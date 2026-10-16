"""Byte-level safetensors I/O and model references.

We parse the safetensors header ourselves instead of going through `safe_open`, so any dtype the format can name
(F8_E4M3, F8_E8M0, packed FP4 stored as I8, ...) can be read, copied byte-for-byte, and written back, whether or not
the installed safetensors/numpy know about it. Tensors are only materialized when a merge actually needs their
values; untouched tensors move as raw bytes.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

# safetensors dtype string -> (torch dtype, bytes per element)
DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "F64": (torch.float64, 8),
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "BOOL": (torch.bool, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
}
if hasattr(torch, "float8_e8m0fnu"):
    DTYPES["F8_E8M0"] = (torch.float8_e8m0fnu, 1)

TORCH_TO_ST = {v[0]: k for k, v in DTYPES.items()}

DTYPE_NAMES = {
    "float32": "F32", "fp32": "F32", "float": "F32",
    "float16": "F16", "fp16": "F16", "half": "F16",
    "bfloat16": "BF16", "bf16": "BF16",
    "float64": "F64",
}


def st_dtype(name: str | None) -> str | None:
    """Config dtype name ("bfloat16", "float16", ...) -> safetensors dtype string."""
    if name is None:
        return None
    key = name.lower().removeprefix("torch.")
    if key not in DTYPE_NAMES:
        raise ValueError(f"unsupported dtype {name!r}")
    return DTYPE_NAMES[key]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str  # safetensors dtype string
    shape: tuple[int, ...]
    file: Path
    start: int  # absolute byte offset in file
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class SafetensorsFile:
    """One .safetensors file, mmapped."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        self.metadata = header.pop("__metadata__", None) or {}
        base = 8 + n
        self.tensors: dict[str, TensorInfo] = {}
        for name, h in header.items():
            s, e = h["data_offsets"]
            self.tensors[name] = TensorInfo(name, h["dtype"], tuple(h["shape"]), self.path, base + s, base + e)
        self._mm: mmap.mmap | None = None

    def raw(self, name: str) -> bytes:
        if self._mm is None:
            with open(self.path, "rb") as f:
                self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        t = self.tensors[name]
        return self._mm[t.start:t.end]

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None


def bytes_to_tensor(data: bytes, dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    if dtype not in DTYPES:
        raise ValueError(f"cannot materialize dtype {dtype} (only copy it raw)")
    tdt, _ = DTYPES[dtype]
    if len(data) == 0:
        return torch.empty(shape, dtype=tdt)
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return buf.view(tdt).reshape(shape)


def tensor_to_bytes(t: torch.Tensor) -> tuple[bytes, str]:
    t = t.detach().contiguous().cpu()
    if t.dtype not in TORCH_TO_ST:
        raise ValueError(f"cannot serialize {t.dtype}")
    return t.view(torch.uint8).numpy().tobytes() if t.numel() else b"", TORCH_TO_ST[t.dtype]


class ModelFiles:
    """All safetensors tensors in a model directory (sharded via index.json, or every *.safetensors file)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        idx = self.path / "model.safetensors.index.json"
        weight_map = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
        if weight_map is not None:
            files = sorted(set(weight_map.values()))
        else:
            files = sorted(p.name for p in self.path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors weights in {self.path}")
        self.files = {f: SafetensorsFile(self.path / f) for f in files}
        self.info: dict[str, TensorInfo] = {}
        self._file_of: dict[str, SafetensorsFile] = {}
        if weight_map is not None:
            # the index is authoritative: an overlay checkpoint's linked shards still hold the replaced tensors
            for name, f in weight_map.items():
                sf = self.files[f]
                if name not in sf.tensors:
                    raise KeyError(f"index maps {name} to {f}, which doesn't contain it")
                self.info[name] = sf.tensors[name]
                self._file_of[name] = sf
        else:
            for sf in self.files.values():
                for name, ti in sf.tensors.items():
                    if name in self.info:
                        raise ValueError(f"{name} appears in more than one shard and there is no index")
                    self.info[name] = ti
                    self._file_of[name] = sf

    def __contains__(self, name: str) -> bool:
        return name in self.info

    def names(self) -> list[str]:
        return list(self.info)

    def raw(self, name: str) -> bytes:
        return self._file_of[name].raw(name)

    def tensor(self, name: str) -> torch.Tensor:
        ti = self.info[name]
        return bytes_to_tensor(self.raw(name), ti.dtype, ti.shape)

    def config(self) -> dict:
        p = self.path / "config.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def close(self):
        for f in self.files.values():
            f.close()


# ---------------------------------------------------------------- model references

HF_ALLOW = ["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja", "*.tiktoken"]


@dataclass(frozen=True)
class ModelRef:
    """`path_or_repo[@revision][+lora_path_or_repo[@revision]]`, the mergekit syntax."""

    model: str
    revision: str | None = None
    lora: str | None = None
    lora_revision: str | None = None

    @classmethod
    def parse(cls, s: str) -> "ModelRef":
        s = str(s)
        model, _, lora = s.partition("+")

        def split_rev(x: str) -> tuple[str, str | None]:
            # local paths may legitimately contain '@'; only treat it as a revision when the path doesn't exist
            if "@" in x and not Path(x).exists():
                a, _, b = x.rpartition("@")
                return a, b
            return x, None

        m, mr = split_rev(model)
        lo, lr = split_rev(lora) if lora else (None, None)
        return cls(m, mr, lo, lr)

    def __str__(self) -> str:
        s = self.model + (f"@{self.revision}" if self.revision else "")
        if self.lora:
            s += "+" + self.lora + (f"@{self.lora_revision}" if self.lora_revision else "")
        return s


def resolve(path_or_repo: str, revision: str | None = None, allow: list[str] | None = None) -> Path:
    """Local directory, or download (cached) from the Hugging Face Hub."""
    p = Path(os.path.expanduser(path_or_repo))
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, revision=revision, allow_patterns=allow or HF_ALLOW))


# ---------------------------------------------------------------- writer


class ShardWriter:
    """Streams tensors into size-capped shards, then names them model-0000i-of-0000N and writes the index.

    `add_raw` writes bytes verbatim (dtype string and shape given), so untouched tensors stay byte-identical.
    """

    def __init__(self, out: Path, max_shard_bytes: int = 5 * 1024**3, prefix: str = "model"):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.max = max_shard_bytes
        self.prefix = prefix
        self._shards: list[list[tuple[str, str, tuple[int, ...], bytes]]] = []
        self._pending: list[tuple[str, str, tuple[int, ...], bytes]] = []
        self._pending_bytes = 0
        self._tmp_files: list[Path] = []
        self._weight_map: dict[str, int] = {}
        self.total_bytes = 0

    def add(self, name: str, t: torch.Tensor):
        data, dt = tensor_to_bytes(t)
        self.add_raw(name, dt, tuple(t.shape), data)

    def add_raw(self, name: str, dtype: str, shape: tuple[int, ...], data: bytes):
        if name in self._weight_map:
            raise ValueError(f"duplicate output tensor {name}")
        if self._pending and self._pending_bytes + len(data) > self.max:
            self._flush()
        self._pending.append((name, dtype, tuple(shape), data))
        self._pending_bytes += len(data)
        self._weight_map[name] = len(self._tmp_files)
        self.total_bytes += len(data)

    def _flush(self):
        if not self._pending:
            return
        tmp = self.out / f".kessel-shard-{len(self._tmp_files):05d}.tmp"
        write_safetensors(tmp, self._pending)
        self._tmp_files.append(tmp)
        self._pending, self._pending_bytes = [], 0

    def finish(self) -> dict[str, str]:
        """Rename shards into place, write the index (if sharded), return name -> file."""
        self._flush()
        n = len(self._tmp_files)
        if n == 1:
            names = [f"{self.prefix}.safetensors"]
        else:
            names = [f"{self.prefix}-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]
        for tmp, final in zip(self._tmp_files, names):
            tmp.rename(self.out / final)
        wm = {k: names[v] for k, v in self._weight_map.items()}
        if n > 1:
            idx = {"metadata": {"total_size": self.total_bytes}, "weight_map": dict(sorted(wm.items()))}
            (self.out / f"{self.prefix}.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")
        return wm


def write_safetensors(path: Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]], metadata: dict | None = None):
    header: dict = {"__metadata__": metadata or {"format": "pt"}}
    off = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        off += len(data)
    if len(header) != len(tensors) + 1:
        raise ValueError(f"{path}: duplicate tensor names")
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for *_, data in tensors:
            f.write(data)


def iter_safetensors(path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    sf = SafetensorsFile(path)
    try:
        for name, ti in sf.tensors.items():
            yield name, bytes_to_tensor(sf.raw(name), ti.dtype, ti.shape)
    finally:
        sf.close()
