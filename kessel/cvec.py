"""llama.cpp control vectors: a residual-stream vector added after decoder layers at inference time.

The file is a GGUF with general.architecture = "controlvector" and one f32 tensor `direction.<k>` of length n_embd
per steered layer. llama.cpp (`--control-vector FILE` / `--control-vector-scaled FILE:S`) adds direction.k to the
output of decoder layer k, counting from 0, and refuses direction.0: the first layer's output can't be steered.

Layer numbering is the easy thing to get wrong. Hugging Face `output_hidden_states` returns n_layers + 1 states and
hidden_states[j] is the output of decoder layer j - 1 (hidden_states[0] is the embeddings), so a vector measured at
hidden_states[j] belongs in direction.(j - 1). Pass hf_index=True when the vectors are indexed that way.

Scale is linear: a file saved at scale s and loaded with --control-vector-scaled FILE:t adds s * t * v.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

_ALIGN = 32
_U32, _I32, _STR, _ARR, _U64 = 4, 5, 8, 9, 10
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def mean_difference(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Per-layer steering vectors from hidden states stacked as [n_examples, n_layers + 1, d] (HF indexing):
    mean(pos) - mean(neg), shape [n_layers + 1, d]. Save with hf_index=True."""
    return pos.float().mean(0) - neg.float().mean(0)


def _layers(vectors, hf_index: bool, layers) -> dict[int, torch.Tensor]:
    if isinstance(vectors, torch.Tensor):
        if vectors.dim() != 2:
            raise ValueError(f"expected [n_layers, d] vectors, got shape {list(vectors.shape)}")
        vectors = dict(enumerate(vectors))
    out = {}
    for j, v in vectors.items():
        j = int(j)
        if layers is not None and j not in layers:
            continue
        k = j - 1 if hf_index else j
        if k < 1:
            if layers is None:
                continue  # embeddings / layer 0 output: llama.cpp can't steer them, so skip unless asked for
            raise ValueError(f"layer {j} maps to direction.{k}; llama.cpp only steers direction.1 and up")
        out[k] = v.detach().float().cpu().reshape(-1)
    if not out:
        raise ValueError("no layers left to write")
    dims = {v.numel() for v in out.values()}
    if len(dims) != 1:
        raise ValueError(f"vectors have different lengths: {sorted(dims)}")
    return dict(sorted(out.items()))


def save_control_vector(vectors: torch.Tensor | dict[int, torch.Tensor], path: str | Path, scale: float = 1.0,
                        layers=None, hf_index: bool = False, model_hint: str = "") -> list[int]:
    """Write a llama.cpp control vector. `vectors` is a [n_layers, d] tensor or {layer: vector}; `layers` keeps a
    subset (in the vectors' own indexing). Returns the llama.cpp layer numbers written."""
    out = _layers(vectors, hf_index, None if layers is None else set(layers))
    kv = [("general.architecture", _STR, "controlvector"), ("controlvector.model_hint", _STR, model_hint),
          ("controlvector.layer_count", _I32, len(out))]
    buf = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(out), len(kv)))

    def s(x: str):
        b = x.encode()
        buf.extend(struct.pack("<Q", len(b)) + b)

    for key, typ, val in kv:
        s(key)
        buf.extend(struct.pack("<I", typ))
        s(val) if typ == _STR else buf.extend(struct.pack("<i", val))
    off = 0
    blobs = []
    for k, v in out.items():
        s(f"direction.{k}")
        buf.extend(struct.pack("<IQIQ", 1, v.numel(), 0, off))  # 1-D, F32, offset into the data section
        b = (v * scale).numpy().astype("<f4").tobytes()
        blobs.append(b + b"\0" * (-len(b) % _ALIGN))
        off += len(blobs[-1])
    buf.extend(b"\0" * (-len(buf) % _ALIGN))
    Path(path).write_bytes(bytes(buf) + b"".join(blobs))
    return list(out)


def load_control_vector(path: str | Path) -> tuple[dict[int, torch.Tensor], dict]:
    """Read a control vector: ({llama.cpp layer: f32 vector}, metadata)."""
    data = Path(path).read_bytes()
    if data[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    ver, nt, nkv = struct.unpack_from("<IQQ", data, 4)
    if ver < 2:
        raise ValueError(f"{path}: GGUF v{ver} is not supported")
    pos = 24

    def rd(fmt):
        nonlocal pos
        r = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return r[0]

    def rs():
        nonlocal pos
        n = rd("Q")
        pos += n
        return data[pos - n:pos].decode()

    def rv(t):
        if t == _STR:
            return rs()
        if t == _ARR:
            et, n = rd("I"), rd("Q")
            return [rv(et) for _ in range(n)]
        return rd(_SCALAR[t])

    meta = {}
    for _ in range(nkv):
        key = rs()
        meta[key] = rv(rd("I"))
    if meta.get("general.architecture") != "controlvector":
        raise ValueError(f"{path}: architecture is {meta.get('general.architecture')!r}, not 'controlvector'")
    infos = []
    for _ in range(nt):
        name, nd = rs(), rd("I")
        shape = [rd("Q") for _ in range(nd)]
        typ, off = rd("I"), rd("Q")
        if typ != 0:
            raise ValueError(f"{path}: {name} is not f32")
        infos.append((name, shape, off))
    base = pos + (-pos % int(meta.get("general.alignment", _ALIGN)))
    out = {}
    for name, shape, off in infos:
        if not name.startswith("direction."):
            continue
        n = 1
        for d in shape:
            n *= d
        k = int(name.split(".", 1)[1])
        v = torch.frombuffer(bytearray(data[base + off:base + off + 4 * n]), dtype=torch.float32)
        out[k] = out[k] + v if k in out else v  # llama.cpp sums repeated layers too
    return dict(sorted(out.items())), meta
