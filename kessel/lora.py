"""PEFT LoRA adapters as per-tensor weight deltas, applied while reading (no model load, no merge_and_unload)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import torch

from .io import SafetensorsFile, bytes_to_tensor, resolve

log = logging.getLogger(__name__)

ADAPTER_ALLOW = ["adapter_config.json", "adapter_model.safetensors"]
_LORA_KEY = re.compile(r"^(?P<module>.+?)\.(?P<kind>lora_A|lora_B|lora_embedding_A|lora_embedding_B)(?:\.[^.]+)?\.?(?:weight)?$")


# tensor groups beside the language model that reuse its module names
SIDE_TOWERS = ("mtp", "visual", "vision_tower", "vision_model")


@dataclass
class _Pair:
    module: str
    a: str
    b: str
    embedding: bool


class LoraAdapter:
    def __init__(self, path_or_repo: str, revision: str | None = None):
        self.path = resolve(path_or_repo, revision, ADAPTER_ALLOW)
        self.cfg = json.loads((self.path / "adapter_config.json").read_text())
        if self.cfg.get("use_dora"):
            raise ValueError(f"{path_or_repo}: DoRA adapters aren't supported")
        files = sorted(self.path.glob("adapter_model*.safetensors"))
        if not files:
            raise FileNotFoundError(f"{path_or_repo}: no adapter_model.safetensors")
        self.files = [SafetensorsFile(f) for f in files]
        self._where = {n: f for f in self.files for n in f.tensors}
        pairs: dict[str, dict] = {}
        self.replacements: dict[str, str] = {}  # modules_to_save: model tensor name -> adapter key
        for key in self._where:
            k = key.removeprefix("base_model.model.")
            m = _LORA_KEY.match(k)
            if m:
                pairs.setdefault(m["module"], {})[m["kind"]] = key
            else:
                self.replacements[k] = key
        self.pairs: list[_Pair] = []
        for mod, d in pairs.items():
            if "lora_A" in d and "lora_B" in d:
                self.pairs.append(_Pair(mod, d["lora_A"], d["lora_B"], False))
            elif "lora_embedding_A" in d and "lora_embedding_B" in d:
                self.pairs.append(_Pair(mod, d["lora_embedding_A"], d["lora_embedding_B"], True))
            else:
                raise ValueError(f"{path_or_repo}: incomplete LoRA pair for {mod}: {sorted(d)}")
        self._targets: dict[str, _Pair] = {}

    def _tensor(self, key: str) -> torch.Tensor:
        sf = self._where[key]
        ti = sf.tensors[key]
        return bytes_to_tensor(sf.raw(key), ti.dtype, ti.shape)

    def _scale(self, module: str) -> float:
        r = self.cfg.get("r", 8)
        alpha = self.cfg.get("lora_alpha", r)
        for pat, v in (self.cfg.get("rank_pattern") or {}).items():
            if module == pat or re.fullmatch(rf"(.*\.)?{pat}", module):
                r = v
                break
        for pat, v in (self.cfg.get("alpha_pattern") or {}).items():
            if module == pat or re.fullmatch(rf"(.*\.)?{pat}", module):
                alpha = v
                break
        return alpha / (r ** 0.5) if self.cfg.get("use_rslora") else alpha / r

    def bind(self, model_names) -> None:
        """Map adapter modules onto the model's tensor names (handles a model./language_model. prefix mismatch)."""
        names = set(model_names)
        by_suffix: dict[str, list[str]] = {}
        for n in names:
            parts = n.split(".")
            for i in range(len(parts)):
                by_suffix.setdefault(".".join(parts[i:]), []).append(n)

        def find(stem: str) -> str:
            if stem in names:
                return stem
            parts = stem.split(".")
            for i in range(len(parts)):
                hits = by_suffix.get(".".join(parts[i:]), [])
                if len(hits) > 1:
                    # a text-only adapter's `layers.0...` also names the MTP head's `mtp.layers.0...`
                    # (Qwen3.5/3.8); unless the adapter names the side tower, take the main stack
                    hits = [h for h in hits if not any(f"{a}." in h and f"{a}." not in stem for a in SIDE_TOWERS)]
                if len(hits) == 1:
                    return hits[0]
                if len(hits) > 1:
                    break
            raise KeyError(f"LoRA target {stem} not found in the model")

        self._targets = {find(p.module + ".weight"): p for p in self.pairs}
        self._repl = {find(k): v for k, v in self.replacements.items()}
        if self._repl:
            log.warning("adapter replaces %d whole tensors (modules_to_save): %s", len(self._repl),
                        sorted(self._repl)[:4])

    def touches(self, name: str) -> bool:
        return name in self._targets or name in self._repl

    def apply(self, name: str, w: torch.Tensor) -> torch.Tensor:
        if name in self._repl:
            t = self._tensor(self._repl[name]).to(w.device, torch.float32)
            if t.shape != w.shape:
                raise ValueError(f"modules_to_save {name}: shape {tuple(t.shape)} vs model {tuple(w.shape)}")
            return t
        p = self._targets.get(name)
        if p is None:
            return w
        a = self._tensor(p.a).to(w.device, torch.float32)
        b = self._tensor(p.b).to(w.device, torch.float32)
        delta = (b @ a) * self._scale(p.module)
        if p.embedding:
            delta = delta.T
        if self.cfg.get("fan_in_fan_out"):
            delta = delta.T
        if delta.shape != w.shape:
            raise ValueError(f"LoRA delta for {name} is {tuple(delta.shape)}, weight is {tuple(w.shape)}")
        return w + delta
