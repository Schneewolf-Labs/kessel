"""Merge config: mergekit's YAML schema (the `models:` and `slices:` forms), parsed into dataclasses.

Parameter values may be a scalar, a gradient (list of numbers interpolated across a slice's layers), or a list of
`{filter: <substring>, value: ...}` conditions, the first match winning. Lookup order is mergekit's: the source's
own parameters in the slice, then the slice's, then the top-level `parameters`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .io import ModelRef

log = logging.getLogger(__name__)


def evaluate(setting: Any, tensor_name: str, t: float) -> Any:
    """mergekit `evaluate_setting`: scalar | gradient | [{filter, value}]. Returns None when no condition matches."""
    if isinstance(setting, (int, float, bool, str)) or setting is None:
        return setting
    if isinstance(setting, dict):
        setting = [setting]
    if isinstance(setting, list):
        if setting and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in setting):
            scaled = t * (len(setting) - 1)
            i0 = int(scaled)
            i1 = min(len(setting) - 1, i0 + 1)
            frac = scaled - i0
            return (1 - frac) * setting[i0] + frac * setting[i1]
        if setting and all(isinstance(e, (int, float, bool, str)) for e in setting):
            return setting[int(t * (len(setting) - 1))]
        for cond in setting:
            if not isinstance(cond, dict) or "value" not in cond:
                raise ValueError(f"bad parameter condition {cond!r}")
            f = cond.get("filter")
            if f is None or f == "*" or (tensor_name and f in tensor_name):
                return evaluate(cond["value"], tensor_name, t)
        return None
    raise ValueError(f"unexpected parameter setting {setting!r}")


@dataclass
class Source:
    model: ModelRef
    layer_range: tuple[int, int] | None = None  # None = all layers
    parameters: dict = field(default_factory=dict)


@dataclass
class Slice:
    sources: list[Source]
    base_model: ModelRef | None = None
    parameters: dict = field(default_factory=dict)


@dataclass
class MergeConfig:
    merge_method: str
    slices: list[Slice] | None = None  # None -> `models:` form, one implicit full-depth slice
    models: list[Source] | None = None
    base_model: ModelRef | None = None
    parameters: dict = field(default_factory=dict)
    dtype: str | None = None
    out_dtype: str | None = None
    tokenizer_source: str | None = None  # None | "base" | model ref
    chat_template: str | None = None
    seed: int = 0

    def referenced(self) -> list[ModelRef]:
        seen: dict[str, ModelRef] = {}
        if self.base_model:
            seen[str(self.base_model)] = self.base_model
        for s in self.models or []:
            seen.setdefault(str(s.model), s.model)
        for sl in self.slices or []:
            for s in sl.sources:
                seen.setdefault(str(s.model), s.model)
            if sl.base_model:
                seen.setdefault(str(sl.base_model), sl.base_model)
        return list(seen.values())

    def parameter(self, name: str, tensor: str, t: float, slice_: Slice | None = None,
                  source: Source | None = None) -> Any:
        """Resolve a parameter for one tensor; None if unset everywhere."""
        layers = []
        if source is not None:
            layers.append(source.parameters)
        if slice_ is not None:
            layers.append(slice_.parameters)
        layers.append(self.parameters)
        for params in layers:
            if params and name in params:
                v = evaluate(params[name], tensor, t)
                if v is not None:
                    return v
        return None


KNOWN_TOP = {"merge_method", "slices", "models", "base_model", "parameters", "dtype", "out_dtype",
             "tokenizer_source", "tokenizer", "chat_template", "seed", "name"}


def _src(d: dict, with_range: bool) -> Source:
    if "model" not in d:
        raise ValueError(f"source without model: {d}")
    lr = d.get("layer_range")
    if with_range and lr is not None:
        lr = (int(lr[0]), int(lr[1]))
    return Source(ModelRef.parse(d["model"]), lr if with_range else None, d.get("parameters") or {})


def parse(data: dict | str | Path) -> MergeConfig:
    if isinstance(data, (str, Path)) and Path(data).exists():
        data = yaml.safe_load(Path(data).read_text())
    elif isinstance(data, str):
        data = yaml.safe_load(data)
    unknown = set(data) - KNOWN_TOP
    if unknown:
        # mergekit silently ignores unknown keys (e.g. a misspelled `tokenzer_source`); we at least say so
        log.warning("ignoring unknown config keys: %s", sorted(unknown))
    if "modules" in data:
        raise ValueError("`modules:` configs aren't supported yet")
    if bool(data.get("slices")) == bool(data.get("models")):
        raise ValueError("specify exactly one of `models:` or `slices:`")
    tok = data.get("tokenizer_source")
    if data.get("tokenizer"):
        if tok:
            raise ValueError("specify tokenizer_source or tokenizer, not both")
        tcfg = data["tokenizer"]
        if tcfg.get("tokens") or tcfg.get("pad_to_multiple_of"):
            raise ValueError("tokenizer.tokens / pad_to_multiple_of aren't supported yet")
        tok = tcfg.get("source")
    if tok == "union":
        raise ValueError("tokenizer_source: union isn't supported yet (vocab merging); use base or a model")
    cfg = MergeConfig(
        merge_method=data["merge_method"],
        base_model=ModelRef.parse(data["base_model"]) if data.get("base_model") else None,
        parameters=data.get("parameters") or {},
        dtype=data.get("dtype"),
        out_dtype=data.get("out_dtype"),
        tokenizer_source=tok,
        chat_template=data.get("chat_template"),
        seed=int(data.get("seed", 0)),
    )
    if data.get("models"):
        cfg.models = [_src(m, with_range=False) for m in data["models"]]
    else:
        cfg.slices = []
        for s in data["slices"]:
            srcs = [_src(x, with_range=True) for x in s["sources"]]
            if any(x.layer_range is None for x in srcs):
                raise ValueError("every slice source needs a layer_range")
            lens = {x.layer_range[1] - x.layer_range[0] for x in srcs}
            if len(lens) != 1:
                raise ValueError(f"slice sources have different lengths: {[x.layer_range for x in srcs]}")
            cfg.slices.append(Slice(srcs, ModelRef.parse(s["base_model"]) if s.get("base_model") else None,
                                    s.get("parameters") or {}))
    return cfg
