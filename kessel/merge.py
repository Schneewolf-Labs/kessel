"""Plan and run a merge, one output tensor at a time.

Tensors are matched by name; there are no per-architecture definitions. A tensor belongs to a layer when its name
contains `<prefix>.<N>.` with prefix ending in layers/h/blocks/layer/mtp (`model.layers.12.mlp.up_proj.weight`,
`layers.3.attn.wo_b.weight`, `mtp.0...`). The group with the most layers is the one `slices:` and layer_range
address; gradients interpolate over a slice's layers (t = i / (n - 1)), and tensors outside layers get t = 0,
as in mergekit.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from . import methods as M
from . import quant as Q
from .config import MergeConfig, Slice, Source, parse
from .io import DTYPES, ModelFiles, ModelRef, ShardWriter, resolve, st_dtype, tensor_to_bytes
from .lora import LoraAdapter

log = logging.getLogger(__name__)

LAYER_RE = re.compile(r"^(?P<prefix>(?:.*?\.)?(?:layers|h|blocks|layer|mtp))\.(?P<idx>\d+)\.(?P<rest>.+)$")
EMBED_RE = re.compile(r"(^|\.)(embed_tokens|embed|wte|tok_embeddings|word_embeddings|lm_head|head|output)\.weight$")
PRE_RE = re.compile(r"(^|\.)(embed_tokens|embed|wte|wpe|tok_embeddings|word_embeddings|embeddings)(\.|$)")
FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
TOKENIZER_FILES = ["tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
                   "added_tokens.json", "merges.txt", "vocab.json", "chat_template.jinja", "chat_template.json"]
TAGALONG_FILES = ["generation_config.json", "preprocessor_config.json", "processor_config.json",
                  "video_preprocessor_config.json"]
LAYER_COUNT_KEYS = ["num_hidden_layers", "n_layers", "num_layers", "n_layer"]


def layer_of(name: str):
    m = LAYER_RE.match(name)
    return (m["prefix"], int(m["idx"]), m["rest"]) if m else None


# ---------------------------------------------------------------- loaded models


class Model:
    """A model's tensors, plus its LoRA (from `model+lora`) applied on read."""

    def __init__(self, ref: ModelRef):
        self.ref = ref
        self.path = resolve(ref.model, ref.revision)
        self.files = ModelFiles(self.path)
        self.info = self.files.info
        self.lora = LoraAdapter(ref.lora, ref.lora_revision) if ref.lora else None
        self._quant: dict[str, Q.QuantSpec | None] = {}
        self.companions: set[str] = set()
        for n in self.info:
            spec = Q.detect(n, self.info)
            self._quant[n] = spec
            if spec:
                self.companions.add(spec.scale_name)
        if self.lora:
            self.lora.bind([n for n in self.info if n not in self.companions])
        groups: dict[str, int] = {}
        for n in self.info:
            lo = layer_of(n)
            if lo:
                groups[lo[0]] = max(groups.get(lo[0], 0), lo[1] + 1)
        self.groups = groups

    def quant(self, name: str) -> Q.QuantSpec | None:
        return self._quant.get(name)

    def touched_by_lora(self, name: str) -> bool:
        return bool(self.lora and self.lora.touches(name))

    def raw_key(self, name: str) -> tuple:
        """Bytes identifying the stored tensor (weight + scale if quantized)."""
        spec = self.quant(name)
        k = (self.info[name].dtype, self.info[name].shape, self.files.raw(name))
        if spec:
            k += (self.files.raw(spec.scale_name),)
        return k

    def dense(self, name: str, device: str) -> torch.Tensor:
        spec = self.quant(name)
        if spec:
            w = Q.dequantize(self.files.tensor(name).to(device), self.files.tensor(spec.scale_name).to(device), spec)
        else:
            w = self.files.tensor(name).to(device, torch.float32)
        if self.lora:
            w = self.lora.apply(name, w)
        return w


# ---------------------------------------------------------------- plan


@dataclass
class Input:
    model: str  # key into Plan.models
    name: str  # tensor name in that model
    params: dict = field(default_factory=dict)
    is_base: bool = False


@dataclass
class OutTensor:
    name: str
    ref: Input  # the input whose dtype/format/shape the output takes
    inputs: list[Input]
    globals: dict
    t: float


@dataclass
class Plan:
    cfg: MergeConfig
    method: M.Method
    models: dict[str, Model]
    tensors: list[OutTensor]
    ref_model: str
    layer_prefix: str | None
    layer_map: list[int] | None  # output layer i <- reference-model layer layer_map[i] (slices only)


def _resolve_globals(cfg, method, name, t, slice_):
    out = {}
    for p, default in method.global_params.items():
        v = cfg.parameter(p, name, t, slice_)
        if v is None:
            if default is M.REQUIRED:
                raise ValueError(f"missing required parameter {p!r} for {name}")
            v = default
        out[p] = v
    return out


def _resolve_model_params(cfg, method, src_name, t, slice_, source, is_base):
    out = {}
    for p, default in method.model_params.items():
        v = cfg.parameter(p, src_name, t, slice_, source)
        if v is None:
            if default is M.REQUIRED and not is_base:
                raise ValueError(f"missing required parameter {p!r} for {source.model}:{src_name}")
            v = None if default is M.REQUIRED else default
        out[p] = v
    return out


def plan(cfg: MergeConfig) -> Plan:
    method = M.get(cfg.merge_method)
    models = {str(r): Model(r) for r in cfg.referenced()}
    if cfg.models is not None:
        return _plan_models(cfg, method, models)
    return _plan_slices(cfg, method, models)


def _plan_models(cfg, method, models) -> Plan:
    sources = list(cfg.models)
    base = str(cfg.base_model) if cfg.base_model else None
    if base and base not in [str(s.model) for s in sources]:
        sources.append(Source(cfg.base_model))
    if method.needs_base and not base:
        raise ValueError(f"{method.name} needs a base_model")
    ref_key = base or str(sources[0].model)
    ref = models[ref_key]
    out = []
    for name in ref.info:
        if name in ref.companions:
            continue
        lo = layer_of(name)
        n = ref.groups[lo[0]] if lo else 0
        t = (lo[1] / (n - 1) if n > 1 else 1.0) if lo else 0.0
        inputs = []
        for s in sources:
            key = str(s.model)
            if name not in models[key].info:
                if key != ref_key:
                    log.warning("%s has no %s; merging that tensor without it", key, name)
                continue
            is_base = key == base
            inputs.append(Input(key, name, _resolve_model_params(cfg, method, name, t, None, s, is_base), is_base))
        out.append(OutTensor(name, next(i for i in inputs if i.model == ref_key), inputs,
                             _resolve_globals(cfg, method, name, t, None), t))
    return Plan(cfg, method, models, out, ref_key, None, None)


def _plan_slices(cfg, method, models) -> Plan:
    first = models[str(cfg.slices[0].sources[0].model)]
    if not first.groups:
        raise ValueError("slices: no layered tensors found")
    prefix = max(first.groups, key=lambda p: (first.groups[p], "layers" in p))
    global_base = str(cfg.base_model) if cfg.base_model else None
    out: list[OutTensor] = []
    layer_map: list[int] = []

    def slice_base(sl: Slice) -> str | None:
        b = str(sl.base_model) if sl.base_model else global_base
        if b and b not in [str(s.model) for s in sl.sources]:
            if method.needs_base:
                raise ValueError(f"slice base_model {b} must be one of the slice's sources")
            return None
        return b

    def ref_source(sl: Slice) -> Source:
        b = slice_base(sl)
        return next((s for s in sl.sources if str(s.model) == b), sl.sources[0])

    def add(name_out, sl, srcs_names, t):
        b = slice_base(sl)
        if method.needs_base and not b:
            raise ValueError(f"{method.name} needs a base_model")
        inputs = []
        for s, src_name in srcs_names:
            key = str(s.model)
            if src_name not in models[key].info:
                log.warning("%s has no %s; merging that tensor without it", key, src_name)
                continue
            is_base = key == b
            inputs.append(Input(key, src_name, _resolve_model_params(cfg, method, src_name, t, sl, s, is_base),
                                is_base))
        rs = ref_source(sl)
        ref_in = next((i for i in inputs if i.model == str(rs.model)), inputs[0])
        out.append(OutTensor(name_out, ref_in, inputs, _resolve_globals(cfg, method, name_out, t, sl), t))

    # tensors outside the sliced layer group: embeddings from the first slice's models, the rest from the last's
    def outside(sl: Slice, pre: bool):
        rm = models[str(ref_source(sl).model)]
        for name in rm.info:
            if name in rm.companions:
                continue
            lo = layer_of(name)
            if lo and lo[0] == prefix:
                continue
            if bool(PRE_RE.search(name)) == pre:
                add(name, sl, [(s, name) for s in sl.sources], 0.0)

    outside(cfg.slices[0], pre=True)
    out_idx = 0
    for sl in cfg.slices:
        rs = ref_source(sl)
        rm = models[str(rs.model)]
        length = rs.layer_range[1] - rs.layer_range[0]
        for i in range(length):
            t = i / (length - 1) if length > 1 else 1.0
            src_layer = rs.layer_range[0] + i
            rests = [layer_of(n)[2] for n in rm.info
                     if n not in rm.companions and (lo := layer_of(n)) and lo[0] == prefix and lo[1] == src_layer]
            if not rests:
                raise ValueError(f"{rs.model} has no layer {src_layer} under {prefix}")
            for rest in rests:
                add(f"{prefix}.{out_idx}.{rest}", sl,
                    [(s, f"{prefix}.{s.layer_range[0] + i}.{rest}") for s in sl.sources], t)
            layer_map.append(src_layer)
            out_idx += 1
    outside(cfg.slices[-1], pre=False)
    ref_key = str(ref_source(cfg.slices[0]).model)
    return Plan(cfg, method, models, out, ref_key, prefix, layer_map)


# ---------------------------------------------------------------- execute


@dataclass
class Report:
    tensors: int = 0
    copied: int = 0
    computed: int = 0
    unchanged_after_compute: int = 0
    requant: Q.RequantStats = field(default_factory=Q.RequantStats)
    changed: list[str] = field(default_factory=list)


def _rename_companion(out_name: str, ref_name: str, scale_name: str) -> str:
    # ref "a.weight" + scale "a.scale" -> out "b.weight" -> "b.scale"; same for "_scale_inv"
    stem_in = ref_name[: -len(".weight")]
    stem_out = out_name[: -len(".weight")]
    assert scale_name.startswith(stem_in)
    return stem_out + scale_name[len(stem_in):]


def run(cfg: MergeConfig | str | Path | dict, out: str | Path, device: str = "cpu", overlay: bool = False,
        max_shard_bytes: int = 5 * 1024**3, keep_quantized: bool = True, progress: bool = True) -> Report:
    if not isinstance(cfg, MergeConfig):
        cfg = parse(cfg)
    out = Path(out)
    p = plan(cfg)
    ref_model = p.models[p.ref_model]
    if overlay and (p.layer_map is not None or ref_model.lora):
        raise ValueError("--overlay needs a `models:` merge whose reference model is a plain checkpoint")
    out_dtype = st_dtype(cfg.out_dtype or cfg.dtype)
    out.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out, max_shard_bytes, prefix="kessel-overlay" if overlay else "model")
    rep = Report()
    linked: dict[str, str] = {}  # overlay: output name -> reference shard file

    it = p.tensors
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc="merging", unit="tensor")
        except ImportError:
            pass

    for ot in it:
        rep.tensors += 1
        rm = p.models[ot.ref.model]
        rinfo = rm.info[ot.ref.name]
        spec = rm.quant(ot.ref.name)
        keep_q = spec is not None and keep_quantized
        target_dtype = rinfo.dtype if (keep_q or rinfo.dtype not in FLOAT_DTYPES or out_dtype is None) else out_dtype
        if spec and not keep_q:
            target_dtype = out_dtype or "BF16"
        mparams = [i.params for i in ot.inputs if not i.is_base]

        def emit_raw(src: Model, name: str):
            spec_s = src.quant(name)
            if overlay and src is ref_model and name == ot.name:
                linked[ot.name] = src.info[name].file.name
                if spec_s:
                    linked[spec_s.scale_name] = src.info[spec_s.scale_name].file.name
                return
            ti = src.info[name]
            writer.add_raw(ot.name, ti.dtype, ti.shape, src.files.raw(name))
            if spec_s:
                si = src.info[spec_s.scale_name]
                writer.add_raw(_rename_companion(ot.name, name, spec_s.scale_name), si.dtype, si.shape,
                               src.files.raw(spec_s.scale_name))

        # 1. non-float tensors (ids, masks, packed ints that aren't a known quant format): copy from the reference
        if rinfo.dtype not in FLOAT_DTYPES and not spec:
            emit_raw(rm, ot.ref.name)
            rep.copied += 1
            continue
        # 2. identical inputs through an idempotent method: copy bytes
        no_lora = not any(p.models[i.model].touched_by_lora(i.name) for i in ot.inputs)
        if (no_lora and target_dtype == rinfo.dtype and p.method.idempotent(ot.globals, mparams)
                and len({p.models[i.model].raw_key(i.name) for i in ot.inputs}) == 1):
            emit_raw(rm, ot.ref.name)
            rep.copied += 1
            continue
        # 3. compute
        dense = {id(i): p.models[i.model].dense(i.name, device) for i in ot.inputs}
        shapes = {tuple(v.shape) for v in dense.values()}
        if len(shapes) > 1 and not (EMBED_RE.search(ot.name) and all(len(s) == 2 for s in shapes)):
            raise ValueError(f"{ot.name}: shape mismatch across models {sorted(shapes)}")
        base_in = next((i for i in ot.inputs if i.is_base), None)
        others = [i for i in ot.inputs if not i.is_base]
        mi = M.MergeInput(
            name=ot.name,
            base=dense[id(base_in)] if base_in else None,
            tensors=[dense[id(i)] for i in others],
            params=[i.params for i in others],
            globals=ot.globals,
            base_params=base_in.params if base_in else {},
            seed=cfg.seed,
        )
        res = p.method.fn(mi).float()
        del dense, mi
        rep.computed += 1
        if keep_q:
            logical = rinfo.shape if spec.kind == "fp8" else (rinfo.shape[0], rinfo.shape[1] * 2)
            if tuple(res.shape) != tuple(logical):
                raise ValueError(f"{ot.name}: merged shape {tuple(res.shape)} can't be requantized")
            w0 = rm.files.tensor(ot.ref.name)
            s0 = rm.files.tensor(spec.scale_name)
            w1, s1 = Q.requantize(res, w0.to(device), s0.to(device), spec, rep.requant)
            wb, wdt = tensor_to_bytes(w1)
            sb, _ = tensor_to_bytes(s1)
            wdt = rinfo.dtype  # keep I8 vs U8 / exact dtype string of the source
            if overlay and ot.name == ot.ref.name and rm is ref_model and wb == rm.files.raw(ot.ref.name) \
                    and sb == rm.files.raw(spec.scale_name):
                rep.unchanged_after_compute += 1
                emit_raw(rm, ot.ref.name)
                continue
            writer.add_raw(ot.name, wdt, tuple(w1.shape), wb)
            si = rm.info[spec.scale_name]
            writer.add_raw(_rename_companion(ot.name, ot.ref.name, spec.scale_name), si.dtype, si.shape, sb)
        else:
            res_t = res.to(DTYPES[target_dtype][0]).cpu()
            data, _ = tensor_to_bytes(res_t)
            if overlay and ot.name == ot.ref.name and rm is ref_model and target_dtype == rinfo.dtype \
                    and data == rm.files.raw(ot.ref.name):
                rep.unchanged_after_compute += 1
                emit_raw(rm, ot.ref.name)
                continue
            writer.add_raw(ot.name, target_dtype, tuple(res_t.shape), data)
        rep.changed.append(ot.name)

    written = writer.finish()
    if overlay:
        _finish_overlay(ref_model, out, written, linked)
    _write_metadata(p, out, cfg)
    return rep


def _finish_overlay(ref: Model, out: Path, written: dict[str, str], linked: dict[str, str]):
    import os

    for fname in sorted(set(linked.values())):
        dst = out / fname
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(os.path.relpath((ref.path / fname).resolve(), out.resolve()))
    wm = {**linked, **written}
    total = sum((out / f).stat().st_size for f in set(wm.values()))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, indent=2) + "\n")
    idx = out / "kessel-overlay.safetensors.index.json"
    if idx.exists():
        idx.unlink()


def _set_layer_count(cfg: dict, n_old: int, n_new: int, layer_map: list[int]):
    for k in LAYER_COUNT_KEYS:
        if cfg.get(k) == n_old:
            cfg[k] = n_new
    for k, v in list(cfg.items()):
        if isinstance(v, list) and len(v) == n_old and k not in LAYER_COUNT_KEYS:
            cfg[k] = [v[i] for i in layer_map]  # per-layer lists, e.g. Qwen3.5 layer_types
        elif isinstance(v, dict) and k.endswith("_config"):
            _set_layer_count(v, n_old, n_new, layer_map)


def _write_metadata(p: Plan, out: Path, cfg: MergeConfig):
    ref = p.models[p.ref_model]
    mcfg = ref.files.config()
    if mcfg:
        if p.layer_map is not None:
            _set_layer_count(mcfg, ref.groups[p.layer_prefix], len(p.layer_map), p.layer_map)
        if cfg.out_dtype or cfg.dtype:
            # transformers >= 4.56 writes `dtype`; older configs carry `torch_dtype`
            key = "dtype" if "dtype" in mcfg or "torch_dtype" not in mcfg else "torch_dtype"
            mcfg[key] = cfg.out_dtype or cfg.dtype
        (out / "config.json").write_text(json.dumps(mcfg, indent=2) + "\n")
    donor = ref
    if cfg.tokenizer_source and cfg.tokenizer_source != "base":
        donor_ref = ModelRef.parse(cfg.tokenizer_source)
        donor = p.models.get(str(donor_ref)) or Model(donor_ref)
    elif cfg.tokenizer_source == "base" and cfg.base_model:
        donor = p.models[str(cfg.base_model)]
    for f in TOKENIZER_FILES + TAGALONG_FILES:
        if (donor.path / f).exists() and not (out / f).exists():
            shutil.copy(donor.path / f, out / f)
    if cfg.chat_template:
        tc = out / "tokenizer_config.json"
        tcfg = json.loads(tc.read_text()) if tc.exists() else {}
        tcfg["chat_template"] = cfg.chat_template
        tc.write_text(json.dumps(tcfg, indent=2) + "\n")
    (out / "kessel_config.yaml").write_text(yaml.safe_dump(_config_dict(cfg), sort_keys=False))


def _config_dict(cfg: MergeConfig) -> dict:
    def src(s: Source) -> dict:
        d = {"model": str(s.model)}
        if s.layer_range:
            d["layer_range"] = list(s.layer_range)
        if s.parameters:
            d["parameters"] = s.parameters
        return d

    d: dict = {}
    if cfg.models is not None:
        d["models"] = [src(s) for s in cfg.models]
    else:
        d["slices"] = [{"sources": [src(s) for s in sl.sources],
                        **({"base_model": str(sl.base_model)} if sl.base_model else {}),
                        **({"parameters": sl.parameters} if sl.parameters else {})} for sl in cfg.slices]
    d["merge_method"] = cfg.merge_method
    if cfg.base_model:
        d["base_model"] = str(cfg.base_model)
    for k in ("parameters", "dtype", "out_dtype", "tokenizer_source", "chat_template"):
        if getattr(cfg, k):
            d[k] = getattr(cfg, k)
    if cfg.seed:
        d["seed"] = cfg.seed
    return d
