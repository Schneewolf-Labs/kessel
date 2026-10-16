"""Merge methods. Each is a plain function over float32 tensors.

Semantics follow mergekit's implementations (same parameter names and defaults) so existing YAML configs give the
same result, with two deliberate differences: math is done in float32 (mergekit uses the config dtype, usually
bf16, for most methods), and random sparsification is seeded per tensor name so a merge is reproducible.

A method receives a `MergeInput`: the base tensor (or None), the other models' tensors in config order, their
per-model parameters, and the global parameters.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class MergeInput:
    name: str
    base: torch.Tensor | None
    tensors: list[torch.Tensor]
    params: list[dict]  # per entry of `tensors`
    globals: dict
    base_params: dict = field(default_factory=dict)
    seed: int = 0

    def generator(self) -> torch.Generator:
        h = int(hashlib.sha256(f"{self.seed}:{self.name}".encode()).hexdigest()[:15], 16)
        dev = (self.base if self.base is not None else self.tensors[0]).device
        g = torch.Generator(device=dev)
        g.manual_seed(h)
        return g


@dataclass(frozen=True)
class Method:
    name: str
    fn: Callable[[MergeInput], torch.Tensor]
    needs_base: bool
    global_params: dict  # name -> default (required if default is REQUIRED)
    model_params: dict
    # identical inputs -> identical output (lets the planner copy bytes instead of computing)
    idempotent: Callable[[dict, list[dict]], bool] = lambda g, p: True


REQUIRED = object()


# ---------------------------------------------------------------- helpers


def common_submatrix(ts: list[torch.Tensor]) -> list[torch.Tensor]:
    """mergekit's embedding rule: if 2-D shapes differ, use the largest common [rows, cols] of all of them."""
    if all(t.shape == ts[0].shape for t in ts):
        return ts
    if not all(t.dim() == 2 for t in ts):
        raise ValueError(f"shape mismatch {[tuple(t.shape) for t in ts]}")
    r = min(t.shape[0] for t in ts)
    c = min(t.shape[1] for t in ts)
    return [t[:r, :c] for t in ts]


def _rescaled(t: torch.Tensor, mask: torch.Tensor, rescale: bool) -> torch.Tensor:
    m = t * mask
    if not rescale:
        return m
    before, after = t.abs().sum(), m.abs().sum()
    if before < 1e-7 or after < 1e-7:
        return m
    return m * (before / after)


def sparsify_magnitude(t, density, rescale, g=None):
    if density >= 1:
        return t
    k = int(density * t.numel())
    if k <= 0:
        raise ValueError("density too low: would zero the whole tensor")
    mask = torch.zeros_like(t)
    idx = torch.argsort(t.abs().reshape(-1), descending=True)[:k]
    mask.view(-1)[idx] = 1
    return _rescaled(t, mask, rescale)


def sparsify_random(t, density, rescale, g=None):
    if density >= 1:
        return t
    mask = torch.bernoulli(torch.full_like(t, density), generator=g)
    return _rescaled(t, mask, rescale)


def sparsify_outliers(t, density, rescale, g=None, gamma=0.01):
    if density >= 1:
        return t
    n = t.numel()
    target, n_top = int(density * n), int(gamma * n)
    n_bot = n - target - n_top
    if n_bot < 0:
        n_top += n_bot
        n_bot = 0
    order = torch.argsort(t.abs().reshape(-1))
    mask = torch.zeros_like(t)
    mask.view(-1)[order[n_bot: n - n_top]] = 1
    return _rescaled(t, mask, rescale)


def sparsify_della(t, density, rescale, g=None, epsilon=0.15):
    if density >= 1:
        return t
    if density <= 0:
        return torch.zeros_like(t)
    if density + epsilon >= 1 or density - epsilon <= 0:
        raise ValueError("della: epsilon must keep density +/- epsilon inside (0, 1)")
    shape = t.shape
    x = t if t.dim() >= 2 else t.unsqueeze(0)
    ranks = torch.argsort(torch.argsort(x.abs(), dim=1), dim=1).to(x.dtype) + 1
    lo, hi = ranks.min(1, keepdim=True).values, ranks.max(1, keepdim=True).values
    rank_norm = ((ranks - lo) / (hi - lo)).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    mask = torch.bernoulli(probs, generator=g)
    return _rescaled(x, mask, rescale).reshape(shape)


SPARSIFIERS = {"magnitude": sparsify_magnitude, "random": sparsify_random,
               "magnitude_outliers": sparsify_outliers, "della_magprune": sparsify_della}


# ---------------------------------------------------------------- task-arithmetic family


def _gta(sparsify: str | None, consensus: str | None):
    def fn(m: MergeInput) -> torch.Tensor:
        base = m.base
        deltas, weights = [], []
        g = m.generator() if sparsify in ("random", "della_magprune") else None
        for t, p in zip(m.tensors, m.params):
            if t.shape != base.shape:
                b, t = common_submatrix([base, t])
                if b.shape != base.shape:  # other model is smaller: pad its delta with zeros
                    full = torch.zeros_like(base)
                    full[: t.shape[0], : t.shape[1]] = t - b
                    d = full
                else:
                    d = t - b
            else:
                d = t - base
            if sparsify:
                kw = {}
                if sparsify == "magnitude_outliers":
                    kw["gamma"] = p.get("gamma", 0.01)
                if sparsify == "della_magprune":
                    kw["epsilon"] = p.get("epsilon", 0.15)
                d = SPARSIFIERS[sparsify](d, p.get("density", 1.0), m.globals["rescale"], g, **kw)
            deltas.append(d)
            weights.append(float(p["weight"]))
        if not deltas:
            return base
        D = torch.stack(deltas)
        W = torch.tensor(weights, dtype=D.dtype, device=D.device).view(-1, *([1] * base.dim()))
        WD = D * W
        if consensus:
            sign = WD.sign()
            if consensus == "sum":
                majority = (WD.sum(0) >= 0).to(D.dtype) * 2 - 1
            else:
                majority = (sign.sum(0) >= 0).to(D.dtype) * 2 - 1
            mask = (sign == majority).to(D.dtype)
            mixed = (WD * mask).sum(0)
            div = (W * mask).sum(0)
            div[div == 0] = 1
        else:
            mixed = WD.sum(0)
            div = W.sum(0).expand_as(mixed).clone()
            div[div.abs() < 1e-8] = 1
        if m.globals["normalize"]:
            mixed = mixed / div
        lam = m.globals.get("lambda", 1.0)
        if lam != 1:
            mixed = mixed * lam
        return base + mixed

    return fn


def _gta_method(name, sparsify, consensus, normalize, rescale):
    mp = {"weight": REQUIRED, "density": 1.0}
    if sparsify == "magnitude_outliers":
        mp["gamma"] = 0.01
    if sparsify == "della_magprune":
        mp["epsilon"] = 0.15
    return Method(name, _gta(sparsify, consensus), True,
                  {"normalize": normalize, "rescale": rescale, "lambda": 1.0, "int8_mask": False}, mp)


# ---------------------------------------------------------------- others


def linear(m: MergeInput) -> torch.Tensor:
    ts = list(m.tensors)
    ps = list(m.params)
    if m.base is not None:
        ts.insert(0, m.base)
        ps.insert(0, m.base_params)
    ts = common_submatrix(ts)
    if any(p.get("weight") is None for p in ps):
        raise ValueError("linear: every model (including base_model, if set) needs a weight")
    w = torch.tensor([float(p["weight"]) for p in ps], dtype=ts[0].dtype, device=ts[0].device)
    w = w.view(-1, *([1] * ts[0].dim()))
    res = (torch.stack(ts) * w).sum(0)
    if m.globals["normalize"]:
        res = res / w.sum(0)
    return res


def _linear_idem(g, ps):
    return bool(g.get("normalize", True)) or abs(sum(float(p.get("weight", 0)) for p in ps) - 1) < 1e-9


def slerp_tensors(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold=0.9995, eps=1e-8) -> torch.Tensor:
    a, b = v0.double(), v1.double()
    na, nb = a.norm(), b.norm()
    a_n = a / na if na > eps else a
    b_n = b / nb if nb > eps else b
    dot = (a_n * b_n).sum()
    if dot.abs() > dot_threshold:
        return ((1 - t) * v0 + t * v1)
    theta0 = torch.arccos(dot)
    s0 = torch.sin(theta0 - theta0 * t) / torch.sin(theta0)
    s1 = torch.sin(theta0 * t) / torch.sin(theta0)
    return (s0 * a + s1 * b).to(v0.dtype)


def slerp(m: MergeInput) -> torch.Tensor:
    if not m.tensors:
        return m.base
    if len(m.tensors) != 1:
        raise ValueError("slerp merges exactly two models (base + one)")
    a, b = common_submatrix([m.base, m.tensors[0]])
    return slerp_tensors(float(m.globals["t"]), a, b)


def nearswap(m: MergeInput) -> torch.Tensor:
    if not m.tensors:
        return m.base
    if len(m.tensors) != 1:
        raise ValueError("nearswap merges exactly two models (base + one)")
    a, b = m.base, m.tensors[0]
    w = (float(m.globals["t"]) / (a - b).abs().clamp(min=1e-6)).clamp(0, 1)
    return w * b + (1 - w) * a


def model_stock(m: MergeInput) -> torch.Tensor:
    if not m.tensors:
        return m.base
    if len(m.tensors) < 2:
        raise ValueError("model_stock needs a base plus at least two models")
    w0, *ws = common_submatrix([m.base] + list(m.tensors))
    shape = w0.shape
    if m.globals["filter_wise"]:
        if w0.dim() == 1:
            w0, ws = w0.unsqueeze(0), [w.unsqueeze(0) for w in ws]
    else:
        w0, ws = w0.reshape(-1), [w.reshape(-1) for w in ws]
    offs = [w - w0 for w in ws]
    cos = []
    for i in range(len(offs)):
        for j in range(i + 1, len(offs)):
            npd = offs[i].norm(dim=-1) * offs[j].norm(dim=-1)
            cos.append(((offs[i] * offs[j]).sum(-1) / npd.clamp(min=1e-6)).clamp(-1, 1))
    c = torch.stack(cos).mean(0).unsqueeze(-1)
    n = len(ws)
    t = (n * c) / (1 + (n - 1) * c)
    avg = sum(ws) / n
    return (t * avg + (1 - t) * w0).reshape(shape)


def arcee_fusion(m: MergeInput) -> torch.Tensor:
    if not m.tensors:
        return m.base
    if len(m.tensors) != 1:
        raise ValueError("arcee_fusion merges exactly two models (base + one)")
    base, other = common_submatrix([m.base, m.tensors[0]])
    eps = 1e-8
    p = torch.softmax(other, dim=-1) + eps
    q = torch.softmax(base, dim=-1) + eps
    kl = (p * torch.log(p / q)).sum(-1)
    imp = (other - base).abs() * kl.unsqueeze(-1)
    flat = imp.reshape(-1)
    if flat.numel() > 1_000_000:
        flat = flat[torch.randperm(flat.numel(), generator=m.generator(), device=flat.device)[:1_000_000]]
    srt, _ = torch.sort(flat)
    n = srt.numel() - 1

    def q_at(x):
        return srt[int(x * n)]

    thr = q_at(0.5) + 1.5 * (q_at(0.75) - q_at(0.25))
    return base + (other - base) * (imp >= thr).to(base.dtype)


def passthrough(m: MergeInput) -> torch.Tensor:
    srcs = ([m.base] if m.base is not None else []) + list(m.tensors)
    if len(srcs) != 1:
        raise ValueError("passthrough takes exactly one source per tensor")
    scale = (m.base_params if m.base is not None else m.params[0]).get("scale")
    return srcs[0] if scale is None else srcs[0] * float(scale)


METHODS: dict[str, Method] = {}


def register(method: Method):
    METHODS[method.name] = method


for _m in [
    _gta_method("task_arithmetic", None, None, False, False),
    _gta_method("ties", "magnitude", "sum", True, False),
    _gta_method("dare_ties", "random", "sum", False, True),
    _gta_method("dare_linear", "random", None, False, True),
    _gta_method("breadcrumbs", "magnitude_outliers", None, False, False),
    _gta_method("breadcrumbs_ties", "magnitude_outliers", "sum", False, False),
    _gta_method("della", "della_magprune", "sum", True, True),
    _gta_method("della_linear", "della_magprune", None, False, True),
    Method("linear", linear, False, {"normalize": True}, {"weight": REQUIRED}, _linear_idem),
    Method("slerp", slerp, True, {"t": REQUIRED}, {}),
    Method("nearswap", nearswap, True, {"t": REQUIRED}, {}),
    Method("model_stock", model_stock, True, {"filter_wise": False}, {}),
    Method("arcee_fusion", arcee_fusion, True, {}, {}),
    Method("passthrough", passthrough, False, {}, {"scale": None},
           lambda g, ps: all(p.get("scale") in (None, 1, 1.0) for p in ps)),
]:
    register(_m)


def get(name: str) -> Method:
    if name not in METHODS:
        raise ValueError(f"unknown merge_method {name!r}; have {sorted(METHODS)}")
    return METHODS[name]

