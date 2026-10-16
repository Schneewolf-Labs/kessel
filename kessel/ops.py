"""Operations around merges: compare checkpoints, recover a rank-1 edit (e.g. an abliteration direction) from a
checkpoint pair, and re-apply it to another model at any strength.

Projection conventions (W is [out, in], r a unit vector):
  output side:  W' = W - lam * r (r^T W)     r lives in the output space (e.g. the residual stream for o_proj/wo_b)
  input side:   W' = W - lam * (W r) r^T     r lives in the input space
lam = 1 removes the direction exactly; lam > 1 over-projects (inverts it).
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from . import quant as Q
from .io import DTYPES, ModelRef, ShardWriter, tensor_to_bytes
from .merge import Model

SIDE_FILES = {".json", ".jinja", ".model", ".txt", ".tiktoken"}


def _model(m: str | Model) -> Model:
    return m if isinstance(m, Model) else Model(ModelRef.parse(m))


# ---------------------------------------------------------------- diff


@dataclass
class TensorDiff:
    name: str
    rel_norm: float  # ||B - A|| / ||A||
    rank1_energy: float | None  # share of ||B - A||^2 in the top singular value (2-D only)
    dtype: str
    shape: tuple


def diff(a: str | Model, b: str | Model, device: str = "cpu", pattern: str | None = None, svd: bool = True):
    """Tensors that differ between two checkpoints. Returns (diffs, only_in_a, only_in_b)."""
    A, B = _model(a), _model(b)
    rx = re.compile(pattern) if pattern else None
    na = {n for n in A.info if n not in A.companions}
    nb = {n for n in B.info if n not in B.companions}
    out = []
    for n in sorted(na & nb):
        if rx and not rx.search(n):
            continue
        if A.raw_key(n) == B.raw_key(n):
            continue
        wa, wb = A.dense(n, device), B.dense(n, device)
        if wa.shape != wb.shape:
            out.append(TensorDiff(n, float("nan"), None, A.info[n].dtype, tuple(A.info[n].shape)))
            continue
        d = wb - wa
        rel = float(d.norm() / wa.norm().clamp(min=1e-12))
        r1 = None
        if svd and d.dim() == 2 and rel > 0:
            s = torch.linalg.svdvals(d)
            r1 = float(s[0] ** 2 / (s ** 2).sum())
        out.append(TensorDiff(n, rel, r1, A.info[n].dtype, tuple(wa.shape)))
    return out, sorted(na - nb), sorted(nb - na)


# ---------------------------------------------------------------- directions


@dataclass
class Direction:
    name: str
    r: torch.Tensor  # unit vector
    side: str  # "output" | "input"
    lam: float  # fitted strength
    fit: float  # cosine between the observed delta and the fitted rank-1 projection


def _fit(W: torch.Tensor, D: torch.Tensor, r: torch.Tensor, side: str) -> tuple[float, float]:
    P = torch.outer(r, r @ W) if side == "output" else torch.outer(W @ r, r)
    pp = (P * P).sum()
    lam = float(-(D * P).sum() / pp) if pp > 0 else 0.0
    fit = float(-(D * P).sum() / (D.norm() * P.norm()).clamp(min=1e-30))
    return lam, fit


def extract_directions(base: str | Model, edited: str | Model, pattern: str | None = None, device: str = "cpu",
                       side: str = "auto") -> list[Direction]:
    """For each 2-D tensor that differs, recover r and lam such that edited ~= project(base, r, lam).

    The top singular vector of the delta gives r; which side it lives on is picked by which projection model
    explains the delta better (reported as `fit`, 1.0 = exactly a rank-1 projection; quantization noise lowers it).
    """
    A, B = _model(base), _model(edited)
    rx = re.compile(pattern) if pattern else None
    res = []
    for n in sorted(n for n in A.info if n in B.info and n not in A.companions):
        if (rx and not rx.search(n)) or A.raw_key(n) == B.raw_key(n):
            continue
        W, E = A.dense(n, device), B.dense(n, device)
        if W.dim() != 2 or W.shape != E.shape:
            continue
        D = E - W
        if min(D.shape) > 1024:
            U, S, V = torch.svd_lowrank(D.double(), q=8, niter=6)
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(D.double(), full_matrices=False)
        cands = []
        for sd, r in (("output", U[:, 0]), ("input", Vh[0])):
            if side in ("auto", sd):
                lam, fit = _fit(W.double(), D.double(), r, sd)
                cands.append((fit, sd, r, lam))
        fit, sd, r, lam = max(cands, key=lambda c: c[0])
        res.append(Direction(n, r.float().cpu(), sd, lam, fit))
    return res


def save_directions(dirs: list[Direction], path: str | Path):
    from safetensors.torch import save_file

    save_file({d.name: d.r.contiguous() for d in dirs}, str(path),
              metadata={"kessel": json.dumps({d.name: {"side": d.side, "lam": d.lam, "fit": d.fit} for d in dirs})})


def load_directions(path: str | Path) -> list[Direction]:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        meta = json.loads((f.metadata() or {}).get("kessel", "{}"))
        return [Direction(k, f.get_tensor(k), meta.get(k, {}).get("side", "output"),
                          meta.get(k, {}).get("lam", 1.0), meta.get(k, {}).get("fit", float("nan")))
                for k in f.keys()]


def project(W: torch.Tensor, r: torch.Tensor, lam: float, side: str) -> torch.Tensor:
    r = (r / r.norm()).to(W.device, W.dtype)
    if side == "output":
        return W - lam * torch.outer(r, r @ W)
    return W - lam * torch.outer(W @ r, r)


# ---------------------------------------------------------------- apply


@dataclass
class ApplyReport:
    edited: list[str]
    requant: Q.RequantStats
    residual: dict  # name -> |r.W'| / |r.W| (target |1 - lam|)


def apply_directions(model: str | Model, dirs: list[Direction], out: str | Path, lam: float | None = None,
                     device: str = "cpu", overlay: bool = True, max_shard_bytes: int = 5 * 1024**3) -> ApplyReport:
    """Project each direction out of its tensor (at `lam`, or each direction's fitted lam) and save.

    overlay=True writes only the edited tensors to a new shard and symlinks the untouched ones, like the published
    DeepSeek-V4 abliteration overlay; overlay=False writes a full copy.
    """
    M = _model(model)
    if M.lora:
        raise ValueError("bake the LoRA first (kessel merge), then apply directions")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    by_name = {d.name: d for d in dirs}
    missing = [n for n in by_name if n not in M.info]
    if missing:
        raise KeyError(f"model has no {missing[:3]}")
    writer = ShardWriter(out, max_shard_bytes, prefix="kessel-overlay" if overlay else "model")
    stats = Q.RequantStats()
    linked: dict[str, str] = {}
    residual = {}
    edited_scales = {M.quant(n).scale_name for n in by_name if M.quant(n)}
    for n, ti in M.info.items():
        if n in by_name:
            d = by_name[n]
            spec = M.quant(n)
            W = M.dense(n, device)
            lam_n = d.lam if lam is None else lam
            Wp = project(W, d.r, lam_n, d.side)
            if spec:
                w1, s1 = Q.requantize(Wp, M.files.tensor(n).to(device), M.files.tensor(spec.scale_name).to(device),
                                      spec, stats)
                writer.add_raw(n, ti.dtype, tuple(w1.shape), tensor_to_bytes(w1)[0])
                si = M.info[spec.scale_name]
                writer.add_raw(spec.scale_name, si.dtype, si.shape, tensor_to_bytes(s1)[0])
                actual = Q.dequantize(w1.to(device), s1.to(device), spec)
            else:
                actual = Wp.to(DTYPES[ti.dtype][0])
                writer.add(n, actual)
                actual = actual.float()
            r = d.r.to(device)
            num = (r @ actual) if d.side == "output" else (actual @ r)
            den = (r @ W) if d.side == "output" else (W @ r)
            residual[n] = float(num.norm() / den.norm().clamp(min=1e-30))
        elif n in edited_scales:
            continue
        elif overlay:
            linked[n] = ti.file.name
        else:
            writer.add_raw(n, ti.dtype, ti.shape, M.files.raw(n))
    written = writer.finish()
    if overlay:
        from .overlay import finish

        finish(M.path, out, written, linked)
    for f in M.path.iterdir():
        if f.is_file() and f.suffix in SIDE_FILES and f.name != "model.safetensors.index.json" \
                and not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    return ApplyReport(sorted(by_name), stats, residual)
