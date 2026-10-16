"""Block-scaled quantized weights: detect, dequantize to float32, requantize.

Formats handled (all as stored in safetensors, a weight tensor plus a sibling scale tensor):

- FP8 e4m3 weight + block scale, the DeepSeek-V3/V4 and HF "fp8" checkpoints. The scale is either ue8m0
  (`<w>.scale`, F8_E8M0, multiply) or float32 (`<w>_scale_inv`, multiply, despite the name). Block shape is
  inferred from the shapes (normally 128x128).
- FP4 e2m1, two values per byte (low nibble first) stored as I8/U8 with an F8_E8M0 scale per 1x32 block along the
  input dim. DeepSeek-V4 routed experts. Layout taken from DeepSeek's own `inference/convert.py`.

Requantization reuses each block's original scale whenever the merged block still fits under it, and an element
whose value didn't change re-encodes to its original byte. So a tensor that a merge didn't touch comes back
byte-identical, and a small edit (e.g. an abliteration) changes only the bytes it actually moved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


@dataclass(frozen=True)
class QuantSpec:
    kind: str  # "fp8" | "fp4"
    scale_name: str
    scale_dtype: str  # "F8_E8M0" | "F32" | ...
    block: tuple[int, int]  # (rows, cols) of logical weight per scale element


def scale_name_for(weight_name: str, names) -> str | None:
    if not weight_name.endswith(".weight"):
        return None
    stem = weight_name[: -len(".weight")]
    for cand in (stem + ".scale", weight_name + "_scale_inv", weight_name + "_scale"):
        if cand in names:
            return cand
    return None


def _infer_block(logical: tuple[int, int], scale_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(scale_shape) != 2:
        raise ValueError(f"expected 2-D block scale, got {scale_shape}")
    out = []
    for n, s in zip(logical, scale_shape):
        b = math.ceil(n / s)
        if math.ceil(n / b) != s:
            raise ValueError(f"cannot infer block size: weight {logical}, scale {scale_shape}")
        out.append(b)
    return tuple(out)


def detect(weight_name: str, info: dict) -> QuantSpec | None:
    """info: name -> object with .dtype (safetensors string) and .shape."""
    w = info[weight_name]
    sname = scale_name_for(weight_name, info)
    if sname is None:
        return None
    s = info[sname]
    if w.dtype == "F8_E4M3" and len(w.shape) == 2:
        return QuantSpec("fp8", sname, s.dtype, _infer_block(w.shape, s.shape))
    if w.dtype in ("I8", "U8") and len(w.shape) == 2 and s.dtype == "F8_E8M0":
        logical = (w.shape[0], w.shape[1] * 2)
        return QuantSpec("fp4", sname, s.dtype, _infer_block(logical, s.shape))
    return None


# ---------------------------------------------------------------- scales


def e8m0_to_float(codes: torch.Tensor) -> torch.Tensor:
    """ue8m0 bytes -> float32 powers of two (255 = NaN per OCP MX; treated as an error)."""
    c = codes.view(torch.uint8).to(torch.int32)
    if (c == 255).any():
        raise ValueError("e8m0 NaN code in scale")
    return torch.ldexp(torch.ones_like(c, dtype=torch.float32), c - 127)


def float_to_e8m0(exp: torch.Tensor) -> torch.Tensor:
    """integer exponents -> ue8m0 bytes."""
    c = (exp + 127).clamp(0, 254).to(torch.uint8)
    return c


def scale_to_float(scale: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "F8_E8M0":
        return e8m0_to_float(scale)
    return scale.float()


def _expand(scale: torch.Tensor, block: tuple[int, int], shape: tuple[int, int]) -> torch.Tensor:
    return scale.repeat_interleave(block[0], 0)[: shape[0]].repeat_interleave(block[1], 1)[:, : shape[1]]


# ---------------------------------------------------------------- fp4 codec


def fp4_unpack(packed: torch.Tensor) -> torch.Tensor:
    """[R, C] bytes -> [R, 2C] float32 e2m1 values (low nibble first)."""
    b = packed.view(torch.uint8).to(torch.int64)
    lo, hi = b & 0x0F, (b >> 4) & 0x0F
    vals = FP4_VALUES.to(packed.device)
    return torch.stack([vals[lo], vals[hi]], dim=-1).flatten(-2)


# magnitude codes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6; midpoints between neighbours
_FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def fp4_encode(x: torch.Tensor) -> torch.Tensor:
    """float values (already divided by scale) -> 4-bit codes, round to nearest, ties to even code."""
    a = x.abs().clamp(max=FP4_MAX)
    mids = _FP4_MIDS.to(x.device)
    idx = torch.bucketize(a, mids)  # a <= mid -> lower; ties land on the lower code
    tie = torch.isin(a, mids)
    # at a tie bucketize picked the lower code; ties-to-even wants the even code
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    sign = (torch.signbit(x) & (idx > 0)).to(torch.int64)
    return idx + 8 * sign


def fp4_pack(codes: torch.Tensor) -> torch.Tensor:
    """[R, 2C] codes -> [R, C] int8 bytes."""
    lo, hi = codes[..., 0::2], codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8).view(torch.int8)


# ---------------------------------------------------------------- dequantize / requantize


def dequantize(w: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    s = scale_to_float(scale, spec.scale_dtype)
    if spec.kind == "fp8":
        v = w.float()
    else:
        v = fp4_unpack(w)
    return v * _expand(s.to(v.device), spec.block, tuple(v.shape))


def _block_amax(x: torch.Tensor, block: tuple[int, int], sshape: tuple[int, int]) -> torch.Tensor:
    R, C = x.shape
    br, bc = block
    pad = torch.zeros(sshape[0] * br, sshape[1] * bc, dtype=x.dtype, device=x.device)
    pad[:R, :C] = x.abs()
    return pad.view(sshape[0], br, sshape[1], bc).amax(dim=(1, 3))


@dataclass
class RequantStats:
    blocks: int = 0
    blocks_rescaled: int = 0
    bytes_changed: int = 0


def requantize(x: torch.Tensor, orig_w: torch.Tensor, orig_scale: torch.Tensor, spec: QuantSpec,
               stats: RequantStats | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 weights -> (codes, scale) in the original format, reusing original block scales where they fit."""
    x = x.float()
    qmax = FP8_MAX if spec.kind == "fp8" else FP4_MAX
    s_old = scale_to_float(orig_scale, spec.scale_dtype).to(x.device)
    amax = _block_amax(x, spec.block, tuple(s_old.shape))
    fits = amax <= qmax * s_old
    if spec.scale_dtype == "F8_E8M0":
        need = torch.ceil(torch.log2(torch.clamp(amax / qmax, min=2.0**-126))).to(torch.int32)
        old_exp = orig_scale.view(torch.uint8).to(torch.int32).to(x.device) - 127
        new_exp = torch.where(fits, old_exp, need)
        s_codes = float_to_e8m0(new_exp)
        s_new = torch.ldexp(torch.ones_like(new_exp, dtype=torch.float32), new_exp)
        scale_out = s_codes.view(torch.float8_e8m0fnu)
    else:
        s_new = torch.where(fits, s_old, torch.clamp(amax / qmax, min=1e-12))
        scale_out = s_new.to(orig_scale.dtype)
    full = _expand(s_new, spec.block, tuple(x.shape))
    # an original byte can only be reused where its block kept its original scale
    kept = _expand(fits.to(torch.float32), spec.block, tuple(x.shape)).bool()
    y = x / full
    if spec.kind == "fp8":
        codes = y.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        old = orig_w.to(x.device)
        same = kept & ((codes.float() * full) == (old.float() * _expand(s_old, spec.block, tuple(x.shape))))
        codes_u8 = torch.where(same, old.view(torch.uint8), codes.view(torch.uint8))
        out_w = codes_u8.view(torch.float8_e4m3fn)
    else:
        c = fp4_encode(y)
        old = orig_w.to(x.device)
        oldc = old.view(torch.uint8).to(torch.int64)
        old_codes = torch.stack([oldc & 0xF, (oldc >> 4) & 0xF], -1).flatten(-2)
        vals = FP4_VALUES.to(x.device)
        same = kept & ((vals[c] * full) == (vals[old_codes] * _expand(s_old, spec.block, tuple(x.shape))))
        c = torch.where(same, old_codes, c)
        out_w = fp4_pack(c).view(orig_w.dtype)
    if stats is not None:
        stats.blocks += fits.numel()
        stats.blocks_rescaled += int((~fits).sum())
        stats.bytes_changed += int((out_w.view(torch.uint8) != orig_w.to(x.device).view(torch.uint8)).sum())
    return out_w.cpu(), scale_out.cpu()


def quantize(x: torch.Tensor, kind: str = "fp8", block: tuple[int, int] | None = None,
             scale_dtype: str = "F8_E8M0") -> tuple[torch.Tensor, torch.Tensor, QuantSpec]:
    """Fresh quantization of float weights (power-of-two ue8m0 or float32 block scales)."""
    x = x.float()
    block = block or ((128, 128) if kind == "fp8" else (1, 32))
    qmax = FP8_MAX if kind == "fp8" else FP4_MAX
    sshape = (math.ceil(x.shape[0] / block[0]), math.ceil(x.shape[1] / block[1]))
    amax = _block_amax(x, block, sshape)
    if scale_dtype == "F8_E8M0":
        exp = torch.ceil(torch.log2(torch.clamp(amax / qmax, min=2.0**-126))).to(torch.int32)
        scale = float_to_e8m0(exp).view(torch.float8_e8m0fnu)
    else:
        scale = torch.clamp(amax / qmax, min=1e-12)
    spec = QuantSpec(kind, "", scale_dtype, block)
    zeros = (torch.zeros(x.shape, dtype=torch.float8_e4m3fn) if kind == "fp8"
             else torch.zeros(x.shape[0], x.shape[1] // 2, dtype=torch.int8))
    w, s = requantize(x, zeros.to(x.device), scale.to(x.device), spec)
    return w, s, spec
