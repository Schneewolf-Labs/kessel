import pytest
import torch

from kessel import quant as Q


def test_fp4_codes_roundtrip():
    codes = torch.arange(16).view(1, 16)
    vals = Q.FP4_VALUES[codes]
    enc = Q.fp4_encode(vals)
    # -0 encodes as +0
    assert torch.equal(Q.FP4_VALUES[enc], vals)
    packed = Q.fp4_pack(codes)
    assert packed.shape == (1, 8)
    assert torch.equal(Q.fp4_unpack(packed), vals)


def test_fp4_ties_to_even():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75])
    assert Q.FP4_VALUES[Q.fp4_encode(x)].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0]


def test_e8m0():
    c = torch.tensor([127, 128, 120], dtype=torch.uint8)
    assert Q.e8m0_to_float(c).tolist() == [1.0, 2.0, 2.0**-7]


@pytest.mark.parametrize("kind,shape", [("fp8", (256, 384)), ("fp8", (200, 130)), ("fp4", (64, 128))])
def test_requantize_is_identity_on_unchanged(kind, shape):
    torch.manual_seed(0)
    x = torch.randn(shape) * 0.02
    w, s, spec = Q.quantize(x, kind)
    deq = Q.dequantize(w, s, spec)
    assert (deq - x).abs().max() < (0.07 if kind == "fp8" else 0.6) * x.abs().max()
    st = Q.RequantStats()
    w2, s2 = Q.requantize(deq, w, s, spec, st)
    assert torch.equal(w2.view(torch.uint8), w.view(torch.uint8))
    assert torch.equal(s2.view(torch.uint8), s.view(torch.uint8))
    assert st.bytes_changed == 0 and st.blocks_rescaled == 0


def test_requantize_small_edit_touches_few_bytes():
    torch.manual_seed(1)
    x = torch.randn(256, 256) * 0.02
    w, s, spec = Q.quantize(x, "fp8")
    deq = Q.dequantize(w, s, spec)
    edited = deq.clone()
    edited[3, :] *= 0.5  # one row, stays under the block scale
    st = Q.RequantStats()
    w2, s2 = Q.requantize(edited, w, s, spec, st)
    changed = (w2.view(torch.uint8) != w.view(torch.uint8))
    assert changed[3].any() and not changed[torch.arange(256) != 3].any()


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
def test_requantize_bumps_scale_on_overflow(kind):
    torch.manual_seed(2)
    x = torch.randn(128, 128) * 0.02
    w, s, spec = Q.quantize(x, kind)
    deq = Q.dequantize(w, s, spec)
    # push one element per block over the top: the block must rescale, and its unchanged elements must NOT keep
    # their old bytes (those would be read at the new scale)
    edited = deq.clone()
    edited[0, 0] = deq.abs().max() * 1.5
    st = Q.RequantStats()
    w2, s2 = Q.requantize(edited, w, s, spec, st)
    assert st.blocks_rescaled >= 1
    out = Q.dequantize(w2, s2, spec)
    tol = 0.07 if kind == "fp8" else 0.3
    assert ((out - edited).abs() <= tol * edited.abs().max()).all()
    assert (out - edited).norm() < (0.05 if kind == "fp8" else 0.3) * edited.norm()


def test_float32_scale_inv():
    torch.manual_seed(3)
    x = torch.randn(256, 256)
    w, s, spec = Q.quantize(x, "fp8", scale_dtype="F32")
    assert s.dtype == torch.float32
    assert torch.allclose(Q.dequantize(w, s, spec), x, rtol=0.07, atol=1e-2)
