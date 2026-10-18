"""End-to-end on a tiny checkpoint laid out like DeepSeek-V4: FP8 attention with ue8m0 128x128 block scales,
FP4 (packed I8) experts with 1x32 scales, bf16/f32 extras, an MTP layer. Covers quantized merges, overlay output
and abliteration-direction recovery and re-application."""

import json

import torch

from kessel import quant as Q
from kessel.io import ModelFiles, tensor_to_bytes, write_safetensors
from kessel.merge import run
from kessel.ops import apply_directions, extract_directions

LAYERS = 3


def _write_model(path, seed, wo_b=None):
    g = torch.Generator().manual_seed(seed)
    tensors = {}

    def put(name, t):
        tensors[name] = t

    def put_q(name, x, kind):
        w, s, _ = Q.quantize(x, kind)
        put(name + ".weight", w)
        put(name + ".scale", s)

    put("embed.weight", (torch.randn(100, 256, generator=g) * 0.02).bfloat16())
    put("head.weight", (torch.randn(100, 256, generator=g) * 0.02).bfloat16())
    put("norm.weight", torch.ones(256).bfloat16())
    put("hc_head_fn", torch.randn(4, 64, generator=g))
    for i, pre in [(i, f"layers.{i}") for i in range(LAYERS)] + [(LAYERS, "mtp.0")]:
        x = torch.randn(256, 512, generator=g) * 0.02  # always drawn, so the rest of the stream is unchanged
        if wo_b and pre in wo_b:
            x = wo_b[pre]
        put_q(f"{pre}.attn.wo_b", x, "fp8")
        put_q(f"{pre}.attn.wq_a", torch.randn(128, 256, generator=g) * 0.02, "fp8")
        put_q(f"{pre}.ffn.experts.0.w1", torch.randn(64, 256, generator=g) * 0.02, "fp4")
        put(f"{pre}.attn_norm.weight", torch.ones(256).bfloat16())
    path.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    half = len(names) // 2
    wm = {}
    for fi, chunk in enumerate([names[:half], names[half:]]):
        fname = f"model-0000{fi + 1}-of-00002.safetensors"
        items = []
        for n in chunk:
            data, dt = tensor_to_bytes(tensors[n])
            items.append((n, dt, tuple(tensors[n].shape), data))
            wm[n] = fname
        write_safetensors(path / fname, items)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (path / "config.json").write_text(json.dumps({"num_hidden_layers": LAYERS, "model_type": "deepseek_v4"}))
    return tensors


def _dense(mf, name):
    spec = Q.detect(name, mf.info)
    return Q.dequantize(mf.tensor(name), mf.tensor(spec.scale_name), spec)


def test_abliteration_roundtrip(tmp_path):
    stock = tmp_path / "stock"
    _write_model(stock, 0)
    torch.manual_seed(5)
    targets = [f"layers.{i}.attn.wo_b.weight" for i in range(LAYERS)] + ["mtp.0.attn.wo_b.weight"]
    from kessel.ops import Direction

    truth = {n: torch.nn.functional.normalize(torch.randn(256), dim=0) for n in targets}
    rep = apply_directions(str(stock), [Direction(n, r, "output", 2.5, 1.0) for n, r in truth.items()],
                           tmp_path / "abl")
    # overlay: only the 4 edited weights + their scales are new bytes
    over = ModelFiles(tmp_path / "abl")
    new_files = {f for f in over.files if f.startswith("kessel-overlay")}
    assert {n for n, ti in over.info.items() if ti.file.name in new_files} == \
        set(targets) | {t.replace(".weight", ".scale") for t in targets}
    assert all((tmp_path / "abl" / f).is_symlink() for f in over.files if f not in new_files)
    assert set(over.info) == set(ModelFiles(stock).info)
    # residual |r.W'|/|r.W| should be near |1 - 2.5| = 1.5 despite fp8 rounding
    assert all(1.3 < v < 1.7 for v in rep.residual.values()), rep.residual

    dirs = {d.name: d for d in extract_directions(str(stock), str(tmp_path / "abl"))}
    assert set(dirs) == set(targets)
    for n, r in truth.items():
        d = dirs[n]
        assert d.side == "output"
        assert abs(float(d.r @ r)) > 0.99, n
        assert 2.0 < d.lam < 2.8, d.lam


def test_rebake_on_finetune_and_quantized_merge(tmp_path):
    stock, ft = tmp_path / "stock", tmp_path / "ft"
    st = _write_model(stock, 0)
    # "finetune": same model, only wo_b changed (like Smaug-Flash)
    g = torch.Generator().manual_seed(9)
    mf = ModelFiles(stock)
    wo = {pre: _dense(mf, f"{pre}.attn.wo_b.weight") + torch.randn(256, 512, generator=g) * 0.002
          for pre in [f"layers.{i}" for i in range(LAYERS)] + ["mtp.0"]}
    _write_model(ft, 0, wo_b=wo)

    # task arithmetic at weight 1.0 reproduces the finetune; overlay over stock writes only wo_b
    cfg = {"models": [{"model": str(ft), "parameters": {"weight": 1.0}}], "merge_method": "task_arithmetic",
           "base_model": str(stock)}
    rep = run(cfg, tmp_path / "soup", overlay=True, progress=False)
    soup = ModelFiles(tmp_path / "soup")
    ftf = ModelFiles(ft)
    for n in soup.info:
        if ".wo_b." not in n:
            assert soup.raw(n) == ftf.raw(n), n  # untouched tensors: byte-identical
        elif n.endswith(".weight"):
            # stock + (ft - stock) requantized under stock's block scales: within fp8 rounding of the finetune
            want = _dense(ftf, n)
            assert (_dense(soup, n) - want).abs().max() <= 0.07 * want.abs().max(), n
    written = {n for n, ti in soup.info.items() if ti.file.name.startswith("kessel-overlay")}
    assert written == {n for n in st if ".wo_b." in n}
    assert rep.requant.blocks_rescaled == 0

    # half-strength soup: still only wo_b changes, values halfway
    cfg["models"][0]["parameters"]["weight"] = 0.5
    run(cfg, tmp_path / "half", overlay=True, progress=False)
    half = ModelFiles(tmp_path / "half")
    n = "layers.1.attn.wo_b.weight"
    mid = (_dense(mf, n) + _dense(ftf, n)) / 2
    assert (_dense(half, n) - mid).abs().max() < 0.07 * mid.abs().max()

    # recover a direction from stock vs abliterated-stock and re-bake it on the finetune
    from kessel.ops import Direction

    r = torch.nn.functional.normalize(torch.randn(256, generator=g), dim=0)
    apply_directions(str(stock), [Direction(n, r, "output", 2.5, 1.0)], tmp_path / "abl")
    dirs = extract_directions(str(stock), str(tmp_path / "abl"))
    rep2 = apply_directions(str(ft), dirs, tmp_path / "ft_abl", lam=2.5)
    assert 1.3 < rep2.residual[n] < 1.7


def test_fp4_expert_merge(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_model(a, 0)
    _write_model(b, 1)
    run({"models": [{"model": str(a), "parameters": {"weight": 0.5}}, {"model": str(b), "parameters": {"weight": 0.5}}],
         "merge_method": "linear"}, tmp_path / "m", progress=False)
    m, fa, fb = ModelFiles(tmp_path / "m"), ModelFiles(a), ModelFiles(b)
    n = "layers.0.ffn.experts.0.w1.weight"
    assert m.info[n].dtype == "I8" and m.info[n].shape == fa.info[n].shape
    avg = (_dense(fa, n) + _dense(fb, n)) / 2
    got = _dense(m, n)
    # fp4 is coarse: check the merge lands near the average in aggregate
    assert float((got - avg).norm() / avg.norm()) < 0.25
    assert m.info["hc_head_fn"].dtype == "F32" and m.info["embed.weight"].dtype == "BF16"


def test_overlay_of_partial_checkpoint_writes_upstream_index(tmp_path, monkeypatch):
    """A partial checkpoint (as `kessel fetch` writes it) -> the overlay's index is the upstream's full index with
    only the edited tensors repointed, plus kessel_upstream.json for `kessel materialize`."""
    from kessel import overlay
    from kessel.ops import Direction

    full = tmp_path / "full"
    _write_model(full, 0)
    upstream_wm = json.loads((full / "model.safetensors.index.json").read_text())["weight_map"]
    # partial = only layer 1's wo_b (+scale), like `kessel fetch --filter`
    part = tmp_path / "part"
    part.mkdir()
    mf = ModelFiles(full)
    keep = ["layers.1.attn.wo_b.weight", "layers.1.attn.wo_b.scale"]
    write_safetensors(part / "kessel-partial.safetensors",
                      [(n, mf.info[n].dtype, mf.info[n].shape, mf.raw(n)) for n in keep])
    (part / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"partial_of": "org/model@abc123"}, "weight_map": {n: "kessel-partial.safetensors" for n in keep}}))
    monkeypatch.setattr(overlay, "upstream_weight_map", lambda repo, rev: dict(upstream_wm))

    r = torch.nn.functional.normalize(torch.randn(256), dim=0)
    apply_directions(str(part), [Direction(keep[0], r, "output", 1.0, 1.0)], tmp_path / "out")
    wm = json.loads((tmp_path / "out" / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(wm) == set(upstream_wm)
    assert {n for n, f in wm.items() if f != upstream_wm[n]} == set(keep)
    up = json.loads((tmp_path / "out" / "kessel_upstream.json").read_text())
    assert up["partial_of"] == "org/model@abc123"
    assert not any(p.is_symlink() for p in (tmp_path / "out").iterdir())


def test_flatten_overlay(tmp_path):
    from kessel.ops import Direction
    from kessel.overlay import flatten

    stock = tmp_path / "stock"
    _write_model(stock, 0)
    n = "layers.2.attn.wo_b.weight"
    r = torch.nn.functional.normalize(torch.randn(256), dim=0)
    apply_directions(str(stock), [Direction(n, r, "output", 2.5, 1.0)], tmp_path / "ov")
    flatten(tmp_path / "ov", tmp_path / "flat")
    ov, fl, st = ModelFiles(tmp_path / "ov"), ModelFiles(tmp_path / "flat"), ModelFiles(stock)
    assert not any(p.name.startswith("kessel-overlay") for p in (tmp_path / "flat").iterdir())
    assert set(fl.files) == set(st.files)
    for name in st.info:
        assert fl.raw(name) == ov.raw(name), name  # every tensor is the overlay's view
    # and no shard holds a stale copy: each tensor appears exactly once across the flat shards
    seen = [t for f in fl.files.values() for t in f.tensors]
    assert len(seen) == len(set(seen)) == len(st.info)
    assert fl.raw(n) != st.raw(n)


def _write_lora(path, module_names, shapes, scale, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for mod, (out_f, in_f) in zip(module_names, shapes):
        a = torch.randn(4, in_f, generator=g)
        b = torch.randn(out_f, 4, generator=g) * scale
        for kind, t in (("lora_A", a), ("lora_B", b)):
            data, dt = tensor_to_bytes(t)
            items.append((f"base_model.model.{mod}.{kind}.weight", dt, tuple(t.shape), data))
    path.mkdir(parents=True, exist_ok=True)
    write_safetensors(path / "adapter_model.safetensors", items)
    (path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 4, "target_modules": ["wq_a"]}))


def test_promote_changed_keeps_small_lora_delta(tmp_path):
    """A LoRA delta far below FP8's step (~0.1% of the weight): requantizing erases it; --promote-changed writes the
    touched weights as bf16, which keeps much of it, drops their scales and lists them as unquantized."""
    base = _write_model(tmp_path / "base", 0)
    mods = [f"layers.{i}.attn.wq_a" for i in range(LAYERS)]
    _write_lora(tmp_path / "lora", mods, [(128, 256)] * LAYERS, scale=1e-5)
    cfg = {"models": [{"model": f"{tmp_path / 'base'}+{tmp_path / 'lora'}", "parameters": {"weight": 1.0}}],
           "merge_method": "task_arithmetic", "base_model": str(tmp_path / "base"), "dtype": "bfloat16"}
    (tmp_path / "base" / "config.json").write_text(json.dumps(
        {"num_hidden_layers": LAYERS, "model_type": "deepseek_v4",
         "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}))
    mf_base = ModelFiles(tmp_path / "base")
    lora = __import__("kessel.lora", fromlist=["LoraAdapter"]).LoraAdapter(str(tmp_path / "lora"))
    w = _dense(mf_base, "layers.1.attn.wq_a.weight")
    a = lora._tensor("base_model.model.layers.1.attn.wq_a.lora_A.weight")
    b = lora._tensor("base_model.model.layers.1.attn.wq_a.lora_B.weight")
    delta = b @ a

    req = tmp_path / "requant"
    run(cfg, req, progress=False)
    got_req = _dense(ModelFiles(req), "layers.1.attn.wq_a.weight") - w

    out = tmp_path / "promoted"
    rep = run(cfg, out, overlay=True, progress=False, promote_changed=True)
    mf = ModelFiles(out)
    assert sorted(rep.promoted) == sorted(m + ".weight" for m in mods)
    for m in mods:
        assert mf.info[m + ".weight"].dtype == "BF16"
        assert m + ".scale" not in mf.info
    # untouched quantized tensors keep their bytes and scales
    assert mf.info["layers.1.attn.wo_b.weight"].dtype == "F8_E4M3" and "layers.1.attn.wo_b.scale" in mf.info
    assert mf.raw("layers.1.attn.wo_b.weight") == mf_base.raw("layers.1.attn.wo_b.weight")
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qc["ignored_layers"] == sorted(mods) and qc["modules_to_not_convert"] == sorted(mods)

    got = mf.tensor("layers.1.attn.wq_a.weight").float() - w
    cos = torch.nn.functional.cosine_similarity
    assert (delta.norm() / w.norm()) < 2e-3
    assert cos(got.flatten(), delta.flatten(), 0) > 0.5       # bf16 keeps much of it
    assert cos(got_req.flatten(), delta.flatten(), 0) < 0.3   # FP8 requantization erases most of it
    assert torch.equal(got, (w + delta).bfloat16().float() - w)
    assert (out / "model.safetensors.index.json").exists()
    assert base["layers.1.attn.wq_a.weight"].dtype == torch.float8_e4m3fn

