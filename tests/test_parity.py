"""Kessel vs mergekit on the same configs (skipped when mergekit isn't installed)."""

import pytest
import torch
import yaml

from kessel import methods as M
from kessel.io import ModelFiles
from kessel.merge import run

mergekit = pytest.importorskip("mergekit")


def _mergekit(cfg_yaml, out):
    from mergekit.config import MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions

    run_merge(MergeConfiguration.model_validate(yaml.safe_load(cfg_yaml)), str(out),
              MergeOptions(copy_tokenizer=False, quiet=True, write_model_card=False, lora_merge_cache=str(out) + "_lc"))


def _load(path):
    mf = ModelFiles(path)
    return {n: mf.tensor(n).float() for n in mf.names()}


CONFIGS = {
    "linear": """
models:
  - model: {ft1}
    parameters: {{weight: 0.5}}
  - model: {ft2}
    parameters: {{weight: 0.3}}
  - model: {ft3}
    parameters: {{weight: 0.2}}
merge_method: linear
dtype: bfloat16
""",
    "task_arithmetic": """
models:
  - model: {ft1}
    parameters: {{weight: 0.7}}
  - model: {ft2}
    parameters: {{weight: [0.2, 0.9]}}
merge_method: task_arithmetic
base_model: {base}
dtype: bfloat16
""",
    "ties": """
models:
  - model: {ft1}
    parameters: {{weight: 1.0, density: 0.5}}
  - model: {ft2}
    parameters: {{weight: 0.5, density: 0.7}}
  - model: {ft3}
    parameters: {{weight: 0.8, density: 0.3}}
merge_method: ties
base_model: {base}
dtype: bfloat16
""",
    "breadcrumbs": """
models:
  - model: {ft1}
    parameters: {{weight: 1.0, density: 0.6, gamma: 0.05}}
  - model: {ft2}
    parameters: {{weight: 0.5, density: 0.6}}
merge_method: breadcrumbs_ties
base_model: {base}
dtype: bfloat16
""",
    "slerp": """
slices:
  - sources:
      - model: {ft1}
        layer_range: [0, 4]
      - model: {ft2}
        layer_range: [0, 4]
merge_method: slerp
base_model: {ft2}
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
""",
    "model_stock": """
models:
  - model: {ft1}
  - model: {ft2}
  - model: {ft3}
merge_method: model_stock
base_model: {base}
dtype: bfloat16
""",
    "nearswap": """
merge_method: nearswap
base_model: {base}
models:
  - model: {ft1}
parameters:
  t: 0.01
dtype: bfloat16
""",
    "frankenmerge": """
slices:
  - sources:
      - model: {ft1}
        layer_range: [0, 3]
  - sources:
      - model: {ft2}
        layer_range: [1, 4]
merge_method: passthrough
dtype: bfloat16
""",
    "lora_bake": """
base_model: {ft1}+{lora1}
dtype: bfloat16
merge_method: task_arithmetic
parameters:
  normalize: false
slices:
- sources:
  - layer_range: [0, 4]
    model: {ft1}+{lora1}
    parameters:
      weight: 1.0
""",
    "arcee_fusion": """
models:
  - model: {ft1}
merge_method: arcee_fusion
base_model: {base}
dtype: bfloat16
""",
    "multislerp": """
models:
  - model: {ft1}
    parameters: {{weight: 0.5}}
  - model: {ft2}
    parameters: {{weight: [0.2, 0.9]}}
  - model: {ft3}
    parameters: {{weight: 0.3}}
merge_method: multislerp
dtype: bfloat16
""",
    "multislerp_base": """
models:
  - model: {ft1}
    parameters: {{weight: 0.6}}
  - model: {ft2}
    parameters: {{weight: 0.4}}
merge_method: multislerp
base_model: {base}
dtype: bfloat16
""",
    "karcher": """
models:
  - model: {ft1}
  - model: {ft2}
  - model: {ft3}
merge_method: karcher
base_model: {base}
parameters:
  max_iter: 20
  tol: 1.0e-6
dtype: bfloat16
""",
    "sce": """
models:
  - model: {ft1}
  - model: {ft2}
  - model: {ft3}
merge_method: sce
base_model: {base}
dtype: bfloat16
""",
}


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_matches_mergekit(name, dtype, tiny, tmp_path):
    cfg = CONFIGS[name].format(**tiny).replace("dtype: bfloat16", f"dtype: {dtype}")
    _mergekit(cfg, tmp_path / "mk")
    run(yaml.safe_load(cfg), tmp_path / "ks", progress=False)
    mk, ks = _load(tmp_path / "mk"), _load(tmp_path / "ks")
    assert set(mk) == set(ks), (set(mk) ^ set(ks))
    worst_fro, worst_frac = 0.0, 0.0
    for n in mk:
        assert mk[n].shape == ks[n].shape, n
        d = (mk[n] - ks[n]).abs()
        fro = float(d.norm() / mk[n].norm().clamp(min=1e-12))
        # elements off by more than 2 bf16 ulps of their own magnitude
        frac = float((d > 2 * 2**-8 * mk[n].abs().clamp(min=1e-3)).float().mean())
        worst_fro, worst_frac = max(worst_fro, fro), max(worst_frac, frac)
    if dtype == "float32":
        print(f"PARITY32 {name}: worst rel fro err {worst_fro:.2e}, worst frac elements off {worst_frac:.2e}")
        if name == "lora_bake":
            # mergekit merges the LoRA into the bf16-loaded model (bf16 rounding); we add the delta in fp32
            assert worst_fro < 4e-3
        else:
            assert worst_fro < 1e-5 and worst_frac == 0
        return
    print(f"PARITY {name}: worst rel fro err {worst_fro:.2e}, worst frac elements off {worst_frac:.2e}")
    # mergekit does most methods in bf16, we do fp32. Thresholding methods (ties top-k, arcee_fusion) can flip
    # elements sitting on the threshold, so allow a small fraction of elements to differ.
    assert worst_fro < TOL.get(name, 1e-2)
    assert worst_frac < FRAC.get(name, 1e-3)


# bf16: mergekit does the math in bf16, we do fp32 and round once. Thresholding methods (top-k, arcee_fusion's
# importance cut) flip elements that sit on the threshold; the float32 runs above show the math itself is identical.
# multislerp/karcher normalize to unit vectors and rescale in bf16 in mergekit, which moves most elements by an ulp.
TOL = {"arcee_fusion": 5e-2, "ties": 3e-2, "breadcrumbs": 3e-2}
FRAC = {"arcee_fusion": 0.3, "ties": 3e-2, "breadcrumbs": 3e-2, "model_stock": 5e-2, "karcher": 5e-2,
        "multislerp": 0.5, "multislerp_base": 0.5, "sce": 5e-2}


# Paths the end-to-end configs can't reach on the tiny models (mergekit fails or writes a wrong shape there because
# untouched tensors such as layernorms are identical across models), compared function-to-function on random data.
def _rand(n, shape=(32, 48), seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(*shape, generator=g)
    return base, [base + 0.1 * torch.randn(*shape, generator=g) for _ in range(n)]


@pytest.mark.parametrize("topk", [0.3, 0.75, 1.0])
def test_sce_select_topk_matches_mergekit(topk):
    from mergekit.merge_methods.sce import sce_merge

    base, ts = _rand(3)
    mk = sce_merge([t.clone() for t in ts], base.clone(), select_topk=topk)
    ks = M.sce(M.MergeInput("w", base, ts, [{}] * 3, {"select_topk": topk, "int8_mask": False}))
    assert torch.equal(mk, ks)


def test_sce_select_topk_untouched_tensor_keeps_shape():
    # mergekit returns a [n, ...] tensor here (sce_mask's k == 0 branch); the right answer is the base
    base = torch.ones(64)
    ks = M.sce(M.MergeInput("ln", base, [base.clone()] * 3, [{}] * 3, {"select_topk": 0.4, "int8_mask": False}))
    assert torch.equal(ks, base)


@pytest.mark.parametrize("with_base", [True, False])
def test_multislerp_three_models_matches_mergekit(with_base):
    from mergekit.merge_methods.multislerp import multislerp

    base, ts = _rand(3, seed=1)
    w = [0.5, 0.2, 0.3]
    mk = multislerp([t.clone() for t in ts], w, base.clone() if with_base else None)
    ks = M.multislerp(M.MergeInput("w", base if with_base else None, ts, [{"weight": x} for x in w],
                                   {"normalize_weights": True, "eps": 1e-8}))
    assert torch.equal(mk, ks)


def test_multislerp_untouched_tensor_is_base():
    # mergekit raises for >2 models when every delta is zero; every model agrees, so the base is the answer
    base = torch.ones(64)
    ks = M.multislerp(M.MergeInput("ln", base, [base.clone()] * 3, [{"weight": 1.0}] * 3,
                                   {"normalize_weights": True, "eps": 1e-8}))
    assert torch.equal(ks, base)


def test_karcher_matches_mergekit():
    from mergekit.merge_methods.karcher import karcher_merge_tensors

    base, ts = _rand(3, seed=2)
    mk = karcher_merge_tensors([base.clone()] + [t.clone() for t in ts], [0.25] * 4, max_iter=10, tol=1e-5)
    ks = M.karcher(M.MergeInput("w", base, ts, [{}] * 3, {"max_iter": 10, "tol": 1e-5}))
    assert torch.equal(mk, ks)
