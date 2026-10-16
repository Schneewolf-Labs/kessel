"""Kessel vs mergekit on the same configs (skipped when mergekit isn't installed)."""

import pytest
import yaml

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
TOL = {"arcee_fusion": 5e-2, "ties": 3e-2, "breadcrumbs": 3e-2}
FRAC = {"arcee_fusion": 0.3, "ties": 3e-2, "breadcrumbs": 3e-2, "model_stock": 5e-2}
