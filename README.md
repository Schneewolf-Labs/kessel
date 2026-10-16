# Kessel

Model merging by tensor name. Kessel reads mergekit's YAML, so existing configs run unchanged, and it also handles the
checkpoints mergekit can't: FP8/FP4 block-quantized models, models without an architecture definition, and rank-1
edits such as abliteration.

Kessel is the merge engine for [Merlina](https://github.com/Schneewolf-Labs/Merlina). Grimoire is its training engine.

## Why

The math of a merge method is a few dozen lines of tensor code. The rest is plumbing: streaming shards, matching
tensors across models, and writing output. mergekit handles that plumbing with per-architecture definitions and
assumes bf16/fp16 weights. That makes it unable to run a merge on a new architecture or a natively quantized
checkpoint until someone upstream adds support.

Kessel matches tensors **by name**. A tensor belongs to a layer when its name contains `<prefix>.<N>.`, where the
prefix ends in `layers`, `h`, `blocks`, `layer` or `mtp`. Anything with the same name in two models can be merged.
There are no architecture files.

## Install

```bash
pip install kessel-merge    # imports as `kessel`; the CLI is `kessel`
pip install -e .            # from a checkout: torch, safetensors, pyyaml, huggingface_hub, tqdm
pip install -e ".[dev]"     # + pytest, ruff, transformers, peft (tests)
```

## Merge

```bash
kessel merge config.yaml ./out [--device cuda] [--overlay] [--dequantize]
```

The config is mergekit's: `models:` or `slices:`, `base_model`, `parameters` (scalars, gradients and `filter`
conditions), `dtype`, `tokenizer_source`, and `model+lora` references.

```yaml
models:
  - model: org/finetune-a
    parameters: {weight: 0.6, density: 0.5}
  - model: org/finetune-b
    parameters: {weight: 0.4, density: 0.5}
merge_method: ties
base_model: org/base
dtype: bfloat16
```

**Methods:** `linear`, `task_arithmetic`, `ties`, `dare_ties`, `dare_linear`, `breadcrumbs`, `breadcrumbs_ties`,
`della`, `della_linear`, `slerp`, `nearswap`, `model_stock`, `arcee_fusion`, `passthrough`. Parameter names and
defaults are mergekit's.

**Parity with mergekit:** with `dtype: float32`, every method is bit-identical to mergekit 0.1.4 on the test models
(`tests/test_parity.py`). With bf16 the results differ by bf16 rounding, because Kessel does the math in float32 and
rounds once. Random sparsification (DARE, DELLA) is seeded per tensor name (`seed:` in the config), so re-running a
merge gives the same model.

**Quantized checkpoints:**
- Detects FP8 e4m3 weights with ue8m0 or float32 block scales (DeepSeek-V3/V4, HF fp8).
- Detects FP4 e2m1 weights, packed two per byte with 1×32 ue8m0 scales (DeepSeek-V4 routed experts).
- Merges in float32, then requantizes to the source format.
- Requantizing keeps each block's original scale wherever the result still fits. Elements whose value didn't change
  get their original byte back, so an untouched tensor stays byte-identical and a small edit only changes the bytes
  it moved.
- `--dequantize` writes plain `dtype` tensors instead.

**`--overlay`** writes only the tensors that differ from the base model. The base's shards are symlinked and a new
index points at both. A merge or edit that touches 1.5 GB of a 160 GB checkpoint therefore costs 1.5 GB.

**LoRA:** `model+adapter` applies a PEFT adapter as a weight delta while reading. There is no model load and no
`merge_and_unload`. Supported: rsLoRA, rank/alpha patterns, embedding LoRA, `modules_to_save`, and prefix mismatches
such as `model.layers` vs `model.language_model.layers`. DoRA is not supported.

**Not yet supported:** `tokenizer_source: union` (vocab merging), `modules:` configs, the
`multislerp`/`karcher`/`sce` methods, and parallel tensor execution.

## Rank-1 edits

```bash
# what differs between two checkpoints, and how rank-1 is each difference
kessel diff org/base org/edited --filter 'attn'

# recover per-tensor directions from base vs abliterated (side and strength are fitted)
kessel directions extract org/base org/base-abliterated -o dirs.safetensors

# re-apply them to another model, at a chosen strength, as an overlay
kessel directions apply org/finetune dirs.safetensors ./finetune-abliterated --lam 2.5
```

The two projection forms (`W` is `[out, in]`, `r` a unit vector):

- **Output side:** `W - lam * r (rᵀW)`
- **Input side:** `W - lam * (W r) rᵀ`

`lam = 1` removes the direction exactly. A larger `lam` over-projects. `apply` reports `|rᵀW'| / |rᵀW|` per tensor,
which should land near `|1 - lam|`. On FP8 weights, part of a small edit rounds away, so check that number.

## Partial fetch

```bash
kessel fetch deepseek-ai/DeepSeek-V4-Flash-0731 ./wo_b --filter '\.attn\.wo_b\.weight$'
```

Downloads only the matching tensors and their scales, using HTTP range requests against the safetensors headers. The
output directory is a normal checkpoint that every other command accepts. Use it when an edit touches a few tensors
of a huge model. The repo's index decides which shard holds each tensor, so overlay repos come out right.

An overlay built from a fetched checkpoint (`kessel merge --overlay` or `kessel directions apply`) gets the
upstream repo's full index, with only the changed tensors pointing at the new shard, plus `kessel_upstream.json`.
`kessel materialize ./out` then downloads the untouched upstream shards next to it, which makes it a complete,
loadable checkpoint. For example, re-baking an abliteration onto a finetune of a 160 GB model:

```bash
kessel fetch org/base           base_wob  --filter '\.attn\.wo_b\.weight$'
kessel fetch org/base-abliterated abl_wob --filter '\.attn\.wo_b\.weight$'
kessel fetch org/finetune       ft_wob    --filter '\.attn\.wo_b\.weight$'
kessel directions extract base_wob abl_wob -o dirs.safetensors
kessel directions apply ft_wob dirs.safetensors ./finetune-abliterated --lam 2.5
kessel materialize ./finetune-abliterated     # when you want to load it
```

## Python

```python
import kessel

report = kessel.run("config.yaml", "./out", device="cuda")
dirs = kessel.extract_directions("base", "base-abliterated")
kessel.apply_directions("finetune", dirs, "./out", lam=2.5)

# a new method is a function over float32 tensors
from kessel import Method, MergeInput, register
register(Method("mean", lambda m: sum(m.tensors) / len(m.tensors), needs_base=False,
                global_params={}, model_params={}))
```

## Tests

```bash
pytest   # parity tests skip if mergekit isn't installed
```
