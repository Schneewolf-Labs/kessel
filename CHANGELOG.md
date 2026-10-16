# Changelog

## [0.1.0] - 2026-10-16

### Added
- `kessel merge`: mergekit-compatible YAML (`models:`, `slices:`, gradients, `filter:` conditions, `model+lora`,
  `tokenizer_source`). Methods: linear, task_arithmetic, ties, dare_ties, dare_linear, breadcrumbs,
  breadcrumbs_ties, della, della_linear, slerp, nearswap, model_stock, arcee_fusion, passthrough. Bit-exact with
  mergekit 0.1.4 in float32.
- Tensor matching by name, no architecture definitions (keeps vision towers and MTP heads).
- FP8 (e4m3, ue8m0/float32 block scales) and FP4 (e2m1, packed) checkpoints: merge in float32, requantize
  reusing original block scales; untouched tensors stay byte-identical.
- `--overlay` output: only changed tensors are written, the base's shards are linked.
- `kessel diff`, `kessel directions extract|apply` (rank-1 edits such as abliteration).
- `kessel fetch`: download only matching tensors of a Hub repo via range requests.
- Overlays of fetched (partial) checkpoints carry the upstream's full index; `kessel materialize` downloads the
  untouched shards; `kessel flatten` writes overlay tensors into the base shards in place (a plain checkpoint any
  loader reads correctly).
