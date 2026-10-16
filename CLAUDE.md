# Kessel

Model merging by tensor name. Merge engine for Merlina (Grimoire is the training engine).

## Philosophy

A merge method is a function over float32 tensors; the hard part is I/O, and that's shared. No per-architecture
definitions: tensors are matched by name, layers found by `<prefix>.<N>.`. Stay compatible with mergekit's YAML.
Untouched tensors move as raw bytes, so outputs are byte-identical wherever a merge didn't change anything.

## Structure

```
kessel/
├── io.py       # byte-level safetensors reader/writer (any dtype incl. F8_E8M0, packed FP4), ModelRef, ShardWriter
├── quant.py    # FP8/FP4 block-scaled codecs: detect, dequantize, requantize (reusing original scales)
├── methods.py  # merge methods (mergekit semantics) + registry
├── config.py   # mergekit YAML -> dataclasses; parameter gradients/filters
├── lora.py     # PEFT adapters applied as per-tensor deltas on read
├── merge.py    # planner (models:/slices:) + executor (copy-if-identical, overlay output)
├── ops.py      # diff, rank-1 direction extract/apply (abliteration)
├── fetch.py    # partial Hub download via range requests
└── cli.py      # `kessel merge|diff|directions|fetch`
```

## Rules

- Parity: `tests/test_parity.py` runs every method against mergekit on tiny Llamas; float32 must be bit-exact.
  A semantic change to a method needs a parity reason.
- The model index is authoritative (overlay checkpoints keep stale tensors in linked shards).
- Ruff: line length 120, rules E4/E7/E9/F.
