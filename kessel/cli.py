"""kessel CLI: merge | diff | directions extract | directions apply | fetch."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _merge(a):
    from .merge import run

    rep = run(a.config, a.out, device=a.device, overlay=a.overlay, max_shard_bytes=int(a.shard_gb * 1024**3),
              keep_quantized=not a.dequantize, progress=not a.quiet)
    print(json.dumps({"tensors": rep.tensors, "copied": rep.copied, "computed": rep.computed,
                      "unchanged_after_compute": rep.unchanged_after_compute,
                      "requant_blocks_rescaled": rep.requant.blocks_rescaled,
                      "requant_bytes_changed": rep.requant.bytes_changed}, indent=1))


def _diff(a):
    from .ops import diff

    diffs, only_a, only_b = diff(a.a, a.b, device=a.device, pattern=a.filter, svd=not a.no_svd)
    for d in diffs:
        r1 = f"{d.rank1_energy:.3f}" if d.rank1_energy is not None else "-"
        print(f"{d.name}\t{d.dtype}\t{list(d.shape)}\trel={d.rel_norm:.3e}\trank1={r1}")
    print(f"# {len(diffs)} tensors differ; only in A: {len(only_a)}; only in B: {len(only_b)}", file=sys.stderr)
    for n in only_a[:20]:
        print(f"# only in A: {n}", file=sys.stderr)
    for n in only_b[:20]:
        print(f"# only in B: {n}", file=sys.stderr)


def _extract(a):
    from .ops import extract_directions, save_directions

    dirs = extract_directions(a.base, a.edited, pattern=a.filter, device=a.device, side=a.side)
    for d in dirs:
        print(f"{d.name}\tside={d.side}\tlam={d.lam:.4f}\tfit={d.fit:.4f}")
    save_directions(dirs, a.out)
    print(f"# {len(dirs)} directions -> {a.out}", file=sys.stderr)


def _apply(a):
    from .ops import apply_directions, load_directions

    dirs = load_directions(a.directions)
    rep = apply_directions(a.model, dirs, a.out, lam=a.lam, device=a.device, overlay=not a.full_copy)
    worst = max(rep.residual.values()) if rep.residual else float("nan")
    print(json.dumps({"edited": len(rep.edited), "requant_blocks_rescaled": rep.requant.blocks_rescaled,
                      "requant_bytes_changed": rep.requant.bytes_changed, "worst_residual": worst}, indent=1))


def _fetch(a):
    from .fetch import fetch

    print(json.dumps(fetch(a.repo, a.filter, a.out, revision=a.revision), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kessel", description="Model merging, by tensor name.")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge", help="run a merge config (mergekit YAML)")
    m.add_argument("config")
    m.add_argument("out")
    m.add_argument("--device", default="cpu")
    m.add_argument("--overlay", action="store_true",
                   help="write only tensors that differ from the base; symlink the base's shards")
    m.add_argument("--dequantize", action="store_true", help="write FP8/FP4 weights as dtype instead of requantizing")
    m.add_argument("--shard-gb", type=float, default=5.0)
    m.add_argument("--quiet", action="store_true")
    m.set_defaults(fn=_merge)

    d = sub.add_parser("diff", help="list tensors that differ between two checkpoints")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--filter", help="regex on tensor names")
    d.add_argument("--device", default="cpu")
    d.add_argument("--no-svd", action="store_true")
    d.set_defaults(fn=_diff)

    dr = sub.add_parser("directions", help="rank-1 edits (abliteration)")
    drs = dr.add_subparsers(dest="dcmd", required=True)
    e = drs.add_parser("extract", help="recover per-tensor directions from base vs edited")
    e.add_argument("base")
    e.add_argument("edited")
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--filter")
    e.add_argument("--side", choices=["auto", "output", "input"], default="auto")
    e.add_argument("--device", default="cpu")
    e.set_defaults(fn=_extract)
    p = drs.add_parser("apply", help="project directions out of a model")
    p.add_argument("model")
    p.add_argument("directions")
    p.add_argument("out")
    p.add_argument("--lam", type=float, help="strength (default: each direction's fitted lam)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--full-copy", action="store_true", help="write every tensor instead of an overlay")
    p.set_defaults(fn=_apply)

    f = sub.add_parser("fetch", help="download only the tensors matching a regex from a Hub repo (range requests)")
    f.add_argument("repo")
    f.add_argument("out")
    f.add_argument("--filter", required=True, help="regex on tensor names (scales come along)")
    f.add_argument("--revision")
    f.set_defaults(fn=_fetch)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
