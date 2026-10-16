"""Overlay checkpoints: only the tensors an operation changed are written, everything else is referenced.

Two cases:
- The source is a full local checkpoint: its shards are symlinked into the output and the index points at them.
- The source is a partial checkpoint from `kessel fetch`: the output index is the *upstream repo's* full index with
  the changed tensors pointed at the overlay shard, and `kessel_upstream.json` records repo + revision. The
  output is then a complete description of the new model; `kessel materialize` downloads the untouched shards
  next to it to make it loadable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

UPSTREAM_FILE = "kessel_upstream.json"


def _split(src: str) -> tuple[str, str]:
    repo, _, rev = src.rpartition("@")
    return repo, rev


def partial_upstream(model_path: Path) -> tuple[str, str] | None:
    """(repo, revision) if `model_path` is a partial checkpoint written by `kessel fetch`."""
    p = model_path / "model.safetensors.index.json"
    if p.exists():
        src = (json.loads(p.read_text()).get("metadata") or {}).get("partial_of")
        if src:
            return _split(src)
    return None


def upstream_weight_map(repo: str, revision: str) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision)).read_text())[
        "weight_map"]


def finish(src_path: Path, out: Path, written: dict[str, str], linked: dict[str, str]):
    """Write the overlay's index. `written`: name -> new shard file; `linked`: name -> unchanged source shard file."""
    up = partial_upstream(src_path)
    if up:
        wm = upstream_weight_map(*up)
        missing = [n for n in written if n not in wm]
        if missing:
            raise KeyError(f"overlay tensors not in {up[0]}: {missing[:3]}")
        wm.update(written)
        (out / UPSTREAM_FILE).write_text(json.dumps({"partial_of": f"{up[0]}@{up[1]}",
                                                     "overlay": sorted(set(written.values()))}, indent=2) + "\n")
        total = None
    else:
        for fname in sorted(set(linked.values())):
            if not (src_path / fname).exists():
                raise FileNotFoundError(f"{src_path / fname} is referenced but missing (materialize the source first)")
            dst = out / fname
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(os.path.relpath((src_path / fname).resolve(), out.resolve()))
        wm = {**linked, **written}
        total = sum((out / f).stat().st_size for f in set(wm.values()))
    meta = {"total_size": total} if total is not None else {}
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": meta, "weight_map": dict(sorted(wm.items()))}, indent=2) + "\n")
    stray = out / "kessel-overlay.safetensors.index.json"
    if stray.exists():
        stray.unlink()


def materialize(out: str | Path) -> list[str]:
    """Download the upstream shards an overlay built from a partial checkpoint still references."""
    from huggingface_hub import hf_hub_download

    out = Path(out)
    if not (out / UPSTREAM_FILE).exists():
        raise ValueError(f"{out} has no {UPSTREAM_FILE}; it isn't an overlay of a fetched partial checkpoint")
    up = _split(json.loads((out / UPSTREAM_FILE).read_text())["partial_of"])
    wm = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    need = sorted(f for f in set(wm.values()) if not (out / f).exists())
    for f in need:
        hf_hub_download(up[0], f, revision=up[1], local_dir=out)
    return need
