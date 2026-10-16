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


def flatten(src: str | Path, out: str | Path | None = None, in_place: bool = False) -> dict:
    """Turn an overlay checkpoint into a plain one: each overlay tensor is written into the shard that held the
    original, at the same byte offsets (same dtype and shape required), so no shard carries a stale copy and
    any loader sees exactly the indexed weights. Unchanged byte ranges stay identical, which keeps uploads to
    chunk-deduplicating storage (HF Xet) small.

    in_place=True patches the shards in `src` (they must be regular files, not symlinks into a cache or another
    checkpoint); otherwise the shards are copied to `out` first.
    """
    import shutil

    from .io import SafetensorsFile

    src = Path(src)
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(wm.values()))
    missing = [f for f in files if not (src / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced shards missing (run `kessel materialize` first): "
                                f"{missing[:3]}")
    headers = {f: SafetensorsFile(src / f) for f in files}
    owner: dict[str, str] = {}  # tensor -> base shard that holds a slot for it
    overlay_files = {f for f in files if f.startswith("kessel-overlay")}
    for f in files:
        if f in overlay_files:
            continue
        for n in headers[f].tensors:
            owner.setdefault(n, f)
    patches: dict[str, list[tuple[str, str]]] = {}  # base shard -> [(tensor, overlay file)]
    for n, f in wm.items():
        if f in overlay_files:
            if n not in owner:
                raise ValueError(f"{n} exists only in the overlay; flatten needs a base slot for every tensor")
            a, b = headers[owner[n]].tensors[n], headers[f].tensors[n]
            if (a.dtype, a.shape) != (b.dtype, b.shape):
                raise ValueError(f"{n}: overlay {b.dtype}{list(b.shape)} vs base {a.dtype}{list(a.shape)}")
            patches.setdefault(owner[n], []).append((n, f))
    if in_place:
        out = src
    else:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
    new_wm = {}
    for f in files:
        if f in overlay_files:
            continue
        dst = out / f
        if in_place:
            if dst.is_symlink():
                raise ValueError(f"{dst} is a symlink; refusing to patch in place (copy it first)")
        else:
            shutil.copyfile(src / f, dst)
        if f in patches:
            with open(dst, "r+b") as fh:
                for n, of in patches[f]:
                    fh.seek(headers[f].tensors[n].start)
                    fh.write(headers[of].raw(n))
        for n in headers[f].tensors:
            if n in wm:
                new_wm[n] = f
    for h in headers.values():
        h.close()
    if set(new_wm) != set(wm):
        raise ValueError(f"flattened index lost tensors: {sorted(set(wm) - set(new_wm))[:3]}")
    total = sum((out / f).stat().st_size for f in set(new_wm.values()))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_wm.items()))}, indent=2) + "\n")
    if not in_place:
        for p in src.iterdir():
            if p.is_file() and p.suffix in {".json", ".jinja", ".model", ".txt", ".tiktoken"} \
                    and p.name not in ("model.safetensors.index.json",) and not (out / p.name).exists():
                shutil.copy(p, out / p.name)
    else:
        for f in overlay_files:
            (out / f).unlink()
    return {"patched_tensors": sum(len(v) for v in patches.values()), "shards_patched": len(patches)}
