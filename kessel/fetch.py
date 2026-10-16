"""Fetch a subset of a Hub checkpoint's tensors with HTTP range requests.

Reads each shard's safetensors header (a few KB), then downloads only the byte ranges of tensors whose names match.
Quantization scales are pulled along with their weights. The result is a normal local checkpoint directory
(one safetensors file + index + the repo's small json/tokenizer files) that every kessel command accepts, so a
rank-1 edit on 46 tensors of a 160 GB model needs ~1.5 GB of traffic instead of the whole checkpoint.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

from . import quant as Q
from .io import TensorInfo, write_safetensors

SMALL_FILES = re.compile(r"^[^/]+\.(json|jinja|model|txt|tiktoken)$")


def _session_get(url: str, headers: dict, rng: tuple[int, int] | None = None) -> bytes:
    from huggingface_hub.utils import get_session

    h = dict(headers)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
    sess = get_session()
    # huggingface_hub >= 1.0 hands out an httpx client, older versions a requests session
    redirect_kw = "follow_redirects" if type(sess).__module__.startswith("httpx") else "allow_redirects"
    r = sess.get(url, headers=h, timeout=120, **{redirect_kw: True})
    r.raise_for_status()
    if rng and len(r.content) != rng[1] - rng[0]:
        raise IOError(f"short range read from {url}: wanted {rng[1] - rng[0]} bytes, got {len(r.content)}")
    return r.content


def fetch(repo: str, pattern: str, out: str | Path, revision: str | None = None, progress: bool = True) -> dict:
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.model_info(repo, revision=revision)
    sha = info.sha
    files = [s.rfilename for s in info.siblings]
    headers = build_hf_headers()
    for f in files:
        if SMALL_FILES.match(f) and f != "model.safetensors.index.json":
            hf_hub_download(repo, f, revision=sha, local_dir=out)

    # the index is authoritative: an overlay repo's original shards still hold the tensors it replaced
    weight_map = None
    if "model.safetensors.index.json" in files:
        idx = hf_hub_download(repo, "model.safetensors.index.json", revision=sha)
        weight_map = json.loads(Path(idx).read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(f for f in files if f.endswith(".safetensors") and "/" not in f)
    rx = re.compile(pattern)
    plan: list[tuple[str, TensorInfo]] = []
    for shard in shards:
        url = hf_hub_url(repo, shard, revision=sha)
        (n,) = struct.unpack("<Q", _session_get(url, headers, (0, 8)))
        header = json.loads(_session_get(url, headers, (8, 8 + n)))
        header.pop("__metadata__", None)
        tinfo = {k: TensorInfo(k, v["dtype"], tuple(v["shape"]), Path(shard), 8 + n + v["data_offsets"][0],
                               8 + n + v["data_offsets"][1]) for k, v in header.items()}
        if weight_map is not None:
            tinfo = {k: v for k, v in tinfo.items() if weight_map.get(k) == shard}
        want = {k for k in tinfo if rx.search(k)}
        for k in list(want):
            spec = Q.scale_name_for(k, tinfo)
            if spec:
                want.add(spec)
        plan += [(url, tinfo[k]) for k in sorted(want)]
    if not plan:
        raise ValueError(f"no tensors in {repo} match {pattern!r}")

    items = []
    it = plan
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(plan, desc=f"fetch {repo}", unit="tensor")
        except ImportError:
            pass
    for url, ti in it:
        data = _session_get(url, headers, (ti.start, ti.end)) if ti.end > ti.start else b""
        items.append((ti.name, ti.dtype, ti.shape, data))
    fname = "kessel-partial.safetensors"
    write_safetensors(out / fname, items, metadata={"format": "pt", "kessel_source": f"{repo}@{sha}",
                                                    "kessel_filter": pattern})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"partial_of": f"{repo}@{sha}", "filter": pattern},
         "weight_map": {name: fname for name, *_ in items}}, indent=2) + "\n")
    return {"repo": repo, "revision": sha, "tensors": len(items), "bytes": sum(len(d) for *_, d in items)}
