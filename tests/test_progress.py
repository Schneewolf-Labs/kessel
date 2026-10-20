from kessel.io import ModelFiles
from kessel.merge import run


def test_progress_callback_counts_every_output_tensor(tiny, tmp_path):
    calls = []
    cfg = {"models": [{"model": str(tiny["ft1"])}, {"model": str(tiny["ft2"])}],
           "merge_method": "model_stock", "base_model": str(tiny["base"])}
    rep = run(cfg, tmp_path / "out", progress=lambda done, total: calls.append((done, total)))
    n = len(ModelFiles(tiny["base"]).names())
    assert rep.tensors == n
    assert calls == [(i, n) for i in range(n + 1)]


def test_missing_local_path_says_so(tmp_path):
    import pytest

    from kessel.io import resolve

    for missing in (str(tmp_path / "nope"), "./nope", "~/nope-kessel", "a/b/c"):
        with pytest.raises(FileNotFoundError, match="model not found"):
            resolve(missing)
