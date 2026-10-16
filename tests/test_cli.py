import json

import yaml

from kessel.cli import main
from kessel.io import ModelFiles


def test_cli_merge_and_diff(tiny, tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"models": [{"model": tiny["ft1"], "parameters": {"weight": 1.0}}],
                                   "merge_method": "task_arithmetic", "base_model": tiny["base"]}))
    main(["merge", str(cfg), str(tmp_path / "out"), "--quiet"])
    rep = json.loads(capsys.readouterr().out)
    assert rep["tensors"] == len(ModelFiles(tiny["base"]).names())
    assert (tmp_path / "out" / "config.json").exists() and (tmp_path / "out" / "kessel_config.yaml").exists()
    main(["diff", tiny["base"], str(tmp_path / "out"), "--filter", "layers\\.0\\."])
    out = capsys.readouterr().out
    assert "model.layers.0.self_attn.q_proj.weight" in out


def test_config_gradients_match_mergekit():
    import pytest

    mk = pytest.importorskip("mergekit.config")
    from kessel.config import evaluate

    cases = [0.5, [0, 1], [0, 0.5, 0.3, 0.7, 1], [1, 2, 3, 4, 5, 6, 7]]
    for c in cases:
        for t in [0, 0.1, 0.25, 0.5, 0.77, 1]:
            assert evaluate(c, "model.layers.3.mlp.up_proj.weight", t) == pytest.approx(
                mk.evaluate_setting("model.layers.3.mlp.up_proj.weight", c, t))
    conds = [{"filter": "self_attn", "value": [0, 1]}, {"filter": "mlp", "value": 0.3}, {"value": 0.5}]
    mk_conds = [mk.ConditionalParameter(**c) for c in conds]
    for name in ["model.layers.1.self_attn.q_proj.weight", "model.layers.1.mlp.up_proj.weight", "lm_head.weight"]:
        assert evaluate(conds, name, 0.4) == pytest.approx(mk.evaluate_setting(name, mk_conds, 0.4))
