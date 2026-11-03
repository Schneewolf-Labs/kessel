import json

import pytest
import torch

from kessel.cli import main
from kessel.cvec import load_control_vector, mean_difference, save_control_vector


def test_roundtrip_hf_index_and_scale(tmp_path):
    torch.manual_seed(0)
    v = torch.randn(6, 16)  # HF-indexed: row j = output of decoder layer j-1
    p = tmp_path / "c.gguf"
    written = save_control_vector(v, p, scale=0.5, layers=range(2, 5), hf_index=True, model_hint="llama")
    assert written == [1, 2, 3]
    got, meta = load_control_vector(p)
    assert meta["general.architecture"] == "controlvector" and meta["controlvector.layer_count"] == 3
    assert meta["controlvector.model_hint"] == "llama"
    for j in range(2, 5):
        assert torch.allclose(got[j - 1], 0.5 * v[j])


def test_layer0_cannot_be_steered(tmp_path):
    v = torch.randn(4, 8)
    with pytest.raises(ValueError, match="direction.0"):
        save_control_vector(v, tmp_path / "c.gguf", layers=[1], hf_index=True)
    # without an explicit layer list, rows that would land on direction.0 (or the embeddings) are skipped
    assert save_control_vector(v, tmp_path / "c.gguf", hf_index=True) == [1, 2]


def test_mean_difference():
    pos, neg = torch.ones(3, 5, 4), torch.zeros(2, 5, 4)
    assert torch.equal(mean_difference(pos, neg), torch.ones(5, 4))


def test_readable_by_gguf_py(tmp_path):
    gguf = pytest.importorskip("gguf")
    v = {3: torch.arange(8.0), 5: -torch.arange(8.0)}
    p = tmp_path / "c.gguf"
    save_control_vector(v, p)
    r = gguf.GGUFReader(str(p))
    assert bytes(r.fields["general.architecture"].parts[-1]).decode() == "controlvector"
    t = {x.name: x.data for x in r.tensors}
    assert sorted(t) == ["direction.3", "direction.5"]
    assert t["direction.5"].tolist() == (-torch.arange(8.0)).tolist()


def test_reads_gguf_py_written_file(tmp_path):
    gguf = pytest.importorskip("gguf")
    import numpy as np

    p = tmp_path / "w.gguf"
    w = gguf.GGUFWriter(str(p), "controlvector")
    w.add_string("controlvector.model_hint", "qwen35")
    w.add_int32("controlvector.layer_count", 2)
    w.add_tensor("direction.1", np.full(16, 2.0, dtype=np.float32))
    w.add_tensor("direction.7", np.arange(16, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    got, meta = load_control_vector(p)
    assert meta["controlvector.model_hint"] == "qwen35"
    assert list(got) == [1, 7] and got[7].tolist() == list(range(16)) and float(got[1].sum()) == 32.0


def test_hf_hidden_states_are_layer_outputs_shifted_by_one():
    # The hf_index convention: hidden_states[j] is decoder layer j-1's output (layer k <-> direction.k in llama.cpp).
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                     num_attention_heads=4, num_key_value_heads=2, vocab_size=64)).eval()
    outs = {}
    hooks = [layer.register_forward_hook(lambda mod, i, o, k=k: outs.__setitem__(k, o[0] if isinstance(o, tuple) else o))
             for k, layer in enumerate(m.model.layers)]
    with torch.no_grad():
        hs = m(torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True).hidden_states
    for h in hooks:
        h.remove()
    for k in range(2):  # the last hidden state has the final norm applied
        assert torch.allclose(hs[k + 1], outs[k])


def test_cli_export_and_show(tmp_path, capsys):
    torch.manual_seed(1)
    src = tmp_path / "dirs.pt"
    torch.save({"search": torch.randn(10, 8), "other": torch.randn(10, 8)}, src)
    out = tmp_path / "s.gguf"
    with pytest.raises(SystemExit):
        main(["cvec", "export", str(src), str(out)])
    capsys.readouterr()
    main(["cvec", "export", str(src), str(out), "--key", "search", "--layers", "4-6", "--hf-index", "--scale", "2"])
    rep = json.loads(capsys.readouterr().out)
    assert rep["llama_cpp_layers"] == [3, 4, 5]
    main(["cvec", "show", str(out)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("# ") and len(lines) == 4 and lines[1].startswith("direction.3\tn_embd=8")
