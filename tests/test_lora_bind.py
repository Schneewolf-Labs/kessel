import json

import pytest
import torch

from kessel.io import tensor_to_bytes, write_safetensors
from kessel.lora import LoraAdapter

# Qwen3.5/3.8 checkpoint layout: the language model under model.language_model, plus an MTP head and a vision tower
QWEN35 = [f"model.language_model.layers.{i}.mlp.up_proj.weight" for i in range(4)] + [
    "mtp.layers.0.mlp.up_proj.weight",
    "model.visual.blocks.0.mlp.up_proj.weight",
]


def _adapter(path, modules):
    items = []
    for mod in modules:
        for kind, shape in (("lora_A", (4, 8)), ("lora_B", (8, 4))):
            data, dt = tensor_to_bytes(torch.zeros(shape))
            items.append((f"base_model.model.{mod}.{kind}.weight", dt, shape, data))
    path.mkdir(parents=True, exist_ok=True)
    write_safetensors(path / "adapter_model.safetensors", items)
    (path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 4}))
    return LoraAdapter(str(path))


def test_text_only_adapter_binds_layer_0_to_the_language_model_not_mtp(tmp_path):
    """A text-only adapter (model.layers.N) on the full checkpoint: layer 0's suffix also matches mtp.layers.0."""
    lora = _adapter(tmp_path, [f"model.layers.{i}.mlp.up_proj" for i in range(4)])
    lora.bind(QWEN35)
    assert sorted(lora._targets) == sorted(QWEN35[:4])


def test_adapter_that_names_the_mtp_head_still_binds_it(tmp_path):
    lora = _adapter(tmp_path, ["mtp.layers.0.mlp.up_proj"])
    lora.bind(QWEN35)
    assert list(lora._targets) == ["mtp.layers.0.mlp.up_proj.weight"]


def test_ambiguity_between_two_main_stacks_still_fails(tmp_path):
    lora = _adapter(tmp_path, ["layers.0.mlp.up_proj"])
    with pytest.raises(KeyError):
        lora.bind(["a.layers.0.mlp.up_proj.weight", "b.layers.0.mlp.up_proj.weight"])
