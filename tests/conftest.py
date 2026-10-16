"""Tiny random Llama checkpoints for merge tests."""

import pytest
import torch


def _llama(path, seed, base=None, noise=0.02, layers=4):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=256, tie_word_embeddings=False, torch_dtype="bfloat16")
    torch.manual_seed(seed)
    m = LlamaForCausalLM(cfg)
    if base is not None:
        sd = base.state_dict()
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.copy_(sd[n] + noise * torch.randn_like(p) * sd[n].float().std().clamp(min=1e-3))
    m = m.to(torch.bfloat16)
    m.save_pretrained(path, safe_serialization=True)
    return m


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    base = _llama(root / "base", 0)
    paths = {"base": root / "base"}
    for i in range(1, 4):
        _llama(root / f"ft{i}", i, base=base)
        paths[f"ft{i}"] = root / f"ft{i}"
    # a LoRA on ft1
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(root / "ft1", torch_dtype=torch.float32)
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"]))
    torch.manual_seed(7)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    pm.save_pretrained(root / "lora1")
    paths["lora1"] = root / "lora1"
    return {k: str(v) for k, v in paths.items()}
