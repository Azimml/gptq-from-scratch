"""
Shared pytest fixtures and helpers.

All test models are built from config (no downloads), so the whole suite runs
fully offline and fast — suitable for CPU-only CI.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed RNGs before every test for reproducibility."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def tiny_gpt2():
    """A minimal GPT-2 (Conv1D layers) built from config, no download."""
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        n_embd=32,
        n_head=2,
        n_layer=2,
        n_positions=64,
        vocab_size=128,
    )
    return GPT2LMHeadModel(cfg).eval()


@pytest.fixture
def tiny_llama():
    """A minimal LLaMA (Linear layers, RoPE, GQA) built from config."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def tiny_qwen2():
    """A minimal Qwen2 model built from config (new architecture support)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(cfg).eval()


def make_calib(n_samples=4, seq_len=16, vocab_size=128):
    """Build synthetic calibration data: list of (1, seq_len) token tensors."""
    return [torch.randint(0, vocab_size, (1, seq_len)) for _ in range(n_samples)]
