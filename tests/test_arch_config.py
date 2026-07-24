"""
Tests for architecture auto-detection and the per-architecture accessors in
``arch_config.py`` — including Qwen2 support and the block-output unwrapping
helper that keeps error propagation correct across transformers versions.
"""

import pytest
import torch

from arch_config import (
    GPT2_CONFIG,
    LLAMA_CONFIG,
    OPT_CONFIG,
    QWEN2_CONFIG,
    _unwrap_block_output,
    get_arch_config,
)


def test_unwrap_block_output_from_tensor():
    """A bare tensor (transformers 5.x GPT-2/OPT) is returned as-is."""
    t = torch.randn(1, 4, 8)
    assert _unwrap_block_output(t) is t


def test_unwrap_block_output_from_tuple():
    """A tuple (older transformers) yields its first element."""
    t = torch.randn(1, 4, 8)
    assert _unwrap_block_output((t, "cache", None)) is t


def test_unwrap_block_output_from_list():
    t = torch.randn(1, 4, 8)
    assert _unwrap_block_output([t]) is t


def test_unwrap_block_output_from_model_output():
    """A ModelOutput-like object exposes .last_hidden_state."""

    class FakeOutput:
        last_hidden_state = torch.randn(1, 4, 8)

    out = FakeOutput()
    assert _unwrap_block_output(out) is out.last_hidden_state


def test_unwrap_block_output_rejects_unknown():
    with pytest.raises(TypeError):
        _unwrap_block_output(object())


def test_detect_gpt2(tiny_gpt2):
    assert get_arch_config(tiny_gpt2) is GPT2_CONFIG


def test_detect_llama(tiny_llama):
    assert get_arch_config(tiny_llama) is LLAMA_CONFIG


def test_detect_qwen2(tiny_qwen2):
    """Qwen2 is the newly added architecture."""
    assert get_arch_config(tiny_qwen2) is QWEN2_CONFIG


def test_unsupported_arch_raises():
    class FakeConfig:
        pass

    class FakeModel:
        config = FakeConfig()

    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_arch_config(FakeModel())


@pytest.mark.parametrize(
    "fixture_name,expected_batched",
    [
        ("tiny_gpt2", False),  # GPT-2 blocks drop the batch dim in tf 5.x
        ("tiny_llama", True),
        ("tiny_qwen2", True),
    ],
)
def test_accessors_and_block_forward(request, fixture_name, expected_batched):
    """
    Every supported architecture must expose working accessors and a
    block_forward that returns a finite hidden-state tensor of the right size.
    """
    model = request.getfixturevalue(fixture_name)
    arch = get_arch_config(model)
    ids = torch.randint(0, 128, (1, 16))

    emb = arch.compute_embeddings(model, ids, "cpu")
    assert (
        emb.shape[-1] == model.config.hidden_size
        if hasattr(model.config, "hidden_size")
        else emb.shape[-1] == model.config.n_embd
    )

    kwargs = arch.get_block_kwargs(model, ids, "cpu")
    blocks = arch.get_blocks(model)
    assert len(blocks) == 2

    out = arch.block_forward(blocks[0], emb, **kwargs)
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).all()
    # Same hidden dim as input, regardless of whether the batch dim survives.
    assert out.shape[-1] == emb.shape[-1]


def test_qwen2_uses_llama_style_sublayer_groups():
    """Qwen2 shares LLaMA's q/k/v -> o -> gate/up -> down grouping."""
    assert QWEN2_CONFIG.sublayer_groups == LLAMA_CONFIG.sublayer_groups
    assert QWEN2_CONFIG.layer_name_prefix == "model.layers"


def test_get_max_seq_len(tiny_qwen2):
    assert QWEN2_CONFIG.get_max_seq_len(tiny_qwen2) == 128


def test_opt_config_registered():
    """OPT remains supported (regression guard)."""
    assert OPT_CONFIG.layer_name_prefix == "model.decoder.layers"
