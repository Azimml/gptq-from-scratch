"""
Fast end-to-end smoke tests for the full block-by-block GPTQ pipeline
(``gptq.quantize_model``) on tiny models built from config.

These run on CPU in a few seconds and never touch the network, so they are safe
for CI. They verify the pipeline completes, mutates weights, and leaves the
model producing finite logits — across GPT-2 (Conv1D) and Qwen2 (the new arch).
"""

import io
from contextlib import redirect_stdout

import pytest
import torch

from gptq import quantize_model
from model_utils import get_weight_and_type
from tests.conftest import make_calib


def _quiet_quantize(model, calib, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return quantize_model(model, calib, device="cpu", **kwargs)


def test_end_to_end_gpt2_changes_weights_and_stays_finite(tiny_gpt2):
    model = tiny_gpt2
    calib = make_calib(n_samples=4, seq_len=16)

    # Snapshot one quantizable weight before quantization.
    first_conv = model.transformer.h[0].attn.c_attn
    before, _ = get_weight_and_type(first_conv)

    stats = _quiet_quantize(model, calib, n_bits=4, block_size=8)

    assert len(stats) > 0
    for s in stats.values():
        assert s["loss"] >= 0
        assert s["time"] >= 0

    after, _ = get_weight_and_type(first_conv)
    assert not torch.allclose(before, after), "weights should be quantized"

    with torch.no_grad():
        logits = model(torch.randint(0, 128, (1, 16))).logits
    assert torch.isfinite(logits).all()


def test_end_to_end_qwen2_new_architecture(tiny_qwen2):
    """The headline new capability: GPTQ on a Qwen2 model, end to end."""
    model = tiny_qwen2
    calib = make_calib(n_samples=4, seq_len=16)

    stats = _quiet_quantize(model, calib, n_bits=4, block_size=8)

    # Qwen2 block: q/k/v/o_proj + gate/up/down_proj = 7 linears per layer * 2.
    assert len(stats) == 14
    assert all(name.startswith("model.layers.") for name in stats)

    with torch.no_grad():
        logits = model(torch.randint(0, 128, (1, 16))).logits
    assert torch.isfinite(logits).all()


def test_end_to_end_qwen2_all_optimizations(tiny_qwen2):
    """Grouping + act-order + true-sequential all run on the new arch."""
    model = tiny_qwen2
    calib = make_calib(n_samples=4, seq_len=16)

    stats = _quiet_quantize(
        model,
        calib,
        n_bits=4,
        block_size=8,
        group_size=16,
        act_order=True,
        true_sequential=True,
    )
    assert len(stats) == 14

    with torch.no_grad():
        logits = model(torch.randint(0, 128, (1, 16))).logits
    assert torch.isfinite(logits).all()


def test_quantize_model_rejects_empty_calibration(tiny_gpt2):
    """An empty calibration set fails fast with a clear message, not an opaque
    IndexError deep in the pipeline."""
    with pytest.raises(ValueError, match="calibration_data is empty"):
        quantize_model(tiny_gpt2, [], device="cpu")


def test_end_to_end_llama(tiny_llama):
    """Regression guard for the LLaMA path (RoPE + GQA)."""
    model = tiny_llama
    calib = make_calib(n_samples=4, seq_len=16)

    stats = _quiet_quantize(model, calib, n_bits=4, block_size=8)
    assert len(stats) == 14

    with torch.no_grad():
        logits = model(torch.randint(0, 128, (1, 16))).logits
    assert torch.isfinite(logits).all()
