"""
Tests for the symmetric-uniform quantization primitives in ``quantize.py``.
"""

import torch

from quantize import (
    compute_row_scales,
    quantize_column,
    quantize_tensor,
    round_to_nearest,
)


def test_quantize_tensor_shape_and_range():
    """Dequantized values lie on a grid bounded by the integer range."""
    w = torch.randn(64, 32)
    n_bits = 4
    w_hat, scale = quantize_tensor(w, n_bits=n_bits)

    assert w_hat.shape == w.shape
    assert scale > 0

    qmax = 2 ** (n_bits - 1) - 1  # 7
    qmin = -(2 ** (n_bits - 1))  # -8
    # Recover integer levels; they must sit within [qmin, qmax].
    levels = torch.round(w_hat / scale)
    assert levels.min() >= qmin
    assert levels.max() <= qmax


def test_quantize_tensor_error_shrinks_with_more_bits():
    """More bits => finer grid => lower reconstruction error."""
    w = torch.randn(128, 128)
    err = {}
    for n_bits in (2, 3, 4, 8):
        w_hat, _ = quantize_tensor(w, n_bits=n_bits)
        err[n_bits] = (w - w_hat).pow(2).mean().item()
    assert err[2] > err[3] > err[4] > err[8]


def test_quantize_tensor_scale_matches_absmax():
    """Per-tensor scale is |w|.max / qmax."""
    w = torch.tensor([-3.0, 1.5, 0.25, 2.0])
    n_bits = 4
    _, scale = quantize_tensor(w, n_bits=n_bits)
    qmax = 2 ** (n_bits - 1) - 1
    assert torch.isclose(scale, w.abs().max() / qmax)


def test_quantize_tensor_handles_all_zeros():
    """A zero tensor must not divide by zero and must dequantize to zero."""
    w = torch.zeros(8, 8)
    w_hat, scale = quantize_tensor(w, n_bits=4)
    assert torch.all(w_hat == 0)
    assert scale >= 1e-10  # clamped, never zero


def test_compute_row_scales_is_per_output_channel():
    """One scale per row (output channel), each = row absmax / qmax."""
    W = torch.tensor([[1.0, -2.0, 0.5], [4.0, 0.0, -1.0]])
    n_bits = 4
    scales = compute_row_scales(W, n_bits=n_bits)
    qmax = 2 ** (n_bits - 1) - 1
    assert scales.shape == (2,)
    assert torch.isclose(scales[0], torch.tensor(2.0 / qmax))
    assert torch.isclose(scales[1], torch.tensor(4.0 / qmax))


def test_quantize_column_uses_per_row_scale():
    """Each element of a column is quantized with its own row scale."""
    W = torch.tensor([[1.0, -2.0], [4.0, -1.0]])
    scales = compute_row_scales(W, n_bits=4)
    col = W[:, 0]
    q = quantize_column(col, scales, n_bits=4)
    assert q.shape == col.shape
    # Reconstruction should be close to the original for well-scaled inputs.
    assert torch.allclose(q, col, atol=scales.max().item())


def test_quantize_column_clamps_to_range():
    """Values beyond the grid are clamped, not wrapped."""
    scales = torch.tensor([1.0])
    col = torch.tensor([100.0])  # far above qmax=7 at 4 bits
    q = quantize_column(col, scales, n_bits=4)
    qmax = 2 ** (4 - 1) - 1
    assert q.item() == qmax * scales.item()


def test_round_to_nearest_matches_quantize_tensor():
    """The RTN baseline is just per-tensor quantization without the scale."""
    w = torch.randn(16, 16)
    rtn = round_to_nearest(w, n_bits=4)
    w_hat, _ = quantize_tensor(w, n_bits=4)
    assert torch.allclose(rtn, w_hat)


def test_symmetric_quantization_preserves_sign():
    """Symmetric quantization keeps the sign of large-magnitude weights."""
    w = torch.tensor([[-5.0, 5.0, -3.0, 3.0]])
    w_hat, _ = quantize_tensor(w, n_bits=4)
    assert torch.sign(w_hat).equal(torch.sign(w))
