"""
Tests for the core GPTQ math in ``gptq.py``: Hessian approximation, the
Cholesky-based inverse, and single-layer quantization on a synthetic linear
layer. These exercise the algorithm's numerics without loading any model.
"""

import torch

from gptq import compute_hessian, gptq_quantize_layer
from quantize import round_to_nearest


def test_hessian_is_symmetric_and_psd():
    """H = X^T X / n + damping is symmetric and positive definite."""
    X = torch.randn(256, 32)
    H = compute_hessian(X, damp_pct=0.01)

    assert H.shape == (32, 32)
    assert torch.allclose(H, H.T, atol=1e-5)
    eigvals = torch.linalg.eigvalsh(H)
    assert eigvals.min() > 0  # damping guarantees PD => Cholesky succeeds


def test_hessian_damping_adds_to_diagonal():
    """Damping raises the diagonal by damp_pct * mean(diag) and nothing else."""
    X = torch.randn(128, 16)
    n = X.shape[0]
    raw = X.T @ X / n
    damped = compute_hessian(X, damp_pct=0.05)

    expected_damp = 0.05 * torch.mean(torch.diag(raw))
    diff = damped - raw
    # Off-diagonal entries unchanged.
    off_diag = diff - torch.diag(torch.diag(diff))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-5)
    # Diagonal raised by exactly the damping term.
    assert torch.allclose(torch.diag(diff), torch.full((16,), expected_damp.item()), atol=1e-4)


def test_gptq_cholesky_path_runs_and_preserves_shape():
    """A well-conditioned Hessian goes through the Cholesky path cleanly."""
    torch.manual_seed(0)
    W = torch.randn(20, 16)
    X = torch.randn(512, 16)
    H = compute_hessian(X)

    Q, loss = gptq_quantize_layer(W, H, n_bits=4, block_size=8)
    assert Q.shape == W.shape
    assert torch.isfinite(Q).all()
    assert loss >= 0


def test_gptq_cholesky_fallback_on_singular_hessian():
    """A rank-deficient Hessian triggers the extra-damping fallback, no crash."""
    torch.manual_seed(0)
    W = torch.randn(8, 12)
    # Rank-deficient activations => near-singular H (only the base damping).
    X = torch.randn(4, 12)  # far fewer rows than columns
    H = X.T @ X / X.shape[0]
    tiny = 1e-8 * torch.mean(torch.diag(H))
    H[range(12), range(12)] += tiny  # barely-PD => forces the fallback branch

    Q, loss = gptq_quantize_layer(W, H, n_bits=4, block_size=4)
    assert torch.isfinite(Q).all()
    assert loss >= 0


def test_gptq_beats_rtn_on_correlated_activations():
    """
    The central GPTQ claim: with a non-trivial Hessian (correlated inputs),
    error-compensated quantization reconstructs the layer output better than
    round-to-nearest. We measure ||W X - Q X|| under both methods.
    """
    torch.manual_seed(0)
    in_features, out_features, n = 48, 24, 1024

    # Correlated activations: a low-rank factor makes H strongly off-diagonal,
    # which is exactly where GPTQ's compensation helps.
    factor = torch.randn(in_features, in_features)
    X = torch.randn(n, in_features) @ factor
    W = torch.randn(out_features, in_features)

    H = compute_hessian(X)

    Q_gptq, _ = gptq_quantize_layer(W, H, n_bits=3, block_size=16)
    Q_rtn = round_to_nearest(W, n_bits=3)

    # Output reconstruction error on the calibration activations.
    err_gptq = (W @ X.T - Q_gptq @ X.T).pow(2).mean().item()
    err_rtn = (W @ X.T - Q_rtn @ X.T).pow(2).mean().item()

    assert err_gptq < err_rtn, (
        f"GPTQ ({err_gptq:.4f}) should beat RTN ({err_rtn:.4f}) on correlated activations"
    )


def test_gptq_act_order_is_permutation_invariant_in_shape():
    """act-order permutes columns internally but returns them in the original
    order, so shapes and finiteness are preserved."""
    torch.manual_seed(0)
    W = torch.randn(16, 24)
    X = torch.randn(512, 24)
    H = compute_hessian(X)

    Q, loss = gptq_quantize_layer(W, H, n_bits=4, block_size=8, act_order=True)
    assert Q.shape == W.shape
    assert torch.isfinite(Q).all()
    assert loss >= 0


def test_gptq_grouping_runs():
    """Per-group scales (group_size > 0) produce a valid quantized matrix."""
    torch.manual_seed(0)
    W = torch.randn(16, 64)
    X = torch.randn(512, 64)
    H = compute_hessian(X)

    Q, loss = gptq_quantize_layer(W, H, n_bits=4, block_size=16, group_size=16)
    assert Q.shape == W.shape
    assert torch.isfinite(Q).all()


def test_gptq_grouping_beats_per_row_on_uneven_column_scales():
    """
    Per-group scales should reconstruct better than a single per-row scale when
    column magnitudes vary wildly across the row: a global row absmax is
    dominated by the loud columns and wastes resolution on the quiet ones, while
    per-group scales adapt to each block of columns.

    We assert on the two whole-matrix quantities that are unambiguously better
    with grouping — the internal GPTQ loss and the output-space reconstruction
    error — rather than any single sub-slice, since error compensation
    redistributes error across columns.
    """
    torch.manual_seed(0)
    out_features, in_features, n = 16, 64, 4096
    group = 16

    # Left half of every row is ~100x larger than the right half.
    W = torch.randn(out_features, in_features)
    W[:, : in_features // 2] *= 100.0

    X = torch.randn(n, in_features)
    H = compute_hessian(X)

    Q_group, loss_group = gptq_quantize_layer(W, H, n_bits=4, block_size=group, group_size=group)
    Q_row, loss_row = gptq_quantize_layer(W, H, n_bits=4, block_size=group, group_size=-1)

    assert torch.isfinite(Q_group).all()
    assert loss_group < loss_row

    err_group = (W @ X.T - Q_group @ X.T).pow(2).mean().item()
    err_row = (W @ X.T - Q_row @ X.T).pow(2).mean().item()
    assert err_group < err_row


def test_gptq_higher_bits_lower_loss():
    """Higher bit-width should yield lower GPTQ reconstruction loss."""
    torch.manual_seed(0)
    W = torch.randn(24, 32)
    X = torch.randn(512, 32)
    H = compute_hessian(X)

    _, loss2 = gptq_quantize_layer(W, H, n_bits=2, block_size=16)
    _, loss4 = gptq_quantize_layer(W, H, n_bits=4, block_size=16)
    assert loss4 < loss2
