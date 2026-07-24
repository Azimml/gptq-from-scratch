"""
Minimal, self-contained demo of the central GPTQ claim, with no model download.

It builds a synthetic linear layer with *correlated* inputs (which makes the
Hessian strongly off-diagonal — exactly where error compensation helps), then
compares the output-reconstruction error of GPTQ against naive round-to-nearest.

Run from the repository root:

    python examples/gptq_vs_rtn_synthetic.py
"""

import os
import sys

import torch

# Allow running as a script from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptq import compute_hessian, gptq_quantize_layer  # noqa: E402
from quantize import round_to_nearest  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    in_features, out_features, n_samples = 256, 128, 4096
    n_bits = 3

    # Correlated activations: X = Z @ factor gives a non-trivial covariance,
    # so the Hessian H = X^T X has meaningful off-diagonal structure.
    factor = torch.randn(in_features, in_features)
    X = torch.randn(n_samples, in_features) @ factor
    W = torch.randn(out_features, in_features)

    H = compute_hessian(X)

    Q_gptq, loss = gptq_quantize_layer(W, H, n_bits=n_bits, block_size=64)
    Q_rtn = round_to_nearest(W, n_bits=n_bits)

    # Output-space reconstruction error on the calibration activations.
    err_gptq = (W @ X.T - Q_gptq @ X.T).pow(2).mean().item()
    err_rtn = (W @ X.T - Q_rtn @ X.T).pow(2).mean().item()

    print(f"Layer: {out_features} x {in_features}, {n_bits}-bit, {n_samples} calibration rows")
    print(f"  GPTQ  output MSE: {err_gptq:.4f}  (internal loss {loss:.2f})")
    print(f"  RTN   output MSE: {err_rtn:.4f}")
    print(f"  GPTQ is {err_rtn / err_gptq:.1f}x lower error than round-to-nearest")


if __name__ == "__main__":
    main()
