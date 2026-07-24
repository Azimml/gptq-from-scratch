# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Developer tooling: `Makefile`, `.editorconfig`, `.gitattributes`, and a
  pre-commit config.
- Contributing guide, GitHub issue templates, and a pull request template.
- `examples/` directory with a runnable quantization script.
- `__all__` declarations on the public modules to define their exported API.
- Input validation for bit-width and calibration data with clear error messages.

## [0.1.0]

### Added
- From-scratch GPTQ implementation: Hessian approximation, Cholesky-based
  inverse, and column-wise error compensation.
- Multi-architecture support (GPT-2, OPT, LLaMA, Mistral, Qwen2) with automatic
  detection behind an `ArchConfig` abstraction.
- Three optimizations from the paper: grouping, act-order, and true-sequential.
- Memory-efficient block-by-block calibration loop.
- WikiText-2 perplexity evaluation with a sliding window.
- pytest suite covering the quantization primitives, GPTQ numerics,
  architecture accessors, and an offline end-to-end smoke test.
- GitHub Actions CI running lint, type-check, and tests on Python 3.10–3.12.

[Unreleased]: https://github.com/Azimml/gptq-from-scratch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Azimml/gptq-from-scratch/releases/tag/v0.1.0
