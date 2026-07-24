# Contributing

Thanks for your interest in improving **GPTQ From Scratch**. This is a small,
educational implementation, so the bar for contributions is simple: keep the
code readable, keep the tests green, and keep changes focused.

## Development setup

```bash
git clone https://github.com/Azimml/gptq-from-scratch.git
cd gptq-from-scratch
pip install -e ".[dev]"
```

The dev extra pulls in `pytest`, `ruff`, and `mypy`. PyTorch is a heavy
dependency; if you only need CPU wheels, install them first:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0,<3.0"
pip install -e ".[dev]"
```

## Before opening a pull request

Run the same checks CI runs (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
A `Makefile` wraps them:

```bash
make lint      # ruff check + ruff format --check
make test      # pytest (offline, CPU, a few seconds)
make check     # lint + type-check + test
```

Or invoke the tools directly:

```bash
ruff check .
ruff format --check .
pytest
```

The test suite builds tiny models from config, so it runs fully offline — no
model downloads or GPU required.

## Guidelines

- **One logical change per pull request.** Small, reviewable diffs merge faster.
- **Add a test** for any behavior change to the quantization math or the
  architecture accessors. New tests should build models from config (see
  `tests/conftest.py`) so they stay offline and fast.
- **Match the existing style.** Ruff enforces formatting and import order; run
  `ruff format .` before committing.
- **Adding a new model architecture?** Implement (or reuse) the accessors in
  `arch_config.py`, register the config class in `_CONFIG_CLASS_MAP`, and add a
  detection test plus a `tiny_*` fixture.

## Reporting bugs

Open an issue using the bug-report template and include the model name, the
`transformers`/`torch` versions, and the full traceback.
