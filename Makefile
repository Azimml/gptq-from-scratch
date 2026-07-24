# Developer shortcuts. Mirrors the checks run in CI (.github/workflows/ci.yml).
# The module list is kept in sync with [tool.setuptools].py-modules.

MODULES := arch_config.py evaluate.py gptq.py main.py model_utils.py quantize.py

.PHONY: help install lint format type test check figures clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with dev tooling
	pip install -e ".[dev]"

lint:  ## Lint and check formatting (no changes made)
	ruff check .
	ruff format --check .

format:  ## Auto-fix lint issues and format in place
	ruff check --fix .
	ruff format .

type:  ## Type-check the source modules with mypy
	mypy $(MODULES)

test:  ## Run the offline test suite
	pytest

check: lint type test  ## Run everything CI runs

figures:  ## Regenerate the README figures
	python generate_figures.py

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
