---
name: Bug report
about: Report something that does not work as expected
title: "[bug] "
labels: bug
---

## Description

A clear description of what went wrong.

## Reproduction

The exact command you ran, e.g.:

```bash
python main.py --model gpt2 --quantize --bits 4
```

## Expected vs. actual

- **Expected:** ...
- **Actual:** ...

## Traceback

```
paste the full traceback here
```

## Environment

- Model: <!-- e.g. gpt2, facebook/opt-1.3b, Qwen/Qwen2.5-0.5B -->
- OS / Python: <!-- e.g. Ubuntu 22.04, Python 3.11 -->
- `torch` version: <!-- python -c "import torch; print(torch.__version__)" -->
- `transformers` version: <!-- python -c "import transformers; print(transformers.__version__)" -->
- Device: <!-- cpu / cuda -->
