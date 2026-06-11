# RESEARCH KNOWLEDGE BASE

## OVERVIEW

`research/` contains optional MLX/PyTorch reference and diagnostic scripts. These
are not default health gates and may require local packages, Apple Silicon, or
ignored model/data assets.

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| MLX MNIST training | `mlx_reference.py` | Reference training/compare script |
| PyTorch MNIST training | `pytorch_reference.py` | Torch/Torchvision comparison path |
| MLX inference | `mlx_inference.py` | MNIST latency/throughput comparison |
| PyTorch inference | `pytorch_inference.py` | MPS/CPU comparison path |
| Bonsai MLX bench | `mlx_bonsai.py` | Full-precision local model benchmark |
| Q4 golden tokens | `mlx_bonsai_q4_golden.py` | Token capture for engine parity |
| Q4 compare | `mlx_q4_compare.py`, `mlx_q4_f16_test.py` | Prototype diagnostics |
| Q4 layers | `mlx_q4_layer*.py` | Activation dump/debug utilities |

## CONVENTIONS

- Treat scripts as local diagnostics. Do not make them prerequisites for
  `engine` or `labrat` default gates.
- Prefer current engine artifact names when comparing results:
  `artifacts/bolt-mnist-run*.json`, `artifacts/bolt-bonsai-bench.json`,
  `artifacts/bolt-q4-golden.json`, and `artifacts/bolt-q4-bench.json`.
- MNIST data should resolve from `data/mnist_torch/MNIST/raw` unless a script
  explicitly documents another checked path.
- Q4 scripts should prefer `data/qwen3-1.7b-q4-gs64`, then diagnostic
  `data/qwen3-1.7b-q4`, then matching `~/models/` directories.
- Full-precision Bonsai defaults to `~/models/bonsai-1.7b`; the local
  `data/bonsai-1.7b` path may be an ignored symlink.
- Keep Python dependencies out of repo manifests unless the project intentionally
  promotes these scripts from optional diagnostics to supported tooling.

## ANTI-PATTERNS

- Do not trust stale help text or glob patterns blindly; several scripts have
  historically written under `benchmarks/` while comparing under `artifacts/`.
- Do not commit `__pycache__`, downloaded datasets, model weights, benchmark
  dumps, or local activation captures.
- Do not treat Q4 layer/dtype scripts as production gates; they are mutation- and
  monkey-patch-heavy diagnostics.
- Do not use research script success to claim engine parity unless the matching
  Zig golden/bench gate also passes.

## COMMANDS

```sh
python research/mlx_reference.py --help
python research/pytorch_reference.py --help
python research/mlx_bonsai_q4_golden.py --help
```

Run actual scripts only after confirming dependencies and local assets exist.
