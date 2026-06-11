# Research scope and drift controls

`research/*.py` scripts are optional diagnostics and parity references. They are
not default product release gates, not default health checks, and not required for
portable release qualification. The default product release path stays Zig-first:
`engine` and `labrat` build steps own supported product verification, while
Python research helpers remain opt-in local comparisons.

## Why research scripts stay out of default CI

- They can require local Python packages (`mlx`, `mlx-lm`, `numpy`, `torch`, or
  `torchvision`) that are intentionally not repo-level dependencies.
- Several paths require Apple Silicon, MPS/Metal availability, downloaded MNIST
  data, local model weights, or ignored `data/` and `artifacts/` files.
- Diagnostic Q4 scripts can be mutation-heavy probes for logits, layer dumps, or
  dtype behavior; they are useful for investigation but are not release
  contracts.
- Product parity claims must be made by the matching Zig gate and artifact
  validator, not by a standalone Python helper succeeding on one workstation.

## Drift-control map

Keep this table synchronized with the current `research/*.py` file list. The
release checker verifies every script is classified here and verifies default
workflows do not run these scripts.

| Script | Classification | Engine assumption or artifact | Drift-control note |
| --- | --- | --- | --- |
| `research/mlx_reference.py` | Optional MNIST parity reference | MNIST input layout and `artifacts/bolt-mnist-run*.json` comparison shape | Run only when MLX and local MNIST data are available; keep out of default CI because it may train/download and depends on local accelerator setup. |
| `research/mlx_inference.py` | Optional MNIST inference diagnostic | MNIST model/input conventions and timing expectations comparable to engine MNIST runs | Use for local latency or throughput comparison only; default Zig tests own release health. |
| `research/pytorch_reference.py` | Optional MNIST parity reference | Torch MNIST dataset layout under `data/mnist_torch/MNIST/raw` and engine MNIST artifact shape | Run only when Torch/Torchvision and data are installed; do not promote to release gate without explicit dependency policy. |
| `research/pytorch_inference.py` | Optional MNIST inference diagnostic | PyTorch CPU/MPS execution comparable to engine MNIST inference artifacts | Use to investigate backend differences; local hardware and package state make it unsuitable for default CI. |
| `research/mlx_bonsai.py` | Optional Bonsai full-precision benchmark reference | Full-precision Bonsai assets such as `~/models/bonsai-1.7b` or ignored `data/bonsai-1.7b`, plus `artifacts/bolt-bonsai-bench.json` | Run after confirming model assets exist; engine Bonsai gates remain the product authority. |
| `research/mlx_bonsai_q4_golden.py` | Optional Q4 golden-token capture reference | Q4 model paths (`data/qwen3-1.7b-q4-gs64`, `data/qwen3-1.7b-q4`, or matching `~/models/`) and `artifacts/bolt-q4-golden.json` | Use to refresh or inspect golden-token expectations; release claims require the Zig Q4 golden gate. |
| `research/mlx_q4_compare.py` | Optional Q4 prompt/logit/token diagnostic | Q4 tokenizer/model assumptions and `artifacts/bolt-q4-bench.json` or golden output comparisons | Investigative only; package and model drift make default CI unreliable. |
| `research/mlx_q4_f16_test.py` | Optional Q4 dtype diagnostic | Q4 quantization dtype assumptions and local MLX conversion behavior | Keep out of default CI because it probes implementation details and may depend on MLX version behavior. |
| `research/mlx_q4_layer0.py` | Optional Q4 layer-0 activation diagnostic | Layer-0 activation naming, tensor layout, and local Q4 model files | Debug-only probe; do not use as product release evidence without matching Zig artifact validation. |
| `research/mlx_q4_layer_dump.py` | Optional Q4 multi-layer activation dump | Multi-layer activation layout, local Q4 model files, and ignored dump artifacts | Debug-only dump utility; generated captures are local artifacts and not default CI inputs. |

## Promotion rule

A research script can become a supported release lane only after a separate plan
adds explicit dependency management, hermetic asset policy, artifact freshness
checks, and CI scoping. Until then, these scripts remain optional diagnostics and
must not be wired into default workflows.
