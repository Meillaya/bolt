# nnzap real-asset checklist — Milestone 0

Use this checklist before any final-gate claim. Synthetic fixtures remain milestone-only.

## MNIST raw dataset
- [ ] Acquisition path recorded: curl/gunzip or torchvision.
- [ ] Local path recorded, default `data/mnist_torch/MNIST/raw`.
- [ ] `train-images-idx3-ubyte` SHA256 recorded.
- [ ] `train-labels-idx1-ubyte` SHA256 recorded.
- [ ] `t10k-images-idx3-ubyte` SHA256 recorded.
- [ ] `t10k-labels-idx1-ubyte` SHA256 recorded.
- [ ] License/access note recorded.

## Bonsai/unquantized transformer assets
- [ ] Local model directory recorded.
- [ ] Model config/metadata checksums recorded.
- [ ] Tokenizer files checksums recorded.
- [ ] Safetensors shard checksums recorded.
- [ ] Provenance and license/access note recorded.
- [ ] Golden prompt set recorded.

## Qwen/Bonsai Q4 assets
- [ ] Local Q4 model directory recorded.
- [ ] Q4 model metadata checksums recorded.
- [ ] Tokenizer files checksums recorded.
- [ ] Q4 safetensors/shard checksums recorded.
- [ ] Quantization format note recorded (`q4_mlx`/reference equivalent).
- [ ] Provenance and license/access note recorded.

## Completion rule
- [ ] `docs/nnzap-asset-manifest-template.json` has been copied to a local manifest or completed in an approved location.
- [ ] Every `TODO` needed by a final-gate row is resolved.
- [ ] Final-gate report cites the completed manifest and command artifacts.
- [ ] If the HTTP MNIST acquisition path is used, expected canonical SHA256 values are recorded before accepting downloaded files for final gates.
