#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_KERNELS = {"add_f32", "relu_f32", "matmul_f32", "bias_add_f32", "reduce_sum_f32", "softmax_f32"}
MNIST_KERNELS = ("matmul_f32", "bias_add_f32", "softmax_f32")
LLM_KERNELS = ("bias_add_f32", "softmax_f32")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def enabled_kernel_names(kernels: dict[str, Any]) -> list[str]:
    return [name for name, enabled in kernels.items() if enabled]


def build_summary(run_dir: Path, timestamp: str) -> tuple[dict[str, Any], str]:
    kernel_path = run_dir / "kernel-bench.json"
    mnist_path = run_dir / "mnist-inference.json"
    llm_path = run_dir / "llm-inference.json"

    kernel = load_json(kernel_path)
    mnist = load_json(mnist_path)
    llm = load_json(llm_path)

    require(kernel.get("schema_version") == 1, "kernel benchmark schema_version must be 1")
    require(kernel.get("status") == "ok", f"benchmark did not complete successfully: {kernel.get('reason', 'unknown reason')}")
    primitive_results = kernel.get("primitive_results", [])
    require(isinstance(primitive_results, list) and primitive_results, "benchmark report did not include primitive_results")

    seen = {entry.get("name") for entry in primitive_results}
    missing = sorted(REQUIRED_KERNELS - seen)
    require(not missing, f"benchmark report missing primitives: {', '.join(missing)}")

    require(mnist.get("backend") == "metal", f"mnist inference benchmark did not use Metal: {mnist.get('backend')}")
    mnist_kernels = mnist.get("dispatched_kernels", {})
    require(isinstance(mnist_kernels, dict), "mnist dispatched_kernels must be an object")
    for name in MNIST_KERNELS:
        require(bool(mnist_kernels.get(name)), f"mnist inference did not dispatch {name}")

    require(llm.get("backend") == "metal", f"llm inference benchmark did not use Metal: {llm.get('backend')}")
    llm_kernels = llm.get("dispatched_kernels", {})
    require(isinstance(llm_kernels, dict), "llm dispatched_kernels must be an object")
    for name in LLM_KERNELS:
        require(bool(llm_kernels.get(name)), f"llm inference did not dispatch {name}")

    correctness_gates = [
        {
            "name": "primitive_benchmark_status",
            "status": "pass",
            "artifact": kernel_path.name,
            "backend": kernel.get("backend"),
        },
        {
            "name": "primitive_coverage",
            "status": "pass",
            "required_primitives": sorted(REQUIRED_KERNELS),
            "observed_primitives": sorted(name for name in seen if isinstance(name, str)),
        },
        {
            "name": "mnist_real_gpu_path",
            "status": "pass",
            "artifact": mnist_path.name,
            "backend": mnist.get("backend"),
            "dispatched_kernels": enabled_kernel_names(mnist_kernels),
            "predicted_label": mnist.get("predicted_label"),
        },
        {
            "name": "llm_real_decoder_path",
            "status": "pass",
            "artifact": llm_path.name,
            "backend": llm.get("backend"),
            "loader": llm.get("loader"),
            "model_name": llm.get("model_name"),
            "weights_format": llm.get("weights_format"),
            "dispatched_kernels": enabled_kernel_names(llm_kernels),
            "preferred_next_token_text": llm.get("preferred_next_token_text"),
        },
    ]

    primitive_observations: list[dict[str, Any]] = []
    for entry in primitive_results:
        observation = {
            "name": entry.get("name"),
            "elapsed_ns": entry.get("elapsed_ns"),
            "checksum": entry.get("checksum"),
        }
        for key in ("elements", "rows", "cols", "inner"):
            if key in entry:
                observation[key] = entry[key]
        primitive_observations.append(observation)

    performance_observations = {
        "note": "Kernel elapsed_ns values are timing observations; smoke fixture outputs are correctness evidence and should not be treated as stable performance thresholds.",
        "kernel_iterations": kernel.get("iterations"),
        "primitive_results": primitive_observations,
        "model_smoke_shapes": {
            "mnist": {
                "input_shape": mnist.get("image_shape") or [mnist.get("rows"), mnist.get("cols")],
                "class_count": len(mnist.get("logits_milli", [])),
                "top_logit_milli": mnist.get("top_logit_milli"),
            },
            "llm": {
                "model_vocab_size": llm.get("model_vocab_size"),
                "model_context_length": llm.get("model_context_length"),
                "conditioned_next_probability_milli": llm.get("conditioned_next_probability_milli"),
            },
        },
    }

    summary_json = {
        "schema_version": 1,
        "benchmark_kind": "kernel-and-inference",
        "timestamp_utc": timestamp,
        "correctness_gates": correctness_gates,
        "performance_observations": performance_observations,
        "artifacts": {
            "kernel_report": kernel_path.name,
            "mnist_inference": mnist_path.name,
            "llm_inference": llm_path.name,
        },
    }

    lines = [
        "benchmark correctness gates",
        f"- primitive benchmark status: pass ({kernel.get('backend')})",
        f"- primitive coverage: pass ({', '.join(sorted(REQUIRED_KERNELS))})",
        f"- mnist real GPU path: pass ({', '.join(enabled_kernel_names(mnist_kernels))})",
        f"- llm real decoder path: pass ({', '.join(enabled_kernel_names(llm_kernels))})",
        "benchmark performance observations",
        f"- backend: {kernel['backend']}",
        f"- iterations: {kernel['iterations']}",
    ]
    for entry in primitive_results:
        lines.append(f"- {entry['name']}: {entry['elapsed_ns']} ns total")
    lines.extend(
        [
            "model smoke evidence",
            f"- mnist predicted_label: {mnist['predicted_label']}",
            f"- mnist top_logit_milli: {mnist['top_logit_milli']}",
            f"- llm loader: {llm['loader']}",
            f"- llm model_name: {llm['model_name']}",
            f"- llm weights_format: {llm['weights_format']}",
            f"- llm preferred_next_token_text: {llm['preferred_next_token_text']}",
            f"- llm conditioned_next_probability_milli: {llm['conditioned_next_probability_milli']}",
        ]
    )
    return summary_json, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate bolt benchmark artifacts and emit summaries.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--latest-dir", type=Path)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()

    summary_json, summary_text = build_summary(args.run_dir, args.timestamp)
    (args.run_dir / "summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")
    (args.run_dir / "summary.txt").write_text(summary_text)
    if args.latest_dir is not None:
        args.latest_dir.mkdir(parents=True, exist_ok=True)
        (args.latest_dir / "summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")
        (args.latest_dir / "summary.txt").write_text(summary_text)
    print(summary_text, end="")


if __name__ == "__main__":
    main()
