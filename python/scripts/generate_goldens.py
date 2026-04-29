#!/usr/bin/env python3
"""Generate deterministic smoke fixtures for the current v1 proof paths."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixtureManifest:
    fixture_version: int
    family: str
    fixture_name: str
    payload_file: str
    expected_file: str
    runtime_bundle_file: str | None
    description: str


def build_manifest(family: str) -> FixtureManifest:
    if family == "mnist":
        return FixtureManifest(
            fixture_version=1,
            family="mnist",
            fixture_name="mnist-smoke",
            payload_file="mnist-smoke.json",
            expected_file="mnist-smoke.expected.json",
            runtime_bundle_file="mnist-smoke.runtime.json",
            description="Deterministic smoke fixture for the compact non-LLM path.",
        )
    if family == "llm":
        return FixtureManifest(
            fixture_version=1,
            family="llm",
            fixture_name="llm-smoke",
            payload_file="llm-smoke.json",
            expected_file="llm-smoke.expected.json",
            runtime_bundle_file="llm-smoke.runtime.json",
            description="Deterministic smoke fixture for the small decoder-only path.",
        )
    raise ValueError(f"unsupported family: {family}")


def write_fixture(output_dir: Path, family: str) -> Path:
    manifest = build_manifest(family)
    family_dir = output_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)

    payload_path = family_dir / manifest.payload_file
    expected_path = family_dir / manifest.expected_file
    runtime_bundle_path = family_dir / manifest.runtime_bundle_file if manifest.runtime_bundle_file else None

    if family == "mnist":
        pixels = [0.0] * 784
        pixels[0] = 1.0
        pixels[111] = 2.0
        pixels[783] = 4.0
        output_size = 10
        dense_weights = [0.0] * (784 * output_size)
        bias = [i / 10.0 for i in range(output_size)]
        dense_weights[(0 * output_size) + 7] = 0.1
        dense_weights[(111 * output_size) + 7] = 0.2
        dense_weights[(783 * output_size) + 7] = 0.3
        matmul = [
            sum(pixels[row] * dense_weights[(row * output_size) + col] for row in range(784))
            for col in range(output_size)
        ]
        logits = [value + bias[index] for index, value in enumerate(matmul)]
        max_logit = max(logits)
        exp_values = [math.exp(value - max_logit) for value in logits]
        exp_sum = sum(exp_values)
        probabilities = [value / exp_sum for value in exp_values]
        logits_milli = [round(value * 1000) for value in logits]
        probabilities_milli = [round(value * 1000) for value in probabilities]
        predicted_label = max(range(output_size), key=lambda index: logits[index])
        payload = {
            "family": family,
            "fixture_name": manifest.fixture_name,
            "image_shape": {"rows": 28, "cols": 28},
            "pixels": pixels,
            "expected_sum": 7,
            "expected_non_zero_count": 3,
            "expected_label": predicted_label,
        }
        expected = {
            "family": family,
            "fixture_name": manifest.fixture_name,
            "backend": "metal",
            "dispatched_kernels": {
                "matmul_f32": True,
                "bias_add_f32": True,
                "softmax_f32": True,
            },
            "rows": 28,
            "cols": 28,
            "element_count": 784,
            "pixel_sum": 7,
            "non_zero_count": 3,
            "predicted_label": predicted_label,
            "logits_milli": logits_milli,
            "probabilities_milli": probabilities_milli,
            "top_logit_milli": logits_milli[predicted_label],
            "top_probability_milli": probabilities_milli[predicted_label],
        }
        runtime_bundle = {
            "weights_file": "mnist-smoke.weights.bin",
        }
        mnist_weights = dense_weights + bias
    else:
        payload = {
            "family": family,
            "fixture_name": manifest.fixture_name,
            "prompt_text": "zig metal bolt",
            "token_ids": [1, 2, 3],
            "logits": [0.1, 0.2, 0.9, 0.4],
            "expected_prompt_sum": 6,
            "expected_top_token": 2,
        }
        expected = {
            "family": family,
            "fixture_name": manifest.fixture_name,
            "backend": "metal",
            "loader": "bolt-runtime-bundle-v1",
            "model_name": "llm-smoke-decoder",
            "model_architecture": "tiny-transition-decoder",
            "model_vocab_size": 4,
            "model_context_length": 8,
            "weights_format": "binary-f32-le",
            "dispatched_kernels": {
                "bias_add_f32": True,
                "softmax_f32": True,
            },
            "prompt_token_count": 3,
            "prompt_sum": 6,
            "logits_count": 4,
            "next_token_id": 2,
            "generated_token_count": 4,
            "generated_last_token": 1,
            "prompt_tail_token_id": 3,
            "prompt_tail_token_text": "bolt",
            "next_token_text": "metal",
            "raw_next_score_milli": 900,
            "conditioned_next_token_id": 1,
            "conditioned_next_token_text": "zig",
            "model_next_token_id": 1,
            "model_next_token_text": "zig",
            "prompt_context_next_token_id": 1,
            "prompt_context_next_token_text": "zig",
            "preferred_route": "prompt_context",
            "preferred_next_token_id": 1,
            "preferred_next_token_text": "zig",
            "preferred_next_score_milli": 2400,
            "route_comparison": {
                "raw": {
                    "route": "raw",
                    "token_id": 2,
                    "token_text": "metal",
                    "score_milli": 900,
                },
                "conditioned": {
                    "route": "conditioned",
                    "token_id": 1,
                    "token_text": "zig",
                    "score_milli": 1000,
                },
                "model": {
                    "route": "model",
                    "token_id": 1,
                    "token_text": "zig",
                    "score_milli": 2300,
                },
                "prompt_context": {
                    "route": "prompt_context",
                    "token_id": 1,
                    "token_text": "zig",
                    "score_milli": 2400,
                    "projection_milli": 1500,
                    "context_bias_milli": 900,
                },
                "preferred": {
                    "route": "prompt_context",
                    "token_id": 1,
                    "token_text": "zig",
                    "score_milli": 2400,
                    "projection_milli": 1500,
                    "context_bias_milli": 900,
                },
                "gains": {
                    "raw_to_conditioned_milli": 100,
                    "conditioned_to_model_milli": 1300,
                    "model_to_prompt_context_milli": 100,
                    "prompt_context_over_raw_milli": 1500,
                    "prompt_context_over_conditioned_milli": 1400,
                    "prompt_context_over_model_milli": 100,
                },
            },
            "generated_projection_milli": 1500,
            "generated_context_bias_milli": 900,
            "prompt_context_over_raw_gain_milli": 1500,
            "prompt_context_over_conditioned_gain_milli": 1400,
            "prompt_context_over_model_gain_milli": 100,
            "raw_conditioning_flipped": True,
            "conditioned_matches_model": True,
            "top_token_weight_milli": 2000,
            "weighted_logit_sum_milli": 2400,
            "prompt_condition_bias_milli": 800,
            "prompt_context_bias_milli": 900,
            "conditioned_next_score_milli": 1000,
            "conditioned_next_probability_milli": 343,
            "model_next_score_milli": 2300,
            "prompt_context_next_score_milli": 2400,
            "model_condition_gap_milli": 1300,
            "logit_margin_milli": 500,
        }
        tokenizer_config = {
            "vocab": ["<pad>", "zig", "metal", "bolt"],
        }
        weights_projection = [1.0, 1.5, 2.0, 0.5]
        transition_bias = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.1, 0.2, 0.0,
            0.0, 0.0, 0.1, 0.2,
            0.0, 0.8, 0.0, 0.1,
        ]
        runtime_bundle = {
            "model_file": "llm-smoke.model.json",
            "tokenizer_file": "llm-smoke.tokenizer.json",
            "weights_file": "llm-smoke.weights.bin",
        }
        model_config = {
            "loader": "bolt-runtime-bundle-v1",
            "model_name": "llm-smoke-decoder",
            "architecture": "tiny-transition-decoder",
            "vocab_size": 4,
            "hidden_size": 4,
            "context_length": 8,
        }

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    expected_path.write_text(json.dumps(expected, indent=2) + "\n")
    if family == "mnist" and runtime_bundle_path:
        import struct

        weights_path = family_dir / runtime_bundle["weights_file"]
        runtime_bundle_path.write_text(json.dumps(runtime_bundle, indent=2) + "\n")
        weights_path.write_bytes(struct.pack("<" + "f" * len(mnist_weights), *mnist_weights))
    elif runtime_bundle_path:
        import struct

        tokenizer_path = family_dir / runtime_bundle["tokenizer_file"]
        model_path = family_dir / runtime_bundle["model_file"]
        weights_path = family_dir / runtime_bundle["weights_file"]
        runtime_bundle_path.write_text(json.dumps(runtime_bundle, indent=2) + "\n")
        model_path.write_text(json.dumps(model_config, indent=2) + "\n")
        tokenizer_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")
        packed_weights = weights_projection + transition_bias
        weights_path.write_bytes(struct.pack("<" + "f" * len(packed_weights), *packed_weights))

    manifest_path = family_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parents[1] / "fixtures",
        type=Path,
    )
    parser.add_argument(
        "--family",
        choices=("mnist", "llm", "all"),
        default="all",
    )
    args = parser.parse_args()

    families = ("mnist", "llm") if args.family == "all" else (args.family,)
    results = []
    for family in families:
        results.append(str(write_fixture(args.output_dir, family)))

    print(json.dumps({"generated": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
