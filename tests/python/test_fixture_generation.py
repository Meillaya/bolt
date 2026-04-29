#!/usr/bin/env python3

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_generation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "python/scripts/generate_goldens.py"),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )

        mnist_manifest = json.loads((output_dir / "mnist/manifest.json").read_text())
        llm_manifest = json.loads((output_dir / "llm/manifest.json").read_text())
        mnist_expected = json.loads((output_dir / "mnist/mnist-smoke.expected.json").read_text())
        llm_payload = json.loads((output_dir / "llm/llm-smoke.json").read_text())
        llm_expected = json.loads((output_dir / "llm/llm-smoke.expected.json").read_text())

        assert mnist_manifest["family"] == "mnist"
        assert llm_manifest["family"] == "llm"
        assert mnist_manifest["fixture_version"] == 1
        assert llm_manifest["fixture_version"] == 1
        assert mnist_manifest["expected_file"] == "mnist-smoke.expected.json"
        assert mnist_manifest["runtime_bundle_file"] == "mnist-smoke.runtime.json"
        assert mnist_expected["element_count"] == 784
        assert mnist_expected["pixel_sum"] == 7
        assert mnist_expected["predicted_label"] == 7
        assert mnist_expected["backend"] == "metal"
        assert mnist_expected["dispatched_kernels"] == {
            "matmul_f32": True,
            "bias_add_f32": True,
            "softmax_f32": True,
        }
        assert mnist_expected["logits_milli"] == [0, 100, 200, 300, 400, 500, 600, 2400, 800, 900]
        assert mnist_expected["top_logit_milli"] == 2400
        assert mnist_expected["top_probability_milli"] == 435
        mnist_runtime_bundle = json.loads((output_dir / "mnist/mnist-smoke.runtime.json").read_text())
        mnist_weights_path = output_dir / "mnist" / mnist_runtime_bundle["weights_file"]
        assert mnist_runtime_bundle["weights_file"] == "mnist-smoke.weights.bin"
        assert mnist_weights_path.exists()
        assert mnist_weights_path.stat().st_size == (784 * 10 + 10) * 4
        assert llm_manifest["expected_file"] == "llm-smoke.expected.json"
        assert llm_manifest["runtime_bundle_file"] == "llm-smoke.runtime.json"
        runtime_bundle = json.loads((output_dir / "llm/llm-smoke.runtime.json").read_text())
        model_path = output_dir / "llm" / runtime_bundle["model_file"]
        tokenizer_path = output_dir / "llm" / runtime_bundle["tokenizer_file"]
        weights_path = output_dir / "llm" / runtime_bundle["weights_file"]
        model_config = json.loads(model_path.read_text())
        tokenizer_config = json.loads(tokenizer_path.read_text())
        assert runtime_bundle["model_file"] == "llm-smoke.model.json"
        assert runtime_bundle["tokenizer_file"] == "llm-smoke.tokenizer.json"
        assert runtime_bundle["weights_file"] == "llm-smoke.weights.bin"
        assert model_config["loader"] == "bolt-runtime-bundle-v1"
        assert model_config["model_name"] == "llm-smoke-decoder"
        assert model_config["architecture"] == "tiny-transition-decoder"
        assert model_config["context_length"] == 8
        assert model_path.exists()
        assert tokenizer_path.exists()
        assert weights_path.exists()
        assert tokenizer_config["vocab"] == ["<pad>", "zig", "metal", "bolt"]
        assert weights_path.stat().st_size == 80
        assert llm_payload["prompt_text"] == "zig metal bolt"
        assert llm_payload["token_ids"] == [1, 2, 3]
        assert llm_expected["prompt_sum"] == 6
        assert llm_expected["backend"] == "metal"
        assert llm_expected["loader"] == "bolt-runtime-bundle-v1"
        assert llm_expected["model_name"] == "llm-smoke-decoder"
        assert llm_expected["dispatched_kernels"] == {
            "bias_add_f32": True,
            "softmax_f32": True,
        }
        assert llm_expected["next_token_id"] == 2
        assert llm_expected["generated_token_count"] == 4
        assert llm_expected["generated_last_token"] == 1
        assert llm_expected["prompt_tail_token_id"] == 3
        assert llm_expected["prompt_tail_token_text"] == "bolt"
        assert llm_expected["next_token_text"] == "metal"
        assert llm_expected["raw_next_score_milli"] == 900
        assert llm_expected["conditioned_next_token_id"] == 1
        assert llm_expected["conditioned_next_token_text"] == "zig"
        assert llm_expected["model_next_token_id"] == 1
        assert llm_expected["model_next_token_text"] == "zig"
        assert llm_expected["prompt_context_next_token_id"] == 1
        assert llm_expected["prompt_context_next_token_text"] == "zig"
        assert llm_expected["preferred_route"] == "prompt_context"
        assert llm_expected["preferred_next_token_id"] == 1
        assert llm_expected["preferred_next_token_text"] == "zig"
        assert llm_expected["preferred_next_score_milli"] == 2400
        assert llm_expected["route_comparison"]["preferred"]["route"] == "prompt_context"
        assert llm_expected["route_comparison"]["prompt_context"]["projection_milli"] == 1500
        assert llm_expected["route_comparison"]["gains"]["raw_to_conditioned_milli"] == 100
        assert llm_expected["route_comparison"]["gains"]["conditioned_to_model_milli"] == 1300
        assert llm_expected["route_comparison"]["gains"]["prompt_context_over_model_milli"] == 100
        assert llm_expected["generated_projection_milli"] == 1500
        assert llm_expected["generated_context_bias_milli"] == 900
        assert llm_expected["prompt_context_over_raw_gain_milli"] == 1500
        assert llm_expected["prompt_context_over_conditioned_gain_milli"] == 1400
        assert llm_expected["prompt_context_over_model_gain_milli"] == 100
        assert llm_expected["raw_conditioning_flipped"] is True
        assert llm_expected["conditioned_matches_model"] is True
        assert llm_expected["top_token_weight_milli"] == 2000
        assert llm_expected["weighted_logit_sum_milli"] == 2400
        assert llm_expected["prompt_condition_bias_milli"] == 800
        assert llm_expected["prompt_context_bias_milli"] == 900
        assert llm_expected["conditioned_next_score_milli"] == 1000
        assert llm_expected["conditioned_next_probability_milli"] == 343
        assert llm_expected["model_next_score_milli"] == 2300
        assert llm_expected["prompt_context_next_score_milli"] == 2400
        assert llm_expected["model_condition_gap_milli"] == 1300
        assert llm_expected["logit_margin_milli"] == 500


def test_benchmark_artifact_validator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        latest_dir = Path(tmp) / "latest"
        run_dir.mkdir()
        (run_dir / "kernel-bench.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": "metal",
                    "status": "ok",
                    "iterations": 2,
                    "primitive_results": [
                        {"name": "add_f32", "elements": 4, "elapsed_ns": 10, "checksum": 0.0},
                        {"name": "relu_f32", "elements": 4, "elapsed_ns": 11, "checksum": 0.0},
                        {"name": "matmul_f32", "rows": 2, "cols": 2, "inner": 2, "elapsed_ns": 12, "checksum": 1.0},
                        {"name": "bias_add_f32", "rows": 2, "cols": 2, "elapsed_ns": 13, "checksum": 1.0},
                        {"name": "reduce_sum_f32", "elements": 4, "elapsed_ns": 14, "checksum": 2.0},
                        {"name": "softmax_f32", "elements": 4, "elapsed_ns": 15, "checksum": 1.0},
                    ],
                }
            )
        )
        (run_dir / "mnist-inference.json").write_text(
            json.dumps(
                {
                    "backend": "metal",
                    "image_shape": [1, 28, 28],
                    "predicted_label": 7,
                    "logits_milli": [0] * 10,
                    "top_logit_milli": 2400,
                    "dispatched_kernels": {
                        "matmul_f32": True,
                        "bias_add_f32": True,
                        "softmax_f32": True,
                    },
                }
            )
        )
        (run_dir / "llm-inference.json").write_text(
            json.dumps(
                {
                    "backend": "metal",
                    "loader": "bolt-runtime-bundle-v1",
                    "model_name": "llm-smoke-decoder",
                    "model_vocab_size": 4,
                    "model_context_length": 8,
                    "weights_format": "binary-f32-le",
                    "preferred_next_token_text": "zig",
                    "conditioned_next_probability_milli": 343,
                    "dispatched_kernels": {
                        "bias_add_f32": True,
                        "softmax_f32": True,
                    },
                }
            )
        )

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "python/scripts/validate_bench_artifacts.py"),
                str(run_dir),
                "--latest-dir",
                str(latest_dir),
                "--timestamp",
                "20260429T000000Z",
            ],
            check=True,
        )

        summary = json.loads((run_dir / "summary.json").read_text())
        latest_summary = json.loads((latest_dir / "summary.json").read_text())
        assert summary == latest_summary
        assert summary["schema_version"] == 1
        assert summary["benchmark_kind"] == "kernel-and-inference"
        assert [gate["status"] for gate in summary["correctness_gates"]] == ["pass"] * 4
        assert summary["performance_observations"]["kernel_iterations"] == 2
        assert summary["performance_observations"]["model_smoke_shapes"]["mnist"]["input_shape"] == [1, 28, 28]
        assert "Kernel elapsed_ns values" in summary["performance_observations"]["note"]


def test_experiment_recipe_contract() -> None:
    runner = load_script_module("run_experiment_recipe", "python/scripts/run_experiment_recipe.py")
    recipe_path = ROOT / "research/recipes/smoke-local.json"
    recipe = runner.load_recipe(recipe_path)

    assert recipe["schema_version"] == 1
    assert recipe["local_only"] is True
    assert recipe["external_services_allowed"] is False
    assert [step["command"][0] for step in recipe["commands"]] == [
        "./research/scripts/run_compare.sh",
        "./research/scripts/run_bench.sh",
        "./research/scripts/run_proof.sh",
    ]

    first_fingerprint = runner.stable_recipe_fingerprint(recipe)
    second_fingerprint = runner.stable_recipe_fingerprint(dict(recipe))
    assert first_fingerprint == second_fingerprint

    with tempfile.TemporaryDirectory() as tmp:
        invalid_recipe = Path(tmp) / "networked.json"
        invalid_recipe.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "networked",
                    "local_only": False,
                    "external_services_allowed": True,
                    "commands": [{"name": "bad", "command": ["curl", "https://example.com"]}],
                }
            )
        )
        try:
            runner.load_recipe(invalid_recipe)
        except SystemExit as exc:
            assert "local_only=true" in str(exc)
        else:
            raise AssertionError("networked experiment recipe unexpectedly passed validation")


if __name__ == "__main__":
    test_fixture_generation()
    test_benchmark_artifact_validator()
    test_experiment_recipe_contract()
    print("python-fixture-test: PASS")
