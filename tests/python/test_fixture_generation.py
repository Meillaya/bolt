#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
        assert mnist_manifest["runtime_bundle_file"] is None
        assert mnist_expected["element_count"] == 784
        assert mnist_expected["pixel_sum"] == 7
        assert mnist_expected["predicted_label"] == 7
        assert llm_manifest["expected_file"] == "llm-smoke.expected.json"
        assert llm_manifest["runtime_bundle_file"] == "llm-smoke.runtime.json"
        runtime_bundle = json.loads((output_dir / "llm/llm-smoke.runtime.json").read_text())
        tokenizer_path = output_dir / "llm" / runtime_bundle["tokenizer_file"]
        weights_path = output_dir / "llm" / runtime_bundle["weights_file"]
        tokenizer_config = json.loads(tokenizer_path.read_text())
        assert runtime_bundle["tokenizer_file"] == "llm-smoke.tokenizer.json"
        assert runtime_bundle["weights_file"] == "llm-smoke.weights.bin"
        assert tokenizer_path.exists()
        assert weights_path.exists()
        assert tokenizer_config["vocab"] == ["<pad>", "zig", "metal", "bolt"]
        assert weights_path.stat().st_size == 80
        assert llm_payload["prompt_text"] == "zig metal bolt"
        assert llm_payload["token_ids"] == [1, 2, 3]
        assert llm_expected["prompt_sum"] == 6
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
        assert llm_expected["model_next_score_milli"] == 2300
        assert llm_expected["prompt_context_next_score_milli"] == 2400
        assert llm_expected["model_condition_gap_milli"] == 1300
        assert llm_expected["logit_margin_milli"] == 500


if __name__ == "__main__":
    test_fixture_generation()
    print("python-fixture-test: PASS")
