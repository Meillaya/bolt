#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/experiments"
ALLOWED_COMMANDS = {
    "./research/scripts/run_compare.sh",
    "./research/scripts/run_bench.sh",
    "./research/scripts/run_proof.sh",
    "./research/scripts/run_debug.sh",
    "./research/scripts/run_check.sh",
    "./research/scripts/run_smoke.sh",
}
ARTIFACT_LINE_RE = re.compile(r"^(compare|benchmark|proof|debug) artifacts: (artifacts/[^\s]+)$", re.MULTILINE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_recipe(path: Path) -> dict[str, Any]:
    try:
        recipe = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid recipe JSON in {path}: {exc}") from exc
    require(recipe.get("schema_version") == 1, "recipe schema_version must be 1")
    require(recipe.get("local_only") is True, "v1 experiment recipes must set local_only=true")
    require(recipe.get("external_services_allowed") is False, "v1 experiment recipes must set external_services_allowed=false")
    commands = recipe.get("commands")
    require(isinstance(commands, list) and commands, "recipe commands must be a non-empty list")
    for index, step in enumerate(commands):
        require(isinstance(step, dict), f"command step {index} must be an object")
        command = step.get("command")
        require(isinstance(command, list) and command, f"command step {index} command must be a non-empty array")
        require(all(isinstance(part, str) for part in command), f"command step {index} command parts must be strings")
        require(command[0] in ALLOWED_COMMANDS, f"command step {index} is not allowlisted: {command[0]}")
    return recipe


def stable_recipe_fingerprint(recipe: dict[str, Any]) -> str:
    stable = {
        "schema_version": recipe.get("schema_version"),
        "name": recipe.get("name"),
        "local_only": recipe.get("local_only"),
        "external_services_allowed": recipe.get("external_services_allowed"),
        "commands": recipe.get("commands"),
        "evaluation": recipe.get("evaluation", {}),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip().lower()).strip("-")
    return slug or "experiment"


def copy_latest(run_dir: Path, latest_dir: Path) -> None:
    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)


def run_step(step: dict[str, Any], index: int, run_dir: Path) -> dict[str, Any]:
    name = str(step.get("name") or f"step-{index + 1}")
    command = step["command"]
    stdout_path = run_dir / f"{index + 1:02d}-{slugify(name)}.stdout.txt"
    stderr_path = run_dir / f"{index + 1:02d}-{slugify(name)}.stderr.txt"
    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    start = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    ended = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)

    artifacts = []
    for match in ARTIFACT_LINE_RE.finditer(completed.stdout + "\n" + completed.stderr):
        artifacts.append({"kind": match.group(1), "path": match.group(2)})

    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "started_at": started,
        "completed_at": ended,
        "elapsed_ms": elapsed_ms,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
        "artifacts": artifacts,
    }


def relative_artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def evaluate_run(recipe: dict[str, Any], command_results: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    recommendations: list[str] = []
    artifact_paths: dict[str, list[str]] = {}
    for result in command_results:
        if result["exit_code"] != 0:
            failures.append(f"command failed: {result['name']} exit_code={result['exit_code']}")
        for artifact in result["artifacts"]:
            artifact_paths.setdefault(artifact["kind"], []).append(artifact["path"])

    latest_bench = ROOT / "artifacts/bench/latest"
    bench_manifest_path = latest_bench / "manifest.json"
    bench_summary_path = latest_bench / "summary.json"
    benchmark_summary: dict[str, Any] | None = None
    if bench_manifest_path.exists() and bench_summary_path.exists():
        benchmark_summary = json.loads(bench_summary_path.read_text())
        gates = benchmark_summary.get("correctness_gates", [])
        failing_gates = [gate.get("name") for gate in gates if gate.get("status") != "pass"]
        if failing_gates:
            failures.append(f"benchmark correctness gates failed: {failing_gates}")
        gate_names = {gate.get("name") for gate in gates}
        if "primitive_coverage" not in gate_names:
            failures.append("benchmark summary did not include primitive_coverage gate")
        if not ({"mnist_real_gpu_path", "llm_real_decoder_path"} & gate_names):
            failures.append("benchmark summary did not include a model-level benchmark gate")
    else:
        failures.append("benchmark manifest/summary missing under artifacts/bench/latest")

    required_artifact_kinds = recipe.get("evaluation", {}).get("required_artifact_kinds", [])
    for kind in required_artifact_kinds:
        if kind == "benchmark":
            if not artifact_paths.get("benchmark"):
                failures.append("required benchmark artifact path was not captured from command output")
        elif not artifact_paths.get(kind):
            failures.append(f"required {kind} artifact path was not captured from command output")

    if failures:
        outcome = "fail"
        recommendations.append("Inspect per-command stderr/stdout and rerun after fixing the failed gate.")
    else:
        outcome = "pass"
        recommendations.append("Control flow and local artifact gates passed; compare summary.json across runs for timing trends.")

    evidence_paths = {
        "benchmark_manifest": relative_artifact_path(bench_manifest_path),
        "benchmark_summary": relative_artifact_path(bench_summary_path),
    }
    if benchmark_summary is not None:
        artifacts = benchmark_summary.get("artifacts", {})
        for key in ("kernel_report", "mnist_inference", "llm_inference"):
            value = artifacts.get(key)
            if value:
                evidence_paths[key] = relative_artifact_path(latest_bench / value)

    return {
        "outcome": outcome,
        "failures": failures,
        "artifact_paths_from_commands": artifact_paths,
        "evidence_paths": evidence_paths,
        "benchmark_correctness_gates": [] if benchmark_summary is None else benchmark_summary.get("correctness_gates", []),
        "recommendations": recommendations,
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Experiment report: {report['recipe']['name']}",
        "",
        f"- outcome: `{report['evaluation']['outcome']}`",
        f"- local_only: `{str(report['recipe']['local_only']).lower()}`",
        f"- external_services_allowed: `{str(report['recipe']['external_services_allowed']).lower()}`",
        f"- control_flow_fingerprint: `{report['control_flow_fingerprint']}`",
        "",
        "## Commands",
    ]
    for command in report["commands_executed"]:
        lines.extend(
            [
                f"- `{command['name']}`: exit `{command['exit_code']}` in {command['elapsed_ms']} ms",
                f"  - command: `{' '.join(command['command'])}`",
                f"  - stdout: `{command['stdout']}`",
                f"  - stderr: `{command['stderr']}`",
            ]
        )
        for artifact in command["artifacts"]:
            lines.append(f"  - {artifact['kind']} artifact: `{artifact['path']}`")
    lines.extend(["", "## Evaluation evidence"])
    for key, value in report["evaluation"]["evidence_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Recommendations"])
    for recommendation in report["evaluation"]["recommendations"]:
        lines.append(f"- {recommendation}")
    if report["evaluation"]["failures"]:
        lines.extend(["", "## Failures"])
        for failure in report["evaluation"]["failures"]:
            lines.append(f"- {failure}")
    path.write_text("\n".join(lines) + "\n")


def run_recipe(recipe_path: Path, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    recipe = load_recipe(recipe_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifact_root / f"{timestamp}-{slugify(str(recipe.get('name', 'experiment')))}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(recipe_path, run_dir / "recipe.json")

    command_results: list[dict[str, Any]] = []
    for index, step in enumerate(recipe["commands"]):
        result = run_step(step, index, run_dir)
        command_results.append(result)
        if result["exit_code"] != 0 and step.get("continue_on_failure") is not True:
            break

    report = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": timestamp,
        "recipe": {
            "path": str(recipe_path),
            "name": recipe.get("name"),
            "description": recipe.get("description", ""),
            "local_only": recipe.get("local_only"),
            "external_services_allowed": recipe.get("external_services_allowed"),
        },
        "control_flow_fingerprint": stable_recipe_fingerprint(recipe),
        "commands_executed": command_results,
        "evaluation": evaluate_run(recipe, command_results),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown_report(report, run_dir / "report.md")
    copy_latest(run_dir, artifact_root / "latest")
    print(f"experiment artifacts: {relative_artifact_path(run_dir)}")
    print(f"experiment outcome: {report['evaluation']['outcome']}")
    if report["evaluation"]["outcome"] != "pass":
        raise SystemExit(1)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local-only bolt experiment recipe.")
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    recipe_path = args.recipe if args.recipe.is_absolute() else ROOT / args.recipe
    artifact_root = args.artifact_root if args.artifact_root.is_absolute() else ROOT / args.artifact_root
    run_recipe(recipe_path, artifact_root)


if __name__ == "__main__":
    main()
