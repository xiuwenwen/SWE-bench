#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_NAME = "SWE-bench/SWE-bench_Verified"
DATASET_FILE = SCRIPT_DIR / "swe_bench_verified_tasks.jsonl"
LEGACY_DATASET_FILES = (SCRIPT_DIR / "swe_bench_lite_tasks.json", SCRIPT_DIR / "swe_bench_verified_tasks.json")
PREDICTIONS_FILE = SCRIPT_DIR / "predictions.jsonl"
CASE_RECORD_FILE = SCRIPT_DIR / "run_state" / "case_results.jsonl"
BLOCKED_TASK_LIST_FILE = SCRIPT_DIR / "run_state" / "blocked_task_list"
DOCKER_CONFIG_DIR = SCRIPT_DIR / "run_state" / "docker_config_no_creds"
WORKSPACE_DIR = SCRIPT_DIR / "temp_workspace"
RUN_LOG_DIR = SCRIPT_DIR / "openorchestra_runs"
EVALUATION_LOG_DIR = SCRIPT_DIR / "logs" / "run_evaluation"
ORCHESTRA_BIN = Path("/Users/shixiuwen/workSpace/myHarnessSystem/orchestra")
DEFAULT_AGENT_NAME = "openorchestra"
DEFAULT_TEST_LIMIT = 2
DOCKER_DESKTOP_START_TIMEOUT_SECONDS = 180
DOCKER_DESKTOP_POLL_SECONDS = 3


@dataclass(frozen=True)
class TaskResult:
    instance_id: str
    patch: str
    success: bool
    project_dir: Path | None
    patch_file: Path | None
    error: str | None = None


@dataclass(frozen=True)
class DatasetSource:
    dataset_name: str
    dataset_file: Path
    sha256: str
    bytes: int
    task_count: int
    generated_at_utc: str


@dataclass(frozen=True)
class EvaluationResult:
    exit_code: int
    resolved: bool | None
    report_file: Path | None
    error: str | None = None


def enforce_jsonl_dataset_file(dataset_file: Path) -> None:
    if dataset_file.suffix != ".jsonl":
        raise SystemExit(f"[ERROR] SWE-bench dataset file must use .jsonl: {dataset_file}")


def remove_legacy_dataset_files(dataset_file: Path) -> None:
    current = dataset_file.resolve()
    for legacy_file in LEGACY_DATASET_FILES:
        legacy = legacy_file.expanduser().resolve()
        if legacy == current or not legacy.exists():
            continue
        legacy.unlink()
        print(f"[*] 已删除旧数据集文件: {legacy}")


def download_dataset(dataset_name: str, dataset_file: Path) -> None:
    enforce_jsonl_dataset_file(dataset_file)
    if dataset_file.exists():
        print(f"[*] 测试集已存在: {dataset_file}")
        return

    print(f"[*] 正在从 Hugging Face 下载数据集: {dataset_name}")
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_name, split="test")
    dataset.to_json(str(dataset_file))
    print(f"[*] 下载完成，共 {len(dataset)} 个任务: {dataset_file}")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_tasks(dataset_file: Path) -> list[dict[str, Any]]:
    with dataset_file.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def dataset_source(dataset_name: str, dataset_file: Path, tasks: list[dict[str, Any]]) -> DatasetSource:
    stat = dataset_file.stat()
    return DatasetSource(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        sha256=sha256_file(dataset_file),
        bytes=stat.st_size,
        task_count=len(tasks),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def dataset_source_payload(source: DatasetSource) -> dict[str, Any]:
    return {
        "dataset_name": source.dataset_name,
        "dataset_file": str(source.dataset_file),
        "sha256": source.sha256,
        "bytes": source.bytes,
        "task_count": source.task_count,
        "generated_at_utc": source.generated_at_utc,
    }


def build_swebench_context(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "openorchestra.swebench_context.v1",
        "source": "swe_bench_case_metadata",
        "instance": {
            "instance_id": task.get("instance_id"),
            "repo": task.get("repo"),
            "base_commit": task.get("base_commit"),
            "created_at": task.get("created_at"),
            "difficulty": task.get("difficulty"),
        },
        "environment_contract_seed": {
            "schema_version": "environment_contract.v1",
            "source": "benchmark_metadata",
            "confidence": "high",
            "runtime": {
                "type": "swebench",
                "repo": task.get("repo"),
                "version": task.get("version"),
                "base_commit": task.get("base_commit"),
                "environment_setup_commit": task.get("environment_setup_commit") or task.get("base_commit"),
            },
            # Harness should treat this as a benchmark adapter input, not a free-form pytest fallback.
            "setup": {
                "mode": "benchmark_spec",
                "spec_source": "SWE-bench TestSpec",
            },
            "unknowns": [],
        },
        "validation_contract_seed": {
            "schema_version": "validation_contract.v1",
            "source": "benchmark_metadata",
            "confidence": "high",
            "runtime": "swebench_official",
            "official_entrypoint": "python -m swebench.harness.run_evaluation",
            "final_check": {
                "authority": "official_swebench_evaluator",
                "timing": "after_harness_delivery",
                "required_result": {"resolved": True},
                "harness_instruction": "Treat the official SWE-bench evaluator as the final validation authority; local agent tests are pre-check evidence, not the final pass criterion.",
            },
            "tests": {
                "mode": "benchmark_spec",
                "spec_source": "SWE-bench TestSpec",
            },
            "pass_criteria": {
                "type": "swebench_resolved",
                "resolved": True,
            },
        },
        "agent_visibility_policy": {
            "reference_patch_visible": False,
            "test_patch_body_visible": False,
            "hints_text_visible": False,
            "hidden_test_identities_visible": False,
            "withheld_fields": ["patch", "test_patch", "hints_text", "FAIL_TO_PASS", "PASS_TO_PASS"],
            "reason": "Keep inference separate from official benchmark evaluation; agents receive benchmark runtime semantics, not gold patches, hidden tests, or issue-discussion hints.",
        },
    }


def write_dataset_source(run_log_dir: Path, source: DatasetSource) -> None:
    run_log_dir.mkdir(parents=True, exist_ok=True)
    (run_log_dir / "dataset_source.json").write_text(
        json.dumps(dataset_source_payload(source), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_instance_dir_name(instance_id: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in instance_id)


def task_workspace_dir(workspace_root: Path, instance_id: str) -> Path:
    return workspace_root / safe_instance_dir_name(instance_id)


def clone_task_repo(task: dict[str, Any], workspace_dir: Path) -> None:
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)
    workspace_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{task['repo']}.git"
    print(f"  -> 克隆 {repo_url}")
    subprocess.run(["git", "clone", repo_url, str(workspace_dir)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", task["base_commit"]], cwd=workspace_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "reset", "--hard", task["base_commit"]], cwd=workspace_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=workspace_dir, check=True, capture_output=True, text=True)


def reset_task_repo(task: dict[str, Any], workspace_dir: Path) -> None:
    subprocess.run(["git", "reset", "--hard", task["base_commit"]], cwd=workspace_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=workspace_dir, check=True, capture_output=True, text=True)


def build_agent_prompt(task: dict[str, Any], swebench_context: dict[str, Any] | None = None) -> str:
    context = swebench_context if swebench_context is not None else build_swebench_context(task)
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            f"SWE-bench instance: {task['instance_id']}",
            f"Repository: {task['repo']}",
            f"Base commit: {task['base_commit']}",
            "",
            "Fix the bug described below with the smallest safe code change.",
            "Do not edit tests just to make the benchmark pass.",
            "Do not use or infer from the reference patch, test_patch, hints_text, FAIL_TO_PASS, or PASS_TO_PASS.",
            "Use the structured SWE-bench metadata below when planning environment setup and validation.",
            "Harness should treat the official SWE-bench evaluator result (`resolved: true`) as the final validation authority after delivery.",
            "The reference patch, test_patch body, hints_text, and hidden test identities are intentionally withheld from agents.",
            "",
            "Structured SWE-bench metadata:",
            "```json",
            context_json,
            "```",
            "",
            "Problem statement:",
            task["problem_statement"],
            "",
        ]
    )


def run_openorchestra(
    task: dict[str, Any],
    *,
    workspace_dir: Path,
    run_dir: Path,
    orchestra_bin: Path,
    backend: str | None,
    serial_agents: bool,
    timeout_seconds: int | None,
    ui: bool,
) -> subprocess.CompletedProcess[str]:
    swebench_context = build_swebench_context(task)
    (run_dir / "swebench_context.json").write_text(
        json.dumps(swebench_context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prompt = build_agent_prompt(task, swebench_context)
    prompt_file = run_dir / "problem_statement.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    command = [
        str(orchestra_bin),
        "--workflow",
        "bugfix",
        "--source-repo",
        str(workspace_dir),
        "--prompt-file",
        str(prompt_file),
    ]
    command.append("--ui" if ui else "--no-ui")
    if backend:
        command.extend(["--backend", backend])
    if serial_agents:
        command.append("--serial-agents")

    (run_dir / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("  -> 运行 OpenOrchestra")
    result = run_process_group(command, cwd=SCRIPT_DIR, timeout_seconds=timeout_seconds, live_output=ui)
    (run_dir / "orchestra_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (run_dir / "orchestra_stderr.log").write_text(result.stderr or "", encoding="utf-8")
    return result


def run_process_group(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int | None,
    live_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if live_output:
        return communicate_process_group_live(process, command, timeout_seconds)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr = terminate_process_group(process)
        raise subprocess.TimeoutExpired(command, timeout_seconds, output=stdout, stderr=stderr)
    except KeyboardInterrupt:
        # OpenOrchestra runs in its own session, so Ctrl+C reaches this wrapper
        # but not necessarily the child process group. Kill it explicitly.
        terminate_process_group(process)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def communicate_process_group_live(
    process: subprocess.Popen[str],
    command: list[str],
    timeout_seconds: int | None,
) -> subprocess.CompletedProcess[str]:
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def read_stream(stream, target, chunks: list[str]) -> None:
        if stream is None:
            return
        for chunk in iter(stream.readline, ""):
            if not chunk:
                break
            chunks.append(chunk)
            target.write(chunk)
            target.flush()

    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, sys.stdout, stdout_chunks), daemon=True)
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, sys.stderr, stderr_chunks), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr = terminate_process_group(process)
        stdout_chunks.append(stdout)
        stderr_chunks.append(stderr)
        raise subprocess.TimeoutExpired(command, timeout_seconds, output="".join(stdout_chunks), stderr="".join(stderr_chunks))
    except KeyboardInterrupt:
        stdout, stderr = terminate_process_group(process)
        stdout_chunks.append(stdout)
        stderr_chunks.append(stderr)
        raise
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    return subprocess.CompletedProcess(command, returncode, "".join(stdout_chunks), "".join(stderr_chunks))


def terminate_process_group(process: subprocess.Popen[str], grace_seconds: float = 5.0) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def parse_project_dir(stdout: str) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("project_dir:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return Path(value).expanduser()
    return None


def final_patch_path(project_dir: Path) -> Path:
    delivery_dir = project_dir.parent if project_dir.name == "source" else project_dir
    return delivery_dir / "patches" / "final.patch"


def validate_patch_applies(patch_file: Path, workspace_dir: Path) -> tuple[bool, str | None]:
    patch = patch_file.read_text(encoding="utf-8", errors="replace")
    if not patch.strip():
        return False, "final.patch is empty"
    result = subprocess.run(
        ["git", "apply", "--check", str(patch_file)],
        cwd=workspace_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "git apply --check failed").strip()
    return True, None


def tail(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def completed_process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_agent_on_task(
    task: dict[str, Any],
    *,
    workspace_dir: Path,
    run_log_root: Path,
    orchestra_bin: Path,
    backend: str | None,
    serial_agents: bool,
    timeout_seconds: int | None,
    dataset: DatasetSource,
    ui: bool,
) -> TaskResult:
    instance_id = task["instance_id"]
    checkout_dir = task_workspace_dir(workspace_dir, instance_id)
    run_dir = run_log_root / instance_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> 开始处理任务: {instance_id}")
    try:
        clone_task_repo(task, checkout_dir)
    except subprocess.CalledProcessError as exc:
        error = f"environment setup failed: {tail(exc.stderr or exc.stdout or str(exc))}"
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "error": error,
                    "dataset": dataset_source_payload(dataset),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  [X] {error}")
        return TaskResult(instance_id, "", False, None, None, error)

    try:
        result = run_openorchestra(
            task,
            workspace_dir=checkout_dir.resolve(),
            run_dir=run_dir,
            orchestra_bin=orchestra_bin,
            backend=backend,
            serial_agents=serial_agents,
            timeout_seconds=timeout_seconds,
            ui=ui,
        )
    except subprocess.TimeoutExpired as exc:
        error = f"OpenOrchestra timed out after {timeout_seconds} seconds"
        (run_dir / "orchestra_stdout.log").write_text(completed_process_text(exc.stdout), encoding="utf-8")
        (run_dir / "orchestra_stderr.log").write_text(completed_process_text(exc.stderr), encoding="utf-8")
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "error": error,
                    "dataset": dataset_source_payload(dataset),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  [X] {error}")
        return TaskResult(instance_id, "", False, None, None, error)
    project_dir = parse_project_dir(result.stdout)
    if result.returncode != 0:
        error = f"OpenOrchestra failed with return code {result.returncode}: {tail(result.stderr or result.stdout)}"
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "project_dir": str(project_dir) if project_dir else None,
                    "error": error,
                    "dataset": dataset_source_payload(dataset),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  [X] {error}")
        return TaskResult(instance_id, "", False, project_dir, None, error)

    if not project_dir:
        error = "OpenOrchestra stdout did not contain project_dir"
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "error": error,
                    "dataset": dataset_source_payload(dataset),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  [X] {error}")
        return TaskResult(instance_id, "", False, None, None, error)

    patch_file = final_patch_path(project_dir)
    if not patch_file.exists():
        error = f"final.patch not found: {patch_file}"
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "project_dir": str(project_dir),
                    "patch_file": str(patch_file),
                    "error": error,
                    "dataset": dataset_source_payload(dataset),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  [X] {error}")
        return TaskResult(instance_id, "", False, project_dir, patch_file, error)

    try:
        reset_task_repo(task, checkout_dir)
    except subprocess.CalledProcessError as exc:
        error = f"failed to reset workspace before patch validation: {tail(exc.stderr or exc.stdout or str(exc))}"
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "project_dir": str(project_dir),
                    "patch_file": str(patch_file),
                    "error": error,
                    "dataset": dataset_source_payload(dataset),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  [X] {error}")
        return TaskResult(instance_id, "", False, project_dir, patch_file, error)

    applies, apply_error = validate_patch_applies(patch_file, checkout_dir)
    if not applies:
        error = f"final.patch does not apply to base checkout: {apply_error}"
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "project_dir": str(project_dir),
                    "patch_file": str(patch_file),
                    "error": error,
                    "dataset": dataset_source_payload(dataset),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  [X] {error}")
        return TaskResult(instance_id, "", False, project_dir, patch_file, error)

    patch = patch_file.read_text(encoding="utf-8", errors="replace")
    status = {
        "success": True,
        "project_dir": str(project_dir),
        "patch_file": str(patch_file),
        "patch_bytes": len(patch.encode("utf-8")),
        "dataset": dataset_source_payload(dataset),
    }
    (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  -> 成功提取并校验 patch: {patch_file} ({len(patch)} chars)")
    return TaskResult(instance_id, patch, True, project_dir, patch_file)


def expand_instance_selectors(tasks: list[dict[str, Any]], selectors: list[str], *, option_name: str) -> set[str]:
    instance_ids = [str(task["instance_id"]) for task in tasks]
    instance_id_set = set(instance_ids)
    resolved: set[str] = set()
    for raw_selector in selectors:
        for selector in str(raw_selector).split(","):
            selector = selector.strip()
            if not selector:
                continue
            if selector in instance_id_set:
                resolved.add(selector)
                continue
            matches = [instance_id for instance_id in instance_ids if instance_id.endswith(selector)]
            if not matches:
                raise SystemExit(f"[ERROR] {option_name} did not match any SWE-bench instance: {selector}")
            if len(matches) > 1:
                raise SystemExit(
                    f"[ERROR] {option_name} selector {selector!r} is ambiguous: " + ", ".join(sorted(matches))
                )
            resolved.add(matches[0])
    return resolved


def load_blocked_task_selectors(blocked_task_list: Path = BLOCKED_TASK_LIST_FILE) -> list[str]:
    if not blocked_task_list.exists():
        return []
    selectors: list[str] = []
    with blocked_task_list.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            selectors.extend(part.strip() for part in line.split(",") if part.strip())
    return selectors


def resolve_blocked_task_ids(tasks: list[dict[str, Any]], selectors: list[str]) -> tuple[set[str], list[str]]:
    instance_ids = [str(task["instance_id"]) for task in tasks]
    instance_id_set = set(instance_ids)
    resolved: set[str] = set()
    unmatched: list[str] = []
    for selector in selectors:
        if selector in instance_id_set:
            resolved.add(selector)
            continue
        matches = [instance_id for instance_id in instance_ids if instance_id.endswith(selector)]
        if not matches:
            unmatched.append(selector)
            continue
        if len(matches) > 1:
            raise SystemExit(
                f"[ERROR] {BLOCKED_TASK_LIST_FILE} selector {selector!r} is ambiguous: "
                + ", ".join(sorted(matches))
            )
        resolved.add(matches[0])
    return resolved, unmatched


def select_tasks(tasks: list[dict[str, Any]], *, instance_ids: set[str]) -> list[dict[str, Any]]:
    if instance_ids:
        tasks = [task for task in tasks if task["instance_id"] in instance_ids]
    return tasks


def apply_task_limit(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit > 0:
        return tasks[:limit]
    return tasks


def load_completed_prediction_ids(predictions_file: Path) -> set[str]:
    if not predictions_file.exists():
        return set()
    completed: set[str] = set()
    with predictions_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("instance_id") and record.get("model_patch"):
                completed.add(str(record["instance_id"]))
    return completed


def load_successful_case_ids(case_record_file: Path) -> set[str]:
    if not case_record_file.exists():
        return set()
    successful: set[str] = set()
    with case_record_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = record.get("instance_id")
            if instance_id and record.get("official_evaluation_passed") is True:
                successful.add(str(instance_id))
    return successful


def case_record_status(
    result: TaskResult,
    *,
    evaluation_enabled: bool,
    official_evaluation_passed: bool | None,
) -> str:
    if not result.success:
        return "prediction_failed"
    if official_evaluation_passed is True:
        return "official_passed"
    if official_evaluation_passed is False:
        return "official_failed"
    if evaluation_enabled:
        return "official_unknown"
    return "prediction_written"


def build_case_record(
    *,
    task: dict[str, Any],
    dataset: DatasetSource,
    args: argparse.Namespace,
    result: TaskResult,
    run_dir: Path,
    evaluation_exit_code: int | None,
    official_evaluation_passed: bool | None,
    evaluation_report_file: Path | None = None,
    evaluation_error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance_id": result.instance_id,
        "repo": task.get("repo"),
        "base_commit": task.get("base_commit"),
        "dataset": dataset_source_payload(dataset),
        "agent_name": args.agent_name,
        "backend": args.backend or "auto",
        "orchestra_success": result.success,
        "prediction_written": result.success,
        "patch_bytes": len(result.patch.encode("utf-8")) if result.success else 0,
        "evaluation_enabled": args.evaluate,
        "evaluation_exit_code": evaluation_exit_code,
        "evaluation_report_file": str(evaluation_report_file) if evaluation_report_file else None,
        "evaluation_error": evaluation_error,
        "official_evaluation_passed": official_evaluation_passed,
        "status": case_record_status(
            result,
            evaluation_enabled=args.evaluate,
            official_evaluation_passed=official_evaluation_passed,
        ),
        "project_dir": str(result.project_dir) if result.project_dir else None,
        "patch_file": str(result.patch_file) if result.patch_file else None,
        "run_dir": str(run_dir),
        "error": result.error,
    }


def append_case_record(case_record_file: Path, record: dict[str, Any]) -> None:
    case_record_file.parent.mkdir(parents=True, exist_ok=True)
    with case_record_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluation_namespace_arg(value: str) -> str | None:
    if value != "auto":
        return "none" if value.strip().lower() in {"", "none", "null"} else value
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "none"
    return None


def evaluation_dataset_name(args: argparse.Namespace) -> str:
    dataset_file = Path(args.dataset_file).expanduser().resolve()
    if dataset_file.exists():
        return str(dataset_file)
    return args.dataset_name


def build_evaluation_command(args: argparse.Namespace, predictions_file: Path) -> list[str]:
    return build_evaluation_command_for_instances(args, predictions_file, instance_ids=args.instance_id or [])


def build_evaluation_command_for_instances(
    args: argparse.Namespace,
    predictions_file: Path,
    *,
    instance_ids: list[str],
    max_workers: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        evaluation_dataset_name(args),
        "--predictions_path",
        str(predictions_file),
        "--max_workers",
        str(max_workers if max_workers is not None else args.eval_max_workers),
        "--run_id",
        args.eval_run_id,
        "--timeout",
        str(args.eval_timeout),
    ]
    namespace = evaluation_namespace_arg(args.eval_namespace)
    if namespace is not None:
        command.extend(["--namespace", namespace])
    if instance_ids:
        command.append("--instance_ids")
        command.extend(instance_ids)
    return command


def evaluation_report_path(args: argparse.Namespace, instance_id: str) -> Path:
    return (
        EVALUATION_LOG_DIR
        / args.eval_run_id
        / args.agent_name.replace("/", "__")
        / instance_id
        / "report.json"
    )


def remove_stale_evaluation_reports(args: argparse.Namespace, instance_ids: list[str]) -> None:
    for instance_id in instance_ids:
        report_path = evaluation_report_path(args, instance_id)
        if report_path.exists():
            report_path.unlink()


def read_evaluation_resolved(report_path: Path, instance_id: str) -> tuple[bool | None, str | None]:
    if not report_path.exists():
        return None, f"evaluation report not found: {report_path}"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"failed to read evaluation report: {exc}"
    try:
        return bool(report[instance_id]["resolved"]), None
    except (KeyError, TypeError) as exc:
        return None, f"evaluation report missing resolved field: {exc}"


def docker_info(docker: str) -> tuple[bool, str]:
    try:
        result = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "docker info timed out after 30s."
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "docker info failed").strip()
        return False, detail
    return True, ""


def start_docker_desktop() -> tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, "automatic Docker Desktop startup is only supported on macOS."
    opener = shutil.which("open")
    if opener is None:
        return False, "macOS open command not found."
    try:
        subprocess.run([opener, "-a", "Docker"], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "open -a Docker timed out after 30s."
    except OSError as exc:
        return False, str(exc)
    return True, ""


def check_docker_available() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if not docker:
        return False, "docker CLI not found on PATH. Install/start Docker Desktop before running official SWE-bench evaluation."

    ok, detail = docker_info(docker)
    if ok:
        return True, ""

    start_ok, start_detail = start_docker_desktop()
    if not start_ok:
        return False, f"Docker daemon is not available and auto-start failed: {start_detail} Detail: {detail}"

    print("[preflight] Docker daemon is not ready; starting Docker Desktop...")
    deadline = time.monotonic() + DOCKER_DESKTOP_START_TIMEOUT_SECONDS
    last_detail = detail
    while time.monotonic() < deadline:
        time.sleep(DOCKER_DESKTOP_POLL_SECONDS)
        ok, last_detail = docker_info(docker)
        if ok:
            return True, ""

    return False, (
        "Docker Desktop was started but the daemon did not become ready within "
        f"{DOCKER_DESKTOP_START_TIMEOUT_SECONDS}s. Detail: {last_detail}"
    )


def evaluation_env() -> dict[str, str]:
    env = os.environ.copy()
    DOCKER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    docker_config = DOCKER_CONFIG_DIR / "config.json"
    if not docker_config.exists():
        docker_config.write_text('{"auths":{}}\n', encoding="utf-8")
    env["DOCKER_CONFIG"] = str(DOCKER_CONFIG_DIR)
    return env


def backend_binary_name(backend: str | None) -> str | None:
    if backend in {None, "auto"}:
        return None
    return {"codex": "codex", "claude": "claude", "gemini": "gemini", "qwen": "qwen"}.get(backend)


def ensure_writable_dir(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".preflight_write_check.{os.getpid()}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(f"[ERROR] {label} is not writable: {path}: {exc}") from exc


def preflight_environment(
    args: argparse.Namespace,
    *,
    orchestra_bin: Path,
    dataset_file: Path,
    predictions_file: Path,
    case_record_file: Path,
    workspace_dir: Path,
    run_log_dir: Path,
) -> None:
    print("[*] preflight: checking local runtime environment")
    if shutil.which("git") is None:
        raise SystemExit("[ERROR] git CLI not found on PATH. Install git before running SWE-bench tasks.")
    print("[preflight] git: ok")

    if not orchestra_bin.exists():
        raise SystemExit(f"[ERROR] missing OpenOrchestra launcher: {orchestra_bin}")
    if not os.access(orchestra_bin, os.X_OK):
        raise SystemExit(f"[ERROR] OpenOrchestra launcher is not executable: {orchestra_bin}")
    print(f"[preflight] OpenOrchestra launcher: {orchestra_bin}")

    backend_binary = backend_binary_name(args.backend)
    if backend_binary:
        backend_path = shutil.which(backend_binary)
        if backend_path is None:
            raise SystemExit(f"[ERROR] backend CLI not found on PATH: {backend_binary}")
        print(f"[preflight] backend {args.backend}: {backend_path}")
    else:
        print("[preflight] backend: auto; OpenOrchestra will resolve it")

    ensure_writable_dir(dataset_file.parent, "dataset directory")
    ensure_writable_dir(predictions_file.parent, "predictions directory")
    ensure_writable_dir(case_record_file.parent, "case record directory")
    ensure_writable_dir(workspace_dir, "workspace directory")
    ensure_writable_dir(run_log_dir, "run log directory")
    print("[preflight] writable paths: ok")

    if args.evaluate:
        try:
            __import__("swebench.harness.run_evaluation")
        except Exception as exc:
            raise SystemExit(f"[ERROR] SWE-bench evaluator import failed: {exc}") from exc
        print("[preflight] SWE-bench evaluator module: ok")
        ok, message = check_docker_available()
        if not ok:
            raise SystemExit(f"[ERROR] {message}")
        namespace = evaluation_namespace_arg(args.eval_namespace)
        namespace_label = namespace if namespace is not None else "default"
        print(f"[preflight] Docker evaluation: ok namespace={namespace_label}")


def run_official_evaluation(
    args: argparse.Namespace,
    predictions_file: Path,
    output_dir: Path,
    *,
    instance_ids: list[str],
    max_workers: int | None = None,
) -> EvaluationResult:
    remove_stale_evaluation_reports(args, instance_ids)
    command = build_evaluation_command_for_instances(
        args,
        predictions_file,
        instance_ids=instance_ids,
        max_workers=max_workers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[*] 开始官方 SWE-bench Docker evaluation")
    result = subprocess.run(command, cwd=SCRIPT_DIR, capture_output=True, text=True, env=evaluation_env())
    (output_dir / "evaluation_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (output_dir / "evaluation_stderr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    report_file = evaluation_report_path(args, instance_ids[0]) if len(instance_ids) == 1 else None
    resolved, error = (None, None)
    if report_file is not None:
        resolved, error = read_evaluation_resolved(report_file, instance_ids[0])
    return EvaluationResult(result.returncode, resolved, report_file, error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenOrchestra on SWE-bench Verified and produce predictions.jsonl.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-file", type=Path, default=DATASET_FILE)
    parser.add_argument("--predictions-file", type=Path, default=PREDICTIONS_FILE)
    parser.add_argument(
        "--case-record-file",
        type=Path,
        default=CASE_RECORD_FILE,
        help="Persistent JSONL case result record. Kept outside run cache/log directories.",
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE_DIR)
    parser.add_argument("--run-log-dir", type=Path, default=RUN_LOG_DIR)
    parser.add_argument("--orchestra-bin", type=Path, default=ORCHESTRA_BIN)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--limit", type=int, default=DEFAULT_TEST_LIMIT, help="Number of tasks to run. Use 0 for all tasks.")
    parser.add_argument("--instance-id", action="append", default=[], help="Run only this instance_id. Can be repeated.")
    parser.add_argument("--append", action="store_true", help="Append to predictions instead of overwriting it.")
    parser.add_argument("--resume", action="store_true", help="Append and skip instance IDs that already have a non-empty patch.")
    parser.add_argument(
        "--skip-successful-cases",
        action="store_true",
        help="Skip instance IDs already recorded as official SWE-bench passed in --case-record-file.",
    )
    parser.add_argument("--backend", choices=["auto", "codex", "claude", "gemini", "qwen"], default=None)
    parser.add_argument("--serial-agents", action="store_true", help="Pass --serial-agents through to orchestra.")
    parser.add_argument("--orchestra-timeout", type=int, default=0, help="Per-instance OpenOrchestra timeout in seconds. Use 0 for no outer timeout.")
    parser.set_defaults(ui=True)
    parser.add_argument("--ui", dest="ui", action="store_true", help="Run OpenOrchestra with its local Web UI enabled and stream its output live. Enabled by default.")
    parser.add_argument("--no-ui", dest="ui", action="store_false", help="Disable the local Web UI and capture OpenOrchestra output to logs only.")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep the final temporary checkout after completion.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed task.")
    parser.set_defaults(evaluate=True)
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="Run official SWE-bench evaluation after each successful prediction. Enabled by default.",
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="Disable official SWE-bench Docker evaluation.",
    )
    parser.add_argument("--eval-run-id", default="openorchestra")
    parser.add_argument("--eval-max-workers", type=int, default=4)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument(
        "--eval-namespace",
        default="auto",
        help="Docker image namespace for official evaluation. Use auto, none, or a concrete namespace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orchestra_bin = args.orchestra_bin.expanduser().resolve()
    dataset_file = args.dataset_file.expanduser().resolve()
    predictions_file = args.predictions_file.expanduser().resolve()
    case_record_file = args.case_record_file.expanduser().resolve()
    workspace_dir = args.workspace.expanduser().resolve()
    run_log_dir = args.run_log_dir.expanduser().resolve()

    enforce_jsonl_dataset_file(dataset_file)
    preflight_environment(
        args,
        orchestra_bin=orchestra_bin,
        dataset_file=dataset_file,
        predictions_file=predictions_file,
        case_record_file=case_record_file,
        workspace_dir=workspace_dir,
        run_log_dir=run_log_dir,
    )
    remove_legacy_dataset_files(dataset_file)
    download_dataset(args.dataset_name, dataset_file)
    all_tasks = load_tasks(dataset_file)
    dataset = dataset_source(args.dataset_name, dataset_file, all_tasks)
    write_dataset_source(run_log_dir, dataset)
    print(f"[*] dataset sha256: {dataset.sha256}")
    selected_instance_ids = expand_instance_selectors(all_tasks, args.instance_id, option_name="--instance-id")
    tasks = select_tasks(all_tasks, instance_ids=selected_instance_ids)
    if not tasks:
        raise SystemExit("no SWE-bench tasks selected")
    blocked_task_selectors = load_blocked_task_selectors()
    blocked_task_ids, unmatched_blocked_selectors = resolve_blocked_task_ids(all_tasks, blocked_task_selectors)
    if unmatched_blocked_selectors:
        print(
            "[*] blocked_task_list: 忽略当前数据集中不存在的条目: "
            + ", ".join(sorted(unmatched_blocked_selectors))
        )
    skipped_blocked_cases = 0
    if blocked_task_ids:
        before = len(tasks)
        tasks = [task for task in tasks if task["instance_id"] not in blocked_task_ids]
        skipped_blocked_cases = before - len(tasks)
        print(
            "[*] blocked_task_list: 跳过 blocked case "
            f"{skipped_blocked_cases} 个: {', '.join(sorted(blocked_task_ids))}"
        )
    completed_ids = load_completed_prediction_ids(predictions_file) if args.resume else set()
    skipped_existing = 0
    if completed_ids:
        before = len(tasks)
        tasks = [task for task in tasks if task["instance_id"] not in completed_ids]
        skipped_existing = before - len(tasks)
        print(f"[*] resume: 跳过已有非空 patch 的任务 {skipped_existing} 个。")
    successful_case_ids = load_successful_case_ids(case_record_file) if args.skip_successful_cases else set()
    skipped_successful_cases = 0
    if successful_case_ids:
        before = len(tasks)
        tasks = [task for task in tasks if task["instance_id"] not in successful_case_ids]
        skipped_successful_cases = before - len(tasks)
        print(f"[*] skip-successful-cases: 跳过历史官方评测通过 case {skipped_successful_cases} 个。")
    tasks = apply_task_limit(tasks, args.limit)
    if not tasks:
        print("[*] 所选任务均已完成。")
        if args.evaluate and not args.skip_successful_cases and skipped_blocked_cases == 0:
            evaluation = run_official_evaluation(
                args,
                predictions_file,
                run_log_dir,
                instance_ids=sorted(selected_instance_ids),
                max_workers=args.eval_max_workers,
            )
            return evaluation.exit_code
        return 0
    if args.limit > 0:
        print(f"[*] 当前执行 {len(tasks)} 个任务。使用 --limit 0 可跑完整 Verified 集。")
    else:
        print(f"[*] 当前执行完整选择集: {len(tasks)} 个任务。")

    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append or args.resume else "w"
    failures = 0
    completed = 0
    evaluation_completed = 0
    evaluation_failures = 0
    successful_instance_ids: list[str] = []
    failed_instance_ids: list[str] = []
    evaluation_failed_instance_ids: list[str] = []
    timeout_seconds = args.orchestra_timeout if args.orchestra_timeout > 0 else None
    with predictions_file.open(mode, encoding="utf-8") as output:
        for task in tasks:
            result = run_agent_on_task(
                task,
                workspace_dir=workspace_dir,
                run_log_root=run_log_dir,
                orchestra_bin=orchestra_bin,
                backend=args.backend,
                serial_agents=args.serial_agents,
                timeout_seconds=timeout_seconds,
                dataset=dataset,
                ui=args.ui,
            )
            evaluation_exit_code: int | None = None
            official_evaluation_passed: bool | None = None
            evaluation_report_file: Path | None = None
            evaluation_error: str | None = None
            if not result.success:
                failures += 1
                failed_instance_ids.append(result.instance_id)
                print(f"  -> 预测未写入: {result.instance_id} 失败，详见 {run_log_dir / result.instance_id / 'status.json'}")
            else:
                completed += 1
                successful_instance_ids.append(result.instance_id)
                output.write(
                    json.dumps(
                        {
                            "instance_id": result.instance_id,
                            "model_patch": result.patch,
                            "model_name_or_path": args.agent_name,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output.flush()
                print(f"  -> 写入预测: {result.instance_id}, patch 长度: {len(result.patch)}")
                if args.evaluate:
                    print(f"  -> 立即评测: {result.instance_id}")
                    evaluation = run_official_evaluation(
                        args,
                        predictions_file,
                        run_log_dir / result.instance_id,
                        instance_ids=[result.instance_id],
                        max_workers=1,
                    )
                    evaluation_exit_code = evaluation.exit_code
                    official_evaluation_passed = evaluation.resolved
                    evaluation_report_file = evaluation.report_file
                    evaluation_error = evaluation.error
                    if official_evaluation_passed is True:
                        evaluation_completed += 1
                        print(f"  -> 官方评测通过: {result.instance_id}")
                    else:
                        evaluation_failures += 1
                        evaluation_failed_instance_ids.append(result.instance_id)
                        print(f"  [X] 官方评测失败: {result.instance_id}，详见 {run_log_dir / result.instance_id}")
            append_case_record(
                case_record_file,
                build_case_record(
                    task=task,
                    dataset=dataset,
                    args=args,
                    result=result,
                    run_dir=run_log_dir / result.instance_id,
                    evaluation_exit_code=evaluation_exit_code,
                    official_evaluation_passed=official_evaluation_passed,
                    evaluation_report_file=evaluation_report_file,
                    evaluation_error=evaluation_error,
                ),
            )
            if args.fail_fast and not result.success:
                break
            if args.fail_fast and evaluation_failed_instance_ids and evaluation_failed_instance_ids[-1] == result.instance_id:
                break

    if not args.keep_workspace and workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)

    print(f"\n[*] predictions: {predictions_file}")
    print(f"[*] case records: {case_record_file}")
    print(f"[*] run logs: {run_log_dir}")
    print(f"[*] completed: {completed}/{len(tasks)}")
    print(f"[*] failures: {failures}/{len(tasks)}")
    if args.evaluate:
        print(f"[*] evaluated: {evaluation_completed}/{completed}")
        print(f"[*] evaluation failures: {evaluation_failures}/{completed}")
    summary = {
        "dataset_name": args.dataset_name,
        "dataset": dataset_source_payload(dataset),
        "predictions_file": str(predictions_file),
        "case_record_file": str(case_record_file),
        "run_log_dir": str(run_log_dir),
        "selected_tasks": len(tasks),
        "completed": completed,
        "failures": failures,
        "evaluation_completed": evaluation_completed,
        "evaluation_failures": evaluation_failures,
        "successful_instance_ids": successful_instance_ids,
        "failed_instance_ids": failed_instance_ids,
        "evaluation_failed_instance_ids": evaluation_failed_instance_ids,
        "blocked_task_list_file": str(BLOCKED_TASK_LIST_FILE),
        "blocked_task_ids": sorted(blocked_task_ids),
        "unmatched_blocked_task_selectors": sorted(unmatched_blocked_selectors),
        "skipped_blocked_cases": skipped_blocked_cases,
        "skipped_existing": skipped_existing,
        "skip_successful_cases": args.skip_successful_cases,
        "skipped_successful_cases": skipped_successful_cases,
    }
    (run_log_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluation_command = build_evaluation_command(args, predictions_file)
    print("[*] 官方评测命令: " + shlex.join(evaluation_command))
    if args.evaluate:
        return 1 if failures or evaluation_failures else 0
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[interrupt] stopped; OpenOrchestra child process group was terminated.", file=sys.stderr)
        raise SystemExit(130)
