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
WORKSPACE_DIR = SCRIPT_DIR / "temp_workspace"
RUN_LOG_DIR = SCRIPT_DIR / "openorchestra_runs"
ORCHESTRA_BIN = Path("/Users/shixiuwen/workSpace/myHarnessSystem/orchestra")
DEFAULT_AGENT_NAME = "openorchestra"
DEFAULT_TEST_LIMIT = 2


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


def build_agent_prompt(task: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"SWE-bench instance: {task['instance_id']}",
            f"Repository: {task['repo']}",
            f"Base commit: {task['base_commit']}",
            "",
            "Fix the bug described below with the smallest safe code change.",
            "Do not edit tests just to make the benchmark pass.",
            "Do not use or infer from the reference patch or test_patch.",
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
    prompt = build_agent_prompt(task)
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


def check_docker_available() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if not docker:
        return False, "docker CLI not found on PATH. Install/start Docker Desktop before running official SWE-bench evaluation."
    try:
        result = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "docker info timed out after 30s. Start or restart Docker Desktop and retry."
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "docker info failed").strip()
        return False, f"Docker daemon is not available. Start Docker Desktop and retry. Detail: {detail}"
    return True, ""


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


def preflight_environment(args: argparse.Namespace, *, orchestra_bin: Path, dataset_file: Path, predictions_file: Path, workspace_dir: Path, run_log_dir: Path) -> None:
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
) -> int:
    command = build_evaluation_command_for_instances(
        args,
        predictions_file,
        instance_ids=instance_ids,
        max_workers=max_workers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[*] 开始官方 SWE-bench Docker evaluation")
    result = subprocess.run(command, cwd=SCRIPT_DIR, capture_output=True, text=True)
    (output_dir / "evaluation_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (output_dir / "evaluation_stderr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenOrchestra on SWE-bench Verified and produce predictions.jsonl.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-file", type=Path, default=DATASET_FILE)
    parser.add_argument("--predictions-file", type=Path, default=PREDICTIONS_FILE)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE_DIR)
    parser.add_argument("--run-log-dir", type=Path, default=RUN_LOG_DIR)
    parser.add_argument("--orchestra-bin", type=Path, default=ORCHESTRA_BIN)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--limit", type=int, default=DEFAULT_TEST_LIMIT, help="Number of tasks to run. Use 0 for all tasks.")
    parser.add_argument("--instance-id", action="append", default=[], help="Run only this instance_id. Can be repeated.")
    parser.add_argument("--append", action="store_true", help="Append to predictions instead of overwriting it.")
    parser.add_argument("--resume", action="store_true", help="Append and skip instance IDs that already have a non-empty patch.")
    parser.add_argument("--backend", choices=["auto", "codex", "claude", "gemini", "qwen"], default=None)
    parser.add_argument("--serial-agents", action="store_true", help="Pass --serial-agents through to orchestra.")
    parser.add_argument("--orchestra-timeout", type=int, default=0, help="Per-instance OpenOrchestra timeout in seconds. Use 0 for no outer timeout.")
    parser.add_argument("--ui", action="store_true", help="Run OpenOrchestra with its local Web UI enabled and stream its output live.")
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
    workspace_dir = args.workspace.expanduser().resolve()
    run_log_dir = args.run_log_dir.expanduser().resolve()

    enforce_jsonl_dataset_file(dataset_file)
    preflight_environment(
        args,
        orchestra_bin=orchestra_bin,
        dataset_file=dataset_file,
        predictions_file=predictions_file,
        workspace_dir=workspace_dir,
        run_log_dir=run_log_dir,
    )
    remove_legacy_dataset_files(dataset_file)
    download_dataset(args.dataset_name, dataset_file)
    all_tasks = load_tasks(dataset_file)
    dataset = dataset_source(args.dataset_name, dataset_file, all_tasks)
    write_dataset_source(run_log_dir, dataset)
    print(f"[*] dataset sha256: {dataset.sha256}")
    tasks = select_tasks(all_tasks, instance_ids=set(args.instance_id))
    if not tasks:
        raise SystemExit("no SWE-bench tasks selected")
    completed_ids = load_completed_prediction_ids(predictions_file) if args.resume else set()
    skipped_existing = 0
    if completed_ids:
        before = len(tasks)
        tasks = [task for task in tasks if task["instance_id"] not in completed_ids]
        skipped_existing = before - len(tasks)
        print(f"[*] resume: 跳过已有非空 patch 的任务 {skipped_existing} 个。")
    tasks = apply_task_limit(tasks, args.limit)
    if not tasks:
        print("[*] 所选任务均已完成。")
        if args.evaluate:
            return run_official_evaluation(
                args,
                predictions_file,
                run_log_dir,
                instance_ids=args.instance_id or [],
                max_workers=args.eval_max_workers,
            )
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
                    eval_code = run_official_evaluation(
                        args,
                        predictions_file,
                        run_log_dir / result.instance_id,
                        instance_ids=[result.instance_id],
                        max_workers=1,
                    )
                    if eval_code == 0:
                        evaluation_completed += 1
                        print(f"  -> 官方评测通过: {result.instance_id}")
                    else:
                        evaluation_failures += 1
                        evaluation_failed_instance_ids.append(result.instance_id)
                        print(f"  [X] 官方评测失败: {result.instance_id}，详见 {run_log_dir / result.instance_id}")
            if args.fail_fast and not result.success:
                break
            if args.fail_fast and evaluation_failed_instance_ids and evaluation_failed_instance_ids[-1] == result.instance_id:
                break

    if not args.keep_workspace and workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)

    print(f"\n[*] predictions: {predictions_file}")
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
        "run_log_dir": str(run_log_dir),
        "selected_tasks": len(tasks),
        "completed": completed,
        "failures": failures,
        "evaluation_completed": evaluation_completed,
        "evaluation_failures": evaluation_failures,
        "successful_instance_ids": successful_instance_ids,
        "failed_instance_ids": failed_instance_ids,
        "evaluation_failed_instance_ids": evaluation_failed_instance_ids,
        "skipped_existing": skipped_existing,
    }
    (run_log_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluation_command = build_evaluation_command(args, predictions_file)
    print("[*] 官方评测命令: " + shlex.join(evaluation_command))
    if args.evaluate:
        return 1 if failures or evaluation_failures else 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
