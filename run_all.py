from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
import warnings

from src.utils.runtime_status import DEFAULT_RUNTIME_STATUS, read_runtime_status


try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)
        import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

ENV_NAME = "torch311"
PANEL_REFRESH_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full elec pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command and exit.")
    return parser.parse_args()


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parent


def parse_project_version(config_path: Path) -> str:
    in_project = False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*#", line):
            continue
        if re.match(r"^project\s*:\s*$", line):
            in_project = True
            continue
        if in_project and re.match(r"^\S", line):
            break
        match = re.match(r"^\s+version\s*:\s*(.+?)\s*(?:#.*)?$", line)
        if in_project and match:
            return match.group(1).strip().strip('"').strip("'")
    raise RuntimeError(f"Unable to parse project.version from {config_path}")


def add_candidate(candidates: list[Path], value: str | None) -> None:
    if not value:
        return
    path = Path(value).expanduser()
    if path not in candidates:
        candidates.append(path)


def resolve_python_executable() -> Path:
    candidates: list[Path] = []
    current = Path(sys.executable).resolve()
    if current.parent.parent.name.lower() == ENV_NAME.lower():
        add_candidate(candidates, str(current))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).name.lower() == ENV_NAME.lower():
        add_candidate(candidates, str(Path(conda_prefix) / "python.exe"))

    mamba_root = os.environ.get("MAMBA_ROOT_PREFIX")
    if mamba_root:
        add_candidate(candidates, str(Path(mamba_root) / "envs" / ENV_NAME / "python.exe"))

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).resolve().parents[1]
        add_candidate(candidates, str(conda_root / "envs" / ENV_NAME / "python.exe"))

    mamba_exe = os.environ.get("MAMBA_EXE")
    if mamba_exe:
        mamba_root_dir = Path(mamba_exe).resolve().parents[2]
        add_candidate(candidates, str(mamba_root_dir / "envs" / ENV_NAME / "python.exe"))

    add_candidate(candidates, fr"D:\miniforge\envs\{ENV_NAME}\python.exe")
    add_candidate(candidates, str(Path.home() / "miniforge3" / "envs" / ENV_NAME / "python.exe"))
    add_candidate(candidates, str(Path.home() / "mambaforge" / "envs" / ENV_NAME / "python.exe"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"Unable to find python.exe for environment '{ENV_NAME}'. Activate it first, or install it under a standard Miniforge/Mambaforge envs directory."
    )


def read_status_snapshot(path: str | Path) -> dict[str, Any]:
    return read_runtime_status(path)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, second = divmod(total_seconds, 60)
    hour, minute = divmod(minutes, 60)
    if hour > 0:
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    return f"{minute:02d}:{second:02d}"


def format_progress_line(
    status: dict[str, Any],
    resources: dict[str, Any],
    elapsed_seconds: float,
    width: int = 160,
) -> str:
    stage = str(status.get("stage") or DEFAULT_RUNTIME_STATUS["stage"])
    phase_name = str(status.get("phase_name") or DEFAULT_RUNTIME_STATUS["phase_name"])
    message = str(status.get("message") or "").strip()
    total_progress = float(status.get("total_progress", 0.0)) * 100.0
    phase_progress = float(status.get("phase_progress", 0.0)) * 100.0
    process_cpu = float(resources.get("process_cpu_percent", 0.0))
    process_memory = float(resources.get("process_memory_gb", 0.0))
    gpu_util = resources.get("gpu_util_percent")
    gpu_used = resources.get("gpu_memory_used_gb")
    gpu_total = resources.get("gpu_memory_total_gb")

    parts = [
        f"[{stage}]",
        phase_name,
        f"总进度 {total_progress:.1f}%",
        f"阶段 {phase_progress:.1f}%",
        f"已耗时 {_format_duration(elapsed_seconds)}",
        f"CPU {process_cpu:.1f}%",
        f"内存 {process_memory:.1f}GB",
    ]
    if gpu_util is not None:
        parts.append(f"GPU {float(gpu_util):.1f}%")
    if gpu_used is not None and gpu_total is not None:
        parts.append(f"显存 {float(gpu_used):.1f}/{float(gpu_total):.1f}GB")
    if message:
        parts.append(message)

    line = " | ".join(parts)
    if len(line) <= width:
        return line
    return line[: max(0, width - 1)] + "…"


def _query_nvidia_smi() -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first = result.stdout.strip().splitlines()[0]
    fields = [item.strip() for item in first.split(",")]
    if len(fields) != 4:
        return None
    gpu_util, _, mem_used_mb, mem_total_mb = fields
    return {
        "gpu_util_percent": float(gpu_util),
        "gpu_memory_used_gb": float(mem_used_mb) / 1024.0,
        "gpu_memory_total_gb": float(mem_total_mb) / 1024.0,
    }


class ResourceMonitor:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.process = psutil.Process(pid) if psutil is not None else None
        if self.process is not None:
            self.process.cpu_percent(interval=None)
            for child in self.process.children(recursive=True):
                child.cpu_percent(interval=None)
        self._nvml_ready = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_ready = True
            except Exception:
                self._nvml_ready = False

    def close(self) -> None:
        if self._nvml_ready and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def sample(self) -> dict[str, Any]:
        payload = {
            "process_cpu_percent": 0.0,
            "process_memory_gb": 0.0,
            "gpu_util_percent": None,
            "gpu_memory_used_gb": None,
            "gpu_memory_total_gb": None,
        }
        if self.process is not None:
            try:
                processes = [self.process] + self.process.children(recursive=True)
                cpu_total = 0.0
                memory_total = 0
                for proc in processes:
                    cpu_total += proc.cpu_percent(interval=None)
                    memory_total += proc.memory_info().rss
                payload["process_cpu_percent"] = cpu_total
                payload["process_memory_gb"] = memory_total / (1024.0**3)
            except Exception:
                pass
        gpu_payload = self._sample_gpu()
        if gpu_payload is not None:
            payload.update(gpu_payload)
        return payload

    def _sample_gpu(self) -> dict[str, float] | None:
        if self._nvml_ready and pynvml is not None:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "gpu_util_percent": float(util.gpu),
                    "gpu_memory_used_gb": float(memory.used) / (1024.0**3),
                    "gpu_memory_total_gb": float(memory.total) / (1024.0**3),
                }
            except Exception:
                pass
        return _query_nvidia_smi()


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root()
    config_path = project_root / "experiment_config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"Configuration file not found: {config_path}")

    version = parse_project_version(config_path)
    python_exe = resolve_python_executable()
    output_dir = project_root / "outputs" / version
    mpl_config_dir = project_root / ".cache" / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    runtime_status_path = output_dir / "logs" / "runtime_status.json"
    stdout_log_path = output_dir / "logs" / "runner_stdout.log"
    stderr_log_path = output_dir / "logs" / "runner_stderr.log"
    run_cmd = [str(python_exe), "-m", "src.scripts.run_pipeline"]

    print(f"Project root: {project_root}")
    print(f"Experiment version: {version}")
    print(f"Output dir: {output_dir}")
    print(f"Runtime status: {runtime_status_path}")
    print(f"Command: {' '.join(run_cmd)}")

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(mpl_config_dir)
    env["ELEC_RUNTIME_STATUS_PATH"] = str(runtime_status_path)
    with stdout_log_path.open("w", encoding="utf-8") as stdout_handle, stderr_log_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(run_cmd, cwd=project_root, env=env, stdout=stdout_handle, stderr=stderr_handle)
        monitor = ResourceMonitor(process.pid)
        start_time = time.perf_counter()
        try:
            while True:
                return_code = process.poll()
                status = read_status_snapshot(runtime_status_path)
                resources = monitor.sample()
                line = format_progress_line(status, resources, elapsed_seconds=time.perf_counter() - start_time, width=180)
                print("\r" + line.ljust(180), end="", flush=True)
                if return_code is not None:
                    break
                time.sleep(PANEL_REFRESH_SECONDS)
        finally:
            monitor.close()
    print()
    if process.returncode != 0:
        print(f"Pipeline failed. Check logs under {output_dir / 'logs'}")
        print(f"stdout log: {stdout_log_path}")
        print(f"stderr log: {stderr_log_path}")
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
