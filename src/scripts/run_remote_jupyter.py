from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import datetime as dt
import html
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener
import uuid
import zipfile

from src.utils.versioning import parse_project_version


DEFAULT_JUPYTER_URL = "http://10.26.27.72:9007/"
DEFAULT_JUPYTER_KERNEL = "python3"
DEFAULT_REMOTE_ENV = "torch311"
DEFAULT_PASSWORD_ENV = "ELEC_JUPYTER_PASSWORD"
DEFAULT_TOKEN_ENV = "ELEC_JUPYTER_TOKEN"
DEFAULT_REMOTE_ROOT_PREFIX = "elec_remote_runs"
DEFAULT_HTTP_TIMEOUT_SECONDS = 300
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 6 * 60 * 60
REMOTE_RESULT_MARKER = "__ELEC_REMOTE_RESULT__"
EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".worktrees",
    "__pycache__",
}
EXCLUDED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".envrc",
}


@dataclass(frozen=True)
class ConnectionConfig:
    url: str
    kernel: str
    password: str | None = field(default=None, repr=False)
    token: str | None = field(default=None, repr=False)
    timeout: int = DEFAULT_HTTP_TIMEOUT_SECONDS


@dataclass(frozen=True)
class RemoteRunSpec:
    run_id: str
    version: str
    archive_remote_path: str
    remote_project_dir: str
    artifact_remote_path: str
    pytest_args: list[str]
    remote_env: str = DEFAULT_REMOTE_ENV
    remote_python: str | None = None
    run_pipeline: bool = True
    run_tests: bool = True


@dataclass(frozen=True)
class KernelExecutionResult:
    status: str
    output: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the current project to Jupyter, run the pipeline/tests remotely, and pull artifacts back.",
    )
    parser.add_argument("--url", help="Jupyter base URL. Defaults to ELEC_JUPYTER_URL or the assigned lab URL.")
    parser.add_argument("--kernel", help="Jupyter kernel spec name. Defaults to ELEC_JUPYTER_KERNEL or python3.")
    parser.add_argument("--remote-env", help="Remote Python environment name used to run the project. Defaults to torch311.")
    parser.add_argument("--remote-python", help="Explicit remote Python executable path. Overrides remote env discovery.")
    parser.add_argument("--password-env", default=DEFAULT_PASSWORD_ENV, help="Environment variable containing the Jupyter password.")
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV, help="Environment variable containing a Jupyter token.")
    parser.add_argument("--remote-root", help="Remote Jupyter contents path for this run.")
    parser.add_argument("--run-id", help="Stable run id. Defaults to a timestamped id.")
    parser.add_argument("--download-dir", help="Local directory for pulled remote artifacts.")
    parser.add_argument("--include-outputs", action="store_true", help="Include local outputs/ in the upload bundle.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Do not run python run_all.py on Jupyter.")
    parser.add_argument("--skip-tests", action="store_true", help="Do not run pytest on Jupyter.")
    parser.add_argument("--probe", action="store_true", help="Only verify Jupyter login and kernel availability.")
    parser.add_argument("--dry-run", action="store_true", help="Describe the remote run without connecting.")
    parser.add_argument("--build-archive-on-dry-run", action="store_true", help="Actually create the source zip during --dry-run.")
    parser.add_argument(
        "--no-sync-local-output",
        action="store_true",
        help="Do not replace local outputs/<version> with the pulled remote outputs after a successful run.",
    )
    parser.add_argument("--http-timeout", type=int, default=DEFAULT_HTTP_TIMEOUT_SECONDS)
    parser.add_argument("--execution-timeout", type=int, default=DEFAULT_EXECUTION_TIMEOUT_SECONDS)
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to src.scripts.run_pytest on Jupyter. Must be last. Defaults to: tests -q",
    )
    return parser.parse_args(argv)


def build_connection_config(args: argparse.Namespace) -> ConnectionConfig:
    url = (args.url or os.environ.get("ELEC_JUPYTER_URL") or DEFAULT_JUPYTER_URL).rstrip("/") + "/"
    kernel = args.kernel or os.environ.get("ELEC_JUPYTER_KERNEL") or DEFAULT_JUPYTER_KERNEL
    password = os.environ.get(args.password_env)
    token = os.environ.get(args.token_env)
    timeout = int(getattr(args, "timeout", None) or getattr(args, "http_timeout", DEFAULT_HTTP_TIMEOUT_SECONDS))
    if not password and not token:
        raise RuntimeError(
            f"Set {args.password_env} or {args.token_env} before connecting to Jupyter. "
            "Do not write the password into tracked project files."
        )
    return ConnectionConfig(url=url, kernel=kernel, password=password, token=token, timeout=timeout)


def default_run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"jupyter-{timestamp}"


def default_download_dir(project_root: Path, version: str, run_id: str) -> Path:
    return project_root / "outputs" / version / "remote_jupyter" / run_id


def should_exclude_from_archive(relative_path: PurePosixPath, is_dir: bool, include_outputs: bool) -> bool:
    parts = set(relative_path.parts)
    if parts & EXCLUDED_DIRECTORY_NAMES:
        return True
    if not include_outputs and relative_path.parts and relative_path.parts[0] == "outputs":
        return True
    name = relative_path.name
    if name in EXCLUDED_FILE_NAMES:
        return True
    if name.startswith(".env.") and name != ".env.example":
        return True
    if not is_dir and name.endswith((".pyc", ".pyo", ".pyd")):
        return True
    return False


def build_source_archive(project_root: Path, archive_path: Path, include_outputs: bool = False) -> dict[str, Any]:
    project_root = project_root.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    byte_count = 0
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for current_root, dir_names, file_names in os.walk(project_root):
            current_path = Path(current_root)
            kept_dirs = []
            for dir_name in sorted(dir_names):
                dir_path = current_path / dir_name
                relative_dir = PurePosixPath(dir_path.relative_to(project_root).as_posix())
                if should_exclude_from_archive(relative_dir, True, include_outputs):
                    continue
                kept_dirs.append(dir_name)
            dir_names[:] = kept_dirs
            if current_path != project_root:
                relative_current = PurePosixPath(current_path.relative_to(project_root).as_posix())
                if not should_exclude_from_archive(relative_current, True, include_outputs):
                    archive.writestr(relative_current.as_posix().rstrip("/") + "/", "")
            for file_name in sorted(file_names):
                path = current_path / file_name
                relative_path = PurePosixPath(path.relative_to(project_root).as_posix())
                if should_exclude_from_archive(relative_path, False, include_outputs):
                    continue
                if not path.is_file():
                    continue
                if path == archive_path:
                    continue
                if archive_path in path.parents:
                    continue
                archive.write(path, relative_path.as_posix())
                file_count += 1
                byte_count += path.stat().st_size
    return {
        "archive_path": str(archive_path),
        "file_count": file_count,
        "byte_count": byte_count,
        "include_outputs": include_outputs,
    }


def extract_kernel_names(kernelspec_payload: dict[str, Any]) -> list[str]:
    kernelspecs = kernelspec_payload.get("kernelspecs", {})
    if not isinstance(kernelspecs, dict):
        return []
    return sorted(str(name) for name in kernelspecs)


def build_remote_python_resolver_code() -> str:
    return r'''
def _python_executable_from_env_path(env_path):
    env_path = Path(env_path)
    for relative in ("bin/python", "python.exe"):
        candidate = env_path / relative
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_remote_python_command(env_name, explicit_python=None):
    explicit = explicit_python or os.environ.get("ELEC_REMOTE_PYTHON")
    if explicit:
        return [explicit]

    candidate_env_paths = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).name == env_name:
        candidate_env_paths.append(conda_prefix)
    for root_env in ("MAMBA_ROOT_PREFIX", "CONDA_ROOT", "CONDA_PREFIX_1"):
        root = os.environ.get(root_env)
        if root:
            candidate_env_paths.append(str(Path(root) / "envs" / env_name))
    home = Path.home()
    candidate_env_paths.extend(
        [
            f"/opt/conda/envs/{env_name}",
            f"/opt/miniconda/envs/{env_name}",
            f"/opt/miniconda3/envs/{env_name}",
            f"/opt/miniforge/envs/{env_name}",
            f"/opt/miniforge3/envs/{env_name}",
            f"/research/miniforge3/envs/{env_name}",
            f"/research/miniforge/envs/{env_name}",
            f"/usr/local/conda/envs/{env_name}",
            f"/usr/local/miniconda/envs/{env_name}",
            str(home / ".conda" / "envs" / env_name),
            str(home / "miniconda" / "envs" / env_name),
            str(home / "miniconda3" / "envs" / env_name),
            str(home / "miniforge" / "envs" / env_name),
            str(home / "miniforge3" / "envs" / env_name),
        ]
    )
    for env_path in candidate_env_paths:
        executable = _python_executable_from_env_path(env_path)
        if executable:
            return [executable]

    for tool_name in ("mamba", "conda"):
        tool = shutil.which(tool_name)
        if not tool:
            continue
        try:
            completed = subprocess.run([tool, "env", "list", "--json"], check=False, capture_output=True, text=True, timeout=20)
            if completed.returncode == 0:
                envs = json.loads(completed.stdout).get("envs", [])
                for env_path in envs:
                    if Path(env_path).name == env_name:
                        executable = _python_executable_from_env_path(env_path)
                        if executable:
                            return [executable]
        except Exception:
            pass
        return [tool, "run", "-n", env_name, "python"]

    raise RuntimeError(f"Unable to find remote Python environment: {env_name}")
'''


def build_probe_code(remote_env: str = DEFAULT_REMOTE_ENV, remote_python: str | None = None) -> str:
    resolver = build_remote_python_resolver_code()
    config_json = json.dumps({"remote_env": remote_env, "remote_python": remote_python}, ensure_ascii=False)
    return f"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CONFIG = json.loads({config_json!r})

{resolver}

payload = {{
    "kernel_sys_executable": sys.executable,
    "kernel_sys_prefix": sys.prefix,
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    "remote_env": CONFIG["remote_env"],
}}
try:
    command = resolve_remote_python_command(CONFIG["remote_env"], CONFIG["remote_python"])
    completed = subprocess.run(
        [*command, "-c", "import json, sys; print(json.dumps({{'sys_executable': sys.executable, 'sys_prefix': sys.prefix}}))"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload["remote_python_command"] = command
    payload["remote_python_probe_returncode"] = completed.returncode
    payload["remote_python_probe_stdout"] = completed.stdout.strip()
    payload["remote_python_probe_stderr"] = completed.stderr.strip()
except Exception as exc:
    payload["remote_python_error"] = str(exc)
print("__ELEC_JUPYTER_PROBE__" + json.dumps(payload, ensure_ascii=False), flush=True)
"""


def build_remote_execution_code(spec: RemoteRunSpec) -> str:
    config = {
        "run_id": spec.run_id,
        "version": spec.version,
        "archive_remote_path": spec.archive_remote_path,
        "remote_project_dir": spec.remote_project_dir,
        "artifact_remote_path": spec.artifact_remote_path,
        "pytest_args": spec.pytest_args,
        "remote_env": spec.remote_env,
        "remote_python": spec.remote_python,
        "run_pipeline": spec.run_pipeline,
        "run_tests": spec.run_tests,
        "result_marker": REMOTE_RESULT_MARKER,
    }
    config_json = json.dumps(config, ensure_ascii=False, indent=2)
    resolver = build_remote_python_resolver_code()
    return f"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

CONFIG = json.loads({config_json!r})

{resolver}


def add_tree(zip_handle, root, archive_root):
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            zip_handle.write(path, f"{{archive_root}}/{{relative}}")


archive_path = Path(CONFIG["archive_remote_path"])
project_dir = Path(CONFIG["remote_project_dir"])
artifact_path = Path(CONFIG["artifact_remote_path"])
project_dir.parent.mkdir(parents=True, exist_ok=True)
if project_dir.exists():
    shutil.rmtree(project_dir)
project_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive_path, "r") as source_zip:
    source_zip.extractall(project_dir)

env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["MPLCONFIGDIR"] = str(project_dir / ".cache" / "matplotlib")
(project_dir / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

commands = []
results = []
returncode = 0
remote_python = []
try:
    remote_python = resolve_remote_python_command(CONFIG["remote_env"], CONFIG["remote_python"])
    print(":: remote python:", " ".join(remote_python), flush=True)
except Exception as exc:
    returncode = 1
    print(":: remote python resolution failed:", str(exc), flush=True)
    results.append({{
        "name": "resolve_python",
        "argv": [CONFIG["remote_env"]],
        "returncode": 1,
        "elapsed_seconds": 0.0,
        "error": str(exc),
    }})
if returncode == 0 and CONFIG["run_pipeline"]:
    commands.append({{"name": "pipeline", "argv": [*remote_python, "run_all.py"]}})
if returncode == 0 and CONFIG["run_tests"]:
    commands.append({{"name": "pytest", "argv": [*remote_python, "-m", "src.scripts.run_pytest", *CONFIG["pytest_args"]]}})
for command in commands:
    started = time.time()
    print(":: remote command start:", command["name"], " ".join(command["argv"]), flush=True)
    completed = subprocess.run(command["argv"], cwd=project_dir, env=env, check=False)
    elapsed = time.time() - started
    print(":: remote command end:", command["name"], "returncode", completed.returncode, "elapsed", round(elapsed, 2), flush=True)
    results.append({{
        "name": command["name"],
        "argv": command["argv"],
        "returncode": int(completed.returncode),
        "elapsed_seconds": elapsed,
    }})
    if completed.returncode != 0:
        returncode = int(completed.returncode)
        break

output_dir = project_dir / "outputs" / CONFIG["version"]
pytest_dir = project_dir / ".cache" / "tests" / "pytest"
manifest = {{
    "run_id": CONFIG["run_id"],
    "version": CONFIG["version"],
    "project_dir": str(project_dir),
    "output_dir": str(output_dir),
    "pytest_dir": str(pytest_dir),
    "commands": results,
    "remote_env": CONFIG["remote_env"],
    "remote_python": remote_python,
    "returncode": returncode,
    "artifact_path": str(artifact_path),
}}
manifest_path = artifact_path.parent / "remote_jupyter_run_manifest.json"
artifact_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
with zipfile.ZipFile(artifact_path, "w", compression=zipfile.ZIP_DEFLATED) as artifacts:
    add_tree(artifacts, output_dir, f"outputs/{{CONFIG['version']}}")
    add_tree(artifacts, pytest_dir, "pytest")
    artifacts.write(manifest_path, "remote_jupyter_run_manifest.json")

print(CONFIG["result_marker"] + json.dumps(manifest, ensure_ascii=False), flush=True)
"""


def _quote_content_path(path: str) -> str:
    return "/".join(quote(part, safe="") for part in PurePosixPath(path).parts if part not in {"."})


def _extract_xsrf_from_html(body: str) -> str | None:
    marker = 'name="_xsrf"'
    marker_index = body.find(marker)
    if marker_index < 0:
        return None
    value_index = body.rfind("value=", 0, marker_index + 200)
    if value_index < 0:
        value_index = body.find("value=", marker_index, marker_index + 200)
    if value_index < 0:
        return None
    quote_char = body[value_index + len("value=") : value_index + len("value=") + 1]
    if quote_char not in {"'", '"'}:
        return None
    start = value_index + len("value=") + 1
    end = body.find(quote_char, start)
    if end < 0:
        return None
    return html.unescape(body[start:end])


class JupyterClient:
    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config
        self.cookies = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookies))

    def login(self) -> None:
        if self.config.token:
            self.request_json("GET", "api")
            return
        login_body = self._request_text("GET", "login")
        xsrf = self._xsrf_cookie() or _extract_xsrf_from_html(login_body)
        form = {"password": self.config.password or ""}
        if xsrf:
            form["_xsrf"] = xsrf
        self._request_text(
            "POST",
            "login?next=%2F",
            data=urlencode(form).encode("utf-8"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": urljoin(self.config.url, "login"),
            },
        )
        self.request_json("GET", "api")

    def request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> Any:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        text = self._request_text(method, path, data=data, headers=headers, query=query)
        return json.loads(text) if text else None

    def make_directory(self, path: str) -> None:
        current_parts: list[str] = []
        for part in PurePosixPath(path).parts:
            if part in {"."}:
                continue
            current_parts.append(part)
            current = PurePosixPath(*current_parts).as_posix()
            if self.content_exists(current):
                continue
            self.request_json("PUT", f"api/contents/{_quote_content_path(current)}", {"type": "directory"})

    def content_exists(self, path: str) -> bool:
        try:
            self.request_json("GET", f"api/contents/{_quote_content_path(path)}", query={"content": "0"})
            return True
        except HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        parent = PurePosixPath(remote_path).parent.as_posix()
        if parent not in {"", "."}:
            self.make_directory(parent)
        content = base64.b64encode(local_path.read_bytes()).decode("ascii")
        self.request_json(
            "PUT",
            f"api/contents/{_quote_content_path(remote_path)}",
            {"type": "file", "format": "base64", "content": content},
        )

    def download_file(self, remote_path: str) -> bytes:
        payload = self.request_json("GET", f"api/contents/{_quote_content_path(remote_path)}", query={"content": "1"})
        if payload.get("format") == "base64":
            return base64.b64decode(payload["content"])
        return str(payload.get("content", "")).encode("utf-8")

    def start_kernel(self, kernel_name: str) -> str:
        payload = self.request_json("POST", "api/kernels", {"name": kernel_name})
        return str(payload["id"])

    def shutdown_kernel(self, kernel_id: str) -> None:
        try:
            self.request_json("DELETE", f"api/kernels/{quote(kernel_id, safe='')}")
        except HTTPError:
            pass

    def execute_python(self, kernel_id: str, code: str, timeout: int) -> KernelExecutionResult:
        try:
            import websocket
        except ImportError as exc:  # pragma: no cover - depends on host environment
            raise RuntimeError("Install websocket-client before using the remote Jupyter runner.") from exc

        session_id = uuid.uuid4().hex
        ws_url = self._websocket_url(kernel_id, session_id)
        headers = []
        cookie_header = self._cookie_header()
        if cookie_header:
            headers.append(f"Cookie: {cookie_header}")
        ws = websocket.create_connection(ws_url, timeout=self.config.timeout, header=headers)
        output_parts: list[str] = []
        msg_id = uuid.uuid4().hex
        message = {
            "header": {
                "msg_id": msg_id,
                "username": "elec",
                "session": session_id,
                "date": dt.datetime.now(dt.timezone.utc).isoformat(),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
            "buffers": [],
        }
        status = "unknown"
        deadline = time.monotonic() + timeout
        try:
            ws.send(json.dumps(message))
            while time.monotonic() < deadline:
                raw = ws.recv()
                payload = json.loads(raw)
                if payload.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                channel = payload.get("channel")
                msg_type = payload.get("msg_type") or payload.get("header", {}).get("msg_type")
                content = payload.get("content", {})
                if channel == "iopub" and msg_type == "stream":
                    text = str(content.get("text", ""))
                    print(text, end="", flush=True)
                    output_parts.append(text)
                elif channel == "iopub" and msg_type == "error":
                    traceback = "\n".join(content.get("traceback", []))
                    print(traceback, flush=True)
                    output_parts.append(traceback)
                elif channel == "shell" and msg_type == "execute_reply":
                    status = str(content.get("status", "unknown"))
                    break
            else:
                raise TimeoutError(f"Remote Jupyter execution exceeded {timeout} seconds.")
        finally:
            ws.close()
        return KernelExecutionResult(status=status, output="".join(output_parts))

    def _request_text(
        self,
        method: str,
        path: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
    ) -> str:
        url = urljoin(self.config.url, path)
        query_items = dict(query or {})
        if self.config.token:
            query_items["token"] = self.config.token
        if query_items:
            separator = "&" if "?" in url else "?"
            url = url + separator + urlencode(query_items)
        request_headers = dict(headers or {})
        xsrf = self._xsrf_cookie()
        if xsrf and method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            request_headers["X-XSRFToken"] = xsrf
        if self.config.token:
            request_headers["Authorization"] = f"token {self.config.token}"
        request = Request(url, data=data, headers=request_headers, method=method.upper())
        with self.opener.open(request, timeout=self.config.timeout) as response:
            return response.read().decode("utf-8")

    def _xsrf_cookie(self) -> str | None:
        for cookie in self.cookies:
            if cookie.name == "_xsrf":
                return str(cookie.value)
        return None

    def _cookie_header(self) -> str:
        return "; ".join(f"{cookie.name}={cookie.value}" for cookie in self.cookies)

    def _websocket_url(self, kernel_id: str, session_id: str) -> str:
        parsed = urlparse(self.config.url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path.rstrip("/")
        query = {"session_id": session_id}
        if self.config.token:
            query["token"] = self.config.token
        return f"{scheme}://{parsed.netloc}{base_path}/api/kernels/{quote(kernel_id, safe='')}/channels?{urlencode(query)}"


def parse_remote_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if REMOTE_RESULT_MARKER in line:
            payload = line.split(REMOTE_RESULT_MARKER, 1)[1]
            return json.loads(payload)
    raise RuntimeError("Remote execution did not emit the expected result marker.")


def parse_probe_result(output: str) -> dict[str, Any]:
    marker = "__ELEC_JUPYTER_PROBE__"
    for line in reversed(output.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1]
            return json.loads(payload)
    raise RuntimeError("Probe did not emit the expected result marker.")


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Refusing to extract path outside destination: {member.filename}")
            archive.extract(member, destination)


def pull_remote_artifacts(client: JupyterClient, remote_path: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    zip_path = local_dir / "remote_artifacts.zip"
    zip_path.write_bytes(client.download_file(remote_path))
    safe_extract_zip(zip_path, local_dir)
    return zip_path


def normalize_tree_attributes(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.chmod(0o777)
    except OSError:
        pass
    if path.is_dir():
        for child in path.rglob("*"):
            try:
                child.chmod(0o777)
            except OSError:
                pass


def count_files(path: Path) -> int:
    if path.is_file():
        return 1
    if not path.exists():
        return 0
    return sum(1 for child in path.rglob("*") if child.is_file())


def remove_existing_path(path: Path) -> None:
    if not path.exists():
        return
    normalize_tree_attributes(path)
    if path.is_dir():
        def _retry_remove(function: Any, failing_path: str, _: Any) -> None:
            os.chmod(failing_path, stat.S_IWRITE)
            function(failing_path)

        shutil.rmtree(path, onerror=_retry_remove)
    else:
        path.chmod(0o777)
        path.unlink()


def sync_pulled_outputs_to_local(
    download_dir: Path,
    project_root: Path,
    version: str,
    preserve_names: tuple[str, ...] = ("remote_jupyter",),
) -> dict[str, Any]:
    source_output_dir = (download_dir / "outputs" / version).resolve()
    project_root = project_root.resolve()
    target_output_dir = (project_root / "outputs" / version).resolve()
    if not source_output_dir.is_dir():
        raise RuntimeError(f"Pulled remote output directory not found: {source_output_dir}")
    if target_output_dir == source_output_dir:
        raise RuntimeError("Refusing to sync remote outputs onto the same directory.")
    if project_root != target_output_dir and project_root not in target_output_dir.parents:
        raise RuntimeError(f"Refusing to sync outside project root: {target_output_dir}")
    if target_output_dir in source_output_dir.parents:
        relative_source = source_output_dir.relative_to(target_output_dir)
        if not relative_source.parts or relative_source.parts[0] not in preserve_names:
            raise RuntimeError(f"Pulled output directory is inside a replaceable local output path: {source_output_dir}")

    target_output_dir.mkdir(parents=True, exist_ok=True)
    removed_names: list[str] = []
    for child in sorted(target_output_dir.iterdir(), key=lambda item: item.name):
        if child.name in preserve_names:
            continue
        remove_existing_path(child)
        removed_names.append(child.name)

    synced_files = 0
    synced_names: list[str] = []
    for child in sorted(source_output_dir.iterdir(), key=lambda item: item.name):
        destination = target_output_dir / child.name
        if destination.name in preserve_names:
            continue
        if destination.exists():
            remove_existing_path(destination)
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)
        synced_files += count_files(destination)
        synced_names.append(child.name)

    return {
        "source_output_dir": str(source_output_dir),
        "target_output_dir": str(target_output_dir),
        "preserved_names": list(preserve_names),
        "removed_names": removed_names,
        "synced_names": synced_names,
        "synced_files": synced_files,
    }


def write_local_client_manifest(
    local_dir: Path,
    connection: ConnectionConfig,
    source_manifest: dict[str, Any],
    remote_manifest: dict[str, Any] | None,
    sync_manifest: dict[str, Any] | None = None,
) -> None:
    payload = {
        "jupyter_url": connection.url,
        "kernel": connection.kernel,
        "source_archive": source_manifest,
        "remote_manifest": remote_manifest,
        "local_output_sync": sync_manifest,
    }
    (local_dir / "remote_jupyter_client_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "experiment_config.yaml"
    version = parse_project_version(config_path)
    remote_env = args.remote_env or os.environ.get("ELEC_REMOTE_ENV") or DEFAULT_REMOTE_ENV
    remote_python = args.remote_python or os.environ.get("ELEC_REMOTE_PYTHON")
    if args.probe:
        connection = build_connection_config(args)
        client = JupyterClient(connection)
        client.login()
        kernels = extract_kernel_names(client.request_json("GET", "api/kernelspecs"))
        print(f"Jupyter URL: {connection.url}")
        print(f"Available kernels: {', '.join(kernels) if kernels else '<none>'}")
        print(f"Remote project env: {remote_env}")
        if connection.kernel not in kernels:
            print(f"Kernel not found: {connection.kernel}")
            return 1
        print(f"Kernel ready: {connection.kernel}")
        kernel_id = client.start_kernel(connection.kernel)
        try:
            execution = client.execute_python(kernel_id, build_probe_code(remote_env, remote_python), timeout=60)
        finally:
            client.shutdown_kernel(kernel_id)
        if execution.status != "ok":
            print(f"Probe execution failed: {execution.status}")
            return 1
        probe = parse_probe_result(execution.output)
        if probe.get("remote_python_error"):
            print(f"Remote Python not resolved: {probe['remote_python_error']}")
            return 1
        if int(probe.get("remote_python_probe_returncode", 1)) != 0:
            print(f"Remote Python probe failed: {probe.get('remote_python_probe_stderr', '')}")
            return 1
        return 0

    run_id = args.run_id or default_run_id()
    remote_root = (args.remote_root or f"{DEFAULT_REMOTE_ROOT_PREFIX}/{version}/{run_id}").strip("/")
    remote_archive_path = f"{remote_root}/source.zip"
    remote_project_dir = f"{remote_root}/project"
    remote_artifact_path = f"{remote_root}/artifacts.zip"
    local_cache = project_root / ".cache" / "remote_jupyter" / version / run_id
    local_archive_path = local_cache / "source.zip"
    if args.dry_run and not args.build_archive_on_dry_run:
        source_manifest = {
            "archive_path": str(local_archive_path),
            "file_count": "not-built",
            "byte_count": "not-built",
            "include_outputs": args.include_outputs,
        }
    else:
        source_manifest = build_source_archive(project_root, local_archive_path, include_outputs=args.include_outputs)
    pytest_args = args.pytest_args if args.pytest_args is not None else ["tests", "-q"]
    spec = RemoteRunSpec(
        run_id=run_id,
        version=version,
        archive_remote_path=remote_archive_path,
        remote_project_dir=remote_project_dir,
        artifact_remote_path=remote_artifact_path,
        pytest_args=pytest_args,
        remote_env=remote_env,
        remote_python=remote_python,
        run_pipeline=not args.skip_pipeline,
        run_tests=not args.skip_tests,
    )
    download_dir = Path(args.download_dir).resolve() if args.download_dir else default_download_dir(project_root, version, run_id)

    print(f"Project root: {project_root}")
    print(f"Experiment version: {version}")
    print(f"Run id: {run_id}")
    print(f"Source archive: {local_archive_path}")
    print(f"Archive files: {source_manifest['file_count']}")
    print(f"Remote root: {remote_root}")
    print(f"Remote project env: {remote_env}")
    print(f"Local artifact dir: {download_dir}")
    if args.dry_run:
        return 0

    connection = build_connection_config(args)
    client = JupyterClient(connection)
    remote_manifest: dict[str, Any] | None = None
    kernel_id: str | None = None
    try:
        client.login()
        client.upload_file(local_archive_path, remote_archive_path)
        kernel_id = client.start_kernel(connection.kernel)
        execution = client.execute_python(kernel_id, build_remote_execution_code(spec), timeout=args.execution_timeout)
        remote_manifest = parse_remote_result(execution.output)
        pull_remote_artifacts(client, remote_artifact_path, download_dir)
        remote_returncode = int(remote_manifest.get("returncode", 1))
        sync_manifest = None
        if args.no_sync_local_output:
            print("Skipped local output sync because --no-sync-local-output was set.")
        elif not spec.run_pipeline:
            print("Skipped local output sync because --skip-pipeline was set.")
        elif remote_returncode == 0:
            sync_manifest = sync_pulled_outputs_to_local(download_dir, project_root, version)
            print(f"Synced remote outputs to: {sync_manifest['target_output_dir']}")
        else:
            print(f"Skipped local output sync because remote returncode was {remote_returncode}.")
        write_local_client_manifest(download_dir, connection, source_manifest, remote_manifest, sync_manifest)
        print(f"Pulled remote artifacts to: {download_dir}")
        return remote_returncode
    finally:
        if kernel_id is not None:
            client.shutdown_kernel(kernel_id)


if __name__ == "__main__":
    raise SystemExit(main())
