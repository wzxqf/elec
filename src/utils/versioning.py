from __future__ import annotations

import re
from pathlib import Path


DEFAULT_PROJECT_CONFIG_FILENAME = "experiment_config.yaml"


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


def load_project_version(project_root: Path, config_path: str | Path = DEFAULT_PROJECT_CONFIG_FILENAME) -> str:
    candidate = Path(config_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    return parse_project_version(candidate.resolve())


def normalize_version_token(version: str) -> str:
    return "".join(character for character in version if character.isalnum())


def build_test_prefix(version: str) -> str:
    return f"test_{normalize_version_token(version)}_"


def build_test_filename(purpose: str, version: str) -> str:
    normalized_purpose = purpose.removeprefix("test_").removesuffix(".py")
    return f"{build_test_prefix(version)}{normalized_purpose}.py"


def build_version_report_filename(version: str) -> str:
    return f"{version}报告.md"
