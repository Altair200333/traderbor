from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
AGENTS_V2_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = AGENTS_V2_ROOT.parent

DATA_DIR = AGENTS_V2_ROOT / "data"
RUNS_DIR = AGENTS_V2_ROOT / "runs"
CHARTS_DIR = AGENTS_V2_ROOT / "charts"
ARTIFACTS_DIR = AGENTS_V2_ROOT / "artifacts"
TMP_DIR = AGENTS_V2_ROOT / "tmp"
WORKLOG_DIR = PROJECT_ROOT / "worklog"


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, RUNS_DIR, CHARTS_DIR, ARTIFACTS_DIR, TMP_DIR, WORKLOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def safe_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes project root: {path}")
    return resolved


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def safe_agents_path(path: str | Path) -> Path:
    raw = str(path).strip()
    normalized = raw.replace("\\", "/")
    for prefix in ("./agents-v2/", "agents-v2/"):
        if normalized.lower().startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = AGENTS_V2_ROOT / candidate

    resolved = candidate.resolve()
    root = AGENTS_V2_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes agents-v2 root: {path}")
    return resolved


def display_agents_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(AGENTS_V2_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)
