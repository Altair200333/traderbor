from __future__ import annotations

import hashlib
from pathlib import Path

from traderbot_ai.paths import PROJECT_ROOT
from traderbot_ai.screener.render import to_canonical_json
from traderbot_ai.screener.screener import ScanResult


DEFAULT_SCREENER_ARTIFACT_ROOT = PROJECT_ROOT / "worklog" / "screener"


def write_scan_artifacts(result: ScanResult, run_id: str, root: str | Path | None = None) -> dict[str, str]:
    base = Path(root) if root is not None else DEFAULT_SCREENER_ARTIFACT_ROOT
    safe_run_id = _safe_part(run_id)
    directory = base / safe_run_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = to_canonical_json(result)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    json_path = directory / f"{result.as_of_ms}.json"
    sha_path = directory / f"{result.as_of_ms}.sha256"
    json_path.write_text(payload, encoding="utf-8")
    sha_path.write_text(digest + "\n", encoding="utf-8")
    return {"artifact_path": str(json_path), "sha256_path": str(sha_path), "sha256": digest}


def _safe_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in str(value)).strip(".-")
    return safe or "scan"
