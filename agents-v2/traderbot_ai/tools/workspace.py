from __future__ import annotations

import difflib
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from agents import function_tool

from traderbot_ai.paths import (
    AGENTS_V2_ROOT,
    TMP_DIR,
    WORKLOG_DIR,
    display_agents_path,
    display_path,
    ensure_runtime_dirs,
    safe_agents_path,
)
from traderbot_ai.runtime.run_context import get_current_run_id, normalize_run_id


ReadUnit = Literal["chars", "lines"]
WriteMode = Literal["overwrite", "append", "prepend", "insert", "replace"]

DEFAULT_READ_CHARS = 20000
MAX_READ_CHARS = 120000
MAX_READ_LINES = 2000
MAX_SEARCH_FILE_BYTES = 2_000_000
MAX_SEARCH_RESULTS = 200


def _ok(**data) -> dict:
    return {"ok": True, **data}


def _error(message: str, **data) -> dict:
    return {"ok": False, "error": message, **data}


def _is_probably_text(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\x00" not in sample


def list_local_files_impl(path: str = ".", pattern: str = "*", max_items: int = 80) -> dict:
    try:
        ensure_runtime_dirs()
        base = safe_agents_path(path)
        if not base.exists():
            return _error("path does not exist", path=display_agents_path(base))
        if not base.is_dir():
            return _error("path is not a directory", path=display_agents_path(base))

        items = []
        limit = max(1, min(max_items, 500))
        for item in sorted(base.glob(pattern)):
            items.append(
                {
                    "path": display_agents_path(item),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
            if len(items) >= limit:
                break
        return _ok(root="agents-v2", path=display_agents_path(base), items=items)
    except Exception as error:
        return _error(str(error), path=path)


def read_local_file_impl(
    path: str,
    unit: ReadUnit = "chars",
    seek: int = 0,
    count: int = DEFAULT_READ_CHARS,
    encoding: str = "utf-8",
) -> dict:
    try:
        file_path = safe_agents_path(path)
        if not file_path.exists():
            return _error("file does not exist", path=display_agents_path(file_path))
        if not file_path.is_file():
            return _error("path is not a file", path=display_agents_path(file_path))
        if not _is_probably_text(file_path):
            return _error("file looks binary; text read refused", path=display_agents_path(file_path))

        text = file_path.read_text(encoding=encoding, errors="replace")
        if unit == "chars":
            start = max(0, seek)
            limit = max(1, min(count, MAX_READ_CHARS))
            end = min(len(text), start + limit)
            return _ok(
                path=display_agents_path(file_path),
                unit=unit,
                seek=start,
                count=end - start,
                next_seek=end if end < len(text) else None,
                total_chars=len(text),
                truncated=end < len(text),
                content=text[start:end],
            )
        if unit == "lines":
            lines = text.splitlines(keepends=True)
            start_line = max(1, seek or 1)
            limit = max(1, min(count, MAX_READ_LINES))
            start_index = start_line - 1
            end_index = min(len(lines), start_index + limit)
            return _ok(
                path=display_agents_path(file_path),
                unit=unit,
                seek=start_line,
                count=end_index - start_index,
                next_seek=(end_index + 1) if end_index < len(lines) else None,
                total_lines=len(lines),
                truncated=end_index < len(lines),
                content="".join(lines[start_index:end_index]),
            )
        return _error("unit must be chars or lines", path=display_agents_path(file_path), unit=unit)
    except LookupError:
        return _error(f"unknown text encoding: {encoding}", path=path)
    except Exception as error:
        return _error(str(error), path=path)


def write_local_file_impl(
    path: str,
    content: str,
    mode: WriteMode = "overwrite",
    seek: int = 0,
    length: int = 0,
    create_dirs: bool = True,
    encoding: str = "utf-8",
) -> dict:
    try:
        file_path = safe_agents_path(path)
        if file_path.exists() and file_path.is_dir():
            return _error("path is a directory", path=display_agents_path(file_path))
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        elif not file_path.parent.exists():
            return _error("parent directory does not exist", path=display_agents_path(file_path.parent))

        old_text = ""
        if file_path.exists():
            if not _is_probably_text(file_path):
                return _error("file looks binary; text write refused", path=display_agents_path(file_path))
            old_text = file_path.read_text(encoding=encoding, errors="replace")
        elif mode in {"append", "prepend", "insert", "replace"}:
            old_text = ""

        if mode == "overwrite":
            new_text = content
        elif mode == "append":
            new_text = old_text + content
        elif mode == "prepend":
            new_text = content + old_text
        elif mode == "insert":
            start = max(0, min(seek, len(old_text)))
            new_text = old_text[:start] + content + old_text[start:]
        elif mode == "replace":
            start = max(0, min(seek, len(old_text)))
            end = max(start, min(start + max(0, length), len(old_text)))
            new_text = old_text[:start] + content + old_text[end:]
        else:
            return _error("mode must be overwrite, append, prepend, insert, or replace", path=path, mode=mode)

        file_path.write_text(new_text, encoding=encoding)
        return _ok(
            path=display_agents_path(file_path),
            mode=mode,
            chars=len(new_text),
            bytes=file_path.stat().st_size,
        )
    except LookupError:
        return _error(f"unknown text encoding: {encoding}", path=path)
    except Exception as error:
        return _error(str(error), path=path)


def _best_fuzzy_score(query: str, text: str) -> float:
    q = query.lower().strip()
    target = text.lower()
    if not q or not target:
        return 0.0
    if q in target:
        return 1.0
    return difflib.SequenceMatcher(None, q, target).ratio()


def _snippet(text: str, start: int, length: int, slice_chars: int) -> str:
    left = max(0, start - slice_chars)
    right = min(len(text), start + length + slice_chars)
    snippet = text[left:right].replace("\r", "")
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."
    return snippet


def search_local_files_impl(
    query: str,
    path: str = ".",
    fuzzy: bool = False,
    count: int = 20,
    page: int = 1,
    slice_chars: int = 80,
    pattern: str = "*",
) -> dict:
    try:
        ensure_runtime_dirs()
        needle = query.strip()
        if not needle:
            return _error("query is empty")
        root = safe_agents_path(path)
        if not root.exists():
            return _error("path does not exist", path=display_agents_path(root))
        if root.is_file():
            files = [root]
        else:
            files = [item for item in root.rglob(pattern) if item.is_file()]

        matches = []
        slice_limit = max(0, min(slice_chars, 500))
        for file_path in files:
            rel = display_agents_path(file_path)
            file_score = _best_fuzzy_score(needle, rel) if fuzzy else (1.0 if needle.lower() in rel.lower() else 0.0)
            if file_score >= (0.58 if fuzzy else 1.0):
                matches.append(
                    {
                        "path": rel,
                        "kind": "path",
                        "line": None,
                        "char": None,
                        "score": round(file_score, 4),
                        "snippet": rel,
                    }
                )

            try:
                size = file_path.stat().st_size
            except OSError:
                continue
            if size > MAX_SEARCH_FILE_BYTES or not _is_probably_text(file_path):
                continue

            text = file_path.read_text(encoding="utf-8", errors="replace")
            if fuzzy:
                offset = 0
                for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
                    score = _best_fuzzy_score(needle, line)
                    if score >= 0.58:
                        matches.append(
                            {
                                "path": rel,
                                "kind": "content",
                                "line": line_number,
                                "char": offset,
                                "score": round(score, 4),
                                "snippet": _snippet(text, offset, len(line), slice_limit),
                            }
                        )
                    offset += len(line)
            else:
                lower = text.lower()
                lower_needle = needle.lower()
                start = lower.find(lower_needle)
                while start != -1:
                    line_number = text.count("\n", 0, start) + 1
                    matches.append(
                        {
                            "path": rel,
                            "kind": "content",
                            "line": line_number,
                            "char": start,
                            "score": 1.0,
                            "snippet": _snippet(text, start, len(needle), slice_limit),
                        }
                    )
                    if len(matches) >= MAX_SEARCH_RESULTS * 5:
                        break
                    start = lower.find(lower_needle, start + max(1, len(lower_needle)))

        matches.sort(key=lambda item: (-float(item["score"]), item["path"], item["line"] or 0, item["char"] or 0))
        page_size = max(1, min(count, MAX_SEARCH_RESULTS))
        page_number = max(1, page)
        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size
        return _ok(
            query=needle,
            root=display_agents_path(root),
            fuzzy=fuzzy,
            page=page_number,
            count=page_size,
            total_matches=len(matches),
            has_more=end_index < len(matches),
            matches=matches[start_index:end_index],
        )
    except Exception as error:
        return _error(str(error), query=query, path=path)


def append_worklog_record(markdown: str, run_id: str | None = None) -> dict:
    ensure_runtime_dirs()
    today = datetime.now().date().isoformat()
    active_run_id = normalize_run_id(run_id) or get_current_run_id()
    root = WORKLOG_DIR / "runs" / active_run_id if active_run_id else WORKLOG_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{today}-record.md"
    header = f"\n\n## {datetime.now().isoformat(timespec='seconds')}\n\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header)
        f.write(markdown.strip())
        f.write("\n")
    return _ok(path=display_path(path), run_id=active_run_id)


def run_python_code_impl(code: str, timeout_seconds: int = 30) -> dict:
    try:
        ensure_runtime_dirs()
        timeout = max(1, min(timeout_seconds, 120))
        script_path = TMP_DIR / f"agent-code-{uuid.uuid4().hex}.py"
        script_path.write_text(code, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=AGENTS_V2_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return _ok(
            exit_code=completed.returncode,
            stdout=completed.stdout[-12000:],
            stderr=completed.stderr[-12000:],
            script=display_agents_path(script_path),
        )
    except subprocess.TimeoutExpired as error:
        return _error(
            "python code timed out",
            exit_code=None,
            stdout=(error.stdout or "")[-12000:] if isinstance(error.stdout, str) else "",
            stderr="Timed out",
        )
    except Exception as error:
        return _error(str(error))


@function_tool
def list_local_files(path: str = ".", pattern: str = "*", max_items: int = 80) -> dict:
    """List files under agents-v2. Path like data or ./data."""
    return list_local_files_impl(path=path, pattern=pattern, max_items=max_items)


@function_tool
def read_local_file(
    path: str,
    unit: ReadUnit = "chars",
    seek: int = 0,
    count: int = DEFAULT_READ_CHARS,
    encoding: str = "utf-8",
) -> dict:
    """Read text under agents-v2. unit chars or lines. seek says where to start."""
    return read_local_file_impl(path=path, unit=unit, seek=seek, count=count, encoding=encoding)


@function_tool
def search_local_files(
    query: str,
    path: str = ".",
    fuzzy: bool = False,
    count: int = 20,
    page: int = 1,
    slice_chars: int = 80,
    pattern: str = "*",
) -> dict:
    """Search files under agents-v2. Returns paths, lines, and small snippets."""
    return search_local_files_impl(
        query=query,
        path=path,
        fuzzy=fuzzy,
        count=count,
        page=page,
        slice_chars=slice_chars,
        pattern=pattern,
    )


@function_tool
def write_local_file(
    path: str,
    content: str,
    mode: WriteMode = "overwrite",
    seek: int = 0,
    length: int = 0,
    create_dirs: bool = True,
    encoding: str = "utf-8",
) -> dict:
    """Write text under agents-v2. Modes: overwrite append prepend insert replace."""
    return write_local_file_impl(
        path=path,
        content=content,
        mode=mode,
        seek=seek,
        length=length,
        create_dirs=create_dirs,
        encoding=encoding,
    )


@function_tool
def append_worklog(markdown: str) -> dict:
    """Add a short markdown note to worklog/YYYY-MM-DD-record.md."""
    return append_worklog_record(markdown)


@function_tool
def run_python_code(code: str, timeout_seconds: int = 30) -> dict:
    """Run small Python code in agents-v2. Returns stdout and stderr."""
    return run_python_code_impl(code=code, timeout_seconds=timeout_seconds)
