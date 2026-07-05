from __future__ import annotations

from agents import SQLiteSession

from traderbot_ai.paths import DATA_DIR, ensure_runtime_dirs


def get_session(session_id: str) -> SQLiteSession:
    ensure_runtime_dirs()
    return SQLiteSession(session_id=session_id, db_path=DATA_DIR / "sessions.sqlite")
