from __future__ import annotations

import json
from contextvars import ContextVar

from traderbot_ai.paths import DATA_DIR, ensure_runtime_dirs
from traderbot_ai.tools.market import parse_time_ms


SIMULATION_CLOCK_PATH = DATA_DIR / "simulation_clock.json"
_PROCESS_CLOCK_MS: ContextVar[int | None] = ContextVar("simulation_clock_ms", default=None)


def set_process_simulation_clock_state(as_of: str | int | float) -> int:
    as_of_ms = _parse_required_as_of(as_of)
    _PROCESS_CLOCK_MS.set(as_of_ms)
    return as_of_ms


def clear_process_simulation_clock_state() -> None:
    _PROCESS_CLOCK_MS.set(None)


def process_simulation_clock_ms() -> int | None:
    return _PROCESS_CLOCK_MS.get()


def set_simulation_clock_state(as_of: str | int | float) -> int:
    as_of_ms = set_process_simulation_clock_state(as_of)
    set_file_simulation_clock_state(as_of_ms)
    return as_of_ms


def set_file_simulation_clock_state(as_of: str | int | float) -> int:
    as_of_ms = _parse_required_as_of(as_of)
    ensure_runtime_dirs()
    SIMULATION_CLOCK_PATH.write_text(json.dumps({"as_of_ms": as_of_ms}, indent=2), encoding="utf-8")
    return as_of_ms


def clear_simulation_clock_state() -> None:
    clear_process_simulation_clock_state()
    clear_file_simulation_clock_state()


def clear_file_simulation_clock_state() -> None:
    if SIMULATION_CLOCK_PATH.exists():
        SIMULATION_CLOCK_PATH.unlink()


def active_simulation_clock_ms() -> int | None:
    process_clock = process_simulation_clock_ms()
    if process_clock is not None:
        return process_clock
    return file_simulation_clock_ms()


def file_simulation_clock_ms() -> int | None:
    if not SIMULATION_CLOCK_PATH.exists():
        return None
    data = json.loads(SIMULATION_CLOCK_PATH.read_text(encoding="utf-8"))
    value = data.get("as_of_ms")
    return int(value) if value is not None else None


def guarded_simulation_as_of(value: str | int | float | None) -> int | None:
    requested = parse_time_ms(value)
    active = active_simulation_clock_ms()
    if active is None:
        return requested
    if requested is None:
        return active
    if requested > active:
        raise ValueError(f"as_of {requested} exceeds simulation clock {active}")
    return requested


def _parse_required_as_of(as_of: str | int | float) -> int:
    as_of_ms = parse_time_ms(as_of)
    if as_of_ms is None:
        raise ValueError("as_of is required")
    return as_of_ms
