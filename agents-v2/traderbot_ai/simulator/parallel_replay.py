"""Two-phase parallel candidate replay.

Phase 1: a hold-provider deterministic replay over the window finds every
candidate bar (hold mode has no entries, so scans see no position/cooldown
feedback). Phase 2: every candidate bar gets an ISOLATED codex session - its
own $budget wallet and exchange state, the decision provider is called on the
first bar only, then runner-owned settlement steps until the wallet is flat
(pending-limit TTL and the 24h max-hold bound every session). Sessions run as
subprocesses with bounded concurrency: env isolation covers the simulation
clock and deterministic-candidates variables that are process-global.

The aggregate is per-opportunity PnL: sum over sessions of
(final equity - budget). It is NOT a portfolio equity curve - sessions do not
share balance, position slots, or cooldowns, so an entry at bar T no longer
suppresses candidates at later bars the way a sequential replay would.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traderbot_ai.simulator.exchange_replay import DATA_DIR
from traderbot_ai.tools.market import parse_time_ms

ONE_HOUR_MS = 3_600_000
AGENTS_V2_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ORDER_STATUSES = {"new", "partiallyfilled"}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_candidate_bars(replay_path: Path) -> list[int]:
    """Candidate-bar timestamps (as_of_ms) from a replay JSONL scan log."""
    bars: list[int] = []
    with open(replay_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "step_completed":
                continue
            payload = record.get("payload") or {}
            scan = payload.get("scan") or {}
            if scan.get("candidates"):
                bars.append(int(payload["as_of_ms"]))
    return bars


def build_session_command(
    python: str,
    *,
    symbols: str,
    start_ms: int,
    end_ms: int,
    execution_interval: str,
    balance_usdt: float,
    fee_rate: float,
    scanner_provider: str | None,
    session_id: str,
    out_dir: Path,
    decision_provider: str = "codex-cli-mcp",
    first_bar_only: bool = True,
    codex_model: str | None = None,
    codex_reasoning_effort: str | None = None,
    codex_timeout_sec: int = 1800,
    codex_output_dir: Path | None = None,
    codex_fail_open: str = "error",
) -> list[str]:
    command = [
        python, "-m", "traderbot_ai.cli", "exchange-replay",
        "--symbols", symbols,
        "--start-time", _iso(start_ms),
        "--end-time", _iso(end_ms),
        "--decision-interval", "1h",
        "--execution-interval", execution_interval,
        "--balance-usdt", str(balance_usdt),
        "--fee-rate", str(fee_rate),
        "--screener-mode", "deterministic",
        "--decision-provider", decision_provider,
        "--no-preload",
        "--run-id", session_id,
        "--state-path", str(out_dir / f"{session_id}.exchange.json"),
        "--events-path", str(out_dir / f"{session_id}.exchange.events.jsonl"),
        "--replay-path", str(out_dir / f"{session_id}.replay.jsonl"),
    ]
    if first_bar_only:
        command += ["--decide-first-bar-only", "--end-when-flat"]
    if scanner_provider:
        command += ["--scanner-provider", scanner_provider]
    if decision_provider == "codex-cli-mcp":
        command += ["--session", f"{session_id}-codex",
                    "--codex-timeout-sec", str(codex_timeout_sec),
                    "--codex-fail-open", codex_fail_open]
        if codex_model:
            command += ["--codex-model", codex_model]
        if codex_reasoning_effort:
            command += ["--codex-reasoning-effort", codex_reasoning_effort]
        if codex_output_dir is not None:
            command += ["--codex-output-dir", str(codex_output_dir)]
    return command


def _session_env(out_dir: Path, session_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TRADERBOT_SIMULATION_CLOCK_PATH"] = str(out_dir / f"{session_id}.clock.json")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(AGENTS_V2_ROOT) + (os.pathsep + existing if existing else "")
    return env


def summarize_session(replay_path: Path, state_path: Path, budget_usdt: float) -> dict[str, Any]:
    """Per-session outcome: first-bar decision, realized PnL vs budget, flatness."""
    steps = 0
    first: dict[str, Any] = {"candidates": [], "decision": None, "decision_symbol": None}
    final_equity: float | None = None
    with open(replay_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            payload = record.get("payload") or {}
            if record.get("type") == "step_completed":
                steps += 1
                if steps == 1:
                    decision = payload.get("decision") or {}
                    first = {
                        "candidates": (payload.get("scan") or {}).get("candidates") or [],
                        "decision": decision.get("final_decision"),
                        "decision_symbol": decision.get("symbol"),
                    }
            elif record.get("type") == "replay_completed":
                totals = (payload.get("final_wallet") or {}).get("totals") or {}
                final_equity = totals.get("equity_usdt")
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    open_positions = state.get("positions") or []
    active_orders = [order for order in (state.get("orders") or [])
                     if str(order.get("status", "")).lower() in ACTIVE_ORDER_STATUSES]
    closed = state.get("closed_positions") or []
    close_reasons = sorted({str(item.get("exit_reason") or item.get("close_reason") or "?") for item in closed})
    return {
        "steps": steps,
        **first,
        "closed_positions": len(closed),
        "close_reasons": close_reasons,
        "flat": not open_positions and not active_orders,
        "final_equity_usdt": final_equity,
        "pnl_usdt": None if final_equity is None else round(float(final_equity) - float(budget_usdt), 4),
    }


def run_parallel_candidate_replay(
    *,
    symbols: str,
    start_time: str | int | float,
    end_time: str | int | float,
    execution_interval: str = "1m",
    balance_usdt: float = 1000.0,
    fee_rate: float = 0.0,
    scanner_provider: str | None = None,
    concurrency: int = 5,
    session_horizon_hours: int = 30,
    run_id: str | None = None,
    out_dir: Path | None = None,
    phase1_replay: Path | None = None,
    codex_model: str | None = None,
    codex_reasoning_effort: str | None = None,
    codex_timeout_sec: int = 1800,
    codex_output_dir: Path | None = None,
    codex_fail_open: str = "error",
) -> dict[str, Any]:
    start_ms = parse_time_ms(start_time)
    end_ms = parse_time_ms(end_time)
    if start_ms is None or end_ms is None:
        raise ValueError("start_time and end_time are required")
    base_run_id = run_id or f"par-{start_ms}-{end_ms}"
    # session subprocesses run with cwd=AGENTS_V2_ROOT: every path they receive must be absolute
    out = (Path(out_dir) if out_dir is not None else DATA_DIR / "parallel" / base_run_id).resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    cwd = str(AGENTS_V2_ROOT)
    codex_out = codex_output_dir if codex_output_dir is not None else out / "codex"

    if phase1_replay is None:
        phase1_replay = out / "phase1.replay.jsonl"
        phase1_command = build_session_command(
            python,
            symbols=symbols,
            start_ms=start_ms,
            end_ms=end_ms,
            execution_interval=execution_interval,
            balance_usdt=balance_usdt,
            fee_rate=fee_rate,
            scanner_provider=scanner_provider,
            session_id="phase1",
            out_dir=out,
            decision_provider="hold",
            first_bar_only=False,
        )
        phase1 = subprocess.run(
            phase1_command, cwd=cwd, env=_session_env(out, "phase1"),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if phase1.returncode != 0:
            return {"ok": False, "error": "phase1 candidate-scan replay failed",
                    "stderr": (phase1.stderr or phase1.stdout or "")[-2000:]}
    bars = extract_candidate_bars(Path(phase1_replay))

    def run_one(as_of_ms: int) -> dict[str, Any]:
        session_id = f"{base_run_id}-{as_of_ms}"
        command = build_session_command(
            python,
            symbols=symbols,
            start_ms=as_of_ms,
            end_ms=as_of_ms + session_horizon_hours * ONE_HOUR_MS,
            execution_interval=execution_interval,
            balance_usdt=balance_usdt,
            fee_rate=fee_rate,
            scanner_provider=scanner_provider,
            session_id=session_id,
            out_dir=out,
            codex_model=codex_model,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_timeout_sec=codex_timeout_sec,
            codex_output_dir=codex_out,
            codex_fail_open=codex_fail_open,
        )
        base = {"as_of_ms": as_of_ms, "as_of_iso": _iso(as_of_ms), "session_id": session_id}
        try:
            proc = subprocess.run(
                command, cwd=cwd, env=_session_env(out, session_id),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=codex_timeout_sec + 900,
            )
        except subprocess.TimeoutExpired:
            return {**base, "ok": False, "error": "session subprocess timed out"}
        if proc.returncode != 0:
            return {**base, "ok": False, "error": (proc.stderr or proc.stdout or "")[-1500:]}
        summary = summarize_session(out / f"{session_id}.replay.jsonl", out / f"{session_id}.exchange.json", balance_usdt)
        if summary.get("pnl_usdt") is None:
            return {**base, "ok": False, "error": "session finished without a final wallet record", **summary}
        return {**base, "ok": True, **summary}

    sessions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        futures = [pool.submit(run_one, as_of_ms) for as_of_ms in bars]
        for future in as_completed(futures):
            sessions.append(future.result())
    sessions.sort(key=lambda row: row["as_of_ms"])

    ok_rows = [row for row in sessions if row.get("ok")]
    failed = [row for row in sessions if not row.get("ok")]
    not_flat = [row["session_id"] for row in ok_rows if not row.get("flat")]
    summary = {
        "ok": not failed,
        "run_id": base_run_id,
        "out_dir": str(out),
        "phase1_replay": str(phase1_replay),
        "candidate_bars": len(bars),
        "sessions_ok": len(ok_rows),
        "sessions_failed": len(failed),
        "sessions_not_flat": not_flat,
        "budget_per_session_usdt": balance_usdt,
        "total_pnl_usdt": round(sum(float(row["pnl_usdt"]) for row in ok_rows), 2),
        "aggregation_note": "per-opportunity PnL: isolated $budget per session, no shared slots/cooldowns/compounding across bars",
        "sessions": sessions,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
