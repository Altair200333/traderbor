"""SQLite journal — the bot's source of truth (WAL mode, reconcile-on-start).

Tables (architecture plan component 8): snapshots, signals, orders, executions,
positions, equity, events, liq_prints, daily_volume, kv. All timestamps are
UTC epoch milliseconds (INTEGER). All writes go through this module.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    bar_ms INTEGER NOT NULL,          -- open time of the completed 1h bar
    symbol TEXT NOT NULL,             -- research pair name (e.g. PEPEUSDT)
    close REAL NOT NULL,              -- lastPrice at bar close
    mark REAL,
    oi REAL,                          -- openInterest (coins)
    funding REAL,
    PRIMARY KEY (bar_ms, symbol)
);
CREATE TABLE IF NOT EXISTS daily_volume (
    day_ms INTEGER NOT NULL,          -- UTC day start
    symbol TEXT NOT NULL,
    quote_volume REAL NOT NULL,
    PRIMARY KEY (day_ms, symbol)
);
CREATE TABLE IF NOT EXISTS signals (   -- every trigger, filled or not (deploy-spec s4)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts_ms INTEGER NOT NULL,          -- trigger bar CLOSE time
    ret6 REAL, doi6 REAL, btc_ret6 REAL,
    weight REAL, mw_tag TEXT, session TEXT,
    limit_px REAL,
    approved INTEGER NOT NULL DEFAULT 0,
    veto_reason TEXT,
    UNIQUE (strategy, symbol, ts_ms)
);
CREATE TABLE IF NOT EXISTS orders (
    order_link_id TEXT PRIMARY KEY,
    exchange_order_id TEXT,
    strategy TEXT NOT NULL,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,             -- exchange symbol
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    price REAL,
    qty REAL NOT NULL,
    status TEXT NOT NULL,             -- pending|open|filled|partial_ttl|cancelled|rejected
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    ttl_deadline_ms INTEGER,
    filled_qty REAL NOT NULL DEFAULT 0,
    avg_fill_px REAL,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS executions (
    exec_id TEXT PRIMARY KEY,
    order_link_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    fee REAL,
    ts_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_px REAL NOT NULL,
    entry_ms INTEGER NOT NULL,
    stop_px REAL,
    exit_due_ms INTEGER,
    status TEXT NOT NULL,             -- open|closed
    exit_px REAL, exit_ms INTEGER, exit_reason TEXT,
    pnl_net REAL,
    weight REAL,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts_ms INTEGER PRIMARY KEY,
    equity_usd REAL NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    level TEXT NOT NULL,              -- info|warn|alarm
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT
);
CREATE TABLE IF NOT EXISTS liq_prints (  -- allLiquidation research feed (day-1 gold)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_liq_ts ON liq_prints (ts_ms);
CREATE TABLE IF NOT EXISTS unlock_events (
    slug TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_date TEXT NOT NULL,         -- YYYY-MM-DD (cliff date, UTC)
    frac_supply REAL NOT NULL,
    PRIMARY KEY (slug, event_date)
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def now_ms() -> int:
    return int(time.time() * 1000)


def link_id(strategy: str, symbol: str, ts_ms: int) -> str:
    """Bybit orderLinkId: letters/digits/dash/underscore, <=45 chars."""
    return f"{strategy}-{symbol}-{ts_ms}"[:45]


class Journal:
    def __init__(self, path: Path | str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(p))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ---------------------------------------------------------- generic --
    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        cur = self.db.execute(sql, args)
        self.db.commit()
        return cur

    def event(self, level: str, kind: str, message: str, data: dict | None = None) -> None:
        self._exec("INSERT INTO events (ts_ms, level, kind, message, data) VALUES (?,?,?,?,?)",
                   (now_ms(), level, kind, message, json.dumps(data) if data else None))

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self._exec("INSERT INTO kv (key, value) VALUES (?,?) "
                   "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # -------------------------------------------------------- snapshots --
    def write_snapshots(self, bar_ms: int, rows: list[dict]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO snapshots (bar_ms, symbol, close, mark, oi, funding) "
            "VALUES (?,?,?,?,?,?)",
            [(bar_ms, r["symbol"], r["close"], r.get("mark"), r.get("oi"),
              r.get("funding")) for r in rows])
        self.db.commit()

    def get_series(self, symbol: str, n_bars: int) -> list[sqlite3.Row]:
        """Last n_bars snapshot rows for symbol, ascending by bar_ms."""
        rows = self.db.execute(
            "SELECT bar_ms, close, oi FROM snapshots WHERE symbol=? "
            "ORDER BY bar_ms DESC LIMIT ?", (symbol, n_bars)).fetchall()
        return list(reversed(rows))

    def last_bar_ms(self) -> int | None:
        row = self.db.execute("SELECT MAX(bar_ms) AS m FROM snapshots").fetchone()
        return row["m"]

    def write_daily_volume(self, symbol: str, rows: list[tuple[int, float]]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO daily_volume (day_ms, symbol, quote_volume) VALUES (?,?,?)",
            [(d, symbol, v) for d, v in rows])
        self.db.commit()

    def median_daily_volume(self, symbol: str, before_day_ms: int, n_days: int = 30) -> float | None:
        """Median quote volume of the last n_days COMPLETED days before before_day_ms."""
        rows = self.db.execute(
            "SELECT quote_volume FROM daily_volume WHERE symbol=? AND day_ms<? "
            "ORDER BY day_ms DESC LIMIT ?", (symbol, before_day_ms, n_days)).fetchall()
        if len(rows) < n_days:
            return None
        vals = sorted(r["quote_volume"] for r in rows)
        n = len(vals)
        mid = n // 2
        return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    # ---------------------------------------------------------- signals --
    def write_signal(self, **kw: Any) -> None:
        self._exec(
            "INSERT OR IGNORE INTO signals (strategy, mode, symbol, ts_ms, ret6, doi6, "
            "btc_ret6, weight, mw_tag, session, limit_px, approved, veto_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (kw["strategy"], kw["mode"], kw["symbol"], kw["ts_ms"], kw.get("ret6"),
             kw.get("doi6"), kw.get("btc_ret6"), kw.get("weight"), kw.get("mw_tag"),
             kw.get("session"), kw.get("limit_px"), int(kw.get("approved", False)),
             kw.get("veto_reason")))

    def set_signal_result(self, strategy: str, symbol: str, ts_ms: int,
                          approved: bool, veto_reason: str | None) -> None:
        self._exec("UPDATE signals SET approved=?, veto_reason=? "
                   "WHERE strategy=? AND symbol=? AND ts_ms=?",
                   (int(approved), veto_reason, strategy, symbol, ts_ms))

    def last_signal_ms(self, strategy: str, symbol: str) -> int | None:
        row = self.db.execute(
            "SELECT MAX(ts_ms) AS m FROM signals WHERE strategy=? AND symbol=?",
            (strategy, symbol)).fetchone()
        return row["m"]

    def recent_signals(self, strategy: str, mode: str, limit: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM signals WHERE strategy=? AND mode=? AND approved=1 "
            "ORDER BY ts_ms DESC LIMIT ?", (strategy, mode, limit)).fetchall()

    # ----------------------------------------------------------- orders --
    def upsert_order(self, **kw: Any) -> None:
        ts = now_ms()
        self._exec(
            "INSERT INTO orders (order_link_id, exchange_order_id, strategy, mode, symbol, "
            "side, order_type, price, qty, status, created_ms, updated_ms, ttl_deadline_ms, "
            "filled_qty, avg_fill_px, meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(order_link_id) DO UPDATE SET exchange_order_id=excluded.exchange_order_id, "
            "status=excluded.status, updated_ms=excluded.updated_ms, price=excluded.price, "
            "filled_qty=excluded.filled_qty, avg_fill_px=excluded.avg_fill_px",
            (kw["order_link_id"], kw.get("exchange_order_id"), kw["strategy"], kw["mode"],
             kw["symbol"], kw["side"], kw.get("order_type", "Limit"), kw.get("price"),
             kw["qty"], kw["status"], kw.get("created_ms", ts), ts,
             kw.get("ttl_deadline_ms"), kw.get("filled_qty", 0.0), kw.get("avg_fill_px"),
             json.dumps(kw["meta"]) if kw.get("meta") else None))

    def set_order_status(self, order_link_id: str, status: str,
                         filled_qty: float | None = None,
                         avg_fill_px: float | None = None) -> None:
        sets, args = ["status=?", "updated_ms=?"], [status, now_ms()]
        if filled_qty is not None:
            sets.append("filled_qty=?")
            args.append(filled_qty)
        if avg_fill_px is not None:
            sets.append("avg_fill_px=?")
            args.append(avg_fill_px)
        args.append(order_link_id)
        self._exec(f"UPDATE orders SET {', '.join(sets)} WHERE order_link_id=?", tuple(args))

    def get_order(self, order_link_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM orders WHERE order_link_id=?",
                               (order_link_id,)).fetchone()

    def open_orders(self, mode: str | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM orders WHERE status IN ('pending','open')"
        args: tuple = ()
        if mode:
            q += " AND mode=?"
            args = (mode,)
        return self.db.execute(q, args).fetchall()

    def write_execution(self, exec_id: str, order_link_id: str | None, symbol: str,
                        side: str, price: float, qty: float, fee: float | None,
                        ts_ms: int) -> bool:
        cur = self._exec(
            "INSERT OR IGNORE INTO executions (exec_id, order_link_id, symbol, side, "
            "price, qty, fee, ts_ms) VALUES (?,?,?,?,?,?,?,?)",
            (exec_id, order_link_id, symbol, side, price, qty, fee, ts_ms))
        return cur.rowcount > 0

    # -------------------------------------------------------- positions --
    def open_position(self, **kw: Any) -> int:
        cur = self._exec(
            "INSERT INTO positions (strategy, mode, symbol, side, qty, entry_px, entry_ms, "
            "stop_px, exit_due_ms, status, weight, meta) VALUES (?,?,?,?,?,?,?,?,?,'open',?,?)",
            (kw["strategy"], kw["mode"], kw["symbol"], kw["side"], kw["qty"], kw["entry_px"],
             kw["entry_ms"], kw.get("stop_px"), kw.get("exit_due_ms"), kw.get("weight"),
             json.dumps(kw["meta"]) if kw.get("meta") else None))
        return int(cur.lastrowid)

    def update_position_qty(self, pos_id: int, qty: float, entry_px: float) -> None:
        self._exec("UPDATE positions SET qty=?, entry_px=? WHERE id=?", (qty, entry_px, pos_id))

    def close_position(self, pos_id: int, exit_px: float, exit_ms: int,
                       exit_reason: str, pnl_net: float | None) -> None:
        self._exec("UPDATE positions SET status='closed', exit_px=?, exit_ms=?, "
                   "exit_reason=?, pnl_net=? WHERE id=?",
                   (exit_px, exit_ms, exit_reason, pnl_net, pos_id))

    def open_positions(self, mode: str | None = None, strategy: str | None = None
                       ) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM positions WHERE status='open'", []
        if mode:
            q += " AND mode=?"
            args.append(mode)
        if strategy:
            q += " AND strategy=?"
            args.append(strategy)
        return self.db.execute(q, tuple(args)).fetchall()

    def closed_trades(self, strategy: str, mode: str, limit: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM positions WHERE strategy=? AND mode=? AND status='closed' "
            "ORDER BY exit_ms DESC LIMIT ?", (strategy, mode, limit)).fetchall()

    # ------------------------------------------------------------ equity --
    def write_equity(self, ts_ms: int, equity_usd: float, source: str) -> None:
        self._exec("INSERT OR REPLACE INTO equity (ts_ms, equity_usd, source) VALUES (?,?,?)",
                   (ts_ms, equity_usd, source))

    def equity_peak(self, source: str) -> float | None:
        row = self.db.execute("SELECT MAX(equity_usd) AS m FROM equity WHERE source=?",
                              (source,)).fetchone()
        return row["m"]

    def last_equity(self, source: str) -> float | None:
        row = self.db.execute("SELECT equity_usd FROM equity WHERE source=? "
                              "ORDER BY ts_ms DESC LIMIT 1", (source,)).fetchone()
        return row["equity_usd"] if row else None

    # -------------------------------------------------------- liq prints --
    def write_liq_prints(self, rows: list[tuple[int, str, str, float, float]]) -> None:
        self.db.executemany(
            "INSERT INTO liq_prints (ts_ms, symbol, side, price, size) VALUES (?,?,?,?,?)", rows)
        self.db.commit()

    # ------------------------------------------------------ unlock events --
    def write_unlock_events(self, rows: list[tuple[str, str, str, float]]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO unlock_events (slug, symbol, event_date, frac_supply) "
            "VALUES (?,?,?,?)", rows)
        self.db.commit()

    def unlock_events(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM unlock_events ORDER BY event_date").fetchall()
