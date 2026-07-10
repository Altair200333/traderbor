"""Download full Deribit DVOL volatility-index history (BTC + ETH) at 1h.

Source: public/get_volatility_index_data (no API key, no auth).
  https://www.deribit.com/api/v2/public/get_volatility_index_data
  params: currency, start_timestamp(ms), end_timestamp(ms), resolution(s)

The endpoint returns up to 1000 points *ending* at end_timestamp and pages
backward via the `continuation` cursor (a ms timestamp to pass as the next
end_timestamp). DVOL history begins 2021-03-24 for both currencies.

Row schema returned by Deribit: [ts_ms, open, high, low, close] where the
value is the DVOL index level in annualized vol *percentage points* (e.g.
39.53 == 39.53% annualized). Output columns: ts, open, high, low, close.

Resumable: if the parquet exists we only fetch the recent tail (page back
from now until we reach the stored max ts), then merge/dedupe. Delete the
file to force a full re-download. ~0.2s sleep between requests; 429 backoff.

Usage:
  python download_dvol.py                     # BTC+ETH, 1h, full/resume
  python download_dvol.py --currencies BTC --resolution 3600
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "research" / "data" / "options"
BASE = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
FLOOR_MS = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
SLEEP = 0.2
COLS = ["ts", "open", "high", "low", "close"]


def _fetch(currency: str, start_ms: int, end_ms: int, resolution: int) -> dict:
    url = (f"{BASE}?currency={currency}&start_timestamp={start_ms}"
           f"&end_timestamp={end_ms}&resolution={resolution}")
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "traderbor-research/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited -> exponential backoff
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed {currency} {start_ms}-{end_ms}")


def _page_back(currency: str, resolution: int, floor_ms: int, stop_at_ms: int | None) -> list[list]:
    """Page backward from now to floor_ms. Stop early once we reach stop_at_ms
    (the max ts already stored) so resumes only fetch the new tail."""
    rows: list[list] = []
    end = int(time.time() * 1000)
    seen: set[int] = set()
    while end > floor_ms:
        resp = _fetch(currency, floor_ms, end, resolution)
        res = resp.get("result", {})
        data = res.get("data", [])
        if not data:
            break
        new = [row for row in data if row[0] not in seen]
        for row in new:
            seen.add(row[0])
        rows.extend(new)
        oldest = min(r[0] for r in data)
        cont = res.get("continuation")
        # resume short-circuit: we've reached data we already have
        if stop_at_ms is not None and oldest <= stop_at_ms:
            break
        if not cont or cont >= end:
            end = oldest - 1
        else:
            end = cont
        time.sleep(SLEEP)
    return rows


def download(currency: str, resolution: int) -> dict:
    label = {3600: "1h", 86400: "1d"}.get(resolution, str(resolution))
    out = OUT_DIR / f"dvol_{currency.lower()}_{label}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = None
    stop_at = None
    if out.exists():
        existing = pd.read_parquet(out)
        if len(existing):
            stop_at = int(existing["ts"].max())

    rows = _page_back(currency, resolution, FLOOR_MS, stop_at)
    df = pd.DataFrame(rows, columns=COLS)
    if existing is not None and len(existing):
        df = pd.concat([existing, df], ignore_index=True)
    if not len(df):
        return {"currency": currency, "resolution": label, "rows": 0, "status": "no_data"}
    df["ts"] = df["ts"].astype("int64")
    for c in COLS[1:]:
        df[c] = df[c].astype("float64")
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_parquet(out, index=False)

    step = resolution * 1000
    expected = (int(df["ts"].iloc[-1]) - int(df["ts"].iloc[0])) // step + 1
    return {
        "currency": currency,
        "resolution": label,
        "file": str(out),
        "rows": int(len(df)),
        "first": pd.Timestamp(int(df["ts"].iloc[0]), unit="ms").isoformat(),
        "last": pd.Timestamp(int(df["ts"].iloc[-1]), unit="ms").isoformat(),
        "missing_bars": int(expected - len(df)),
        "status": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currencies", default="BTC,ETH")
    ap.add_argument("--resolution", type=int, default=3600, help="seconds: 3600=1h, 86400=1d")
    args = ap.parse_args()
    for cur in args.currencies.split(","):
        cur = cur.strip().upper()
        res = download(cur, args.resolution)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
