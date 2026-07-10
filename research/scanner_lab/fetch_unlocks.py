"""Fetch historical token-unlock (emissions/vesting) data from DefiLlama's FREE
datasets CDN and map it onto our 149-pair Binance-perp universe.

PHASE A data source (verified 2026-07):
    - Protocol list:  https://defillama-datasets.llama.fi/emissionsProtocolsList
                      -> JSON array of ~339 protocol slugs that have emission schedules.
    - Per protocol:   https://defillama-datasets.llama.fi/emissions/{slug}
                      -> JSON. Relevant fields:
                         metadata.unlockEvents : [ {timestamp,
                                                    cliffAllocations:[{amount,...}],
                                                    linearAllocations:[...],
                                                    summary:{totalTokensCliff}} ]
                         supplyMetrics.maxSupply / metadata.total : denominator
                         gecko_id, name, metadata.token ("chain:address")
    - Symbol map:     https://api.llama.fi/protocols  (FREE, 7799 rows)
                      -> slug -> symbol, gecko_id -> symbol.  e.g. arbitrum -> ARB.

    NOTE: https://api.llama.fi/emissions* is now 402 Payment Required (moved to the
    paid "pro" API). The datasets CDN above is still free and machine-readable and is
    what the public defillama.com/unlocks page consumes.

We treat a CLIFF as one discrete unlock "tranche" (the event-study unit). Linear
(continuous) allocations are recorded for context but are NOT events (any single
day's linear drip is tiny; the Keyrock finding concerns cliff tranches).

Outputs (research/data/unlocks/):
    emissions_meta.json    per-slug slim record + universe match status
    unlock_events.parquet  all cliff events for universe-matched tokens (all sizes)
    unlock_events.csv      same, human-readable
    coverage.json          Phase-A coverage summary
    raw/{slug}.json        slim raw extract per fetched protocol (audit trail)
"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from universe import load_universe  # noqa: E402

OUT = HERE.parents[0] / "data" / "unlocks"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

CDN = "https://defillama-datasets.llama.fi"
UA = {"User-Agent": "Mozilla/5.0 research"}


def getj(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    universe = load_universe()
    uni_syms = {c.symbol.upper() for c in universe}
    print(f"universe: {len(universe)} pairs, {len(uni_syms)} symbols")

    # --- symbol maps from /protocols (free) ---
    protocols = getj("https://api.llama.fi/protocols")
    slug2sym, gecko2sym = {}, {}
    for p in protocols:
        s = p.get("symbol")
        if not s or s == "-":
            continue
        if p.get("slug"):
            slug2sym[p["slug"]] = s.upper()
        if p.get("gecko_id"):
            gecko2sym.setdefault(p["gecko_id"], s.upper())
    print(f"/protocols: {len(protocols)} rows, slug2sym={len(slug2sym)}")

    # CoinGecko coins/list is the AUTHORITATIVE id->symbol map. The emissions payload
    # carries the token's coingecko id in `gecko_id`, so this resolves the real ticker
    # (e.g. ethena -> ENA) where /protocols slug picks the wrong sibling protocol
    # (e.g. Ethena's USDe stablecoin).
    cg = getj("https://api.coingecko.com/api/v3/coins/list")
    id2sym = {c["id"]: c["symbol"].upper() for c in cg if c.get("symbol")}
    print(f"coingecko coins/list: {len(id2sym)} ids")

    slugs = getj(f"{CDN}/emissionsProtocolsList")
    print(f"emissionsProtocolsList: {len(slugs)} slugs")

    # Fetch ALL payloads; mapping is done after we read each payload's gecko_id.
    to_fetch = list(slugs)
    print(f"fetching {len(to_fetch)} payloads")

    def fetch_one(slug: str):
        for attempt in range(2):
            try:
                d = getj(f"{CDN}/emissions/{slug}", timeout=90)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 1:
                    return slug, None, repr(e)[:120]
        meta = d.get("metadata", {}) or {}
        supply = d.get("supplyMetrics", {}) or {}
        max_supply = supply.get("maxSupply") or meta.get("total")
        gecko = d.get("gecko_id")
        sym = (id2sym.get(gecko) if gecko else None) or slug2sym.get(slug) \
            or (gecko2sym.get(gecko) if gecko else None)
        cliffs = []
        for ev in meta.get("unlockEvents", []) or []:
            ts = ev.get("timestamp")
            summ = ev.get("summary", {}) or {}
            tok = summ.get("totalTokensCliff")
            if tok is None:
                tok = sum(float(a.get("amount", 0) or 0)
                          for a in ev.get("cliffAllocations", []) or [])
            if ts and tok and float(tok) > 0:
                cliffs.append({"timestamp": int(ts), "cliff_tokens": float(tok)})
        rec = {
            "slug": slug, "gecko_id": gecko, "name": d.get("name"),
            "token": meta.get("token"), "symbol": sym,
            "max_supply": float(max_supply) if max_supply else None,
            "n_cliff_events": len(cliffs), "cliffs": cliffs,
        }
        return slug, rec, None

    metas, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, s): s for s in to_fetch}
        for i, fut in enumerate(as_completed(futs), 1):
            slug, rec, err = fut.result()
            if err:
                errors.append((slug, err))
                continue
            metas.append(rec)
            (RAW / f"{slug}.json").write_text(json.dumps(rec), encoding="utf-8")
            if i % 25 == 0:
                print(f"  fetched {i}/{len(to_fetch)}")
    print(f"fetched ok={len(metas)} errors={len(errors)}")
    if errors:
        print("  sample errors:", errors[:5])

    # --- match to universe + build events table ---
    events = []
    matched_syms, meta_slim = set(), []
    for r in metas:
        sym = r["symbol"]
        matched = sym in uni_syms
        pair = f"{sym}USDT" if matched else None
        meta_slim.append({k: r[k] for k in
                          ("slug", "gecko_id", "name", "token", "symbol",
                           "max_supply", "n_cliff_events")} | {"matched": matched})
        if not matched or not r["max_supply"]:
            continue
        matched_syms.add(sym)
        ms = r["max_supply"]
        for c in r["cliffs"]:
            dt = datetime.fromtimestamp(c["timestamp"], tz=timezone.utc)
            events.append({
                "symbol": sym, "pair": pair, "slug": r["slug"],
                "gecko_id": r["gecko_id"], "max_supply": ms,
                "event_ts": c["timestamp"],
                "event_date": dt.strftime("%Y-%m-%d"),
                "cliff_tokens": c["cliff_tokens"],
                "frac_supply": c["cliff_tokens"] / ms,
            })

    ev_df = pd.DataFrame(events).sort_values(["symbol", "event_date"]).reset_index(drop=True)
    ev_df.to_parquet(OUT / "unlock_events.parquet", index=False)
    ev_df.to_csv(OUT / "unlock_events.csv", index=False)
    (OUT / "emissions_meta.json").write_text(json.dumps(meta_slim, indent=1), encoding="utf-8")

    unmatched_uni = sorted(uni_syms - matched_syms)
    cov = {
        "emissions_slugs_total": len(slugs),
        "payloads_fetched": len(metas),
        "fetch_errors": len(errors),
        "universe_symbols": len(uni_syms),
        "universe_matched": len(matched_syms),
        "universe_matched_syms": sorted(matched_syms),
        "universe_unmatched_syms": unmatched_uni,
        "total_cliff_events_matched": int(len(ev_df)),
        "events_ge_1pct": int((ev_df["frac_supply"] >= 0.01).sum()) if len(ev_df) else 0,
        "events_ge_3pct": int((ev_df["frac_supply"] >= 0.03).sum()) if len(ev_df) else 0,
        "event_date_span": [ev_df["event_date"].min(), ev_df["event_date"].max()]
        if len(ev_df) else None,
    }
    (OUT / "coverage.json").write_text(json.dumps(cov, indent=1), encoding="utf-8")
    print("COVERAGE:", json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()
