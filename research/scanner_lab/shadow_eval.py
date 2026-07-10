"""Shadow evaluation of the FROZEN Phase D forward champion on pristine data.

The champion is frozen BY RECIPE, identical to the Phase D forward opening
(harness_v3.open_vault_eval): LGBM on the pruned-hardened feature list, trained
on resolved dev rows with as_of < forward-vault start minus 24h embargo, both
vaults excluded. No refit, ever — the training fingerprint is written to the
artifact so any drift across dataset rebuilds is visible.

Modes:
  --parity          score the burnt forward-vault window and compare p_hat /
                    EV>0 selection against the saved Phase D predictions
                    (plumbing check only — the vault stays burnt evidence-wise)
  (default)         score pristine events (as_of >= --test-from, label-safe)
                    and report market vs retest-limit streams vs take-all
                    control at 25/10 bps, plus a slot-3 capacity sim

Usage:
  python shadow_eval.py --dataset research/data/events_v3a2.parquet --parity \
      --parity-preds <phase_d_forward dir or preds parquet>
  python shadow_eval.py --dataset research/data/events_v3a2_shadow.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval as te  # noqa: E402
from harness_v3 import (  # noqa: E402
    EMBARGO_MS, add_hardened_features, _fit_fold, btc_month_regime,
    in_vault_or_embargo, run_slots, slot_report, vault_bounds,
)
from universe import REPO_ROOT  # noqa: E402

ART_ROOT = REPO_ROOT / "research" / "data" / "v3" / "artifacts"
FEATURES_DEFAULT = (ART_ROOT / "20260707T184423Z_a3_oi_verdict"
                    / "pruned-hardened.features.txt")
HORIZON_MS = 24 * 3_600_000


def frozen_train_mask(ev: pd.DataFrame) -> pd.Series:
    """EXACT Phase D forward training set: resolved, pre-vault minus embargo,
    both vaults + embargoes excluded."""
    v = next(x for x in vault_bounds() if x["name"] == "forward")
    resolved = ev["outcome"].isin(["tp", "sl"])
    not_vault = ~in_vault_or_embargo(ev["as_of"].to_numpy())
    return (ev["as_of"] < v["vs"] - EMBARGO_MS) & resolved & not_vault


def train_fingerprint(train: pd.DataFrame) -> dict:
    keys = sorted(f"{s}:{a}" for s, a in zip(train["symbol"], train["as_of"]))
    sha = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    return {"n_train": int(len(train)),
            "train_min_as_of": pd.Timestamp(int(train["as_of"].min()), unit="ms",
                                            tz="UTC").isoformat(),
            "train_max_as_of": pd.Timestamp(int(train["as_of"].max()), unit="ms",
                                            tz="UTC").isoformat(),
            "train_keys_sha256": sha}


def add_ev(test: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    test = test.copy()
    test["ev_val"] = (test["p_hat"] * test["tp_rr"] - (1 - test["p_hat"])
                      - cost_bps / 1e4 / test["d_final"])
    test["selected_ev"] = test["ev_val"] > 0
    return test


def _dedup_stream(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day"] = pd.to_datetime(df["as_of"], unit="ms", utc=True).dt.date
    return (df.sort_values("as_of").groupby(["symbol", "day"], as_index=False)
            .head(1).sort_values("as_of"))


def stream_report(test: pd.DataFrame, cost_bps: float, entry: str,
                  selected_only: bool = True) -> dict:
    """Per-event + slot-3 money for one policy.

    entry='market': every event trades at next-open (r_mkt_eval).
    entry='retest': only retest_filled rows trade (r_ret_eval); unfilled = skip.
    Cost applied identically to both streams (cb/1e4/d_final), matching the
    Phase D addendum accounting.
    """
    sel = test[test["selected_ev"]] if selected_only else test
    if entry == "retest":
        traded = sel[sel["retest_filled"].fillna(False).astype(bool)].copy()
        traded["rm"] = traded["r_ret_eval"] - cost_bps / 1e4 / traded["d_final"]
    else:
        traded = sel.copy()
        traded["rm"] = traded["r_mkt_eval"] - cost_bps / 1e4 / traded["d_final"]
    res = traded[traded["outcome"].isin(["tp", "sl"])]
    out = {"entry": entry, "cost_bps": cost_bps,
           "n_selected": int(len(sel)), "n_traded": int(len(traded)),
           "fill_rate": round(len(traded) / len(sel), 4) if len(sel) else None,
           "n_resolved": int(len(res)),
           "precision": round(float((res["outcome"] == "tp").mean()), 4)
           if len(res) else None,
           "sum_R": round(float(traded["rm"].sum()), 3),
           "avg_R": round(float(traded["rm"].mean()), 4) if len(traded) else None}
    if len(traded):
        stream = _dedup_stream(traded)
        if "t_exit_min" not in stream.columns:
            stream["t_exit_min"] = np.nan
        taken = run_slots(stream, 3)
        out["slot3"] = slot_report(taken, f"{entry} slots=3")
    return out


def run_parity(test: pd.DataFrame, preds_path: Path) -> dict:
    """Compare freshly scored vault window against saved Phase D predictions."""
    saved = pd.read_parquet(preds_path)
    m = test.merge(saved[["symbol", "as_of", "p_hat", "selected_ev"]],
                   on=["symbol", "as_of"], suffixes=("", "_saved"))
    agree = (m["selected_ev"] == m["selected_ev_saved"]).mean()
    corr = float(np.corrcoef(m["p_hat"], m["p_hat_saved"])[0, 1])
    mad = float((m["p_hat"] - m["p_hat_saved"]).abs().max())
    return {"n_test": int(len(test)), "n_saved": int(len(saved)),
            "n_joined": int(len(m)), "p_hat_corr": round(corr, 6),
            "p_hat_max_abs_diff": round(mad, 6),
            "selection_agreement": round(float(agree), 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--features-file", default=str(FEATURES_DEFAULT))
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--test-from", default="2026-07-06T01:00",
                    help="pristine window start (UTC); default = first hour "
                         "strictly after the burnt forward vault + its edge")
    ap.add_argument("--parity", action="store_true",
                    help="score the burnt forward vault window and diff "
                         "against --parity-preds instead of pristine eval")
    ap.add_argument("--parity-preds", default="",
                    help="saved Phase D forward preds parquet (or dir)")
    ap.add_argument("--out-dir", default="",
                    help="artifact dir root (default research/data/v3/artifacts)")
    args = ap.parse_args()

    feats = [ln.strip() for ln in Path(args.features_file).read_text().splitlines()
             if ln.strip()]
    # prep() ordering (symbol-major) MUST be preserved: fit_predict's val slice
    # for early stopping + Platt calibration is positional (train.iloc[-n_val:]),
    # so row order is part of the frozen Phase D recipe. Do not re-sort.
    ev = te.prep(pd.read_parquet(args.dataset))
    ev, _ = add_hardened_features(ev)
    missing = [f for f in feats if f not in ev.columns]
    if missing:
        raise SystemExit(f"dataset lacks frozen features: {missing}")

    tr_mask = frozen_train_mask(ev)
    train = ev[tr_mask]
    fp = train_fingerprint(train)
    print(f"frozen train: {fp['n_train']} rows "
          f"({fp['train_min_as_of']} .. {fp['train_max_as_of']})")

    v = next(x for x in vault_bounds() if x["name"] == "forward")
    if args.parity:
        test = ev[(ev["as_of"] >= v["vs"]) & (ev["as_of"] < v["ve"])].copy()
        label = "parity"
    else:
        t_from = int(pd.Timestamp(args.test_from, tz="UTC").timestamp() * 1000)
        if t_from < v["ve"]:
            raise SystemExit("--test-from must be after the burnt forward vault")
        # label-safe: the event needs its full 24h outcome window inside data
        data_end = int(ev["as_of"].max())
        t_to = data_end - HORIZON_MS
        test = ev[(ev["as_of"] >= t_from) & (ev["as_of"] <= t_to)].copy()
        label = "pristine"
        print(f"pristine window: {pd.Timestamp(t_from, unit='ms', tz='UTC')} .. "
              f"{pd.Timestamp(t_to, unit='ms', tz='UTC')} ({len(test)} events)")

    if not len(test):
        raise SystemExit("no test events in window")

    test["p_hat"] = _fit_fold(train, test, feats, args.model)
    regime_map, _ = btc_month_regime()
    test["month"] = pd.to_datetime(test["as_of"], unit="ms", utc=True).dt.strftime("%Y-%m")
    test["regime"] = test["month"].map(regime_map).fillna("?")
    test = add_ev(test, args.cost_bps)

    out: dict = {"mode": label, "dataset": args.dataset, "model": args.model,
                 "features_file": args.features_file, "n_features": len(feats),
                 "run_utc": datetime.now(timezone.utc).isoformat(),
                 "train_fingerprint": fp}

    if args.parity:
        pp = Path(args.parity_preds)
        if pp.is_dir():
            cands = sorted(pp.glob("*.preds.parquet")) or sorted(pp.glob("*.parquet"))
            pp = cands[0]
        out["parity"] = run_parity(test, pp)
        print(json.dumps(out["parity"], indent=2))
    else:
        reports = []
        for cb in (args.cost_bps, 10.0):
            t_cb = add_ev(test, cb)
            reports.append(stream_report(t_cb, cb, "market"))
            reports.append(stream_report(t_cb, cb, "retest"))
            reports.append({**stream_report(t_cb, cb, "market", selected_only=False),
                            "entry": "control_take_all_market"})
        out["streams"] = reports
        out["n_events"] = int(len(test))
        out["n_selected_ev"] = int(test["selected_ev"].sum())
        for r in reports:
            print(f"{r['entry']:>28s} @{r['cost_bps']:>4.0f}bps: "
                  f"traded={r['n_traded']:>4d} sum_R={r['sum_R']:>8.3f} "
                  f"avg_R={r['avg_R']} precision={r['precision']} "
                  f"fill={r.get('fill_rate')}")

    root = Path(args.out_dir) if args.out_dir else ART_ROOT
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    art = root / f"{ts}_shadow_eval_{label}"
    art.mkdir(parents=True, exist_ok=False)
    (art / "result.json").write_text(json.dumps(out, indent=2, default=str),
                                     encoding="utf-8")
    keep = [c for c in ["as_of", "month", "regime", "symbol", "side", "pattern",
                        "outcome", "p_hat", "tp_rr", "d_final", "r_mkt_eval",
                        "r_ret_eval", "retest_filled", "t_exit_min",
                        "ev_val", "selected_ev", "is_hard", "is_marginal"]
            if c in test.columns]
    test[keep].to_parquet(art / "preds.parquet", index=False)
    print(f"\nartifacts -> {art}")


if __name__ == "__main__":
    main()
