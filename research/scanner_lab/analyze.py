"""Post-training analysis of walk-forward predictions.

Reads research/data/predictions.parquet (from train_eval.py), writes a markdown
report to research/data/analysis_report.md and prints it.

Usage: python analyze.py [--model lgbm] [--cost-bps 25]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT  # noqa: E402

PRED = REPO_ROOT / "research" / "data" / "predictions.parquet"
OUT = REPO_ROOT / "research" / "data" / "analysis_report.md"
RETEST_P = 0.4


def _r_cols(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    df = df.copy()
    cost_r = cost_bps / 1e4 / df["d_final"]
    df["rm"] = df["r_mkt_eval"] - cost_r
    df["rr_"] = df["r_ret_eval"] - cost_r * df["retest_filled"]
    return df


def _seltab(df: pd.DataFrame, name: str, buf: io.StringIO) -> None:
    res = df[df["outcome"].isin(["tp", "sl"])]
    tp = (res["outcome"] == "tp").mean() if len(res) else np.nan
    fill = df["retest_filled"].mean() if len(df) else np.nan
    buf.write(f"| {name} | {len(df)} | {tp:.3f} | {df['rm'].sum():+.1f} | "
              f"{df['rm'].mean():+.3f} | {df['rr_'].sum():+.1f} | {fill:.2f} |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--cost-bps", type=float, default=25.0)
    args = ap.parse_args()

    df = pd.read_parquet(PRED)
    df = df[df["model"] == args.model].copy()
    df = _r_cols(df, args.cost_bps)
    df["ev"] = df["p_hat"] * df["tp_rr"] - (1 - df["p_hat"]) \
        - args.cost_bps / 1e4 / df["d_final"]
    buf = io.StringIO()
    w = buf.write
    w(f"# Scanner v2 analysis ({args.model}, cost {args.cost_bps}bps RT)\n\n")
    w(f"OOS events: {len(df)}, months {df['month'].min()}..{df['month'].max()}\n\n")

    # 1. calibration by p_hat decile
    w("## Calibration (resolved events, all folds pooled)\n\n")
    res = df[df["outcome"].isin(["tp", "sl"])].copy()
    res["dec"] = pd.qcut(res["p_hat"], 10, duplicates="drop")
    cal = res.groupby("dec", observed=True).agg(
        n=("p_hat", "size"), p_mean=("p_hat", "mean"),
        tp_rate=("outcome", lambda s: (s == "tp").mean()))
    w("| decile | n | p_hat | realized tp | \n|---|---|---|---|\n")
    for iv, r in cal.iterrows():
        w(f"| {iv} | {r['n']:.0f} | {r['p_mean']:.3f} | {r['tp_rate']:.3f} |\n")

    # 2. money by decile (all outcomes)
    w("\n## Money by p_hat decile (all outcomes, per event)\n\n")
    df["dec"] = pd.qcut(df["p_hat"], 10, duplicates="drop", labels=False)
    dec = df.groupby("dec").agg(n=("rm", "size"), avg_rm=("rm", "mean"),
                                avg_rr=("rr_", "mean"), avg_ev=("ev", "mean"))
    w("| dec | n | avg R mkt | avg R retest | model EV |\n|---|---|---|---|---|\n")
    for d, r in dec.iterrows():
        w(f"| {d} | {r['n']:.0f} | {r['avg_rm']:+.3f} | {r['avg_rr']:+.3f} | {r['avg_ev']:+.3f} |\n")

    # 3. policy comparison
    w("\n## Policies (whole OOS)\n\n")
    w("| policy | n | tp_rate | R_mkt | avg | R_retest | fill_rate |\n"
      "|---|---|---|---|---|---|---|\n")
    _seltab(df, "all pool", buf)
    _seltab(df[df["is_hard"]], "hard (prod)", buf)
    _seltab(df[df["is_hard"] | df["is_marginal"]], "hard+marginal (prod)", buf)
    sel = df[df["ev"] > 0]
    _seltab(sel, "ML EV>0", buf)
    for thr in (0.30, 0.35, 0.40):
        _seltab(df[df["p_hat"] >= thr], f"ML p>={thr}", buf)
    # flow-matched top-N per month
    nflow = df.groupby("month").apply(
        lambda g: g[g["is_hard"] | g["is_marginal"]].shape[0]).to_dict()
    topn = pd.concat([g.nlargest(nflow.get(m, 0), "p_hat")
                      for m, g in df.groupby("month")])
    _seltab(topn, "ML top-N (flow-matched)", buf)
    # symbol-capped variant of EV>0: max 25% of monthly picks per symbol
    capped = []
    for m, g in sel.groupby("month"):
        cap = max(int(0.25 * len(g)), 2)
        capped.append(g.sort_values("p_hat", ascending=False).groupby("symbol").head(cap))
    if capped:
        _seltab(pd.concat(capped), "ML EV>0 + symbol cap 25%", buf)

    # 4. monthly detail for ML EV>0
    w("\n## ML EV>0 by month\n\n| month | n | R_mkt | R_retest | top symbol (share) |\n|---|---|---|---|---|\n")
    for m, g in sel.groupby("month"):
        ts = g["symbol"].value_counts()
        w(f"| {m} | {len(g)} | {g['rm'].sum():+.1f} | {g['rr_'].sum():+.1f} | "
          f"{ts.index[0]} ({ts.iloc[0] / len(g):.0%}) |\n")
    cum = sel.sort_values("as_of")["rm"].cumsum()
    mdd = (cum - cum.cummax()).min() if len(cum) else np.nan
    w(f"\ncumulative R_mkt max drawdown: {mdd:+.1f}R\n")

    # 5. slices
    w("\n## Slices of ML EV>0\n\n| slice | n | tp_rate | R_mkt | R_retest |\n|---|---|---|---|---|\n")
    for name, mask in [
        ("long", sel["side"] == "long"), ("short", sel["side"] == "short"),
        ("T1", sel["tier"] == 1), ("T2", sel["tier"] == 2), ("T3", sel["tier"] == 3),
        ("btc>ema50", sel["btc_above_ema50"] == 1), ("btc<ema50", sel["btc_above_ema50"] == 0),
        ("P1", sel["pattern"] == "P1"), ("P1H", sel["pattern"] == "P1H"),
        ("P2", sel["pattern"] == "P2"), ("P3", sel["pattern"] == "P3"),
        ("was hard/marg", sel["is_hard"] | sel["is_marginal"]),
        ("pool-only find", ~(sel["is_hard"] | sel["is_marginal"])),
    ]:
        g = sel[mask]
        res_g = g[g["outcome"].isin(["tp", "sl"])]
        tp = (res_g["outcome"] == "tp").mean() if len(res_g) else np.nan
        w(f"| {name} | {len(g)} | {tp if tp == tp else float('nan'):.3f} | "
          f"{g['rm'].sum():+.1f} | {g['rr_'].sum():+.1f} |\n")

    # 6. cost sensitivity (EV recomputed per cost)
    w("\n## Cost sensitivity (ML EV>0 re-selected per cost)\n\n"
      "| cost bps RT | n | R_mkt | R_retest |\n|---|---|---|---|\n")
    for cb in (0, 10, 25, 50):
        d2 = _r_cols(df, cb)
        d2["ev"] = d2["p_hat"] * d2["tp_rr"] - (1 - d2["p_hat"]) - cb / 1e4 / d2["d_final"]
        s2 = d2[d2["ev"] > 0]
        w(f"| {cb} | {len(s2)} | {s2['rm'].sum():+.1f} | {s2['rr_'].sum():+.1f} |\n")

    text = buf.getvalue()
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
