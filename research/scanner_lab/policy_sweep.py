"""Policy sweep on saved walk-forward predictions: which combination of
hard filters (side/pattern/regime) + ML selection gives stable positive R.

Answers: does ML add anything WITHIN the good slice, or is the edge just
long+P1+regime? Usage: python policy_sweep.py [--model lgbm] [--cost-bps 25]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import REPO_ROOT  # noqa: E402

PRED = REPO_ROOT / "research" / "data" / "predictions.parquet"


def stats(sub: pd.DataFrame, months: int) -> dict:
    if len(sub) == 0:
        return dict(n=0)
    m = sub.groupby("month")["rm"].sum()
    res = sub[sub["outcome"].isin(["tp", "sl"])]
    return dict(
        n=len(sub), per_mo=round(len(sub) / months, 1),
        tp=round((res["outcome"] == "tp").mean(), 3) if len(res) else np.nan,
        R=round(sub["rm"].sum(), 1), avg=round(sub["rm"].mean(), 3),
        R_ret=round(sub["rr_"].sum(), 1),
        mo_pos=f"{(m > 0).sum()}/{len(m)}",
        worst_mo=round(m.min(), 1),
        sharpe_mo=round(m.mean() / m.std(), 2) if len(m) > 2 and m.std() > 0 else np.nan,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--cost-bps", type=float, default=25.0)
    args = ap.parse_args()

    df = pd.read_parquet(PRED)
    df = df[df["model"] == args.model].copy()
    cost_r = args.cost_bps / 1e4 / df["d_final"]
    df["rm"] = df["r_mkt_eval"] - cost_r
    df["rr_"] = df["r_ret_eval"] - cost_r * df["retest_filled"]
    df["ev"] = df["p_hat"] * df["tp_rr"] - (1 - df["p_hat"]) - cost_r
    df["day"] = pd.to_datetime(df["as_of"], unit="ms", utc=True).dt.date
    months = df["month"].nunique()

    long_ = df["side"] == "long"
    p1 = df["pattern"] == "P1"
    bull = df["btc_above_ema50"] == 1
    breadth = df["breadth_ema50"] >= 0.5

    def topk_day(sub: pd.DataFrame, k: int) -> pd.DataFrame:
        return sub.sort_values("p_hat", ascending=False).groupby("day").head(k)

    policies: list[tuple[str, pd.DataFrame]] = [
        ("long only", df[long_]),
        ("long P1", df[long_ & p1]),
        ("long P1 btc>ema50", df[long_ & p1 & bull]),
        ("long P1 breadth>=.5", df[long_ & p1 & breadth]),
        ("long P1 bull (no ML), top3/day by vol_ratio",
         df[long_ & p1 & bull].sort_values("vol_ratio", ascending=False).groupby("day").head(3)),
        ("long P1 bull + EV>0", df[long_ & p1 & bull & (df["ev"] > 0)]),
        ("long P1 bull + p top3/day", topk_day(df[long_ & p1 & bull], 3)),
        ("long P1 bull + p top1/day", topk_day(df[long_ & p1 & bull], 1)),
        ("long P1 + p top3/day", topk_day(df[long_ & p1], 3)),
        ("long + p top3/day", topk_day(df[long_], 3)),
        ("long P1 bull + p BOTTOM3/day", df[long_ & p1 & bull].sort_values("p_hat").groupby("day").head(3)),
        ("long P1 bull breadth + EV>0", df[long_ & p1 & bull & breadth & (df["ev"] > 0)]),
        ("prod hard+marg (ref)", df[df["is_hard"] | df["is_marginal"]]),
        ("prod hard+marg long P1 bull", df[(df["is_hard"] | df["is_marginal"]) & long_ & p1 & bull]),
    ]
    base = df[long_ & p1 & bull & (df["ev"] > 0)]
    # capacity-realistic variants: EV>0 but at most k/day (EV-ranked), symbol<=1/day
    for k in (2, 3, 4):
        capped = (base.sort_values("ev", ascending=False)
                  .groupby(["day", "symbol"]).head(1)
                  .sort_values("ev", ascending=False).groupby("day").head(k))
        policies.append((f"long P1 bull EV>0 cap{k}/day sym1", capped))
    for buf_ in (0.1, 0.2, 0.3):
        policies.append((f"long P1 bull EV>{buf_}", df[long_ & p1 & bull & (df["ev"] > buf_)]))

    rows = []
    for name, sub in policies:
        rows.append({"policy": name, **stats(sub, months)})
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))

    # monthly breakdown + concentration for the realistic pick
    pick = (base.sort_values("ev", ascending=False)
            .groupby(["day", "symbol"]).head(1)
            .sort_values("ev", ascending=False).groupby("day").head(3))
    print("\n=== long P1 bull EV>0 cap3/day sym1: monthly ===")
    g = pick.groupby("month").agg(n=("rm", "size"), R=("rm", "sum"), R_ret=("rr_", "sum"))
    print(g.round(1).to_string())
    vc = pick["symbol"].value_counts()
    print(f"\ntop symbols: {dict(vc.head(6))}, total syms: {len(vc)}")
    top_r = pick.nlargest(5, "rm")[["month", "symbol", "rm"]]
    print(f"top-5 single-event R: {[round(x, 1) for x in top_r['rm']]}")
    print(f"R without top-10 events: {pick['rm'].sum() - pick.nlargest(10, 'rm')['rm'].sum():+.1f}")
    # cost grid on the capped policy (re-selected per cost)
    print("\ncost grid (re-selected):")
    for cb in (10, 25, 50):
        cr = cb / 1e4 / df["d_final"]
        ev2 = df["p_hat"] * df["tp_rr"] - (1 - df["p_hat"]) - cr
        b2 = df[long_ & p1 & bull & (ev2 > 0)].copy()
        b2["rm2"] = df["r_mkt_eval"] - cr
        p2 = (b2.sort_values("ev", ascending=False).groupby(["day", "symbol"]).head(1)
              .sort_values("ev", ascending=False).groupby("day").head(3))
        m2 = p2.groupby("month")["rm2"].sum()
        print(f"  {cb}bps: n={len(p2)} R={p2['rm2'].sum():+.1f} avg={p2['rm2'].mean():+.3f} "
              f"mo_pos={(m2 > 0).sum()}/{len(m2)} worst={m2.min():+.1f}")

    # ML lift check inside the slice: decile split of long-P1-bull by p_hat
    sl = df[long_ & p1 & bull].copy()
    if len(sl) > 500:
        sl["q"] = pd.qcut(sl["p_hat"], 5, labels=False, duplicates="drop")
        print("\nlong-P1-bull by p_hat quintile (avg rm | avg rr_ | n):")
        g = sl.groupby("q").agg(n=("rm", "size"), avg_rm=("rm", "mean"),
                                avg_rr=("rr_", "mean"))
        print(g.to_string())


if __name__ == "__main__":
    main()
