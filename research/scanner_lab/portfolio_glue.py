"""
PORTFOLIO GLUE: liqrev v2 (with overlay) x unlock S1 (pre-cliff basket-hedged short)
x unlock S2 (post-cliff long, >=3%). Question #20 support analysis (descriptive,
no selection, no new strategy parameters).

Pre-registered scope (frozen before run):
- Inputs: research/data/liqrev/daily_equity_v2.parquet,
  research/data/unlocks/daily_returns_s1.parquet, daily_returns_s2.parquet.
  Align on the intersection of dates (liqrev span 2022-07..2026-07 binds).
  Missing sleeve days inside span = 0 return (no position).
- Metrics per sleeve and per combo: CAGR, maxDD, ann.vol, Sharpe(rf=0),
  worst month, full period + 2025-26 sub-window.
- Combos (declared, daily-rebalanced weight sums): A 50/50 liqrev+S1,
  B 50/50 liqrev+S2, C 1/3 each, D 70/20/10 liqrev/S1/S2.
- Correlations: daily + monthly pairwise.
- Storm section: monthly returns of each sleeve+combos for 2022-11, 2023-06,
  2024-08, 2025-02, 2026-01..06; conditional tails: mean sleeve return on the
  other sleeve's worst-5% days; 10 worst days of combo C with per-sleeve split.
- Output: research/data/unlocks/results_glue.json. No verdict rule — this is
  descriptive; the deployment decision stays with the frozen liqrev v2 spec.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIQ = ROOT / "research/data/liqrev/daily_equity_v2.parquet"
S1 = ROOT / "research/data/unlocks/daily_returns_s1.parquet"
S2 = ROOT / "research/data/unlocks/daily_returns_s2.parquet"
OUT = ROOT / "research/data/unlocks/results_glue.json"


def load(path, name):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    s = df.set_index("date")["ret"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s.name = name
    return s


def metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return None
    eq = (1 + r).cumprod()
    days = (r.index[-1] - r.index[0]).days or 1
    cagr = float(eq.iloc[-1] ** (365.0 / days) - 1)
    dd = float((eq / eq.cummax() - 1).min())
    vol = float(r.std() * np.sqrt(365))
    shp = float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else np.nan
    m = (1 + r).resample("ME").prod() - 1
    return {"cagr": round(cagr, 4), "maxdd": round(dd, 4), "vol": round(vol, 4),
            "sharpe": round(shp, 2), "worst_month": round(float(m.min()), 4),
            "n_days": int(len(r)), "total": round(float(eq.iloc[-1] - 1), 4)}


liq, s1, s2 = load(LIQ, "liqrev"), load(S1, "s1_short"), load(S2, "s2_long")
idx = pd.date_range(liq.index.min(), liq.index.max(), freq="D")
df = pd.DataFrame(index=idx)
for s in (liq, s1, s2):
    df[s.name] = s.reindex(idx).fillna(0.0)

combos = {"A_50liq_50s1": {"liqrev": .5, "s1_short": .5},
          "B_50liq_50s2": {"liqrev": .5, "s2_long": .5},
          "C_equal_thirds": {"liqrev": 1/3, "s1_short": 1/3, "s2_long": 1/3},
          "D_70_20_10": {"liqrev": .7, "s1_short": .2, "s2_long": .1}}
for name, w in combos.items():
    df[name] = sum(df[c] * wt for c, wt in w.items())

res = {"span": [str(idx[0].date()), str(idx[-1].date())]}
res["metrics_full"] = {c: metrics(df[c]) for c in df.columns}
sub = df[df.index >= "2025-01-01"]
res["metrics_2025_26"] = {c: metrics(sub[c]) for c in df.columns}

monthly = (1 + df[["liqrev", "s1_short", "s2_long"]]).resample("ME").prod() - 1
res["corr_daily"] = df[["liqrev", "s1_short", "s2_long"]].corr().round(3).to_dict()
res["corr_monthly"] = monthly.corr().round(3).to_dict()

storm_months = ["2022-11", "2023-06", "2024-08", "2025-02",
                "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
mm = (1 + df).resample("ME").prod() - 1
mm.index = mm.index.strftime("%Y-%m")
res["storm_monthly"] = mm.loc[mm.index.isin(storm_months)].round(4).to_dict("index")

cond = {}
for a, b in [("liqrev", "s1_short"), ("liqrev", "s2_long"),
             ("s1_short", "liqrev"), ("s1_short", "s2_long")]:
    active = df[df[a] != 0]
    if len(active) < 20:
        continue
    thr = active[a].quantile(0.05)
    bad = active[active[a] <= thr]
    cond[f"when_{a}_worst5pct_vs_{b}"] = {
        "n_days": int(len(bad)), f"{a}_mean": round(float(bad[a].mean()), 4),
        f"{b}_mean": round(float(bad[b].mean()), 4),
        f"{b}_share_negative": round(float((bad[b] < 0).mean()), 3)}
res["conditional_tails"] = cond

worst = df.nsmallest(10, "C_equal_thirds")[["C_equal_thirds", "liqrev", "s1_short", "s2_long"]]
res["worst10_days_comboC"] = {str(d.date()): {k: round(float(v), 4) for k, v in row.items()}
                              for d, row in worst.iterrows()}

OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
