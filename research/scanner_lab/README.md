# scanner_lab — momentum scanner research pipeline

2y x 149-coin lab for scanner research. Full story:
`docs/notes/2026-07-07/sota-scanner-{master,findings}.md`.

## Quick start (venv: repo .venv, needs pandas/pyarrow/sklearn/lightgbm)

```
python download_klines.py --intervals 1h,5m     # ~5 min, data.binance.vision
python dataset.py                               # 25s -> research/data/events.parquet
python train_eval.py --model both               # walk-forward, monthly folds
python slot_sim.py                              # capacity-honest policy eval
python train_final.py                           # deploy artifacts
python validate_scanner.py                      # runtime parity check (run after ANY feature change!)
python scanner_v2.py                            # demo scan on latest cached hour
```

## Modules

- universe.py — parses the 149-coin table from the bybit-trading-universe note
- download_klines.py — monthly zips + daily fallback, parquet + manifest
- candidates.py — vectorized replica of production triggers P1/P1H/P2/P3 + gates
  S1-S11 (parity-checked vs prod tp rates); relaxed pool + is_hard/is_marginal flags
- labeling.py — triple-barrier on 5m paths (matrix.py parity) + retest sim + MFE/MAE
- features.py — ~50 features; NB the breadth epoch-leak fix lives here
- dataset.py — scan+label+features -> events.parquet
- train_eval.py — LR/LGBM walk-forward (24h purge, uniqueness weights, Platt)
- slot_sim.py / policy_sweep*.py / selection_final.py / exit_sweep.py /
  rules_eval.py / analyze.py — evaluation harnesses
- train_final.py — deployable artifacts -> research/artifacts/scanner_v2/
- scanner_v2.py — runtime scanner (EXPERIMENTAL: ML score is advisory, no edge
  claim; see artifacts meta.json)
- validate_scanner.py — runtime-vs-dataset parity (this catches leaks; keep it)

## Verdict snapshot (2026-07-08)

No deployable edge at 25bps in 1h breakout scanning (rules or ML). Survived
findings: long-only, drop P2, BTC>EMA50 gate, RR 3-4 @ 24h hold, stops >= 1.25x
noise floor, no retest on filtered flow, costs dominate (10bps ~ breakeven).
