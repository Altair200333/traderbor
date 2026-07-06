# Deterministic momentum screener

The screener is the runner-owned signal source for deterministic replay mode.
It reads only closed local-cache candles, computes the S1-S9 momentum gates,
detects P1/P2/P3 setups, applies replay state gates, and returns compact rows
plus plan primitives for candidates.

## Timeframes

- Signal data: closed `1h` candles.
- Deterministic replay scan cadence: every closed `1h` bar.
- Execution and TP/SL resolution in the simulator: normally `1m`.

This matches the strategy horizon: the screener finds 4h-24h momentum setups
from 1h structure without missing one-bar triggers, the runner calls the model only when candidates exist, and
the simulator resolves orders on the finest cached execution interval.

## Manual scan

From the repository root:

```powershell
$env:PYTHONPATH='agents-v2'; .venv\Scripts\python.exe -m traderbot_ai.screener --symbols "BTCUSDT,ETHUSDT,SOLUSDT" --as-of "2026-07-01T16:00:00Z" --cache-path "agents-v2\data\broad-2w-20260706\market_cache.sqlite3"
```

Write canonical replay artifacts:

```powershell
$env:PYTHONPATH='agents-v2'; .venv\Scripts\python.exe -m traderbot_ai.screener --symbols "BTCUSDT,ETHUSDT,SOLUSDT" --as-of "2026-07-01T16:00:00Z" --cache-path "agents-v2\data\broad-2w-20260706\market_cache.sqlite3" --run-id manual-check --write-artifact
```

JSON output:

```powershell
$env:PYTHONPATH='agents-v2'; .venv\Scripts\python.exe -m traderbot_ai.screener --symbols "BTCUSDT,ETHUSDT,SOLUSDT" --as-of "2026-07-01T16:00:00Z" --cache-path "agents-v2\data\broad-2w-20260706\market_cache.sqlite3" --json
```

Window summary with candidates, near-misses, and forward labels:

```powershell
$env:PYTHONPATH='agents-v2'; .venv\Scripts\python.exe -m traderbot_ai.screener.matrix --symbols "BTCUSDT,ETHUSDT,SOLUSDT" --start "2026-06-22T00:00:00Z" --end "2026-07-06T00:00:00Z" --step-interval 1h --cache-path "agents-v2\data\broad-2w-20260706\market_cache.sqlite3" --output "agents-v2\artifacts\screener-window-summary.json"
```

## Runner mode

Use the screener in exchange replay:

```powershell
$env:PYTHONPATH='agents-v2'; .venv\Scripts\python.exe -m traderbot_ai.cli exchange-replay --screener-mode deterministic --decision-provider codex-cli-mcp --codex-model gpt-5.5 --codex-reasoning-effort medium --decision-interval 1h --execution-interval 1m --symbols "BTCUSDT,ETHUSDT,SOLUSDT"
```

In deterministic mode, broad scans are done before the model call. If there are
no candidates, the runner records an auto-hold and skips the model.
Runner-owned max-hold and impulse-break maintenance also run on this cadence.
