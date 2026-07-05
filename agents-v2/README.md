# Traderbot Agents V2

Fresh Python agent runtime using the OpenAI Agents SDK.

## Setup

From repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r agents-v2\requirements.txt
```

The runtime loads `.env` from repo root. Required:

```text
OPENAI_API_KEY=...
TRADERBOT_MODEL=gpt-5.5
TRADERBOT_VISION_MODEL=gpt-5.5
TRADERBOT_REASONING_EFFORT=medium
```

## Commands

List tools:

```powershell
$env:PYTHONPATH='agents-v2'; .\.venv\Scripts\python.exe -m traderbot_ai.cli tools
```

Check configured providers without an API call:

```powershell
$env:PYTHONPATH='agents-v2'; .\.venv\Scripts\python.exe -m traderbot_ai.cli providers
```

Run a live provider probe:

```powershell
$env:PYTHONPATH='agents-v2'; .\.venv\Scripts\python.exe -m traderbot_ai.cli providers --live
```

Render a chart without the agent:

```powershell
$env:PYTHONPATH='agents-v2'; .\.venv\Scripts\python.exe -m traderbot_ai.cli chart --symbol BTCUSDT --interval 15m --limit 80
```

Run the agent:

```powershell
$env:PYTHONPATH='agents-v2'; .\.venv\Scripts\python.exe -m traderbot_ai.cli run --session btc-demo --prompt "Analyze BTCUSDT on 15m, use tools, validate risk, write worklog, and return a paper-trading decision."
```

Show local session history:

```powershell
$env:PYTHONPATH='agents-v2'; .\.venv\Scripts\python.exe -m traderbot_ai.cli history --session btc-demo
```

## Storage

- SQLite chat history: `agents-v2/data/sessions.sqlite`
- Paper portfolio: `agents-v2/data/portfolio.json`
- Run audit logs: `agents-v2/runs/YYYY-MM-DD.jsonl`
- Market and chart artifacts: `agents-v2/artifacts/`
- Agent worklog tool output: `worklog/YYYY-MM-DD-record.md`

## Design Notes

- The agent can read/write files under `agents-v2` through workspace tools.
- Workspace paths accept `abc/asd`, `./abc/asd`, and backslashes.
- Workspace paths cannot escape upward out of `agents-v2`.
- File read supports char or line mode with `seek` and `count`.
- File search returns paginated snippets instead of whole files.
- Code execution is exposed through a bounded Python snippet tool rooted at `agents-v2`.
- Market artifacts save compact `{t,o,h,l,c,v}` candles with metadata.
- Chart artifacts save PNG, metadata JSON, and source candle JSON.
- Vision tools can inspect saved chart/image files and return short visual notes.
- The experimental OpenAI Codex tool is enabled when available.
- Live trading tools are not implemented by default. Paper trading and risk validation are present first.
- Deterministic risk validation owns order geometry and sizing checks.
- SQLite sessions preserve conversation state. Stable prompts plus `prompt_cache_retention="24h"` help OpenAI prompt caching, but cache hits are not guaranteed.
