from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from traderbot_ai.agents.trading import build_trading_instructions
from traderbot_ai.config import Settings
from traderbot_ai.decision.prompts import build_exchange_replay_prompt
from traderbot_ai.paths import AGENTS_V2_ROOT, PROJECT_ROOT, RUNS_DIR
from traderbot_ai.runtime.usage import UsageTotals, usage_totals
from traderbot_ai.schemas import TradeDecision


CodexFailOpen = Literal["error", "hold"]


@dataclass(frozen=True)
class CodexCliOptions:
    model: str | None = None
    reasoning_effort: str | None = None
    profile: str | None = None
    sandbox: str = "read-only"
    timeout_sec: int = 1800
    output_dir: Path | None = None
    fail_open: CodexFailOpen = "error"
    codex_executable: str = "codex"
    python_executable: str = sys.executable
    ignore_user_config: bool = True
    full_auto: bool = True
    mcp_server_name: str = "traderbot"
    include_mcp_config: bool = True


class CodexCliMcpDecisionProvider:
    name = "codex-cli-mcp"

    def __init__(
        self,
        *,
        settings: Settings,
        session_name: str,
        options: CodexCliOptions | None = None,
    ) -> None:
        self._settings = settings
        self._session_name = session_name
        self._options = options or CodexCliOptions()

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        paths = self._step_paths(context)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        _clear_step_artifacts(paths)
        paths.schema_path.write_text(json.dumps(_trade_decision_output_schema(), indent=2), encoding="utf-8")
        prompt = self._build_prompt(context)
        paths.prompt_path.write_text(prompt, encoding="utf-8")
        env = self._build_env()
        command = self._build_command(paths)
        result: subprocess.CompletedProcess[str] | None = None
        try:
            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self._options.timeout_sec,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired as error:
            paths.stdout_path.write_text(error.stdout or "", encoding="utf-8")
            paths.stderr_path.write_text(error.stderr or "", encoding="utf-8")
            return self._handle_error(f"codex timed out after {self._options.timeout_sec}s", paths, context)

        paths.stdout_path.write_text(result.stdout or "", encoding="utf-8")
        paths.stderr_path.write_text(result.stderr or "", encoding="utf-8")
        codex_usage = _codex_usage_from_jsonl_text(result.stdout or "")
        if result.returncode != 0:
            return self._handle_error(f"codex exited with code {result.returncode}", paths, context)

        final_text = paths.final_message_path.read_text(encoding="utf-8") if paths.final_message_path.exists() else ""
        try:
            decision = _parse_trade_decision(final_text)
        except ValueError as error:
            return self._handle_error(str(error), paths, context)

        output = decision.model_dump(mode="json")
        output.update(
            {
                "codex_stdout_log": str(paths.stdout_path),
                "codex_stderr_log": str(paths.stderr_path),
                "codex_prompt_path": str(paths.prompt_path),
                "codex_final_message_path": str(paths.final_message_path),
                "codex_decision_path": str(paths.decision_path),
                "codex_mcp_audit_path": str(paths.audit_path),
                "codex_usage": codex_usage,
            }
        )
        paths.decision_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        return output

    def close(self) -> None:
        return None

    def _build_prompt(self, context: dict[str, Any]) -> str:
        system_prompt = build_trading_instructions(self._settings, exchange_replay=True)
        user_prompt = build_exchange_replay_prompt(context)
        return f"""
You are running Traderbot V2 through Codex CLI.

Use the configured traderbot MCP tools for market data, wallet state, risk validation, and simulator exchange writes.
Do not use live market data.
Do not read raw cache/state files directly when an MCP tool can provide the same fact.
Do not use filesystem write tools or shell commands to mutate simulator state.
The replay runner owns clock movement and settlement.

Codex MCP tool order:
1. Use get_wallet_compact before scanning. Use full get_wallet only if compact output is missing a specific fact needed for an entry or maintenance close.
2. Use get_recent_trade_events for risk/cooldown reconstruction.
3. Use scan_momentum_universe once for the broad symbol scan. This replaces raw per-symbol 4h get_candles calls for coarse screening.
4. Use get_candidate_detail for at most 2 finalists. This replaces raw 1h get_candles calls for deep checks when it returns the needed facts.
5. Use get_candles only as a fallback for missing screener/detail facts, never as the default broad scan.
6. Use validate_order and calculate_position_size before any entry. calculate_position_size.amount is USDT notional; TradeDecision.amount is USDT notional; linear place_order.qty is base-asset quantity, so use qty = notional / current entry price. Do not pass USDT notional as linear qty and do not use marketUnit for linear orders.
7. Then set_leverage and place_order only for a real long/short decision.

System instructions:
{system_prompt}

Replay step prompt:
{user_prompt}

Return only the final JSON object matching the TradeDecision schema.
"""

    def _build_command(self, paths: "_CodexStepPaths") -> list[str]:
        executable = shutil.which(self._options.codex_executable) or self._options.codex_executable
        command = [
            executable,
            "exec",
            "--ephemeral",
            "--json",
            "--sandbox",
            self._options.sandbox,
            "--cd",
            str(PROJECT_ROOT),
            "--output-schema",
            str(paths.schema_path),
            "--output-last-message",
            str(paths.final_message_path),
        ]
        if self._options.full_auto:
            command.append("--full-auto")
        command.extend(["-c", 'approval_policy="never"'])
        if self._options.ignore_user_config:
            command.append("--ignore-user-config")
        if self._options.model:
            command.extend(["--model", self._options.model])
        if self._options.reasoning_effort:
            command.extend(["-c", f"model_reasoning_effort={_toml_string(self._options.reasoning_effort)}"])
        if self._options.profile:
            command.extend(["--profile", self._options.profile])
        if self._options.include_mcp_config:
            command.extend(self._mcp_config_overrides(paths))
        command.append("-")
        return command

    def _mcp_config_overrides(self, paths: "_CodexStepPaths") -> list[str]:
        prefix = f"mcp_servers.{self._options.mcp_server_name}"
        env = self._mcp_env(paths)
        overrides = [
            "-c",
            f"{prefix}.command={_toml_string(self._options.python_executable)}",
            "-c",
            f"{prefix}.args=[\"-m\",\"traderbot_ai.mcp.server\"]",
            "-c",
            f"{prefix}.startup_timeout_sec=60",
            "-c",
            f"{prefix}.tool_timeout_sec=120",
            "-c",
            f'{prefix}.default_tools_approval_mode="approve"',
            "-c",
            f"{prefix}.required=true",
        ]
        for key, value in env.items():
            overrides.extend(["-c", f"{prefix}.env.{key}={_toml_string(value)}"])
        return overrides

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        _prepend_env_path(env, "PYTHONPATH", str(AGENTS_V2_ROOT))
        return env

    def _mcp_env(self, paths: "_CodexStepPaths") -> dict[str, str]:
        env = {
            "PYTHONPATH": _join_env_path(str(AGENTS_V2_ROOT), os.environ.get("PYTHONPATH")),
            "TRADERBOT_MCP_AUDIT_PATH": str(paths.audit_path),
            "TRADERBOT_MCP_RUN_ID": self._session_name,
            "TRADERBOT_MCP_STEP_ID": paths.audit_path.name.replace(".mcp-audit.jsonl", ""),
        }
        for key in (
            "TRADERBOT_EXCHANGE_BACKEND",
            "TRADERBOT_EXCHANGE_STATE_PATH",
            "TRADERBOT_EXCHANGE_EVENTS_PATH",
            "TRADERBOT_EXCHANGE_FEE_RATE",
            "TRADERBOT_MARKET_CACHE_PATH",
            "TRADERBOT_EXCHANGE_EXECUTION_INTERVAL",
            "TRADERBOT_SIMULATION_CLOCK_PATH",
        ):
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def _step_paths(self, context: dict[str, Any]) -> "_CodexStepPaths":
        output_root = Path(self._options.output_dir) if self._options.output_dir is not None else RUNS_DIR / "codex"
        output_dir = output_root / _safe_file_part(self._session_name)
        step_id = _safe_file_part(str(context.get("as_of_ms") or uuid.uuid4().hex))
        return _CodexStepPaths(
            output_dir=output_dir,
            schema_path=output_dir / "trade-decision.schema.json",
            prompt_path=output_dir / f"{step_id}.prompt.txt",
            stdout_path=output_dir / f"{step_id}.jsonl",
            stderr_path=output_dir / f"{step_id}.stderr.txt",
            final_message_path=output_dir / f"{step_id}.final.txt",
            decision_path=output_dir / f"{step_id}.decision.json",
            audit_path=output_dir / f"{step_id}.mcp-audit.jsonl",
        )

    def _handle_error(self, message: str, paths: "_CodexStepPaths", context: dict[str, Any]) -> dict[str, Any]:
        if self._options.fail_open == "hold":
            return {
                "final_decision": "hold",
                "symbol": context["symbols"][0],
                "timeframe": context["decision_interval"],
                "thesis": "Codex provider failed; fail-open hold.",
                "amount": 0.0,
                "confidence": 0.0,
                "risk_summary": message,
                "tool_summary": ["codex-cli-mcp failed before valid TradeDecision"],
                "codex_stdout_log": str(paths.stdout_path),
                "codex_stderr_log": str(paths.stderr_path),
                "codex_prompt_path": str(paths.prompt_path),
                "codex_final_message_path": str(paths.final_message_path),
                "codex_mcp_audit_path": str(paths.audit_path),
                "codex_usage": _codex_usage_from_file(paths.stdout_path),
            }
        raise RuntimeError(f"Codex CLI decision failed: {message}")


@dataclass(frozen=True)
class _CodexStepPaths:
    output_dir: Path
    schema_path: Path
    prompt_path: Path
    stdout_path: Path
    stderr_path: Path
    final_message_path: Path
    decision_path: Path
    audit_path: Path


def _parse_trade_decision(text: str) -> TradeDecision:
    stripped = text.strip()
    if not stripped:
        raise ValueError("codex final message was empty")
    candidates = [stripped]
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))
    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            _validate_strict_trade_decision_object(data)
            return TradeDecision.model_validate(data)
        except Exception as error:
            last_error = error
    raise ValueError(f"codex final message did not match TradeDecision JSON: {last_error}")


def _validate_strict_trade_decision_object(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("TradeDecision output must be a JSON object")
    schema = _trade_decision_output_schema()
    properties = set(schema.get("properties") or {})
    required = set(schema.get("required") or properties)
    keys = set(data)
    missing = sorted(required - keys)
    extra = sorted(keys - properties)
    if missing:
        raise ValueError(f"TradeDecision output missing required fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"TradeDecision output has unexpected fields: {', '.join(extra)}")


def _clear_step_artifacts(paths: _CodexStepPaths) -> None:
    for path in (
        paths.stdout_path,
        paths.stderr_path,
        paths.final_message_path,
        paths.decision_path,
        paths.audit_path,
    ):
        if path.exists():
            path.unlink()


def _codex_usage_from_file(path: Path) -> dict[str, int | float]:
    if not path.exists():
        return _usage_dict(UsageTotals())
    return _codex_usage_from_jsonl_text(path.read_text(encoding="utf-8"))


def _codex_usage_from_jsonl_text(text: str) -> dict[str, int | float]:
    records: list[Any] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("usage"), dict):
            records.append(record["usage"])
        else:
            records.append(record)
    return _usage_dict(usage_totals(records))


def _usage_dict(totals: UsageTotals) -> dict[str, int | float]:
    total_tokens = totals.total_tokens or (totals.input_tokens + totals.output_tokens)
    return {
        "input_tokens": totals.input_tokens,
        "cached_input_tokens": totals.cached_input_tokens,
        "uncached_input_tokens": totals.uncached_input_tokens,
        "output_tokens": totals.output_tokens,
        "total_tokens": total_tokens,
        "cache_hit_ratio": totals.cache_hit_ratio,
    }


def _trade_decision_output_schema() -> dict[str, Any]:
    schema = TradeDecision.model_json_schema()
    return _strict_json_schema(schema)


def _strict_json_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_strict_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _strict_json_schema(item) for key, item in value.items() if key != "default"}
    if result.get("type") == "object" or "properties" in result:
        properties = result.get("properties")
        if isinstance(properties, dict):
            result["required"] = list(properties.keys())
        result["additionalProperties"] = False
    return result


def _safe_file_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-") or "codex"


def _toml_string(value: str) -> str:
    return json.dumps(str(value))


def _join_env_path(first: str, rest: str | None) -> str:
    if rest:
        return first + os.pathsep + rest
    return first


def _prepend_env_path(env: dict[str, str], key: str, first: str) -> None:
    env[key] = _join_env_path(first, env.get(key))
