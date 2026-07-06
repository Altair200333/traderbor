from __future__ import annotations

import argparse
import json

from traderbot_ai.screener.artifacts import write_scan_artifacts
from traderbot_ai.screener.render import to_markdown_table
from traderbot_ai.screener.screener import scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic momentum screener.")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--cache-path")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-artifact", action="store_true")
    args = parser.parse_args()
    result = scan(args.symbols, as_of_ms=args.as_of, cache_path=args.cache_path)
    if args.write_artifact:
        artifacts = write_scan_artifacts(result, args.run_id)
    else:
        artifacts = None
    if args.json:
        payload = result.model_dump(mode="json")
        if artifacts:
            payload["artifacts"] = artifacts
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(to_markdown_table(result))
        if artifacts:
            print(json.dumps(artifacts, ensure_ascii=False))


if __name__ == "__main__":
    main()
