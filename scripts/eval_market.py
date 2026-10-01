"""Compare an exported fixed-scope Market delivery against annotated gold."""
from __future__ import annotations

import argparse
from pathlib import Path

from shop_agent_langgraph.agents.eval import build_eval_agent
from shop_agent_langgraph.agents.eval.io import load_actual, load_gold, save_report
from shop_agent_langgraph.agents.eval.tools import validate_scope


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate Market name/description coverage and binary schema differences.")
    parser.add_argument("--gold", type=Path, default=root / "eval/cases/gamepad/gold.json")
    parser.add_argument("--actual", type=Path, required=True, help="Completed MarketResult, Market cache, or CriteriaAttributeSet JSON")
    parser.add_argument("--output", type=Path, default=root / ".cache/eval/gamepad")
    args = parser.parse_args()
    gold, metadata = load_gold(args.gold)
    actual = load_actual(args.actual)
    validate_scope(gold, actual)
    report = build_eval_agent().invoke(gold, actual)
    report.gold_metadata = metadata
    json_path, markdown_path = save_report(report, args.output)
    print(f"match groups: {report.metrics.match_group_count}; missing: {report.metrics.missing_count}; extra: {report.metrics.extra_count}")
    print(f"classification errors: {report.metrics.classification_error_count}; granularity errors: {report.metrics.granularity_error_count}")
    print(f"Report: {markdown_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
