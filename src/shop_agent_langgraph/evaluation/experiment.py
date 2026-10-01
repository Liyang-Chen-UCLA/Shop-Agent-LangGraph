from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile
from uuid import uuid4

from langfuse import Evaluation
from langfuse.langchain import CallbackHandler

from ..agents.eval.graph import EvalAgent
from ..agents.eval.io import load_actual, save_report
from ..agents.market.aggregation import MarketAggregationAgent
from ..agents.market.graph import MarketAgent
from ..agents.research.graph import build_research_agent
from ..core.llm import build_deepseek_model
from ..core.tracing import redact
from ..domain.criteria import CriteriaAttributeSet
from .cases import validate_case, write_json


def versions():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    files = ["agents/research/prompt.md", "agents/market/aggregation_prompt.md", "agents/eval/prompt.md",
             "agents/eval/schema_prompt.md", "agents/eval/jev_rules.md"]
    return {"git_commit": result.stdout.strip(), "working_tree_dirty": bool(dirty.stdout.strip()),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            "judge_model": os.getenv("EVAL_MODEL", "deepseek-flash"),
            "code_hashes": {str(p.relative_to(root / "src/shop_agent_langgraph")).replace("\\", "/"):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "src/shop_agent_langgraph").rglob("*.py")},
            "prompt_hashes": {name: hashlib.sha256((root / "src/shop_agent_langgraph" / name).read_bytes()).hexdigest() for name in files}}


def report_scores(report):
    m = report.metrics
    scores = [Evaluation(name=f"market.{name}", value=float(getattr(m, name)), data_type="NUMERIC")
              for name in ("gold_count", "actual_count", "matched_gold_count", "matched_actual_count",
                           "missing_count", "extra_count", "classification_error_count", "granularity_error_count")]
    scores += [Evaluation(name=f"market.{field}", value=value, data_type="NUMERIC")
               for field, value in m.field_scores.items() if value is not None]
    scores += [Evaluation(name=f"market.{kind}.{field}", value=float(getattr(coverage, field)), data_type="NUMERIC")
               for kind, coverage in m.by_kind.items()
               for field in ("gold_count", "actual_count", "matched_gold_count", "matched_actual_count", "missing_count", "extra_count")]
    return scores


def run_experiment(case, output_dir, *, client=None, jev=None, actual_path=None, model=None, judge_model=None, run_name=None,
                   dataset=None):
    """Execute the same task/evaluator locally or through Langfuse's runner."""
    gold = validate_case(case)
    if jev is not None:
        jev.preflight()  # fail before spending on nine research calls
        jev.trace_client = client
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "case.json", case)
    root = Path(__file__).resolve().parents[3]
    with ZipFile(directory / "source.zip", "w") as archive:
        sources = list((root / "src/shop_agent_langgraph").rglob("*.py")) + list((root / "src/shop_agent_langgraph").rglob("*.md"))
        sources += [root / "pyproject.toml", root / "uv.lock", root / "scripts/market_evaluation.py"]
        for path in sources:
            archive.write(path, str(path.relative_to(root)))
    supplied_model = model
    model = model or build_deepseek_model().model_copy(update={"request_timeout": 120, "max_retries": 1})
    judge_model = judge_model or supplied_model or build_deepseek_model().model_copy(update={
        "model_name": os.getenv("EVAL_MODEL", "deepseek-flash"), "request_timeout": 120, "max_retries": 1})
    metadata = {**case["metadata"], **versions(), "judge_mode": "jev-shadow" if jev else "llm",
                "jev_model": jev.model if jev else None,
                "run_mode": "existing-delivery" if actual_path else "research-and-delivery"}
    manifest = {"status": "running", "stage": "market", "run_name": run_name or uuid4().hex,
                "metadata": metadata}
    write_json(directory / "manifest.json", manifest)
    config = {"callbacks": [CallbackHandler()]} if client else {}

    async def task(*, item, **kwargs):
        if actual_path:
            collection = load_actual(actual_path)
            if set(collection.source_item_ids) != set(gold.source_item_ids):
                raise ValueError("existing delivery must use the case's fixed product IDs")
            output = dict(status="completed", item_ids=collection.source_item_ids,
                          criteria=[x.model_dump(mode="json") for x in collection.criteria],
                          attributes=[x.model_dump(mode="json") for x in collection.attributes])
        else:
            inp = item["input"] if isinstance(item, dict) else item.input
            market = MarketAgent(None, MarketAggregationAgent(model))
            result = await market.ainvoke_fixed_products(inp["source_item_ids"], inp["target"],
                product_contexts=inp["product_contexts"],
                research=build_research_agent(model, allow_external_search=False), config=config)
            output = result.model_dump(mode="json")
        write_json(directory / "market.json", output)
        print("Market delivery saved", flush=True)
        return output

    async def evaluator(*, output, expected_output, **kwargs):
        manifest["stage"] = "evaluation"
        write_json(directory / "manifest.json", manifest)
        if not output or output.get("status") != "completed":
            raise ValueError("cannot score a failed or pending Market task")
        actual = CriteriaAttributeSet.model_validate({"source_item_ids": output["item_ids"],
                   "criteria": output["criteria"], "attributes": output["attributes"]})
        report = await EvalAgent(judge_model, jev=jev, config=config,
                                 use_deterministic_fields=True, review_batch_size=4).ainvoke(
            CriteriaAttributeSet.model_validate(expected_output), actual)
        report.gold_metadata = metadata
        save_report(report, directory)
        manifest["evaluation_complete"] = True
        manifest["metrics"] = report.metrics.model_dump(mode="json")
        write_json(directory / "manifest.json", manifest)
        return report_scores(report)

    try:
        if client:
            kwargs = dict(name="market-fixed-products", run_name=manifest["run_name"],
                task=task, evaluators=[evaluator], max_concurrency=1,
                metadata={"input_sha256": metadata["input_sha256"], "gold_sha256": metadata["expected_output_sha256"],
                          "git_commit": metadata["git_commit"], "judge_mode": metadata["judge_mode"]})
            if dataset is None:
                result = client.run_experiment(data=[{k: case[k] for k in ("input", "expected_output", "metadata")}], **kwargs)
            else:
                result = dataset.run_experiment(**kwargs)
            manifest["experiment_id"] = result.experiment_id
            manifest["experiment_url"] = result.dataset_run_url
            manifest["trace_ids"] = [r.trace_id for r in result.item_results]
        else:
            async def local():
                output = await task(item=case)
                await evaluator(output=output, expected_output=case["expected_output"])
            asyncio.run(local())
        if not manifest.get("evaluation_complete"):
            raise RuntimeError("experiment task/evaluator failed; no completed report was produced")
        manifest.update(status="completed", stage="done")
    except Exception as error:
        manifest.update(status="failed", error=redact(str(error)))
        raise
    finally:
        write_json(directory / "manifest.json", manifest)
        if client:
            client.flush()
    return manifest
