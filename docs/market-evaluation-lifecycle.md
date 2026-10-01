# 固定商品集的 Market Evaluation

范围是独立手柄的固定 9 商品：读取冻结的商品内容，执行 Research 和 Market 聚合，再审核交付。此流程不执行搜索或筛选，也不启用外部搜索。一个 Dataset item 表示完整的 9 商品用例，不是一个商品。

当前 gold 有 62 条维度及商品原文证据，状态为 `draft_pending_human_review`。评分衡量交付与这份草稿的差异，不能视为经过人工确认的质量结论。

## 1. 冻结用例与发布 Dataset

```powershell
uv run python scripts/market_evaluation.py prepare
uv run python scripts/market_evaluation.py publish-case
# 审阅 dataset-draft.json 后发布：
uv run python scripts/market_evaluation.py publish-case --apply --output .cache/eval/gamepad/published-dataset.json
```

用例保存商品全文、固定 ID、标准 schema gold、证据标注和内容哈希。Langfuse Dataset `market/fixed-products/gamepad` 保存这些内容；发布结果包含精确版本时间。相同用例使用稳定的 Dataset item ID，后续人工修订更新该 item，并保留历史 Dataset 版本。

## 2. 运行固定版本实验

在 `.env` 配置已有 DeepSeek、Langfuse 连接；Jev 使用 `TYPESAFE_API_KEY`，不要把密钥放进用例或标注。未配置 Jev 时可以先运行 LLM baseline：

```powershell
uv run python scripts/market_evaluation.py run --dataset market/fixed-products/gamepad --dataset-version <发布结果的version> --judge llm --langfuse --name baseline
uv run python scripts/market_evaluation.py run --dataset market/fixed-products/gamepad --dataset-version <同一version> --judge jev-shadow --langfuse --name jev-shadow
```

默认 Jev 模型固定为 `jev-1.13.0`。`JEV_MIN_CONFIDENCE` 默认留空，不能在没有校准数据时假定一个可靠阈值。当前 Langfuse 固定为 `4.15.6`、TypeSafe SDK 为 `0.7.2`，保持与项目的 LangGraph/OpenTelemetry 依赖兼容。

Research 只收到商品输入，不接触 gold；遇到未确认相关的商品时，实验失败并保留记录，不悄悄缩小商品范围。Market 一次聚合全部结果。每次实验在 `.cache/eval/gamepad/runs/<时间>/` 保存输入、源码快照、交付、报告与版本清单，同时关联 Langfuse Dataset run 和 trace。

已有交付可通过 `--actual <market.json>` 重评；清单会标明 `existing-delivery`，不能把它当作重新执行 Research/Market 的实验。

## 3. 匹配与细粒度评分

第一阶段仅根据 `name`、`description` 划分 match、missing、extra；允许多个维度进入同一匹配组。第二阶段审核匹配项的全部适用字段，每项为 0/1，并记录 criteria/attribute 误分类、合并多个 item 的粒度错误。没有适用样本的字段为 N/A，不计作 0。

可确定的 kind、type、字段缺失和空公式由程序判断；剩余语义字段由 baseline LLM 审核，按 4 个 gold item 分批校验完整覆盖。Jev 对语义字段并行给出 `correct / incorrect / unclear` 及置信度、概率，保留原始返回。只有明确选择对应候选 0/1；`unclear` 保持未决，不冒充错误标签。

Jev 当前是 shadow：不会替换 baseline 分数。未校准、低置信度、未决或与 baseline 不一致的判断需要人工复核。Jev 调用作为 generation 记录输入、问题、模型、返回和用量。

## 4. Annotation queue MVP

队列 `market-dimension-review` 使用 BOOLEAN 配置 `human_correct`。每个任务对应独立 observation，稳定任务 ID 与本地 ledger 防止重试重复入队。

```powershell
uv run python scripts/market_evaluation.py review --report <run目录>/report.json --kinds matching schema_field --limit 5 --output .cache/eval/gamepad/delivery-review-tasks.json
uv run python scripts/market_evaluation.py publish-review --tasks .cache/eval/gamepad/delivery-review-tasks.json
```

任务有四种：gold 单项及证据审核、gold 全集漏项审核、完整匹配划分审核、单个 schema 字段审核。审核员先独立判断：正确为 1，错误为 0，并写理由。gold 或匹配有误时，用 Langfuse corrected output 编辑完整 JSON；字段任务直接使用 0/1 和备注。

为降低锚定影响，自动评分不放入待审 observation。人工完成后，导出流程再向同一 observation 附上 LLM/Jev 评分，供 Score Analytics 比较。不要代填人工标签。

```powershell
uv run python scripts/market_evaluation.py export-review --report <run目录>/report.json --output .cache/eval/gamepad/review-export
```

只读取已完成且存在唯一有效人工标签的任务。导出验证用例版本、固定商品范围、schema 和原文引文，产生 `gold-candidate.json`、`case-candidate.json`、`human-reviews.json`、`jev-calibration.json`，不会覆盖原始 gold。导出的候选仍待人工最终确认；完成队列任务不等于批准整份 gold。

## 5. 校准与回归

```powershell
uv run python scripts/market_evaluation.py calibrate --labels .cache/eval/gamepad/review-export/jev-calibration.json --output .cache/eval/gamepad/calibration/run-1 --langfuse
uv run python scripts/market_evaluation.py compare --baseline <baseline目录>/manifest.json --candidate <candidate目录>/manifest.json --output .cache/eval/gamepad/comparison.json
```

校准时 Jev 只看输入，不看人工答案。结果按字段统计一致数量、未决数量和正负样本覆盖，不自动批准模型上线。需先积累包含正确、错误及容易混淆项的真实人工样本，再决定规则、模型和阈值；更新后重新校准。

运行比较要求输入、gold、裁判模型及评估代码/提示词一致；输出 missing、extra、误分类、粒度错误及字段均值差异。没有预设总分权重或自动发布门槛。改变 gold 或裁判版本后，应在新版本上重跑 baseline 与 candidate。

官方参考：[Jev](https://langfuse.com/docs/evaluation/evaluation-methods/jev-as-a-judge)、[Dataset experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk)、[Annotation queues](https://langfuse.com/docs/evaluation/evaluation-methods/annotation-queues)、[Corrections](https://langfuse.com/docs/observability/features/corrections)。
