# Rubric（MVP）

每条 rubric 是一个由代码回答的“是/否”问题，只报告通过率，不做门禁、不分级、不阻断交付。范围限定为 **gamepad 冻结 9 商品的 Research + Aggregation**，即实验实际执行的部分。

## 资产与入口

| 位置 | 内容 |
|---|---|
| `src/shop_agent_langgraph/evaluation/rubrics/{research,aggregation}.yaml` | rubric 定义：`id` `question` `pass` `fail` `data_source` |
| `eval/cases/gamepad/rubric.yaml` | gamepad overlay：`core_ids`（AG-1）、`excluded_keywords`（AG-6） |
| `evaluation/rubric.py` | 加载与校验、通过率汇总、schema 字段措辞的单一来源 |
| `evaluation/rubric_checks.py` | 每条 rubric 一个确定性、无网络的检查器 |
| `evaluation/rubric_run.py` | 对一次实验计算 rubric、落盘、加载多个 run |

## Rubric 清单

每个检查单元的结果是 `pass` / `fail` / `unresolved`（代码无法判定）/ `na`（不适用）。通过率 = pass / (pass + fail + unresolved)；`na` 不计入；分母为 0 时为空，不是 100%。

### Research（输入：单商品冻结原文与 Research 审计）

| ID | 问题 |
|---|---|
| RS-1 | 每条 evidence.source_text 是否是商品原文的子串？ |
| RS-2 | 固定相关商品的 relevance 是否为 relevant？ |
| RS-3 | evidence 的 value 与 unit 是否出现在其 source_text 中？ |

### Aggregation（输入：Market 交付与 `EvalReport`）

| ID | 问题 |
|---|---|
| AG-1 | 核心维度是否全部一对一匹配？ |
| AG-3 | 已匹配维度的 kind、type 与字段存在性是否正确？ |
| AG-6 | 输出维度是否不含排除类别？ |
| AG-7 | 输出 ID 是否唯一，且 source_item_ids 等于输入？ |

AG-1 的失败原因分三种：核心维度 missing、合并（多个 gold 对 1 个输出）、拆分（1 个 gold 对多个输出）。编号保留原值，空号不再使用。

## 实现时做的取舍

- 与原文比对前一律去除所有空白字符（含全角空格、零宽字符），不做其他归一化。因此 OCR 里 `$ \leq8M $` 这类公式写法与模型抽取的 `≤8` 会判为不一致，这是有意的严格。
- AG-1：核心 ID 不在 gold 里时记 `unresolved`。AG-3：混合 kind/type 的合并组无法由代码判定，记 `unresolved`，不猜。
- 重放已有交付（`--actual`）没有 Research 审计，RS-* 会缺席。
- gold 与 overlay 目前仍是草稿（`draft_pending_human_review`），依赖 gold 的通过率需结合人工复核解读。

## 输出

`run_experiment` 生成 `EvalReport` 后计算全部 rubric，保存 `rubric.json`（rubric 文件 hash、各单元结果、通过率汇总）与 `rubric.md`，并向 Langfuse 上报 `rubric.<id>`（通过率，NUMERIC）。

失败聚类与归因读取这些单元结果（见 [market-evaluation-lifecycle.md](market-evaluation-lifecycle.md)）。`rubric.json` 缺失或其 rubric hash 与当前文件不同时，会用 `market.json`、`report.json`、`case.json` 重新计算。

`versions()` 把 rubric YAML、`schema_fields.yaml` 和 overlay 的 hash 放入 `prompt_hashes`；`compare_runs` 在这些 hash 或 `evaluation/rubric*.py` 不同时拒绝比较。

## Schema 字段措辞的单一来源

`evaluation/schema_fields.yaml` 是 kind、type、units、formula、direction、values、value_domain、aliases 字段措辞的唯一来源：

- LLM 审核提示词：`agents/eval/schema_prompt.md` 的 `{{FIELD_DEFINITIONS}}` 在构造 `EvalAgent` 时渲染；
- Jev 指令：`jev_rules.md` + `Judge ONLY <field>.` + 同一定义；
- 人工题面：`review_question` 模板 + 同一定义（`REVIEW_VERSION = "v2"`，措辞变化会改变任务 ID）。

## 后续（不在 MVP 内）

E2E、Intent、Route、Market 选品的 rubric；LLM 裁判类 rubric（证据主体、gold 维度召回、字段语义）；门禁与阈值；多次运行稳定性、人工复标和人机一致率等验收协议。需要时再按实际失败补回。
