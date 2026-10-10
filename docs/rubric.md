# Rubric 与评分门禁（初版，覆盖到 Market Agent）

每条 rubric 是一个“是/否”问题，由代码（code）或模型（llm）回答。规则写在 YAML 里，检查器和门禁都从同一份 YAML 读取，不在代码里另写一套。

## 资产与入口

| 位置 | 内容 |
|---|---|
| `src/shop_agent_langgraph/evaluation/rubrics/*.yaml` | 6 个模块文件：`e2e` `intent` `route` `market_screening` `research` `aggregation` |
| `eval/cases/gamepad/rubric.yaml` | gamepad overlay：`core_ids`、`excluded_keywords`，状态 `draft_pending_human_review` |
| `eval/tasks/<module>/*.json` | 手写 task：`{input, reference, rubric_ids, metadata}`；intent 5、route 5、e2e 3、research 9 |
| `evaluation/rubric.py` | 加载与校验、门禁、单一来源渲染、晋升判定 |
| `evaluation/rubric_checks.py` | 每条 code rubric 一个确定性、无网络的检查器 |
| `evaluation/rubric_run.py` | 对一次实验计算 research + aggregation rubric、落盘、按 run 目录做门禁 |

字段：`id` `question` `pass` `fail` `judge`(code\|llm) `severity`(veto\|gate\|monitor) `threshold`(仅 gate) `maturity`(shadow\|active) `depends_on_gold` `data_source`。加载时强制：`judge=llm` 必须 `shadow`，`judge=code` 必须 `active`，只有 gate 有阈值（0–1]，ID 带模块前缀且全局唯一。

research task 只引用 gold 的路径和商品 ID，不复制 gold；商品原文取自冻结的商品库，期望证据在计算时从 gold 读取。所有 task 的 `reference_status` 都是 `draft_pending_human_review`，没有人工批准过。

## Rubric 清单

每个检查单元：`pass` / `fail` / `unresolved`（无法判定，如页面原文缺失、混合 kind 的合并组）/ `na`（不适用，不计入分母）。

### E2E（输入：用户消息；数据：trace 工具序列与最终回复）

| ID | 问题 | 判定 | 级别 | 状态 | 依赖 gold |
|---|---|---|---|---|---|
| E2E-1 | 调用链是否正确（create/switch 依次 intent→route→market；ambiguous 不调 market；query 不调 route/market） | code | veto | active | 否 |
| E2E-2 | market 的 node_id 是否等于 route 的 resolved node_id | code | veto | active | 否 |
| E2E-3 | resolved node_id 是否等于参考或在允许集合内 | code | gate 0.9 | active | 是 |
| E2E-4 | Market 是否返回 completed | code | gate 0.9 | active | 否 |
| E2E-5 | 回复是否没有声称工具结果中不存在的内容 | llm | veto | shadow | 否 |
| E2E-6 | 回复是否不含 JSON、工具名、agent 名、内部 ID | code | gate 1.0 | active | 否 |
| E2E-7 | ambiguous 时回复是否给出 route 返回的每个候选名 | code | gate 1.0 | active | 否 |

### Intent（输入：消息与上下文；参考：IntentResult）

| ID | 问题 | 判定 | 级别 | 状态 | 依赖 gold |
|---|---|---|---|---|---|
| INT-1 | action 是否等于参考 | code | gate 0.95 | active | 是 |
| INT-2 | category 是否正确（参考为 null 则输出 null，否则归一化后相等） | code | gate 0.9 | active | 是 |
| INT-3 | 每条输出偏好在用户原话中是否有依据 | llm | veto | shadow | 否 |
| INT-4 | 参考偏好是否被召回（criteria/attribute 分别计） | llm | gate 0.8 | shadow | 是 |
| INT-5 | 消息中的数字是否在输出中原样保留 | code | veto | active | 否 |

### Route（输入：品类名；参考：status、node_id 或可接受候选集）

| ID | 问题 | 判定 | 级别 | 状态 | 依赖 gold |
|---|---|---|---|---|---|
| RT-1 | status 是否等于参考 | code | gate 0.9 | active | 是 |
| RT-2 | resolved 时 node_id 是否等于参考 | code | gate 0.9 | active | 是 |
| RT-3 | 返回的所有 node 是否存在于 taxonomy 且 name/path 一致 | code | veto | active | 否 |
| RT-4 | ambiguous 时参考节点是否在 candidates 中 | code | gate 0.8 | active | 是 |
| RT-5 | children 是否恰为 resolved 节点的直接子节点 | code | veto | active | 否 |

### Market 选品（数据：`MarketResult.audit`）

| ID | 问题 | 判定 | 级别 | 状态 | 依赖 gold |
|---|---|---|---|---|---|
| MS-1 | 入选 ID 是否都被搜索到、读过页面、无重复 | code | veto | active | 否 |
| MS-2 | 筛选证据 `source_text` 是否是所读页面原文的子串 | code | veto | active | 否 |
| MS-3 | 入选商品是否不含参考标为不相关的 | code | veto | active | 是 |
| MS-4 | 入选数量是否在 [3, 研究样本上限] | code | gate 1.0 | active | 否 |
| MS-5 | 参考相关商品的召回率 | code | monitor | active | 是 |

### Research（输入：单商品冻结原文；参考：gold 中该商品的 evidence）

| ID | 问题 | 判定 | 级别 | 状态 | 依赖 gold |
|---|---|---|---|---|---|
| RS-1 | 每条 `source_text` 是否是商品原文的子串 | code | veto | active | 否 |
| RS-2 | 固定相关商品是否 `relevance=relevant` | code | veto | active | 否 |
| RS-3 | `value` 与 `unit` 是否出现在其 `source_text` 中 | code | veto | active | 否 |
| RS-4 | 竞品/广告/对比表内容是否没有被标为 `subject=product` | llm | gate 0.9 | shadow | 否 |
| RS-5 | gold 中该商品有证据的维度是否至少有一条对应 evidence | llm | gate 0.7 | shadow | 是 |

### Aggregation（9 商品固定用例与 `EvalReport`）

| ID | 问题 | 判定 | 级别 | 状态 | 依赖 gold |
|---|---|---|---|---|---|
| AG-1 | 核心维度是否全部被 match | code | gate 1.0 | active | 是 |
| AG-2 | 非核心维度的覆盖率 | code | monitor | active | 是 |
| AG-3 | 已匹配维度的 kind/type/字段存在性是否正确（`deterministic_score`） | code | gate 0.95 | active | 是 |
| AG-4 | 核心维度是否一对一匹配（granularity=1） | code | gate 1.0 | active | 是 |
| AG-5 | units/direction/values/formula/value_domain 语义是否正确（现有 LLM 裁判的字段分） | llm | gate 0.9 | shadow | 是 |
| AG-6 | 输出维度的名称与别名是否不含排除类别关键词 | code | veto | active | 否 |
| AG-7 | 输出 ID 是否唯一，`source_item_ids` 是否等于输入 | code | veto | active | 否 |

## 实现时做的取舍

- 与原文比对前一律去除所有空白字符（含全角空格、零宽字符），不做其他归一化（不改大小写、不转全角）。因此 OCR 里 `$ \leq8M $` 这类公式写法与模型抽取的 `≤8` 会判为不一致，这是有意的严格。
- E2E-1 只覆盖 create/switch/query；update/remove/confirm 记 `na`。E2E-3 在参考要求 ambiguous 时记 `na`。
- E2E-6 的“内部 ID”只认键名（`node_id` 等）、UUID、9 位以上数字，以及带“节点/ID/编号”标签的节点 ID；裸数字（如预算 301）不算泄漏。E2E-7 只检查候选名是否出现，不检查有没有多说别的候选。
- INT-5 只检查阿拉伯数字，在 create/update/switch 上执行；数字出现在类别或偏好任一处即算保留（避免 “PS5” 这类型号数字被误判）。
- MS-4 的上限取 `CONFIG.market.max_search_products`（配置里没有 `max_research_products`）。MS-5 受该上限约束，召回率天然偏低，所以只做监控。
- AG-3 对混合 kind/type 的合并组无法由代码判定，记 `unresolved`，不猜。AG-4 把“核心维度没匹配上”也算失败（与 AG-1 同源，不是重复扣分的独立问题）。
- AG-5 不新增模型调用，直接读 `EvalReport` 里 0/1 字段分；`aliases` 是辅助字段，不计入。
- llm rubric 只暴露可注入的 judge 回调（`run_module(..., judges={id: fn})`）；没有 judge 时该 rubric 显示 `not_run`，不会伪造通过率。

## 门禁规则

1. **block**：任一 active 的 veto 有 fail；或 active 的 gate 低于阈值。
2. **needs_review**：
   - `depends_on_gold` 的 rubric 失败，但 gold 未 approved（case 状态与 overlay 状态都必须是 `approved` 才算批准）——失败降级；
   - veto 无 fail 但有 `unresolved`；gate 现在低于阈值，但把 `unresolved` 全算通过后能达标。
3. **pass**：其余情况。优先级 `block > needs_review > pass`。
4. shadow 与 monitor 只报告通过率；shadow 另外给出“若为 active 会是什么结论”（`would_be`），不影响总结论。
5. 通过率 = pass / (pass + fail + unresolved)；`na` 不计入；分母为 0 时为空，不是 100%。
6. 多 trial：veto 要求所有 trial 都没有 fail；gate 取各 trial 通过率的平均值（没有适用单元的 trial 跳过）。

## 命令

实验会在 `run_experiment` 生成 `EvalReport` 后计算 research + aggregation rubric，保存 `rubric.json`（含 rubric 文件 hash、各单元结果和门禁）与 `rubric.md`，并向 Langfuse 上报分数 `rubric.<id>`（通过率）与 `rubric.gate`（pass/needs_review/block）。重放已有交付（`--actual`）没有 Research 审计，research rubric 会缺席。

```powershell
# 对一个或多个 run 目录做门禁；多个目录视为同一用例的多次 trial。block 时退出码为 1。
uv run python scripts/market_evaluation.py gate --runs .cache/eval/gamepad/runs/<时间> [<时间2> ...]
# 额外把 needs_review 也当失败；保存 gate.json / gate.md
uv run python scripts/market_evaluation.py gate --runs <run目录> --fail-on-review --output .cache/eval/gamepad/gate
```

没有 `rubric.json` 的旧 run 会用 `market.json`、`report.json`、`case.json` 重新计算。混合不同 rubric 文件 hash、不同用例或不同 gold 版本的 run 会被拒绝。

`versions()` 把所有 rubric YAML 和 overlay 的 hash 放入 `prompt_hashes`；`compare_runs` 在这些 hash 或 `evaluation/rubric*.py` 不同时拒绝比较。

## 单一来源

`aggregation.yaml` 里 AG-3（kind、type）和 AG-5（units、formula、direction、values、value_domain、aliases）的 `fields[].definition` 是 schema 字段措辞的唯一来源：

- LLM 审核提示词：`agents/eval/schema_prompt.md` 的 `{{FIELD_DEFINITIONS}}` 在构造 `EvalAgent` 时渲染；
- Jev 指令：`jev_rules.md` + `Judge ONLY <field>.` + 同一定义；
- 人工题面：rubric 的 `review_question` 模板 + 同一定义。人工任务版本为 `REVIEW_VERSION = "v2"`，措辞变化会改变任务 ID。

改动一个字段的 `definition` 后三处同步变化。Jev 指令此前用的是更短的独立措辞，现改为与 LLM 提示词一致的完整定义；已有 Jev 校准结果基于旧指令，需要重新校准。

## shadow → active 晋升

llm rubric 初始都是 shadow。晋升必须同时满足（读校准 manifest 的 `by_field`，`rubric.promotion_status` 实现了此判定）：

1. 人机一致率 ≥ 0.9（`correct / count`，`unclear` 算不一致）；
2. 人工样本 ≥ 30 条；
3. 人工标注里正例和负例都有。

晋升时把该条 `maturity` 改为 `active`，并同时修改校验允许的组合（当前加载器要求 llm 必须 shadow，需随晋升流程一起调整）。晋升后重跑 baseline，因为 rubric hash 变化，旧 run 不能与新 run 直接比较。
