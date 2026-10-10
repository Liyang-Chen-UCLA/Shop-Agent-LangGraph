# Gamepad gold：按人工裁决完成规则复审

2026-10-10；依据rubric提交 `5a23c02`。人工逐项裁决与后续AI规则复审分开记录，整份gold及overlay仍为 `draft_pending_human_review`。

## 当前结论

保留 **61项维度：5 criteria、56 attributes；202条证据；18项候选核心**。原稿为62项/215条证据。核心集合不因新增功能自动扩张；其选择和AG-4一对一要求尚未得到整体验收。

## 对用户规则的理解

- 先确定通用能力或物理量，再决定值与证据；商品、软件和品牌名称不是独立标准。
- 允许明确厂商量化声明；指标含义、数值和单位须可确定，未知含义不补猜。
- 模式支持取并集并保留条件；SKU数值分别记录，无法对应SKU的冲突数值不纳入。
- 硬件类型、功能支持、数量、配置分别记录；固定布局切换不证明自定义映射。
- 上位类别可以保留原文确定的范围，不强行细分；有电池不自动推出可充电。
- 布尔能力存在与具体档数分开；档数冲突不自动否定原文明确的能力存在。
- 有优劣方向且条件可比的量列criteria；配置、偏好和不能无条件排序的量列attributes。

## 本次自主处理

- 摇杆分辨率：去掉没有充分依据的“输出”限定，保留明确级数/位数，不自动互换或推有效精度。
- 供电方式：只记录电池供电/USB供电，避免容量→内置可充电的过度推断。
- 灯光可调：5/6种冲突只影响档数，能力有无仍有明确支持；未新增档数标准。
- 充电时间：保留原文量化时间，不要求或虚构原文未披露的空到满测试端点。
- 死区/曲线可调：改用完整“自由调节…”连续原文，软件是实现上下文。
- 按键结构及连接方式：删掉未获支持的具体轴体和已排除商品形态的定义内容。

## 已执行的关键人工裁决

- 删除只有定性宣传的输入延迟，删除含义不明的0.5ms传输延时；保留回报率。
- 鲲鹏5400+不能消歧，移出证据；九尾狐磁变阻归磁阻，不补TMR子类。
- 硬件扳机类型/快速模式、马达类型/对称配置分开；键位映射/固定布局分开。
- 软件名称不作独立标准，移除配套配置软件条目，具体功能各自保留。
- Q37按X/S/P三档平台取并集；尺寸按用户看图确认的横向125、纵向61.5、厚20.8mm记录。
- 车载SKU未知>10H移出，只保留8小时及电竞/机械款10–12小时；不误删Q37明确型号>10H。

## rubric 如何使用这份gold

- RS-1：202条quote均能在指定页面及冻结Research正文找到；页面哈希、图片路径、记录ID和范围一致。
- RS-3：约束Research原始value/unit；gold的规范单位属于Market表示，不能反向要求改写原始引文。
- RS-4：只用目标商品/部件/列及对应SKU的原文，不用广告/竞品支持目标配置。
- RS-5：按每个商品实际有证据的维度召回，不要求9个商品每个覆盖61项。当前实验未注入RS-4/RS-5 judge，不声称已通过这些语义裁判。
- AG-1/AG-4：18项核心候选；合理拆分摇杆/整机回报率也可能受一对一门槛惩罚，这属于评分政策，不是商品事实，尚未改动。
- AG-2：其余43项覆盖只作监控；AG-3/AG-5审核schema及语义，字段不应要求超出原文的细分。
- AG-6/AG-7：无排除关键词、重复维度ID或范围不一致；schema能加载，冻结用例能生成。

## 全量61项的判断

| ID | 标准 | 类型 | 核心 | 证据数 | 按既定规则的判断 |
|---|---|---|---|---:|---|
| `report_rate` | 回报率 | criteria/numeric | 是 | 5 | 保留Hz上报频率；区分摇杆/整机、协议与超频SKU，禁止倒数推成传输延时。 |
| `button_lifetime` | 按键寿命 | criteria/numeric | 是 | 4 | 保留有数字的按键循环声明；部件范围另有原文，测试条件未知不编造；不使用质保或摇杆寿命。 |
| `stick_lifetime` | 摇杆寿命 | criteria/numeric | 是 | 2 | 保留明确摇杆部件循环次数；不推出永久无漂移或整机寿命。 |
| `battery_runtime` | 电池续航时间 | criteria/numeric | 是 | 3 | 版本可对应的续航分别保留；车载未知SKU的>10H已删除，Q37明确型号>10H保留。 |
| `charging_time` | 充电时间 | criteria/numeric | 是 | 2 | 保留2小时/约3小时量化充电声明；不擅自补出从空到满、规定电源或独立实测条件。 |
| `compatible_platform` | 兼容平台 | attributes/categorical | 是 | 12 | 各档平台取并集；Q37三档完整表已补齐，条件不丢失，不排除明确的PS4/5厂商声明。 |
| `connection_mode` | 连接方式 | attributes/categorical | 是 | 11 | 输入连接方式与供电/充电分开；按明确模式/SKU取并集，独立手柄之外的触屏传导不混入。 |
| `bluetooth_version` | 蓝牙版本 | attributes/categorical | 否 | 3 | 原文5.0/5.3是版本类别；不凭版本大小设优劣或推出延迟。 |
| `stick_sensor` | 摇杆传感器类型 | attributes/categorical | 是 | 6 | 用户裁决九尾狐归磁阻；TMR明示者保留子类，不从上位名称补猜子类。 |
| `trigger_mode` | 扳机类型 | attributes/categorical | 是 | 7 | 按用户拆分保留扳机输入类型；快速模式只在独立布尔条目，不据快速推双切。 |
| `rapid_trigger_support` | 快速扳机模式支持 | attributes/boolean | 否 | 1 | 原文明示快速模式支持，可记布尔能力；不推具体算法或性能，不自动升级为核心。 |
| `trigger_resolution` | 扳机输出分级 | attributes/numeric | 否 | 3 | 保留明确0–255/256阶原文口径；不将输出级数当作位移误差或行程。 |
| `trigger_travel` | 扳机行程 | attributes/numeric | 否 | 2 | 0.3mm短行程与8mm线性行程分别带模式，不能把精度0.1mm当行程。 |
| `trigger_position_precision` | 扳机位移精度声明 | attributes/numeric | 否 | 1 | 保留0.1mm级厂商精度声明，实际误差定义未知，不宣称实测精度或推优劣方向。 |
| `stick_resolution` | 摇杆分辨率 | attributes/numeric | 否 | 3 | 保留明确级数/位数的原文声明；删除输出限定，不在级与bit之间自动换算或推有效精度；5400+排除。 |
| `deadzone_adjustable` | 摇杆死区可调 | attributes/boolean | 否 | 2 | 保留明确可调死区/反死区；完整语句已补齐，0死区本身不证明可调。 |
| `stick_curve_adjustable` | 摇杆输出曲线可调 | attributes/boolean | 否 | 2 | 保留方/圆调教和明确可调曲线，不与灵敏度或死区合并。 |
| `stick_sensitivity_adjustable` | 摇杆灵敏度可调 | attributes/boolean | 否 | 1 | 明确灵敏度自定义支持布尔能力，不等同输出分辨率。 |
| `stick_tension_adjustable` | 摇杆阻尼可调 | attributes/boolean | 否 | 2 | 原文可调阻尼支持布尔能力；范围另列，不预设阻尼越大越好。 |
| `stick_tension_range` | 摇杆阻尼调节力范围 | attributes/numeric | 否 | 1 | 保留30–90gf明确力范围，数值属性不带单向偏好。 |
| `stick_calibration` | 摇杆校准/归中功能 | attributes/boolean | 否 | 3 | 保留自动校准/归中能力，不转成回中误差=0。 |
| `button_mechanism` | 按键结构 | attributes/categorical | 是 | 7 | 按目标部件原文记录结构/用料，不补黑轴等未披露细分，也不移入竞品导电胶。 |
| `button_travel` | 按键触发行程 | attributes/numeric | 否 | 2 | 保留动作按键0.3/0.7mm，扳机行程独立，不转为输入延迟。 |
| `button_actuation_force` | 按键触发力 | attributes/numeric | 否 | 1 | 保留70gf按压力声明；重量单位g与力单位gf不混用，不设置无条件偏好。 |
| `remapping` | 键位映射 | attributes/boolean | 否 | 2 | 保留自定义/编程/导入键位映射；固定布局互换证据已分到布局切换。 |
| `macro_recording` | 宏录制/编程 | attributes/boolean | 是 | 5 | 保留明确宏编程/多步连招能力；不把单键映射或简单连发当宏。 |
| `turbo` | 连发功能 | attributes/boolean | 否 | 3 | 明确连发开启方法或TURBO支持，保留布尔能力；游戏射速宣传不是连发证据。 |
| `turbo_speed_levels` | 连发调速档数 | attributes/numeric | 否 | 1 | 保留三种连发频率的档数，不把档数当Hz或默认越多越好。 |
| `motion_control` | 体感功能 | attributes/boolean | 是 | 6 | 明确姿态输入能力保留原生/映射/平台条件；摇杆360度文案不证明体感。 |
| `vibration` | 震动反馈 | attributes/boolean | 是 | 9 | 明确震动或震动马达支持能力；不推反馈品质或所有平台均可用。 |
| `vibration_motor_type` | 震动马达类型 | attributes/categorical | 是 | 2 | 仅保留明确音圈线性/纽扣结构；数量和对称配置独立。 |
| `vibration_symmetry` | 震动对称配置 | attributes/categorical | 否 | 3 | 按用户拆分，保留NS对称/PC非对称条件；不推机械布局，不自动升级为核心。 |
| `vibration_motor_count` | 震动马达数量 | attributes/numeric | 否 | 6 | 按部件数量保留；八位堂1+1+2原文对应位置，不凭立体/丰富宣传猜数量。 |
| `trigger_vibration` | 扳机震动 | attributes/boolean | 否 | 3 | 明确扳机部位震动且保留SKU；双切功能不能单独证明扳机震动。 |
| `vibration_adjustable` | 震动可调 | attributes/boolean | 否 | 4 | 明确强度/模式/分区可调；与震动存在、马达类型、数量分别记录。 |
| `dock_support` | 座充支持 | attributes/boolean | 否 | 3 | 支持座充与套餐含座充分开；明确另购不等于不支持。 |
| `dock_auto_power` | 座充自动休眠/唤醒 | attributes/boolean | 否 | 1 | 仅保留K10原文明确自动休眠/唤醒；九尾狐智能启停不补猜这些行为。 |
| `dock_hub` | 底座HUB拓展 | attributes/boolean | 否 | 1 | K10明确可作扩展坞；不推广为所有底座或所有USB功能。 |
| `headset_jack` | 耳麦接口 | attributes/boolean | 否 | 2 | 明确3.5mm耳机/耳麦孔支持；USB充电口不证明音频能力。 |
| `nfc` | NFC功能 | attributes/boolean | 否 | 1 | 以明确NFC刷Amiibo原文为依据，不根据NEC OCR猜测。 |
| `console_wake` | 主机唤醒 | attributes/boolean | 否 | 2 | 明确Switch/NS主机唤醒保留；手柄自身回连和对象不明唤醒不纳入。 |
| `multiplayer_controllers` | 多手柄同时连接 | attributes/boolean | 否 | 3 | 原文明示多手柄同时连接，保留平台/接收器/分线器条件，不推出无限数量。 |
| `rgb_lighting` | RGB灯效 | attributes/boolean | 否 | 3 | 保留明确RGB声明，普通灯条/指示灯不归RGB。 |
| `lighting_adjustable` | 灯光可调 | attributes/boolean | 否 | 5 | 是否可调是布尔标准；5/6种冲突不否定能力，只禁止确定档数的推断。 |
| `replaceable_faceplate` | 面壳可更换 | attributes/boolean | 否 | 1 | 明确磁吸可替换面壳，颜色SKU不证明可更换。 |
| `replaceable_stick_caps` | 摇杆帽可更换 | attributes/boolean | 否 | 2 | 明确可拆卸/替换摇杆帽或高度，原文部件范围保留。 |
| `extra_programmable_buttons` | 扩展编程按键 | attributes/boolean | 否 | 3 | 明确扩展编程按键能力；实体按键数量本身不证明可编程。 |
| `extra_button_count` | 扩展按键数量 | attributes/numeric | 否 | 2 | 明确背键/扩展键数量且带部件范围，不把全机总键数当扩展键。 |
| `back_button_lock` | 背键物理锁定/收纳 | attributes/boolean | 否 | 1 | 明确物理伸缩/上锁能力；不由物理锁推出电子键值锁，新旧版本保持区别。 |
| `button_layout_switch` | 按键布局切换 | attributes/boolean | 否 | 3 | 固定ABXY/预设布局/实体字符切换保留，不用于证明自定义映射。 |
| `voice_configuration` | 语音修改配置 | attributes/boolean | 否 | 2 | 语音修改配置是通用功能；软件名仅是实现上下文，保留后台与音频设备条件。 |
| `body_weight` | 机身重量 | attributes/numeric | 是 | 6 | 保留机身质量及SKU；底座62g不纳入机身，不预设更轻一定更好。 |
| `body_dimensions` | 机身尺寸 | attributes/categorical | 是 | 5 | 机身三轴尺寸原文保留，包装尺寸排除；Q37轴次已由用户看原图确认。 |
| `battery_capacity` | 电池容量 | attributes/numeric | 是 | 5 | 保留自身电池容量及SKU，不推算续航、供电模式或把无电池填0。 |
| `power_source` | 供电方式 | attributes/categorical | 是 | 6 | 收窄到证据支持的电池/USB来源；容量不自动证明内置、可拆卸、可充电。 |
| `wireless_range` | 无线传输距离 | attributes/numeric | 否 | 3 | 保留距离及协议条件；≤8m是上界，不倒置为至少8m，不把无延迟当0ms。 |
| `standby_time` | 待机时间 | attributes/numeric | 否 | 2 | 满电静置30天是待机量，不能替代持续游戏续航。 |
| `cable_length` | 连接线长度 | attributes/numeric | 否 | 2 | 保留2m/2.5m线长，不转成无线距离或设更长更好。 |
| `detachable_cable` | 连接线可拔插 | attributes/boolean | 否 | 2 | 明确拔线/可拔插设计，Type-C充电口本身不作依据。 |
| `material` | 外壳/接触材料 | attributes/categorical | 否 | 3 | 按外壳/按键/接触部位记录ABS、硅胶、软橡胶，不把局部用料推成整机。 |
| `anti_slip_grip` | 防滑设计 | attributes/boolean | 否 | 3 | 明确防滑纹理/橡胶设计支持布尔能力；舒适或漂亮宣传不转为量化性能。 |

## 使用边界

本次完成AI依据用户规则的复审，没有代填61项人工标签或把整个gold/overlay标为approved。历史Dataset、队列、run未修改。新旧schema与证据哈希变化，应在同一新用例版本上重跑baseline和candidate。人工原始回答见human-review-decisions.json；逐条来源检查见gold-rubric-review.json。
