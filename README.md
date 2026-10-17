
基于蛋白语言模型与机器学习的 UGT91C1 糖基转移酶突变体智能设计与活性优化

## 一、项目背景

糖基转移酶是一类能够催化糖基从供体分子转移到底物分子上的重要酶，在天然产物修饰、药物合成、功能食品开发以及合成生物学中具有广泛应用。通过糖基化修饰，底物分子的水溶性、稳定性、生物利用度和生物活性往往能够得到显著改善，因此糖基转移酶的研究在生物化工领域具有重要意义。

UGT91C1 是来源于植物的一种 UDP-糖基转移酶，能够参与特定天然产物的糖基化修饰过程，在高值产物生物合成和天然甜味剂改造等方向具有较大应用潜力。然而，天然来源的酶往往存在催化效率有限、底物适应性不足、工业应用条件下稳定性不够等问题。传统酶工程通常采用“经验选点—构建突变体—逐个验证”的方式进行优化，这种方法虽然可行，但存在筛选周期长、实验成本高、成功率低等不足。

随着人工智能和蛋白质计算设计的发展，利用蛋白序列大数据训练得到的蛋白语言模型，已经能够从序列中提取包含进化和结构约束信息的高维特征，为小样本酶工程问题提供了新的解决思路。本项目拟将蛋白语言模型 ESM 与机器学习算法 XGBoost 相结合，构建 UGT91C1 突变体活性预测模型，用于指导下一轮突变设计与实验验证，从而形成“预测—验证—再学习”的闭环优化流程，提高酶工程筛选效率。

## 二、项目目标

本项目以 UGT91C1 为研究对象，建立一套面向糖基转移酶改造的机器学习辅助设计平台。项目的总体目标包括以下几个方面：

建立 UGT91C1 突变体标准化数据集，整理 WT 及各类突变体的序列、比酶活或相对活性等实验数据。

利用蛋白语言模型 ESM 对 UGT91C1 及其突变体序列进行表征，提取高维 embedding 特征。

基于已有实验数据训练 XGBoost 回归模型，学习“序列特征—酶活性”之间的关系。

构建候选突变体库，对可能有利的位点进行单点或组合突变扫描，并对候选进行活性预测与优先级排序。

对预测结果进行实验验证，筛选获得活性提升的突变体。

将新增实验结果回灌至模型中，形成迭代优化机制。

项目最终希望实现：在减少实验盲目筛选数量的前提下，提高获得高活性突变体的成功率，为后续糖基转移酶定向进化与工业应用提供方法支撑。

## 三、研究意义

1. 理论意义

本项目探索将蛋白语言模型应用于糖基转移酶的小样本酶工程问题，验证序列表征学习在特定酶活性预测中的有效性。该研究有助于推动生物化工领域中“数据驱动酶设计”的方法发展。

2. 应用意义

UGT91C1 的活性提升有助于提高目标产物的糖基化转化效率，降低生物催化成本，对天然产物修饰、甜味剂改造、功能化合物生物制造等方向具有应用价值。

3. 方法学意义

本项目建立的是一种可推广的技术路线，不仅适用于 UGT91C1，也可扩展到其他糖基转移酶、氧化还原酶、水解酶等酶类的理性改造与智能筛选。

## 四、项目核心思路

本项目的核心思想可以概括为一句话：

利用人工智能模型学习已有突变体数据中的规律，从而优先推荐最可能提高活性的突变体，减少传统试错式实验筛选的工作量。

传统酶改造往往依赖研究者经验选取少量位点进行突变，而本项目则通过机器学习模型在大量潜在突变方案中进行初步筛选，把原本需要做几十个甚至上百个突变体的工作，缩减到只优先验证若干个最有希望的候选，从而显著提升效率。

## 五、技术路线

本项目的技术路线分为六个主要模块：

1. 数据收集与标准化

收集 UGT91C1 WT 及突变体的实验数据，包括突变体名称、突变位点、完整序列、比酶活或相对活性等信息，并统一实验条件和数据格式，构建训练数据集。

2. WT 序列确定与突变体序列生成

从实验实际使用的 UGT91C1 构建出发，确定 WT 蛋白全长氨基酸序列。根据 mutation_list 自动将突变施加到 WT 序列中，生成每个突变体的完整蛋白序列。

3. ESM 序列表征

将 WT 及突变体完整序列输入蛋白语言模型 ESM，得到每条序列的 embedding 向量，用于表征该蛋白在进化和结构约束空间中的特征。

4. 机器学习建模

以 ESM embedding 作为输入，以实验测得的活性数据作为输出，利用 XGBoost 训练监督学习模型，并通过交叉验证评估模型性能。

5. 候选突变体设计与筛选

基于文献、结构分析及模型预测结果，选择关键位点构建候选突变体库，并用模型预测各候选的潜在活性，按预测值进行排序，选择 Top 突变体进行实验验证。

6. 实验验证与模型迭代

通过分子生物学和酶学实验构建并检测候选突变体，将新测得的数据补充回数据集，再次训练模型，实现模型与突变体性能的同步优化。

## 六、技术原理
1. 蛋白语言模型 ESM 的原理

ESM 是一种基于深度学习的蛋白质语言模型，其训练方式与自然语言处理中的预训练模型类似。它通过在海量蛋白序列上进行自监督学习，学会识别序列中的规律，例如哪些位点更保守、哪些氨基酸替换更可能保持结构稳定、哪些局部模式与特定功能相关。

在本项目中，ESM 不直接输出“活性高低”，而是将蛋白序列转化为一个高维向量。这个向量可看作该蛋白的“数字化特征表示”，其中已经包含了大量潜在的进化与结构信息。

2. XGBoost 的原理

XGBoost 是一种梯度提升树算法，适合处理小样本、非线性、高维特征的监督学习问题。它通过迭代训练多棵决策树，不断修正前一轮预测误差，从而建立一个性能较强的回归模型。

在本项目中，XGBoost 的输入是 ESM embedding，输出是实验活性值。模型学习的本质是“从序列表征中寻找影响酶活性的规律”。

3. 闭环优化原理

本项目并不是一次性建完模型后就结束，而是强调“预测—验证—回灌”的迭代。随着新实验数据不断加入，模型会越来越贴近 UGT91C1 这一具体体系的真实规律，预测结果也会逐步提高可靠性。

## 七、项目创新点

本项目的创新点主要体现在以下几个方面：

1. 将蛋白语言模型引入糖基转移酶改造

相较于传统依赖人工设计理化特征的方法，本项目利用 ESM 自动学习蛋白序列中的潜在信息，减少人工经验依赖。

2. 面向小样本场景的监督学习建模

糖基转移酶实验数据数量有限，本项目选用小样本友好的 XGBoost 算法，适配酶工程中常见的数据规模。

3. 构建酶工程闭环优化体系

项目不局限于“建一个模型”，而是建立从数据、模型、筛选到实验验证的完整闭环流程，具有较强的工程应用价值。

4. 强调生物化工应用导向

本项目以实际酶活提升和转化效率改善为目标，兼顾模型预测能力与实验可落地性，符合生物化工方向大创项目特点。

## 八、项目实施内容
第一阶段：基础数据整理

确定 UGT91C1 WT 蛋白序列

整理已有突变体活性数据

建立 mutants.csv 数据表

第二阶段：模型构建

提取突变体 ESM embedding

训练 XGBoost 模型

使用交叉验证评估模型预测性能

第三阶段：候选设计

围绕关键位点构建单点或组合突变候选

用模型预测新候选活性

选择优先实验对象

第四阶段：实验验证

构建候选突变体

测定比酶活、转化率等指标

对照 WT 分析提升效果

第五阶段：迭代优化

将实验结果补充至训练集

再训练模型

进行下一轮筛选与验证

## 九、预期成果
本项目预期取得以下成果：

构建一套适用于 UGT91C1 的机器学习辅助酶工程流程。

建立标准化的突变体—活性数据集。

训练出可用于候选突变体排序的活性预测模型。

筛选获得若干活性提高的 UGT91C1 突变体。

形成项目代码、数据文档、技术报告及结题材料。

为后续论文撰写、竞赛展示或进一步产业应用奠定基础。

## 十、项目优势

与传统酶工程方法相比，本项目具有以下优势：

筛选效率更高：优先测试最有潜力的候选，而非盲目逐个验证。

适用于小样本：适合大创阶段的数据规模。

具有扩展性：后续可加入 docking、分子动力学、稳定性预测等模块。

兼具科研与应用价值：既能做方法研究，也能产出实际高活性酶突变体。

## 十一、项目局限性与风险

项目也存在一定局限：

初始样本量较少时，模型预测的泛化能力有限。

活性标签易受实验条件、批次差异影响。

模型学到的是统计关联，而非完整的物理机制。

若 WT 序列编号与实验突变编号不一致，会直接导致建模失败。

对应的解决办法包括：

严格统一实验条件；

每批设置 WT 对照并做相对活性归一化；

优先从单点突变开始，逐步扩展到组合突变；

确保使用实验实际构建对应的 WT 蛋白序列。

## 十二、总结

本项目围绕 UGT91C1 的活性优化问题，构建了一条结合蛋白语言模型与机器学习的智能酶工程路线。通过 ESM 对突变体序列进行高维表征，再用 XGBoost 学习序列与活性之间的关系，可以在小样本条件下初步实现突变体活性预测与候选排序。结合实验验证与数据回灌，该项目有望形成一套适用于糖基转移酶优化的通用方法，为生物化工领域中的酶工程研究提供新的技术思路。

## 十三、先验与硬约束接入说明

当前代码已支持在 ESM embedding 外，拼接三类先验特征，并在预测阶段执行稳定性硬过滤：

1. 进化先验（MSA / PSSM / 协变）
2. 结构先验（口袋 / 几何）
3. 稳定性先验（ΔΔG）

### 1) 初始化先验模板文件

运行：

```bash
python scripts/init_prior_templates.py
```

会生成：

- `data/priors/pssm.csv`
- `data/priors/coevolution.csv`
- `data/priors/residue_structure.csv`
- `data/priors/pocket.csv`
- `data/priors/ddg.csv`
- `data/prior_config.json`

### 2) 按需填充真实先验

- `pssm.csv`：每个 `pos` 对 20 个氨基酸的 PSSM 分数。
- `coevolution.csv`：`pos_i,pos_j,score` 协变强度。
- `residue_structure.csv`：每个残基到供体/受体/催化质心距离、ASA、二面角、二级结构。
- `pocket.csv`：口袋体积与疏水性（全局特征）。
- `ddg.csv`：可按 `variant` 或 `mutation` 提供 ΔΔG。
- `prior_config.json`：可配置 `active_sites`，用于“突变位点到活性位点”协变特征。

### 3) 训练与预测

训练：

```bash
python scripts/train_xgb.py
```

预测：

```bash
python scripts/predict.py
```

预测输出：

- `results/predictions.csv`：全部候选预测结果。
- `results/predictions_filtered_by_ddg.csv`：应用 ΔΔG 硬过滤后保留的候选。

说明：`scripts/predict.py` 里的 `DDG_THRESHOLD` 默认为 `2.0`，可按体系调整。

## 十四、UGT91C1 家族序列检索与样本扩增

当实验样本较少时，可以先检索 UGT91C1/UGT91 家族同源序列，再基于家族支持位点扩增候选库。

### 1) 抓取家族序列（UniProt）

```bash
python scripts/fetch_family_sequences.py --max-records 3000
```

输出：

- `data/family/ugt91_family_metadata.tsv`
- `data/family/ugt91_family_raw.fasta`
- `data/family/ugt91_family_dedup.fasta`

### 2) 基于家族支持突变扩增候选样本

```bash
python scripts/augment_candidates_from_family.py --top-k-alt-per-pos 3 --min-support 2
```

输出：

- `data/new_candidates_family.csv`：家族支持候选。
- `data/new_candidates_merged.csv`：与原 `new_candidates.csv` 去重合并后候选。

说明：

- 当前版本使用全局比对将同源序列映射到 WT 位点，不要求同源序列与 WT 等长。
- `scripts/predict.py` 会优先读取 `data/new_candidates_merged.csv`（若存在），否则回退到 `data/new_candidates.csv`。

## 十五、用家族特征 + mutants 实验数据预测突变体活性

### 1) 从家族序列构建进化先验（PSSM + 协变）

```bash
python scripts/build_family_evo_priors.py --min-mi 0.01 --top-k 30000
```

输出：

- `data/priors/pssm.csv`
- `data/priors/coevolution.csv`

### 2) 用 mutants 实验数据训练并预测候选活性

```bash
python scripts/train_predict_family_model.py
```

默认行为：

- 训练集：`data/mutants.csv`
- 候选集：优先 `data/new_candidates_merged.csv`，不存在则用 `data/new_candidates.csv`
- 目标列：优先 `activity_rel`，否则 `activity`

输出：

- `models/xgb_family_priors.pkl`
- `results/predictions_family_model.csv`
- `results/predictions_family_model_filtered_by_ddg.csv`
- `results/family_model_cv_metrics.csv`

## 十六、项目预测过程逻辑

本节描述当前项目从“候选突变体”到“预测结果文件”的完整链路，便于排查结果差异与口径问题。

### 1) 任务定义与两类数据

项目里有两类不同用途的数据：

1. 训练/验证数据：`data/mutants.csv`  
	含实验标签（`activity` 或 `activity_rel`），用于训练模型和做交叉验证。
2. 候选预测数据：`data/new_candidates.csv` 或 `data/new_candidates_merged.csv`  
	通常无实验标签，仅用于模型打分排序。

关键点：

- 5 折交叉验证衡量的是模型在 `mutants.csv` 上的泛化能力。
- `results/predictions.csv` 是对候选库的外推预测，不能直接当作“验证集误差”。

### 2) 候选集选择逻辑

在 `scripts/predict.py` 中，候选输入优先级为：

1. 若存在 `data/new_candidates_merged.csv`，优先使用它。
2. 否则回退到 `data/new_candidates.csv`。

因此，同一模型在不同时间预测，若候选文件不同，输出分布会明显变化。

### 3) 单条样本如何进入模型

每个候选变体会经过两路特征构建：

1. ESM 序列表征（高维）
	- 对 `sequence` 跑 ESM2（`scripts/predict.py` 中调用 `esm2_t33_650M_UR50D`）。
	- 对 token 表征做平均池化，得到每条序列 embedding。
2. 先验特征（低维）
	- 通过 `scripts/prior_features.py` 的 `PriorFeatureBuilder` 构建进化/结构/稳定性先验。
	- 先验来源于 `data/priors/*.csv` 与 `data/prior_config.json`。

最终输入是拼接特征：

- `X_all = [ESM embedding, prior features]`

即“序列语义 + 生物先验”的联合表示。

### 4) 模型推理与输出字段

模型文件 `models/xgb_esm.pkl` 由 `scripts/train_xgb.py` 训练并保存。预测时：

1. 读取模型包（含 XGBoost 模型与先验维度信息）。
2. 对候选 `X_all` 做回归预测，生成 `pred_activity`。
3. 依据 WT 活性换算得到 `pred_activity_rel`，并按 `pred_activity_rel` 降序排序。

输出到：

1. `results/predictions.csv`：所有候选。
2. `results/predictions_filtered_by_ddg.csv`：执行 ΔΔG 硬过滤后的候选。

说明：

- `pred_activity`：绝对活性预测值（保留用于兼容旧流程）。
- `pred_activity_rel`：与 `mutants.csv` 同口径的相对活性预测值（推荐用于排序与对比）。

### 5) ΔΔG 硬过滤逻辑

`scripts/predict.py` 中默认参数：

- `DDG_THRESHOLD = 2.0`
- `ALLOW_UNKNOWN_DDG = True`

含义：

1. 若样本可计算到 `ddg_total/ddg_max` 且超过阈值，则过滤。
2. 若缺少 ΔΔG（未知），在 `ALLOW_UNKNOWN_DDG=True` 时保留。

这会导致一个常见现象：

- 当 `ddg.csv` 缺失较多时，`predictions.csv` 与 `predictions_filtered_by_ddg.csv` 可能几乎相同。

### 6) 训练与验证口径（避免误差误读）

项目中常见两种标签口径：

1. 绝对活性：`activity`
2. 相对活性：`activity_rel`（通常相对 WT 归一化）

如果模型输出是绝对活性 `pred_activity`，则：

1. 与 `activity` 比较是同口径。
2. 与 `activity_rel` 直接比较是异口径，会出现“误差很大”的假象。

若需要对比 `activity_rel`，应先换算：

`pred_activity_rel = pred_activity / WT_activity`

### 7) 为什么候选文件看起来“误差难评估”

`results/predictions.csv` 的多数行是新候选，通常在 `mutants.csv` 中没有真值。  
因此其准确度只能通过“重叠样本”评估，而不是全表评估。

建议评估顺序：

1. 先看 5 折 CV（训练集内泛化）。
2. 再看预测文件与 mutants 的重叠样本误差（若重叠很少，仅作参考）。
3. 最终以实验回测（湿实验）作为外推候选的真实性能判据。

### 8) 推荐的结果解读流程

对每轮预测，建议按以下顺序做决策：

1. 从 `results/predictions_filtered_by_ddg.csv` 取 Top N。
2. 检查 `design_tier`（优先从较低阶组合开始验证，风险更可控）。
3. 检查 `family_support_sum`（优先家族支持高的方案）。
4. 合并实验可行性约束（位点可构建性、结构风险、成本）。
5. 输出小批量候选进入实验，实验结果再回灌训练集。

这对应项目的闭环：

预测 -> 实验 -> 回灌 -> 重训 -> 再预测

---

## 十七、突变体活性预测平台说明与启动流程

本项目已集成一个基于 Streamlit 的前端预测平台，用于将 UGT91C1 突变体活性预测流程可视化、交互化。平台面向四类使用场景：

1. 直接用现有实验数据 + 候选库，给出下一批建议实验的突变体序列并导出（候选推荐）。
2. 上传候选突变体 CSV，自动完成零样本排序、活性预测、候选排序和结果导出。
3. 拿到新一批实验数据后，把数据并入训练集并自动重训，人工确认后采纳新模型（数据入库）。
4. 查看已有 prediction 结果，对 Top 候选、预测分数分布、热点突变位点和家族支持关系进行可视化分析；并把预测值与实测值对比评估模型好坏（对比分析）。

### 1）平台主要功能

预测平台入口：

```text
http://localhost:8501
```

侧边导航栏共六个页面，「候选推荐」为默认首页，「平台说明」放最后：

- **候选推荐**：不依赖上传，对候选库统一打分后直接给出下一批建议实验的序列；
- **候选预测**：上传 CSV 后执行零样本排序 + 模型预测，并可导出结果；
- **数据入库**：把新批次实验数据并入训练集并自动重训，人工确认后采纳新模型；
- **对比分析**：预测值与实测值并排对比，评估模型泛化能力与本轮推荐命中率；
- **结果浏览**：读取 `results/` 目录下已有预测结果并可视化；
- **平台说明**：平台定位、当前模型配置、按位点留出的泛化能力指标与使用须知、数据来源与评测协议、六套方案对比与后续优先级（由原「概览」与「模型与数据」两页合并）。

语言与主题控件位于侧边栏导航菜单**上方**（不占用页面正文），提供两个开关（默认简体中文 + 白天）：

- **中英文切换**（`中文 / English`）：切换后导航标题与全部界面文案立即刷新，词条集中在 `i18n.py`；未收录的字符串自动回退中文，不会因漏译报错。
- **白天 / 夜间模式**：两套「黑—蓝—金」配色定义在 `theme.py`。切换分两层：先由 `apply_native_theme()` 把主题写进 Streamlit 运行时配置（内置表格、上传框、下拉、提示条等组件换成深色底 + 浅色字），再注入 CSS 变量统一页面背景、指标卡、表格与图表的配色，以保证夜间模式下不出现「深底深字」。

「候选推荐」页面读取 `results/candidate_recommendations.csv`（由
`python scripts/build_candidate_recommendations.py` 生成），把
`results/zeroshot_candidate_ranking.csv` 与 `results/predictions.csv` 两个候选池按突变体去重合并后统一打分：

- 候选的突变位点若在训练数据中出现过（已见非背景位点），交给部署模型预测；
- 未出现过的位点交给零样本 masked-marginal 分数；
- 两个分数尺度不同，按**名次百分位**加权融合成「融合名次分」（`w_model = 已见非背景位数 / 非背景位点总数`）；
- 页面支持 Top-N 选择、按突变类型（单突变 / 组合）与位点覆盖筛选，并可导出推荐序列 CSV。

注意：组合突变叠加仍是最弱环节，且零样本分数是各突变贡献之和，突变数越多该值天然越大，纯按零样本分排序时 3~4 个突变的组合容易排到前面，建议配合筛选器按需取用。


「候选预测」页面支持：

- 上传 CSV 候选突变体文件；
- 自动读取 `variant`、`sequence`、`activity_rel` 等字段，`sequence` 留空会按 WT 序列自动补齐；
- 构建模型特征；
- 计算 masked-marginal 零样本分数，作为候选筛选的第一道排序器；
- 对带标签数据执行五折交叉验证；
- 对无标签候选数据执行活性预测排序；
- 显示预测流程真实进度；
- 展示 Top 候选预测分布；
- 导出预测结果 CSV。

「结果浏览」页面用于读取 `results/` 目录下已有预测结果，包括：

- `results/zeroshot_candidate_ranking.csv`
- `results/predictions.csv`
- `results/predictions_family_model_filtered_by_ddg.csv`
- `results/预测结果-家族支持候选.csv`

可视化内容包括：

- Top 候选排序柱状图；
- 预测分数分布直方图；
- 热点突变位点频次图；
- 家族支持与预测值散点图；
- Top 候选明细表。

「数据入库」页面用于把新一批实验数据并入统一训练集并重训模型，流程为：

1. 上传新批次 CSV，最小格式只需 `variant` 与 `activity` 两列（WT 行写 `WT`）；
2. 平台按**批内 WT** 归一化算出 `activity_rel = activity / wt_batch`，并把突变写法与 WT 序列逐位点核对（WT 残基对不上直接报错），补齐 `mutation_list / sequence / sites / sites_str / n_mut / is_background_only`；
3. 勾选确认后执行入库：先用当前线上模型给这批突变体打一份「入库前预测」快照写入 `results/active_learning_log.csv`，再把数据追加进 `data/train_dataset.csv`（追加前自动备份为 `data/train_dataset.csv.bak-<时间戳>`）；
4. 重跑 `scripts/extract_features_v2.py` 整体重建 ESM 特征缓存（行序必须与新的训练集逐行对齐，不能只 append）；
5. 重跑 `scripts/retrain_deploy_model.py` 产出**候选模型**到 `models/candidates/run_<时间戳>/`，并同时输出 `report.json` 与 `oof.csv`；
6. 页面上并排展示「现役模型 vs 候选模型」的按位点留出指标，由人工点「采纳新模型」才会替换 `models/xgb_esm.pkl`（旧模型归档进 `models/history/`）；丢弃候选不会删除已入库的实验数据。

页面提供一个「自动调优（小网格搜索）」开关：默认关闭时沿用线上配方快速重训；打开后额外试 4 组超参，在同一数据集、同一协议下按按位点留出的 Macro Spearman 择优（并列看 Macro AUC）。

「对比分析」页面分两个板块：

- **板块一 · 全量按位点留出的折外预测 vs 实测**：读取 `models/active/report.json` 与 `models/active/oof.csv`，给出 Macro Spearman / Macro AUC / Recall@10、预测 vs 实测散点（含 y = x 参考线）、残差分布，以及按批次、按突变数分层的指标表。首个报告可用下面这条命令为现役模型补算（只写报告、不改变模型本身）：
  ```powershell
  python scripts/retrain_deploy_model.py --eval-only --model models/xgb_esm.pkl --report-dir models/active
  ```
- **板块二 · 本轮新增数据：入库前预测 vs 实测**：读取 `results/active_learning_log.csv`，按轮次查看「入库前模型怎么预测」与实测活性的对比、Top-5 命中数与明细表。

两个板块口径不同，不可混读：板块一衡量外推到新位点的泛化能力，板块二衡量本轮推荐的命中率（单轮样本少，仅作趋势参考）。

报告文件统一命名为 `report.json` / `oof.csv`，用目录区分归属：`models/active/` 放现役模型的报告，`models/candidates/run_<时间戳>/` 放每次重训的候选模型与报告。采纳新模型时，旧模型会归档到 `models/history/xgb_esm_<时间戳>.pkl`，采纳记录写入 `models/training_history.jsonl`。

模型训练入口仍统一为 `scripts/retrain_deploy_model.py`（无参数运行即为原有的「覆盖写线上模型」行为），不要再使用其它旧训练脚本。

### 2）启动前准备

确认当前目录位于项目根目录：

```powershell
cd g:\ugt_ml
```

安装依赖：

```powershell
C:\Users\11716\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

如果环境中已经配置了 `python` 命令，也可以使用：

```powershell
python -m pip install -r requirements.txt
```

### 3）启动前端平台

在项目根目录运行：

```powershell
python -m streamlit run streamlit_app.py
```

启动成功后，终端会显示类似：

```text
Local URL: http://localhost:8501
```

然后在浏览器中打开：

```text
http://localhost:8501
```

组内多人共用一台公用机器时，请改用下一节的局域网部署方式。

### 4）组内局域网部署（多人共用一台机器）

给组内多人在同一台公用机器上用时，按下面三步走。

**第一步：设置组内共享口令**

把 `.streamlit/secrets.toml.example` 复制成 `.streamlit/secrets.toml`，把 `app_password` 改成组内约定的口令。
该文件已加入 `.gitignore`，不会进仓库；**不配置则不启用口令门**，只适合本地开发。

**第二步：一键启动**

在公用机器上双击 `start_server.bat`。脚本会依次完成：激活 conda 环境 `ugt91c1`（没有则回退到项目内 `.venv`）
→ 依赖自检 → 把 ESM 权重缓存固定在 `models/torch_cache/`（只下载一次，之后可离线）→ 打印本机可用地址
→ 以 `--server.address 0.0.0.0` 启动服务。

组内成员用脚本打印的地址访问，例如：

```text
http://192.168.1.20:8501
```

**第三步：放行防火墙**

公用机器首次启动时 Windows 会弹出防火墙询问，需勾选允许访问；若没弹窗，手动放行 TCP 8501 端口。

共享部署下的几条约定：

- **同时只允许一人重训**：「数据入库」页用 `models/.retrain.lock` 做互斥，两人同时点会提示等待；进程异常退出留下的锁会在 30 分钟后自动失效。
- **服务关闭了文件监视**：启动脚本传了 `--server.fileWatcherType none`，避免入库写数据时把所有人的会话反复重跑；改完代码需要重启脚本才生效。
- **口令门**：登录口令由 `auth.py` 在进入导航前校验，定位是「防组外人员误入」的轻量门禁，不是强安全边界。
- **GPU 可选**：环境里装了 NVIDIA 驱动 + CUDA 版 torch 时会自动用 GPU（启动日志会打印 `CUDA 可用: True`）；默认从 PyPI 装的 torch 是 CPU 版，纯 CPU 也能跑，只是零样本打分慢一些。
- **主题与语言是会话级的**：多人各自切换互不影响，刷新页面回到默认的「中文 + 白天」。

页面脚本位于 `app_pages/`，共享业务逻辑集中在 `prediction_core.py`，展示辅助函数在 `ui_helpers.py`，
中英文词条在 `i18n.py`，白天 / 夜间配色在 `theme.py`。
端口与默认配色在 `.streamlit/config.toml` 中配置；运行时由 `theme.py` 的 `apply_native_theme()` 覆盖（切换主题无需改配置文件，刷新页面后回到默认的白天模式）。

### 5）上传数据格式

最小可用 CSV 只需要包含 `variant` 列：

```csv
variant,sequence
F208M,
F208M;V129A,
M15I,
```

如果希望平台根据本次上传数据计算五折预测精准度，需要同时提供真实活性标签：

```csv
variant,sequence,activity_rel
F208M,,2.00
F208M;V129A,,3.28
M15I,,1.25
```

字段说明：

- `variant`：突变体名称，例如 `F208M`、`F208M;V129A`；
- `sequence`：突变体完整氨基酸序列，可为空；
- `activity_rel`：相对活性标签，用于本次上传数据的五折交叉验证；
- `activity`：绝对活性标签，也可用于五折交叉验证。

如果 CSV 不包含 `activity_rel` 或 `activity`，平台只进行候选预测排序，不计算本次数据的精准度。

### 6）六步预测流程

平台中的六步流程对应真实后端执行阶段：

1. 读取 CSV  
   检查文件格式，读取候选突变体数据；`sequence` 留空时按 WT 序列自动补齐。

2. 构建特征  
   生成 ESM-2 序列表征 1280 维，并叠加家族先验、进化先验等 23 维特征。

3. 零样本排序  
   对 WT 序列逐位点 mask，以 Σ[log p(突变) − log p(WT)] 打分，作为第一道排序器。

4. 五折预测  
   如果上传文件包含足够的真实活性标签（≥5 条），则执行 5 折交叉验证；否则跳过数值精度评估。

5. 候选排序  
   按零样本分数降序给出名次，并同时给出模型预测名次作为第二道参考。

6. 导出报告  
   生成可下载的预测结果 CSV，并在页面中展示图表和表格。

### 7）结果解读

平台输出的排序结果应理解为模型推荐顺序，而不是最终实验结论。

**仅按名次取前 N 个候选，勿用预测绝对值做阈值筛选。** 该模型是排序器而非绝对活性测量仪：
按位点留出评测下排序能力较强（Macro Spearman 0.314、Macro AUC 0.663），但 RMSE 约 0.974，
预测绝对值只反映数值拟合水平，不能直接当作实验活性使用。

推荐使用方式：

1. 先按名次查看 Top 候选；
2. 再结合家族支持、突变数量、位点合理性和实验构建难度；
3. 选择少量候选进入下一轮实验验证；
4. 将实验结果回填训练集，用于后续模型迭代。

因此，预测平台的作用是减少盲目筛选范围，为实验设计提供优先级参考。
