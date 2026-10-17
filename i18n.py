"""平台中英文文案与语言状态。

两种用法：
  * ``t(text)``  —— 短文案，以中文原文为键查英文表，``zh`` 直接返回原文，
    未收录的字符串回退中文，保证不会因漏译而报错。
  * ``tb(key, **kw)`` —— 命名块，用于多行段落、HTML 提示块与含占位符的模板，
    在 ``BLOCKS`` 里按 ``key`` 存放 ``zh`` / ``en`` 两版文案。
"""

from __future__ import annotations

import streamlit as st

DEFAULT_LANG = "zh"
# 语言选择控件里「显示文本 -> 语言码」的对应关系。
LANG_OPTIONS = {"中文": "zh", "English": "en"}


def lang() -> str:
    """当前语言码（``zh`` / ``en``）。"""
    return st.session_state.get("lang", DEFAULT_LANG)


def t(text: str) -> str:
    """短文案翻译：中文原文 -> 当前语言。"""
    if lang() == "zh":
        return text
    return EN.get(text, text)


def tb(key: str, **kw) -> str:
    """命名块翻译：按 ``key`` 取当前语言整段文案并用 ``kw`` 填充占位符。"""
    block = BLOCKS.get(key)
    if block is None:
        return key
    text = block.get(lang()) or block.get("zh", key)
    return text.format(**kw) if kw else text


# --------------------------------------------------------------------------- #
# 短文案表（中文 -> 英文）
# --------------------------------------------------------------------------- #
EN: dict[str, str] = {
    # ---- 导航 / 全局 ----
    "平台说明": "Platform Guide",
    "候选推荐": "Candidate Picks",
    "候选预测": "Candidate Scoring",
    "数据入库": "Data Ingestion",
    "对比分析": "Comparison",
    "结果浏览": "Results",
    "语言": "Language",
    "夜间模式": "Dark mode",
    # ---- 指标卡标签 ----
    "预测目标": "Prediction target",
    "特征数量": "Feature count",
    "训练数据": "Training data",
    "回归器": "Regressor",
    "候选总数": "Total candidates",
    "可模型预测": "Model-scorable",
    "已见位点候选": "Seen-site candidates",
    "最高融合分": "Top fused score",
    "候选数量": "Candidates",
    "最高分": "Top score",
    "中位分": "Median",
    "平均分": "Mean",
    "最高零样本分数": "Top zero-shot score",
    "中位零样本分数": "Median zero-shot score",
    "五折 RMSE": "5-fold RMSE",
    "五折 MAE": "5-fold MAE",
    "参与样本": "Samples",
    "评估突变体数": "Variants scored",
    "轮次": "Round",
    "突变体数": "Variants",
    "Top-5 命中": "Top-5 hits",
    # ---- 表格列名 / 控件标签 ----
    "突变体": "Variant",
    "突变列表": "Mutations",
    "突变数": "Mutations",
    "位点": "Sites",
    "位点覆盖": "Site coverage",
    "融合名次分": "Fused rank score",
    "模型预测活性": "Model activity",
    "零样本分数": "Zero-shot score",
    "家族支持度": "Family support",
    "来源候选池": "Source pool",
    "全局名次": "Global rank",
    "推荐名次": "Recommended rank",
    "名次": "Rank",
    "零样本名次": "Zero-shot rank",
    "模型名次": "Model rank",
    "预测活性": "Predicted activity",
    "真实相对活性": "True relative activity",
    "真实活性": "True activity",
    "预测误差": "Prediction error",
    "排序分数": "Ranking score",
    "排序依据": "Ranked by",
    "来源": "Source",
    "同源池": "Homolog pool",
    "零样本总分": "Zero-shot sum",
    "零样本最低": "Zero-shot min",
    "零样本均值": "Zero-shot mean",
    "批次": "Batch",
    "样本数": "Samples",
    "折外预测": "Out-of-fold",
    "实测": "Measured",
    "实测减预测": "Measured − predicted",
    "入库前预测": "Pre-ingest",
    "原推荐名次": "Original rank",
    "残差": "Residual",
    "特征数": "Features",
    "方案": "Scheme",
    "文件名": "File",
    "说明": "Description",
    "状态": "Status",
    "已就绪": "Ready",
    "文件缺失": "Missing",
    "推荐数量 Top-N": "Top-N count",
    "突变类型": "Mutation type",
    "全部": "All",
    "单突变": "Single",
    "组合突变": "Combo",
    "结果来源": "Result source",
    "查看范围": "View range",
    "全部轮次": "All rounds",
    "上传候选 CSV": "Upload candidate CSV",
    "开始预测": "Run prediction",
    "下载模板": "Download template",
    "下载结果 CSV": "Download results CSV",
    "下载完整结果 CSV": "Download full results CSV",
    "下载折外预测明细 CSV": "Download out-of-fold CSV",
    "下载本轮对比明细 CSV": "Download round comparison CSV",
    "下载新批次模板 CSV": "Download new-batch template CSV",
    "入库并重训": "Ingest & retrain",
    "采纳新模型（替换线上）": "Adopt new model",
    "丢弃候选模型": "Discard candidate",
    "批次标签": "Batch label",
    "自动调优（小网格搜索）": "Auto tuning (grid search)",
    "新批次 CSV（需含 variant 与 activity 列）": "New batch CSV (needs variant & activity)",
    "新批次 CSV 格式（点开查看 / 下载模板）": "New batch CSV format (view / download template)",
    "批内 WT 比酶活（CSV 已含 WT 行时留 0 即可）": "Within-batch WT activity (0 if CSV has a WT row)",
    # ---- 图表轴 / 图例标题 ----
    "分数区间": "Score bin",
    "出现次数": "Count",
    "突变位点": "Mutation site",
    "候选突变体": "Candidate variant",
    "融合名次分（0~1，越高越优先）": "Fused rank score (0–1, higher = better)",
    "模型预测活性（相对 WT）": "Model activity (relative to WT)",
    "预测 activity_rel": "Predicted activity_rel",
    "实测 activity_rel": "Measured activity_rel",
    "入库前预测 activity_rel": "Pre-ingest predicted activity_rel",
    "残差（预测 − 实测）": "Residual (predicted − measured)",
    "残差区间": "Residual bin",
    "样本数量": "Samples",
    "覆盖类型": "Coverage type",
    # ---- 分块标题 ----
    "数据": "Data",
    "评测协议": "Evaluation protocol",
    "方案对比": "Scheme comparison",
    "后续优先级": "Next priorities",
    "当前模型": "Current model",
    "泛化能力（按位点留出）": "Generalization (leave-one-site-out)",
    "预测流程": "Prediction pipeline",
    "页面导航": "Page navigation",
    "已知短板": "Known weaknesses",
    "实验批次": "Batches",
    "批间差异": "Inter-batch difference",
    "背景位点": "Background site",
    "推荐优先级（融合名次分）": "Recommendation priority (fused rank score)",
    "整库位点覆盖分布": "Whole-pool site coverage",
    "模型预测 vs 零样本分数（按位点覆盖着色）": "Model vs zero-shot score (colored by coverage)",
    "推荐明细": "Recommendation details",
    "Top 16 候选": "Top 16 candidates",
    "零样本分数分布": "Zero-shot score distribution",
    "本次上传数据的数值精度（参考）": "Numerical accuracy on this upload (reference)",
    "候选明细": "Candidate details",
    "折外预测明细": "Out-of-fold details",
    "板块一 · 全量按位点留出的折外预测 vs 实测": "Panel 1 · Full leave-one-site-out predictions vs measured",
    "板块二 · 本轮新增数据：入库前预测 vs 实测": "Panel 2 · This round's new data: pre-ingest predictions vs measured",
    "预测 vs 实测（按批次着色）": "Predicted vs measured (colored by batch)",
    "入库前预测 vs 实测（按轮次着色）": "Pre-ingest predicted vs measured (colored by round)",
    "残差分布": "Residual distribution",
    "按批次分层": "By batch",
    "按突变数分层": "By mutation count",
    "明细": "Details",
    "分数分布": "Score distribution",
    "突变位点热区": "Mutation-site hotspots",
    "分数与家族支持度": "Score vs family support",
    "① 上传新批次数据并预览": "1. Upload and preview the new batch",
    "② 确认入库并重训": "2. Confirm ingestion and retrain",
    "③ 现役模型 vs 候选模型": "3. Live model vs candidate model",
    "模型采纳历史": "Model adoption history",
    "自动调优：各候选配方的按位点留出表现": "Auto tuning: leave-one-site-out by scheme",
    "候选模型的折外预测明细（前 200 行）": "Candidate model out-of-fold details (first 200 rows)",
    "使用须知": "Usage note",
    "结果来源": "Result sources",
    # ---- 提示 / 说明单行 ----
    "当前筛选条件下没有候选，请放宽条件。": "No candidates under the current filters; please relax them.",
    "当前筛选下没有同时具备模型预测与零样本分数的候选。": "No candidate has both a model prediction and a zero-shot score under the current filters.",
    "两个分数尺度不同、不可直接比较；点位越靠右上通常越值得优先实验。": "The two scores are on different scales and are not directly comparable; points toward the top-right are usually worth testing first.",
    "模型预测活性为 activity_rel 口径（WT = 1），仅用于名次参考，勿做绝对值阈值筛选。": "Model-predicted activity uses the activity_rel scale (WT = 1); use it for ranking only, never for absolute-value thresholding.",
    "还没有预测结果：上传 CSV 后点击「开始预测」。": "No prediction yet: upload a CSV and click 'Run prediction'.",
    "随机五折只衡量数值拟合，不代表候选排序能力；挑候选请以名次为准。": "Random 5-fold measures numerical fit only, not candidate ranking ability; rank by position when picking candidates.",
    "该结果文件不含 family_support 列。": "This result file has no family_support column.",
    "已隐藏 462 位的 sequence 列，避免表格被长文本撑开；完整字段请下载 CSV。": "The 462-char sequence column is hidden to keep the table narrow; download the CSV for all fields.",
    "还没有采纳过任何新模型。入库重训后点「采纳新模型」即会在此留下记录。": "No model adopted yet. Ingest & retrain, then click 'Adopt new model' to record an entry here.",
    "数据仍在训练集里。修好问题后重跑本页即可继续，无需重复入库。": "The data is still in the training set. Fix the issue and rerun this page; no need to re-ingest.",
    "数据仍在训练集里，修好问题后重跑本页即可。": "The data is still in the training set; fix the issue and rerun this page.",
    "已丢弃候选模型。已入库的实验数据保留在训练集里。": "Candidate model discarded. The ingested data stays in the training set.",
    "本批次只有 WT 行，无需快照。": "This batch has only a WT row; no snapshot needed.",
    "请上传新批次 CSV 后继续。": "Continue after uploading a new-batch CSV.",
    "没有可入库的新记录，已在上方说明原因。": "No new records to ingest; see the reason above.",
    "未找到 `results/tier_cv_metrics.csv`，跳过方案对比。": "`results/tier_cv_metrics.csv` not found; skipping scheme comparison.",
    "检测到已有其他人在执行「入库并重训」，请等其结束后再试。": "Another user is currently running ingestion & retrain. Wait for it to finish, then try again.",
    "没有可用的折外预测点。": "No usable out-of-fold points.",
    "报告里没有按批次的分层指标。": "The report has no batch-level metrics.",
    "报告里没有按突变数的分层指标。": "The report has no mutation-count metrics.",
    "当前范围内没有同时含预测与实测的记录。": "No records with both prediction and measured values in this range.",
    "数据不足，暂不绘制残差分布。": "Not enough data to draw a residual distribution.",
    "样本不足 3 条或该组内真值无方差时，分层指标留空（显示为 None），不做无意义的计算。": "When a group has fewer than 3 samples or no variance in the truth, stratified metrics are left blank (shown as None) rather than computed meaninglessly.",
    "`相对 WT 倍数` 才是进训练集的标签；WT 行自身恒为 1，作为本批次的归一化基准。": "Only the 'relative to WT' value enters the training set as the label; the WT row is always 1 and serves as this batch's normalization baseline.",
    "还没有折外预测报告，本板块暂时为空。": "No out-of-fold report yet; this panel is empty for now.",
    "结论：参数无关的零样本分数排序能力最强；位点窗口表征是负贡献；在 67 条数据规模下加训练头反而拖累排序（0.48 → 0.17）。": "Conclusion: the parameter-free zero-shot score ranks best; site-window representations hurt; with only 67 samples, adding a trained head drags ranking down (0.48 → 0.17).",
    "模型预测活性为 activity_rel 口径（WT = 1）。": "Model-predicted activity uses the activity_rel scale (WT = 1).",
    "解决后重跑本页即可。": "Fix it and rerun this page.",
    "五折 RMSE": "5-fold RMSE",
    "评估突变体数": "Variants scored",
    "已完成": "Done",
    "跳过": "Skipped",
    "按位点留出 Macro Spearman": "LSO Macro Spearman",
    "按位点留出 Macro AUC": "LSO Macro AUC",
    "RMSE（仅参考）": "RMSE (reference only)",
    "配方": "Scheme",
    "候选目录": "Candidate dir",
    "备注": "Note",
    "实测 − 预测": "Measured − predicted",
    "上传候选突变体 CSV，平台按「零样本分数 → 模型预测」的顺序输出候选名次。": "Upload a candidate-variant CSV; the platform outputs candidate ranks in the order 'zero-shot score → model prediction'.",
    # ---- 补充：说明 / 帮助文本 ----
    "仅按名次取前 N 个候选，勿用预测绝对值做阈值筛选": "Use rank order only; never threshold on absolute predictions",
    "ESM-2 序列表征 1280 维 + 进化 / 结构 / 稳定性先验 23 维": "ESM-2 sequence representation (1280-d) + evolutionary / structural / stability priors (23-d)",
    "两个实验批次按各自批内 WT 归一化后合并": "Two batches merged after within-batch WT normalization",
    "留出整个突变位点后外推，最贴近「挑候选突变体」的真实场景": "Extrapolate after holding out an entire site; closest to the real 'pick candidates' scenario",
    "区分「高于 WT」与「低于 WT」的能力": "Ability to separate 'above WT' from 'below WT'",
    "前 10 名候选覆盖真实高活性突变的比例": "Share of true high-activity variants covered by the top-10 picks",
    "只反映数值精度，挑候选时参考价值有限": "Reflects numerical accuracy only; limited value for picking candidates",
    "批次 A：WT = 0.021；批次 B：WT = 3.7": "Batch A: WT = 0.021; Batch B: WT = 3.7",
    "按各自批内 WT 归一化后再合并，避免引入批间偏差": "Merged after normalizing to each batch's own WT to avoid batch bias",
    "出现在几乎所有批次 B 变体中，评测与排序时剔除": "Present in almost all batch-B variants; removed during evaluation and ranking",
    "单点突变 + F208M 打底组合，按突变体去重合并": "Single mutations + F208M-anchored combos, deduplicated by variant",
    "至少含 1 个训练集已见非背景位点的候选": "Candidates with at least 1 seen non-background site",
    "含 ≥1 个已见位点，模型可直接外推": "Contains ≥1 seen site; the model can extrapolate directly",
    "融合名次分，越接近 1 越优先": "Fused rank score; closer to 1 = higher priority",
    "本页不依赖上传：平台从现有候选库里，对**已见位点用模型预测、未见位点用零样本分数**分别打分，再按名次百分位融合，直接给出下一批最值得做实验的序列。": "This page needs no upload: from the existing candidate pool the platform scores **seen sites with the model and unseen sites with the zero-shot score**, then fuses them by rank percentile to directly return the sequences most worth testing next.",
    "按批次分组后的平均排序相关性": "Average ranking correlation grouped by batch",
    "把 activity_rel > 1 视为高活性后的区分能力": "Discrimination treating activity_rel > 1 as high activity",
    "取前 10 个候选能覆盖多少真实高活性突变": "How many true high-activity variants the top-10 picks cover",
    "每次入库对应一轮": "Each ingestion round",
    "按预测取前 5 个，其中实测高于 WT 的个数": "Top-5 by prediction; count of those exceeding WT",
    "单轮样本少，仅作趋势参考": "Few samples per round; trend reference only",
    "相对 WT 倍数": "Relative to WT",
    "原始活性": "Raw activity",
    "突变位点": "Mutation sites",
    "仅背景位点": "Background only",
    "配置": "Scheme",
    "超参数": "Hyperparameters",
    "留出突变体数": "Held-out variants",
    "采纳时间": "Adopted at",
    "训练样本数": "Training samples",
    "归档旧模型": "Archived model",
    "条": "rows",
    "2 个": "2",
    "1.76 倍": "1.76×",
    # ---- 方案名 / 结果来源 / 位点覆盖 ----
    "零样本分数（当前排序器）": "Zero-shot score (current ranker)",
    "线上配方（mean-pool+先验）": "Live recipe (mean-pool + priors)",
    "零样本+先验+XGBoost": "Zero-shot + priors + XGBoost",
    "位点窗口+零样本+PCA": "Site-window + zero-shot + PCA",
    "位点窗口+零样本+XGBoost": "Site-window + zero-shot + XGBoost",
    "融合特征+PCA-Ridge": "Fused features + PCA-Ridge",
    "零样本分数排序（第一道排序器）": "Zero-shot score ranking (first ranker)",
    "参数无关的 masked-marginal 分数；按位点留出评测下排序能力最强。": "Parameter-free masked-marginal score; best ranking under leave-one-site-out evaluation.",
    "ESM + XGBoost 预测结果": "ESM + XGBoost predictions",
    "线上配方（ESM mean-pool + 先验特征 + XGBoost）对候选库的外推预测。": "Live recipe (ESM mean-pool + priors + XGBoost) extrapolated over the candidate pool.",
    "家族模型预测结果": "Family-model predictions",
    "家族进化先验模型结果，并已通过 ΔΔG 硬过滤。": "Family evolutionary prior model, hard-filtered by ΔΔG.",
    "家族支持候选结果": "Family-supported candidate results",
    "基于同源家族支持位点扩增出的候选及其打分。": "Candidates expanded from homolog family-supported sites and their scores.",
    "参数无关零样本分数": "Parameter-free zero-shot score",
    "线上配方外推预测": "Live-recipe extrapolation",
    "家族先验 + ΔΔG 过滤": "Family priors + ΔΔG filter",
    "家族支持位点扩增": "Family-supported site expansion",
    "已见位点": "Seen sites",
    "已见+未见": "Seen + unseen",
    "未见位点": "Unseen sites",
    "仅背景位点": "Background only",
    # ---- 核心模块：校验 / 流程 / 报错 ----
    "CSV 至少需要包含 variant 列，例如 F208M 或 F208M;V129A。": "The CSV must contain at least a variant column, e.g. F208M or F208M;V129A.",
    "没有找到有效的突变体记录。": "No valid variant records found.",
    "上传文件为空。": "The uploaded file is empty.",
    "结果文件里没有找到 pred_activity_rel / pred_activity_family / pred_activity / zs_nonbg_sum / zs_sum 列。": "The result file has none of the pred_activity_rel / pred_activity_family / pred_activity / zs_nonbg_sum / zs_sum columns.",
    "还没有候选推荐表，请先运行：python scripts/build_candidate_recommendations.py": "No recommendation table yet; run: python scripts/build_candidate_recommendations.py",
    "本次上传数据未提供至少 5 条可用的 activity_rel / activity 标签，无法计算数值精度。": "This upload provides fewer than 5 usable activity_rel / activity labels; numerical accuracy cannot be computed.",
    "有效标签少于 5 条，无法执行五折交叉验证。": "Fewer than 5 valid labels; 5-fold cross-validation cannot run.",
    "已基于本次上传数据完成五折交叉验证（随机折，仅衡量数值拟合）。": "5-fold cross-validation finished on this upload (random folds, measures numerical fit only).",
    "读取并规范化上传 CSV": "Read and normalize the uploaded CSV",
    "生成 ESM 序列表征": "Generate ESM sequence representations",
    "构建融合特征": "Build fused features",
    "计算零样本分数（第一道排序器）": "Compute zero-shot scores (first ranker)",
    "执行五折评估与模型预测": "Run 5-fold evaluation and model prediction",
    "生成候选名次": "Produce candidate ranks",
    "正在加载 ESM-2 与 XGBoost 模型…": "Loading ESM-2 and XGBoost models…",
}


# --------------------------------------------------------------------------- #
# 命名块（多行段落 / HTML 提示块 / 含占位符模板）
# --------------------------------------------------------------------------- #
BLOCKS: dict[str, dict[str, str]] = {
    # ---- 平台说明页（合并 概览 + 模型与数据）----
    "about_intro": {
        "zh": "面向 UGT91C1 糖基转移酶的突变体活性预测与候选排序工具：用现有实验数据，对候选库里的突变体统一打分，直接给出下一批最值得实验的序列。",
        "en": "A mutation-activity prediction and candidate-ranking tool for the UGT91C1 glycosyltransferase: it scores every variant in the candidate pool against existing experimental data and returns the sequences most worth testing next.",
    },
    "about_eval_caption": {
        "zh": "评测协议 {protocol} · {n_variants} 个变体 / {n_sites} 个突变位点",
        "en": "Protocol {protocol} · {n_variants} variants / {n_sites} mutation sites",
    },
    "about_usage_note": {
        "zh": (
            '<div class="ugt-callout">'
            "<strong>使用须知</strong>：{note}。"
            "平台的作用是把实验范围从几十个候选收敛到前 N 名，而不是给出可信的绝对活性。"
            "</div>"
        ),
        "en": (
            '<div class="ugt-callout">'
            "<strong>Usage note</strong>: {note}. "
            "The platform narrows the experimental scope from dozens of candidates down to the top N; "
            "it does not provide trustworthy absolute activities."
            "</div>"
        ),
    },
    "about_flow": {
        "zh": (
            "1. **读取候选库** 合并单点突变池与 F208M 打底的组合池（共两千余条），按突变体去重。\n"
            "2. **位点分流** 候选的突变位点若曾在训练数据中出现，交给模型；未出现则交给零样本分数。\n"
            "3. **模型预测** 已见位点候选：ESM-2 序列表征 1280 维 + 进化 / 结构 / 稳定性先验 23 维 → XGBoost。\n"
            "4. **零样本打分** 未见位点候选：对 WT 逐位点 mask，以 Σ[log p(突变) − log p(WT)] 打分。\n"
            "5. **名次融合** 两分数尺度不同，按名次百分位加权合并成「融合名次分」，越接近 1 越优先。\n"
            "6. **导出序列** 在「候选推荐」页选 Top-N 并下载下一批建议实验的突变体 CSV。\n"
        ),
        "en": (
            "1. **Read candidate pool** Merge the single-mutation pool and the F208M-anchored combo pool (2000+ entries) and deduplicate by variant.\n"
            "2. **Route by site** If a candidate's mutation site appears in the training data, send it to the model; otherwise use the zero-shot score.\n"
            "3. **Model prediction** Seen-site candidates: ESM-2 sequence representation (1280-d) + evolutionary / structural / stability priors (23-d) → XGBoost.\n"
            "4. **Zero-shot scoring** Unseen-site candidates: mask the WT position by position and score with Σ[log p(mutant) − log p(WT)].\n"
            "5. **Rank fusion** The two scores are on different scales; combine them by rank percentile into a 'fused rank score' (closer to 1 = higher priority).\n"
            "6. **Export sequences** On the 'Candidate Picks' page choose Top-N and download the CSV of variants to test next.\n"
        ),
    },
    "about_nav": {
        "zh": (
            "- **候选推荐**：不用上传，直接给出下一批建议实验的序列与导出表。\n"
            "- **候选预测**：自己准备一批候选 CSV 上传，得到零样本名次 + 模型名次。\n"
            "- **数据入库**：把新一批实验数据并入训练集并自动重训，人工确认后采纳新模型。\n"
            "- **对比分析**：把预测值与实测值放在一起看，评估模型泛化能力与本轮推荐命中率。\n"
            "- **结果浏览**：查看 results/ 下已有的各套预测结果分布与明细。\n"
        ),
        "en": (
            "- **Candidate Picks**: no upload needed; directly returns the sequences to test next plus an export table.\n"
            "- **Candidate Scoring**: upload your own candidate CSV to get zero-shot ranks + model ranks.\n"
            "- **Data Ingestion**: merge a new experimental batch into the training set and auto-retrain; adopt the new model after manual confirmation.\n"
            "- **Comparison**: view predictions next to measured values to assess generalization and this round's hit rate.\n"
            "- **Results**: explore the distribution and details of the existing prediction result sets under results/.\n"
        ),
    },
    "about_weakness": {
        "zh": (
            "- **组合突变排序偏弱**：批次 A 单点突变零样本 Spearman 0.72，批次 B 组合突变仅 0.23，这是目前唯一被验证的短板。\n"
            "- **位点窗口表征已停用**：在 67 条数据规模下为负贡献，不再投入。\n"
            "- **暂不做微调**：数据量积累到 200 条以上再考虑 LoRA / 适配器方案。\n"
        ),
        "en": (
            "- **Weak combo-mutation ranking**: zero-shot Spearman is 0.72 for batch-A single mutations but only 0.23 for batch-B combos; this is the only verified weakness so far.\n"
            "- **Site-window representations retired**: they contributed negatively at the 67-sample scale, so we no longer invest in them.\n"
            "- **No fine-tuning yet**: consider LoRA / adapters only after the data grows past 200 samples.\n"
        ),
    },
    "about_protocol": {
        "zh": (
            "采用**按位点留出交叉验证**（leave-one-site-out）：每次把某个突变位点的全部变体留作测试集，"
            "用其余位点训练并预测，从而外推到从未见过的位点。\n\n"
            "- 这是「挑候选突变体」的真实场景，因此作为主指标；随机折的结果会虚高泛化能力，不作参考。\n"
            "- 主指标为排序能力（Spearman / AUC / Recall@10）；R² / RMSE 受标签尺度影响，参考价值有限。\n"
            "- 评测集为剔除背景位点 208 后的 27 个突变位点 / 64 个变体。\n"
        ),
        "en": (
            "We use **leave-one-site-out cross-validation**: each time, all variants at one mutation site are held out as the test set, "
            "the model is trained on the remaining sites, and predictions extrapolate to a site never seen before.\n\n"
            "- This mirrors the real 'pick candidate variants' scenario, so it is the primary metric; random folds inflate generalization and are not used.\n"
            "- Primary metrics are ranking ability (Spearman / AUC / Recall@10); R² / RMSE depend on label scale and are of limited value.\n"
            "- The evaluation set is the 27 mutation sites / 64 variants after removing background site 208.\n"
        ),
    },
    "about_priority": {
        "zh": (
            "- **P0 已完成** 用本地同配方重训的模型替换线上工件（旧工件在 XGBoost 3.0.5 下丢失截距，预测整体偏移约 1.4），并修复平台首页的评估展示。\n"
            "- **P1 待新数据** 主动学习回灌，专门补组合突变样本；每轮实验后按批内 WT 归一化并入训练集。\n"
            "- **P2 已完成** 零样本分数作为候选筛选的第一道排序器已上线；位点窗口特征方向已放弃。\n"
            "- **P3 待数据量** 数据积累到 200 条以上后，再考虑轻量微调（LoRA r=4 仅挂最后两层）或适配器方案。\n\n"
            "当前泛化能力：Macro Spearman {spearman} ·\nMacro AUC {auc} ·\nRecall@10 {recall10}。\n{note}。\n"
        ),
        "en": (
            "- **P0 done** Replaced the live artifact with a locally retrained model of the same recipe (the old artifact lost its intercept under XGBoost 3.0.5, shifting all predictions by ~1.4) and fixed the home-page evaluation display.\n"
            "- **P1 pending new data** Active-learning feedback to specifically add combo-mutation samples; after each round, merge into the training set with within-batch WT normalization.\n"
            "- **P2 done** The zero-shot score is live as the first-stage candidate ranker; the site-window feature direction has been dropped.\n"
            "- **P3 pending data volume** Reconsider lightweight fine-tuning (LoRA r=4 on the last two layers) or adapters only after data grows past 200 samples.\n\n"
            "Current generalization: Macro Spearman {spearman} ·\nMacro AUC {auc} ·\nRecall@10 {recall10}.\n{note}.\n"
        ),
    },
    # ---- 候选推荐页 ----
    "rec_hint": {
        "zh": (
            '<div class="ugt-callout">'
            "<strong>融合口径</strong>：候选内已见非背景位点占比越高，模型预测的权重越大；"
            "已见位点全用模型预测，未见位点全用零样本 masked-marginal 分数，两者按名次百分位加权后合并成「融合名次分」。"
            "<strong>{note}</strong>。"
            "</div>"
        ),
        "en": (
            '<div class="ugt-callout">'
            "<strong>Fusion rule</strong>: the higher the share of seen non-background sites in a candidate, the more weight the model prediction gets; "
            "seen sites use the model prediction entirely, unseen sites use the zero-shot masked-marginal score entirely, and the two are combined by rank percentile into a 'fused rank score'. "
            "<strong>{note}</strong>."
            "</div>"
        ),
    },
    "rec_shortcoming": {
        "zh": (
            "已知短板：**组合突变叠加**的排序能力最弱（批次 B 组合突变零样本 Spearman 仅 0.23，单点约 0.72）。"
            "另外零样本分数是各突变贡献之和，突变数越多该值天然越大，因此纯按零样本分排序时 3~4 个突变的组合容易排到前面。"
            "做满前 N 个候选时覆盖真实高活性突变的能力与旧配方持平，但单个组合候选的把握度较低；"
            "建议先用「突变类型」筛成单突变或组合突变，按需取用。"
        ),
        "en": (
            "Known weakness: **combo mutations** rank worst (zero-shot Spearman is only 0.23 for batch-B combos vs ~0.72 for single mutations). "
            "Also, the zero-shot score sums per-mutation contributions, so variants with more mutations naturally score higher; when ranking purely by zero-shot score, 3–4 mutation combos tend to float to the top. "
            "Filling the full top-N covers true high-activity variants on par with the old recipe, but confidence in any single combo candidate is low; "
            "use the 'Mutation type' filter to isolate single or combo mutations as needed."
        ),
    },
    "rec_table_title": {
        "zh": "推荐序列（筛选后 {n} 条中的前 {m} 条）",
        "en": "Recommended sequences (top {m} of {n} after filtering)",
    },
    # ---- 候选预测页 ----
    "predict_intro": {
        "zh": "上传候选突变体 CSV，平台按「零样本分数 → 模型预测」的顺序输出候选名次。",
        "en": "Upload a candidate-variant CSV; the platform outputs candidate ranks in the order 'zero-shot score → model prediction'.",
    },
    "predict_columns": {
        "zh": (
            "- 必需列：`variant`（如 `F208M`、`F208M;V129A`），多个突变用 `;` 或 `-` 分隔。\n"
            "- 可选列：`sequence`（留空会按 WT 序列自动补齐）、`activity_rel` / `activity`（≥5 条时额外给出数值精度）。\n"
        ),
        "en": (
            "- Required column: `variant` (e.g. `F208M`, `F208M;V129A`); separate multiple mutations with `;` or `-`.\n"
            "- Optional columns: `sequence` (auto-filled from the WT sequence if empty), `activity_rel` / `activity` (extra numerical accuracy when ≥5 rows).\n"
        ),
    },
    "predict_table_title": {
        "zh": "候选排序（{n} 条候选）",
        "en": "Candidate ranking ({n} candidates)",
    },
    "predict_value_caption": {
        "zh": "预测活性为 activity_rel 口径（WT = 1）。{extra}",
        "en": "Predicted activity uses the activity_rel scale (WT = 1). {extra}",
    },
    # ---- 数据入库页 ----
    "ingest_intro": {
        "zh": (
            "拿到新一批实验数据后，在本页把数据并入统一训练集，并用统一入口重训模型。"
            "训练数据与模型采纳是两个独立决定：数据入库后即使不采纳新模型，实验测量值也会保留在训练集里。"
        ),
        "en": (
            "After obtaining a new experimental batch, merge it into the unified training set on this page and retrain via the unified entry point. "
            "Data ingestion and model adoption are independent decisions: even if you don't adopt the new model, the measured values remain in the training set."
        ),
    },
    "ingest_norm_note": {
        "zh": (
            '<div class="ugt-callout">'
            "<strong>为什么必须按批内 WT 归一化</strong>：不同批次的比酶活存在批间漂移，"
            "直接混用绝对值会让模型学到「批次」而不是「突变」。本页会用该批次的 WT 活性做除数，"
            "算出 activity_rel（相对 WT 倍数，WT = 1）后再入库。"
            "</div>"
        ),
        "en": (
            '<div class="ugt-callout">'
            "<strong>Why within-batch WT normalization is required</strong>: specific activities drift between batches, "
            "so mixing absolute values directly makes the model learn 'batch' instead of 'mutation'. This page divides by that batch's WT activity "
            "to compute activity_rel (relative to WT, WT = 1) before ingestion."
            "</div>"
        ),
    },
    "ingest_format_table": {
        "zh": (
            "最小格式只有两列：\n\n"
            "| 列名 | 必填 | 说明 |\n"
            "| --- | --- | --- |\n"
            "| `variant` | 是 | 突变体写法，`F208M` 或 `F208M;V129A`（分隔符 `-` 与 `;` 均可）；WT 行写 `WT` |\n"
            "| `activity` | 是 | 该批次的**原始**比酶活实测值；WT 必须为正数 |\n"
        ),
        "en": (
            "The minimal format has just two columns:\n\n"
            "| Column | Required | Description |\n"
            "| --- | --- | --- |\n"
            "| `variant` | yes | Variant notation, `F208M` or `F208M;V129A` (both `-` and `;` work); write `WT` for the wild type |\n"
            "| `activity` | yes | The batch's **raw** measured specific activity; WT must be positive |\n"
        ),
    },
    "ingest_format_explain": {
        "zh": (
            "平台会自动补齐 `mutation_list / sequence / sites / sites_str / n_mut / is_background_only`，"
            "并按 `activity_rel = activity / 批内 WT` 归一化。突变写法会与 WT 序列逐位点核对，"
            "WT 残基对不上会直接报错（挡住手误）。"
        ),
        "en": (
            "The platform auto-fills `mutation_list / sequence / sites / sites_str / n_mut / is_background_only` "
            "and normalizes with `activity_rel = activity / within-batch WT`. Mutation notations are checked position by position against the WT sequence; "
            "a mismatched WT residue raises an error immediately (guarding against typos)."
        ),
    },
    "ingest_dataset_caption": {
        "zh": "当前训练集 {n} 行，来自批次 {b}。",
        "en": "Training set currently has {n} rows from batches {b}.",
    },
    "ingest_pending": {
        "zh": "**待入库 {n} 条（批内 WT = {wt}）**",
        "en": "**{n} rows to ingest (within-batch WT = {wt})**",
    },
    "ingest_confirm": {
        "zh": "我已核对上述 {n} 条数据无误，同意追加进训练集（会先自动备份原文件）",
        "en": "I have verified the {n} rows above and agree to append them to the training set (the original file is backed up first)",
    },
    "ingest_step1": {
        "zh": "① 用当前线上模型给新批次打「入库前预测」快照…",
        "en": "1. Snapshotting this batch with the current live model (pre-ingest prediction)…",
    },
    "ingest_step1_done": {"zh": "① 入库前预测快照已记录", "en": "1. Pre-ingest prediction snapshot recorded"},
    "ingest_step1_fail": {
        "zh": "① 快照失败，流程已中断（未改动任何文件）",
        "en": "1. Snapshot failed; aborted without changing any file",
    },
    "ingest_snapshot_written": {
        "zh": "已写入预测快照，轮次 {round_id}（共 {n} 条突变体）",
        "en": "Snapshot written for round {round_id} ({n} variants)",
    },
    "ingest_step2": {"zh": "② 追加进训练集（先备份原文件）…", "en": "2. Appending to the training set (backing up first)…"},
    "ingest_backup": {"zh": "备份：{name}", "en": "Backup: {name}"},
    "ingest_rows": {"zh": "行数：{before} → {after}", "en": "Rows: {before} → {after}"},
    "ingest_step2_done": {"zh": "② 已入库，训练集 {n} 行", "en": "2. Ingested; training set now has {n} rows"},
    "ingest_step2_fail": {"zh": "② 入库失败，流程已中断", "en": "2. Ingestion failed; process aborted"},
    "ingest_step3_done": {"zh": "③ 特征缓存已重建", "en": "3. Feature cache rebuilt"},
    "ingest_step3_fail": {
        "zh": "③ 特征重建失败：数据已入库，但模型未更新",
        "en": "3. Feature rebuild failed: data is ingested but the model was not updated",
    },
    "ingest_step4_done": {"zh": "④ 候选模型已就绪", "en": "4. Candidate model ready"},
    "ingest_step4_fail": {
        "zh": "④ 重训失败：数据已入库，但模型未更新",
        "en": "4. Retrain failed: data is ingested but the model was not updated",
    },
    "ingest_live_header": {"zh": "现役模型（{n} 条训练样本）", "en": "Live model ({n} training samples)"},
    "ingest_cand_header": {"zh": "候选模型（{n} 条训练样本）", "en": "Candidate model ({n} training samples)"},
    "ingest_report_caption": {
        "zh": "候选报告生成于 {generated_at}；留出评估覆盖 {n_sites} 个位点、{n_variants} 条突变体。",
        "en": "Candidate report generated at {generated_at}; hold-out evaluation covers {n_sites} sites and {n_variants} variants.",
    },
    "ingest_grid_caption": {"zh": "网格择优选中的配方：{label}", "en": "Recipe selected by the grid search: {label}"},
    "ingest_note_caption": {
        "zh": (
            "口径须知：主指标来自**按位点留出**（外推到训练中从未出现的突变位点），与「挑候选突变体」"
            "的目标一致；{note}。"
        ),
        "en": (
            "Note: the primary metrics come from **leave-one-site-out** (extrapolation to sites never seen in training), "
            "aligned with the goal of 'picking candidate variants'; {note}."
        ),
    },
    "ingest_adopt_success": {
        "zh": (
            "已采纳新模型：Macro Spearman {spearman}、Macro AUC {auc}、Recall@10 {recall10}。"
            "旧模型已归档为 {archived}。"
        ),
        "en": (
            "New model adopted: Macro Spearman {spearman}, Macro AUC {auc}, Recall@10 {recall10}. "
            "Old model archived as {archived}."
        ),
    },
    # ---- 对比分析页 ----
    "compare_intro": {
        "zh": (
            "把模型给出的预测值和实测活性放在一起对比。两个板块口径不同："
            "上板块看**外推到新位点的泛化能力**，下板块看**本轮推荐的命中情况**。"
        ),
        "en": (
            "Compare the model's predictions against measured activity. The two panels differ in scope: "
            "the upper panel shows **generalization to new sites**, the lower shows **this round's recommendation hits**."
        ),
    },
    "compare_read_note": {
        "zh": (
            '<div class="ugt-callout">'
            "<strong>读图须知</strong>：散点越贴近金色虚线（y = x）说明预测越准；"
            "点云整体在虚线上方表示模型系统性高估、下方表示系统性低估。"
            "排序能力（Spearman / AUC / Recall@K）才是挑候选的核心，数值精度只是参考。<strong>{note}</strong>。"
            "</div>"
        ),
        "en": (
            '<div class="ugt-callout">'
            "<strong>How to read</strong>: the closer a point lies to the gold dashed line (y = x), the more accurate the prediction; "
            "a cloud above the line means systematic overestimation, below means underestimation. "
            "Ranking ability (Spearman / AUC / Recall@K) is what matters for picking candidates; numerical accuracy is only a reference. <strong>{note}</strong>."
            "</div>"
        ),
    },
    "compare_empty": {
        "zh": (
            "还没有折外预测报告，本板块暂时为空。\n\n"
            "这份报告记录的是「每次留出一个突变位点、用其余数据训练后对该位点样本做出的预测」，"
            "是模型泛化能力的估计，需要在重训或补算时才会生成。"
        ),
        "en": (
            "No out-of-fold report yet; this panel is empty for now.\n\n"
            "This report records 'the prediction made for a site's samples after training on the other sites and holding out that site'; "
            "it estimates generalization and is produced during retraining or a backfill run."
        ),
    },
    "compare_first_gen": {
        "zh": "首次生成（对当前线上模型补算一份，不改变模型本身，报告落在 `{dir}/`）：",
        "en": "First-time generation (backfill for the current live model; the model itself is unchanged; report saved under `{dir}/`):",
    },
    "compare_oof_caption": {
        "zh": (
            "报告生成于 {generated_at}，训练集 {dataset_rows} 行；"
            "折外预测来自「同评测协议重新训练各折」，是外部泛化的估计，不是现役模型对训练样本的拟合值。"
        ),
        "en": (
            "Report generated at {generated_at}; training set has {dataset_rows} rows. "
            "Out-of-fold predictions come from retraining each fold under the same protocol; they estimate external generalization, not the live model's fit to training samples."
        ),
    },
    "compare_residual_caption": {
        "zh": "残差均值 {mean:+.3f}，中位 {median:+.3f}",
        "en": "Residual mean {mean:+.3f}, median {median:+.3f}",
    },
    "compare_panel2_caption": {
        "zh": (
            "如实说明：单轮新增样本通常只有几条，命中率与相关系数噪声很大，只适合看趋势；"
            "组合突变叠加是已验证的短板（多突变候选的把握度低于单点），不要据此对单个候选下强结论。"
        ),
        "en": (
            "Honest note: a single round usually adds only a few samples, so hit rate and correlation are noisy and only reflect trends; "
            "combo mutations are a verified weakness (confidence in multi-mutation candidates is lower than for single ones), so don't draw strong conclusions about any single candidate."
        ),
    },
    # ---- 结果浏览页 ----
    "results_source_caption": {"zh": "来源文件 `{file}` · {note}", "en": "Source file `{file}` · {note}"},
    "results_rank_caption": {
        "zh": "排序依据 `{score}`（降序）；该文件不含真实标签，仅反映候选优先级。",
        "en": "Ranked by `{score}` (descending); this file has no true labels and only reflects candidate priority.",
    },
    "results_detail_header": {
        "zh": "候选明细（共 {n} 条，显示前 200 条）",
        "en": "Candidate details ({n} rows, first 200 shown)",
    },
    # ---- 核心模块 ----
    "core_missing_seq": {
        "zh": "发现 {n} 条记录缺少 sequence，且 data/wt.fasta 不可用，无法生成序列。",
        "en": "{n} records are missing a sequence and data/wt.fasta is unavailable, so sequences cannot be generated.",
    },
    "core_unknown_source": {"zh": "未知结果来源：{key}", "en": "Unknown result source: {key}"},
    "core_missing_file": {"zh": "结果文件不存在：{name}", "en": "Result file not found: {name}"},
    "core_subproc_fail": {
        "zh": "子进程失败（退出码 {code}）：{name}\n{tail}",
        "en": "Subprocess failed (exit code {code}): {name}\n{tail}",
    },
    "core_model_params": {
        "zh": "XGBoost {n} 棵 × 深度 {depth} × lr {lr}",
        "en": "XGBoost {n} trees × depth {depth} × lr {lr}",
    },
    "compare_combo_help": {
        "zh": "含 {n} 条组合突变",
        "en": "includes {n} combo mutations",
    },
    "ingest_grid_rows": {
        "zh": "训练集 {n} 行",
        "en": "{n} training rows",
    },
    "core_dim_mismatch": {
        "zh": "ESM 表征维度不匹配：期望 {a}，实际 {b}",
        "en": "ESM representation dimension mismatch: expected {a}, got {b}",
    },
    "core_missing_model": {
        "zh": "候选目录里没有 model.pkl：{path}",
        "en": "No model.pkl in the candidate directory: {path}",
    },
    "rec_download": {
        "zh": "下载前 {n} 条推荐序列 CSV",
        "en": "Download top {n} recommended sequences CSV",
    },
    "predict_value_extra": {
        "zh": (
            "本次上传带标签，该列由上传数据的五折外推得到（{n} 条），"
            "样本较少时波动大，仅作参考。"
        ),
        "en": "This upload has labels, so the column comes from 5-fold extrapolation of the uploaded data ({n} rows); it is noisy with few samples and is only a reference.",
    },
    "predict_value_extra_none": {
        "zh": "本次上传未带标签，该列来自平台预训练模型。",
        "en": "This upload has no labels, so the column comes from the platform's pretrained model.",
    },
    "ingest_feature_rebuild": {
        "zh": "**③ 重建 ESM 特征缓存**（行序必须与新的训练集逐行对齐，因此整体重算）",
        "en": "**3. Rebuild the ESM feature cache** (row order must align line-by-line with the new training set, so it is recomputed in full)",
    },
    "ingest_retrain_label": {
        "zh": "**④ 重训候选模型**（输出到 `models/candidates/{name}/`，不覆盖线上模型）",
        "en": "**4. Retrain the candidate model** (written to `models/candidates/{name}/`, the live model is untouched)",
    },
    "ingest_step3_status": {
        "zh": "正在重建特征缓存（需要 GPU，可能较久）…",
        "en": "Rebuilding the feature cache (uses the GPU, may take a while)…",
    },
    "ingest_step4_status": {
        "zh": "正在重训并做按位点留出评估…",
        "en": "Retraining and running leave-one-site-out evaluation…",
    },
    "predict_progress": {
        "zh": "{step}／6　{message}",
        "en": "Step {step}/6 · {message}",
    },
    "predict_read_fail": {
        "zh": "读取 CSV 失败：{err}",
        "en": "Failed to read CSV: {err}",
    },
    "predict_missing_upload": {
        "zh": "请先选择一个 CSV 文件。",
        "en": "Please choose a CSV file first.",
    },
    "predict_running": {"zh": "正在预测…", "en": "Predicting…"},
    "predict_done": {"zh": "预测完成", "en": "Prediction complete"},
    "predict_failed": {"zh": "预测失败", "en": "Prediction failed"},
    "predict_no_labels": {
        "zh": "未提供标签，跳过数值精度评估。",
        "en": "No labels provided; skipping numerical accuracy evaluation.",
    },
    "predict_upload_help": {
        "zh": "UTF-8 或 GBK 编码的 CSV",
        "en": "CSV encoded as UTF-8 or GBK",
    },
    "results_core_note": {
        "zh": (
            "「结果浏览」页只读展示 results/ 下已存在的预测结果文件，不做任何写操作；"
            "结果来源与评测协议见「平台说明」页。"
        ),
        "en": (
            "The 'Results' page only reads the existing prediction files under results/ and never writes; "
            "see the 'Platform Guide' page for result sources and the evaluation protocol."
        ),
    },
    "results_choose_hint": {
        "zh": "选择要查看的结果文件，右侧显示其分布与明细。",
        "en": "Choose a result file to view; distributions and details appear on the right.",
    },
    "compare_site_help": {
        "zh": "覆盖 {n} 个位点",
        "en": "covers {n} sites",
    },
    "compare_oof_expander": {
        "zh": "折外预测明细（共 {n} 条）",
        "en": "Out-of-fold details ({n} rows)",
    },
    "compare_no_log": {
        "zh": (
            "还没有入库前的预测记录。到「数据入库」页上传新批次数据并执行入库，"
            "平台会先用当时的线上模型给这批突变体打一份预测快照，再把它与实测值一起记在这里。"
        ),
        "en": (
            "No pre-ingest prediction records yet. On the 'Data Ingestion' page upload a new batch and run ingestion; "
            "the platform first snapshots predictions for these variants with the then-live model, then logs them here alongside measured values."
        ),
    },
    "compare_retrain_hint": {
        "zh": "也可以在「数据入库」页重训一次并采纳新模型，报告会随采纳自动落盘。",
        "en": "You can also retrain once and adopt a new model on the 'Data Ingestion' page; the report is saved automatically upon adoption.",
    },
    "compare_log_hint": {
        "zh": "提示：下方板块二已有入库前的预测记录，可直接查看。",
        "en": "Tip: panel 2 below already has pre-ingest prediction records that you can view directly.",
    },
    "ingest_parse_fail": {
        "zh": "解析失败，未做任何写入。请修正后再上传：\n\n",
        "en": "Parsing failed; nothing was written. Fix the issues and re-upload:\n\n",
    },
    "ingest_grid_help": {
        "zh": (
            "关闭＝沿用线上配方快速重训；打开＝另外试 4 组超参，"
            "按按位点留出的 Macro Spearman 择优，耗时明显更长。"
        ),
        "en": (
            "Off = quick retrain with the live recipe; On = additionally try 4 hyperparameter sets and pick by "
            "leave-one-site-out Macro Spearman, which takes noticeably longer."
        ),
    },
    "ingest_grid_choice": {
        "zh": "网格择优选中的配方：{label}",
        "en": "Recipe chosen by the grid search: {label}",
    },
    "ingest_no_oof": {
        "zh": "候选报告里没有折外预测明细。",
        "en": "The candidate report has no out-of-fold details.",
    },
    "ingest_adopt_help": {
        "zh": "旧模型会先归档进 models/history/，可随时回退。",
        "en": "The old model is archived to models/history/ first and can be rolled back at any time.",
    },
    "ingest_adopt_note": {
        "zh": "注意：候选推荐表仍是按重训前的模型算的。如需同步刷新，请到「候选推荐」页重建推荐表。",
        "en": "Note: the recommendation table was still computed with the pre-retrain model. To refresh it, rebuild the table on the 'Candidate Picks' page.",
    },
    "ingest_discard_help": {
        "zh": "只删除候选模型与报告，已入库的实验数据不会受影响。",
        "en": "Deletes only the candidate model and report; ingested experimental data is unaffected.",
    },
}
