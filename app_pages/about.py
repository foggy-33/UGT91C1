"""平台说明页：合并原「概览」与「模型与数据」，集中介绍模型、数据、评测与后续优先级。"""

import streamlit as st

from i18n import t, tb
from prediction_core import (
    RESULT_SOURCES,
    get_resources,
    model_evaluation,
    model_params_text,
    tier_cv_metrics,
)
from ui_helpers import metric_row

MODEL_LABELS = {
    "ZS_zeroshot_only": "零样本分数（当前排序器）",
    "M0_meanpool_prior_XGB": "线上配方（mean-pool+先验）",
    "M1_zeroshot_prior_XGB": "零样本+先验+XGBoost",
    "M2_sitelocal_zs_prior_PCA_Ridge": "位点窗口+零样本+PCA",
    "M3_sitelocal_zs_prior_XGB": "位点窗口+零样本+XGBoost",
    "M4_fusion_PCA_Ridge": "融合特征+PCA-Ridge",
}

evaluation = model_evaluation()
resources = get_resources()
n_train = evaluation["n_train"]

st.caption(tb("about_intro"))

st.subheader(t("当前模型"))
metric_row(
    [
        (t("预测目标"), evaluation["target"], t(evaluation["target_note"]) or None),
        (
            t("特征数量"),
            str(len(resources["feature_names"])),
            t("ESM-2 序列表征 1280 维 + 进化 / 结构 / 稳定性先验 23 维"),
        ),
        (t("训练数据"), f"{n_train} {t('条')}", t("两个实验批次按各自批内 WT 归一化后合并")),
        (t("回归器"), model_params_text()),
    ]
)

st.subheader(t("泛化能力（按位点留出）"))
metric_row(
    [
        (
            "Macro Spearman",
            evaluation["text"]["spearman_macro"],
            t("留出整个突变位点后外推，最贴近「挑候选突变体」的真实场景"),
        ),
        ("Macro AUC", evaluation["text"]["auc_macro"], t("区分「高于 WT」与「低于 WT」的能力")),
        ("Recall@10", evaluation["text"]["recall10_macro"], t("前 10 名候选覆盖真实高活性突变的比例")),
        (
            "RMSE",
            evaluation["text"]["rmse"],
            f"{evaluation['rmse_protocol']} · {t('只反映数值精度，挑候选时参考价值有限')}",
        ),
    ]
)
st.caption(
    tb(
        "about_eval_caption",
        protocol=evaluation["protocol"],
        n_variants=evaluation["n_variants"],
        n_sites=evaluation["n_sites"],
    )
)

st.markdown(
    tb("about_usage_note", note=t(evaluation["rank_only_note"])),
    unsafe_allow_html=True,
)

st.subheader(t("预测流程"))
st.markdown(tb("about_flow"))

st.subheader(t("页面导航"))
st.markdown(tb("about_nav"))

st.subheader(t("已知短板"))
st.markdown(tb("about_weakness"))

st.subheader(t("数据"))
metric_row(
    [
        (t("训练数据"), f"{n_train} {t('条')}"),
        (t("实验批次"), t("2 个"), t("批次 A：WT = 0.021；批次 B：WT = 3.7")),
        (t("批间差异"), t("1.76 倍"), t("按各自批内 WT 归一化后再合并，避免引入批间偏差")),
        (t("背景位点"), "F208M", t("出现在几乎所有批次 B 变体中，评测与排序时剔除")),
    ]
)

st.subheader(t("评测协议"))
st.markdown(tb("about_protocol"))

st.subheader(t("方案对比"))
metrics = tier_cv_metrics()
if metrics.empty:
    st.info(t("未找到 `results/tier_cv_metrics.csv`，跳过方案对比。"))
else:
    show = metrics.assign(方案=metrics["model"].map(lambda key: t(MODEL_LABELS.get(key, key))))
    show = show[["方案", "n_features", "macro_spearman", "macro_auc", "macro_recall10"]]
    show.columns = [t("方案"), t("特征数"), "Macro Spearman", "Macro AUC", "Recall@10"]
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            t("方案"): st.column_config.TextColumn(width="medium"),
            t("特征数"): st.column_config.NumberColumn(width="small"),
            "Macro Spearman": st.column_config.NumberColumn(width="small", format="%.3f"),
            "Macro AUC": st.column_config.NumberColumn(width="small", format="%.3f"),
            "Recall@10": st.column_config.NumberColumn(width="small", format="%.2f"),
        },
    )
    st.caption(
        t(
            "结论：参数无关的零样本分数排序能力最强；位点窗口表征是负贡献；"
            "在 67 条数据规模下加训练头反而拖累排序（0.48 → 0.17）。"
        )
    )

st.subheader(t("结果来源"))
st.dataframe(
    [
        {
            t("文件名"): info["path"].name,
            t("说明"): t(info["short"]),
            t("状态"): t("已就绪") if info["path"].exists() else t("文件缺失"),
        }
        for info in RESULT_SOURCES.values()
    ],
    hide_index=True,
    width="stretch",
    column_config={
        t("文件名"): st.column_config.TextColumn(width="medium"),
        t("说明"): st.column_config.TextColumn(width="medium"),
        t("状态"): st.column_config.TextColumn(width="small"),
    },
)

st.subheader(t("后续优先级"))
st.markdown(
    tb(
        "about_priority",
        spearman=evaluation["text"]["spearman_macro"],
        auc=evaluation["text"]["auc_macro"],
        recall10=evaluation["text"]["recall10_macro"],
        note=t(evaluation["rank_only_note"]),
    )
)
