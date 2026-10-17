"""候选推荐页：直接用现有实验数据 + 候选库，给出下一批建议实验的突变体序列。

无需上传文件：候选库（单点 + F208M 打底组合）已由
scripts/build_candidate_recommendations.py 预先打分并融合排序。
"""

import numpy as np
import streamlit as st

from i18n import t, tb
from prediction_core import (
    COVERAGE_LABELS,
    RANK_ONLY_NOTE,
    load_recommendations,
    recommendation_summary,
)
from ui_helpers import coverage_bar, fused_bar, metric_row, model_vs_zero_scatter, render_chart

st.caption(
    t(
        "本页不依赖上传：平台从现有候选库里，对**已见位点用模型预测、未见位点用零样本分数**"
        "分别打分，再按名次百分位融合，直接给出下一批最值得做实验的序列。"
    )
)

try:
    frame = load_recommendations()
except Exception as exc:
    st.error(str(exc))
    st.stop()

summary = recommendation_summary(frame)
frame["coverage_label"] = frame["coverage"].map(lambda key: t(COVERAGE_LABELS.get(key, key)))

metric_row(
    [
        (t("候选总数"), str(summary["total"]), t("单点突变 + F208M 打底组合，按突变体去重合并")),
        (t("可模型预测"), str(summary["with_model"]), t("至少含 1 个训练集已见非背景位点的候选")),
        (t("已见位点候选"), str(summary["seen_sites"]), t("含 ≥1 个已见位点，模型可直接外推")),
        (t("最高融合分"), f"{summary['top_fused']:.3f}", t("融合名次分，越接近 1 越优先")),
    ]
)

st.markdown(
    tb("rec_hint", note=t(RANK_ONLY_NOTE)),
    unsafe_allow_html=True,
)

st.info(tb("rec_shortcoming"), icon=":material/info:")

st.divider()

# --------------------------------------------------------------------------- #
# 筛选
# --------------------------------------------------------------------------- #
left, mid, right = st.columns([1.1, 1, 1.6], vertical_alignment="bottom")
with left:
    top_n = st.slider(t("推荐数量 Top-N"), min_value=5, max_value=60, value=15, step=5)
with mid:
    mut_choice = st.segmented_control(
        t("突变类型"),
        [t("全部"), t("单突变"), t("组合突变")],
        default=t("全部"),
        key="rec_mut_type",
    )
with right:
    coverage_choice = st.multiselect(
        t("位点覆盖"),
        options=list(COVERAGE_LABELS),
        default=list(COVERAGE_LABELS),
        format_func=lambda key: t(COVERAGE_LABELS[key]),
        key="rec_coverage",
    )

view = frame.copy()
mut_choice = mut_choice or t("全部")
if mut_choice == t("单突变"):
    view = view[view["n_mut"].eq(1)]
elif mut_choice == t("组合突变"):
    view = view[view["n_mut"].gt(1)]
if coverage_choice:
    view = view[view["coverage"].isin(coverage_choice)]
view = view.reset_index(drop=True)
view["view_rank"] = np.arange(1, len(view) + 1)

if view.empty:
    st.warning(t("当前筛选条件下没有候选，请放宽条件。"))
    st.stop()

top = view.head(top_n).copy()
top["排序标签"] = top["view_rank"].astype(str) + ". " + top["variant"]

st.subheader(tb("rec_table_title", n=len(view), m=len(top)))

chart_left, chart_right = st.columns([1.4, 1])
with chart_left:
    st.markdown(f"**{t('推荐优先级（融合名次分）')}**")
    render_chart(fused_bar(top, value_col="fused_score", label_col="排序标签", limit=top_n))
with chart_right:
    st.markdown(f"**{t('整库位点覆盖分布')}**")
    render_chart(coverage_bar(view))

st.markdown(f"**{t('模型预测 vs 零样本分数（按位点覆盖着色）')}**")
scatter = model_vs_zero_scatter(view, limit=800)
if scatter is None:
    st.info(t("当前筛选下没有同时具备模型预测与零样本分数的候选。"))
else:
    render_chart(scatter)
st.caption(t("两个分数尺度不同、不可直接比较；点位越靠右上通常越值得优先实验。"))

st.subheader(t("推荐明细"))
display = top[
    [
        "view_rank",
        "variant",
        "mutation_list",
        "n_mut",
        "positions",
        "coverage_label",
        "fused_score",
        "model_pred",
        "zero_score",
        "family_support",
        "pool_source",
        "rank",
    ]
]
st.dataframe(
    display,
    hide_index=True,
    width="stretch",
    height=440,
    column_config={
        "view_rank": st.column_config.NumberColumn(t("推荐名次"), width="small"),
        "variant": st.column_config.TextColumn(t("突变体"), width="medium"),
        "mutation_list": st.column_config.TextColumn(t("突变列表"), width="medium"),
        "n_mut": st.column_config.NumberColumn(t("突变数"), width="small"),
        "positions": st.column_config.TextColumn(t("位点"), width="small"),
        "coverage_label": st.column_config.TextColumn(t("位点覆盖"), width="small"),
        "fused_score": st.column_config.NumberColumn(t("融合名次分"), format="%.4f"),
        "model_pred": st.column_config.NumberColumn(t("模型预测活性"), format="%.3f"),
        "zero_score": st.column_config.NumberColumn(t("零样本分数"), format="%.3f"),
        "family_support": st.column_config.NumberColumn(t("家族支持度"), format="%.1f"),
        "pool_source": st.column_config.TextColumn(t("来源候选池"), width="medium"),
        "rank": st.column_config.NumberColumn(t("全局名次"), width="small"),
    },
)

st.caption(t("模型预测活性为 activity_rel 口径（WT = 1），仅用于名次参考，勿做绝对值阈值筛选。"))

st.download_button(
    tb("rec_download", n=len(top)),
    top.to_csv(index=False).encode("utf-8-sig"),
    "ugt91c1_recommended_candidates.csv",
    "text/csv",
    type="primary",
    icon=":material/download:",
    key="download_recommend",
)
