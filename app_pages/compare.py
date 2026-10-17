"""对比分析页：把模型给出的预测值和实测值放在一起看，评估模型好坏。

两个板块（口径不同，不要混读）：
  板块一 · 全量按位点留出的折外预测 vs 实测
     衡量「模型外推到没见过的新位点」的泛化能力，是模型好坏的主口径。
     数据来自 models/training_report.csv（重训或补算时产出的 OOF 预测）。
  板块二 · 本轮新增数据：入库前预测 vs 实测
     衡量「这一轮推荐的候选实际做得怎么样」，即主动学习的命中率。
     数据来自 results/active_learning_log.csv（入库时用当时线上模型打的快照）。
"""

import numpy as np
import pandas as pd
import streamlit as st

from i18n import t, tb
from prediction_core import (
    RANK_ONLY_NOTE,
    deploy_paths,
    load_active_learning_log,
    load_training_report,
    report_metric_row,
)
from ui_helpers import metric_row, pred_vs_actual_scatter, render_chart, residual_histogram

ACTIVE_THRESHOLD = 1.0  # activity_rel > 1 视为高于 WT


def spearman(a, b) -> float | None:
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def topk_hits(y_true, y_pred, k):
    """按预测取前 k 个，其中实测高于 WT 的个数。"""
    k = min(k, len(y_pred))
    order = np.argsort(-np.asarray(y_pred, dtype=float))[:k]
    return int((np.asarray(y_true, dtype=float)[order] > ACTIVE_THRESHOLD).sum()), k


def grouped_table(grouped: dict | None, key_title: str) -> pd.DataFrame:
    """把 report.json 里的分组指标转成展示表。"""
    if not grouped:
        return pd.DataFrame()
    frame = pd.DataFrame(grouped).T.reset_index()
    frame.columns = [key_title, t("样本数"), "Spearman", "AUC", "Recall@5", "Recall@10"]
    return frame


st.caption(tb("compare_intro"))

report, oof = load_training_report()
metrics_view = report_metric_row(report)

st.markdown(
    tb("compare_read_note", note=t(RANK_ONLY_NOTE)),
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# 板块一：全量按位点留出
# --------------------------------------------------------------------------- #
st.subheader(t("板块一 · 全量按位点留出的折外预测 vs 实测"))

if oof.empty or report is None:
    paths = deploy_paths()
    st.info(tb("compare_empty"))
    st.markdown(tb("compare_first_gen", dir=paths["active_report_dir"].name))
    st.code(
        f"python scripts/retrain_deploy_model.py --eval-only "
        f"--model models/xgb_esm.pkl --report-dir {paths['active_report_dir'].as_posix()}",
        language="powershell",
    )
    st.caption(tb("compare_retrain_hint"))
    if not load_active_learning_log().empty:
        st.caption(tb("compare_log_hint"))
else:
    eval_rows = oof[oof["oof_pred"].notna()].copy()
    eval_rows["residual"] = eval_rows["oof_pred"] - eval_rows["activity_rel"]

    metric_row(
        [
            ("Macro Spearman", metrics_view["spearman_macro"], t("按批次分组后的平均排序相关性")),
            ("Macro AUC", metrics_view["auc_macro"], t("把 activity_rel > 1 视为高活性后的区分能力")),
            ("Recall@10", metrics_view["recall10_macro"], t("取前 10 个候选能覆盖多少真实高活性突变")),
            (
                t("评估突变体数"),
                str(metrics_view["n_variants"]),
                tb("compare_site_help", n=metrics_view["n_sites"]),
            ),
        ]
    )
    st.caption(
        tb(
            "compare_oof_caption",
            generated_at=metrics_view["generated_at"],
            dataset_rows=metrics_view["dataset_rows"],
        )
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(f"**{t('预测 vs 实测（按批次着色）')}**")
        chart = pred_vs_actual_scatter(
            eval_rows,
            color_col="batch",
            title_pred=t("预测 activity_rel"),
            title_actual=t("实测 activity_rel"),
            limit=400,
        )
        if chart is None:
            st.info(t("没有可用的折外预测点。"))
        else:
            render_chart(chart)
    with right:
        st.markdown(f"**{t('残差分布')}**")
        render_chart(residual_histogram(eval_rows["residual"]))
        st.caption(
            tb(
                "compare_residual_caption",
                mean=eval_rows["residual"].mean(),
                median=eval_rows["residual"].median(),
            )
        )

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{t('按批次分层')}**")
        frame = grouped_table((report or {}).get("by_batch"), t("批次"))
        if frame.empty:
            st.info(t("报告里没有按批次的分层指标。"))
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
    with right:
        st.markdown(f"**{t('按突变数分层')}**")
        frame = grouped_table((report or {}).get("by_n_mut"), t("突变数"))
        if frame.empty:
            st.info(t("报告里没有按突变数的分层指标。"))
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(t("样本不足 3 条或该组内真值无方差时，分层指标留空（显示为 None），不做无意义的计算。"))

    with st.expander(tb("compare_oof_expander", n=len(eval_rows)), expanded=False):
        st.dataframe(
            eval_rows[
                ["variant", "batch", "n_mut", "activity_rel", "oof_pred", "residual"]
            ].head(400),
            hide_index=True,
            width="stretch",
            height=340,
            column_config={
                "variant": st.column_config.TextColumn(t("突变体"), width="medium"),
                "batch": st.column_config.TextColumn(t("批次"), width="small"),
                "n_mut": st.column_config.NumberColumn(t("突变数"), width="small"),
                "activity_rel": st.column_config.NumberColumn(t("实测"), format="%.3f"),
                "oof_pred": st.column_config.NumberColumn(t("折外预测"), format="%.3f"),
                "residual": st.column_config.NumberColumn(t("残差"), format="%.3f"),
            },
        )
        st.download_button(
            t("下载折外预测明细 CSV"),
            eval_rows.to_csv(index=False).encode("utf-8-sig"),
            "training_report_oof.csv",
            "text/csv",
            icon=":material/download:",
            key="download_oof",
        )

# --------------------------------------------------------------------------- #
# 板块二：本轮新增数据
# --------------------------------------------------------------------------- #
st.divider()
st.subheader(t("板块二 · 本轮新增数据：入库前预测 vs 实测"))

log = load_active_learning_log()
if log.empty:
    st.info(tb("compare_no_log"))
else:
    log = log.copy()
    log["actual_activity_rel"] = pd.to_numeric(log["actual_activity_rel"], errors="coerce")
    log["pred_before"] = pd.to_numeric(log["pred_before"], errors="coerce")

    rounds = sorted(log["round_id"].astype(str).unique())
    all_rounds = t("全部轮次")
    options = [all_rounds] + list(reversed(rounds))
    choice = st.selectbox(t("查看范围"), options, index=0, key="compare_round")

    view = log if choice == all_rounds else log[log["round_id"].astype(str) == choice]
    view = view.reset_index(drop=True)
    valid = view.dropna(subset=["pred_before", "actual_activity_rel"])

    hit_count, hit_k = (0, 0)
    if not valid.empty:
        hit_count, hit_k = topk_hits(
            valid["actual_activity_rel"], valid["pred_before"], min(5, len(valid))
        )

    metric_row(
        [
            (t("轮次"), t("全部") if choice == all_rounds else str(choice), t("每次入库对应一轮")),
            (
                t("突变体数"),
                str(len(view)),
                tb("compare_combo_help", n=int(view["n_mut"].gt(1).sum())),
            ),
            (
                t("Top-5 命中"),
                f"{hit_count}/{hit_k}" if hit_k else "-",
                t("按预测取前 5 个，其中实测高于 WT 的个数"),
            ),
            (
                "Spearman",
                "-" if spearman(valid["actual_activity_rel"], valid["pred_before"]) is None
                else f"{spearman(valid['actual_activity_rel'], valid['pred_before']):.3f}",
                t("单轮样本少，仅作趋势参考"),
            ),
        ]
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(f"**{t('入库前预测 vs 实测（按轮次着色）')}**")
        color = "round_id" if choice == all_rounds else None
        chart = pred_vs_actual_scatter(
            valid,
            pred_col="pred_before",
            actual_col="actual_activity_rel",
            color_col=color,
            title_pred=t("入库前预测 activity_rel"),
            title_actual=t("实测 activity_rel"),
            limit=400,
        )
        if chart is None:
            st.info(t("当前范围内没有同时含预测与实测的记录。"))
        else:
            render_chart(chart)
    with right:
        st.markdown(f"**{t('残差分布')}**")
        chart = residual_histogram(valid["pred_before"] - valid["actual_activity_rel"])
        if chart is None:
            st.info(t("数据不足，暂不绘制残差分布。"))
        else:
            render_chart(chart)

    st.markdown(f"**{t('明细')}**")
    ratio_col = t("实测减预测")
    detail = view.assign(
        **{ratio_col: view["actual_activity_rel"] - view["pred_before"]}
    )[
        [
            "round_id",
            "batch",
            "variant",
            "n_mut",
            "pred_before",
            "zero_score",
            "rank_before",
            "actual_activity_rel",
            ratio_col,
        ]
    ]
    st.dataframe(
        detail,
        hide_index=True,
        width="stretch",
        height=340,
        column_config={
            "round_id": st.column_config.TextColumn(t("轮次"), width="small"),
            "batch": st.column_config.TextColumn(t("批次"), width="small"),
            "variant": st.column_config.TextColumn(t("突变体"), width="medium"),
            "n_mut": st.column_config.NumberColumn(t("突变数"), width="small"),
            "pred_before": st.column_config.NumberColumn(t("入库前预测"), format="%.3f"),
            "zero_score": st.column_config.NumberColumn(t("零样本分数"), format="%.3f"),
            "rank_before": st.column_config.NumberColumn(t("原推荐名次"), width="small"),
            "actual_activity_rel": st.column_config.NumberColumn(t("实测"), format="%.3f"),
            ratio_col: st.column_config.NumberColumn(t("实测 − 预测"), format="%.3f"),
        },
    )
    st.caption(tb("compare_panel2_caption"))
    st.download_button(
        t("下载本轮对比明细 CSV"),
        detail.to_csv(index=False).encode("utf-8-sig"),
        "active_learning_compare.csv",
        "text/csv",
        icon=":material/download:",
        key="download_active_log",
    )
