"""候选预测页：上传 CSV → 零样本排序 + 模型预测 → 导出结果。"""

from datetime import datetime

import streamlit as st

from i18n import t, tb
from prediction_core import prediction_payload, read_upload_bytes, sample_template
from ui_helpers import metric_row, render_chart, score_bar, score_histogram

st.caption(t("上传候选突变体 CSV，平台按「零样本分数 → 模型预测」的顺序输出候选名次。"))
st.markdown(tb("predict_columns"))

uploaded = st.file_uploader(
    t("上传候选 CSV"), type=["csv"], help=tb("predict_upload_help")
)

left, right = st.columns([3, 1], vertical_alignment="bottom")
with left:
    run = st.button(t("开始预测"), type="primary", icon=":material/play_arrow:")
with right:
    st.download_button(
        t("下载模板"),
        sample_template(),
        "ugt_prediction_template.csv",
        "text/csv",
        icon=":material/download:",
        key="download_template",
    )

if run:
    if uploaded is None:
        st.error(tb("predict_missing_upload"))
    else:
        try:
            raw_df = read_upload_bytes(uploaded.getvalue())
        except Exception as exc:
            st.error(tb("predict_read_fail", err=exc))
        else:
            with st.status(tb("predict_running"), expanded=True) as status:

                def progress(step, message):
                    label = tb("predict_progress", step=step, message=message)
                    status.update(label=label)
                    st.write(label)

                try:
                    st.session_state.predict_result = prediction_payload(raw_df, progress=progress)
                    status.update(label=tb("predict_done"), state="complete", expanded=False)
                except Exception as exc:
                    status.update(label=tb("predict_failed"), state="error")
                    st.error(str(exc))

payload = st.session_state.get("predict_result")
if not payload:
    st.info(t("还没有预测结果：上传 CSV 后点击「开始预测」。"))
    st.stop()

frame = payload["frame"]
score_col = payload["score_col"]

st.divider()
st.subheader(tb("predict_table_title", n=payload["counts"]["input"]))
metric_row(
    [
        (t("候选数量"), str(payload["counts"]["input"])),
        (t("最高零样本分数"), f"{frame[score_col].max():.3f}"),
        (t("中位零样本分数"), f"{frame[score_col].median():.3f}"),
        (t("特征数量"), str(payload["counts"]["features"])),
    ]
)

left_chart, right_chart = st.columns(2)
with left_chart:
    st.markdown(f"**{t('Top 16 候选')}**")
    render_chart(score_bar(frame, "variant", score_col, limit=16))
with right_chart:
    st.markdown(f"**{t('零样本分数分布')}**")
    render_chart(score_histogram(frame[score_col]))

st.subheader(t("本次上传数据的数值精度（参考）"))
evaluation = payload["evaluation"]
if evaluation.get("available"):
    metric_row(
        [
            (t("五折 RMSE"), f"{evaluation['rmse_mean']:.3f}"),
            (t("五折 MAE"), f"{evaluation['mae_mean']:.3f}"),
            ("R²", "—" if evaluation["r2"] is None else f"{evaluation['r2']:.2f}"),
            (t("参与样本"), str(evaluation["n_samples"])),
        ]
    )
    st.caption(t("随机五折只衡量数值拟合，不代表候选排序能力；挑候选请以名次为准。"))
else:
    st.info(evaluation.get("message", tb("predict_no_labels")))

st.subheader(t("候选明细"))
display_columns = [
    col
    for col in (
        "rank",
        "variant",
        "mutation_list",
        score_col,
        "rank_model",
        "pred_activity",
        "activity_rel",
        "activity",
        "cv_error",
    )
    if col in frame.columns
]
st.dataframe(
    frame[display_columns],
    hide_index=True,
    width="stretch",
    height=420,
    column_config={
        "rank": st.column_config.NumberColumn(t("零样本名次"), width="small"),
        "variant": st.column_config.TextColumn(t("突变体"), width="medium"),
        "mutation_list": st.column_config.TextColumn(t("突变列表"), width="medium"),
        score_col: st.column_config.NumberColumn(t("零样本分数"), format="%.3f"),
        "rank_model": st.column_config.NumberColumn(t("模型名次"), width="small"),
        "pred_activity": st.column_config.NumberColumn(t("预测活性"), format="%.3f"),
        "activity_rel": st.column_config.NumberColumn(t("真实相对活性"), format="%.3f"),
        "activity": st.column_config.NumberColumn(t("真实活性"), format="%.3f"),
        "cv_error": st.column_config.NumberColumn(t("预测误差"), format="%.3f"),
    },
)
extra = (
    tb("predict_value_extra", n=evaluation["n_samples"])
    if evaluation.get("available")
    else tb("predict_value_extra_none")
)
st.caption(tb("predict_value_caption", extra=extra))

st.download_button(
    t("下载结果 CSV"),
    frame.to_csv(index=False).encode("utf-8-sig"),
    f"ugt91c1_predictions_{datetime.now():%Y%m%d_%H%M}.csv",
    "text/csv",
    type="primary",
    icon=":material/download:",
    key="download_predictions",
)
