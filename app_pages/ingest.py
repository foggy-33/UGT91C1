"""数据入库页：把新一批实验数据并入训练集，并用统一训练入口重训模型。

流程（页面自上而下即操作顺序）：
  上传新批次 CSV → 校验与批内 WT 归一化 → 预览确认 → 入库前打预测快照
  → 追加进 data/train_dataset.csv → 重建 ESM 特征缓存 → 重训候选模型
  → 并排对照现役/候选指标 → 人工决定是否采纳

安全约定：
  * 追加训练集前会先写带时间戳的备份；
  * 重训只产出候选模型（models/candidates/run_*），绝不直接覆盖线上的 xgb_esm.pkl；
  * 只有人工点「采纳新模型」才会替换线上模型，旧模型归档进 models/history/；
  * 丢弃候选不会删除已入库的实验数据。
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ingest_batch as ib  # noqa: E402

from i18n import t, tb  # noqa: E402
from prediction_core import (  # noqa: E402
    RANK_ONLY_NOTE,
    acquire_retrain_lock,
    adopt_candidate,
    deploy_paths,
    discard_candidate,
    latest_candidate_dir,
    load_candidate_report,
    load_training_history,
    model_evaluation,
    new_candidate_dir,
    release_retrain_lock,
    report_metric_row,
    run_stage_subprocess,
    score_with_current_model,
)

st.session_state.setdefault("ingest_candidate_dir", None)

paths = deploy_paths()

st.caption(tb("ingest_intro"))

st.markdown(
    tb("ingest_norm_note"),
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# 1. 格式说明与模板
# --------------------------------------------------------------------------- #
with st.expander(t("新批次 CSV 格式（点开查看 / 下载模板）"), expanded=False):
    st.markdown(tb("ingest_format_table"))
    st.markdown(tb("ingest_format_explain"))
    template = pd.DataFrame(
        {"variant": ["WT", "F208M-V129W", "F208M-H93C"], "activity": [3.5, 7.0, 9.0]}
    )
    st.download_button(
        t("下载新批次模板 CSV"),
        template.to_csv(index=False).encode("utf-8-sig"),
        "ugt91c1_new_batch_template.csv",
        "text/csv",
        icon=":material/download:",
        key="download_batch_template",
    )

# --------------------------------------------------------------------------- #
# 2. 上传与预览
# --------------------------------------------------------------------------- #
st.subheader(t("① 上传新批次数据并预览"))

upload = st.file_uploader(
    t("新批次 CSV（需含 variant 与 activity 列）"), type=["csv"], key="ingest_upload"
)

batch_default = ib.next_batch_label(paths["dataset"])
col_batch, col_wt = st.columns(2)
with col_batch:
    batch_label = st.text_input(t("批次标签"), value=batch_default, key="ingest_batch_label")
with col_wt:
    wt_input = st.number_input(
        t("批内 WT 比酶活（CSV 已含 WT 行时留 0 即可）"),
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="ingest_wt",
    )
wt_activity = None if wt_input == 0.0 else float(wt_input)

rows = pd.DataFrame(columns=ib.COLUMNS)
warnings: list[str] = []
errors: list[str] = []

if upload is None:
    st.info(t("请上传新批次 CSV 后继续。"), icon=":material/upload_file:")
    st.stop()

try:
    raw = pd.read_csv(upload, encoding="utf-8-sig")
except UnicodeDecodeError:
    upload.seek(0)
    raw = pd.read_csv(upload, encoding="gbk")

existing = ib.load_dataset(paths["dataset"])
rows, warnings, errors = ib.parse_new_batch(
    raw, batch_label, wt_activity=wt_activity, existing=existing
)

if errors:
    st.error(tb("ingest_parse_fail") + "\n".join(f"- {e}" for e in errors))
    st.stop()

for msg in warnings:
    st.warning(msg)

st.caption(
    tb(
        "ingest_dataset_caption",
        n=len(existing),
        b=sorted(existing["batch"].astype(str).unique()),
    )
)

if rows.empty:
    st.info(t("没有可入库的新记录，已在上方说明原因。"))
    st.stop()

st.markdown(tb("ingest_pending", n=len(rows), wt=f"{rows['wt_batch'].iloc[0]:g}"))
st.dataframe(
    rows[
        [
            "variant",
            "activity",
            "activity_rel",
            "sites_str",
            "n_mut",
            "is_background_only",
        ]
    ],
    hide_index=True,
    width="stretch",
    height=260,
    column_config={
        "variant": st.column_config.TextColumn(t("突变体"), width="medium"),
        "activity": st.column_config.NumberColumn(t("原始活性"), format="%.3f"),
        "activity_rel": st.column_config.NumberColumn(t("相对 WT 倍数"), format="%.3f"),
        "sites_str": st.column_config.TextColumn(t("突变位点"), width="small"),
        "n_mut": st.column_config.NumberColumn(t("突变数"), width="small"),
        "is_background_only": st.column_config.CheckboxColumn(t("仅背景位点"), width="small"),
    },
)
st.caption(t("`相对 WT 倍数` 才是进训练集的标签；WT 行自身恒为 1，作为本批次的归一化基准。"))

# --------------------------------------------------------------------------- #
# 3. 确认并执行
# --------------------------------------------------------------------------- #
st.subheader(t("② 确认入库并重训"))

col_confirm, col_grid = st.columns([1.2, 1])
with col_confirm:
    confirmed = st.checkbox(
        tb("ingest_confirm", n=len(rows)),
        key="ingest_confirm",
    )
with col_grid:
    use_grid = st.toggle(
        t("自动调优（小网格搜索）"),
        value=False,
        key="ingest_grid",
        help=tb("ingest_grid_help"),
    )

candidate_dir = st.session_state.get("ingest_candidate_dir")
if candidate_dir:
    candidate_dir = Path(candidate_dir)

run = st.button(
    t("入库并重训"),
    type="primary",
    disabled=not confirmed,
    icon=":material/model_training:",
    key="ingest_run",
)

if run:
    # 多人共用一台机器时，重训必须串行：抢不到锁就直接拒绝，不进入任何写操作。
    if not acquire_retrain_lock():
        st.error(t("检测到已有其他人在执行「入库并重训」，请等其结束后再试。"))
        st.stop()

    snapshot_rows = rows[rows["variant"].astype(str).str.upper().ne("WT")].reset_index(drop=True)
    target_dir = new_candidate_dir()

    # ---- ① 入库前快照：必须早于任何模型改动，记录「当时模型怎么预测这批」----
    with st.status(tb("ingest_step1"), expanded=True) as status:
        try:
            if snapshot_rows.empty:
                st.write(t("本批次只有 WT 行，无需快照。"))
            else:
                scored = score_with_current_model(snapshot_rows[["variant", "sequence"]])
                log_rows = snapshot_rows[["variant", "activity_rel", "n_mut"]].merge(
                    scored[["variant", "pred_before", "zero_score", "rank_before"]],
                    on="variant",
                    how="left",
                )
                round_id = None
                from prediction_core import append_active_learning_log  # 延迟导入，避免页面顶部堆叠

                round_id = append_active_learning_log(log_rows, batch_label=batch_label)
                st.write(tb("ingest_snapshot_written", round_id=round_id, n=len(log_rows)))
            status.update(label=tb("ingest_step1_done"), state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label=tb("ingest_step1_fail"), state="error")
            st.exception(exc)
            release_retrain_lock()
            st.stop()

    # ---- ② 追加进训练集 ----
    with st.status(tb("ingest_step2"), expanded=True) as status:
        try:
            info = ib.append_rows(rows, paths["dataset"])
            st.write(tb("ingest_backup", name=Path(info["backup"]).name))
            st.write(tb("ingest_rows", before=info["n_before"], after=info["n_after"]))
            status.update(label=tb("ingest_step2_done", n=info["n_after"]), state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label=tb("ingest_step2_fail"), state="error")
            st.exception(exc)
            release_retrain_lock()
            st.stop()

    # ---- ③ 重建 ESM 特征缓存（行序必须与新的 CSV 对齐，因此整体重算）----
    st.write(tb("ingest_feature_rebuild"))
    feature_holder = st.empty()
    with st.status(tb("ingest_step3_status"), expanded=True) as status:
        try:
            lines: list[str] = []
            for line in run_stage_subprocess([paths["features_script"]]):
                lines.append(line)
                feature_holder.code("\n".join(lines[-20:]), language="text")
            status.update(label=tb("ingest_step3_done"), state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label=tb("ingest_step3_fail"), state="error")
            st.error(str(exc))
            st.warning(t("数据仍在训练集里。修好问题后重跑本页即可继续，无需重复入库。"))
            release_retrain_lock()
            st.stop()

    # ---- ④ 重训候选模型（不覆盖线上 pkl）----
    retrain_args = [
        paths["retrain_script"],
        "--out",
        target_dir / "model.pkl",
        "--report-dir",
        target_dir,
    ]
    if use_grid:
        retrain_args.append("--grid")
    st.write(tb("ingest_retrain_label", name=target_dir.name))
    retrain_holder = st.empty()
    with st.status(tb("ingest_step4_status"), expanded=True) as status:
        try:
            lines = []
            for line in run_stage_subprocess(retrain_args):
                lines.append(line)
                retrain_holder.code("\n".join(lines[-20:]), language="text")
            status.update(label=tb("ingest_step4_done"), state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label=tb("ingest_step4_fail"), state="error")
            st.error(str(exc))
            st.warning(t("数据仍在训练集里，修好问题后重跑本页即可。"))
            release_retrain_lock()
            st.stop()

    release_retrain_lock()
    st.session_state["ingest_candidate_dir"] = str(target_dir)
    st.rerun()

# --------------------------------------------------------------------------- #
# 4. 现役 / 候选 指标对照
# --------------------------------------------------------------------------- #
candidate_dir = st.session_state.get("ingest_candidate_dir")
if not candidate_dir:
    fallback = latest_candidate_dir()
    if fallback is not None:
        candidate_dir = fallback
if candidate_dir:
    candidate_dir = Path(candidate_dir)

if candidate_dir and (candidate_dir / "model.pkl").exists():
    st.divider()
    st.subheader(t("③ 现役模型 vs 候选模型"))

    live = model_evaluation()
    cand_report, cand_oof = load_candidate_report(candidate_dir)
    cand = report_metric_row(cand_report)

    from ui_helpers import metric_compare_row  # 延迟导入，保持顶部 import 精简

    metric_compare_row(
        tb("ingest_live_header", n=live.get("n_train")),
        [
            (t("按位点留出 Macro Spearman"), live["text"]["spearman_macro"]),
            (t("按位点留出 Macro AUC"), live["text"]["auc_macro"]),
            ("Recall@10", live["text"]["recall10_macro"]),
            (t("RMSE（仅参考）"), live["text"]["rmse"]),
        ],
        tb("ingest_cand_header", n=cand["dataset_rows"]),
        [
            (t("按位点留出 Macro Spearman"), cand["spearman_macro"]),
            (t("按位点留出 Macro AUC"), cand["auc_macro"]),
            ("Recall@10", cand["recall10_macro"]),
            (t("RMSE（仅参考）"), cand["rmse"]),
        ],
    )

    if cand["available"]:
        st.caption(
            tb(
                "ingest_report_caption",
                generated_at=cand["generated_at"],
                n_sites=cand["n_sites"],
                n_variants=cand["n_variants"],
            )
        )
        if cand["selected_label"]:
            st.caption(tb("ingest_grid_choice", label=cand["selected_label"]))
        grid = (cand_report or {}).get("grid")
        if grid:
            st.markdown(f"**{t('自动调优：各候选配方的按位点留出表现')}**")
            grid_frame = pd.DataFrame(grid)
            grid_frame["params"] = grid_frame["params"].apply(
                lambda p: ", ".join(f"{k}={v}" for k, v in dict(p).items() if k != "verbosity")
            )
            st.dataframe(
                grid_frame.rename(
                    columns={
                        "label": t("配方"),
                        "params": t("超参数"),
                        "spearman_macro": "Macro Spearman",
                        "auc_macro": "Macro AUC",
                        "recall10_macro": "Recall@10",
                        "rmse": "RMSE",
                        "n_variants": t("留出突变体数"),
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        st.caption(tb("ingest_note_caption", note=t(RANK_ONLY_NOTE)))

    with st.expander(t("候选模型的折外预测明细（前 200 行）")):
        if cand_oof.empty:
            st.info(tb("ingest_no_oof"))
        else:
            st.dataframe(cand_oof.head(200), hide_index=True, width="stretch", height=320)

    col_adopt, col_discard = st.columns(2)
    with col_adopt:
        if st.button(
            t("采纳新模型（替换线上）"),
            type="primary",
            icon=":material/check_circle:",
            key="ingest_adopt",
            help=tb("ingest_adopt_help"),
        ):
            entry = adopt_candidate(candidate_dir)
            st.session_state["ingest_candidate_dir"] = None
            st.success(
                tb(
                    "ingest_adopt_success",
                    spearman=entry["spearman_macro"],
                    auc=entry["auc_macro"],
                    recall10=entry["recall10_macro"],
                    archived=Path(entry["archived_model"]).name
                    if entry["archived_model"]
                    else "（无）",
                )
            )
            st.info(tb("ingest_adopt_note"))
    with col_discard:
        if st.button(
            t("丢弃候选模型"),
            icon=":material/delete:",
            key="ingest_discard",
            help=tb("ingest_discard_help"),
        ):
            discard_candidate(candidate_dir)
            st.session_state["ingest_candidate_dir"] = None
            st.warning(t("已丢弃候选模型。已入库的实验数据保留在训练集里。"))

# --------------------------------------------------------------------------- #
# 5. 训练历史
# --------------------------------------------------------------------------- #
st.divider()
st.subheader(t("模型采纳历史"))
history = load_training_history()
if history.empty:
    st.caption(t("还没有采纳过任何新模型。入库重训后点「采纳新模型」即会在此留下记录。"))
else:
    show = history.rename(
        columns={
            "adopted_at": t("采纳时间"),
            "dataset_rows": t("训练样本数"),
            "spearman_macro": "Macro Spearman",
            "auc_macro": "Macro AUC",
            "recall10_macro": "Recall@10",
            "rmse": "RMSE",
            "archived_model": t("归档旧模型"),
            "candidate_dir": t("候选目录"),
            "note": t("备注"),
        }
    )
    st.dataframe(
        show[
            [
                t("采纳时间"),
                t("训练样本数"),
                "Macro Spearman",
                "Macro AUC",
                "Recall@10",
                "RMSE",
                t("归档旧模型"),
            ]
        ],
        hide_index=True,
        width="stretch",
    )
