"""UGT91C1 突变体活性预测平台的共享业务逻辑。

页面脚本（app_pages/）只负责界面；模型加载、特征构建、零样本打分、五折评估与
结果汇总集中在本模块，避免同一份逻辑散落在多个页面里。

排序口径（重要）：本平台用于挑候选突变体，排序结果才是决策依据；
预测绝对值不参与阈值筛选。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from i18n import t, tb  # noqa: E402
from prior_features import PriorFeatureBuilder  # noqa: E402

MODEL_PATH = BASE_DIR / "models" / "xgb_esm.pkl"
WT_FASTA = BASE_DIR / "data" / "wt.fasta"

BACKGROUND_SITES = {208}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
ESM_BATCH_SIZE = 4
REPR_LAYER = 33

RESULT_SOURCES = {
    "zeroshot": {
        "label": "零样本分数排序（第一道排序器）",
        "path": BASE_DIR / "results" / "zeroshot_candidate_ranking.csv",
        "note": "参数无关的 masked-marginal 分数；按位点留出评测下排序能力最强。",
        "short": "参数无关零样本分数",
    },
    "esm": {
        "label": "ESM + XGBoost 预测结果",
        "path": BASE_DIR / "results" / "predictions.csv",
        "note": "线上配方（ESM mean-pool + 先验特征 + XGBoost）对候选库的外推预测。",
        "short": "线上配方外推预测",
    },
    "family": {
        "label": "家族模型预测结果",
        "path": BASE_DIR / "results" / "predictions_family_model_filtered_by_ddg.csv",
        "note": "家族进化先验模型结果，并已通过 ΔΔG 硬过滤。",
        "short": "家族先验 + ΔΔG 过滤",
    },
    "family_candidates": {
        "label": "家族支持候选结果",
        "path": BASE_DIR / "results" / "预测结果-家族支持候选.csv",
        "note": "基于同源家族支持位点扩增出的候选及其打分。",
        "short": "家族支持位点扩增",
    },
}

RANK_ONLY_NOTE = "仅按名次取前 N 个候选，勿用预测绝对值做阈值筛选"

# 「候选推荐」页读取的预计算表（由 scripts/build_candidate_recommendations.py 产出）。
RECOMMEND_PATH = BASE_DIR / "results" / "candidate_recommendations.csv"
COVERAGE_LABELS = {
    "all_seen": "已见位点",
    "mixed": "已见+未见",
    "none_seen": "未见位点",
    "background_only": "仅背景位点",
}



# --------------------------------------------------------------------------- #
# 重训互斥锁
# --------------------------------------------------------------------------- #
# 单用户本地使用时无所谓，但实验室多人共用一台机器时，两个人同时点
# 「入库并重训」会交错写训练集、并互相覆盖 ESM 特征缓存与候选模型目录，
# 因此用锁文件保证同一时间只有一个人重训。
RETRAIN_LOCK = BASE_DIR / "models" / ".retrain.lock"
RETRAIN_LOCK_STALE_SECONDS = 30 * 60


def _retrain_lock_is_stale() -> bool:
    """锁文件是否已过期（进程异常退出留下的残留）。"""
    try:
        age = datetime.now().timestamp() - RETRAIN_LOCK.stat().st_mtime
    except OSError:
        return False
    return age >= RETRAIN_LOCK_STALE_SECONDS


def acquire_retrain_lock() -> bool:
    """抢占重训锁；返回 False 表示已有其他人在重训。

    超过 ``RETRAIN_LOCK_STALE_SECONDS`` 未释放的锁视为残留并自动接管，
    避免一次崩溃之后永远无法重训。注意：调用方在重训结束后必须调用
    ``release_retrain_lock()``（包括每个失败分支）。
    """
    RETRAIN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(RETRAIN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if not _retrain_lock_is_stale():
            return False
        RETRAIN_LOCK.unlink(missing_ok=True)
        try:
            fd = os.open(RETRAIN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
    with os.fdopen(fd, "w") as handle:
        handle.write(f"pid={os.getpid()} at={datetime.now():%Y-%m-%d %H:%M:%S}")
    return True


def release_retrain_lock() -> None:
    """释放重训锁；未持有时调用也不报错。"""
    try:
        RETRAIN_LOCK.unlink(missing_ok=True)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# 模型与资源
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="正在加载 ESM-2 与 XGBoost 模型…")
def get_resources() -> dict:
    """加载平台所需的全部长生命周期资源（模型、先验构建器、ESM）。"""
    import esm

    pack = joblib.load(MODEL_PATH)
    model = pack["model"] if isinstance(pack, dict) else pack
    meta = pack if isinstance(pack, dict) else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device).eval()

    return {
        "xgb_model": model,
        "meta": meta,
        "feature_names": list(meta.get("feature_names", []) or []),
        "prior_builder": PriorFeatureBuilder(BASE_DIR),
        "esm_model": esm_model,
        "alphabet": alphabet,
        "converter": alphabet.get_batch_converter(),
        "device": device,
    }


def model_evaluation() -> dict:
    """模型泛化能力。

    主指标为按位点留出（外推到新突变位点）协议下的排序能力；
    RMSE / R² 只反映数值精度，仅作参考。
    """

    def num(key):
        value = get_resources()["meta"].get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(value) else value

    def fmt(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    meta = get_resources()["meta"]
    spearman = num("cv_spearman_macro")
    auc = num("cv_auc_macro")
    recall10 = num("cv_recall10_macro")
    rmse = num("cv_rmse")

    return {
        "target": meta.get("target", "activity_rel"),
        "target_note": meta.get("target_note", ""),
        "protocol": meta.get("cv_protocol", "leave-one-site-out"),
        "n_train": meta.get("n_train"),
        "n_sites": meta.get("cv_n_sites"),
        "n_variants": meta.get("cv_n_variants"),
        "spearman_macro": spearman,
        "auc_macro": auc,
        "recall10_macro": recall10,
        "rmse": rmse,
        "r2": num("cv_r2"),
        "rmse_protocol": meta.get("cv_rmse_protocol", "5-fold KFold"),
        "rank_only_note": meta.get("rank_only_note", RANK_ONLY_NOTE),
        "text": {
            "spearman_macro": fmt(spearman),
            "auc_macro": fmt(auc),
            "recall10_macro": fmt(recall10, 2),
            "rmse": fmt(rmse),
            "r2": fmt(num("cv_r2"), 2),
        },
    }


def model_params_text() -> str:
    params = get_resources()["meta"].get("model_params", {})
    if not params:
        return "XGBoost"
    return tb(
        "core_model_params",
        n=params.get("n_estimators"),
        depth=params.get("max_depth"),
        lr=params.get("learning_rate"),
    )


# --------------------------------------------------------------------------- #
# WT 序列与突变解析
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def wt_sequence() -> str:
    if not WT_FASTA.exists():
        return ""
    seq = []
    with open(WT_FASTA, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith(">"):
                seq.append(line)
    return "".join(seq)


def parse_mutations(row) -> list[tuple[int, str, str]]:
    """把 variant / mutation_list 解析为 [(位点, WT 氨基酸, 突变氨基酸)]。"""
    raw = row.get("mutation_list")
    if pd.isna(raw) or not str(raw).strip():
        raw = str(row.get("variant", "")).replace("-", ";")
    out = []
    for token in str(raw).split(";"):
        match = MUT_RE.match(token.strip())
        if match:
            out.append((int(match.group(2)), match.group(1), match.group(3)))
    return out


def build_sequence(mutations) -> str:
    """按突变列表在 WT 序列上生成突变体序列。"""
    seq = list(wt_sequence())
    for pos, _, mut_aa in mutations:
        if 1 <= pos <= len(seq):
            seq[pos - 1] = mut_aa
    return "".join(seq)


# --------------------------------------------------------------------------- #
# 上传数据准备
# --------------------------------------------------------------------------- #
def prepare_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化上传 CSV；缺失的 sequence 按突变列表在 WT 上补齐。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "variant" not in df.columns:
        raise ValueError(t("CSV 至少需要包含 variant 列，例如 F208M 或 F208M;V129A。"))
    df["variant"] = df["variant"].astype(str).str.strip()
    df = df[df["variant"].ne("") & df["variant"].str.lower().ne("nan")].reset_index(drop=True)
    if df.empty:
        raise ValueError(t("没有找到有效的突变体记录。"))

    if "mutation_list" not in df.columns:
        df["mutation_list"] = df["variant"].str.replace("-", ";", regex=False)

    seq = df["sequence"].astype(str).str.strip() if "sequence" in df.columns else pd.Series("", index=df.index)
    missing = seq.eq("") | seq.str.lower().eq("nan")
    if missing.any():
        if not wt_sequence():
            raise ValueError(tb("core_missing_seq", n=int(missing.sum())))
        seq.loc[missing] = df.loc[missing].apply(
            lambda row: build_sequence(parse_mutations(row)), axis=1
        )
    df["sequence"] = seq
    return df


def read_upload_bytes(content: bytes) -> pd.DataFrame:
    import io

    if not content:
        raise ValueError(t("上传文件为空。"))
    try:
        return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(content), encoding="gbk")


def sample_template() -> bytes:
    sample = pd.DataFrame(
        {
            "variant": ["F208M", "F208M;V129A", "M15I", "I364Q"],
            "sequence": ["", "", "", ""],
            "activity_rel": ["", "", "", ""],
        }
    )
    return sample.to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------------------- #
# 特征与打分
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def esm_embeddings(sequences: tuple[str, ...]) -> np.ndarray:
    """ESM-2 表征：逐序列取最后一层隐状态、去除首尾特殊 token 后平均池化。"""
    res = get_resources()
    esm_model, converter, device = res["esm_model"], res["converter"], res["device"]

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(sequences), ESM_BATCH_SIZE):
            batch = [("protein", s) for s in sequences[start : start + ESM_BATCH_SIZE]]
            _, _, tokens = converter(batch)
            out = esm_model(tokens.to(device), repr_layers=[REPR_LAYER])
            reps = out["representations"][REPR_LAYER]
            for i in range(reps.shape[0]):
                embeddings.append(reps[i, 1:-1].mean(0).float().cpu().numpy())
    return np.vstack(embeddings)


_SITE_LOGP: dict[int, np.ndarray] = {}
_SITE_LOCK = threading.Lock()


def site_logp_table(positions) -> dict[int, np.ndarray]:
    """对 WT 序列逐位点 mask，得到该位点的 log-softmax 分布（按位点缓存）。"""
    seq = wt_sequence()
    if not seq:
        return {}
    missing = sorted({int(p) for p in positions if 1 <= int(p) <= len(seq)} - set(_SITE_LOGP))
    if missing:
        res = get_resources()
        esm_model, alphabet, converter, device = (
            res["esm_model"], res["alphabet"], res["converter"], res["device"],
        )
        _, _, wt_tokens = converter([("wt", seq)])
        chunk = 16
        with _SITE_LOCK, torch.no_grad():
            for start in range(0, len(missing), chunk):
                batch_sites = missing[start : start + chunk]
                tokens = wt_tokens.repeat(len(batch_sites), 1).to(device)
                for j, pos in enumerate(batch_sites):
                    tokens[j, pos] = alphabet.mask_idx
                logits = esm_model(tokens)["logits"]
                for j, pos in enumerate(batch_sites):
                    _SITE_LOGP[pos] = torch.log_softmax(logits[j, pos].float(), dim=-1).cpu().numpy()
    return _SITE_LOGP


def zeroshot_scores(df: pd.DataFrame) -> pd.DataFrame:
    """masked-marginal 零样本分数：Σ [log p(mut) − log p(wt)]。

    背景位点（208）在所有候选间近似常数，因此额外给出剔除背景位点的版本，
    并把它作为主排序键。
    """
    mutations = [parse_mutations(row) for _, row in df.iterrows()]
    table = site_logp_table([pos for muts in mutations for pos, _, _ in muts])
    aa_idx = {aa: get_resources()["alphabet"].get_idx(aa) for aa in AA_ORDER}

    rows = []
    for muts in mutations:
        values, non_bg = [], []
        for pos, wt_aa, mut_aa in muts:
            logp = table.get(pos)
            if logp is None:
                continue
            delta = float(logp[aa_idx[mut_aa]] - logp[aa_idx[wt_aa]])
            values.append(delta)
            if pos not in BACKGROUND_SITES:
                non_bg.append(delta)
        rows.append(
            {
                "zs_sum": sum(values),
                "zs_nonbg_sum": sum(non_bg),
                "zs_mean": float(np.mean(values)) if values else 0.0,
                "zs_nonbg_mean": float(np.mean(non_bg)) if non_bg else 0.0,
            }
        )
    return pd.DataFrame(rows, index=df.index)


# --------------------------------------------------------------------------- #
# 交叉验证与候选排序
# --------------------------------------------------------------------------- #
def choose_target_column(df: pd.DataFrame) -> str | None:
    for col in ("activity_rel", "activity"):
        if col in df.columns:
            if pd.to_numeric(df[col], errors="coerce").notna().sum() >= 5:
                return col
    return None


def run_upload_cv(x_matrix, df: pd.DataFrame):
    """对上传数据做 5 折交叉验证（数值精度参考，不代表候选排序能力）。"""
    target_col = choose_target_column(df)
    if target_col is None:
        return {
            "available": False,
            "message": t("本次上传数据未提供至少 5 条可用的 activity_rel / activity 标签，无法计算数值精度。"),
        }, None

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna().to_numpy()
    x_valid, y_valid = x_matrix[valid], y[valid].to_numpy(dtype=float)
    source_index = df.index[valid].to_numpy()

    if len(y_valid) < 5:
        return {
            "available": False,
            "message": t("有效标签少于 5 条，无法执行五折交叉验证。"),
        }, None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(df), np.nan, dtype=float)
    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(x_valid), start=1):
        model = clone(get_resources()["xgb_model"])
        model.fit(x_valid[train_idx], y_valid[train_idx])
        pred = model.predict(x_valid[test_idx])
        oof[source_index[test_idx]] = pred
        true = y_valid[test_idx]
        folds.append(
            {
                "fold": fold_id,
                "rmse": float(mean_squared_error(true, pred) ** 0.5),
                "mae": float(mean_absolute_error(true, pred)),
                "r2": float(r2_score(true, pred)) if len(true) > 1 else np.nan,
            }
        )

    rows = df.copy()
    rows["cv_pred_activity"] = oof
    rows["cv_error"] = pd.to_numeric(rows[target_col], errors="coerce") - rows["cv_pred_activity"]

    valid_pred = rows["cv_pred_activity"].notna()
    y_true = pd.to_numeric(rows.loc[valid_pred, target_col], errors="coerce").to_numpy(dtype=float)
    y_pred = rows.loc[valid_pred, "cv_pred_activity"].to_numpy(dtype=float)

    return {
        "available": True,
        "message": t("已基于本次上传数据完成五折交叉验证（随机折，仅衡量数值拟合）。"),
        "target": target_col,
        "folds": folds,
        "rmse_mean": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae_mean": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "n_samples": int(len(y_true)),
    }, rows


def prediction_payload(raw_df: pd.DataFrame, progress=None) -> dict:
    """完整预测流程：特征 → 零样本 → 五折 → 候选排序。"""
    res = get_resources()

    def report(step, message):
        if progress is not None:
            progress(step, message)

    report(1, t("读取并规范化上传 CSV"))
    df = prepare_candidates(raw_df)

    report(2, t("生成 ESM 序列表征"))
    x_esm = esm_embeddings(tuple(df["sequence"].tolist()))
    esm_dim = res["meta"].get("esm_dim")
    if esm_dim is not None and int(x_esm.shape[1]) != int(esm_dim):
        raise ValueError(tb("core_dim_mismatch", a=esm_dim, b=x_esm.shape[1]))

    report(3, t("构建融合特征"))
    x_prior, _ = res["prior_builder"].build_matrix(df)
    x_all = np.hstack([x_esm, x_prior])

    report(4, t("计算零样本分数（第一道排序器）"))
    zs = zeroshot_scores(df)

    report(5, t("执行五折评估与模型预测"))
    evaluation, cv_rows = run_upload_cv(x_all, df)
    pred = (
        cv_rows["cv_pred_activity"].to_numpy(dtype=float)
        if cv_rows is not None
        else res["xgb_model"].predict(x_all)
    )

    report(6, t("生成候选名次"))
    out = cv_rows.copy() if cv_rows is not None else df.copy()
    out["pred_activity"] = pred
    out = out.join(zs)
    # 第一道排序器用零样本分数（参数无关、外推最稳），模型预测名次作为第二道参考。
    out = out.sort_values("zs_nonbg_sum", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["rank_model"] = out["pred_activity"].rank(ascending=False, method="min").astype(int)

    return {
        "frame": out,
        "counts": {"input": int(len(df)), "features": int(x_all.shape[1])},
        "evaluation": evaluation,
        "score_col": "zs_nonbg_sum",
    }


# --------------------------------------------------------------------------- #
# 结果文件汇总
# --------------------------------------------------------------------------- #
def score_column_name(df: pd.DataFrame) -> str:
    for col in ("pred_activity_rel", "pred_activity_family", "pred_activity", "zs_nonbg_sum", "zs_sum"):
        if col in df.columns:
            return col
    raise ValueError(
        t(
            "结果文件里没有找到 pred_activity_rel / pred_activity_family / "
            "pred_activity / zs_nonbg_sum / zs_sum 列。"
        )
    )


@st.cache_data(show_spinner=False, ttl="10m", max_entries=8)
def load_result_frame(source_key: str) -> pd.DataFrame:
    info = RESULT_SOURCES.get(source_key)
    if info is None:
        raise ValueError(tb("core_unknown_source", key=source_key))
    path = info["path"]
    if not path.exists():
        raise FileNotFoundError(tb("core_missing_file", name=path.name))

    df = pd.read_csv(path)
    col = score_column_name(df)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df[col].notna()].sort_values(col, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


@st.cache_data(show_spinner=False, ttl="10m", max_entries=2)
def tier_cv_metrics() -> pd.DataFrame:
    """六套方案的按位点留出对比结果。"""
    path = BASE_DIR / "results" / "tier_cv_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def source_summary(source_key: str) -> dict:
    """结果来源的概览统计（表格之外的轻量汇总）。"""
    info = RESULT_SOURCES[source_key]
    df = load_result_frame(source_key)
    col = score_column_name(df)
    scores = df[col]
    return {
        "label": info["label"],
        "note": info["note"],
        "file": info["path"].name,
        "score_col": col,
        "total": int(len(df)),
        "top_score": float(scores.max()) if len(scores) else 0.0,
        "median_score": float(scores.median()) if len(scores) else 0.0,
        "mean_score": float(scores.mean()) if len(scores) else 0.0,
    }


# --------------------------------------------------------------------------- #
# 候选推荐
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl="10m", max_entries=2)
def load_recommendations() -> pd.DataFrame:
    """读取预计算的候选推荐表（融合了模型预测与零样本分数）。"""
    if not RECOMMEND_PATH.exists():
        raise FileNotFoundError(
            t("还没有候选推荐表，请先运行：python scripts/build_candidate_recommendations.py")
        )
    # coverage_label 由页面按当前语言生成，此处只返回原始覆盖类型。
    return pd.read_csv(RECOMMEND_PATH)


def recommendation_summary(df: pd.DataFrame) -> dict:
    """候选推荐表的概览统计。"""
    has_model = df["model_pred"].notna()
    return {
        "total": int(len(df)),
        "with_model": int(has_model.sum()),
        "seen_sites": int(df["n_seen_sites"].gt(0).sum()),
        "single": int(df["n_mut"].eq(1).sum()),
        "combo": int(df["n_mut"].gt(1).sum()),
        "top_fused": float(df["fused_score"].max()) if len(df) else 0.0,
        "pool_files": "zeroshot_candidate_ranking.csv + predictions.csv",
    }


# --------------------------------------------------------------------------- #
# 数据入库与重训：路径、子进程、指标报告
# --------------------------------------------------------------------------- #
def deploy_paths() -> dict:
    """模型迭代相关产物路径的统一出处（新增产物都集中在 models/ 下）。

    报告文件统一叫 report.json / oof.csv，因此用目录区分归属：
      * models/active/            —— 现役模型的折外预测报告
      * models/candidates/run_*   —— 每次重训产出的候选模型与报告
    """
    models = BASE_DIR / "models"
    active = models / "active"
    return {
        "models": models,
        "model": MODEL_PATH,
        "history": models / "history",
        "candidates": models / "candidates",
        "active_report_dir": active,
        "report_json": active / "report.json",
        "report_csv": active / "oof.csv",
        "history_log": models / "training_history.jsonl",
        "active_log": BASE_DIR / "results" / "active_learning_log.csv",
        "dataset": BASE_DIR / "data" / "train_dataset.csv",
        "ingest_script": SCRIPTS_DIR / "ingest_batch.py",
        "features_script": SCRIPTS_DIR / "extract_features_v2.py",
        "retrain_script": SCRIPTS_DIR / "retrain_deploy_model.py",
    }


def new_candidate_dir() -> Path:
    """为一次重训分配一个带时间戳的候选目录。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = deploy_paths()["candidates"] / f"run_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_candidate_dir() -> Path | None:
    """最近一次重训的候选目录（按目录名倒序，名字含时间戳）。"""
    root = deploy_paths()["candidates"]
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")])
    return runs[-1] if runs else None


def run_stage_subprocess(args, *, tail_lines: int = 400):
    """逐行执行外部脚本并把输出实时 yield 出来，供页面在 st.status 里滚动展示。

    args 形如 [脚本路径, "--flag", ...]，统一用当前解释器执行、工作目录为项目根。
    子进程非 0 退出时抛 RuntimeError，方便页面中断后续阶段。
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-u", *[str(a) for a in args]]
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    buffer: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        text = line.rstrip()
        buffer.append(text)
        if len(buffer) > tail_lines:
            buffer = buffer[-tail_lines:]
        yield text
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            tb(
                "core_subproc_fail",
                code=proc.returncode,
                name=" ".join(Path(a).name for a in args[:1]),
                tail="\n".join(buffer[-40:]),
            )
        )


@st.cache_data(show_spinner=False, ttl="30m", max_entries=2)
def load_training_report() -> tuple[dict | None, pd.DataFrame]:
    """现役模型的按位点留出折外预测报告（由重训/补算产出）。"""
    paths = deploy_paths()
    report = None
    if paths["report_json"].exists():
        report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    oof = pd.DataFrame()
    if paths["report_csv"].exists():
        oof = pd.read_csv(paths["report_csv"])
    return report, oof


@st.cache_data(show_spinner=False, ttl="1m", max_entries=2)
def load_active_learning_log() -> pd.DataFrame:
    """历轮「入库前预测 vs 实测」记录。"""
    path = deploy_paths()["active_log"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, ttl="1m", max_entries=2)
def load_training_history() -> pd.DataFrame:
    """模型采纳历史（每行一次采纳）。"""
    path = deploy_paths()["history_log"]
    if not path.exists():
        return pd.DataFrame()
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(entries)


def load_candidate_report(candidate_dir: Path) -> tuple[dict | None, pd.DataFrame]:
    """读取某个候选目录的 report.json 与 oof.csv。"""
    candidate_dir = Path(candidate_dir)
    report_path = candidate_dir / "report.json"
    oof_path = candidate_dir / "oof.csv"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    oof = pd.read_csv(oof_path) if oof_path.exists() else pd.DataFrame()
    return report, oof


def report_metric_row(report: dict | None) -> dict:
    """把 report.json 的主指标整理成可展示的字典。"""
    metrics = (report or {}).get("metrics") or {}

    def fmt(key, digits=3):
        value = metrics.get(key)
        return "-" if value is None else f"{float(value):.{digits}f}"

    return {
        "available": report is not None,
        "spearman_macro": fmt("spearman_macro"),
        "auc_macro": fmt("auc_macro"),
        "recall10_macro": fmt("recall10_macro", 2),
        "spearman_pooled": fmt("spearman_pooled"),
        "rmse": "-" if (report or {}).get("rmse") is None else f"{float(report['rmse']):.3f}",
        "n_variants": (report or {}).get("n_variants"),
        "n_sites": (report or {}).get("n_sites"),
        "dataset_rows": (report or {}).get("dataset_rows"),
        "generated_at": (report or {}).get("generated_at", "-"),
        "selected_label": (report or {}).get("selected_label"),
    }


# --------------------------------------------------------------------------- #
# 数据入库与重训：入库前快照、采纳候选
# --------------------------------------------------------------------------- #
def score_with_current_model(df: pd.DataFrame) -> pd.DataFrame:
    """用**当前线上模型**给一批突变体打分，作为「入库前预测」快照。

    返回带 pred_before / zero_score / rank_before 的 DataFrame。
    """
    res = get_resources()
    frame = prepare_candidates(df)
    x_esm = esm_embeddings(tuple(frame["sequence"].tolist()))
    esm_dim = res["meta"].get("esm_dim")
    if esm_dim is not None and int(x_esm.shape[1]) != int(esm_dim):
        raise ValueError(tb("core_dim_mismatch", a=esm_dim, b=x_esm.shape[1]))
    x_prior, _ = res["prior_builder"].build_matrix(frame)
    x_all = np.hstack([x_esm, x_prior])

    frame = frame.join(zeroshot_scores(frame))
    frame["pred_before"] = res["xgb_model"].predict(x_all)
    frame["zero_score"] = frame["zs_nonbg_sum"]

    # 顺带记下该突变体在候选推荐表里的原有名次（查不到则留空）
    ranks: dict[str, int] = {}
    if RECOMMEND_PATH.exists():
        rec = pd.read_csv(RECOMMEND_PATH, usecols=["rank", "variant"])
        ranks = dict(zip(rec["variant"].astype(str), rec["rank"].astype(int)))
    frame["rank_before"] = frame["variant"].astype(str).map(ranks)
    return frame


def append_active_learning_log(rows: pd.DataFrame, *, batch_label: str, round_id: str | None = None) -> str:
    """把入库前的预测快照追加进 results/active_learning_log.csv。"""
    path = deploy_paths()["active_log"]
    round_id = round_id or datetime.now().strftime("R%Y%m%d-%H%M%S")
    entries = pd.DataFrame(
        {
            "round_id": round_id,
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            "batch": batch_label,
            "variant": rows["variant"].astype(str),
            "n_mut": rows["n_mut"],
            "actual_activity_rel": pd.to_numeric(rows["activity_rel"], errors="coerce"),
            "pred_before": pd.to_numeric(rows["pred_before"], errors="coerce"),
            "zero_score": pd.to_numeric(rows["zero_score"], errors="coerce"),
            "rank_before": pd.to_numeric(rows.get("rank_before"), errors="coerce")
            if "rank_before" in rows.columns
            else np.nan,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        entries = pd.concat([pd.read_csv(path), entries], ignore_index=True)
    entries.to_csv(path, index=False)
    return round_id


def adopt_candidate(candidate_dir: Path, *, note: str = "") -> dict:
    """把候选模型提升为线上模型：归档旧模型 → 复制候选 → 记录采纳历史 → 清缓存。"""
    paths = deploy_paths()
    candidate_dir = Path(candidate_dir)
    cand_model = candidate_dir / "model.pkl"
    if not cand_model.exists():
        raise FileNotFoundError(tb("core_missing_model", path=candidate_dir))

    cand_report, _ = load_candidate_report(candidate_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    paths["history"].mkdir(parents=True, exist_ok=True)

    archived = None
    if paths["model"].exists():
        archived = paths["history"] / f"xgb_esm_{stamp}.pkl"
        shutil.copy2(paths["model"], archived)

    shutil.copy2(cand_model, paths["model"])
    if (candidate_dir / "report.json").exists():
        paths["active_report_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_dir / "report.json", paths["report_json"])
    if (candidate_dir / "oof.csv").exists():
        paths["active_report_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_dir / "oof.csv", paths["report_csv"])

    metrics = (cand_report or {}).get("metrics") or {}
    entry = {
        "adopted_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_dir": str(candidate_dir),
        "archived_model": str(archived) if archived else None,
        "dataset_rows": (cand_report or {}).get("dataset_rows"),
        "model_params": (cand_report or {}).get("model_params"),
        "spearman_macro": metrics.get("spearman_macro"),
        "auc_macro": metrics.get("auc_macro"),
        "recall10_macro": metrics.get("recall10_macro"),
        "rmse": (cand_report or {}).get("rmse"),
        "note": note,
    }
    with open(paths["history_log"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    st.cache_resource.clear()
    st.cache_data.clear()
    return entry


def discard_candidate(candidate_dir: Path) -> None:
    """丢弃候选产物（数据已入库，因此只删模型与报告，不动训练集）。"""
    candidate_dir = Path(candidate_dir)
    if candidate_dir.exists() and candidate_dir.is_dir() and "candidates" in candidate_dir.parts:
        shutil.rmtree(candidate_dir)

