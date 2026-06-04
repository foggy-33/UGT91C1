import io
import sys
import threading
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import esm
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prior_features import PriorFeatureBuilder  # noqa: E402


MODEL_PATH = BASE_DIR / "models" / "xgb_esm.pkl"
METRICS_PATH = None
OUTPUT_DIR = BASE_DIR / "results" / "web_predictions"
ALLOWED_EXTENSIONS = {".csv"}
RESULT_SOURCES = {
    "esm": BASE_DIR / "results" / "predictions.csv",
    "family": BASE_DIR / "results" / "predictions_family_model_filtered_by_ddg.csv",
    "family_candidates": BASE_DIR / "results" / "预测结果-家族支持候选.csv",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

model_pack = joblib.load(MODEL_PATH)
xgb_model = model_pack["model"] if isinstance(model_pack, dict) else model_pack
model_meta = model_pack if isinstance(model_pack, dict) else {}
prior_builder = PriorFeatureBuilder(BASE_DIR)
ESM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESM_MODEL, ESM_ALPHABET = esm.pretrained.esm2_t33_650M_UR50D()
ESM_MODEL = ESM_MODEL.to(ESM_DEVICE)
ESM_MODEL.eval()
ESM_BATCH_CONVERTER = ESM_ALPHABET.get_batch_converter()
ESM_BATCH_SIZE = 4
JOBS = {}


def _set_job(job_id, **updates):
    job = JOBS.get(job_id)
    if job is not None:
        job.update(updates)


def _model_evaluation():
    folds = [float(x) for x in (model_meta.get("cv_rmse_all") or [])]
    mean_rmse = model_meta.get("cv_rmse")
    if mean_rmse is None:
        mean_rmse = float(np.mean(folds)) if folds else 0.0
    else:
        mean_rmse = float(mean_rmse)
    std_rmse = float(np.std(folds)) if folds else 0.0
    n_train = None

    if METRICS_PATH and METRICS_PATH.exists():
        try:
            metrics = pd.read_csv(METRICS_PATH)
            if len(metrics):
                row = metrics.iloc[0]
                mean_rmse = float(row.get("cv_rmse_mean", mean_rmse))
                std_rmse = float(row.get("cv_rmse_std", std_rmse))
                n_train = int(row["n_train"]) if "n_train" in row and pd.notna(row["n_train"]) else n_train
        except Exception:
            pass

    accuracy_index = 1 / (1 + mean_rmse) if mean_rmse >= 0 else 0
    return {
        "method": "5-fold cross validation",
        "folds": folds,
        "fold_count": len(folds) or 5,
        "rmse_mean": mean_rmse,
        "rmse_std": std_rmse,
        "accuracy_index": float(accuracy_index),
        "target": model_meta.get("target", "activity_rel"),
        "n_train": n_train,
    }


def _choose_target_column(df):
    for col in ("activity_rel", "activity"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() >= 5:
                return col
    return None


def _run_upload_cv(x_matrix, df):
    target_col = _choose_target_column(df)
    if target_col is None:
        return {
            "available": False,
            "message": "本次上传数据未提供至少 5 条可用的 activity_rel/activity 标签，无法基于上传文件计算五折精准度。",
            "target": None,
            "folds": [],
        }, None

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = y.notna().to_numpy()
    x_valid = x_matrix[valid_mask]
    y_valid = y[valid_mask].to_numpy(dtype=float)
    source_index = df.index[valid_mask].to_numpy()

    if len(y_valid) < 5:
        return {
            "available": False,
            "message": "有效标签少于 5 条，无法执行五折交叉验证。",
            "target": target_col,
            "folds": [],
        }, None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.full(len(df), np.nan, dtype=float)
    folds = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(x_valid), start=1):
        model = clone(xgb_model)
        model.fit(x_valid[train_idx], y_valid[train_idx])
        pred = model.predict(x_valid[test_idx])
        true = y_valid[test_idx]
        oof_pred[source_index[test_idx]] = pred

        rmse = mean_squared_error(true, pred) ** 0.5
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred) if len(true) > 1 else np.nan
        folds.append(
            {
                "fold": fold_id,
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": None if np.isnan(r2) else float(r2),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )

    cv_rows = df.copy()
    cv_rows["cv_pred_activity"] = oof_pred
    cv_rows["cv_error"] = pd.to_numeric(cv_rows[target_col], errors="coerce") - cv_rows["cv_pred_activity"]

    valid_pred = cv_rows["cv_pred_activity"].notna()
    y_true = pd.to_numeric(cv_rows.loc[valid_pred, target_col], errors="coerce").to_numpy(dtype=float)
    y_pred = cv_rows.loc[valid_pred, "cv_pred_activity"].to_numpy(dtype=float)
    rmse_mean = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae_mean = float(mean_absolute_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    return {
        "available": True,
        "message": "已基于本次上传 CSV 完成五折交叉验证。",
        "method": "uploaded CSV 5-fold cross validation",
        "target": target_col,
        "folds": folds,
        "rmse_mean": rmse_mean,
        "mae_mean": mae_mean,
        "r2": None if np.isnan(r2) else float(r2),
        "rmse_std": float(np.std([x["rmse"] for x in folds])),
        "accuracy_index": float(1 / (1 + rmse_mean)),
        "n_samples": int(len(y_true)),
    }, cv_rows


def _read_upload_bytes(content, filename):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("请上传 CSV 文件。")
    if not content:
        raise ValueError("上传文件为空。")

    try:
        return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(content), encoding="gbk")


def _read_upload(file_storage):
    return _read_upload_bytes(file_storage.read(), file_storage.filename)


def _prepare_candidates(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "variant" not in df.columns:
        raise ValueError("CSV 至少需要包含 variant 列，例如 F208M 或 F208M;V129A。")
    if "sequence" not in df.columns:
        raise ValueError("CSV 必须包含 sequence 列用于生成 ESM embedding。")

    df["variant"] = df["variant"].astype(str).str.strip()
    df = df[df["variant"].ne("") & df["variant"].str.lower().ne("nan")].reset_index(drop=True)
    if df.empty:
        raise ValueError("没有找到有效的突变体记录。")

    df["sequence"] = df["sequence"].astype(str).str.strip()
    seq_mask = df["sequence"].ne("") & df["sequence"].str.lower().ne("nan")
    if not seq_mask.all():
        missing = int((~seq_mask).sum())
        raise ValueError(f"发现 {missing} 条记录缺少 sequence，无法生成 embedding。")

    if "mutation_list" not in df.columns:
        df["mutation_list"] = df["variant"].str.replace("-", ";", regex=False)
    return df


def _as_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _records_for_ui(df, limit=80):
    preferred = [
        "rank",
        "variant",
        "mutation_list",
        "activity_rel",
        "activity",
        "pred_activity",
        "pred_activity_family",
        "cv_pred_activity",
        "cv_error",
        "family_support",
        "homolog_pool_size",
        "source",
    ]
    cols = [c for c in preferred if c in df.columns]
    return [
        {col: _as_json_value(row[col]) for col in cols}
        for _, row in df.head(limit).iterrows()
    ]


def _score_column(df):
    for col in ("pred_activity_rel", "pred_activity_family", "pred_activity"):
        if col in df.columns:
            return col
    raise ValueError("预测结果文件中没有找到 pred_activity_rel / pred_activity_family / pred_activity 列。")


def _parse_positions(variants):
    import re

    counts = {}
    for variant in variants:
        for pos in re.findall(r"[A-Z](\d+)[A-Z]", str(variant)):
            counts[pos] = counts.get(pos, 0) + 1
    return [
        {"position": pos, "count": count}
        for pos, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def _histogram(values, bins=10):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return []
    counts, edges = np.histogram(values, bins=bins)
    return [
        {
            "label": f"{edges[i]:.2f}-{edges[i + 1]:.2f}",
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def _result_summary(source):
    path = RESULT_SOURCES.get(source)
    if path is None:
        raise ValueError("未知结果来源。")
    if not path.exists():
        raise ValueError(f"结果文件不存在：{path.name}")

    df = pd.read_csv(path)
    score_col = _score_column(df)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[df[score_col].notna()].sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    top = df.head(20)
    top_records = [
        {
            "rank": int(row["rank"]),
            "variant": _as_json_value(row.get("variant")),
            "score": float(row[score_col]),
            "family_support": _as_json_value(
                row.get("family_support", row.get("family_support_sum", None))
            ),
            "source": _as_json_value(row.get("source", None)),
            "mutation_list": _as_json_value(row.get("mutation_list", None)),
        }
        for _, row in top.iterrows()
    ]

    support_col = "family_support" if "family_support" in df.columns else "family_support_sum"
    support_points = []
    if support_col in df.columns:
        tmp = df[["variant", score_col, support_col]].copy()
        tmp[support_col] = pd.to_numeric(tmp[support_col], errors="coerce")
        tmp = tmp[tmp[support_col].notna()].head(120)
        support_points = [
            {
                "variant": str(row["variant"]),
                "score": float(row[score_col]),
                "support": float(row[support_col]),
            }
            for _, row in tmp.iterrows()
        ]

    return {
        "source": source,
        "file": path.name,
        "score_col": score_col,
        "total": int(len(df)),
        "top_score": float(df[score_col].max()) if len(df) else 0.0,
        "median_score": float(df[score_col].median()) if len(df) else 0.0,
        "mean_score": float(df[score_col].mean()) if len(df) else 0.0,
        "top": top_records,
        "histogram": _histogram(df[score_col].to_numpy(), bins=12),
        "positions": _parse_positions(df["variant"].head(300).tolist())[:16] if "variant" in df.columns else [],
        "support_points": support_points,
    }


def _build_esm_embeddings(df: pd.DataFrame):
    sequences = df["sequence"].tolist()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(sequences), ESM_BATCH_SIZE):
            batch = [("protein", s) for s in sequences[start:start + ESM_BATCH_SIZE]]
            _, _, tokens = ESM_BATCH_CONVERTER(batch)
            tokens = tokens.to(ESM_DEVICE)
            out = ESM_MODEL(tokens, repr_layers=[33])
            reps = out["representations"][33]
            for i in range(reps.shape[0]):
                emb = reps[i, 1:-1].mean(0).float().cpu().numpy()
                embeddings.append(emb)
    return np.vstack(embeddings)


def _prediction_payload(raw_df, progress=None):
    def report(step, message):
        if progress is not None:
            progress(step, message)

    report(0, "读取上传 CSV")
    df = _prepare_candidates(raw_df)

    report(1, "生成 ESM embedding")
    x_esm = _build_esm_embeddings(df)
    esm_dim = model_meta.get("esm_dim")
    if esm_dim is not None and int(x_esm.shape[1]) != int(esm_dim):
        raise ValueError(f"ESM embedding 维度不匹配：期望 {esm_dim}，实际 {x_esm.shape[1]}")

    report(2, "构建模型特征")
    x_prior, feature_names = prior_builder.build_matrix(df)
    x_all = np.hstack([x_esm, x_prior])

    report(3, "执行五折预测")
    upload_eval, cv_rows = _run_upload_cv(x_all, df)

    report(4, "生成活性排序")
    pred = cv_rows["cv_pred_activity"].to_numpy(dtype=float) if cv_rows is not None else xgb_model.predict(x_all)
    out = cv_rows.copy() if cv_rows is not None else df.copy()
    out["pred_activity"] = pred
    out["pred_activity_family"] = pred
    out = out.sort_values("pred_activity_family", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    report(5, "写入预测报告")
    run_id = uuid.uuid4().hex[:12]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"{run_id}_all.csv"
    out.to_csv(all_path, index=False, encoding="utf-8-sig")

    score_col = out["pred_activity_family"]
    top_score = float(score_col.max()) if len(score_col) else 0.0
    median_score = float(score_col.median()) if len(score_col) else 0.0

    return {
        "run_id": run_id,
        "counts": {
            "input": int(len(df)),
            "features": int(x_all.shape[1]),
        },
        "summary": {
            "top_score": top_score,
            "median_score": median_score,
            "target": model_meta.get("target", "activity_rel"),
            "cv_rmse": model_meta.get("cv_rmse"),
        },
        "evaluation": upload_eval,
        "records": _records_for_ui(out),
        "downloads": {
            "all": f"/api/download/{run_id}/all",
        },
    }


def _run_prediction_job(job_id, content, filename):
    try:
        def progress(step, message):
            _set_job(job_id, step=step, message=message)

        raw_df = _read_upload_bytes(content, filename)
        result = _prediction_payload(raw_df, progress=progress)
        _set_job(job_id, state="done", step=5, message="流程完成", result=result)
    except Exception as exc:
        _set_job(job_id, state="error", message=str(exc))


@app.get("/")
def index():
    return render_template(
        "index.html",
        target=model_meta.get("target", "activity_rel"),
        cv_rmse=model_meta.get("cv_rmse"),
        feature_count=len(model_meta.get("feature_names", []) or []),
        evaluation=_model_evaluation(),
    )


@app.get("/prediction-results")
def prediction_results():
    return render_template("prediction_results.html")


@app.get("/api/results/summary")
def results_summary():
    source = request.args.get("source", "esm")
    try:
        return jsonify(_result_summary(source))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/sample")
def sample_csv():
    sample = pd.DataFrame(
        {
            "variant": ["F208M", "F208M;V129A", "M15I", "I364Q"],
            "sequence": ["", "", "", ""],
            "activity_rel": ["", "", "", ""],
        }
    )
    data = sample.to_csv(index=False).encode("utf-8-sig")
    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="ugt_prediction_template.csv",
    )


@app.post("/api/predict")
def predict():
    try:
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("请先选择一个 CSV 文件。")
        raw_df = _read_upload(upload)
        return jsonify(_prediction_payload(raw_df))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/predict/start")
def predict_start():
    try:
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("请先选择一个 CSV 文件。")

        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "state": "running",
            "step": 0,
            "message": "任务已创建",
            "result": None,
        }
        thread = threading.Thread(
            target=_run_prediction_job,
            args=(job_id, upload.read(), upload.filename),
            daemon=True,
        )
        thread.start()
        return jsonify({"job_id": job_id})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/predict/status/<job_id>")
def predict_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在。"}), 404
    return jsonify(job)


@app.get("/api/download/<run_id>/<kind>")
def download(run_id, kind):
    if kind != "all":
        return jsonify({"error": "未知导出类型。"}), 404
    path = OUTPUT_DIR / f"{run_id}_{kind}.csv"
    if not path.exists():
        return jsonify({"error": "结果文件不存在或已被移动。"}), 404
    return send_file(path, as_attachment=True, download_name="预测结果.csv", mimetype="text/csv")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
