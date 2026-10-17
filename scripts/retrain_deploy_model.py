"""P0: 用线上同配方在本地重训并覆盖 models/xgb_esm.pkl，写入诚实指标与元数据。

为什么必须重训
--------------
线上 xgb_esm.pkl 由旧版 XGBoost 序列化（当前环境 xgboost 3.0.5），反序列化时会丢失
base_score 截距，导致所有预测整体偏移（预测均值约 -0.34/0.12，而真值均值约 1.05/1.52），
预测绝对值完全不可用。本地重训即可消除该偏差。此外旧 pkl 未存任何评估字段，导致平台
首页"特征数量 / 准确度"显示为 0。

配方（与线上一致）
------------------
  特征：ESM-2 (t33 650M) mean-pool 1280 维 + 先验特征 23 维 = 1303 维
  模型：XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05, ...)
  训练集：data/train_dataset.csv（Tier 0 合并后的 67 条，批内 WT 归一化）
  标签：activity_rel（相对同批次 WT 的倍数，WT=1）

诚实评估
--------
  主指标：按位点留出（leave-one-site-out）——外推到从未见过的突变位点，与"挑候选突变体"
          目标一致。给出 pooled/macro Spearman、pooled/macro AUC、macro Recall@5/@10。
  参考值：5 折 KFold RMSE（数值精度，参考价值有限，非本任务核心）。

命令行
------
无参数运行 = 原有行为，覆盖写 models/xgb_esm.pkl：用于维护现役模型。

    python scripts/retrain_deploy_model.py

新增参数（供平台「数据入库」页调用，默认值即原有行为）：

    --out PATH        模型输出路径，默认 models/xgb_esm.pkl。
                      平台入库时指向候选路径，避免直接覆盖现役模型。
    --report-dir DIR  额外输出 report.json 与 oof.csv（折外预测明细），
                      供「对比分析」页展示；不传则只打日志、不落盘。
    --grid            小网格自动调优：线上配方 + 4 组变体，每组做同一套按位点留出
                      评估，同一数据集同协议下按 Macro Spearman 择优（并列看 Macro AUC）。
    --eval-only       只评估、不训练、不落模型；配 --model 给现役 pkl 补算一份 OOF 报告。
    --dataset PATH / --features PATH
                      指定训练集 CSV 与 ESM 特征 npy，默认取 data/ 下的正式文件。
                      这两个只用于隔离测试（指向临时副本，不碰正式数据）。

输出：默认 models/xgb_esm.pkl；--out 指定时写指定路径。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from prior_features import PriorFeatureBuilder  # noqa: E402

DS_PATH = BASE_DIR / "data" / "train_dataset.csv"
X_PATH = BASE_DIR / "data" / "feat_meanpool.npy"
MODEL_OUT = BASE_DIR / "models" / "xgb_esm.pkl"

BACKGROUND = {208}
RANDOM_STATE = 42
ACTIVE_THRESHOLD = 1.0  # activity_rel > 1 视为"高于 WT"

MODEL_PARAMS = dict(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
)

# --grid 时的变体组（第一组是线上配方本身，作为同数据集同协议的基线）
GRID_VARIANTS = [
    dict(max_depth=3),
    dict(max_depth=5),
    dict(n_estimators=400, learning_rate=0.03),
    dict(n_estimators=800),
]


def parse_sites(raw):
    if pd.isna(raw) or not str(raw).strip():
        return []
    return [int(x) for x in str(raw).split(";") if str(x).strip().isdigit()]


def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def topk_overlap(y_true, y_pred, k):
    k = min(k, len(y_true))
    op = np.argsort(-np.asarray(y_pred))[:k]
    ot = np.argsort(-np.asarray(y_true))[:k]
    return len(set(op.tolist()) & set(ot.tolist())) / k


def ranking_metrics(y_true, y_pred, batch):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    batch = np.asarray(batch)
    label = (y_true > ACTIVE_THRESHOLD).astype(int)
    out = {
        "spearman_pooled": spearman(y_true, y_pred),
        "auc_pooled": float(roc_auc_score(label, y_pred)) if 0 < label.sum() < len(label) else np.nan,
        "recall5_pooled": topk_overlap(y_true, y_pred, 5),
        "recall10_pooled": topk_overlap(y_true, y_pred, 10),
    }
    macro = {"spearman": [], "auc": [], "recall5": [], "recall10": []}
    for b in sorted(set(batch.tolist())):
        m = batch == b
        yt, yp = y_true[m], y_pred[m]
        if len(yt) < 3 or np.std(yt) == 0:
            continue
        macro["spearman"].append(spearman(yt, yp))
        lab = (yt > ACTIVE_THRESHOLD).astype(int)
        if 0 < lab.sum() < len(lab):
            macro["auc"].append(float(roc_auc_score(lab, yp)))
        macro["recall5"].append(topk_overlap(yt, yp, 5))
        macro["recall10"].append(topk_overlap(yt, yp, 10))
    for k, v in macro.items():
        out[f"{k}_macro"] = float(np.mean(v)) if v else np.nan
    return out


def _num(value, digits=6):
    """把 numpy/NaN 值转成可 JSON 序列化的数字（NaN → None）。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else round(value, digits)


def _params(**overrides):
    params = dict(MODEL_PARAMS)
    params.update(overrides)
    params["verbosity"] = 0
    return params


def grid_candidates():
    """--grid 的候选配方：线上配方 + 四组变体。"""
    return [_params()] + [_params(**v) for v in GRID_VARIANTS]


def load_inputs(dataset_path: Path, features_path: Path):
    """读取训练集 CSV 与 ESM 特征 npy，拼上先验特征。

    行序强耦合：npy 的第 i 行必须对应 CSV 的第 i 行，因此这里做一次显式校验。
    """
    df = pd.read_csv(dataset_path)
    X = np.load(features_path)
    if X.shape[0] != len(df):
        raise ValueError(
            f"特征行数({X.shape[0]}) 与训练集行数({len(df)}) 不一致："
            f"追加数据后需要先重跑 extract_features_v2.py 刷新特征缓存。"
        )
    prior, prior_names = PriorFeatureBuilder(BASE_DIR).build_matrix(df)
    X_all = np.hstack([X, prior])
    feature_names = [f"esm_meanpool_{i}" for i in range(X.shape[1])] + list(prior_names)
    return df, X_all, feature_names, prior_names


def leave_one_site_out(X_all, y, sites_list, params):
    """按位点留出：逐位点把含该位点的样本全部留出，训练其余样本后预测。"""
    eval_sites = sorted({p for s in sites_list for p in s if p not in BACKGROUND})
    has_nbg = np.array([any(p not in BACKGROUND for p in s) for s in sites_list])
    eval_idx = np.where(has_nbg)[0]

    oof = np.full(len(y), np.nan, dtype=float)
    for p in eval_sites:
        in_site = np.array([p in s for s in sites_list])
        tr, te = np.where(~in_site)[0], np.where(in_site)[0]
        if not len(te) or not len(tr):
            continue
        m = xgb.XGBRegressor(**params)
        m.fit(X_all[tr], y[tr])
        oof[te] = m.predict(X_all[te])
    return oof, eval_sites, eval_idx


def kfold_numeric(X_all, y, params):
    """5 折 KFold 的数值精度（RMSE / R²），仅作参考。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmses, r2s = [], []
    for tr, te in kf.split(X_all):
        m = xgb.XGBRegressor(**params)
        m.fit(X_all[tr], y[tr])
        pred = m.predict(X_all[te])
        rmses.append(float(mean_squared_error(y[te], pred) ** 0.5))
        ss_res = float(((y[te] - pred) ** 2).sum())
        ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
        r2s.append(1 - ss_res / ss_tot if ss_tot else np.nan)
    return rmses, r2s


def group_metrics(y, oof, groups, mask):
    """按分组键（批次 / 突变数分档）给出分层指标；样本不足或方差为 0 时留空。"""
    y = np.asarray(y, dtype=float)
    oof = np.asarray(oof, dtype=float)
    groups = np.asarray(groups)
    out = {}
    for key in sorted(set(groups[mask].tolist()), key=str):
        sel = (groups == key) & mask & np.isfinite(oof)
        yt, yp = y[sel], oof[sel]
        row = {"n": int(len(yt)), "spearman": None, "auc": None, "recall5": None, "recall10": None}
        if len(yt) >= 3 and np.std(yt) > 0:
            row["spearman"] = _num(spearman(yt, yp))
            lab = (yt > ACTIVE_THRESHOLD).astype(int)
            if 0 < lab.sum() < len(lab):
                row["auc"] = _num(roc_auc_score(lab, yp))
            row["recall5"] = _num(topk_overlap(yt, yp, 5))
            row["recall10"] = _num(topk_overlap(yt, yp, 10))
        out[str(key)] = row
    return out


def n_mut_bucket(n_mut):
    n = int(n_mut)
    return "3+" if n >= 3 else str(n)


def evaluate(df, X_all, y, params):
    """对一组配方跑完整评估，返回指标字典 + 折外预测 + 分组统计。"""
    sites_list = [parse_sites(s) for s in df["sites_str"]]
    oof, eval_sites, eval_idx = leave_one_site_out(X_all, y, sites_list, params)
    mask = np.zeros(len(df), dtype=bool)
    mask[eval_idx] = True
    site_cv = ranking_metrics(y[eval_idx], oof[eval_idx], df["batch"].to_numpy()[eval_idx])
    rmses, r2s = kfold_numeric(X_all, y, params)

    by_batch = group_metrics(y, oof, df["batch"], mask)
    buckets = np.array([n_mut_bucket(n) for n in df["n_mut"]])
    by_n_mut = group_metrics(y, oof, buckets, mask)

    return {
        "params": {k: v for k, v in params.items() if k != "verbosity"},
        "metrics": key_metrics(site_cv),
        "metrics_full": {k: _num(v) for k, v in site_cv.items()},
        "n_sites": len(eval_sites),
        "n_variants": int(len(eval_idx)),
        "rmse": _num(np.mean(rmses)),
        "rmse_all": [_num(x) for x in rmses],
        "r2": _num(np.nanmean(r2s)),
        "by_batch": by_batch,
        "by_n_mut": by_n_mut,
        "_oof": oof,
        "_eval_idx": eval_idx,
    }


def key_metrics(site_cv):
    """平台展示用的主指标子集。"""
    return {
        "spearman_pooled": _num(site_cv["spearman_pooled"]),
        "spearman_macro": _num(site_cv["spearman_macro"]),
        "auc_pooled": _num(site_cv["auc_pooled"]),
        "auc_macro": _num(site_cv["auc_macro"]),
        "recall5_macro": _num(site_cv["recall5_macro"]),
        "recall10_macro": _num(site_cv["recall10_macro"]),
    }


def selection_score(metrics):
    """网格择优：先看 Macro Spearman，并列看 Macro AUC，再并列看 pooled Spearman。"""
    def val(key):
        v = metrics.get(key)
        return -np.inf if v is None else float(v)
    return (val("spearman_macro"), val("auc_macro"), val("spearman_pooled"))


def build_oof_frame(df, result):
    return pd.DataFrame(
        {
            "variant": df["variant"],
            "batch": df["batch"],
            "n_mut": df["n_mut"],
            "activity_rel": np.asarray(df["activity_rel"], dtype=float).round(6),
            "oof_pred": np.round(result["_oof"], 6),
            "in_eval": np.isfinite(result["_oof"]),
        }
    )


def write_artifacts(report_dir: Path, report: dict, oof_frame: pd.DataFrame):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    oof_frame.to_csv(report_dir / "oof.csv", index=False)
    print("Saved:", report_dir / "report.json", report_dir / "oof.csv")


def save_model(out_path: Path, model, meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**meta, "model": model}, out_path)
    print("Saved:", out_path)


def run_eval_only(args, report_dir):
    """--eval-only：给一个现役 pkl 补算一份折外预测报告，不训练、不落模型。"""
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到要评估的模型：{model_path}")
    pack = joblib.load(model_path)
    meta = pack if isinstance(pack, dict) else {}
    params = dict(meta.get("model_params") or MODEL_PARAMS)
    params["verbosity"] = 0

    dataset_path = Path(args.dataset)
    features_path = Path(args.features)
    df, X_all, _, _ = load_inputs(dataset_path, features_path)
    y = df["activity_rel"].to_numpy(dtype=float)

    print(f"[eval-only] model={model_path}  rows={len(df)}")
    result = evaluate(df, X_all, y, params)
    print(
        f"[leave-one-site-out] sites={result['n_sites']}  variants={result['n_variants']}  "
        + "  ".join(f"{k}={v}" for k, v in result["metrics"].items())
    )
    print(f"[5-fold KFold] RMSE={result['rmse']}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "eval-only",
        "eval_model": str(model_path),
        "dataset": str(dataset_path),
        "dataset_rows": int(len(df)),
        "batches": sorted(df["batch"].astype(str).unique().tolist()),
        "cv_protocol": "leave-one-site-out",
        "model_params": result["params"],
        "metrics": result["metrics"],
        "metrics_full": result["metrics_full"],
        "n_sites": result["n_sites"],
        "n_variants": result["n_variants"],
        "rmse": result["rmse"],
        "rmse_all": result["rmse_all"],
        "r2": result["r2"],
        "by_batch": result["by_batch"],
        "by_n_mut": result["by_n_mut"],
        "grid": None,
        "adopted": False,
    }
    write_artifacts(report_dir, report, build_oof_frame(df, result))


def main(argv=None):
    parser = argparse.ArgumentParser(description="UGT91C1 模型统一训练入口")
    parser.add_argument("--dataset", default=str(DS_PATH), help="训练集 CSV 路径")
    parser.add_argument("--features", default=str(X_PATH), help="ESM mean-pool 特征 npy 路径")
    parser.add_argument("--out", default=str(MODEL_OUT), help="模型输出路径")
    parser.add_argument("--report-dir", default=None, help="额外输出 report.json / oof.csv 的目录")
    parser.add_argument("--grid", action="store_true", help="小网格自动调优并按 Macro Spearman 择优")
    parser.add_argument("--eval-only", action="store_true", help="只评估、不训练、不落模型")
    parser.add_argument("--model", default=str(MODEL_OUT), help="--eval-only 时要评估的模型 pkl")
    args = parser.parse_args(argv)

    report_dir = Path(args.report_dir) if args.report_dir else None
    if args.eval_only:
        if report_dir is None:
            raise SystemExit("--eval-only 需要同时指定 --report-dir 以便落盘报告。")
        run_eval_only(args, report_dir)
        return

    df, X_all, feature_names, prior_names = load_inputs(Path(args.dataset), Path(args.features))
    y = df["activity_rel"].to_numpy(dtype=float)
    print(
        f"rows={len(df)}  esm={X_all.shape[1] - len(prior_names)}  "
        f"prior={len(prior_names)}  all={X_all.shape[1]}"
    )

    # ---------- 1) 候选配方评估（--grid 时多组，否则只有线上配方一组） ----------
    candidates = grid_candidates() if args.grid else [_params()]
    results = []
    for i, params in enumerate(candidates, start=1):
        label = "线上配方" if i == 1 else f"变体{i - 1}"
        print(f"[{label}] {params}")
        result = evaluate(df, X_all, y, params)
        print(
            f"  leave-one-site-out: sites={result['n_sites']}  variants={result['n_variants']}  "
            + "  ".join(f"{k}={v}" for k, v in result["metrics"].items()),
            flush=True,
        )
        print(f"  5-fold KFold RMSE={result['rmse']}", flush=True)
        result["label"] = label
        results.append(result)

    best = max(results, key=lambda r: selection_score(r["metrics"]))
    print(f"\n选定配方：{best['label']}  {best['params']}")

    grid_table = [
        {
            "label": r["label"],
            "params": r["params"],
            "spearman_macro": r["metrics"]["spearman_macro"],
            "auc_macro": r["metrics"]["auc_macro"],
            "recall10_macro": r["metrics"]["recall10_macro"],
            "rmse": r["rmse"],
            "n_variants": r["n_variants"],
        }
        for r in results
    ]

    # ---------- 2) 全量重训并保存 ----------
    final = xgb.XGBRegressor(**best["params"])
    final.fit(X_all, y)
    meta = {
        "prior_feature_names": list(prior_names),
        "feature_names": feature_names,
        "esm_dim": int(X_all.shape[1] - len(prior_names)),
        "target": "activity_rel",
        "target_note": "相对同批次 WT 的比酶活倍数（WT=1，批内归一化）",
        "wt_scale": 1.0,
        "n_train": int(len(df)),
        "model_params": best["params"],
        "cv_protocol": "leave-one-site-out",
        "cv_n_sites": int(best["n_sites"]),
        "cv_n_variants": int(best["n_variants"]),
        "cv_spearman_pooled": float(best["metrics_full"]["spearman_pooled"]),
        "cv_spearman_macro": float(best["metrics_full"]["spearman_macro"]),
        "cv_auc_pooled": float(best["metrics_full"]["auc_pooled"]),
        "cv_auc_macro": float(best["metrics_full"]["auc_macro"]),
        "cv_recall5_macro": float(best["metrics_full"]["recall5_macro"]),
        "cv_recall10_macro": float(best["metrics_full"]["recall10_macro"]),
        "cv_rmse": float(best["rmse"]),
        "cv_rmse_all": best["rmse_all"],
        "cv_r2": float(best["r2"]) if best["r2"] is not None else np.nan,
        "cv_rmse_protocol": "5-fold KFold on activity_rel",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "rank_only": True,
        "rank_only_note": "仅按名次取前 N 个候选，勿用预测绝对值做阈值筛选",
    }
    save_model(Path(args.out), final, meta)

    if report_dir is not None:
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "train",
            "dataset": str(args.dataset),
            "dataset_rows": int(len(df)),
            "batches": sorted(df["batch"].astype(str).unique().tolist()),
            "cv_protocol": "leave-one-site-out",
            "selected_label": best["label"],
            "model_params": best["params"],
            "metrics": best["metrics"],
            "metrics_full": best["metrics_full"],
            "n_sites": best["n_sites"],
            "n_variants": best["n_variants"],
            "rmse": best["rmse"],
            "rmse_all": best["rmse_all"],
            "r2": best["r2"],
            "by_batch": best["by_batch"],
            "by_n_mut": best["by_n_mut"],
            "grid": grid_table if args.grid else None,
            "model_out": str(args.out),
            "adopted": False,
        }
        write_artifacts(report_dir, report, build_oof_frame(df, best))

    back = joblib.load(args.out)["model"]
    print(f"reload check: pred mean={back.predict(X_all).mean():.3f}  true mean={y.mean():.3f}")


if __name__ == "__main__":
    main()
