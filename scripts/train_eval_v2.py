"""Tier 2: 按位点留出（leave-one-site-out）交叉验证，对比基线与新方案。

评测协议（严禁随机折，随机折会虚高泛化能力）：
  对每个"非背景"突变位点 p（背景位点 F208M 的 208 已在 Tier 0 中剔除）：
      测试集 = 所有在 p 位发生突变的变体；训练集 = 其余变体。
  一个变体若含多个突变位点，会在多个折中被预测，取平均作为其 OOF 预测。
  这评测的是"模型能否外推到从未见过的突变位点"，与"挑候选突变体"目标一致。

注意：数据来自两个批次，批内已按各自 WT 归一化，但分数仍可能存在批次标定差异。
因此指标同时给出：
  - pooled（原始拼接）
  - pooled_ranknorm（先在批内做秩归一化再拼接，消除批次标定影响）
  - macro（各批次分别计算后取平均，贴近"同一候选池内排序"的真实场景）

对比模型（全部冻结特征 + 浅头）：
  ZS       : 参数无关的 masked-marginal 零样本分数（仅非背景位点）
  M0       : mean-pool(1280) + 先验(23)                     -> XGBoost   （旧部署配方）
  M1       : 零样本(7) + 先验(23)                            -> XGBoost
  M2       : 位点窗口(仅非背景,1280) + 零样本 + 先验          -> 标准化+PCA(32)+Ridge
  M3       : 位点窗口 + 零样本 + 先验                         -> XGBoost
  M4       : mean-pool + 位点窗口 + 零样本 + 先验              -> 标准化+PCA(32)+Ridge

输出:
  results/tier_cv_metrics.csv
  results/tier_oof_predictions.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from prior_features import PriorFeatureBuilder  # noqa: E402

DS_PATH = BASE_DIR / "data" / "train_dataset.csv"
OUT_METRICS = BASE_DIR / "results" / "tier_cv_metrics.csv"
OUT_OOF = BASE_DIR / "results" / "tier_oof_predictions.csv"
BACKGROUND = {208}
RANDOM_STATE = 42


def parse_sites(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [int(x) for x in str(s).split(";") if str(x).strip()]


def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def topk_scores(y_true, y_pred, k):
    op = np.argsort(-np.asarray(y_pred))
    ot = np.argsort(-np.asarray(y_true))
    return len(set(op[:k].tolist()) & set(ot[:k].tolist())) / k


def batch_rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def metrics(y_true, y_pred, batch):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    batch = np.asarray(batch)
    out = {"spearman_pooled": spearman(y_true, y_pred)}
    out["spearman_pooled_ranknorm"] = spearman(batch_rank(y_true), batch_rank(y_pred))
    label = (y_true > 1.0).astype(int)
    out["auc_pooled"] = float(roc_auc_score(label, y_pred)) if 0 < label.sum() < len(label) else np.nan
    out["recall10_pooled"] = topk_scores(y_true, y_pred, 10)
    out["cover_true_top5_by_pred_top10"] = len(
        set(np.argsort(-y_pred)[:10].tolist()) & set(np.argsort(-y_true)[:5].tolist())
    )

    macro = {"spearman": [], "auc": [], "recall5": [], "recall10": []}
    for b in sorted(set(batch.tolist())):
        m = batch == b
        yt, yp = y_true[m], y_pred[m]
        if len(yt) < 3 or np.std(yt) == 0:
            continue
        macro["spearman"].append(spearman(yt, yp))
        lab = (yt > 1.0).astype(int)
        if 0 < lab.sum() < len(lab):
            macro["auc"].append(roc_auc_score(lab, yp))
        macro["recall5"].append(topk_scores(yt, yp, min(5, len(yt))))
        macro["recall10"].append(topk_scores(yt, yp, min(10, len(yt))))
    for k, v in macro.items():
        out[f"macro_{k}"] = float(np.mean(v)) if v else np.nan
    return out


def model_specs():
    xgb_fn = lambda: xgb.XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=RANDOM_STATE, verbosity=0,
    )
    ridge_fn = lambda: Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=32, random_state=RANDOM_STATE)),
        ("ridge", Ridge(alpha=10.0)),
    ])
    return [
        ("ZS_zeroshot_only", None, "fixed"),
        ("M0_meanpool_prior_XGB", ["meanpool", "prior"], xgb_fn),
        ("M1_zeroshot_prior_XGB", ["zs", "prior"], xgb_fn),
        ("M2_sitelocal_zs_prior_PCA_Ridge", ["site_local", "zs", "prior"], ridge_fn),
        ("M3_sitelocal_zs_prior_XGB", ["site_local", "zs", "prior"], xgb_fn),
        ("M4_fusion_PCA_Ridge", ["meanpool", "site_local", "zs", "prior"], ridge_fn),
    ]


def main():
    df = pd.read_csv(DS_PATH)
    df["sites_list"] = df["sites_str"].apply(parse_sites)
    n = len(df)

    blocks = {
        "meanpool": np.load(BASE_DIR / "data" / "feat_meanpool.npy"),
        "site": np.load(BASE_DIR / "data" / "feat_site.npy"),
        "site_local": np.load(BASE_DIR / "data" / "feat_site_local.npy"),
        "zs": np.load(BASE_DIR / "data" / "feat_zeroshot.npy"),
    }
    prior, _ = PriorFeatureBuilder(BASE_DIR).build_matrix(df)
    blocks["prior"] = prior

    eval_sites = sorted({p for s in df["sites_list"] for p in s if p not in BACKGROUND})
    has_site = df["sites_list"].apply(lambda s: any(p not in BACKGROUND for p in s)).to_numpy()
    eval_idx = np.where(has_site)[0]
    y_all = df["activity_rel"].to_numpy(dtype=float)
    batch_all = df["batch"].to_numpy()
    zs_fixed = blocks["zs"][:, 5]  # zs_nonbg_sum
    print(f"Rows={n}, non-background sites={len(eval_sites)}, evaluated variants={len(eval_idx)}")

    results = []
    oof_table = df.loc[eval_idx, ["variant", "batch", "mutation_list", "activity_rel", "sites_str"]].copy()

    for name, keys, model_fn in model_specs():
        if model_fn == "fixed":
            oof = np.full(n, np.nan)
            oof[eval_idx] = zs_fixed[eval_idx]
            n_feat = 1
        else:
            X = np.hstack([blocks[k] for k in keys])
            oof = np.full(n, np.nan)
            for p in eval_sites:
                in_site = df["sites_list"].apply(lambda s: p in s).to_numpy()
                tr, te = np.where(~in_site)[0], np.where(in_site)[0]
                model = model_fn()
                model.fit(X[tr], y_all[tr])
                oof[te] = model.predict(X[te])
            n_feat = int(X.shape[1])

        m = metrics(y_all[eval_idx], oof[eval_idx], batch_all[eval_idx])
        m["model"] = name
        m["n_features"] = n_feat
        results.append(m)
        oof_table["pred_" + name] = oof[eval_idx]

        print(f"\n[{name}] n_feat={n_feat}")
        print("  " + "  ".join(f"{k}={m[k]:.3f}" for k in
                               ("spearman_pooled", "spearman_pooled_ranknorm", "macro_spearman",
                                "auc_pooled", "macro_auc", "macro_recall5", "macro_recall10")))

    cols = ["model", "n_features", "spearman_pooled", "spearman_pooled_ranknorm", "macro_spearman",
            "auc_pooled", "macro_auc", "recall10_pooled", "macro_recall5", "macro_recall10",
            "cover_true_top5_by_pred_top10"]
    res_df = pd.DataFrame(results)[cols]
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_METRICS, index=False)
    oof_table.to_csv(OUT_OOF, index=False)
    print("\n=== Summary ===")
    print(res_df.to_string(index=False))
    print("\nSaved:", OUT_METRICS, OUT_OOF)


if __name__ == "__main__":
    main()
