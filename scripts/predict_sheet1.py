"""用已部署配方预测 Sheet1（batch A）的 18 个真实实验突变体。

背景：models/xgb_esm.pkl 是用旧版 xgboost 序列化的，在 xgboost 3.0.5 下加载会丢失
base_score 截距，预测值整体偏移约 1.4（真值均值 1.05，预测均值 -0.34），无法直接使用。
因此这里用与之完全相同的配方在本环境重新训练：
    特征 = ESM2-t33 mean-pool(1280) + 23 维先验
    模型 = XGBRegressor(n_estimators=600, max_depth=4, lr=0.05, ...)
    目标 = activity（批内 WT=3.7，预测后除以 3.7 得到 activity_rel）
训练集只用 data/mutants.csv（batch B，48 条），**不含 Sheet1**，因此 Sheet1 的预测是
真正的样本外预测，可直接与实验真值对比作图。

同时给出参数无关的 masked-marginal 零样本分数作为对照。

输出: results/sheet1_predictions.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from prior_features import PriorFeatureBuilder  # noqa: E402

OUT_CSV = BASE_DIR / "results" / "sheet1_predictions.csv"
WT_SCALE = 3.7  # batch B 批内 WT（预测目标所处尺度）


def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ds = pd.read_csv(BASE_DIR / "data" / "train_dataset.csv")
    meanpool = np.load(BASE_DIR / "data" / "feat_meanpool.npy")
    zs = np.load(BASE_DIR / "data" / "feat_zeroshot.npy")
    prior, _ = PriorFeatureBuilder(BASE_DIR).build_matrix(ds)
    X = np.hstack([meanpool, prior])

    is_b = (ds["batch"] == "B").to_numpy()
    is_a = (ds["batch"] == "A").to_numpy()

    # 与线上 train_xgb.py 完全一致的配方；训练集只用 batch B（不含 Sheet1）
    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )
    model.fit(X[is_b], ds.loc[is_b, "activity"].to_numpy(dtype=float))

    out = ds.loc[is_a, ["variant", "mutation_list", "activity"]].copy()
    out["activity_rel_true"] = ds.loc[is_a, "activity_rel"].to_numpy()
    out["pred_activity"] = model.predict(X[is_a])
    out["pred_activity_rel"] = out["pred_activity"] / WT_SCALE
    out["zeroshot_score"] = zs[is_a, 5]  # 仅非背景位点的 log-odds 之和

    out["rank_true"] = out["activity_rel_true"].rank(ascending=False, method="min").astype(int)
    out["rank_pred"] = out["pred_activity_rel"].rank(ascending=False, method="min").astype(int)
    out["rank_zeroshot"] = out["zeroshot_score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values("rank_true").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("Sheet1 (batch A) 预测结果，n =", len(out))
    print(out[["variant", "activity_rel_true", "pred_activity_rel", "zeroshot_score",
               "rank_true", "rank_pred", "rank_zeroshot"]].to_string(index=False))
    y = out["activity_rel_true"].to_numpy()
    print("\n排序一致性（Spearman，越高越好）：")
    print("  ESM+XGB 配方 : %.3f" % spearman(y, out["pred_activity_rel"]))
    print("  零样本分数   : %.3f" % spearman(y, out["zeroshot_score"]))
    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
