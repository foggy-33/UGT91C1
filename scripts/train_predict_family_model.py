import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

from prior_features import PriorFeatureBuilder


def choose_target(df: pd.DataFrame, target: str):
    if target == "auto":
        if "activity_rel" in df.columns:
            return "activity_rel"
        if "activity" in df.columns:
            return "activity"
        raise ValueError("No target found: need activity_rel or activity")
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description="Train/predict activity with family priors + mutants")
    parser.add_argument("--train-csv", type=str, default="data/mutants.csv")
    parser.add_argument("--cand-csv", type=str, default="data/new_candidates_merged.csv")
    parser.add_argument("--fallback-cand-csv", type=str, default="data/new_candidates.csv")
    parser.add_argument("--target", type=str, default="auto")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ddg-threshold", type=float, default=2.0)
    parser.add_argument("--allow-unknown-ddg", action="store_true", default=True)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    train_path = base_dir / args.train_csv
    cand_path = base_dir / args.cand_csv
    fallback_cand_path = base_dir / args.fallback_cand_csv

    if not cand_path.exists():
        cand_path = fallback_cand_path

    df_train = pd.read_csv(train_path)
    df_cand = pd.read_csv(cand_path)
    target_col = choose_target(df_train, args.target)

    prior_builder = PriorFeatureBuilder(base_dir)
    x_train, feat_names = prior_builder.build_matrix(df_train)
    y_train = df_train[target_col].values.astype(float)

    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
    )

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    rmses = []
    for tr, te in kf.split(x_train):
        model.fit(x_train[tr], y_train[tr])
        pred = model.predict(x_train[te])
        rmses.append(mean_squared_error(y_train[te], pred) ** 0.5)

    model.fit(x_train, y_train)

    x_cand, _ = prior_builder.build_matrix(df_cand)
    pred_cand = model.predict(x_cand)
    out = df_cand.copy()
    out["pred_activity_family"] = pred_cand

    out_all_path = base_dir / "results" / "predictions_family_model.csv"
    out_all_path.parent.mkdir(parents=True, exist_ok=True)
    out = out.sort_values("pred_activity_family", ascending=False)
    out.to_csv(out_all_path, index=False)

    out_pass, out_block = prior_builder.ddg_filter(
        out,
        threshold=args.ddg_threshold,
        allow_unknown=args.allow_unknown_ddg,
    )
    out_pass = out_pass.sort_values("pred_activity_family", ascending=False)
    out_pass_path = base_dir / "results" / "predictions_family_model_filtered_by_ddg.csv"
    out_pass.to_csv(out_pass_path, index=False)

    model_path = base_dir / "models" / "xgb_family_priors.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feat_names,
            "target": target_col,
            "cv_rmse": float(np.mean(rmses)),
            "cv_rmse_all": [float(x) for x in rmses],
        },
        model_path,
    )

    metrics_path = base_dir / "results" / "family_model_cv_metrics.csv"
    pd.DataFrame(
        {
            "target": [target_col],
            "cv_rmse_mean": [float(np.mean(rmses))],
            "cv_rmse_std": [float(np.std(rmses))],
            "n_train": [len(df_train)],
            "n_cand": [len(df_cand)],
            "n_cand_pass_ddg": [len(out_pass)],
            "n_cand_block_ddg": [len(out_block)],
        }
    ).to_csv(metrics_path, index=False)

    print("Train target:", target_col)
    print("CV RMSE mean:", float(np.mean(rmses)), "all:", [round(x, 4) for x in rmses])
    print("Saved model:", model_path)
    print("Saved predictions:", out_all_path)
    print("Saved ddG-filtered predictions:", out_pass_path)
    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
