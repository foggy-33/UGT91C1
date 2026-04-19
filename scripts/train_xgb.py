import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from pathlib import Path

from prior_features import PriorFeatureBuilder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "mutants.csv"
X_PATH = BASE_DIR / "data" / "X_esm.npy"
MODEL_OUT = BASE_DIR / "models" / "xgb_esm.pkl"

df = pd.read_csv(DATA_PATH)
X = np.load(X_PATH)
y = df["activity"].values

# Concatenate domain priors (evolution/structure/stability) with ESM embeddings.
prior_builder = PriorFeatureBuilder(BASE_DIR)
X_prior, prior_names = prior_builder.build_matrix(df)
X_all = np.hstack([X, X_prior])
print("ESM shape:", X.shape, "Prior shape:", X_prior.shape, "All shape:", X_all.shape)

model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for tr, te in kf.split(X_all):
    model.fit(X_all[tr], y[tr])
    pred = model.predict(X_all[te])
    rmse = mean_squared_error(y[te], pred) ** 0.5
    rmses.append(rmse)

print("5-fold RMSE:", sum(rmses)/len(rmses), "all:", rmses)

# 用全数据训练最终模型
model.fit(X_all, y)
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "model": model,
        "prior_feature_names": prior_names,
        "esm_dim": int(X.shape[1]),
    },
    MODEL_OUT,
)
print("Saved model:", MODEL_OUT)