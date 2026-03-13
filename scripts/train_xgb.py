import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

DATA_PATH = r"..\data\mutants.csv"
X_PATH = r"..\data\X_esm.npy"
MODEL_OUT = r"..\models\xgb_esm.pkl"

df = pd.read_csv(DATA_PATH)
X = np.load(X_PATH)
y = df["activity"].values

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

for tr, te in kf.split(X):
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    rmse = mean_squared_error(y[te], pred) ** 0.5
    rmses.append(rmse)

print("5-fold RMSE:", sum(rmses)/len(rmses), "all:", rmses)

# 用全数据训练最终模型
model.fit(X, y)
joblib.dump(model, MODEL_OUT)
print("Saved model:", MODEL_OUT)