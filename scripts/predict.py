import numpy as np
import pandas as pd
import torch
import esm
import joblib
from tqdm import tqdm
from pathlib import Path

from prior_features import PriorFeatureBuilder

BASE_DIR = Path(__file__).resolve().parents[1]
CAND_PATH = BASE_DIR / "data" / "new_candidates.csv"
MERGED_CAND_PATH = BASE_DIR / "data" / "new_candidates_merged.csv"
MUTANTS_PATH = BASE_DIR / "data" / "mutants.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_esm.pkl"
OUT_PATH = BASE_DIR / "results" / "predictions.csv"
FILTERED_OUT_PATH = BASE_DIR / "results" / "predictions_filtered_by_ddg.csv"

DDG_THRESHOLD = 2.0
ALLOW_UNKNOWN_DDG = True

device = "cuda" if torch.cuda.is_available() else "cpu"

model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_esm = model_esm.to(device)
model_esm.eval()
batch_converter = alphabet.get_batch_converter()

model_pack = joblib.load(MODEL_PATH)
if isinstance(model_pack, dict) and "model" in model_pack:
    xgb_model = model_pack["model"]
else:
    # Backward compatibility: old checkpoints may only store estimator.
    xgb_model = model_pack

input_cand = MERGED_CAND_PATH if MERGED_CAND_PATH.exists() else CAND_PATH
print("Using candidate file:", input_cand)
df = pd.read_csv(input_cand)

# Keep prediction output on the same scale as mutants.activity_rel.
wt_activity = 1.0
if MUTANTS_PATH.exists():
    df_mut = pd.read_csv(MUTANTS_PATH)
    wt_mask = df_mut["variant"].astype(str).str.upper() == "WT"
    if wt_mask.any() and "activity" in df_mut.columns:
        wt_val = float(df_mut.loc[wt_mask, "activity"].iloc[0])
        if wt_val > 0:
            wt_activity = wt_val
print("WT activity used for rel scale:", wt_activity)

embeddings = []

with torch.no_grad():
    for seq in tqdm(df["sequence"].tolist(), desc="Embedding new"):
        data = [("protein", seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model_esm(tokens, repr_layers=[33])
        reps = out["representations"][33]
        emb = reps[0, 1:-1].mean(0).float().cpu().numpy()
        embeddings.append(emb)

X_new = np.vstack(embeddings)
prior_builder = PriorFeatureBuilder(BASE_DIR)
X_prior, prior_names = prior_builder.build_matrix(df)
X_all = np.hstack([X_new, X_prior])
pred = xgb_model.predict(X_all)

df["pred_activity"] = pred
df["pred_activity_rel"] = df["pred_activity"] / wt_activity
df = df.sort_values("pred_activity_rel", ascending=False)

# Hard filter by predicted/known stability prior (ddG).
df_pass, df_block = prior_builder.ddg_filter(
    df,
    threshold=DDG_THRESHOLD,
    allow_unknown=ALLOW_UNKNOWN_DDG,
)
df_pass = df_pass.sort_values("pred_activity_rel", ascending=False)
df_block = df_block.sort_values("pred_activity_rel", ascending=False)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
df_pass.to_csv(FILTERED_OUT_PATH, index=False)
print("Saved:", OUT_PATH)
print("Saved:", FILTERED_OUT_PATH)
print("Prior features:", prior_names)
print("Candidates kept after ddG filter:", len(df_pass), "/", len(df))
if len(df_block) > 0:
    print("Top blocked by ddG:")
    print(df_block[["variant", "pred_activity_rel", "pred_activity", "ddg_total", "ddg_max"]].head(5))
print(df_pass.head(10))