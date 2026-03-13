import numpy as np
import pandas as pd
import torch
import esm
import joblib
from tqdm import tqdm

CAND_PATH = r"..\data\new_candidates.csv"
MODEL_PATH = r"..\models\xgb_esm.pkl"
OUT_PATH = r"..\results\predictions.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_esm = model_esm.to(device)
model_esm.eval()
batch_converter = alphabet.get_batch_converter()

xgb_model = joblib.load(MODEL_PATH)

df = pd.read_csv(CAND_PATH)
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
pred = xgb_model.predict(X_new)

df["pred_activity"] = pred
df = df.sort_values("pred_activity", ascending=False)
df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
print(df.head(10))