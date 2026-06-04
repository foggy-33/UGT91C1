import numpy as np
import pandas as pd
import torch
import esm
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "mutants.csv"
OUT_X = BASE_DIR / "data" / "X_esm.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 650M模型：效果好但更吃显存；
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()

batch_converter = alphabet.get_batch_converter()

df = pd.read_csv(DATA_PATH)
seqs = df["sequence"].tolist()

embeddings = []
with torch.no_grad():
    for seq in tqdm(seqs, desc="ESM embedding"):
        data = [("protein", seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        out = model(tokens, repr_layers=[33])
        reps = out["representations"][33]  # (1, L, dim)

        # 去掉特殊符号 <cls> <eos>，对残基向量做 mean pooling
        emb = reps[0, 1:-1].mean(0).float().cpu().numpy()
        embeddings.append(emb)

X = np.vstack(embeddings)
np.save(OUT_X, X)
print("Saved:", OUT_X, "shape=", X.shape)