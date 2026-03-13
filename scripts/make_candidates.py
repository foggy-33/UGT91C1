import pandas as pd
import random

AA = list("ACDEFGHIKLMNPQRSTVWY")

# 从 mutants.csv 里取第0条当作 WT（demo 数据下都很像）
df = pd.read_csv(r"..\data\mutants.csv")
wt_seq = df.loc[0, "sequence"]  # demo: 每条只差一个位点，随便取一条当近似WT也能跑通演示

cands = []
random.seed(42)

for i in range(200):
    seq = list(wt_seq)
    pos = random.randint(0, len(seq)-1)
    old = seq[pos]
    new = random.choice([a for a in AA if a != old])
    seq[pos] = new
    seq2 = "".join(seq)
    cands.append({"variant": f"{old}{pos+1}{new}", "sequence": seq2})

out = pd.DataFrame(cands)
out.to_csv(r"..\data\new_candidates.csv", index=False)
print("Saved: ..\\data\\new_candidates.csv  rows=", len(out))