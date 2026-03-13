import pandas as pd
import random
from pathlib import Path

AA = list("ACDEFGHIKLMNPQRSTVWY")
BASE_DIR = Path(__file__).resolve().parents[1]
WT_FASTA = BASE_DIR / "data" / "wt.fasta"
OUT_CSV = BASE_DIR / "data" / "new_candidates.csv"


def read_fasta(path: Path) -> str:
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).strip()


wt_seq = read_fasta(WT_FASTA)
if not wt_seq:
    raise ValueError(f"Empty WT FASTA: {WT_FASTA}")

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
out.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV, "rows=", len(out))