import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
XLSX_PATH = BASE_DIR / "data" / "raw.xlsx"
WT_FASTA = BASE_DIR / "data" / "wt.fasta"
OUT_CSV = BASE_DIR / "data" / "mutants.csv"

def read_fasta(path: Path) -> str:
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).strip()

mut_pat = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")

def apply_mutations(wt_seq: str, mutation_list: str) -> str:
    """mutation_list: 'F208M;W22A'  (1-based positions)"""
    seq = list(wt_seq)
    if not mutation_list:
        return wt_seq

    muts = [m.strip() for m in mutation_list.split(";") if m.strip()]
    for m in muts:
        m = m.replace(" ", "")
        mm = mut_pat.match(m)
        if not mm:
            raise ValueError(f"Bad mutation format: {m} (expect like F208M)")
        old, pos, new = mm.group(1), int(mm.group(2)), mm.group(3)
        i = pos - 1
        if i < 0 or i >= len(seq):
            raise ValueError(f"Position out of range: {m} for length {len(seq)}")
        if seq[i] != old:
            raise ValueError(f"WT mismatch at {pos}: WT has {seq[i]}, but mutation says {old} in {m}")
        seq[i] = new
    return "".join(seq)

# 1) 读 WT 序列
wt_seq = read_fasta(WT_FASTA)
print("WT length:", len(wt_seq))

# 2) 读 Excel（自动找列）
df = pd.read_excel(XLSX_PATH)

# 兼容你的中文列名
col_variant = None
col_activity = None
for c in df.columns:
    c_str = str(c).strip()
    c_low = c_str.lower()
    if "突变" in c_str or c_low in {"variant", "mutation", "mutant"}:
        col_variant = c
    if (
        "比酶活" in c_str
        or "mU" in c_str
        or "酶活" in c_str
        or c_low in {"activity", "activity_rel", "relative_activity"}
    ):
        col_activity = c

if col_variant is None or col_activity is None:
    raise ValueError(f"Cannot find needed columns. Got columns: {list(df.columns)}")

df = df[[col_variant, col_activity]].rename(columns={col_variant: "variant", col_activity: "activity"})
df["variant"] = df["variant"].astype(str).str.strip()
df["activity"] = pd.to_numeric(df["activity"], errors="coerce")

df = df.dropna(subset=["variant", "activity"]).copy()

# 3) 生成 mutation_list（把 F208M-W22A 变成 F208M;W22A）
def to_mutation_list(v: str) -> str:
    v = v.strip()
    if v.upper() == "WT":
        return ""
    # 你表里常见：F208M-H93A（用 - 连接）
    parts = [p.strip() for p in v.split("-") if p.strip()]
    # 如果第一段不是标准突变（比如名字里带前缀），你也可以在这里扩展规则
    return ";".join(parts)

df["mutation_list"] = df["variant"].apply(to_mutation_list)

# 4) 生成 sequence，并做 WT 位点一致性检查
seqs = []
for ml in df["mutation_list"].tolist():
    seqs.append(apply_mutations(wt_seq, ml))
df["sequence"] = seqs

# 5) 建议加一个相对活性（WT=1），更抗批次漂移（如果同表内有WT）
wt_rows = df[df["variant"].str.upper() == "WT"]
if len(wt_rows) == 1 and wt_rows["activity"].iloc[0] > 0:
    wt_act = wt_rows["activity"].iloc[0]
    df["activity_rel"] = df["activity"] / wt_act
else:
    df["activity_rel"] = df["activity"]

df.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
print(df.head())