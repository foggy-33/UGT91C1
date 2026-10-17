"""Tier 0: 构建统一的 UGT91C1 专用训练集。

合并两个实验批次：
  - batch B: data/raw.xlsx（权威标签，48 条，F208M 背景，批内 WT=3.7）
  - batch A: data/Sheet1.csv（19 条，单点突变，批内 WT=0.021）

两个批次的原始比酶活存在批间漂移，因此各自按批内 WT 归一化为 activity_rel（WT=1）。
同时标注每条记录的突变位点，并自动识别在所有记录中普遍出现的"背景位点"
（如 F208M 的 208 位），供按位点留出交叉验证使用。

输出: data/train_dataset.csv
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
WT_FASTA = BASE_DIR / "data" / "wt.fasta"
RAW_XLSX = BASE_DIR / "data" / "raw.xlsx"
SHEET1 = BASE_DIR / "data" / "Sheet1.csv"
OUT_CSV = BASE_DIR / "data" / "train_dataset.csv"

MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
BACKGROUND_FRACTION = 0.5


def read_fasta(path: Path) -> str:
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq)


def to_mutation_list(variant: str) -> str:
    v = str(variant).strip()
    if v.upper() == "WT":
        return ""
    return ";".join(p.strip() for p in v.split("-") if p.strip())


def parse_mutations(mutation_list: str):
    muts = []
    for token in [x.strip() for x in str(mutation_list).split(";") if x.strip()]:
        m = MUT_RE.match(token)
        if not m:
            raise ValueError(f"Bad mutation token: {token}")
        muts.append({"wt": m.group(1), "pos": int(m.group(2)), "mut": m.group(3)})
    return muts


def apply_mutations(wt_seq: str, mutation_list: str) -> str:
    seq = list(wt_seq)
    for m in parse_mutations(mutation_list):
        i = m["pos"] - 1
        if i < 0 or i >= len(seq):
            raise ValueError(f"Position out of range: {m}")
        if seq[i] != m["wt"]:
            raise ValueError(
                f"WT mismatch at {m['pos']}: WT has {seq[i]}, mutation says {m['wt']}"
            )
        seq[i] = m["mut"]
    return "".join(seq)


def main():
    wt_seq = read_fasta(WT_FASTA)
    print(f"WT length: {len(wt_seq)}")

    batch_b = pd.read_excel(RAW_XLSX)[["variant", "activity"]].copy()
    batch_b["batch"] = "B"

    batch_a = pd.read_csv(SHEET1, header=None, names=["variant", "activity"])
    batch_a["batch"] = "A"

    df = pd.concat([batch_a, batch_b], ignore_index=True)
    df["variant"] = df["variant"].astype(str).str.strip()
    df["activity"] = pd.to_numeric(df["activity"], errors="coerce")
    df = df.dropna(subset=["variant", "activity"]).reset_index(drop=True)
    df["mutation_list"] = df["variant"].apply(to_mutation_list)
    df["sequence"] = [apply_mutations(wt_seq, ml) for ml in df["mutation_list"]]

    # 批内 WT 归一化
    wt_map = {}
    for batch, g in df.groupby("batch"):
        wt_rows = g[g["variant"].str.upper() == "WT"]
        if len(wt_rows) == 0:
            raise ValueError(f"Batch {batch} has no WT reference row")
        wt_map[batch] = float(wt_rows["activity"].iloc[0])
    df["wt_batch"] = df["batch"].map(wt_map)
    df["activity_rel"] = df["activity"] / df["wt_batch"]

    # 位点标注
    df["sites"] = df["mutation_list"].apply(lambda ml: [m["pos"] for m in parse_mutations(ml)])
    df["sites_str"] = df["sites"].apply(lambda xs: ";".join(str(x) for x in xs))
    df["n_mut"] = df["sites"].apply(len)

    # 背景位点：出现在 >= BACKGROUND_FRACTION 记录中的位点
    counter = Counter()
    for sites in df["sites"]:
        for pos in set(sites):
            counter[pos] += 1
    n_rows = len(df)
    background = sorted(
        pos for pos, k in counter.items() if k >= BACKGROUND_FRACTION * n_rows
    )
    df["is_background_only"] = df["sites"].apply(
        lambda xs: len(xs) > 0 and all(p in background for p in xs)
    )

    df.to_csv(OUT_CSV, index=False)

    print(f"Rows: {n_rows}  (batch A={len(batch_a)}, batch B={len(batch_b)})")
    print(f"Batch WT: {wt_map}")
    print(f"Background sites (>= {BACKGROUND_FRACTION:.0%} of rows): {background}")
    print("activity_rel range: %.3f - %.3f" % (df["activity_rel"].min(), df["activity_rel"].max()))
    print(df[["variant", "batch", "mutation_list", "activity", "activity_rel", "sites_str"]].head(12).to_string(index=False))
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
