"""Tier 3(实用): 用 masked-marginal 零样本分数对候选突变体库排序。

依据 Tier 2 的按位点留出评测结论：在"外推到新突变位点"这一真实挑候选场景下，
参数无关的零样本分数（log p(mut) - log p(wt)）比任何在 1280 维 ESM 表征上训练的
浅头都更稳（macro Spearman 0.48 vs 0.29）。因此这里直接以该分数排序候选库。

做法：
  - 以 WT 序列为模板，逐位点 mask，取该位点氨基酸分布；
  - 候选突变位点通过"候选序列 vs WT 序列"的差异自动识别（不依赖变体名字符串）；
  - 分数 = 各突变位点 log-odds 之和；同时给出剔除背景位点(208)后的版本。

输出: results/zeroshot_candidate_ranking.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm

BASE_DIR = Path(__file__).resolve().parents[1]
WT_FASTA = BASE_DIR / "data" / "wt.fasta"
CAND_CSV = BASE_DIR / "data" / "new_candidates_merged.csv"
OUT_CSV = BASE_DIR / "results" / "zeroshot_candidate_ranking.csv"
BACKGROUND = {208}


def read_fasta(path: Path) -> str:
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq)


def diff_mutations(wt_seq, seq):
    muts = []
    for i, (a, b) in enumerate(zip(wt_seq, seq)):
        if a != b:
            muts.append((i + 1, a, b))
    return muts


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    wt_seq = read_fasta(WT_FASTA)
    df = pd.read_csv(CAND_CSV)
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["mutations"] = df["sequence"].apply(lambda s: diff_mutations(wt_seq, s))
    df["n_mut"] = df["mutations"].apply(len)
    df["positions"] = df["mutations"].apply(lambda ms: ";".join(str(p) for p, _, _ in ms))

    all_sites = sorted({p for ms in df["mutations"] for p, _, _ in ms})
    print(f"Candidates={len(df)}, unique mutated sites={len(all_sites)}")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    _, _, wt_tokens = batch_converter([("wt", wt_seq)])

    aa_idx = {aa: alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    site_logp = {}
    chunk = 16
    with torch.no_grad():
        for i in range(0, len(all_sites), chunk):
            batch_sites = all_sites[i : i + chunk]
            tokens = wt_tokens.repeat(len(batch_sites), 1).to(device)
            for j, p in enumerate(batch_sites):
                tokens[j, p] = mask_idx
            logits = model(tokens)["logits"]
            for j, p in enumerate(batch_sites):
                site_logp[p] = torch.log_softmax(logits[j, p].float(), dim=-1).cpu().numpy()

    zs_sum, zs_nonbg, zs_min, zs_mean, zs_nonbg_mean = [], [], [], [], []
    for ms in df["mutations"]:
        vals, nb = [], []
        for pos, wt_aa, mut_aa in ms:
            logp = site_logp.get(pos)
            if logp is None:
                continue
            d = float(logp[aa_idx[mut_aa]] - logp[aa_idx[wt_aa]])
            vals.append(d)
            if pos not in BACKGROUND:
                nb.append(d)
        zs_sum.append(sum(vals) if vals else 0.0)
        zs_nonbg.append(sum(nb) if nb else 0.0)
        zs_min.append(min(vals) if vals else 0.0)
        zs_mean.append(float(np.mean(vals)) if vals else 0.0)
        zs_nonbg_mean.append(float(np.mean(nb)) if nb else 0.0)

    df["zs_sum"] = zs_sum
    df["zs_nonbg_sum"] = zs_nonbg
    df["zs_min"] = zs_min
    df["zs_mean"] = zs_mean
    df["zs_nonbg_mean"] = zs_nonbg_mean

    out = df.sort_values("zs_nonbg_sum", ascending=False).reset_index(drop=True)
    out.insert(0, "rank_zeroshot", np.arange(1, len(out) + 1))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV, f"({len(out)} rows)")
    print("\nTop 15 by zero-shot score (non-background sites):")
    print(out.head(15)[["rank_zeroshot", "variant", "n_mut", "positions", "zs_nonbg_sum", "zs_sum"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
