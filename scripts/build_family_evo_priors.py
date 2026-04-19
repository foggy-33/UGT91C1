import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)


def read_fasta(path: Path):
    header = None
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)


def read_wt(path: Path) -> str:
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq)


def global_align(a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -2):
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 1=diag,2=up,3=left

    for i in range(1, n + 1):
        score[i][0] = i * gap
        trace[i][0] = 2
    for j in range(1, m + 1):
        score[0][j] = j * gap
        trace[0][j] = 3

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            diag = score[i - 1][j - 1] + (match if ai == bj else mismatch)
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            best = max(diag, up, left)
            score[i][j] = best
            trace[i][j] = 1 if best == diag else (2 if best == up else 3)

    i, j = n, m
    al_a = []
    al_b = []
    while i > 0 or j > 0:
        t = trace[i][j] if i >= 0 and j >= 0 else 0
        if i > 0 and j > 0 and t == 1:
            al_a.append(a[i - 1])
            al_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or t == 2):
            al_a.append(a[i - 1])
            al_b.append("-")
            i -= 1
        else:
            al_a.append("-")
            al_b.append(b[j - 1])
            j -= 1
    return "".join(reversed(al_a)), "".join(reversed(al_b))


def map_homolog_to_wt_positions(wt_seq: str, homolog_seq: str):
    aw, ah = global_align(wt_seq, homolog_seq)
    wt_pos = 0
    mapping = {}
    for x, y in zip(aw, ah):
        if x != "-":
            wt_pos += 1
            if y != "-":
                mapping[wt_pos] = y
    return mapping


def build_pssm(wt_seq: str, mapped_sequences, pseudocount: float = 1.0):
    bg = 1.0 / len(AA_ORDER)
    rows = []
    for i in range(1, len(wt_seq) + 1):
        cnt = Counter()
        total = 0
        for m in mapped_sequences:
            aa = m.get(i)
            if aa in AA_SET:
                cnt[aa] += 1
                total += 1

        row = {"pos": i}
        for aa in AA_ORDER:
            p = (cnt.get(aa, 0) + pseudocount) / (total + pseudocount * len(AA_ORDER))
            row[aa] = math.log2(p / bg)
        rows.append(row)
    return pd.DataFrame(rows)


def mutual_information(col_i, col_j):
    pairs = [(a, b) for a, b in zip(col_i, col_j) if (a in AA_SET and b in AA_SET)]
    n = len(pairs)
    if n < 2:
        return 0.0

    cnt_xy = Counter(pairs)
    cnt_x = Counter([x for x, _ in pairs])
    cnt_y = Counter([y for _, y in pairs])

    mi = 0.0
    for (x, y), cxy in cnt_xy.items():
        pxy = cxy / n
        px = cnt_x[x] / n
        py = cnt_y[y] / n
        mi += pxy * math.log2(pxy / (px * py))
    return float(mi)


def build_coevolution(wt_seq: str, mapped_sequences, min_mi: float = 0.01, top_k: int = 30000):
    l = len(wt_seq)
    cols = []
    for i in range(1, l + 1):
        cols.append([m.get(i, "-") for m in mapped_sequences])

    rows = []
    for i in range(l):
        for j in range(i + 1, l):
            mi = mutual_information(cols[i], cols[j])
            if mi >= min_mi:
                rows.append((i + 1, j + 1, mi))

    rows.sort(key=lambda x: x[2], reverse=True)
    if top_k > 0:
        rows = rows[:top_k]
    return pd.DataFrame(rows, columns=["pos_i", "pos_j", "score"])


def main():
    parser = argparse.ArgumentParser(description="Build family-derived PSSM/coevolution priors")
    parser.add_argument("--wt", type=str, default="data/wt.fasta")
    parser.add_argument("--family-fasta", type=str, default="data/family/ugt91_family_dedup.fasta")
    parser.add_argument("--min-mi", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=30000)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    wt_path = base_dir / args.wt
    family_fasta = base_dir / args.family_fasta
    out_dir = base_dir / "data" / "priors"
    out_dir.mkdir(parents=True, exist_ok=True)

    wt_seq = read_wt(wt_path)
    if not wt_seq:
        raise ValueError(f"Empty WT sequence: {wt_path}")

    homologs = [s for _, s in read_fasta(family_fasta)]
    if not homologs:
        raise ValueError(f"No homologs found in: {family_fasta}")

    mapped = [map_homolog_to_wt_positions(wt_seq, s) for s in homologs]
    mapped = [m for m in mapped if m]
    if not mapped:
        raise ValueError("Failed to map homolog sequences to WT positions.")

    pssm = build_pssm(wt_seq, mapped)
    coev = build_coevolution(wt_seq, mapped, min_mi=args.min_mi, top_k=args.top_k)

    pssm_path = out_dir / "pssm.csv"
    coev_path = out_dir / "coevolution.csv"
    pssm.to_csv(pssm_path, index=False)
    coev.to_csv(coev_path, index=False)

    print("Saved:", pssm_path, "rows=", len(pssm))
    print("Saved:", coev_path, "rows=", len(coev))
    print("Homologs total:", len(homologs), "mapped:", len(mapped))


if __name__ == "__main__":
    main()
