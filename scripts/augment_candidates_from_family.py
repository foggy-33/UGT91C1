import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


AA = set("ACDEFGHIKLMNPQRSTVWY")


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
    """Needleman-Wunsch global alignment for small/medium protein sequences."""
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


def main():
    parser = argparse.ArgumentParser(description="Generate family-supported mutation candidates")
    parser.add_argument("--family-fasta", type=str, default="data/family/ugt91_family_dedup.fasta")
    parser.add_argument("--wt", type=str, default="data/wt.fasta")
    parser.add_argument("--out", type=str, default="data/new_candidates_family.csv")
    parser.add_argument("--merge-out", type=str, default="data/new_candidates_merged.csv")
    parser.add_argument("--top-k-alt-per-pos", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=2)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    family_fasta = base_dir / args.family_fasta
    wt_path = base_dir / args.wt
    out_path = base_dir / args.out
    merge_out_path = base_dir / args.merge_out

    wt_seq = read_wt(wt_path)
    if not wt_seq:
        raise ValueError(f"Empty WT sequence: {wt_path}")

    homologs = [s for _, s in read_fasta(family_fasta)]
    pos_alt_counter = defaultdict(Counter)
    used_homologs = 0
    for seq in homologs:
        mapping = map_homolog_to_wt_positions(wt_seq, seq)
        if not mapping:
            continue
        used_homologs += 1
        for i, w in enumerate(wt_seq, start=1):
            a = mapping.get(i)
            if not a:
                continue
            if a != w and a in AA and w in AA:
                pos_alt_counter[i][a] += 1

    rows = []
    for pos, cnt in pos_alt_counter.items():
        wt_aa = wt_seq[pos - 1]
        for mut_aa, support in cnt.most_common(args.top_k_alt_per_pos):
            if support < args.min_support:
                continue
            seq = list(wt_seq)
            seq[pos - 1] = mut_aa
            rows.append(
                {
                    "variant": f"{wt_aa}{pos}{mut_aa}",
                    "sequence": "".join(seq),
                    "source": "family_homolog",
                    "family_support": int(support),
                    "homolog_pool_size": int(used_homologs),
                }
            )

    cand_df = pd.DataFrame(rows).drop_duplicates(subset=["variant", "sequence"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cand_df.to_csv(out_path, index=False)

    # Merge with existing randomly generated candidates if available.
    existing_path = base_dir / "data" / "new_candidates.csv"
    if existing_path.exists():
        old = pd.read_csv(existing_path)
        old["source"] = old.get("source", "random")
        merged = pd.concat([old, cand_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["variant", "sequence"]).reset_index(drop=True)
        merged.to_csv(merge_out_path, index=False)
        print("Saved merged candidates:", merge_out_path, "rows=", len(merged))

    print("Saved family candidates:", out_path, "rows=", len(cand_df))
    print("Homologs total:", len(homologs), "alignment-used:", used_homologs)


if __name__ == "__main__":
    main()
