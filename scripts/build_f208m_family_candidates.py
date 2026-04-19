import argparse
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


AA = set("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


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


def parse_mut_token(token: str):
    m = MUT_RE.match(token.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def parse_mutation_list(mut_list: str):
    if not isinstance(mut_list, str) or not mut_list.strip():
        return []
    out = []
    for t in [x.strip() for x in mut_list.split(";") if x.strip()]:
        p = parse_mut_token(t)
        if p is not None:
            out.append((t, p[0], p[1], p[2]))
    return out


def apply_mutations(wt_seq: str, mut_tokens):
    seq = list(wt_seq)
    norm_tokens = []
    for t in mut_tokens:
        parsed = parse_mut_token(t)
        if parsed is None:
            continue
        old, pos, new = parsed
        idx = pos - 1
        if idx < 0 or idx >= len(seq):
            continue
        if seq[idx] != old:
            continue
        seq[idx] = new
        norm_tokens.append(f"{old}{pos}{new}")
    norm_tokens = sorted(set(norm_tokens), key=lambda x: int(MUT_RE.match(x).group(2)))
    return "".join(seq), norm_tokens


def top_family_alternatives(wt_seq: str, homologs, min_support: int, top_k_alt_per_pos: int):
    pos_alt_counter = defaultdict(Counter)
    used = 0
    for seq in homologs:
        mapping = map_homolog_to_wt_positions(wt_seq, seq)
        if not mapping:
            continue
        used += 1
        for pos, wt_aa in enumerate(wt_seq, start=1):
            aa = mapping.get(pos)
            if aa in AA and aa != wt_aa:
                pos_alt_counter[pos][aa] += 1

    pos_top = {}
    for pos, c in pos_alt_counter.items():
        vals = [(aa, s) for aa, s in c.most_common(top_k_alt_per_pos) if s >= min_support]
        if vals:
            pos_top[pos] = vals
    return pos_top, used


def f208m_hotspots_from_mutants(mutants_df: pd.DataFrame, min_rel_activity: float = 1.0):
    score = Counter()
    if "mutation_list" not in mutants_df.columns:
        return score

    for _, row in mutants_df.iterrows():
        ml = row.get("mutation_list", "")
        if not isinstance(ml, str) or "F208M" not in ml:
            continue
        rel = float(row.get("activity_rel", row.get("activity", 0.0)))
        if rel < min_rel_activity:
            continue
        for token, old, pos, new in parse_mutation_list(ml):
            if pos == 208:
                continue
            score[pos] += rel
    return score


def main():
    parser = argparse.ArgumentParser(description="Build family-augmented multi-point candidates around F208M")
    parser.add_argument("--wt", type=str, default="data/wt.fasta")
    parser.add_argument("--family-fasta", type=str, default="data/family/ugt91_family_dedup.fasta")
    parser.add_argument("--mutants", type=str, default="data/mutants.csv")
    parser.add_argument("--base-candidates", type=str, default="data/new_candidates.csv")
    parser.add_argument("--out", type=str, default="data/new_candidates_family.csv")
    parser.add_argument("--merge-out", type=str, default="data/new_candidates_merged.csv")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--top-k-alt-per-pos", type=int, default=3)
    parser.add_argument("--top-hotspots", type=int, default=8)
    parser.add_argument("--max-extra-mutations", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=3000)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    wt_seq = read_wt(base_dir / args.wt)
    if not wt_seq:
        raise ValueError("WT sequence is empty")

    homologs = [s for _, s in read_fasta(base_dir / args.family_fasta)]
    if not homologs:
        raise ValueError("No homolog sequences found")

    mutants_df = pd.read_csv(base_dir / args.mutants)

    # Family-supported alternatives at each WT position.
    pos_top, homolog_used = top_family_alternatives(
        wt_seq,
        homologs,
        min_support=args.min_support,
        top_k_alt_per_pos=args.top_k_alt_per_pos,
    )

    # Prioritize positions observed in high-activity F208M backgrounds.
    hotspot_score = f208m_hotspots_from_mutants(mutants_df)
    prioritized_positions = [p for p, _ in hotspot_score.most_common() if p in pos_top and p != 208]

    if len(prioritized_positions) < args.top_hotspots:
        family_rank = sorted(
            [(p, max(s for _, s in alts)) for p, alts in pos_top.items() if p != 208],
            key=lambda x: x[1],
            reverse=True,
        )
        for p, _ in family_rank:
            if p not in prioritized_positions:
                prioritized_positions.append(p)
            if len(prioritized_positions) >= args.top_hotspots:
                break

    # Ensure seed F208M is always included.
    seed = "F208M"
    candidates = {}

    def add_candidate(tokens, tier, family_support_sum):
        seq, norm_tokens = apply_mutations(wt_seq, tokens)
        if "F208M" not in norm_tokens:
            return
        key = tuple(norm_tokens)
        if key in candidates:
            return
        variant = "-".join(norm_tokens)
        candidates[key] = {
            "variant": variant,
            "mutation_list": ";".join(norm_tokens),
            "sequence": seq,
            "source": "family_homolog_f208m_focus",
            "focus_seed": "F208M",
            "design_tier": tier,
            "family_support_sum": int(family_support_sum),
            "homolog_pool_size": int(homolog_used),
        }

    add_candidate([seed], tier="seed_only", family_support_sum=0)

    hotspot_top_alt = {}
    for p in prioritized_positions:
        hotspot_top_alt[p] = pos_top.get(p, [])

    for extra_mut_n in range(1, args.max_extra_mutations + 1):
        for pos_combo in itertools.combinations(prioritized_positions, extra_mut_n):
            alt_lists = [hotspot_top_alt[p] for p in pos_combo]
            if any(len(v) == 0 for v in alt_lists):
                continue
            for alt_combo in itertools.product(*alt_lists):
                toks = [seed]
                support_sum = 0
                ok = True
                for pos, (aa, support) in zip(pos_combo, alt_combo):
                    wt = wt_seq[pos - 1]
                    if aa == wt:
                        ok = False
                        break
                    toks.append(f"{wt}{pos}{aa}")
                    support_sum += support
                if not ok:
                    continue
                add_candidate(
                    toks,
                    tier=f"seed_plus_{extra_mut_n}",
                    family_support_sum=support_sum,
                )
                if len(candidates) >= args.max_candidates:
                    break
            if len(candidates) >= args.max_candidates:
                break
        if len(candidates) >= args.max_candidates:
            break

    out_df = pd.DataFrame(candidates.values())
    out_df = out_df.sort_values(
        ["design_tier", "family_support_sum", "variant"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    out_path = base_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Merge with base candidates for downstream prediction.
    base_path = base_dir / args.base_candidates
    merged_path = base_dir / args.merge_out
    if base_path.exists():
        base_df = pd.read_csv(base_path)
        if "source" not in base_df.columns:
            base_df["source"] = "random"
        merged = pd.concat([base_df, out_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["variant", "sequence"]).reset_index(drop=True)
        merged.to_csv(merged_path, index=False)
        print("Saved merged candidates:", merged_path, "rows=", len(merged))

    print("Saved F208M-focused family candidates:", out_path, "rows=", len(out_df))
    print("Hotspot positions:", prioritized_positions)
    print("Homologs total:", len(homologs), "alignment-used:", homolog_used)


if __name__ == "__main__":
    main()
