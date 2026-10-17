"""生成「下一批推荐候选」结果表 results/candidate_recommendations.csv。

目的
----
平台要根据**现有实验数据**，从候选库里直接挑出下一批最可能活性更高的序列，
而不需要用户先上传文件。本脚本把两个候选池合并、打分、融合，产出一张排名表，
供 Streamlit「候选推荐」页直接读取。

候选池
------
  A: results/zeroshot_candidate_ranking.csv —— 1371 条，全为单点突变，含现成的零样本分数。
  B: results/predictions.csv              —— 1358 条，F208M 打底的多突变组合（n_mut 1~4）为主。

打分与融合
----------
  * 已见位点（训练集出现过的非背景位点）→ 用线上 XGBoost 配方（ESM mean-pool + 先验特征）预测。
  * 未见位点 → 用零样本 masked-marginal 分数（参数无关，外推到新位点最稳）。
  * 两个分数尺度不可比，按**名次百分位**加权融合：
        w_model = 候选内已见非背景位数 / 候选内非背景位总数
        fused   = w_model * pct_rank(model) + (1 - w_model) * pct_rank(zero)

已知短板
--------
组合突变叠加（多突变）的排序能力弱于单点，注释必须如实标注。

输出：results/candidate_recommendations.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from prior_features import PriorFeatureBuilder  # noqa: E402

POOL_A = BASE_DIR / "results" / "zeroshot_candidate_ranking.csv"
POOL_B = BASE_DIR / "results" / "predictions.csv"
TRAIN_DS = BASE_DIR / "data" / "train_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_esm.pkl"
WT_FASTA = BASE_DIR / "data" / "wt.fasta"
OUT_PATH = BASE_DIR / "results" / "candidate_recommendations.csv"

BACKGROUND_SITES = {208}
MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")
ESM_BATCH_SIZE = 8
REPR_LAYER = 33


def load_wt() -> str:
    seq = []
    with open(WT_FASTA, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith(">"):
                seq.append(line)
    return "".join(seq)


def parse_sites(variant: str) -> list[int]:
    return [int(m.group(2)) for m in MUT_RE.finditer(str(variant))]


def canonical_mutation_list(variant: str) -> str:
    return ";".join(f"{m.group(1)}{m.group(2)}{m.group(3)}" for m in MUT_RE.finditer(str(variant)))


def seen_nonbg_sites() -> set[int]:
    df = pd.read_csv(TRAIN_DS)
    sites = set()
    for raw in df["sites_str"].dropna():
        for token in str(raw).split(";"):
            token = token.strip()
            if token.isdigit():
                sites.add(int(token))
    return sites - BACKGROUND_SITES


# --------------------------------------------------------------------------- #
# 合并候选池
# --------------------------------------------------------------------------- #
def merge_pools() -> pd.DataFrame:
    a = pd.read_csv(POOL_A)
    b = pd.read_csv(POOL_B)

    a = a.rename(columns={"family_support": "family_support_a"})
    a_small = a[
        ["variant", "sequence", "source", "family_support_a", "homolog_pool_size", "zs_nonbg_sum", "zs_sum"]
    ].rename(columns={"source": "source_a"})

    b_small = b[
        [
            "variant",
            "sequence",
            "source",
            "focus_seed",
            "design_tier",
            "family_support_sum",
            "pred_activity_rel",
        ]
    ].rename(columns={"source": "source_b", "pred_activity_rel": "pool_b_pred"})

    merged = pd.merge(a_small, b_small, on="variant", how="outer", suffixes=("", "_b"))

    if "sequence_b" in merged.columns:
        merged["sequence"] = merged["sequence"].where(
            merged["sequence"].notna() & merged["sequence"].astype(str).ne(""),
            merged["sequence_b"],
        )
        merged = merged.drop(columns=["sequence_b"])

    merged["in_pool_a"] = merged["source_a"].notna()
    merged["in_pool_b"] = merged["source_b"].notna()
    merged["pool_source"] = merged["source_b"].fillna(merged["source_a"])
    merged["family_support"] = merged["family_support_a"].fillna(merged["family_support_sum"])
    merged = merged.drop(columns=["family_support_a", "family_support_sum"], errors="ignore")

    merged.insert(0, "variant", merged.pop("variant"))
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 零样本分数（masked-marginal）
# --------------------------------------------------------------------------- #
def zeroshot_scores(variants, wt_seq, esm_model, alphabet, converter, device) -> np.ndarray:
    """Σ[log p(mut) − log p(wt)]，仅统计非背景位点（208 在候选间近似常数）。"""
    positions = sorted({p for v in variants for p in parse_sites(v)})
    _, _, wt_tokens = converter([("wt", wt_seq)])

    logp: dict[int, np.ndarray] = {}
    chunk = 16
    with torch.inference_mode():
        for start in range(0, len(positions), chunk):
            batch_sites = positions[start : start + chunk]
            tokens = wt_tokens.repeat(len(batch_sites), 1).to(device)
            for j, pos in enumerate(batch_sites):
                tokens[j, pos] = alphabet.mask_idx
            logits = esm_model(tokens)["logits"]
            for j, pos in enumerate(batch_sites):
                logp[pos] = torch.log_softmax(logits[j, pos].float(), dim=-1).cpu().numpy()
            print(f"  零样本位点 {min(start + chunk, len(positions))}/{len(positions)}", flush=True)

    values = []
    for variant in variants:
        total = 0.0
        for m in MUT_RE.finditer(str(variant)):
            pos, wt_aa, mut_aa = int(m.group(2)), m.group(1), m.group(3)
            if pos in BACKGROUND_SITES or pos not in logp:
                continue
            idx_wt = alphabet.get_idx(wt_aa)
            idx_mut = alphabet.get_idx(mut_aa)
            total += float(logp[pos][idx_mut] - logp[pos][idx_wt])
        values.append(total)
    return np.asarray(values, dtype=float)


# --------------------------------------------------------------------------- #
# 模型预测（ESM mean-pool + 先验特征 + XGBoost）
# --------------------------------------------------------------------------- #
def esm_embeddings(sequences, esm_model, converter, device) -> np.ndarray:
    out = []
    total = len(sequences)
    with torch.inference_mode():
        for start in range(0, total, ESM_BATCH_SIZE):
            batch = [("protein", s) for s in sequences[start : start + ESM_BATCH_SIZE]]
            _, _, tokens = converter(batch)
            reps = esm_model(tokens.to(device), repr_layers=[REPR_LAYER])["representations"][REPR_LAYER]
            for i in range(reps.shape[0]):
                out.append(reps[i, 1:-1].mean(0).float().cpu().numpy())
            if (start // ESM_BATCH_SIZE) % 20 == 0:
                print(f"  ESM 表征 {min(start + ESM_BATCH_SIZE, total)}/{total}", flush=True)
    return np.vstack(out)


def main() -> None:
    import esm

    wt_seq = load_wt()
    seen = seen_nonbg_sites()
    print(f"WT 长度 {len(wt_seq)}；训练集已见非背景位点 {len(seen)} 个")

    merged = merge_pools()
    merged["mutation_list"] = merged["variant"].map(canonical_mutation_list)
    merged["positions"] = merged["variant"].map(lambda v: ";".join(str(p) for p in parse_sites(v)))
    merged["n_mut"] = merged["variant"].map(lambda v: len(parse_sites(v)))

    non_bg_sites = merged["variant"].map(lambda v: [p for p in parse_sites(v) if p not in BACKGROUND_SITES])
    merged["n_nonbg_sites"] = non_bg_sites.map(len)
    merged["n_seen_sites"] = non_bg_sites.map(lambda ps: sum(1 for p in ps if p in seen))

    def coverage(ps):
        if not ps:
            return "background_only"
        hit = sum(1 for p in ps if p in seen)
        if hit == 0:
            return "none_seen"
        if hit == len(ps):
            return "all_seen"
        return "mixed"

    merged["coverage"] = non_bg_sites.map(coverage)
    print("合并候选数:", len(merged))
    print(merged["coverage"].value_counts().to_dict())
    print("来源:", merged["pool_source"].value_counts().to_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device).eval()
    converter = alphabet.get_batch_converter()

    # ---- 零样本分数：全体候选统一计算 ----
    print("计算零样本分数…")
    merged["zero_score"] = zeroshot_scores(
        merged["variant"].tolist(), wt_seq, esm_model, alphabet, converter, device
    )

    # ---- 模型预测：仅对含≥1 个已见非背景位点的候选 ----
    pack = joblib.load(MODEL_PATH)
    xgb_model = pack["model"]
    prior_builder = PriorFeatureBuilder(BASE_DIR)

    need_model = merged["n_seen_sites"] > 0
    merged["model_pred"] = np.nan
    sub = merged[need_model].reset_index(drop=True)
    print(f"需要模型预测的候选 {len(sub)} 条，开始计算 ESM 表征…")
    if len(sub):
        x_esm = esm_embeddings(sub["sequence"].astype(str).tolist(), esm_model, converter, device)
        x_prior, _ = prior_builder.build_matrix(sub)
        x_all = np.hstack([x_esm, x_prior])
        merged.loc[need_model, "model_pred"] = xgb_model.predict(x_all)

    # ---- 名次百分位融合 ----
    merged["rank_zero"] = merged["zero_score"].rank(ascending=False, method="min")

    has_model = merged["model_pred"].notna()
    merged["rank_model"] = np.nan
    merged.loc[has_model, "rank_model"] = merged.loc[has_model, "model_pred"].rank(
        ascending=False, method="min"
    )

    def pct(rank_series, n):
        """把名次（1 最优）转为 [0,1] 百分位，名次越靠前百分位越高。"""
        if n <= 1:
            return pd.Series(0.5, index=rank_series.index)
        return (n - rank_series + 1) / n

    n_model = int(has_model.sum())
    n_total = len(merged)
    pct_model = pd.Series(0.0, index=merged.index)
    pct_model.loc[has_model] = pct(merged.loc[has_model, "rank_model"], n_model)
    pct_zero = pct(merged["rank_zero"], n_total)

    w_model = np.where(
        merged["n_nonbg_sites"] > 0,
        merged["n_seen_sites"] / merged["n_nonbg_sites"].replace(0, np.nan),
        0.0,
    )
    w_model = pd.Series(w_model, index=merged.index).fillna(0.0)
    merged["w_model"] = w_model
    merged.loc[~has_model, "w_model"] = 0.0

    merged["fused_score"] = merged["w_model"] * pct_model + (1 - merged["w_model"]) * pct_zero
    merged = merged.sort_values("fused_score", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)

    ordered = [
        "rank",
        "variant",
        "sequence",
        "mutation_list",
        "n_mut",
        "positions",
        "coverage",
        "n_nonbg_sites",
        "n_seen_sites",
        "fused_score",
        "model_pred",
        "rank_model",
        "zero_score",
        "rank_zero",
        "w_model",
        "family_support",
        "homolog_pool_size",
        "pool_source",
        "in_pool_a",
        "in_pool_b",
        "design_tier",
        "focus_seed",
    ]
    merged = merged[[c for c in ordered if c in merged.columns]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print("已写出:", OUT_PATH)
    print("\nTop 15 预览：")
    print(
        merged.head(15)[
            ["rank", "variant", "n_mut", "coverage", "fused_score", "model_pred", "zero_score"]
        ].to_string(index=False)
    )
    print("\n覆盖率分布:", merged["coverage"].value_counts().to_dict())


if __name__ == "__main__":
    main()
