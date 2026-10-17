"""Tier 1: 为统一训练集抽取 ESM-2 (650M) 冻结特征。

三类特征（全部冻结点积，不做任何微调）：
  1. mean-pool：整条序列残基表征的均值（1280 维）—— 旧基线所用表征。
  2. site-window：仅在突变位点 ± 邻域取残基表征并平均（1280 维），聚焦局部环境。
     同时输出 site-local（仅非背景位点，剔除所有变体共有的 F208M 常数项）。
  3. zero-shot：masked-marginal 对数几率分数，log p(mut) - log p(wt)。
     在 WT 序列上逐位点 mask 后取该位点分布，对所有突变位点求和（ESM-1v 风格）。
     同时输出 zs_nonbg（仅非背景位点），避免背景位点常数造成跨批次偏移。

前两项在"突变后"的序列上算（上下文反映该突变），第三项在 WT 上下文上算（衡量进化接受度）。

输出:
  data/feat_meanpool.npy   (N, 1280)
  data/feat_site.npy       (N, 1280)  全部突变位点窗口
  data/feat_site_local.npy (N, 1280)  仅非背景位点窗口
  data/feat_zeroshot.npy   (N, 7)  [zs_sum, zs_mean, zs_min, zs_max, zs_n, zs_nonbg_sum, zs_nonbg_n]

行序约束：npy 第 i 行对应 train_dataset.csv 第 i 行。因此训练集一旦追加新数据，
必须整体重跑本脚本，不能只 append。

命令行
------
  python scripts/extract_features_v2.py

  --dataset PATH  训练集 CSV 路径，默认 data/train_dataset.csv。
  --out-dir DIR   特征输出目录，默认 data/。这两个参数只用于隔离测试
                  （指向临时副本，不覆盖正式特征缓存）。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm

BASE_DIR = Path(__file__).resolve().parents[1]
DS_PATH = BASE_DIR / "data" / "train_dataset.csv"
OUT_MEAN = BASE_DIR / "data" / "feat_meanpool.npy"
OUT_SITE = BASE_DIR / "data" / "feat_site.npy"
OUT_SITE_LOCAL = BASE_DIR / "data" / "feat_site_local.npy"
OUT_ZS = BASE_DIR / "data" / "feat_zeroshot.npy"

REPR_LAYER = 33
WINDOW = 2
BACKGROUND = {208}  # 由 Tier 0 自动识别出的背景位点


def parse_sites(sites_str):
    if pd.isna(sites_str) or not str(sites_str).strip():
        return []
    return [int(x) for x in str(sites_str).split(";") if str(x).strip()]


def main():
    parser = argparse.ArgumentParser(description="抽取统一训练集的 ESM-2 冻结特征")
    parser.add_argument("--dataset", default=str(DS_PATH), help="训练集 CSV 路径")
    parser.add_argument("--out-dir", default=str(BASE_DIR / "data"), help="特征输出目录")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mean = out_dir / OUT_MEAN.name
    out_site = out_dir / OUT_SITE.name
    out_site_local = out_dir / OUT_SITE_LOCAL.name
    out_zs = out_dir / OUT_ZS.name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    df = pd.read_csv(ds_path)
    df["sites_list"] = df["sites_str"].apply(parse_sites)
    print(f"训练集 {ds_path}：{len(df)} 行", flush=True)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    # ---------- 1) 每条序列的 token 表征 ----------
    mean_feats, site_feats, site_local_feats = [], [], []
    total = len(df)
    with torch.no_grad():
        for idx, (seq, sites) in enumerate(
            zip(df["sequence"].tolist(), df["sites_list"]), start=1
        ):
            _, _, tokens = batch_converter([("p", seq)])
            tokens = tokens.to(device)
            res = model(tokens, repr_layers=[REPR_LAYER])["representations"][REPR_LAYER][0][1 : len(seq) + 1]
            L = len(seq)
            mp = res.mean(0).float().cpu().numpy()
            mean_feats.append(mp)

            def window_mean(positions):
                vecs = []
                for p in positions:
                    lo = max(0, p - 1 - WINDOW)
                    hi = min(L, p - 1 + WINDOW + 1)
                    vecs.append(res[lo:hi].mean(0))
                return torch.stack(vecs).mean(0).float().cpu().numpy() if vecs else None

            site_feats.append(window_mean(sites) if sites else mp)
            local = [p for p in sites if p not in BACKGROUND]
            site_local_feats.append(window_mean(local) if local else mp)
            if idx % 10 == 0 or idx == total:
                print(f"  序列表征 {idx}/{total}", flush=True)

    mean_feats = np.asarray(mean_feats, dtype=np.float32)
    site_feats = np.asarray(site_feats, dtype=np.float32)
    site_local_feats = np.asarray(site_local_feats, dtype=np.float32)
    print("meanpool:", mean_feats.shape, "site:", site_feats.shape, "site_local:", site_local_feats.shape)

    # ---------- 2) masked-marginal 零样本：逐位点 mask WT ----------
    wt_seq = df.loc[df["variant"].str.upper().eq("WT"), "sequence"].iloc[0]
    all_sites = sorted({p for sites in df["sites_list"] for p in sites})
    print(f"Unique masked sites: {len(all_sites)}")

    _, _, wt_tokens = batch_converter([("wt", wt_seq)])
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

    aa_idx = {aa: alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    def deltas_for(muts):
        out = []
        tokens = [t for t in str(muts).split(";") if t.strip()] if pd.notna(muts) else []
        for token in tokens:
            wt_aa, pos, mut_aa = token[0], int(token[1:-1]), token[-1]
            logp = site_logp.get(pos)
            if logp is None:
                continue
            out.append((pos, float(logp[aa_idx[mut_aa]] - logp[aa_idx[wt_aa]])))
        return out

    zs_feats = []
    for muts in df["mutation_list"]:
        d_all = deltas_for(muts)
        vals = [v for _, v in d_all]
        nb = [v for p, v in d_all if p not in BACKGROUND]
        row = [sum(vals), float(np.mean(vals)), min(vals), max(vals), len(vals)] if vals else [0.0] * 5
        row += [sum(nb), len(nb)] if nb else [0.0, 0.0]
        zs_feats.append(row)
    zs_feats = np.asarray(zs_feats, dtype=np.float32)
    print("zeroshot:", zs_feats.shape)

    np.save(out_mean, mean_feats)
    np.save(out_site, site_feats)
    np.save(out_site_local, site_local_feats)
    np.save(out_zs, zs_feats)
    print("Saved:", out_mean, out_site, out_site_local, out_zs, flush=True)

    df["zs_nonbg_sum"] = zs_feats[:, 5]
    print("\n各批次内 zs_nonbg_sum vs activity_rel 的 Spearman:")
    for batch, g in df.groupby("batch"):
        r = g["zs_nonbg_sum"].corr(g["activity_rel"], method="spearman")
        print(f"  batch {batch}: {r:.3f} (n={len(g)})")


if __name__ == "__main__":
    main()
