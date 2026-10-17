"""生成 UGT91C1 项目报告用的可视化图集。

输出目录: results/figures/
配色统一为 黑 + 蓝 + 金 科技风，不使用红色系。
"""

from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_BLUE = "#1f4e9c"
C_BLUE_L = "#4a7fc1"
C_GOLD = "#c8a657"
C_BLACK = "#1a1a1a"
C_GREY = "#8a8a8a"
C_A = C_GOLD
C_B = C_BLUE

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.edgecolor": "#5a5a5a",
    "axes.labelcolor": C_BLACK,
    "text.color": C_BLACK,
    "figure.facecolor": "white",
})


def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def save(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", p.name)


def clean(ax, grid_axis=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.18, linestyle=":")


ds = pd.read_csv(BASE_DIR / "data" / "train_dataset.csv")
ds["sites_list"] = ds["sites_str"].fillna("").apply(
    lambda s: [int(x) for x in str(s).split(";") if str(x).strip()])
ds["batch_label"] = ds["batch"].map({"A": "批次 A（Sheet1 单点突变）", "B": "批次 B（F208M 组合库）"})


# ---------- 1. 数据总览：两批次原始活性尺度 ----------
def fig01():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    data = [ds.loc[ds.batch == "A", "activity"], ds.loc[ds.batch == "B", "activity"]]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.6))
    for patch, c in zip(bp["boxes"], [C_A, C_B]):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    for i, (d, c) in enumerate(zip(data, [C_A, C_B]), start=1):
        x = np.random.default_rng(i).normal(i, 0.055, len(d))
        ax.scatter(x, d, s=26, color=C_BLACK, alpha=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xticklabels(["批次 A\n(n=19)", "批次 B\n(n=48)"])
    ax.set_ylabel("原始比酶活（log 刻度）")
    ax.set_title("A  两批次原始活性尺度差异", fontsize=12, loc="left")
    clean(ax, "y")
    ax.annotate("批内 WT = 0.021", (1, 0.021), textcoords="offset points", xytext=(28, 6),
                fontsize=9.5, color="#333")
    ax.annotate("批内 WT = 3.700", (2, 3.7), textcoords="offset points", xytext=(28, 6),
                fontsize=9.5, color="#333")

    ax = axes[1]
    bars = ax.bar([0, 1], [len(ds[ds.batch == "A"]), len(ds[ds.batch == "B"])],
                  width=0.5, color=[C_A, C_B])
    for b, v in zip(bars, [19, 48]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, str(v), ha="center",
                fontsize=14, color=C_BLACK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["批次 A", "批次 B"])
    ax.set_ylabel("样本数"); ax.set_ylim(0, 56)
    ax.set_title("B  数据规模（共 67 条）", fontsize=12, loc="left")
    clean(ax, "y")
    fig.suptitle("两批次实验数据概览：批间存在 1.76 倍尺度漂移", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig01_data_overview.png")


# ---------- 2. 批内 WT 归一化前后 ----------
def fig02():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for b, c, lab in (("A", C_A, "批次 A"), ("B", C_B, "批次 B")):
        v = ds.loc[ds.batch == b, "activity"]
        ax.hist(np.log10(v[v > 0]), bins=14, color=c, alpha=0.7, label=lab)
    ax.set_xlabel("log10(原始比酶活)"); ax.set_ylabel("频数")
    ax.set_title("A  归一化前：两批次明显分离", fontsize=12, loc="left")
    ax.legend(frameon=False); clean(ax, "y")

    ax = axes[1]
    for b, c, lab in (("A", C_A, "批次 A"), ("B", C_B, "批次 B")):
        v = ds.loc[ds.batch == b, "activity_rel"]
        ax.hist(v, bins=16, color=c, alpha=0.7, label=lab)
    ax.axvline(1.0, color=C_BLACK, linestyle="--", linewidth=1.2)
    ax.text(1.05, ax.get_ylim()[1] * 0.92, "WT 基准 = 1.0", fontsize=9.5, color="#333")
    ax.set_xlabel("activity_rel（批内 WT 归一）"); ax.set_ylabel("频数")
    ax.set_title("B  归一化后：分布对齐，可直接合并", fontsize=12, loc="left")
    ax.legend(frameon=False); clean(ax, "y")
    fig.suptitle("批内 WT 归一化消除批间偏差", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig02_normalization.png")


# ---------- 3. 突变位点图谱 ----------
def fig03():
    L = 462
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot([0, L], [1, 1], color=C_GREY, linewidth=1.4, zorder=1)
    sites_a = sorted({p for s in ds.loc[ds.batch == "A", "sites_list"] for p in s})
    sites_b = sorted({p for s in ds.loc[ds.batch == "B", "sites_list"] for p in s})
    ax.scatter(sites_a, [1.22] * len(sites_a), marker="v", s=110, color=C_A, zorder=3,
               label="批次 A 位点（%d 个）" % len(sites_a))
    ax.scatter(sites_b, [0.78] * len(sites_b), marker="^", s=110, color=C_B, zorder=3,
               label="批次 B 位点（%d 个）" % len(sites_b))
    ax.scatter([208], [1.0], marker="D", s=150, color=C_BLACK, zorder=4, label="F208M 背景位点")
    hotspots = {93: "H93", 129: "V129", 379: "F379", 208: "F208"}
    for p, name in hotspots.items():
        ax.annotate(name, (p, 1.0), textcoords="offset points", xytext=(0, 22),
                    ha="center", fontsize=10, color=C_BLACK)
    ax.set_xlim(0, L + 6); ax.set_ylim(0.55, 1.5)
    ax.set_yticks([]); ax.set_xlabel("UGT91C1 序列位置（1–462 aa）")
    ax.set_title("实验突变位点在序列上的分布", fontsize=12.5, loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.15, linestyle=":")
    fig.tight_layout()
    save(fig, "fig03_mutation_map.png")


# ---------- 4. 突变数 vs 活性 ----------
def fig04():
    d = ds[ds.batch == "B"]
    groups = sorted(d["n_mut"].unique())
    data = [d.loc[d.n_mut == g, "activity_rel"].to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.6))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BLUE); patch.set_alpha(0.8)
    rng = np.random.default_rng(7)
    for i, (g, arr) in enumerate(zip(groups, data), start=1):
        ax.scatter(rng.normal(i, 0.06, len(arr)), arr, s=34, color=C_BLACK, alpha=0.6, zorder=3)
        ax.text(i, -0.28, f"n={len(arr)}", ha="center", fontsize=9.5, color="#555")
    ax.axhline(1.0, color=C_GOLD, linestyle="--", linewidth=1.2)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{g} 个突变" for g in groups])
    ax.set_ylim(-0.35, 3.6)
    ax.set_ylabel("activity_rel（批内 WT 归一）")
    ax.set_title("批次 B：突变数越多，活性越难维持", fontsize=12.5, loc="left")
    clean(ax, "y")
    fig.tight_layout()
    save(fig, "fig04_nmut_vs_activity.png")


# ---------- 5. 位点 × 取代 热图 ----------
def fig05():
    d = ds[ds.batch == "B"].copy()
    rows = {}
    for _, r in d.iterrows():
        if pd.isna(r["mutation_list"]) or not str(r["mutation_list"]).strip():
            continue
        for tok in str(r["mutation_list"]).split(";"):
            tok = tok.strip()
            if len(tok) < 3 or not tok[1:-1].isdigit():
                continue
            pos = int(tok[1:-1])
            if pos == 208:
                continue
            rows.setdefault(pos, {})[tok[-1]] = max(rows.setdefault(pos, {}).get(tok[-1], 0), r["activity_rel"])
    positions = sorted(rows)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    M = np.full((len(positions), len(aas)), np.nan)
    for i, p in enumerate(positions):
        for j, aa in enumerate(aas):
            if aa in rows[p]:
                M[i, j] = rows[p][aa]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=3.5)
    ax.set_xticks(range(len(aas))); ax.set_xticklabels(aas, fontsize=10)
    ax.set_yticks(range(len(positions))); ax.set_yticklabels([f"{p}" for p in positions], fontsize=10)
    ax.set_xlabel("取代氨基酸"); ax.set_ylabel("位点")
    for i in range(len(positions)):
        for j in range(len(aas)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 1.9 else "#1a1a1a")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("activity_rel（该位点观察到的最高值）", fontsize=10)
    ax.set_title("批次 B 各位点的取代–活性图谱（空白 = 未测）", fontsize=12.5, loc="left")
    fig.tight_layout()
    save(fig, "fig05_site_substitution_heatmap.png")


# ---------- 6. 模型对比 ----------
def fig06():
    m = pd.read_csv(BASE_DIR / "results" / "tier_cv_metrics.csv")
    label_map = {
        "ZS_zeroshot_only": "零样本分数\n(参数无关)",
        "M0_meanpool_prior_XGB": "mean-pool+先验\nXGB (线上配方)",
        "M1_zeroshot_prior_XGB": "零样本+先验\nXGB",
        "M2_sitelocal_zs_prior_PCA_Ridge": "位点窗口+零样本\n+PCA Ridge",
        "M3_sitelocal_zs_prior_XGB": "位点窗口+零样本\nXGB",
        "M4_fusion_PCA_Ridge": "全特征融合\n+PCA Ridge",
    }
    m["short"] = m["model"].map(label_map)
    m = m.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    y = np.arange(len(m))

    ax = axes[0]
    cols = [C_BLUE if v > 0 else "#b0b0b0" for v in m["macro_spearman"]]
    ax.barh(y, m["macro_spearman"], color=cols, height=0.55)
    ax.barh(y, m["spearman_pooled"], color=C_GOLD, height=0.22, alpha=0.9)
    ax.axvline(0, color=C_BLACK, linewidth=1)
    for i, (v1, v2) in enumerate(zip(m["macro_spearman"], m["spearman_pooled"])):
        ax.text(v1 + (0.02 if v1 >= 0 else -0.02), i - 0.22, f"{v1:.2f}", va="center",
                ha="left" if v1 >= 0 else "right", fontsize=9.5, color=C_BLUE)
        ax.text(v2 + (0.02 if v2 >= 0 else -0.02), i + 0.22, f"{v2:.2f}", va="center",
                ha="left" if v2 >= 0 else "right", fontsize=9, color="#8a6f2a")
    ax.set_yticks(y); ax.set_yticklabels(m["short"], fontsize=9.5)
    ax.set_xlabel("Spearman 相关系数（按位点留出 CV）")
    ax.set_title("A  排序能力：macro（粗条）vs pooled（细条）", fontsize=12, loc="left")
    ax.set_xlim(-0.45, 0.62)
    clean(ax, "x")

    ax = axes[1]
    ax.barh(y - 0.19, m["macro_auc"], color=C_BLUE, height=0.34, label="AUC (activity_rel>1)")
    ax.barh(y + 0.19, m["macro_recall10"], color=C_GOLD, height=0.34, label="Recall@10")
    ax.axvline(0.5, color=C_GREY, linestyle="--", linewidth=1)
    ax.text(0.51, len(m) - 0.5, "AUC 随机线 0.5", fontsize=9, color="#555")
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("指标值"); ax.set_xlim(0, 0.92)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.set_title("B  判别力与 top-10 命中率", fontsize=12, loc="left")
    clean(ax, "x")
    fig.suptitle("按位点留出交叉验证：模型排序能力对比（27 个位点 / 64 个变体）",
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig06_model_comparison.png")


# ---------- 7. OOF 预测 vs 实测 ----------
def fig07():
    d = pd.read_csv(BASE_DIR / "results" / "tier_oof_predictions.csv")
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    for b, c, lab in (("A", C_A, "批次 A"), ("B", C_B, "批次 B")):
        g = d[d.batch == b]
        ax.scatter(g["activity_rel"], g["pred_M0_meanpool_prior_XGB"], s=58, color=c,
                   edgecolor="white", linewidth=1.0, label=f"{lab}（Spearman {spearman(g.activity_rel, g['pred_M0_meanpool_prior_XGB']):.2f}）",
                   alpha=0.9)
    lim = [-0.4, d["activity_rel"].max() * 1.1]
    ax.plot(lim, [v / 3.7 for v in lim], "--", color=C_GREY, linewidth=1.2, label="理想线")
    ax.set_xlabel("实测 activity_rel"); ax.set_ylabel("留出预测 activity_rel")
    ax.set_title("线上配方（mean-pool+先验 XGB）留出预测 vs 实测", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9.5)
    clean(ax)
    ax.grid(alpha=0.16, linestyle=":")
    fig.tight_layout()
    save(fig, "fig07_oof_scatter.png")


# ---------- 8. 零样本分数 vs 实测 ----------
def fig08():
    d = pd.read_csv(BASE_DIR / "results" / "tier_oof_predictions.csv")
    zs = pd.read_csv(BASE_DIR / "data" / "train_dataset.csv")
    zmap = dict(zip(zs["variant"], np.load(BASE_DIR / "data" / "feat_zeroshot.npy")[:, 5]))
    d = d[d.variant != "WT"].copy()
    d["zs"] = d["variant"].map(zmap)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    for b, c, lab in (("A", C_A, "批次 A"), ("B", C_B, "批次 B")):
        g = d[d.batch == b]
        ax.scatter(g["zs"], g["activity_rel"], s=56, color=c, edgecolor="white",
                   linewidth=1.0, alpha=0.9,
                   label=f"{lab}：Spearman {spearman(g.zs, g.activity_rel):.2f}")
    ax.axhline(1.0, color=C_GREY, linestyle="--", linewidth=1.1)
    ax.set_xlabel("零样本 masked-marginal 分数（非背景位点 log-odds 之和）")
    ax.set_ylabel("实测 activity_rel")
    ax.set_title("A  零样本分数 vs 实测活性（分批次）", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9.5); clean(ax); ax.grid(alpha=0.16, linestyle=":")

    ax = axes[1]
    names = ["零样本分数", "线上配方\nXGB", "零样本+先验\nXGB", "位点窗口\n+PCA Ridge"]
    vals = [0.477, 0.287, 0.170, -0.123]
    cols = [C_BLUE, C_BLUE, C_BLUE, "#b0b0b0"]
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:.3f}", ha="center", fontsize=10.5)
    ax.axhline(0, color=C_BLACK, linewidth=1)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("macro Spearman"); ax.set_ylim(-0.28, 0.58)
    ax.set_title("B  各方法宏观排序能力汇总", fontsize=12, loc="left")
    clean(ax, "y")
    fig.suptitle("零样本进化分数是最稳的排序信号", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig08_zeroshot_vs_activity.png")


# ---------- 9. 候选库分数分布 ----------
def fig09():
    c = pd.read_csv(BASE_DIR / "results" / "zeroshot_candidate_ranking.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.hist(c["zs_nonbg_sum"], bins=40, color=C_BLUE, alpha=0.85)
    ax.axvline(0, color=C_BLACK, linestyle="--", linewidth=1.2)
    ax.set_xlabel("零样本分数（非背景位点合计）"); ax.set_ylabel("候选数")
    ax.set_title(f"A  候选库分数分布（n={len(c)}）", fontsize=12, loc="left")
    clean(ax, "y")

    ax = axes[1]
    cnt = c["n_mut"].value_counts().sort_index()
    bars = ax.bar(cnt.index, cnt.values, color=C_GOLD, width=0.6)
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontsize=10)
    ax.set_xlabel("每个候选携带的突变数"); ax.set_ylabel("候选数")
    ax.set_title("B  候选库突变数构成", fontsize=12, loc="left")
    clean(ax, "y")
    fig.suptitle("候选突变体库概览（1371 条）", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig09_candidate_score_dist.png")


# ---------- 10. 候选库 Top20 ----------
def fig10():
    c = pd.read_csv(BASE_DIR / "results" / "zeroshot_candidate_ranking.csv").head(20)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    y = np.arange(len(c))[::-1]
    ax.barh(y, c["zs_nonbg_sum"], color=C_BLUE, height=0.6)
    for i, (yy, v) in enumerate(zip(y, c["zs_nonbg_sum"])):
        ax.text(v + 0.08, yy, f"{v:.2f}", va="center", fontsize=9.5, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(c["variant"], fontsize=10)
    ax.set_xlabel("零样本分数（非背景位点合计）")
    ax.set_xlim(0, c["zs_nonbg_sum"].max() * 1.15)
    ax.set_title("候选库零样本排名 Top 20", fontsize=12.5, loc="left")
    clean(ax, "x")
    fig.tight_layout()
    save(fig, "fig10_candidate_top20.png")


# ---------- 11. 突变数 vs 零样本分数（sum 偏差说明） ----------
def fig11():
    c = pd.read_csv(BASE_DIR / "results" / "zeroshot_candidate_ranking.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, col, title in ((axes[0], "zs_nonbg_sum", "A  zs_sum：会随突变数累加负分"),
                           (axes[1], "zs_nonbg_mean", "B  zs_mean：按位点平均，消除数量偏差")):
        groups = sorted(c["n_mut"].unique())
        data = [c.loc[c.n_mut == g, col].to_numpy() for g in groups]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(C_BLUE if col.endswith("sum") else C_GOLD); patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([f"{g}" for g in groups])
        ax.set_xlabel("突变数"); ax.set_title(title, fontsize=12, loc="left")
        clean(ax, "y")
    fig.suptitle("零样本分数的两种聚合方式：多突变场景建议同时看 zs_mean",
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig11_nmut_vs_zeroshot.png")


# ---------- 12. Top-k 命中曲线 ----------
def fig12():
    d = pd.read_csv(BASE_DIR / "results" / "sheet1_predictions.csv")
    d = d[d.variant.str.upper() != "WT"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    ks = range(1, len(d) + 1)
    for col, c, lab in (("pred_activity_rel", C_BLUE, "ESM+XGB 预测"),
                        ("zeroshot_score", C_GOLD, "零样本分数")):
        hits = []
        for k in ks:
            top_pred = set(d.nlargest(k, col).index)
            top_true = set(d.nlargest(k, "activity_rel_true").index)
            hits.append(len(top_pred & top_true) / k)
        ax.plot(list(ks), hits, marker="o", markersize=4.5, color=c, label=lab, linewidth=1.8)
    ax.axhline(1.0, color=C_GREY, linestyle=":", linewidth=1)
    ax.set_xlabel("取前 k 个候选"); ax.set_ylabel("命中率 (Recall@k)")
    ax.set_ylim(0, 1.08); ax.set_xticks(list(ks))
    ax.set_title("A  Sheet1 真实突变体：top-k 命中曲线", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=10); clean(ax); ax.grid(alpha=0.16, linestyle=":")

    ax = axes[1]
    top_pred = d.nlargest(10, "pred_activity_rel")["variant"].tolist()
    top_zs = d.nlargest(10, "zeroshot_score")["variant"].tolist()
    true5 = d.nlargest(5, "activity_rel_true")["variant"].tolist()
    rows = [("实测 top5", true5, C_BLACK), ("ESM+XGB top10", top_pred, C_BLUE),
            ("零样本 top10", top_zs, C_GOLD)]
    for i, (name, lst, c) in enumerate(rows):
        ax.barh(i, len(lst), color="#efefef", height=0.5)
        for j, v in enumerate(lst):
            hit = v in true5
            ax.text(j + 0.5, i, v, ha="center", va="center", fontsize=8.5,
                    color="white" if hit else "#444",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=c if hit else "#f4f4f4", edgecolor="none"))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0, 10); ax.set_xticks([]); ax.invert_yaxis()
    ax.set_title("B  高活性突变命中情况（深色 = 命中真实 top5）", fontsize=12, loc="left")
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    fig.suptitle("Sheet1 实验数据上的候选排序命中分析（18 个真实单点突变体）",
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig12_topk_curve.png")


for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12):
    fn()
print("\nAll figures written to", FIG_DIR)
