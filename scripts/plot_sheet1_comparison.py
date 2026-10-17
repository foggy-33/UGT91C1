"""生成 Sheet1 实验真值 vs 模型预测的对比图。

左图：XGB 配方预测值 vs 实测 activity_rel 散点（含 y=x 参考线）。
右图：名次对齐图（实测 / ESM+XGB 预测 / 零样本分数 三种名次）。

配色统一为黑 + 蓝 + 金，避免红色系。
输出: results/sheet1_prediction_comparison.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
IN_CSV = BASE_DIR / "results" / "sheet1_predictions.csv"
OUT_PNG = BASE_DIR / "results" / "sheet1_prediction_comparison.png"

C_TRUE = "#1a1a1a"
C_XGB = "#1f4e9c"
C_ZS = "#c8a657"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.edgecolor": "#4d4d4d",
    "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a",
})


def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    df = pd.read_csv(IN_CSV)
    wt = df[df["variant"].str.upper() == "WT"]
    mut = df[df["variant"].str.upper() != "WT"].copy()

    r_xgb = spearman(mut["activity_rel_true"], mut["pred_activity_rel"])
    r_zs = spearman(mut["activity_rel_true"], mut["zeroshot_score"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})

    # ---- 左：预测 vs 实测散点 ----
    ax = axes[0]
    lim = [0, max(df["activity_rel_true"].max(), df["pred_activity_rel"].max()) * 1.12]
    ax.plot(lim, lim, "--", color="#9a9a9a", linewidth=1.2, zorder=1, label="理想预测线 y=x")
    ax.axhline(1.0, color="#d8d8d8", linewidth=1.0, zorder=0)
    ax.axvline(1.0, color="#d8d8d8", linewidth=1.0, zorder=0)
    ax.scatter(mut["activity_rel_true"], mut["pred_activity_rel"],
               s=70, color=C_XGB, edgecolor="white", linewidth=1.2, zorder=3, label="单点突变体")
    ax.scatter(wt["activity_rel_true"], wt["pred_activity_rel"],
               s=110, color=C_TRUE, marker="D", edgecolor="white", linewidth=1.2, zorder=4, label="WT 基准")
    for _, row in df[df["activity_rel_true"] > 1.0].iterrows():
        ax.annotate(row["variant"], (row["activity_rel_true"], row["pred_activity_rel"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=9.5, color="#333333")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("实测 activity_rel（批内 WT 归一）")
    ax.set_ylabel("ESM2 + XGB 预测 activity_rel")
    ax.set_title(f"A  预测值 vs 实测值\nSpearman = {r_xgb:.3f}", fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.grid(alpha=0.18, linestyle=":")

    # ---- 右：名次对齐 ----
    ax = axes[1]
    mut = mut.sort_values("rank_true").reset_index(drop=True)
    y = np.arange(len(mut))
    for i, row in mut.iterrows():
        ax.plot([row["rank_true"], row["rank_zeroshot"], row["rank_pred"]], [i, i, i],
                color="#c9c9c9", linewidth=1.2, zorder=1)
    ax.scatter(mut["rank_true"], y, s=62, color=C_TRUE, zorder=3, label="实测名次")
    ax.scatter(mut["rank_pred"], y, s=62, color=C_XGB, zorder=3, label="ESM+XGB 预测名次")
    ax.scatter(mut["rank_zeroshot"], y, s=62, color=C_ZS, zorder=3, label="零样本分数名次")
    ax.set_yticks(y)
    ax.set_yticklabels(mut["variant"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("名次（1 = 活性最高，越靠左越好）")
    ax.set_xticks(range(1, len(mut) + 1))
    ax.set_xlim(0.2, len(mut) + 0.8)
    ax.set_ylim(len(mut) - 0.35, -0.7)
    ax.axvspan(0.2, 5.5, color="#1f4e9c", alpha=0.06, zorder=0)
    ax.text(2.85, len(mut) - 0.5, "真实 top5 区间", ha="center", va="bottom",
            fontsize=9.5, color="#4d4d4d")
    ax.set_title(f"B  名次对齐（真实 top5 覆盖 5/5）\nXGB Spearman = {r_xgb:.3f}   |   零样本 Spearman = {r_zs:.3f}",
                 fontsize=12, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.18, linestyle=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
        axes[0].spines[s].set_visible(False)

    fig.suptitle("UGT91C1 Sheet1 实验数据（18 个单点突变体）模型预测对比",
                 fontsize=13.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, facecolor="white")
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
