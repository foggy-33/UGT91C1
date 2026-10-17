"""把新一批实验数据并入统一训练集 data/train_dataset.csv。

为什么单独一个脚本
------------------
新批次实验的原始比酶活与历史批次存在批间漂移，必须先按**批内 WT**归一化
（activity_rel = activity / wt_batch，WT=1）才能进训练集；同时要补齐位点标注与
序列列，保持与 scripts/build_dataset.py（Tier 0）完全一致的列口径与列顺序。
追加是破坏性写操作，因此这里强制先备份原文件。

本模块只依赖 pandas，不碰 torch / GPU，既可作为 CLI 使用，也可被
Streamlit「数据入库」页 import 复用（预览与真正入库共用同一套解析逻辑）。

页面调用约定
------------
  parse_new_batch(df, batch_label, wt_activity, existing) -> (rows, warnings, errors)
      rows     : 规范化后的新行 DataFrame（列顺序与训练集一致，可直接 concat）
      warnings : 软问题清单（跨批次重复、同批次重复被跳过等）
      errors   : 硬问题清单（突变 token 非法、WT 残基不匹配、缺 WT、活性非正数等）
      errors 非空时页面必须 st.stop()，不得入库。

CLI
---
  python scripts/ingest_batch.py --input new_batch.csv --batch C [--wt-activity 3.7] [--dry-run]

  --dry-run  只解析预览、不写任何文件（测试用）。
  --dataset  指定训练集路径，默认 data/train_dataset.csv；测试时指向临时副本，
             确保不碰正式数据。
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from build_dataset import (  # noqa: E402
    MUT_RE,
    apply_mutations,
    parse_mutations,
    read_fasta,
    to_mutation_list,
)

DS_PATH = BASE_DIR / "data" / "train_dataset.csv"
WT_FASTA = BASE_DIR / "data" / "wt.fasta"

BACKGROUND = {208}
COLUMNS = [
    "variant",
    "activity",
    "batch",
    "mutation_list",
    "sequence",
    "wt_batch",
    "activity_rel",
    "sites",
    "sites_str",
    "n_mut",
    "is_background_only",
]


# --------------------------------------------------------------------------- #
# 读取与规范化
# --------------------------------------------------------------------------- #
def load_dataset(dataset_path: Path = DS_PATH) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"训练集缺少列：{missing}")
    return df[COLUMNS]


def next_batch_label(dataset_path: Path = DS_PATH) -> str:
    """返回下一个未使用的批次字母（A、B 已用 → C）。"""
    used = set()
    if Path(dataset_path).exists():
        for value in pd.read_csv(dataset_path, usecols=["batch"])["batch"].dropna():
            text = str(value).strip().upper()
            if len(text) == 1 and text.isalpha():
                used.add(text)
    for code in range(ord("A"), ord("Z") + 1):
        letter = chr(code)
        if letter not in used:
            return letter
    return "Z"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def parse_new_batch(df, batch_label, wt_activity=None, existing=None):
    """解析并规范化一批新实验数据；返回 (rows, warnings, errors)。"""
    warnings: list[str] = []
    errors: list[str] = []
    df = _normalize_columns(df)

    for col in ("variant", "activity"):
        if col not in df.columns:
            errors.append(f"缺少必需列 `{col}`（最小格式：variant, activity）")
    if errors:
        return pd.DataFrame(columns=COLUMNS), warnings, errors

    df["variant"] = df["variant"].astype(str).str.strip()
    df["activity"] = pd.to_numeric(df["activity"], errors="coerce")
    df = df[df["variant"].ne("") & df["variant"].str.lower().ne("nan")].reset_index(drop=True)
    if df.empty:
        errors.append("没有找到有效的突变体记录")
        return pd.DataFrame(columns=COLUMNS), warnings, errors

    # ---------- WT 参考值 ----------
    wt_rows = df[df["variant"].str.upper().eq("WT")]
    if len(wt_rows) > 1:
        errors.append(f"同一批次只能有一行 WT，当前有 {len(wt_rows)} 行")
    wt_value = None
    if len(wt_rows) == 1:
        wt_value = wt_rows["activity"].iloc[0]
        if pd.isna(wt_value) or float(wt_value) <= 0:
            errors.append(f"WT 行的 activity 必须为正数，当前为 {wt_rows['activity'].iloc[0]}")
            wt_value = None
    if wt_value is None and wt_activity is not None:
        if float(wt_activity) <= 0:
            errors.append(f"--wt-activity 必须为正数，当前为 {wt_activity}")
        else:
            wt_value = float(wt_activity)
    if wt_value is None and len(wt_rows) != 1:
        errors.append("缺少 WT 参考值：请在 CSV 中放一行 variant=WT，或用 --wt-activity 指定")
    if errors:
        return pd.DataFrame(columns=COLUMNS), warnings, errors
    wt_batch = float(wt_value)

    wt_seq = read_fasta(WT_FASTA)
    batch_label = str(batch_label).strip().upper()

    rows = []
    for _, row in df.iterrows():
        variant = row["variant"]
        activity = row["activity"]
        is_wt = variant.upper() == "WT"
        if not is_wt and (pd.isna(activity) or float(activity) < 0):
            errors.append(f"{variant}：activity 不能为空或负数（当前 {row['activity']}）")
            continue

        if is_wt:
            mutation_list, sequence, sites = "", wt_seq, []
        else:
            mutation_list = to_mutation_list(variant)
            tokens = [t for t in mutation_list.split(";") if t]
            if not tokens:
                errors.append(f"{variant}：无法解析出任何突变（格式应为 F208M 或 F208M;V129A）")
                continue
            bad = [t for t in tokens if not MUT_RE.match(t)]
            if bad:
                errors.append(f"{variant}：突变写法非法 {bad}（应为 通配符-位点-氨基酸，如 V129A）")
                continue
            try:
                sequence = apply_mutations(wt_seq, mutation_list)
            except ValueError as exc:
                errors.append(f"{variant}：{exc}")
                continue
            sites = [m["pos"] for m in parse_mutations(mutation_list)]

        rows.append(
            {
                "variant": variant,
                "activity": float(activity),
                "batch": batch_label,
                "mutation_list": mutation_list,
                "sequence": sequence,
                "wt_batch": wt_batch,
                "activity_rel": 1.0 if is_wt else float(activity) / wt_batch,
                "sites": sites,
                "sites_str": ";".join(str(p) for p in sites),
                "n_mut": len(sites),
                "is_background_only": bool(sites) and all(p in BACKGROUND for p in sites),
            }
        )

    if errors:
        return pd.DataFrame(columns=COLUMNS), warnings, errors

    new = pd.DataFrame(rows, columns=COLUMNS)

    # ---------- 去重 ----------
    if existing is None:
        existing = load_dataset()
    existing_pairs = set(zip(existing["variant"].astype(str), existing["batch"].astype(str)))
    existing_variants = set(existing["variant"].astype(str))

    dup_in_batch = new["variant"].duplicated(keep="first")
    if dup_in_batch.any():
        dup_names = new.loc[dup_in_batch, "variant"].tolist()
        warnings.append(f"本批次内重复出现，只保留第一条：{dup_names}")
        new = new[~dup_in_batch].reset_index(drop=True)

    # 同批次同突变体 = 完全重复 → 跳过，不静默覆盖既有测量
    dup_exact = [(str(v), batch_label) in existing_pairs for v in new["variant"]]
    if any(dup_exact):
        skipped = new.loc[dup_exact, "variant"].tolist()
        warnings.append(f"与训练集完全重复（同批次同突变体），本次跳过：{skipped}")
        new = new[~pd.Series(dup_exact, index=new.index)].reset_index(drop=True)

    # 其它批次已有同名突变体 → 保留（属于新批次的独立测量），但明确提示
    # 注意排除 WT：每个批次都必然带一行 WT，提示它「已有测量」是噪音。
    repeats = [v for v in new["variant"] if str(v).upper() != "WT" and v in existing_variants]
    if repeats:
        warnings.append(f"以下突变体在其它批次已有测量，本批次作为新测量一并入库：{repeats}")

    if new.empty:
        warnings.append("去重后没有可入库的新记录")
    else:
        warnings.append(f"待入库 {len(new)} 条；批内 WT = {wt_batch:g}")

    return new, warnings, errors


# --------------------------------------------------------------------------- #
# 追加写盘
# --------------------------------------------------------------------------- #
def _coerce_sites(df: pd.DataFrame) -> pd.DataFrame:
    """统一 sites 列为 Python list，避免新旧行写法不一致。"""
    def fix(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = ast.literal_eval(value)
                return list(parsed) if isinstance(parsed, (list, tuple)) else []
            except (ValueError, SyntaxError):
                return []
        return []

    df = df.copy()
    df["sites"] = df["sites"].apply(fix)
    return df


def append_rows(rows: pd.DataFrame, dataset_path: Path = DS_PATH) -> dict:
    """备份后把新行追加进训练集 CSV；返回 (备份路径, 追加前行数, 追加后行数)。"""
    dataset_path = Path(dataset_path)
    rows = _coerce_sites(rows)[COLUMNS]
    existing = _coerce_sites(load_dataset(dataset_path))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = dataset_path.with_name(f"{dataset_path.name}.bak-{stamp}")
    shutil.copy2(dataset_path, backup)

    merged = pd.concat([existing, rows], ignore_index=True)[COLUMNS]
    merged.to_csv(dataset_path, index=False)
    return {
        "backup": str(backup),
        "n_before": int(len(existing)),
        "n_after": int(len(merged)),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="把新一批实验数据并入统一训练集")
    parser.add_argument("--input", required=True, help="新批次 CSV（需含 variant、activity 列）")
    parser.add_argument("--batch", required=True, help="批次标签，如 C")
    parser.add_argument("--wt-activity", type=float, default=None, help="批内 WT 比酶活；CSV 里已有 WT 行时可不给")
    parser.add_argument("--dataset", default=str(DS_PATH), help="训练集路径（测试时指向临时副本）")
    parser.add_argument("--dry-run", action="store_true", help="只解析预览，不写任何文件")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    raw = pd.read_csv(args.input)
    existing = load_dataset(dataset_path) if dataset_path.exists() else None
    rows, warnings, errors = parse_new_batch(
        raw, args.batch, wt_activity=args.wt_activity, existing=existing
    )

    print(f"解析结果：可入库 {len(rows)} 条")
    if not rows.empty:
        cols = ["variant", "activity", "activity_rel", "sites_str", "n_mut", "is_background_only"]
        print(rows[cols].to_string(index=False))
    for msg in warnings:
        print("告警:", msg)
    for msg in errors:
        print("错误:", msg)
    if errors:
        print("解析失败，未做任何写入。")
        return 1

    if args.dry_run:
        print("dry-run：未写入任何文件。")
        return 0

    info = append_rows(rows, dataset_path)
    print(f"备份：{info['backup']}")
    print(f"行数：{info['n_before']} → {info['n_after']}")
    print("注意：训练集已变更，需重跑 extract_features_v2.py 刷新特征缓存后才能重训。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
