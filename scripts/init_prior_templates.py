import json
from pathlib import Path

import pandas as pd


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path) -> str:
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).strip()


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    prior_dir = data_dir / "priors"
    prior_dir.mkdir(parents=True, exist_ok=True)

    wt_path = data_dir / "wt.fasta"
    wt_seq = read_fasta(wt_path)
    if not wt_seq:
        raise ValueError(f"Empty WT FASTA: {wt_path}")

    pssm_path = prior_dir / "pssm.csv"
    if not pssm_path.exists():
        rows = []
        for i in range(1, len(wt_seq) + 1):
            row = {"pos": i}
            for aa in AA_ORDER:
                row[aa] = 0.0
            rows.append(row)
        pd.DataFrame(rows).to_csv(pssm_path, index=False)

    cov_path = prior_dir / "coevolution.csv"
    if not cov_path.exists():
        pd.DataFrame(columns=["pos_i", "pos_j", "score"]).to_csv(cov_path, index=False)

    res_struct_path = prior_dir / "residue_structure.csv"
    if not res_struct_path.exists():
        rows = []
        for i in range(1, len(wt_seq) + 1):
            rows.append(
                {
                    "pos": i,
                    "donor_dist": 0.0,
                    "acceptor_dist": 0.0,
                    "catalytic_centroid_dist": 0.0,
                    "asa": 0.0,
                    "res_hydrophobicity": 0.0,
                    "phi": 0.0,
                    "psi": 0.0,
                    "ss": "C",
                }
            )
        pd.DataFrame(rows).to_csv(res_struct_path, index=False)

    pocket_path = prior_dir / "pocket.csv"
    if not pocket_path.exists():
        pd.DataFrame(
            [{"pocket_volume": 0.0, "pocket_hydrophobicity": 0.0}]
        ).to_csv(pocket_path, index=False)

    ddg_path = prior_dir / "ddg.csv"
    if not ddg_path.exists():
        pd.DataFrame(columns=["variant", "mutation", "ddg"]).to_csv(ddg_path, index=False)

    cfg_path = data_dir / "prior_config.json"
    if not cfg_path.exists():
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"active_sites": []}, f, ensure_ascii=False, indent=2)

    print("Initialized prior templates in:", prior_dir)
    print("Config:", cfg_path)


if __name__ == "__main__":
    main()
