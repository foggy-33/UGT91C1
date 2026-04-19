import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


class PriorFeatureBuilder:
    """Build evolutionary/structure/stability prior features from optional files."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.prior_dir = self.data_dir / "priors"
        self.config = self._load_config()

        self.pssm = self._load_pssm()
        self.cov = self._load_covariance()
        self.residue_struct = self._load_residue_structure()
        self.pocket = self._load_pocket()
        self.ddg_variant, self.ddg_mut = self._load_ddg()
        self.active_sites = set(int(x) for x in self.config.get("active_sites", []))

    @staticmethod
    def feature_names():
        return [
            "prior_mut_count",
            "evo_pssm_mut_mean",
            "evo_pssm_mut_min",
            "evo_pssm_mut_max",
            "evo_pssm_delta_mean",
            "evo_pssm_conservation_mean",
            "evo_cov_mutpair_mean",
            "evo_cov_mutpair_max",
            "evo_cov_to_active_mean",
            "geom_min_donor_dist",
            "geom_min_acceptor_dist",
            "geom_min_catalytic_centroid_dist",
            "pocket_volume",
            "pocket_hydrophobicity",
            "res_asa_mean",
            "res_hydrophobicity_mean",
            "res_phi_mean",
            "res_psi_mean",
            "res_ss_frac_H",
            "res_ss_frac_E",
            "res_ss_frac_C",
            "stab_ddg_total",
            "stab_ddg_max",
        ]

    def build_matrix(self, df: pd.DataFrame):
        feats = [self._row_features(r) for _, r in df.iterrows()]
        return np.asarray(feats, dtype=np.float32), self.feature_names()

    def add_ddg_columns(self, df: pd.DataFrame):
        ddg_total = []
        ddg_max = []
        for _, row in df.iterrows():
            total, m = self._ddg_stats(row)
            ddg_total.append(total)
            ddg_max.append(m)
        out = df.copy()
        out["ddg_total"] = ddg_total
        out["ddg_max"] = ddg_max
        return out

    def ddg_filter(self, df: pd.DataFrame, threshold: float, allow_unknown: bool = True):
        x = self.add_ddg_columns(df)
        if allow_unknown:
            mask = x["ddg_total"].isna() | (x["ddg_total"] <= threshold)
        else:
            mask = x["ddg_total"].notna() & (x["ddg_total"] <= threshold)
        return x[mask].reset_index(drop=True), x[~mask].reset_index(drop=True)

    def _load_config(self):
        cfg_path = self.data_dir / "prior_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"active_sites": []}

    def _load_pssm(self):
        path = self.prior_dir / "pssm.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if "pos" not in df.columns:
            return {}
        pssm = {}
        for _, row in df.iterrows():
            pos = int(row["pos"])
            scores = {}
            for aa in AA_ORDER:
                if aa in row.index:
                    scores[aa] = float(row[aa])
            pssm[pos] = scores
        return pssm

    def _load_covariance(self):
        path = self.prior_dir / "coevolution.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        need = {"pos_i", "pos_j", "score"}
        if not need.issubset(df.columns):
            return {}
        cov = {}
        for _, row in df.iterrows():
            i = int(row["pos_i"])
            j = int(row["pos_j"])
            s = float(row["score"])
            cov[(i, j)] = s
            cov[(j, i)] = s
        return cov

    def _load_residue_structure(self):
        path = self.prior_dir / "residue_structure.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if "pos" not in df.columns:
            return {}
        idx = {}
        for _, row in df.iterrows():
            pos = int(row["pos"])
            idx[pos] = row
        return idx

    def _load_pocket(self):
        path = self.prior_dir / "pocket.csv"
        if not path.exists():
            return {"pocket_volume": 0.0, "pocket_hydrophobicity": 0.0}
        df = pd.read_csv(path)
        if len(df) == 0:
            return {"pocket_volume": 0.0, "pocket_hydrophobicity": 0.0}
        row = df.iloc[0]
        return {
            "pocket_volume": float(row.get("pocket_volume", 0.0)),
            "pocket_hydrophobicity": float(row.get("pocket_hydrophobicity", 0.0)),
        }

    def _load_ddg(self):
        path = self.prior_dir / "ddg.csv"
        if not path.exists():
            return {}, {}
        df = pd.read_csv(path)
        by_variant = {}
        by_mut = {}
        if {"variant", "ddg"}.issubset(df.columns):
            for _, row in df.iterrows():
                by_variant[str(row["variant"]).strip()] = float(row["ddg"])
        if {"mutation", "ddg"}.issubset(df.columns):
            for _, row in df.iterrows():
                by_mut[str(row["mutation"]).strip()] = float(row["ddg"])
        return by_variant, by_mut

    def _parse_mutations(self, row):
        raw = ""
        if "mutation_list" in row and pd.notna(row["mutation_list"]):
            raw = str(row["mutation_list"]).strip()
        elif "variant" in row and pd.notna(row["variant"]):
            raw = str(row["variant"]).strip().replace("-", ";")
        if not raw or raw.upper() == "WT":
            return []

        muts = []
        for token in [x.strip() for x in raw.split(";") if x.strip()]:
            m = MUT_RE.match(token)
            if not m:
                continue
            muts.append(
                {
                    "token": token,
                    "wt": m.group(1),
                    "pos": int(m.group(2)),
                    "mut": m.group(3),
                }
            )
        return muts

    def _safe_mean(self, values):
        return float(np.mean(values)) if values else 0.0

    def _safe_min(self, values):
        return float(np.min(values)) if values else 0.0

    def _safe_max(self, values):
        return float(np.max(values)) if values else 0.0

    def _ddg_stats(self, row):
        variant = str(row.get("variant", "")).strip()
        if variant in self.ddg_variant:
            v = float(self.ddg_variant[variant])
            return v, v

        muts = self._parse_mutations(row)
        vals = [self.ddg_mut[m["token"]] for m in muts if m["token"] in self.ddg_mut]
        if not vals:
            return np.nan, np.nan
        return float(sum(vals)), float(max(vals))

    def _row_features(self, row):
        muts = self._parse_mutations(row)
        positions = [m["pos"] for m in muts]

        pssm_mut = []
        pssm_delta = []
        pssm_cons = []
        for m in muts:
            scores = self.pssm.get(m["pos"], {})
            if not scores:
                continue
            s_mut = scores.get(m["mut"], 0.0)
            s_wt = scores.get(m["wt"], 0.0)
            pssm_mut.append(s_mut)
            pssm_delta.append(s_mut - s_wt)
            pssm_cons.append(max(scores.values()))

        pair_cov = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                s = self.cov.get((positions[i], positions[j]))
                if s is not None:
                    pair_cov.append(float(s))

        cov_to_active = []
        if self.active_sites:
            for p in positions:
                vals = [self.cov[(p, a)] for a in self.active_sites if (p, a) in self.cov]
                if vals:
                    cov_to_active.append(float(max(vals)))

        donor = []
        acceptor = []
        centroid = []
        asa = []
        hydro = []
        phi = []
        psi = []
        ss = []
        for p in positions:
            rec = self.residue_struct.get(p)
            if rec is None:
                continue
            donor.append(float(rec.get("donor_dist", np.nan)))
            acceptor.append(float(rec.get("acceptor_dist", np.nan)))
            cat = rec.get("catalytic_centroid_dist", np.nan)
            if pd.isna(cat):
                d = rec.get("donor_dist", np.nan)
                a = rec.get("acceptor_dist", np.nan)
                cat = np.nanmin([d, a]) if (not pd.isna(d) or not pd.isna(a)) else np.nan
            centroid.append(float(cat))
            asa.append(float(rec.get("asa", np.nan)))
            hydro.append(float(rec.get("res_hydrophobicity", np.nan)))
            phi.append(float(rec.get("phi", np.nan)))
            psi.append(float(rec.get("psi", np.nan)))
            s = str(rec.get("ss", "C")).strip().upper()
            ss.append(s if s in {"H", "E", "C"} else "C")

        donor = [x for x in donor if not np.isnan(x)]
        acceptor = [x for x in acceptor if not np.isnan(x)]
        centroid = [x for x in centroid if not np.isnan(x)]
        asa = [x for x in asa if not np.isnan(x)]
        hydro = [x for x in hydro if not np.isnan(x)]
        phi = [x for x in phi if not np.isnan(x)]
        psi = [x for x in psi if not np.isnan(x)]

        ddg_total, ddg_max = self._ddg_stats(row)
        ddg_total = 0.0 if np.isnan(ddg_total) else ddg_total
        ddg_max = 0.0 if np.isnan(ddg_max) else ddg_max

        ss_h = ss.count("H") / len(ss) if ss else 0.0
        ss_e = ss.count("E") / len(ss) if ss else 0.0
        ss_c = ss.count("C") / len(ss) if ss else 0.0

        return [
            float(len(muts)),
            self._safe_mean(pssm_mut),
            self._safe_min(pssm_mut),
            self._safe_max(pssm_mut),
            self._safe_mean(pssm_delta),
            self._safe_mean(pssm_cons),
            self._safe_mean(pair_cov),
            self._safe_max(pair_cov),
            self._safe_mean(cov_to_active),
            self._safe_min(donor),
            self._safe_min(acceptor),
            self._safe_min(centroid),
            float(self.pocket.get("pocket_volume", 0.0)),
            float(self.pocket.get("pocket_hydrophobicity", 0.0)),
            self._safe_mean(asa),
            self._safe_mean(hydro),
            self._safe_mean(phi),
            self._safe_mean(psi),
            float(ss_h),
            float(ss_e),
            float(ss_c),
            float(ddg_total),
            float(ddg_max),
        ]
