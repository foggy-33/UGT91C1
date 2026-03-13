import argparse
from pathlib import Path


def _feature_name(feature) -> str:
    q = feature.qualifiers
    for k in ("label", "gene", "product", "note"):
        if k in q:
            v = q[k]
            if isinstance(v, list) and v:
                return str(v[0])
            return str(v)
    return ""


def _pick_feature(record, key: str):
    key_lower = key.lower()
    candidates = []
    for feature in record.features:
        if feature.type != "CDS":
            continue
        name = _feature_name(feature)
        if key_lower in name.lower():
            candidates.append((feature, name))
    if not candidates:
        return None, None
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SnapGene .dna file to FASTA (DNA or protein)."
    )
    parser.add_argument("input", help="Path to SnapGene .dna file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output FASTA path (default: same name with .fasta)",
    )
    parser.add_argument(
        "--protein",
        action="store_true",
        help="Translate DNA CDS to protein FASTA (stop at first stop codon)",
    )
    parser.add_argument(
        "--feature",
        default=None,
        help="Extract a specific CDS feature by keyword (e.g. UGT91C1)",
    )
    args = parser.parse_args()

    try:
        from Bio import SeqIO
    except Exception as exc:
        raise SystemExit(
            "Biopython is required. Install with: pip install biopython\n"
            f"Import error: {exc}"
        )

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_suffix(".fasta")

    record = SeqIO.read(str(in_path), "snapgene")

    if args.feature:
        feature, feature_name = _pick_feature(record, args.feature)
        if feature is None:
            raise SystemExit(f"No CDS feature matched keyword: {args.feature}")
        seq = feature.extract(record.seq)
        record.id = feature_name or args.feature
    else:
        seq = record.seq

    if args.protein:
        seq = seq.translate(to_stop=True)

    record.seq = seq
    record.description = ""

    SeqIO.write(record, str(out_path), "fasta")
    print(f"Saved: {out_path}")
    print(f"Length: {len(seq)}")


if __name__ == "__main__":
    main()
