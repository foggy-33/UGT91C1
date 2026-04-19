import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen


UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
LINK_NEXT_RE = re.compile(r"<([^>]+)>;\s*rel=\"next\"")


def http_get(url: str, retries: int = 3, sleep_sec: float = 1.5) -> Tuple[bytes, Dict[str, str]]:
    last_err = None
    for _ in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "ugt-ml/1.0"})
            with urlopen(req, timeout=60) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.read(), headers
        except Exception as e:  # pragma: no cover - network variability
            last_err = e
            time.sleep(sleep_sec)
    raise RuntimeError(f"HTTP GET failed after retries: {url}; err={last_err}")


def parse_fasta(text: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header = line[1:].strip()
            seq = []
        else:
            seq.append(line)
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def fetch_tsv(query: str, max_records: int) -> List[Dict[str, str]]:
    fields = "accession,id,protein_name,organism_name,length,gene_names"
    url = (
        f"{UNIPROT_BASE}?query={quote(query)}&fields={fields}"
        f"&format=tsv&size=500"
    )
    rows: List[Dict[str, str]] = []
    while url and len(rows) < max_records:
        body, headers = http_get(url)
        text = body.decode("utf-8", errors="replace")
        reader = csv.DictReader(text.splitlines(), delimiter="\t")
        for r in reader:
            rows.append(r)
            if len(rows) >= max_records:
                break
        link = headers.get("link", "")
        m = LINK_NEXT_RE.search(link)
        url = m.group(1) if m else ""
    return rows


def fetch_fasta_for_accessions(accessions: List[str]) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    chunk_size = 200
    for i in range(0, len(accessions), chunk_size):
        chunk = accessions[i : i + chunk_size]
        query = " OR ".join([f"accession:{x}" for x in chunk])
        url = f"{UNIPROT_BASE}?query={quote(query)}&format=fasta&size=500"
        body, _ = http_get(url)
        text = body.decode("utf-8", errors="replace")
        records.extend(parse_fasta(text))
    return records


def dedup_by_sequence(records: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = {}
    for h, s in records:
        if s not in seen:
            seen[s] = h
    return [(h, s) for s, h in seen.items()]


def main():
    parser = argparse.ArgumentParser(description="Fetch UGT91C1 family homolog sequences from UniProt")
    parser.add_argument("--max-records", type=int, default=2000)
    parser.add_argument(
        "--query",
        type=str,
        default=(
            "(gene_exact:UGT91C1 OR protein_name:\"UDP-glycosyltransferase 91\" "
            "OR (family:\"UDP-glycosyltransferase\" AND protein_name:UGT91)) "
            "AND taxonomy_id:33090"
        ),
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "data" / "family"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = fetch_tsv(args.query, max_records=args.max_records)
    if not meta:
        raise RuntimeError("No family records fetched from UniProt. Please check network or query.")

    accessions = [r.get("Entry", "").strip() for r in meta if r.get("Entry")]
    accessions = [x for x in accessions if x]

    fasta_records = fetch_fasta_for_accessions(accessions)
    fasta_dedup = dedup_by_sequence(fasta_records)

    meta_path = out_dir / "ugt91_family_metadata.tsv"
    raw_fa_path = out_dir / "ugt91_family_raw.fasta"
    dedup_fa_path = out_dir / "ugt91_family_dedup.fasta"

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        if meta:
            w = csv.DictWriter(f, fieldnames=list(meta[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(meta)

    with open(raw_fa_path, "w", encoding="utf-8") as f:
        for h, s in fasta_records:
            f.write(f">{h}\n{s}\n")

    with open(dedup_fa_path, "w", encoding="utf-8") as f:
        for h, s in fasta_dedup:
            f.write(f">{h}\n{s}\n")

    print("Saved metadata:", meta_path)
    print("Saved raw fasta:", raw_fa_path, "records=", len(fasta_records))
    print("Saved dedup fasta:", dedup_fa_path, "records=", len(fasta_dedup))


if __name__ == "__main__":
    main()
