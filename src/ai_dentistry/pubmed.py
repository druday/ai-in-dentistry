from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from Bio import Entrez, Medline


def configure_entrez(entrez_cfg: dict[str, Any]) -> None:
    email_env = entrez_cfg.get("contact_email_env", "NCBI_EMAIL")
    api_key_env = entrez_cfg.get("api_key_env", "NCBI_API_KEY")

    email = os.getenv(email_env)
    if not email:
        raise EnvironmentError(
            f"Missing required environment variable {email_env}. "
            "Set it in .env (or export it in shell), e.g. "
            "`set -a; source .env; set +a`."
        )

    Entrez.email = email
    api_key = os.getenv(api_key_env)
    if api_key:
        Entrez.api_key = api_key


def search_pmids(query: str, entrez_cfg: dict[str, Any]) -> list[str]:
    handle = Entrez.esearch(
        db=entrez_cfg.get("db", "pubmed"),
        term=query,
        retmax=entrez_cfg.get("max_records_per_query") or 0,
    )
    result = Entrez.read(handle)
    count = int(result["Count"])

    handle = Entrez.esearch(
        db=entrez_cfg.get("db", "pubmed"),
        term=query,
        retmax=count,
    )
    result = Entrez.read(handle)
    return list(result.get("IdList", []))


def fetch_medline_records(pmids: Iterable[str], entrez_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    batch_size = int(entrez_cfg.get("batch_size", 100))
    retmode = entrez_cfg.get("retmode", "text")
    rettype = entrez_cfg.get("rettype", "medline")
    db = entrez_cfg.get("db", "pubmed")
    records: list[dict[str, Any]] = []

    pmid_list = list(pmids)
    for index in range(0, len(pmid_list), batch_size):
        batch = pmid_list[index : index + batch_size]
        handle = Entrez.efetch(db=db, id=batch, rettype=rettype, retmode=retmode)
        batch_records = list(Medline.parse(handle))
        records.extend(batch_records)
        time.sleep(0.34)

    return records


def write_jsonl(records: Iterable[dict[str, Any]], out_path: str | Path) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def fetch_query_snapshot(
    query_id: str,
    query_text: str,
    entrez_cfg: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    configure_entrez(entrez_cfg)
    pmids = search_pmids(query_text, entrez_cfg)
    records = fetch_medline_records(pmids, entrez_cfg)

    out_path = Path(output_dir) / f"pubmed_records_{query_id}.jsonl"
    write_jsonl(records, out_path)
    return out_path
