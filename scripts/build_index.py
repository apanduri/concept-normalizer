#!/usr/bin/env python3
"""Build a SapBERT+FAISS index for one target vocabulary.

Generic version of CP's backend/build_snomed_index.py — the mechanism (encode
each concept with SapBERT, L2-normalize, add to a FAISS inner-product index) is
the same; only the input source changes per target.

Input format: a CSV with at minimum `code` and `name` columns. Anything else
rides along as metadata ("extra") on each hit and is not used for retrieval.
That lets a BSO-AD build and a LOINC build share the same code without the
builder having to understand either vocabulary's schema.

Usage:

  python scripts/build_index.py --target snomed \\
      --from concepts.csv --out indexes/snomed \\
      [--model cambridgeltl/SapBERT-from-PubMedBERT-fulltext] \\
      [--batch-size 128]

The output directory ends up with two files:

  faiss_index.bin             the FAISS IndexFlatIP, ntotal == row count
  concepts_metadata.json      list of per-row metadata dicts, same order as the index
"""

from __future__ import annotations

import os
# macOS: HF tokenizers + FAISS OpenMP collide after fork, causing a
# silent segfault partway through a large build. Must be set before any
# transformers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True,
                   help="short name (snomed, loinc, omop, bso_ad) — "
                        "determines the output subdirectory default")
    p.add_argument("--from", dest="source", type=Path, required=True,
                   help="CSV with at least `code` and `name` columns")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory (will be created)")
    p.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                   help="HF model id for the biomedical encoder")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-rows", type=int, default=None,
                   help="limit for development; omit to index everything")
    args = p.parse_args()

    # Imports deferred so --help works on a machine without torch/faiss.
    # Import order matters on macOS: torch and faiss-cpu both bundle libomp,
    # and whichever one loads second gets a broken OpenMP. Torch tolerates
    # faiss's libomp but not vice-versa, so torch MUST import first.
    import torch
    from transformers import AutoModel, AutoTokenizer
    import numpy as np
    import faiss

    rows = _load_rows(args.source, max_rows=args.max_rows)
    if not rows:
        print(f"error: no rows loaded from {args.source}", file=sys.stderr)
        return 2
    print(f"[build_index] loaded {len(rows):,} rows from {args.source}")

    print(f"[build_index] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[build_index] device: {device}")

    # The CP paper uses the CLS token with L2-normalized output. We follow suit.
    # Inner-product on L2-normalized vectors is equivalent to cosine similarity.
    dim = model.config.hidden_size
    index = faiss.IndexFlatIP(dim)

    t0 = time.time()
    names = [r["name"] for r in rows]
    for start in range(0, len(names), args.batch_size):
        batch = names[start:start + args.batch_size]
        toks = tokenizer.batch_encode_plus(
            batch,
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            embs = model(**toks)[0][:, 0, :].cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embs = embs / norms
        index.add(embs)
        if (start // args.batch_size) % 20 == 0:
            done = min(start + args.batch_size, len(names))
            print(f"[build_index] {done:,}/{len(names):,} ({done/len(names):.0%}) "
                  f"elapsed {time.time()-t0:.0f}s")

    print(f"[build_index] indexed {index.ntotal:,} vectors in {time.time()-t0:.0f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out / "faiss_index.bin"))
    with (args.out / "concepts_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    with (args.out / "build_info.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "target": args.target,
            "source": str(args.source),
            "model": args.model,
            "rows": len(rows),
            "dim": dim,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, fh, indent=2)
    print(f"[build_index] wrote {args.out}")
    return 0


def _load_rows(path: Path, *, max_rows: int | None = None) -> list[dict]:
    """Read the CSV into dicts. Keeps every column so vocabulary-specific metadata
    (preferred_term, concept_class_id, parent_label, ...) rides along to the
    retriever as `extra`."""
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "name" not in reader.fieldnames \
                or "code" not in reader.fieldnames:
            raise SystemExit(
                f"{path}: CSV must have `code` and `name` columns; "
                f"got {reader.fieldnames!r}"
            )
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            code = (row.get("code") or "").strip()
            name = (row.get("name") or "").strip()
            if not code or not name:
                continue
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


if __name__ == "__main__":
    sys.exit(main())
