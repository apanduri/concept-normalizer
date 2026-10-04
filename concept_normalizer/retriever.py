"""Semantic retrieval — a target vocabulary's concepts, indexed and queryable by name.

Ported from CP's snomed_mapper.py (Abdulnazar et al. 2023 — SapBERT + FAISS +
Levenshtein) and generalized so the target is a runtime argument, not a hardcoded
vocabulary. One `Retriever` wraps one target's embedding index; a target is any
concept table on disk (SNOMED, LOINC, OMOP, BSO-AD, a flat list), each produced
by scripts/build_index.py as a (faiss_index.bin, concepts_metadata.json) pair.

Why this exists even though the normalizer package already has an exact-name
matcher:

  * free text ("Transgender and Gender Nonconforming Identification in EHRs")
    almost never matches a concept name exactly; the exact-name matcher returns
    nothing. SapBERT embeddings put semantically related phrases next to each
    other, so retrieval still finds "Gender identity finding".
  * CP's existing SNOMED pipeline uses this mechanism successfully. Generalizing
    it means other targets inherit the behaviour rather than growing their own
    matchers that drift.

Deliberately kept separate from the alias layer: a reviewed decision still beats
retrieval, so an AliasTable short-circuits before this is called.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Published default. Overridable per-target if a downstream project has fine-tuned
# a different biomedical encoder it would rather use.
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# How many FAISS candidates to pull before syntactic re-ranking. Ten is CP's
# value; it is a tradeoff between recall (more is better) and the risk that the
# syntactic re-ranker picks a stylistically-similar but semantically-wrong hit.
TOP_K = 10

# Short English function words that carry no clinical meaning. Dropping them
# before embedding keeps "Type 2 Diabetes" and "the Type 2 Diabetes" from
# producing different vectors.
_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "and", "or", "with", "by", "as", "is", "are", "was", "be",
})


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation and stop words. Deterministic by design."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return " ".join(tokens).strip()


@dataclass(slots=True, frozen=True)
class RetrievedConcept:
    """One hit from the retriever.

    `semantic_score` is the FAISS cosine similarity against the query embedding
    (higher is better; the index is L2-normalized so this is a true cosine).
    `syntactic_score` is a RapidFuzz partial ratio against the original query
    text (0-100). The two are kept separate rather than merged into one score
    because the caller — usually an LLM selector agent — benefits from seeing
    both signals.
    """

    code: str                   # the concept id in its vocabulary (SNOMED id, LOINC code, OMOP id, etc.)
    name: str
    semantic_score: float
    syntactic_score: float = 0.0
    extra: dict = None          # vocabulary-specific metadata the index chose to carry


class Retriever:
    """Semantic + syntactic retrieval over one target vocabulary.

    One instance per target. Loads SapBERT once, loads the FAISS index once, and
    is safe to reuse across many queries — which is why the module exposes a
    `get_retriever(target)` singleton helper below.
    """

    def __init__(self, index_dir: Path, model_name: str = SAPBERT_MODEL):
        self.index_dir = Path(index_dir)
        self.model_name = model_name

        faiss_path = self.index_dir / "faiss_index.bin"
        meta_path = self.index_dir / "concepts_metadata.json"
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. "
                f"Build it first with: python -m concept_normalizer build-index "
                f"--target <name> --from <concepts.csv> --out {self.index_dir}"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Index metadata not found at {meta_path}. The index and metadata "
                f"must be built together."
            )

        # Heavy imports kept inside __init__ so the rest of the package stays
        # importable without torch/faiss installed (the exact-name path and
        # alias tables both work fine without them).
        import faiss
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._np = np
        self._torch = torch

        log.info("retriever: loading SapBERT model %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        log.info("retriever: loading FAISS index at %s", faiss_path)
        self.index = faiss.read_index(str(faiss_path))

        with meta_path.open(encoding="utf-8") as fh:
            self.metadata = json.load(fh)

        log.info("retriever: ready — %d vectors indexed", self.index.ntotal)

    def _embed(self, text: str):
        """SapBERT CLS token, L2-normalized float32. The vector is a 1xD matrix
        because that's what FAISS.search expects (one row per query)."""
        toks = self.tokenizer.batch_encode_plus(
            [text],
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with self._torch.no_grad():
            cls = self.model(**toks)[0][0, 0, :].cpu().numpy().astype(self._np.float32)

        norm = self._np.linalg.norm(cls)
        if norm > 0:
            cls = cls / norm
        return cls.reshape(1, -1)

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedConcept]:
        """Return top-K candidates, re-ranked by syntactic similarity to `query`.

        Returns [] on an empty query or when preprocessing strips it to nothing,
        rather than raising. Downstream code already handles 'no candidates'; a
        raised exception on an obviously-empty input would be a worse interface.
        """
        if not query or not query.strip():
            return []
        processed = preprocess(query)
        if not processed:
            return []

        from rapidfuzz import fuzz

        vec = self._embed(processed)
        scores, indices = self.index.search(vec, top_k)

        hits: list[RetrievedConcept] = []
        query_lower = query.lower()
        for sem_score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            # The metadata file carries at minimum `code` and `name`; anything
            # else — SNOMED preferred_term flags, LOINC class, etc. — rides on
            # `extra`. That keeps the dataclass stable across targets.
            name = m.get("name") or m.get("preferred_term") or ""
            code = str(m.get("code") or m.get("concept_id") or m.get("snomed_id") or "")
            if not code or not name:
                continue
            hits.append(RetrievedConcept(
                code=code,
                name=name,
                semantic_score=float(sem_score),
                syntactic_score=float(fuzz.partial_ratio(query_lower, name.lower())),
                extra={k: v for k, v in m.items() if k not in ("code", "name")},
            ))

        # Syntactic re-rank. The CP paper's finding is that FAISS's top hit is
        # often not the best lexical match, so a RapidFuzz partial-ratio pass
        # catches cases where the vector got the right neighbourhood but the
        # top hit is a nearby wrong concept.
        hits.sort(key=lambda h: h.syntactic_score, reverse=True)
        return hits


# ---------------------------------------------------------------------------
# Per-target singletons
# ---------------------------------------------------------------------------

# Loading SapBERT and a FAISS index takes several seconds and a few hundred MB.
# Downstream code calls get_retriever(target) once per request, so caching is
# worth doing — but keyed on the index path (not the target name), because a
# project could legitimately want two indexes for the same vocabulary.
_RETRIEVERS: dict[Path, Retriever] = {}


def get_retriever(index_dir: Path, model_name: str = SAPBERT_MODEL) -> Retriever:
    index_dir = Path(index_dir).resolve()
    key = index_dir
    if key not in _RETRIEVERS:
        _RETRIEVERS[key] = Retriever(index_dir, model_name=model_name)
    return _RETRIEVERS[key]


def resolve_index_dir(target: str) -> Path:
    """Where this target's index lives on disk.

    Resolution order:
      1. the explicit env var CONCEPT_NORMALIZER_INDEX_DIR_<TARGET>
      2. the generic CONCEPT_NORMALIZER_INDEX_ROOT / <target>
      3. ./indexes/<target> under the current working directory

    No hardcoded paths and no defaults that only work on one developer's machine.
    """
    env_specific = os.getenv(f"CONCEPT_NORMALIZER_INDEX_DIR_{target.upper()}")
    if env_specific:
        return Path(env_specific)
    root = os.getenv("CONCEPT_NORMALIZER_INDEX_ROOT")
    if root:
        return Path(root) / target.lower()
    return Path.cwd() / "indexes" / target.lower()
