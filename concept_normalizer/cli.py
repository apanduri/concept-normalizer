"""CLI: python -m concept_normalizer <command>

    normalize   normalize text (or a file of terms) against a target vocabulary
    coverage    how much of a source ontology exists in the target
    register    register a source ontology as OMOP custom concepts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from . import ontology as ontology_loader
from . import aliases as alias_mod
from .normalize import Status, normalize
from .registry import VocabularyRegistrar
from .target import OmopVocabulary

# Domains an exact name match may be auto-accepted into. Narrower than the set of
# domains a fact can be WRITTEN to: Device and Procedure matches from a
# behavioural or descriptive ontology are usually the wrong sense of the word, so
# they need a human before becoming clinical facts.
AUTO_ACCEPT_DOMAINS = frozenset({"Observation", "Condition", "Measurement"})

ROOT = Path(__file__).resolve().parent.parent


def _target(args: argparse.Namespace) -> OmopVocabulary:
    vocabs = None
    if args.target.upper() != "OMOP":
        # "SNOMED", "LOINC", "SNOMED,LOINC" — all live inside the OMOP vocabulary,
        # so selecting a target is a filter, not a different implementation.
        vocabs = tuple(v.strip() for v in args.target.split(",") if v.strip())
    return OmopVocabulary(args.vocab, vocabulary_ids=vocabs)


def _aliases(args: argparse.Namespace):
    if not getattr(args, "aliases", None):
        return None
    as_path = Path(args.aliases)
    table = alias_mod.load(as_path) if as_path.exists() else alias_mod.load_builtin(args.aliases)
    print(f"[aliases] {table}")
    return table


def cmd_normalize(args: argparse.Namespace) -> int:
    target = _target(args)
    table = _aliases(args)
    terms: list[str] = list(args.text or [])
    if args.terms_file:
        terms.extend(
            line.strip() for line in args.terms_file.read_text().splitlines() if line.strip()
        )
    if not terms:
        print("error: give one or more terms, or --terms-file", file=sys.stderr)
        return 2

    results = [normalize(t, target, aliases=table) for t in terms]
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(f"target: {target.name}   ({len(target.index):,} distinct names indexed)\n")
        for r in results:
            if r.status is Status.MAPPED:
                print(f"  MAPPED     {r.text!r}\n             -> {r.concept}"
                      f"  domain={r.concept.domain_id}")
            elif r.status is Status.AMBIGUOUS:
                print(f"  AMBIGUOUS  {r.text!r}  ({r.detail})")
                for c in r.candidates[:4]:
                    print(f"             ? {c.concept}  domain={c.concept.domain_id}")
            elif r.status is Status.NOT_IN_TARGET:
                print(f"  REVIEWED-NO {r.text!r}  ({r.detail})")
            else:
                print(f"  UNMAPPED   {r.text!r}  ({r.detail})")
    counts = Counter(r.status.value for r in results)
    print(f"\n{len(results)} terms: " + ", ".join(f"{v} {k}" for k, v in counts.most_common()))
    target.close()
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    concepts = ontology_loader.load(args.ontology)
    target = _target(args)
    print(f"[coverage] {len(concepts)} source concepts vs target {target.name}")

    rows = []
    for c in concepts:
        r = normalize(c.display_name, target)
        rows.append({
            "subtree": c.entity_type,
            "name": c.concept_name,
            "source_id": c.source_id or "",
            "parent": c.parent_name or "",
            "depth": c.depth if c.depth is not None else "",
            "status": r.status.value,
            "concept_id": r.concept.concept_id if r.concept else "",
            "concept_name": r.concept.name if r.concept else "",
            "domain_id": r.concept.domain_id if r.concept else "",
            "n_candidates": len(r.candidates),
            "detail": r.detail or "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    counts = Counter(r["status"] for r in rows)
    print()
    for status in ("mapped", "ambiguous", "unmapped"):
        n = counts.get(status, 0)
        print(f"  {status:<12} {n:>5}  ({n / total:>5.1%})")
    print(f"\n  full results: {args.out}")
    print("  NOTE: exact-name matching only — a floor on coverage. Synonyms and "
          "alternate\n        phrasings are not checked, so true coverage is higher.")
    target.close()
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    concepts = ontology_loader.load(args.ontology)
    print(f"[register] {len(concepts)} concepts from {args.ontology.name}")

    mappings: dict[str, tuple[int, str, str]] = {}
    skipped_domain = 0
    if args.mappings:
        with args.mappings.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "mapped" or not row.get("concept_id"):
                    continue
                domain = row.get("domain_id") or ""
                if domain not in AUTO_ACCEPT_DOMAINS:
                    # Exact name match, suspect meaning. OMOP's "Treadmill"
                    # (4004884) is a Device — a physical object — while a
                    # behavioural ontology means "the patient uses one". Accepting
                    # it would assert device exposures no note described. Held for
                    # review rather than auto-accepted.
                    skipped_domain += 1
                    continue
                mappings[f"{row['subtree']}|{row['name']}"] = (
                    int(row["concept_id"]), domain, "coverage:mapped"
                )
        print(f"[register] {len(mappings)} mappings applied")
        if skipped_domain:
            print(f"[register] {skipped_domain} exact matches held for review "
                  f"(matched outside {sorted(AUTO_ACCEPT_DOMAINS)})")

    with VocabularyRegistrar(args.registry, vocabulary_id=args.vocabulary_id) as reg:
        report = reg.register(concepts, mappings=mappings, commit=args.commit,
                              now=args.now or "")

    print(f"\nnew: {len(report.inserted)}   existing: {len(report.unchanged)}   "
          f"ancestor rows: {report.ancestor_rows}   maps-to: {report.maps_to_rows}")
    print("domains:      " + ", ".join(f"{k}={v}" for k, v in
                                       sorted(report.domain_counts.items(), key=lambda kv: -kv[1])))
    print("decided by:   " + ", ".join(f"{k}={v}" for k, v in
                                       sorted(report.domain_source_counts.items(), key=lambda kv: -kv[1])))
    print("\n[register] DRY RUN — nothing written. Re-run with --commit."
          if report.dry_run else f"\n[register] committed to {args.registry}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m concept_normalizer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--vocab", type=Path, required=True,
                        help="OMOP concept.db backing the target")
        sp.add_argument("--aliases",
                        help="reviewed alias table: a built-in name (e.g. 'acts') or a "
                             "path to a CSV. Consulted before searching.")
        sp.add_argument("--target", default="OMOP",
                        help="OMOP (standard concepts) or a vocabulary filter such "
                             "as SNOMED, LOINC, 'SNOMED,LOINC'")

    n = sub.add_parser("normalize", help="normalize terms against a target")
    common(n)
    n.add_argument("text", nargs="*", help="term(s) to normalize")
    n.add_argument("--terms-file", type=Path, help="one term per line")
    n.add_argument("--json", action="store_true")
    n.set_defaults(func=cmd_normalize)

    c = sub.add_parser("coverage", help="how much of an ontology exists in the target")
    common(c)
    c.add_argument("--ontology", type=Path, required=True)
    c.add_argument("--out", type=Path, default=ROOT / "build" / "coverage.csv")
    c.set_defaults(func=cmd_coverage)

    r = sub.add_parser("register", help="register an ontology as OMOP custom concepts")
    r.add_argument("--ontology", type=Path, required=True)
    r.add_argument("--registry", type=Path, default=ROOT / "build" / "custom_vocab.db")
    r.add_argument("--vocabulary-id", default="CUSTOM")
    r.add_argument("--mappings", type=Path,
                   help="coverage CSV supplying reviewed 'Maps to' targets")
    r.add_argument("--now", default="")
    r.add_argument("--commit", action="store_true")
    r.set_defaults(func=cmd_register)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
