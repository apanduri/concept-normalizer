"""CLI: python -m concept_normalizer <command>

    normalize   normalize text (or a file of terms) against a target vocabulary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from . import aliases as alias_mod
from .normalize import Status, normalize
from .target import OmopVocabulary

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


def cmd_agentic(args):
    """Thin CLI entry point around the two-agent pipeline.

    Resolves the shipped profile by name, lets --index-dir override the default
    on-disk index path, and prints a per-input report with the chosen code and
    the all_candidates audit trail — same shape of information CP's existing
    SNOMED pipeline prints.
    """
    from .agentic import normalize as agentic_normalize
    from .profiles import by_name as profile_by_name
    from .retriever import Retriever

    try:
        profile = profile_by_name(args.profile)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.index_dir:
        profile = profile.__class__(
            name=profile.name, style_hint=profile.style_hint,
            purpose=profile.purpose, examples=profile.examples,
            index_dir=str(args.index_dir),
        )
    retriever = Retriever(Path(profile.index_dir))

    for text in args.text:
        result = agentic_normalize(text, profile, retriever=retriever)
        if result.code is None:
            print(f"UNMAPPED  {text!r}  ({result.chosen_via})")
            continue
        print(f"MAPPED    {text!r}")
        print(f"          -> {result.code} {result.name!r}  "
              f"via={result.chosen_via}  candidate={result.candidate_used!r}")
        print(f"          semantic={result.semantic_score:.4f}  "
              f"syntactic={result.syntactic_score:.0f}")
    return 0


def cmd_tree(args):
    """Thin CLI entry around the tree-navigation engine.

    Loads an ontology JSON, builds a TreeTargetProfile, and prints one hit per
    input phrase. --entity-type is repeatable, matching how a project might
    narrow BSO-AD to just the SDOH branches without the Dementia one.
    """
    from .tree_ontology import TreeTargetProfile, normalize_tree

    only = tuple(args.entity_type) if args.entity_type else None
    target = TreeTargetProfile(
        name=args.name or args.ontology.stem,
        ontology_path=args.ontology,
        only_entity_types=only,
    )
    for text in args.text:
        result = normalize_tree(text, target)
        if result.is_novel:
            print(f"NOVEL     {text!r}  ({result.raw_response!r})")
        else:
            print(f"MAPPED    {text!r}")
            print(f"          -> {result.label} ({result.entity_type})")
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

    a = sub.add_parser("agentic",
                       help="normalize via the two-agent LLM pipeline (CP-ported)")
    a.add_argument("text", nargs="+", help="raw input phrase(s)")
    a.add_argument("--profile", required=True,
                   help="shipped profile name: snomed, loinc, omop, bso_ad")
    a.add_argument("--index-dir", type=Path,
                   help="override the profile's index_dir")
    a.set_defaults(func=cmd_agentic)

    t = sub.add_parser("tree",
                       help="normalize via tree navigation (BSO-AD-style ontology)")
    t.add_argument("text", nargs="+", help="raw input phrase(s)")
    t.add_argument("--ontology", type=Path, required=True,
                   help="BSO-AD-shaped concepts.json")
    t.add_argument("--name", default=None,
                   help="display name for this ontology (used in the prompt)")
    t.add_argument("--entity-type", action="append",
                   help="restrict to one or more entity_types (default: all)")
    t.set_defaults(func=cmd_tree)

    n = sub.add_parser("normalize", help="normalize terms against a target")
    common(n)
    n.add_argument("text", nargs="*", help="term(s) to normalize")
    n.add_argument("--terms-file", type=Path, help="one term per line")
    n.add_argument("--json", action="store_true")
    n.set_defaults(func=cmd_normalize)

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
