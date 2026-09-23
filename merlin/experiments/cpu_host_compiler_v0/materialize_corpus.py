#!/usr/bin/env python3
"""Materialize the frozen generic CPU/RVV capsule descriptors under the canonical artifact root."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from merlin.common.artifacts import new_product
from merlin.common.paths import repo_root
from merlin.mining.corpus import materialize_definition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", default=str(
        repo_root() / "merlin/benchmarks/rvv_paper/development_corpus_v2.yaml"))
    args = parser.parse_args(argv)
    definition = Path(args.definition).resolve()
    definition_metadata = yaml.safe_load(definition.read_text(encoding="utf-8"))
    product = new_product(
        "rvv-development-corpus", version=int(definition_metadata["version"]),
        target="k1_cpu", sources=[str(definition)],
        notes="Generic holdout-safe compiler-development capsules; sealed split is grader-only.")
    lock = materialize_definition(definition, product.path)
    for relpath in ("public/train.jsonl", "public/validation.jsonl",
                    "sealed/heldout.jsonl", "corpus_lock.yaml"):
        product.add_artifact(relpath)
    product.write_manifest()
    print(yaml.safe_dump({"product": str(product.path), **lock}, sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
