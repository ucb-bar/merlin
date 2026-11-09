#!/usr/bin/env python3
"""Propose op/form recipes to an explicit fresh output; never modify source corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.experiments.phase0.generation import _require_distinct_corpus_destinations
from merlin.experiments.phase0.op_cell_synth import synthesize_op_cells
from merlin.targetgen.corpus_spec import derive_binding
from merlin.targetgen.eligibility import capability_map_from_contract
from merlin.targetgen.opset_coverage_input import read_request
from merlin.targetgen.target_experiment import load_target_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-capsules", required=True, type=int)
    parser.add_argument("--max-output-elements", type=int)
    parser.add_argument("--defer", action="append", default=[])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("synthesis output must be a new file")
    doc, report, bounds = read_request(args.input)
    descriptor_bytes = args.descriptor.read_bytes()
    te = load_target_experiment(args.descriptor)
    _require_distinct_corpus_destinations(te, output_root=args.output, evidence_root=None)
    if te.target != doc["target"]:
        raise ValueError("descriptor and coverage target differ")
    if not isinstance(doc.get("datapath"), dict):
        raise ValueError("explicit datapath numeric policy required")
    binding = derive_binding(te, doc["datapath"], contract=doc["contract"], facts=doc["facts"])
    result = synthesize_op_cells(
        doc["target"],
        report=report,
        bounds=bounds,
        binding=binding,
        capability_map=capability_map_from_contract(doc["contract"]),
        max_capsules=args.max_capsules,
        max_output_elements=args.max_output_elements,
        skip_ops=args.defer,
    )
    if args.descriptor.read_bytes() != descriptor_bytes:
        raise ValueError("descriptor changed during synthesis")
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != report["input"]["sha256"]:
        raise ValueError("coverage input changed during synthesis")
    result["provenance"]["input"] = report["input"]
    result["provenance"]["descriptor"] = {
        "path": str(args.descriptor.resolve()),
        "sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
    }
    # Existing writer/run APIs own materialization, execution and hidden answers.
    with args.output.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
