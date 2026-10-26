#!/usr/bin/env python3
"""Generate the iteration coverage table from the 193 corpus and bounded suite.

Run with PYTHONPATH=src:packages/merlin-experiments/src in a torch environment:
  python build_tools/scripts/make_core_aten_coverage.py --corpus-193 PATH --bounded PATH --facts PATH \
    --target TARGET --descriptor PATH --host-capabilities PATH --write

Without --write, print the missing overload/dtype/case list. The canonical
admitted overloads define the missing set; bounded cases fill additional dtypes.
No dtype conversion, shape synthesis or argument rewriting is performed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml
from merlin_experiments.corpus.publication import _excluded_cores, operand_extents

from merlin.common.paths import artifacts_dir, repo_root
from merlin.targetgen.core_aten_capture import loader_source
from merlin.targetgen.functional_truth import derive
from merlin.targetgen.functional_truth_admission import admit, select_admission
from merlin.targetgen.functional_truth_types import Case

GENERATOR = "build_tools/scripts/make_core_aten_coverage.py"
WORKLOAD = Path("examples/workloads/core_aten_coverage/loader.py")
BEGIN = "# BEGIN GENERATED\n"
END = "# END GENERATED\n"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensors(value):
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            yield value
        else:
            for child in value.values():
                yield from tensors(child)
    elif isinstance(value, list):
        for child in value:
            yield from tensors(child)


def case_dtype(case):
    assigned = case.get("partition_assignment", {}).get("dtype")
    if assigned:
        return assigned
    first = next(tensors(case["arguments"]), None)
    if first:
        return first["dtype"]
    # Factories carry their dtype in kwargs, and scalar_tensor defaults to f32.
    return case["arguments"]["kwargs"].get("dtype", {}).get("value", "float32")


def capsule_document(overload, operands):
    """Keep *all* tensor operands, including weights and auxiliary tensors.

    The neutral spelling retains the complete operand multiset even for backward
    calls with several weights; publication's contraction normalizer must not
    discard an operand because its schema role is unfamiliar.
    """
    return {
        "operation": {"op": "core_aten_coverage", "attributes": {"overload": overload}},
        "inputs": [
            {"name": f"operand_{index}", "role": "operand", "shape": item["shape"] or [1]}
            for index, item in enumerate(operands)
            if all(item["shape"])  # Empty tensors cannot equal a held-out layer's positive extents.
        ],
    }


def excluded_cores(descriptor):
    from merlin.targetgen.composition_corpus import held_out_shapes

    root = repo_root()
    descriptor = yaml.safe_load(Path(descriptor).read_bytes())
    experiment = SimpleNamespace(source_root=root, **descriptor["answer_surfaces"])
    cores = _excluded_cores(experiment)
    manifest = root / "merlin/contract/capsules/MANIFEST.yaml"
    for shape in held_out_shapes(manifest):
        if len(shape) == 3:
            m, k, n = shape
            document = capsule_document("matmul", [{"shape": [m, k]}, {"shape": [k, n]}])
            cores.append(Counter(operand_extents(document)))
    return cores


def held_out(document, excluded):
    extents = operand_extents(document)
    if extents is None:
        raise ValueError("cannot identify workload operands against held-out capsules")
    return any(not core - Counter(extents) for core in excluded)


def boundary_safe(case, excluded):
    operands = list(tensors(case["arguments"]))
    # No inputs is legitimate for a factory. Empty tensors have no layer identity.
    operands = [item for item in operands if all(item["shape"])]
    if not operands:
        return True
    documents = [capsule_document(case["overload"], operands)]
    documents.extend(capsule_document(case["overload"], [item]) for item in operands)
    return not any(held_out(document, excluded) for document in documents)


def workload_census(descriptor):
    """Export the selected roster profiles into Core ATen when captures are absent."""
    import torch

    roster_path = Path(descriptor).parent / "phase0-workloads.yaml"
    applications = yaml.safe_load(roster_path.read_bytes())["applications"]
    output = {}
    scratch = artifacts_dir() / "core-aten-coverage"
    scratch.mkdir(parents=True, exist_ok=True)
    for label, entry in sorted(applications.items()):
        if label.startswith("core_aten_coverage_"):
            continue
        source = (roster_path.parent / entry["loader"]).resolve()
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            staged = Path(directory) / "loader.py"
            shutil.copyfile(source, staged)
            if "profile" in entry:
                staged.with_name("profile.json").write_text(json.dumps(entry["profile"]))
            spec = importlib.util.spec_from_file_location(label, staged)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model, inputs = module.get_model_and_inputs()
            with torch.no_grad():
                graph = torch.export.export(model, inputs).run_decompositions().graph
            output[label] = {
                "loader_sha256": sha(source),
                "profile": entry.get("profile"),
                "ops": sorted(
                    {str(node.target) for node in graph.nodes if isinstance(node.target, torch._ops.OpOverload)}
                ),
            }
    return output


def admission(facts, *, target, descriptor, host_capabilities):
    from merlin.targetgen.target_registry import observed_contract

    target_root = Path(descriptor).parent
    contract_path = target_root / "contracts/target_contract.yaml"
    with observed_contract(target, yaml.safe_load(contract_path.read_bytes()), source_path=contract_path):
        return select_admission(
            target,
            rtl_facts=facts,
            software_spec=target_root / "software-spec.yaml",
            host_capabilities=host_capabilities,
        )


def decision(case, selected):
    source = Case(case, "phase0")
    return admit(source, case, derive(source).family, selected)


def select_rows(canonical, bounded, selected, census, excluded):
    canonical_cases = {case["overload"]: case for case in canonical["cases"]}
    admitted = {op for op, case in canonical_cases.items() if decision(case, selected)["status"] == "admitted"}
    covered = {op for entry in census.values() for op in entry["ops"]}
    missing = sorted(admitted - covered)
    candidates = defaultdict(list)
    for case in bounded["selected_cases"]:
        if case["overload"] in missing:
            candidates[case["overload"], case_dtype(case)].append(case)
    rows = []
    for op in missing:
        canonical_case = canonical_cases[op]
        dtype = case_dtype(canonical_case)
        if boundary_safe(canonical_case, excluded):
            rows.append(table_row("corpus_193", canonical_case, decision(canonical_case, selected)))
        for (candidate_op, candidate_dtype), cases in sorted(candidates.items()):
            if candidate_op != op or (candidate_dtype == dtype and boundary_safe(canonical_case, excluded)):
                continue
            ordered = sorted(
                cases,
                key=lambda c: (
                    sum(math.prod(t["shape"]) for t in tensors(c["arguments"])),
                    c.get("case_id", c["overload"]),
                ),
            )
            for case in ordered:
                screened = decision(case, selected)
                if screened["status"] == "admitted" and boundary_safe(case, excluded):
                    rows.append(table_row("bounded", case, screened))
                    break
        if not any(row["overload"] == op and row["dtype"] == dtype for row in rows):
            raise ValueError(f"no held-out-safe admitted case for {op}/{dtype}")
    return missing, sorted(rows, key=lambda row: (row["group"], row["overload"], row["dtype"]))


def table_row(source, case, screened):
    return {
        "source": source,
        "case_id": case.get("case_id") or case["overload"],
        "overload": case["overload"],
        "dtype": case_dtype(case),
        "group": screened["observed_signature"]["family"],
        "arguments": case["arguments"],
    }


def generated_section(rows, census, pins, admission_pins):
    # Reuse the corpus capture's standalone decoder verbatim. Staged loaders must
    # work in a capture environment with torch alone, without an installed Merlin.
    sample = {"overload": rows[0]["overload"], "arguments": rows[0]["arguments"], "expected": {"kind": "tensor"}}
    source = loader_source(sample)
    tree = ast.parse(source)
    helpers = "\n\n".join(
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_decode", "_op"}
    )
    lines = [BEGIN.rstrip(), f"# Generator: {GENERATOR}"]
    lines.extend(f"# {key} sha256: {value}" for key, value in sorted(pins.items()))
    for name, value in (("SUITE_SHA256", pins), ("ADMISSION_SHA256", admission_pins), ("CENSUS", census)):
        lines.append(f"{name} = json.loads({json.dumps(value, sort_keys=True, separators=(',', ':'))!r})")
    lines.append("ROWS = [")
    lines.extend(f"    json.loads({json.dumps(row, sort_keys=True, separators=(',', ':'))!r})," for row in rows)
    lines.append("]")
    return "\n".join(lines) + "\n\n" + helpers + "\n" + END


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-193", type=Path, required=True)
    parser.add_argument("--bounded", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--target", required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--host-capabilities", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {"corpus_193": args.corpus_193, "bounded": args.bounded}
    suites = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
    census = workload_census(args.descriptor)
    selected = admission(
        args.facts, target=args.target, descriptor=args.descriptor, host_capabilities=args.host_capabilities
    )
    missing, rows = select_rows(
        suites["corpus_193"], suites["bounded"], selected, census, excluded_cores(args.descriptor)
    )
    print(
        json.dumps(
            {
                "missing": missing,
                "rows": [{key: value for key, value in row.items() if key != "arguments"} for row in rows],
            },
            indent=2,
        )
    )
    if args.write:
        path = repo_root() / WORKLOAD
        before, rest = path.read_text().split(BEGIN)
        _, after = rest.split(END)
        admission_pins = {
            "facts": sha(args.facts),
            "software_spec": sha(args.descriptor.parent / "software-spec.yaml"),
            "host_capabilities": sha(args.host_capabilities),
            "contract": sha(args.descriptor.parent / "contracts/target_contract.yaml"),
        }
        path.write_text(
            before
            + generated_section(rows, census, {key: sha(path) for key, path in paths.items()}, admission_pins)
            + after
        )


if __name__ == "__main__":
    main()
