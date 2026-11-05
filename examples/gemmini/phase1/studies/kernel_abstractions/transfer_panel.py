"""Operator-only, frozen one-axis transfer panel from canonical public sources.

This creates source contracts and independent mathematical fixtures, never a
compiler or an author input. Ordinary placement admission is recomputed only
when an exported Phase 0 evidence root is explicitly supplied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from merlin.targetgen.component_program import render
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

SCHEMA = "merlin.study.transfer_panel.v1"
FIXTURES = "merlin.study.transfer_fixtures.v1"
GEMM_ANCHOR = {"M": 16, "K": 2048, "N": 2048}
QK_ANCHOR = {"query_heads": 32, "kv_heads": 4, "D": 64, "query": 16, "key": 16}
OPERATOR_POLICY = {
    "operator_only": True, "author_visible": False, "allow_author_mount": False,
    "helper_imports_allowed": False, "author_feedback_allowed": False,
    "evaluation": "frozen compiler snapshots only; never select cases from their measured results",
    "scope": "complete declared invocation and every live output; no full-model equivalence",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def pin(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("a pinned panel input must be an ordinary file")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def specifications():
    """34 preregistered OFAT cells, with each anchor represented exactly once."""
    cases = []
    for family, anchor, axes, stress in (
        ("gemm", GEMM_ANCHOR,
         {"M": (8, 16, 32, 48, 64), "K": (1024, 1536, 2048, 2560, 3072, 4096),
          "N": (1024, 1536, 2048, 2560, 3072, 4096)},
         {"M": (15, 17), "K": (2047, 2049), "N": (2047, 2049)}),
        ("qk", QK_ANCHOR, {"query": (8, 16, 32, 48, 64), "key": (8, 16, 32, 48, 64)},
         {"query": (15, 17), "key": (15, 17)}),
    ):
        rows = [("anchor", "anchor", dict(anchor))]
        rows += [("scaling", axis, {**anchor, axis: value}) for axis, values in axes.items()
                 for value in values if value != anchor[axis]]
        rows += [("stress", axis, {**anchor, axis: value}) for axis, values in stress.items() for value in values]
        for stratum, axis, geometry in rows:
            name = (f"gemm_m{geometry['M']}k{geometry['K']}n{geometry['N']}" if family == "gemm"
                    else f"qk_q32kv4d64_s{geometry['query']}l{geometry['key']}")
            exposure = []
            if stratum == "anchor":
                exposure.append("public_xs_exact_contract")
            if family == "gemm" and geometry == {"M": 32, "K": 2048, "N": 2048}:
                exposure.append("fresh6_public_exact_contract")
            if family == "gemm" and geometry == {"M": 16, "K": 2048, "N": 2560}:
                exposure.append("public_xs_geometry_different_i8_readout")
            cases.append(dict(case="transfer_" + name, family=family, split="heldout", stratum=stratum,
                              axis=axis, geometry=geometry, exposure=exposure))
    return cases


def _template(path):
    path = Path(path).resolve(strict=True)
    path = path / "capsule.yaml" if path.is_dir() else path
    cap = yaml.safe_load(path.read_text())
    source = (path.parent / cap["interface_mlir"]).resolve(strict=True)
    if source.parent != path.parent or cap.get("label") != "public":
        raise ValueError("panel anchor requires an explicit public local source")
    return cap, source.read_text(), {"capsule": pin(path), "source": pin(source)}


def _gemm_template(cap, text):
    cb = parse_interface_mlir(text)
    commands = cb["commands"]
    if [row["opcode"] for row in commands] != ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"]:
        raise ValueError("GEMM anchor requires one complete resident contraction")
    pack, multiply, commit, evict = [row["operands"] for row in commands]
    if (multiply["rhs"] != pack["dst"] or commit["src"] != multiply["dst"]
        or evict["handle"] != pack["dst"] or len(cb["tensors"]) != 2
        or commands[2].get("attributes") != {"epilogue": [], "output_dtype": "i32"}
        or cb["tensors"][multiply["lhs"]] != {"shape": [16, 2048], "dtype": "i8", "role": "input"}
        or cb["tensors"][pack["src"]] != {"shape": [2048, 2048], "dtype": "i8", "role": "weight"}
        or cap["operation"]["op"] != "matmul" or cap.get("stimulus_range") != [-4, 3]
        or cap.get("numeric_policy") != {"compare": "exact_int", "dtype": "i32"}
        or cap["inputs"] != [{"name": name, **row} for name, row in cb["tensors"].items()]
        or cap["operation"]["attributes"] != dict(lhs=multiply["lhs"], weight=pack["src"],
            out=commit["dst"], epilogue=[], output_dtype="i32")):
        raise ValueError("GEMM anchor differs from the selected raw-i32 source contract")
    if emit_interface_mlir(cb) != text:
        raise ValueError("GEMM anchor source does not round-trip exactly")
    return cb


def _qk_template(cap, text):
    if (cap["operation"]["op"] != "component_program" or cap.get("stimulus_range") != [-4, 3]
        or cap.get("numeric_policy") != {"compare": "exact_int", "dtype": "i32"}):
        raise ValueError("QK anchor requires the original bounded component source")
    program = copy.deepcopy(cap["operation"]["attributes"]["program"])
    typed, source = render(program, operand_dtype="i8", accumulator_dtype="i32")
    expected = [{"name": f"s{h}", "op": "matmul", "inputs": [f"Q{h}", f"KT{h // 8}"]} for h in range(32)]
    inputs = [{"name": f"KT{g}", "role": "input", "shape": [64, 16], "dtype": "operand"} for g in range(4)]
    inputs += [{"name": f"Q{h}", "role": "input", "shape": [16, 64], "dtype": "operand"} for h in range(32)]
    outputs = [{"name": f"S{h}", "value": f"s{h}"} for h in range(32)]
    if (program["nodes"] != expected or program["inputs"] != inputs or program["outputs"] != outputs
        or source != text or typed != cap["component_program"] or typed["inputs"] != cap["inputs"]):
        raise ValueError("QK anchor lost original GQA sharing, output coverage, roles or source identity")
    return program


def _semantics(qk, facts):
    """Join declared arithmetic to observed carrier widths; do not infer it from storage i32."""
    sem = copy.deepcopy(qk.get("numerical_semantics") or {})
    internal = sem.get("internal_arithmetic") or {}
    if (sem.get("model", {}).get("engine") != "integer_reference"
        or sem.get("accumulator_dtype") != "i32"
        or internal.get("full_operation_overflow_policy") != "bounded_exact_requires_each_partial_sum"):
        raise ValueError("panel requires the canonical selected exact partial-sum contract")
    rows = facts.get("facts", facts).get("compute_datapaths", [])
    for role, key in (("input", "signed_operand_bits"), ("accumulator", "mac_result_bits")):
        widths = {row.get("elem_bits") for row in rows if row.get("name") == role}
        if widths != {internal.get(key)} or None in widths:
            raise ValueError("selected arithmetic widths differ from observed compute datapaths")
    return sem


def make_case(spec, gemm_cap, gemm_cb, qk_cap, qk_program, semantics):
    """Pure source+metadata construction; no candidate, oracle result or scheduling policy is read."""
    from merlin_experiments.phase0.component_execution_budget import source_for_capsule
    from merlin_experiments.phase0.component_integer_bounds import derive

    from merlin.targetgen.operation_numerics import integer_partial_sum_bound

    cap = copy.deepcopy(gemm_cap if spec["family"] == "gemm" else qk_cap)
    for key in ("software_screen", "integer_partial_sum_bound", "performance_demand", "chain_demand",
                "performance", "comparison_group", "extends", "golden", "input_palette"):
        cap.pop(key, None)
    cap.update(name=spec["case"], source_role="handauthored_compiler_test", label="public",
               source_reference="Frozen operator-only OFAT transfer contract; prior exposure is explicitly recorded.",
               required_oracle_tiers=["L0", "L1", "L2"], performance_scope="scored")
    cap["numerical_semantics"] = copy.deepcopy(semantics)
    cap["interface_mlir"] = "capsule.interface.mlir"
    g = spec["geometry"]
    if spec["family"] == "gemm":
        cb = copy.deepcopy(gemm_cb)
        pack, multiply, commit, _ = [row["operands"] for row in cb["commands"]]
        cb["tensors"][multiply["lhs"]]["shape"] = [g["M"], g["K"]]
        cb["tensors"][pack["src"]]["shape"] = [g["K"], g["N"]]
        source = emit_interface_mlir(cb)
        cap["inputs"] = [{"name": name, **row} for name, row in cb["tensors"].items()]
        cap["operation"] = {"op": "matmul", "attributes": dict(lhs=multiply["lhs"], weight=pack["src"],
                              out=commit["dst"], epilogue=[], output_dtype="i32")}
        outputs = [{"name": commit["dst"], "shape": [g["M"], g["N"]], "dtype": "i32"}]
        bound = integer_partial_sum_bound(semantics, reduction_extent=g["K"], lhs_values=[-4, 3], rhs_values=[-4, 3])
    else:
        program = copy.deepcopy(qk_program)
        for row in program["inputs"]:
            row["shape"] = [64, g["key"]] if row["name"].startswith("KT") else [g["query"], 64]
        typed, source = render(program, operand_dtype="i8", accumulator_dtype="i32")
        cap["inputs"], cap["component_program"] = typed["inputs"], typed
        cap["linalg_mlir"] = cap["interface_mlir"]
        cap["operation"]["attributes"].update(program=program, out=typed["outputs"][0]["name"],
            outs=[row["name"] for row in typed["outputs"]],
            arg_order=[row["name"] for row in typed["inputs"] + typed["outputs"]])
        outputs = [{key: row[key] for key in ("name", "shape", "dtype")} for row in typed["outputs"]]
        bound = derive(source_for_capsule(cap), semantics)
    if bound["status"] != "proven_safe":
        raise ValueError("panel stimulus exceeds the selected internal MAC bound")
    cap["integer_partial_sum_bound"] = bound
    contract = dict(inputs=cap["inputs"], outputs=outputs, operand_dtype="i8", output_dtype="i32",
                    epilogue=[], numerical_semantics=semantics, stimulus_range=[-4, 3],
                    scope="complete invocation, all live outputs", hardware_execution="not_run")
    return cap, source, contract


def write_fixtures(cap, directory, *, tensor_contract=None):
    """Ordinary deterministic leaves and independent mathematical goldens, without compiler code."""
    from merlin_experiments.phase0.component_numerics import evaluate

    from merlin.targetgen.capsule_golden import golden
    from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
    from merlin.targetgen.golden_store import _archive_bytes, write_golden

    outputs = evaluate(cap) if cap["operation"]["op"] == "component_program" else golden(cap)
    leaves = materialize_capsule_leaves(cap)
    dtypes = {"i8": np.int8, "i32": np.int32}
    declarations = {row["name"]: row["dtype"] for row in (cap.get("component_program") or {}).get("outputs", [])}
    output_dtype = cap["operation"].get("attributes", {}).get(
        "output_dtype", cap.get("numeric_policy", {}).get("dtype", "i32"))
    declared_outputs = {name: declarations.get(name, output_dtype) for name in outputs}
    if any(dtype not in dtypes for dtype in [*(row.dtype for row in leaves.values()), *declared_outputs.values()]):
        raise ValueError("physical fixtures require declared i8/i32 tensor dtypes")
    inputs = {name: np.asarray(row.data, dtype=dtypes[row.dtype]).reshape(row.shape) for name, row in leaves.items()}
    arrays = {name: np.asarray(value, dtype=dtypes[declared_outputs[name]]) for name, value in outputs.items()}
    if tensor_contract is not None:
        if set(tensor_contract) != set(inputs) | set(arrays) or set(inputs) & set(arrays):
            raise ValueError("fixture tensor contract must name every input and output exactly once")
        for name, array in {**inputs, **arrays}.items():
            spec = tensor_contract[name]
            if (spec.get("dtype") not in dtypes or array.dtype != dtypes[spec["dtype"]]
                    or list(array.shape) != spec.get("shape")
                    or (spec.get("role") == "output") != (name in arrays)):
                raise ValueError("fixture tensor dtype/shape/role differs from original contract")
    for filename, values in (("inputs.npz", inputs), ("outputs.npz", arrays)):
        (directory / filename).write_bytes(_archive_bytes(values))
    write_golden(directory, {"golden_source": "independent_tensor_integer_reference",
        "integer_partial_sum_bound": cap.get("integer_partial_sum_bound"),
        "qualification": "mathematical reference only; target execution and physical ordering unverified",
        "outputs": outputs})
    return dict(schema=FIXTURES, inputs=pin(directory / "inputs.npz"), outputs=pin(directory / "outputs.npz"),
                stimulus="ordinary materialize_capsule_leaves / Tensor.deterministic; no RNG or candidate input",
                golden=pin(directory / "golden.yaml"), golden_arrays=pin(directory / "golden.npz"))


def prepare_panel(output, *, gemm_capsule, qk_capsule, facts, frozen_revision=None, evidence_root=None):
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("select a new immutable operator-only panel directory")
    gemm, gemm_text, gemm_pin = _template(gemm_capsule)
    qk, qk_text, qk_pin = _template(qk_capsule)
    cb, program = _gemm_template(gemm, gemm_text), _qk_template(qk, qk_text)
    facts_pin = pin(facts)
    semantics = _semantics(qk, json.loads(Path(facts_pin["path"]).read_bytes()))
    evidence = None
    if evidence_root is not None:
        from merlin_experiments.phase0.evidence import load_exported_evidence
        evidence_root = Path(evidence_root).resolve(strict=True)
        evidence = load_exported_evidence(evidence_root)
    output.mkdir(parents=True, mode=0o700)
    _write(output / "operator-policy.json", OPERATOR_POLICY)
    cases = []
    for spec in specifications():
        cap, source, contract = make_case(spec, gemm, cb, qk, program, semantics)
        directory = output / "sources" / spec["case"]
        directory.mkdir(parents=True, mode=0o700)
        (directory / "capsule.interface.mlir").write_text(source)
        fixtures = write_fixtures(cap, directory)
        admission = {"status": "source_ready_admission_pending", "target_execution": "not_run"}
        if evidence is not None:
            from merlin_experiments.phase0.program_admission import accelerator_placement_refusal, screen_written
            screen = screen_written(cap, directory, target=cb["target"], evidence=evidence)
            refusal = accelerator_placement_refusal(screen)
            if refusal:
                raise ValueError(f"panel admission refused {spec['case']}: {refusal}")
            cap["software_screen"] = screen
            admission = {"status": "admitted_source_only", "target_execution": "not_run", "screen": screen}
        cap_path = directory / "capsule.yaml"
        cap_path.write_text(yaml.safe_dump(cap, sort_keys=False))
        _write(directory / "contract.json", contract)
        _write(directory / "program.json", cap.get("component_program") or parse_interface_mlir(source))
        _write(directory / "admission.json", admission)
        cases.append({**spec, "operator_only": True, "source": pin(directory / "capsule.interface.mlir"),
                      "capsule": pin(cap_path), "contract": contract, "contract_pin": pin(directory / "contract.json"),
                      "program": pin(directory / "program.json"),
                      "program_sha256": _digest(cap["operation"]), "fixtures": fixtures,
                      "admission": pin(directory / "admission.json"),
                      "template": gemm_pin if spec["family"] == "gemm" else qk_pin,
                      "original_source_bytes": source == (gemm_text if spec["family"] == "gemm" else qk_text)})
    # Pin again after preparation; an advancing operator input cannot silently mix panel identities.
    if (_template(gemm_capsule)[2] != gemm_pin or _template(qk_capsule)[2] != qk_pin or pin(facts) != facts_pin):
        raise ValueError("canonical panel inputs changed during preparation")
    document = dict(schema=SCHEMA, operator_only=True, author_visible=False, allow_author_mount=False,
        visibility=OPERATOR_POLICY, frozen_revision=frozen_revision, facts=facts_pin, cases=cases,
        counts=dict(total=len(cases), by_family=dict(Counter(row["family"] for row in cases)),
                    by_stratum=dict(Counter(row["stratum"] for row in cases))),
        axis_design="one factor at a time; independent axes; anchors included once; not Cartesian",
        selection_sha256=_digest(specifications()), anchor_roundtrip={"gemm": True, "qk": True},
        admission="admitted_source_only" if evidence else "source_ready_admission_pending",
        qualification_issued=False,
        evidence_manifest=pin(evidence_root / "evidence-manifest.json") if evidence else None)
    _write(output / "manifest.json", document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "gemm-capsule", "qk-capsule", "facts"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--frozen-revision")
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args(argv)
    document = prepare_panel(args.output, gemm_capsule=args.gemm_capsule, qk_capsule=args.qk_capsule,
        facts=args.facts, frozen_revision=args.frozen_revision, evidence_root=args.evidence_root)
    print(json.dumps({"manifest": pin(args.output / "manifest.json"), "counts": document["counts"],
                      "admission": document["admission"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
