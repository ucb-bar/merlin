"""Targeted runtime-scale evidence using the ordinary admission/Spike pipeline.

This produces a smoke report, never a Seal validation ledger. Execute only in
an explicitly selected integration checkout and a fresh output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = (
    "aligned",
    "partial",
    "rne_saturation",
    "conversion_boundary",
    "runtime_scale_zero",
    "runtime_scale_quarter",
    "runtime_scale_half",
    "runtime_scale_nonpow2",
)
SELECTION = dict(
    state="candidate",
    structure="shared_input",
    dataflow="WS",
    nodes=2,
    producer="unseeded i8 matmul",
    exits=["runtime_input requant", "raw accumulator"],
    strategy="f32_product_rne",
    granularity="tensor",
    activation="optional accumulator ReLU",
    excluded=["residual", "pool"],
)


def _narrow(node):
    regime, readout = node.get("regime", {}), node.get("readout", {})
    stages = readout.get("stages", [])
    kinds = [stage.get("kind") for stage in stages]
    activation = [s for s in stages if s.get("kind") == "activation"]
    arithmetic = [s for s in stages if s.get("kind") != "activation"]
    if (
        readout.get("exit") != "requant"
        or regime.get("scale_source") != "runtime_input"
        or kinds
        not in (
            ["convert", "scale", "requant"],
            ["activation", "convert", "scale", "requant"],
            ["convert", "activation", "scale", "requant"],
        )
    ):
        return False
    if any(s.get("function") != "relu" or s.get("domain") != "accumulator" for s in activation):
        return False
    convert, scale, quantize = arithmetic
    return (
        convert.get("to") == "f32"
        and convert.get("rounding") == "half_even"
        and scale.get("dtype") == "f32"
        and scale.get("rounding") == "half_even"
        and scale.get("granularity") in ("tensor", "per_command")
        and quantize.get("to") == "i8"
        and quantize.get("rounding") == "half_even"
        and quantize.get("clamp") == [-128, 127]
    )


def qualifies(row):
    semantic = row.get("semantic", {})
    nodes = semantic.get("nodes", [])
    if (
        row.get("state") != "candidate"
        or semantic.get("structure") != "shared_input"
        or semantic.get("dataflow") != "WS"
        or len(nodes) != 2
    ):
        return False
    for node in nodes:
        regime = node.get("regime", {})
        if (
            node.get("form") != "contraction"
            or node.get("realization") != "matmul"
            or node.get("seed") is not False
            or node.get("residual") is not None
            or regime.get("readout_strategy") != "f32_product_rne"
            or regime.get("weight_scale_granularity") != "tensor"
        ):
            return False

    def raw(node):
        return node.get("readout", {}).get("exit") == "accumulator" and node["readout"].get("stages") == []

    return (_narrow(nodes[0]) and raw(nodes[1])) or (_narrow(nodes[1]) and raw(nodes[0]))


def select_candidates(manifest, candidate_id=None):
    rows = [r for r in manifest["rows"] if qualifies(r)]
    if candidate_id is not None:
        rows = [r for r in rows if r["candidate_id"] == candidate_id]
    if not rows:
        raise ValueError("empty runtime_scale_mixed_ws selection (candidate ID must qualify)")
    ids = [r["candidate_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate identity")
    return sorted(rows, key=lambda r: r["candidate_id"])


def validate_payload(payload):
    """Source/ABI witnesses are required before the ordinary provider admission."""
    import numpy as np

    from merlin.common import mlir_query as mq

    from .runtime_scale_reference import needs_runtime_scale, runtime_multiplier

    module = mq.parse(payload["source"])
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    if len(function.body.block.args) != len(payload["inputs"]):
        raise ValueError("source arguments differ from generated input ABI")
    scales = []
    for index, cap in enumerate(payload["nodes"]):
        if not needs_runtime_scale(cap):
            continue
        name = cap["operation"]["attributes"]["case"]["runtime_scale_input"]
        merged = f"node{index}_{name}"
        matches = [(i, leaf) for i, leaf in enumerate(payload["inputs"]) if leaf["name"] == merged]
        if len(matches) != 1:
            raise ValueError("generated runtime scale is absent after merge")
        ordinal, leaf = matches[0]
        value = np.asarray(payload["input_values"][merged], dtype=np.float32)
        runtime_multiplier(cap, {name: value})
        arg = function.body.block.args[ordinal]
        if str(arg.type) != "tensor<1xf32>" or not arg.uses:
            raise ValueError("merged runtime scale argument is not consumed by source")
        scales.append(dict(argument=ordinal, name=merged, dtype=leaf["dtype"], shape=leaf["shape"]))
    refs = payload["references"]
    if len(scales) != 1 or len(refs) != 2 or {r["dtype"] for r in refs} != {"i8", "i32"}:
        raise ValueError("smoke requires one actual runtime input and two typed references")
    return scales


def passed(outcome):
    return outcome.get("outcome") == "pass" and all(outcome.get(k) is True for k in ("numeric", "placement", "trace"))


def _run_case(candidate, label, manifest, raw_facts, selection, output):
    try:
        return _execute_selected_case(candidate, label, manifest, raw_facts, selection, output)
    except Exception as error:
        # A failed generated shard stays visible and earns no admission.
        import traceback

        from .fusion_cases import digest

        case_id = digest(dict(candidate_id=candidate["candidate_id"], shape_class=label))
        directory = Path(output) / case_id
        directory.mkdir(exist_ok=True)
        (directory / "failure.txt").write_text(traceback.format_exc())
        return dict(
            case_id=case_id,
            candidate_id=candidate["candidate_id"],
            shape_class=label,
            outcome="not_admitted",
            numeric=False,
            placement=False,
            trace=False,
            reason=f"generation: {type(error).__name__}: {error}",
        )


def _execute_selected_case(candidate, label, manifest, raw_facts, selection, output):
    from merlin.targetgen import composition_corpus as corpus

    from .fusion_cases import (
        build_payload,
        canonical,
        digest,
        exclusion_context,
        facts_for_cases,
        file_digest,
        write_frozen,
    )
    from .fusion_execution import execute_case

    case_id = digest(dict(candidate_id=candidate["candidate_id"], shape_class=label))
    directory = Path(output) / case_id
    directory.mkdir()
    schemes = {s["id"]: s for s in manifest["scheme_resolution"]["schemes"]}
    payload = build_payload(
        candidate,
        schemes,
        facts_for_cases(manifest, raw_facts),
        label,
        None,
        corpus.held_out_shapes(),
        exclusion_context(),
        directory / "screen",
    )
    scales = validate_payload(payload)
    write_frozen(directory / "payload.json", canonical(payload))
    outcome = execute_case(payload, candidate, selection, directory)
    readback = json.loads((directory / "output.json").read_bytes()) if (directory / "output.json").is_file() else None
    retained = directory / "oracle-obligations.json"
    oracle = json.loads(retained.read_bytes()) if retained.is_file() else None
    audit = directory / "composition-audit.json"
    loads = json.loads(audit.read_bytes()).get("runtime_scale_loads") if audit.is_file() else None
    evidence = [
        dict(member=str(path.relative_to(directory)), sha256=file_digest(path), bytes=path.stat().st_size)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    return dict(
        case_id=case_id,
        candidate_id=candidate["candidate_id"],
        shape_class=label,
        evidence=evidence,
        source_sha256=payload["source_sha256"],
        inputs_sha256=payload["inputs_sha256"],
        references_sha256=payload["references_sha256"],
        runtime_scales=scales,
        results=[{k: r[k] for k in ("name", "dtype", "shape")} for r in payload["references"]],
        readback=readback,
        oracle_obligations=oracle,
        scale_load_evidence=None if loads is None else loads["evidence"],
        **outcome,
    )


def runtime_dependence(rows):
    """Check changed input effects and invariant raw outputs across fixed labels.

    These are separately compiled calls. Same-ELF reuse needs additional evidence.
    """
    results = []
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        calls = [r for r in rows if r["candidate_id"] == candidate_id and r["shape_class"].startswith("runtime_scale_")]
        valid = len(calls) == 4 and all(passed(r) and r.get("readback") for r in calls)
        raw, narrow, sources = set(), set(), set()
        if valid:
            for call in calls:
                sources.add(call["source_sha256"])
                for spec, data in zip(call["results"], call["readback"]["bytes"], strict=True):
                    (raw if spec["dtype"] == "i32" else narrow).add(data)
            valid = len(raw) == 1 and len(narrow) > 1 and len(sources) == 1
        results.append(
            dict(
                candidate_id=candidate_id,
                source_invariant=len(sources) == 1,
                raw_invariant=len(raw) == 1,
                narrow_varies=len(narrow) > 1,
                status="pass" if valid else "fail",
            )
        )
    return results


def print_selection(selected):
    print(json.dumps(dict(predicate=SELECTION, candidate_ids=[r["candidate_id"] for r in selected])), flush=True)


def run(args):
    from merlin.targetgen.core_aten_device import load_execution_provider

    from .fusion_cases import canonical, file_digest, reference_identity, write_frozen
    from .fusion_execution import pinned_extension
    from .fusion_validation import verify_snapshot, worker_pool

    manifest = json.loads(args.candidates.read_bytes())
    selected = select_candidates(manifest, args.candidate_id)
    verify_snapshot(argparse.Namespace(**vars(args), snapshot=None), manifest)
    provider = load_execution_provider(args.target, args.provider, candidate_writable_roots=())
    extension = pinned_extension(args.target, args.extension, provider.runner_options())
    selection = dict(
        target=args.target,
        package=str(args.package.resolve()),
        provider=str(args.provider.resolve()),
        spec=str(args.spec.resolve()),
        facts=str(args.facts.resolve()),
        extension=str(extension.resolve()),
        extension_sha256=file_digest(extension),
        timeout=600,
        candidates={"normalized_facts": manifest["normalized_facts"]},
    )
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(
        candidate_manifest_sha256=file_digest(args.candidates),
        facts_sha256=file_digest(args.facts),
        spec_sha256=file_digest(args.spec),
        provider_sha256=file_digest(args.provider),
        extension=str(extension.resolve()),
        extension_sha256=file_digest(extension),
        reference_implementation_sha256=reference_identity(),
    )
    write_frozen(args.output / "execution-selection.json", canonical(selection))
    write_frozen(
        args.output / "selection.json",
        canonical(
            dict(
                predicate=SELECTION,
                identity=identity,
                candidate_ids=[r["candidate_id"] for r in selected],
                labels=LABELS,
            )
        ),
    )
    print_selection(selected)
    raw_facts = json.loads(args.facts.read_bytes())
    with worker_pool(args.workers) as pool:
        futures = [
            pool.submit(_run_case, row, label, manifest, raw_facts, selection, args.output)
            for row in selected
            for label in LABELS
        ]
        rows = [future.result() for future in futures]
    dependence = runtime_dependence(rows)
    success = all(passed(row) for row in rows) and all(r["status"] == "pass" for r in dependence)
    report = dict(
        schema="merlin.runtime_scale_smoke.v1",
        qualification="fixed nonnegative finite f32 palette",
        identity=identity,
        predicate=SELECTION,
        candidate_ids=[r["candidate_id"] for r in selected],
        same_elf_reuse="not_measured",
        # Each admitted row carries a discharged lint oracle-obligation record
        # and a retired load-to-CONFIG_ST rs2 chain for its scale input.
        oracle_obligations="discharged per case: nonnegative_test_input or parity_proved_extension",
        scale_load_provenance="retired_load_to_config_st_rs2",
        signed_relu_spike_probe="not_run",
        status="pass" if success else "fail",
        seal_admission=False,
        runtime_dependence=dependence,
        rows=rows,
    )
    write_frozen(args.output / "runtime-scale-smoke.json", canonical(report))
    return 0 if success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    for name in ("candidates", "facts", "spec", "package", "provider", "extension", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--selection", required=True, choices=["runtime_scale_mixed_ws"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-id")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.extension.is_absolute():
        parser.error("extension must be an absolute pinned library path")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
