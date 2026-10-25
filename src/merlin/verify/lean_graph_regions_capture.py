"""Generate actual connected-region evidence from the explicitly pinned production source.

The worker imports production APIs only in an isolated subprocess. All new code
lives in the primary checkout; this does not copy or integrate experimental code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def _ref(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _load_projection():
    path = Path(__file__).with_name("lean_graph_regions_projection.py")
    spec = importlib.util.spec_from_file_location("primary_graph_regions_projection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("primary projection loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(shape: tuple[int, ...], observed: bool) -> str:
    if not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("positive static shape required")
    dims = "x".join(map(str, shape))
    typ = f"tensor<{dims}xi16>"
    coordinates = ",".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({coordinates})->({coordinates})>"
    results = ["%p", "%a", "%b"] if observed else ["%a", "%b"]
    types = ",".join([typ] * len(results))
    text = f"builtin.module {{ func.func @forward(%x:{typ}, %y:{typ}) -> ({types}) {{\n"
    text += "\n".join(f"%{name}init = tensor.empty() : {typ}" for name in ("p", "a", "b")) + "\n"
    for result, arguments, scalar in (
        ("p", ["%x", "%y"], "arith.addi %s0, %s1 : i16"),
        ("a", ["%p"], "arith.muli %s0, %s0 : i16"),
        ("b", ["%p"], "arith.subi %s0, %s0 : i16"),
    ):
        maps = ",".join([identity] * (len(arguments) + 1))
        iterators = ",".join(["#linalg.iterator_type<parallel>"] * len(shape))
        args = ",".join([*arguments, f"%{result}init"])
        block = ",".join([*(f"%s{i}:i16" for i in range(len(arguments))), "%old:i16"])
        inputs = ",".join([typ] * (len(arguments) + 1))
        text += f"""%{result} = "linalg.generic"({args}) <{{indexing_maps=[{maps}],
          iterator_types=[{iterators}], operandSegmentSizes=array<i32:{len(arguments)},1>}}>
          ({{ ^bb0({block}): %v = {scalar}
            "linalg.yield"(%v):(i16)->() }}) : ({inputs}) -> {typ}\n"""
    return text + f"func.return {','.join(results)} : {types}\n}} }}\n"


def _worker(args) -> None:
    reference = args.reference.resolve(strict=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference, text=True).strip() != REVISION:
        raise ValueError("production reference revision differs")
    projection = _load_projection()
    from merlin.compile.connected_regions import plan_connected_region
    from merlin.compile.fusion_discovery import discover
    from merlin.compile.fusion_workflow import realize_candidate

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    fixture_ref, options, selected_origin, selected_inputs = None, None, None, {}
    if args.realize:
        fixture = reference / "merlin/tests/infra/test_multioutput_fusion.py"
        expected = subprocess.check_output(
            ["git", "show", f"{REVISION}:merlin/tests/infra/test_multioutput_fusion.py"], cwd=reference
        )
        if fixture.read_bytes() != expected:
            raise ValueError("public synthetic fixture differs from pinned source")
        spec = importlib.util.spec_from_file_location("pinned_public_multioutput_fixture", fixture)
        if spec is None or spec.loader is None:
            raise RuntimeError("public fixture loader unavailable")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        settings = destination / "selected-fixture"
        settings.mkdir()
        module, operations, options = helper.model(settings, width=16, shape=tuple(args.shape))
        # Execution-context identity includes the selected input PATHS as well
        # as bytes. Fresh replay must reread those original immutable inputs,
        # not silently select relocated copies with a different context hash.
        selected_origin = (args.selected_fixture_origin or settings).resolve(strict=True)
        for name in ("facts.json", "software.yaml"):
            original = selected_origin / name
            if original.read_bytes() != (settings / name).read_bytes():
                raise ValueError("selected original differs from the pinned synthetic recipe")
            selected_inputs[name] = _ref(original)
        from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

        options["selected_contracts"] = read_scheduling_contracts(
            selected_origin / "facts.json", selected_origin / "software.yaml", target="neutral"
        )
        options["selected_contracts"].revalidate()
        (destination / "selected-contracts.json").write_text(
            json.dumps(options["selected_contracts"].record(), indent=2, sort_keys=True) + "\n"
        )
        fixture_ref = _ref(fixture)
        if not args.no_observer:
            from xdsl.dialects import builtin, func

            function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
            term = tuple(function.body.block.ops)[-1]
            term.operands = (operations[0].results[0], *term.operands)
            function.properties["function_type"] = builtin.FunctionType.from_lists(
                function.function_type.inputs.data, [v.type for v in term.operands]
            )
            module.verify()
    else:
        source = _source(tuple(args.shape), not args.no_observer)
        module = projection.parse_source(source)
        operations = tuple(op for op in module.walk() if op.name == "linalg.generic")
    if args.suffix_observer:
        from xdsl.dialects import builtin, func

        function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
        term = tuple(function.body.block.ops)[-1]
        initializer = operations[-1].operands[-1].owner.clone()
        observer = operations[-1].clone()
        initializer.results[0].name_hint = "observer_init"
        observer.results[0].name_hint = "observer"
        observer.operands = (operations[-1].results[0], initializer.results[0])
        function.body.block.insert_op_before(initializer, term)
        function.body.block.insert_op_before(observer, term)
        term.operands = (*term.operands, observer.results[0])
        function.properties["function_type"] = builtin.FunctionType.from_lists(
            function.function_type.inputs.data, [v.type for v in term.operands]
        )
        module.verify()
    selected = operations[: args.stages]
    before = projection.ir_text(module)
    region = plan_connected_region(module, selected)
    region.revalidate()
    projected, ordinals, value_ids = projection.project(module)
    proposed = projection.planner_record(region, ordinals, value_ids, projected)
    page = discover(module, capability_contract={}, numerical_policy_sha256="0" * 64)
    row = next(
        r
        for r in page["candidates"]
        if r["operation_ordinals"] == [ordinals[o] for o in selected]
        and (not args.realize or r["route"] == "multioutput")
    )
    fallback = realize_candidate(
        module, row, capability_contract={}, numerical_policy_sha256="0" * 64, **(options or {})
    )
    if projection.ir_text(module) != before:
        raise ValueError("planning/declined realization changed the source")
    (destination / "source.mlir").write_text(before)
    (destination / "retained-source.mlir").write_text(projection.ir_text(module))
    for name, value in (
        ("source-projection.json", projected),
        ("plan.json", proposed),
        ("native-plan.json", region.record()),
        ("discovery.json", page),
        ("realization.json", fallback.receipt),
    ):
        (destination / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    realization = {
        "status": fallback.receipt["status"],
        "native_structural_conformance": False,
        "composition_law_truth_independently_qualified": False,
        "hardware_execution_qualified": False,
    }
    if args.realize:
        (destination / "declared-composition-law.json").write_text(
            json.dumps(asdict(options["composition_law"]), indent=2, sort_keys=True) + "\n"
        )
        (destination / "before-kernel.json").write_text(json.dumps(asdict(options["kernel"]), indent=2) + "\n")
    if args.realize and fallback.receipt["status"] == "applied":
        product = fallback.artifact
        product.revalidate()
        images = product.images()
        (destination / "after-kernel.json").write_text(json.dumps(asdict(product.compiled.kernel), indent=2) + "\n")
        (destination / "native-conformance.json").write_text(
            json.dumps(product.conformance, indent=2, sort_keys=True) + "\n"
        )
        image_refs = []
        for index, data in enumerate(images):
            image = destination / f"actual-image-{index}.bin"
            image.write_bytes(data)
            image_refs.append(_ref(image))
        realization.update(
            native_structural_conformance=True,
            actual_images=image_refs,
            source_passes=[asdict(p) for p in product.source_passes],
        )
        by_ordinal = {n["ordinal"]: n for n in projected["nodes"]}
        node_keys = ("ordinal", "semanticKey", "operands", "results", "operandTypes", "resultTypes")
        fused = {
            "body": [{k: by_ordinal[ordinal][k] for k in node_keys} for ordinal in product.operation_ordinals],
            "inputs": [value_ids[v] for v in product.inputs],
            "outputs": [value_ids[v] for v in product.outputs],
        }
        (destination / "fused-body.json").write_text(json.dumps(fused, indent=2, sort_keys=True) + "\n")
    sources = []
    for name, mod in sorted(sys.modules.items()):
        if (name != "merlin" and not name.startswith("merlin.")) or not getattr(mod, "__file__", None):
            continue
        path = Path(mod.__file__).resolve()
        relative = path.relative_to(reference)
        expected = subprocess.check_output(["git", "show", f"{REVISION}:{relative.as_posix()}"], cwd=reference)
        if expected != path.read_bytes():
            raise ValueError(f"production source differs from pinned commit: {name}")
        sources.append({"module": name, "file": _ref(path)})
    scope = {
        "schema": "merlin.graph-regions.capture.v1",
        "production_revision": REVISION,
        "production_root": str(reference),
        "python": sys.version,
        "python_executable": _ref(Path(sys.executable)),
        "generator": _ref(Path(__file__)),
        "projector": _ref(Path(projection.__file__)),
        "production_sources": sources,
        "selected_ordinals": [ordinals[o] for o in selected],
        "shape": args.shape,
        "public_synthetic_fixture": fixture_ref,
        "realization": realization,
        "intermediate_observed": not args.no_observer,
        "suffix_observer": args.suffix_observer,
        "selected_fixture_origin": str(selected_origin) if selected_origin is not None else None,
        "selected_inputs": selected_inputs,
        "source": _ref(destination / "source.mlir"),
        "plan": _ref(destination / "plan.json"),
        "projection": _ref(destination / "source-projection.json"),
        "retained_source": _ref(destination / "retained-source.mlir"),
        "numerical_policy_status": "unadmitted discovery placeholder; no numerical gate discharged",
        "projection_trust": "primary-owned xDSL parser and complete SSA walk; not a Lean-MLIR operational bridge",
        "supported_realization_checked": False,
        "approved_pass_verification": False,
        "native_or_hardware_fidelity_proved": False,
    }
    scope["artifacts"] = {
        str(p.relative_to(destination)): _ref(p) for p in sorted(destination.rglob("*")) if p.is_file()
    }
    (destination / "CAPTURE.json").write_text(json.dumps(scope, indent=2, sort_keys=True) + "\n")


def capture(
    reference: Path,
    destination: Path,
    *,
    shape=(3, 5),
    stages=3,
    observed=True,
    realize=False,
    suffix_observer=False,
    selected_fixture_origin: Path | None = None,
) -> dict:
    reference, destination = reference.resolve(), destination.resolve()
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(reference / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--reference",
        str(reference),
        "--destination",
        str(destination),
        "--stages",
        str(stages),
        "--shape",
        *map(str, shape),
    ]
    if not observed:
        command.append("--no-observer")
    if realize:
        command.append("--realize")
    if suffix_observer:
        command.append("--suffix-observer")
    if selected_fixture_origin is not None:
        command.extend(("--selected-fixture-origin", str(selected_fixture_origin.resolve(strict=True))))
    result = subprocess.run(command, env=environment, cwd=reference, capture_output=True, text=True, timeout=120)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = destination.parent / (destination.name + "-capture-process.json")
    log.write_text(
        json.dumps(
            {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
            indent=2,
        )
        + "\n"
    )
    result.check_returncode()
    return json.loads((destination / "CAPTURE.json").read_text())


def reproduce_native(scope: dict, destination: Path) -> dict:
    """Freshly issue a native product and compare actual inputs/specification/images.

    This deliberately supports the pinned public synthetic recipe only. A saved
    realization status is not authority, and a different legal physical layout
    is not silently substituted for the captured artifact.
    """
    if scope.get("public_synthetic_fixture") is None:
        raise ValueError("no reproducible supported native recipe in this capture")
    if scope.get("production_revision") != REVISION:
        raise ValueError("changed production revision")
    for key, filename in (
        ("source", "source.mlir"),
        ("retained_source", "retained-source.mlir"),
        ("plan", "plan.json"),
        ("projection", "source-projection.json"),
    ):
        if scope.get(key) != scope.get("artifacts", {}).get(filename):
            raise ValueError("capture original/artifact association differs")
    fresh = capture(
        Path(scope["production_root"]),
        destination,
        shape=tuple(scope["shape"]),
        stages=len(scope["selected_ordinals"]),
        observed=scope["intermediate_observed"],
        realize=True,
        suffix_observer=scope.get("suffix_observer", False),
        selected_fixture_origin=Path(scope["selected_fixture_origin"]),
    )
    if fresh["realization"]["status"] != "applied":
        raise ValueError("fresh native realization did not admit this source/observer boundary")
    for key in (
        "schema",
        "production_revision",
        "production_root",
        "python",
        "python_executable",
        "generator",
        "projector",
        "production_sources",
        "selected_ordinals",
        "shape",
        "public_synthetic_fixture",
        "intermediate_observed",
        "suffix_observer",
        "selected_fixture_origin",
        "selected_inputs",
        "numerical_policy_status",
        "projection_trust",
        "supported_realization_checked",
        "approved_pass_verification",
        "native_or_hardware_fidelity_proved",
    ):
        if type(scope.get(key)) is not type(fresh[key]) or scope[key] != fresh[key]:
            raise ValueError(f"captured production/recipe/tool identity differs: {key}")
    if set(fresh["artifacts"]) != set(scope["artifacts"]):
        raise ValueError("captured and reproduced native artifact domains differ")
    for name, expected in scope["artifacts"].items():
        actual = fresh["artifacts"][name]
        if (actual["sha256"], actual["bytes"]) != (expected["sha256"], expected["bytes"]):
            raise ValueError(f"actual native source/specification/result differs: {name}")
        if Path(actual["path"]).read_bytes() != Path(expected["path"]).read_bytes():
            raise ValueError(f"actual reproduced native bytes differ: {name}")
    from merlin.verify import lean_graph_regions as base

    if base.content_identity(scope["realization"]) != base.content_identity(fresh["realization"]):
        raise ValueError("actual native realization/context identity differs")
    result = {
        "status": "fresh production replay matched every native artifact byte",
        "captured_revision": scope["production_revision"],
        "reproduced_capture": _ref(destination / "CAPTURE.json"),
        "artifacts_compared": len(scope["artifacts"]),
        "composition_declaration_truth_proved": False,
        "native_or_hardware_fidelity_proved": False,
    }
    (destination.parent / (destination.name + "-reproduction.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--stages", type=int, choices=(2, 3), default=3)
    parser.add_argument("--no-observer", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--realize", action="store_true")
    parser.add_argument("--suffix-observer", action="store_true")
    parser.add_argument("--selected-fixture-origin", type=Path)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        result = capture(
            args.reference,
            args.destination,
            shape=tuple(args.shape),
            stages=args.stages,
            observed=not args.no_observer,
            realize=args.realize,
            suffix_observer=args.suffix_observer,
            selected_fixture_origin=args.selected_fixture_origin,
        )
        print(
            json.dumps(
                {
                    "source": result["source"],
                    "selected_ordinals": result["selected_ordinals"],
                    "approved_pass_verification": False,
                }
            )
        )


if __name__ == "__main__":
    main()
