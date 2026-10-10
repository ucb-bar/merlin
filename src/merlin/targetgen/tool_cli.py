"""Installed operator entrypoints for target-neutral support primitives.

These tools are also imported by selected OOT providers. Explicit input and output arguments keep
their use independent of a Merlin checkout or any in-tree accelerator implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _boot(args: argparse.Namespace) -> int:
    from .fixed_format import boot

    preamble = Path(args.asm_preamble).read_text() if args.asm_preamble else ""
    result = boot.build_boot_object(
        args.source,
        args.out,
        target=args.target,
        clang=args.clang,
        march=args.march,
        mabi=args.mabi,
        asm_preamble=preamble,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _link(args: argparse.Namespace) -> int:
    from .fixed_format import link

    result = link.link_fork_free(
        args.object,
        args.linker_script,
        args.out,
        target=args.target,
        linker=args.linker,
    )
    print(result)
    return 0


def _register_slices(args: argparse.Namespace) -> int:
    from .rtl import register_slices

    raw = json.loads(Path(args.inputs).read_text())
    if not isinstance(raw, dict) or any(
        not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str) or type(value[1]) is not int
        for value in raw.values()
    ):
        raise ValueError("--inputs must map hardware SSA names to [input_label, bit_width]")
    result = register_slices.derive_register_slices(
        Path(args.hw).read_text(),
        module=args.module,
        registers=args.register,
        selector=args.selector,
        selector_value=int(args.selector_value, 0),
        inputs={key: (value[0], value[1]) for key, value in raw.items()},
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(rendered)
    else:
        print(rendered, end="")
    return 0


def _spike_extension(args: argparse.Namespace) -> int:
    from . import spike_extension

    result = spike_extension.resolve(
        args.target,
        default_library_dir=args.default_library_dir,
        default_extension_name=args.default_extension_name,
    )
    print(
        json.dumps(
            {
                "target": result.target,
                "declared": result.declared,
                "extension_name": result.extension_name,
                "extlib": str(result.extlib) if result.extlib else None,
                "library_dir": str(result.library_dir),
                "sha256": result.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rtl_source_audit(args: argparse.Namespace) -> int:
    """Compare selected source ports with extracted facts and authored audit questions."""
    from .rtl import source_audit

    return source_audit.main(
        [
            "--source-bundle",
            args.source_bundle,
            "--facts",
            args.facts,
            "--hardware-spec",
            args.hardware_spec,
            "--output",
            args.output,
        ]
    )


def _counter_source_observation(args: argparse.Namespace) -> int:
    """Check complete source-local rows; retain all physical timer unknowns."""
    from dataclasses import asdict, fields

    from merlin.common.strict_json import loads

    from .rtl.hw_combinational import EvaluationLimits
    from .rtl.hw_counter_intervals import (
        CounterEndpointSelection,
        CounterInterval,
        CounterIntervalLimits,
        CounterPhaseSample,
        observe_counter_intervals,
    )
    from .rtl.hw_counter_state_timelines import (
        StateGetterSelection,
        StatePhaseSample,
        StateTimelineLimits,
        observe_state_getter_timeline,
    )

    def read(path, limit):
        if type(limit) is not int or not 0 < limit < 1 << 63:
            raise ValueError("Counter observation input bound is unavailable.")
        try:
            with Path(path).open("rb") as handle:
                value = handle.read(limit + 1)
        except OSError:
            raise ValueError("Counter observation input is unavailable.") from None
        if len(value) > limit:
            raise ValueError("Counter observation input exceeds its byte bound.")
        return value

    def record(cls, value, *, arrays=()):
        if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
            raise ValueError("Counter observation request fields are incomplete.")
        converted = dict(value)
        for name in arrays:
            if type(converted[name]) is not list:
                raise ValueError("Counter observation request array is unavailable.")
            converted[name] = tuple(converted[name])
        return cls(**converted)

    request_bytes = read(args.request, args.max_request_bytes)
    request = loads(request_bytes, max_bytes=args.max_request_bytes)
    keys = {"schema", "kind", "selection", "limits", "expected_phases", "samples", "intervals"}
    if type(request) is not dict or set(request) != keys or request["schema"] != "merlin.counter_source_request.v1":
        raise ValueError("Counter observation request schema is unsupported.")
    kind = request["kind"]
    if kind not in ("unit_counter", "state_getter"):
        raise ValueError("Counter observation request kind is unsupported.")
    limit_type = CounterIntervalLimits if kind == "unit_counter" else StateTimelineLimits
    raw_limits = request["limits"]
    if type(raw_limits) is not dict or set(raw_limits) != {field.name for field in fields(limit_type)}:
        raise ValueError("Counter observation limit roster is incomplete.")
    limits = record(limit_type, {**raw_limits, "expressions": record(EvaluationLimits, raw_limits["expressions"])})
    selection_type = CounterEndpointSelection if kind == "unit_counter" else StateGetterSelection
    selection = record(
        selection_type, request["selection"], arrays=() if kind == "unit_counter" else ("register_ordinals",)
    )
    phases, samples, intervals = (request[name] for name in ("expected_phases", "samples", "intervals"))
    if (
        type(phases) is not int
        or phases < 2
        or phases % 2
        or phases > limits.expressions.cases
        or type(samples) is not list
        or len(samples) != phases
        or type(intervals) is not list
        or not 0 < len(intervals) <= limits.intervals
    ):
        raise ValueError("Counter observation phase or interval roster is incomplete.")
    sample_type = CounterPhaseSample if kind == "unit_counter" else StatePhaseSample
    arrays = ("inputs", "outputs") if kind == "unit_counter" else ("inputs", "states", "outputs")
    samples = tuple(record(sample_type, row, arrays=arrays) for row in samples)
    intervals = tuple(record(CounterInterval, row) for row in intervals)
    source_bytes = read(args.source, limits.expressions.source_bytes)
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeError:
        raise ValueError("Counter observation source encoding is unsupported.") from None
    observer = observe_counter_intervals if kind == "unit_counter" else observe_state_getter_timeline
    result = observer(
        source, selection=selection, limits=limits, expected_phases=phases, samples=samples, intervals=intervals
    )
    rendered = {
        "schema": "merlin.counter_source_observation.v1",
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "observation": asdict(result),
        "scope": "source-local values only; sample custody, physical units, runtime and timing remain unknown",
    }
    with Path(args.out).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(rendered, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


def _semantic_search(args: argparse.Namespace) -> int:
    """Inspect real linalg-on-tensors MLIR with a selected instruction model."""
    from .contract.linalg_iface import parse_linalg_mlir
    from .instruction_semantics import validate_normalized_instruction_model
    from .semantic_search import SearchLimits, search_linalg_inventory

    mlir_raw = Path(args.mlir).read_bytes()
    model_raw = Path(args.instruction_model).read_bytes()
    model = json.loads(model_raw)
    if not isinstance(model, dict) or model.get("target") != args.target:
        raise ValueError("selected instruction model target differs from --target")
    if model.get("schema") != "merlin.instruction_semantics.v1":
        raise ValueError("unsupported instruction model schema")
    # Phase 0 emits an explicit diagnostic stub when no OOT description was
    # selected. Only that non-selectable case may lack normalized source hashes.
    if not (model.get("status") == "UNKNOWN" and model.get("instructions") == []):
        model = validate_normalized_instruction_model(model, expected_target=args.target)
    parsed = parse_linalg_mlir(mlir_raw.decode("utf-8"))
    defaults = SearchLimits()
    result = search_linalg_inventory(
        parsed,
        model,
        limits=SearchLimits(
            max_candidates=args.max_candidates if args.max_candidates is not None else defaults.max_candidates,
            timeout_ms=args.timeout_ms if args.timeout_ms is not None else defaults.timeout_ms,
        ),
    )
    receipt = {
        "schema": "merlin.semantic_search_invocation.v1",
        "target": args.target,
        "inputs": {
            "mlir_sha256": hashlib.sha256(mlir_raw).hexdigest(),
            "instruction_model_sha256": hashlib.sha256(model_raw).hexdigest(),
        },
        "result": result,
        "qualification": "selection and modeled allocation only; no target code or execution proof",
    }
    destination = Path(args.out).absolute()
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("semantic-search output may not traverse a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": str(destination), "regions": len(result["regions"])}, sort_keys=True))
    return 0


def _outline_integer_matmuls(args: argparse.Namespace) -> int:
    """Materialize diagnostic model-to-kernel slices for OOT compiler experiments."""
    from .contract.model_kernel_outline import outline_integer_matmuls

    source = Path(args.mlir)
    if not source.is_file() or source.is_symlink():
        raise ValueError("--mlir must name a regular, non-symlink model file")
    selected = {}
    for label, argument in (("software spec", args.software_spec), ("capability contract", args.capability_contract)):
        path = Path(argument)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"--{label.replace(' ', '-')} must name a regular, non-symlink selected file")
        selected[label] = path.read_bytes()
    result = outline_integer_matmuls(
        source.read_bytes(),
        target=args.target,
        software_spec=selected["software spec"],
        capability_contract=selected["capability contract"],
    )
    destination = Path(args.out).absolute()
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("outline output may not traverse a symlink")
    destination.mkdir(parents=True, exist_ok=False)
    for candidate in result["candidates"]:
        name = f"kernel-{candidate['ordinal']:06d}.interface.mlir"
        (destination / name).write_text(candidate.pop("interface_mlir"))
        candidate["interface_file"] = name
    (destination / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"out": str(destination), "candidates": len(result["candidates"]), "refused": len(result["refused"])},
            sort_keys=True,
        )
    )
    return 0


def _probe_integer_model_route(args: argparse.Namespace) -> int:
    """Record selected OOT command emission for exact captured integer kernels."""
    from .contract.model_kernel_route import probe_integer_model_kernels

    selected = {}
    for label, argument in (
        ("mlir", args.mlir),
        ("software-spec", args.software_spec),
        ("capability-contract", args.capability_contract),
    ):
        path = Path(argument)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"--{label} must name a regular, non-symlink file")
        selected[label] = path.read_bytes()
    result = probe_integer_model_kernels(
        selected["mlir"],
        target=args.target,
        software_spec=selected["software-spec"],
        capability_contract=selected["capability-contract"],
        package_dir=args.package,
        timeout=args.timeout,
    )
    destination = Path(args.out).absolute()
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("route-probe output may not traverse a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "out": str(destination),
                "candidate_count": result["candidate_count"],
                "emission_counts": result["emission_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


def _stage_integer_model_admission(args: argparse.Namespace) -> int:
    """Write an exact, fail-closed SW-admission development review artifact."""
    from merlin.llvmlower.staged_admission import build_staged_candidate, stage_integer_model_admission

    selected = {}
    inputs = [
        ("mlir", args.mlir),
        ("software-spec", args.software_spec),
        ("capability-contract", args.capability_contract),
    ]
    if args.rtl_facts:
        inputs.append(("rtl-facts", args.rtl_facts))
    for label, argument in inputs:
        path = Path(argument)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"--{label} must name a regular, non-symlink file")
        selected[label] = path.read_bytes()
    result = stage_integer_model_admission(
        selected["mlir"],
        target=args.target,
        software_spec=selected["software-spec"],
        capability_contract=selected["capability-contract"],
        package_dir=args.package,
        operation_id=args.operation_id,
        rtl_facts=selected.get("rtl-facts"),
        timeout=args.timeout,
    )
    destination = Path(args.out).absolute()
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("admission-review output may not traverse a symlink")
    if not args.build_dir and (args.cflag or args.codegen_target != "riscv"):
        raise ValueError("--cflag and --codegen-target require --build-dir")
    if args.build_dir:
        if "rtl-facts" not in selected:
            raise ValueError("--build-dir requires exact --rtl-facts bytes")
        build_dir = Path(args.build_dir).absolute()
        if build_dir.exists() or build_dir.is_symlink() or any(parent.is_symlink() for parent in build_dir.parents):
            raise ValueError("--build-dir must be fresh and may not traverse a symlink")
        if destination.is_relative_to(build_dir):
            raise ValueError("admission review artifact must be outside the fresh build directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    summary = {
        "out": str(destination),
        "candidate_count": result["candidate_count"],
        "review_required": result["review_required"],
    }
    if args.build_dir:
        receipt = build_staged_candidate(
            result,
            model=selected["mlir"],
            target=args.target,
            software_spec=selected["software-spec"],
            capability_contract=selected["capability-contract"],
            package_dir=args.package,
            operation_id=args.operation_id,
            rtl_facts=selected["rtl-facts"],
            workdir=build_dir,
            codegen_target=args.codegen_target,
            cflags=args.cflag,
            timeout=args.timeout,
        )
        receipt_path = build_dir / "candidate-build-receipt.json"
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        summary["build_receipt"] = str(receipt_path)
        summary["build_status"] = receipt["status"]
    print(json.dumps(summary, sort_keys=True))
    return 0


def _mem_perturb_variant(args: argparse.Namespace) -> int:
    """Build the memory-perturbing variant of a Verilator simulator (:mod:`.mem_perturb`).

    The declared digest of the stock simulator comes from the pin registry by artifact name, so the
    control relink is compared with what was pinned rather than with whatever sits on disk."""
    from merlin.common import provenance

    from . import mem_perturb

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; a variant is built into a fresh directory, never over another")
    base_digest = ""
    if args.base_artifact:
        artifacts = provenance.load_artifacts()
        if args.base_artifact not in artifacts:
            raise SystemExit(f"no artifact named {args.base_artifact!r} in the pin registry")
        base_digest = artifacts[args.base_artifact].digest
        if not base_digest:
            raise SystemExit(f"{args.base_artifact!r} declares no digest, so the control relink has nothing to match")
    receipt = mem_perturb.build_verilator_variant(
        args.obj_dir, out, makefile=args.makefile, control=not args.no_control, base_digest=base_digest
    )
    print(json.dumps(receipt, sort_keys=True))
    control = receipt["control_relink"]
    return 0 if args.no_control or control.get("matches_declared_base") is not False else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merlin-target-tools", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("fixed-boot", help="stock-assemble and transcode a fixed-format boot object")
    boot.add_argument("--target", required=True)
    boot.add_argument("--source", required=True)
    boot.add_argument("--out", required=True)
    boot.add_argument("--clang", required=True)
    boot.add_argument("--asm-preamble", help="target-owned assembler preamble file")
    boot.add_argument("--march", default="rv32im")
    boot.add_argument("--mabi", default="ilp32")
    boot.set_defaults(func=_boot)

    link = sub.add_parser("fixed-link", help="stock-link and patch derived fixed-format relocations")
    link.add_argument("--target", required=True)
    link.add_argument("--object", action="append", required=True)
    link.add_argument("--linker-script", required=True)
    link.add_argument("--out", required=True)
    link.add_argument("--linker")
    link.set_defaults(func=_link)

    slices = sub.add_parser("register-slices", help="derive conditional RTL register input slices")
    slices.add_argument("--hw", required=True, help="elaborated HW MLIR file")
    slices.add_argument("--module", required=True)
    slices.add_argument("--register", action="append", required=True, help="hardware SSA register reference")
    slices.add_argument("--selector", required=True, help="hardware SSA selector reference")
    slices.add_argument("--selector-value", required=True, help="integer selector value (base 0)")
    slices.add_argument("--inputs", required=True, help="JSON mapping SSA refs to [input_label, bit_width]")
    slices.add_argument("--out", help="JSON output path; stdout when omitted")
    slices.set_defaults(func=_register_slices)

    spike = sub.add_parser("spike-extension", help="verify the selected target's L2 model identity")
    spike.add_argument("--target", required=True)
    spike.add_argument("--default-library-dir", required=True)
    spike.add_argument("--default-extension-name", required=True)
    spike.set_defaults(func=_spike_extension)

    audit = sub.add_parser("rtl-source-audit", help="audit extracted facts against exact selected RTL ports")
    audit.add_argument("--source-bundle", required=True)
    audit.add_argument("--facts", required=True)
    audit.add_argument("--hardware-spec", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=_rtl_source_audit)

    counter = sub.add_parser("counter-source-observation", help="check complete source-local counter/state rows")
    counter.add_argument("--source", required=True, help="exact selected HW MLIR source")
    counter.add_argument("--request", required=True, help="closed source selection, limits and complete sample JSON")
    counter.add_argument("--max-request-bytes", required=True, type=int, help="explicit request byte bound")
    counter.add_argument("--out", required=True, help="fresh data-only JSON observation destination")
    counter.set_defaults(func=_counter_source_observation)

    selection = sub.add_parser(
        "semantic-search", help="inspect linalg-on-tensors kernels against selected OOT instruction semantics"
    )
    selection.add_argument("--target", required=True)
    selection.add_argument("--mlir", required=True, help="exact model2MLIR/capsule linalg-on-tensors MLIR")
    selection.add_argument("--instruction-model", required=True, help="frozen Phase 0 instruction-semantics JSON")
    selection.add_argument("--out", required=True, help="fresh JSON receipt destination")
    # Do not import the host-private implementation merely to display/help or run an
    # unrelated operator command. The semantic-search action resolves its defaults.
    selection.add_argument("--max-candidates", type=int)
    selection.add_argument("--timeout-ms", type=int)
    selection.set_defaults(func=_semantic_search)

    outline = sub.add_parser(
        "outline-int-mm", help="materialize diagnostic i8×i8→i32 contraction slices from model MLIR"
    )
    outline.add_argument("--target", required=True)
    outline.add_argument("--mlir", required=True, help="captured linalg-on-tensors model MLIR")
    outline.add_argument("--software-spec", required=True, help="selected Phase 0 software-spec YAML or JSON")
    outline.add_argument("--capability-contract", required=True, help="selected Phase 0 capability contract")
    outline.add_argument("--out", required=True, help="fresh directory for kernel interfaces and source bindings")
    outline.set_defaults(func=_outline_integer_matmuls)

    route = sub.add_parser("probe-int-mm-route", help="observe OOT command emission for exact model integer matmuls")
    route.add_argument("--target", required=True)
    route.add_argument("--mlir", required=True, help="exact captured model MLIR")
    route.add_argument("--software-spec", required=True, help="selected software-spec YAML or JSON")
    route.add_argument("--capability-contract", required=True, help="selected capability contract")
    route.add_argument("--package", required=True, help="selected OOT compiler package")
    route.add_argument("--timeout", type=int, default=30, help="seconds per distinct kernel interface")
    route.add_argument("--out", required=True, help="fresh JSON diagnostic receipt destination")
    route.set_defaults(func=_probe_integer_model_route)

    stage = sub.add_parser(
        "stage-int-mm-admission", help="bind one integer model candidate to unreviewed compiler/shim evidence"
    )
    stage.add_argument("--target", required=True)
    stage.add_argument("--mlir", required=True, help="exact captured model MLIR")
    stage.add_argument("--software-spec", required=True, help="selected authored software spec")
    stage.add_argument("--capability-contract", required=True, help="selected capability contract")
    stage.add_argument("--package", required=True, help="selected OOT compiler package")
    stage.add_argument("--operation-id", required=True, help="exact model SHA-bound operation ID")
    stage.add_argument("--rtl-facts", help="selected same-target RTL facts for a reproducible shim tile edge")
    stage.add_argument(
        "--build-dir", help="fresh directory for one diagnostic kernel and shim; stage report survives build failure"
    )
    stage.add_argument("--codegen-target", choices=("riscv", "x86"), default="riscv")
    stage.add_argument("--cflag", action="append", help="explicit board C flag; repeat as --cflag=-flag")
    stage.add_argument("--timeout", type=int, default=30, help="seconds per selected package entrypoint")
    stage.add_argument("--out", required=True, help="fresh JSON review artifact destination")
    stage.set_defaults(func=_stage_integer_model_admission)

    perturb = sub.add_parser(
        "mem-perturb-variant", help="relink a Verilator simulator with the seeded response-reordering memory model"
    )
    perturb.add_argument("--obj-dir", required=True, help="the simulator's Verilator object directory (read only)")
    perturb.add_argument("--makefile", required=True, help="the makefile in --obj-dir that links the simulator")
    perturb.add_argument("--out", required=True, help="fresh directory for the variant and its build receipt")
    perturb.add_argument("--base-artifact", help="pin-registry artifact the stock simulator is declared as")
    perturb.add_argument("--no-control", action="store_true", help="skip the control relink of unmodified objects")
    perturb.set_defaults(func=_mem_perturb_variant)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
