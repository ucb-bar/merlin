"""Fresh ordinary per-arm execution and fixed raw-event consumption.

The original source, numerical outputs, actual linked ELF and captured native
stream are reopened. OOT support owns stage boundaries and timer semantics;
neither a complete event roster nor this data product issues measurement roles.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common import strict_json as json_owner
from merlin.perf import component_coherent_measurement as coherent_owner
from merlin.perf import component_measurement_stream as stream_owner
from merlin.perf.component_coherent_measurement import CoherentMeasurementPlan
from merlin.perf.component_measurement_stream import RawMeasurementPlan, parse_measurement_stream
from merlin.runtime import commandbuffer as types_owner
from merlin.runtime.backends import base as parser_owner
from merlin.runtime.backends.base import decode_float_readback
from merlin.runtime.commandbuffer import declared_output_dtypes
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import native_component_execution as binding_owner
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.execution_service import FunctionalExecutionService
from merlin.targetgen.native_component_execution import _bind, _tree, execute_component

from . import component_decode_products as decode_owner


def _selection(plan, service, policy):
    if type(plan) is RawMeasurementPlan:
        if policy.transport != RB.FULL_VALUES_B64:
            raise ValueError("raw measurement accounting requires selected complete text full-value readback")
    elif type(plan) is CoherentMeasurementPlan:
        if (
            policy.transport != RB.COHERENT_DUMP_V1
            or service.process_transport is None
            or service.process_transport.prepared_readback is None
            or not _same_json(service.process_transport.prepared_readback.record(), plan.readback.record())
        ):
            raise ValueError("coherent measurement requires the same explicitly selected prepared process plan")
    else:
        raise TypeError("measurement accounting requires its explicit raw observation plan")
    plan.record()


def _member(path, *, owner=None):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
        or owner is not None
        and not path.is_relative_to(owner)
    ):
        raise ValueError("measurement accounting requires canonical original product membership")
    return {"path": str(path), "sha256": file_digest(path)}


def _bounded_bytes(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("measurement accounting product exceeds its selected byte budget")
    return data


def _environments(selection):
    if type(selection) is not tuple or not 1 <= len(selection) <= 64:
        raise ValueError("measurement accounting requires explicit full native environments")
    environments = {}
    metadata_bytes = 0
    for row in selection:
        if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str or row[0] in environments:
            raise ValueError("measurement accounting has ambiguous native environment membership")
        stage, values = row
        if (
            type(values) is not tuple
            or len(values) > 64
            or any(
                type(pair) is not tuple or len(pair) != 2 or any(type(value) is not str for value in pair)
                for pair in values
            )
            or len(dict(values)) != len(values)
        ):
            raise ValueError("measurement accounting has unsupported native environment values")
        if len(stage) > 128 or any(len(key) > 256 or len(value) > 8192 for key, value in values):
            raise ValueError("measurement accounting native environment exceeds its metadata budget")
        metadata_bytes += sum(len(key.encode()) + len(value.encode()) for key, value in values)
        if metadata_bytes > 1024 * 1024:
            raise ValueError("measurement accounting aggregate native environments exceed their metadata budget")
        I.environment_identity(dict(values))
        environments[stage] = dict(values)
    return environments


def _same_json(left, right):
    # Serialized scalar identity also rejects bool/int and int/float substitution.
    options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    return json.dumps(left, **options) == json.dumps(right, **options)


def collect_component_measurement(
    *, result_path, plan, build_service, execution_service, native_environments, original_inputs
):
    """Reopen fixed ordinary products and the live actual process consumption."""
    if type(plan) not in (RawMeasurementPlan, CoherentMeasurementPlan):
        raise TypeError("measurement accounting requires its explicit raw observation plan")
    plan.record()
    environments = _environments(native_environments)
    if type(build_service) is not BuildOnlyService or type(execution_service) is not FunctionalExecutionService:
        raise TypeError("measurement accounting requires exact ordinary build and functional services")
    if execution_service.process_transport is None:
        raise ValueError("measurement accounting requires actual selected recorded process consumption")
    result_path = Path(result_path)
    result_pin = _member(result_path)
    root = result_path.parent
    from merlin.common.strict_json import loads

    result = loads(_bounded_bytes(result_path, 8 * 1024 * 1024).decode("utf-8"))
    target = result["target"]
    build_service.verify(target)
    service = execution_service.verify(target, execution_service.simulator)
    if result.get("status") != "numeric_match_diagnostic" or result["numeric_report"].get("status") != "pass":
        raise ValueError("measurement accounting requires the original complete numerical output gate")
    if type(original_inputs) is not dict or set(original_inputs) != {"package", "capsule", "contract"}:
        raise ValueError("measurement accounting requires the original complete input tree selection")
    if not _same_json(result["inputs"], original_inputs):
        raise ValueError("measurement accounting original input selection differs from returned metadata")
    pins, owners = [result_pin], {}
    for name in ("package", "capsule", "contract"):
        original = result["inputs"][name]
        paths = [Path(row["path"]) for row in original.values()]
        if not paths:
            raise ValueError("measurement accounting omitted original input membership")
        # The ordinary record stores relative paths, preserving complete tree membership.
        first = next(iter(original))
        owner = paths[0]
        for _ in Path(first).parts:
            owner = owner.parent
        if _tree(owner) != original:
            raise ValueError("measurement accounting original source or compiler membership changed")
        pins.extend(_member(path, owner=owner) for path in paths)
        owners[name] = owner
    for row in result["emission"].values():
        pin = _member(row["path"])
        if pin["sha256"] != row["sha256"]:
            raise ValueError("measurement accounting emitted source product changed")
        pins.append(pin)
    elf = _member(result["elf"]["path"], owner=root)
    console_pin = _member(result["console"]["path"], owner=root)
    if any(pin["sha256"] != result[name]["sha256"] for name, pin in (("elf", elf), ("console", console_pin))):
        raise ValueError("measurement accounting actual ELF or console changed")
    console_path = Path(console_pin["path"])
    console = _bounded_bytes(console_path, plan.max_console_bytes)
    consumed = execution_service.consumption(elf=Path(elf["path"]), console=console)
    records, actuals, required_environments = [], [], set()
    # Validate containment before reading a newly introduced alias/record.
    for path in root.rglob("invocation.json"):
        if len(records) >= 512:
            raise ValueError("measurement accounting invocation roster exceeds its bound")
        pin = _member(path, owner=root)
        _bounded_bytes(path, 4 * 1024 * 1024)
        actual = I.verify(path)
        if actual["kind"] == "subprocess":
            stage = actual["stage"]
            if stage not in environments:
                raise ValueError("measurement accounting omitted an actual native environment")
            I.require_environment(path, environment=environments[stage])
            required_environments.add(stage)
        actuals.append(actual)
        records.append(pin)
    if required_environments != set(environments):
        raise ValueError("measurement accounting native environment roster differs from actual execution")
    declared = [{"path": row["record"]["path"], "sha256": row["record"]["sha256"]} for row in result["invocations"]]
    if sorted(records, key=lambda row: row["path"]) != sorted(declared, key=lambda row: row["path"]):
        raise ValueError("measurement accounting ordinary invocation membership changed")
    source = next(pin for pin in pins if pin["path"] == result["emission"]["source_interface"]["path"])
    lowered = next(pin for pin in pins if pin["path"] == result["emission"]["lowered_mlir"]["path"])
    source_calls = [row for row in actuals if row["stage"] == "component_source_lowering" and source in row["inputs"]]
    native_calls = [row for row in actuals if row["stage"] == "component_native_execution" and lowered in row["inputs"]]
    linked = [row for row in actuals if row["kind"] == "subprocess" and row["stage"] == "elf" and elf in row["outputs"]]
    selected_sources = [
        {"path": path, "sha256": digest}
        for path, digest in (*build_service.source_pins, *execution_service.source_pins)
    ]
    if (
        len(source_calls) != 1
        or len(native_calls) != 1
        or len(linked) != 1
        or any(
            _member(result["emission"][name]["path"]) not in source_calls[0]["outputs"]
            for name in ("command_buffer", "target_mlir", "lowered_mlir")
        )
        or any(
            _member(result["emission"][name]["path"]) not in native_calls[0]["inputs"]
            for name in ("bound_command_buffer", "input_projection")
        )
        or any(pin not in native_calls[0]["dependencies"] for pin in selected_sources)
    ):
        raise ValueError("measurement accounting lacks the actual selected source/build/linked ELF join")
    translations = [row for row in actuals if row["stage"] == "llvm_translation"]
    objects = [row for row in actuals if row["stage"] == "object"]
    if (
        len(translations) != 1
        or len(translations[0]["inputs"]) != 1
        or translations[0]["inputs"][0]["sha256"] != lowered["sha256"]
        or _member(translations[0]["inputs"][0]["path"], owner=root) != translations[0]["inputs"][0]
        or len(objects) != 1
        or not any(pin in objects[0]["inputs"] for pin in translations[0]["outputs"])
        or not any(pin in linked[0]["inputs"] for pin in objects[0]["outputs"])
    ):
        raise ValueError("measurement accounting lacks actual emitted LLVM/object/linked ELF correspondence")
    for support in build_service.recipe.support_sources:
        pin = _member(support)
        if support.suffix in (".c", ".S", ".s"):
            producers = [row for row in actuals if row["stage"] == "harness_object" and pin in row["inputs"]]
            if len(producers) != 1 or not any(product in linked[0]["inputs"] for product in producers[0]["outputs"]):
                raise ValueError("measurement accounting selected support was not actually compiled and linked")
        elif pin not in linked[0]["inputs"]:
            raise ValueError("measurement accounting selected support was not consumed by the actual linker")
    policy = RB.ReadbackPolicy.from_record(result["readback_policy"])
    _selection(plan, execution_service, policy)
    # Replay the original ordinary binding and selected parser; saved output/report
    # fields cannot substitute for the actual process stream or original answers.
    capsule = CC.load_capsule(owners["capsule"], contract=owners["contract"])
    source_path = owners["capsule"] / capsule.get("interface_mlir", "capsule.interface.mlir")
    if _member(source_path, owner=owners["capsule"]) != source:
        raise ValueError("measurement accounting changed the original source interface")
    cb = loads(_bounded_bytes(result["emission"]["command_buffer"]["path"], 8 * 1024 * 1024))
    bound, inputs, bindings = _bind(capsule, cb, source_path)
    actual_bound = loads(_bounded_bytes(result["emission"]["bound_command_buffer"]["path"], 8 * 1024 * 1024))
    actual_projection = loads(_bounded_bytes(result["emission"]["input_projection"]["path"], 8 * 1024 * 1024))
    if not _same_json(bound, actual_bound) or not _same_json(
        {"inputs": inputs, "bindings": bindings}, actual_projection
    ):
        raise ValueError("measurement accounting original complete input/ABI projection changed")
    text = console.decode("utf-8")
    parsed, metrics = execution_service.parse_output(text)
    coherent = None
    if type(plan) is CoherentMeasurementPlan:
        joined = decode_owner.join_component_decode_products(result_path=result_path, execution_root=root)
        decoder = joined.record()
        prepared = consumed.get("prepared_readback")
        if (
            type(prepared) is not dict
            or decoder["elf"] != elf
            or decoder["payload"] != prepared["output"]
            or prepared["objects"] != [{"symbol": name, "bytes": size} for name, size in plan.readback.bind(bound)]
            or prepared["payload_bytes"] != prepared["product_bytes"]
        ):
            raise ValueError("coherent measurement decoder differs from the actual complete prepared output")
        payload_pin = _member(decoder["payload"]["path"], owner=root)
        payload = _bounded_bytes(payload_pin["path"], plan.readback.max_payload_bytes)
        coherent = plan.decode(payload, cb=bound, inputs=inputs)
        parsed = coherent["outputs"]
        RB.require_memory_value_roster(bound, parsed)
        pins.extend(_member(decoder[name]["path"], owner=root) for name in ("decoder_product", "payload"))
        pins.append(_member(prepared["request"]["path"], owner=root))
        coherent["decoder_products"] = decoder
    else:
        RB.require_full_value_roster(bound, text, parsed, policy=policy)
    RB.require_current_build_receipt(
        cb=bound,
        target=target,
        workdir=root / "build",
        elf_path=Path(elf["path"]),
        policy=policy,
        build_service=build_service,
    )
    parsed = decode_float_readback(parsed, declared_output_dtypes(bound))
    if not _same_json(parsed, result["native"]["outputs"]) or not _same_json(metrics, result["native"]["raw_metrics"]):
        raise ValueError("measurement accounting returned native outputs differ from actual full stream parsing")
    expected = CG.golden(capsule, owners["capsule"])
    if set(expected) != set(bindings["outputs"]):
        raise ValueError("measurement accounting original reference output roster is incomplete")
    numeric_policy = capsule.get("numeric_policy")
    if type(numeric_policy) is not dict or numeric_policy.get("compare") not in ("exact_int", "tolerance_float"):
        raise ValueError("measurement accounting original numerical policy is unavailable")
    report = CG.compare(
        expected,
        {source: parsed[emitted] for source, emitted in bindings["outputs"].items()},
        numeric_policy,
        golden_source=CG.golden_source(capsule, owners["capsule"]),
    )
    if report["status"] != "pass" or not _same_json(report, result["numeric_report"]):
        raise ValueError("measurement accounting returned numerical report differs from original complete comparison")
    if coherent is None:
        events = parse_measurement_stream(console, plan=plan)
    else:
        reports = [
            CG.compare(
                expected,
                {name: values[emitted] for name, emitted in bindings["outputs"].items()},
                numeric_policy,
                golden_source=CG.golden_source(capsule, owners["capsule"]),
            )
            for values in coherent["call_outputs"]
        ]
        if any(row["status"] != "pass" for row in reports):
            raise ValueError("coherent measurement failed an original complete per-call numerical gate")
        coherent["per_call_numeric_reports"] = reports
        events = coherent
    # Reopen actual parsing/output invocation pins; raw returned text cannot substitute.
    build_service.verify(target)
    if execution_service.verify(target, execution_service.simulator) != service:
        raise ValueError("measurement accounting functional selection changed")
    if execution_service.consumption(elf=Path(elf["path"]), console=console) != consumed:
        raise ValueError("measurement accounting actual consumption changed")
    for pin in (*pins, elf, console_pin, *records):
        if _member(pin["path"]) != pin:
            raise ValueError("measurement accounting source or native product changed during collection")
    if any(_tree(owner) != original_inputs[name] for name, owner in owners.items()):
        raise ValueError("measurement accounting original input tree changed during output replay")
    return {
        "schema": "merlin.component_measurement_execution.v1"
        if coherent is None
        else "merlin.component_measurement_execution.v2",
        "ordinary_result": result_pin,
        "source_and_product_pins": pins,
        "ordinary_invocations": records,
        "elf": elf,
        "console": console_pin,
        "actual_consumption": consumed,
        "native_environments": {stage: I.environment_identity(values) for stage, values in environments.items()},
        "consumption_environment": I.environment_identity(dict(execution_service.process_transport.environment)),
        "numeric_report": report,
        "replayed_outputs": parsed,
        "original_input_bindings": bindings,
        "raw_events": events,
        "scope": "fresh per-arm full-output and raw-event attribution only; no stage/timer/measurement authority",
    }


def execute_component_measurement(*, plan, out_dir, native_environments, **ordinary_arguments):
    """Run one fresh arm through the existing ordinary compiler/runtime path."""
    if type(plan) not in (RawMeasurementPlan, CoherentMeasurementPlan):
        raise TypeError("measurement execution requires an explicit raw plan")
    plan.record()
    _environments(native_environments)
    policy = ordinary_arguments["readback_policy"]
    if type(policy) is not RB.ReadbackPolicy:
        raise ValueError("raw measurement execution requires selected complete text full-value readback")
    output = Path(out_dir)
    if (
        not output.is_absolute()
        or output.resolve() != output
        or any(part.is_symlink() for part in (output, *output.parents))
        or output.exists()
    ):
        raise ValueError("measurement execution requires a fresh canonical destination")
    service = ordinary_arguments["execution_service"]
    if type(service) is not FunctionalExecutionService or service.process_transport is None:
        raise ValueError("measurement execution requires actual selected recorded process consumption")
    _selection(plan, service, policy)
    for name in ("package_dir", "capsule_dir", "contract_root"):
        original = Path(ordinary_arguments[name])
        if output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError("measurement destination overlaps original source or compiler membership")
    original_inputs = {
        name: _tree(Path(ordinary_arguments[argument]))
        for name, argument in (("package", "package_dir"), ("capsule", "capsule_dir"), ("contract", "contract_root"))
    }
    output.mkdir(parents=True, mode=0o700)
    result = execute_component(out_dir=output / "ordinary", **ordinary_arguments)
    product = output / "raw_measurement.json"
    inputs = tuple(Path(row["path"]) for row in (result["elf"], result["console"]))
    selected_environments = _environments(native_environments)
    arguments = {
        "plan": plan.record(),
        "original_input_trees_sha256": RB.canonical_sha256(original_inputs),
        "native_environments": {
            stage: I.environment_identity(values) for stage, values in selected_environments.items()
        },
        "consumption_environment": I.environment_identity(dict(service.process_transport.environment)),
        "build_selection_sha256": RB.canonical_sha256(ordinary_arguments["build_service"].verify(result["target"])),
        "functional_selection_sha256": RB.canonical_sha256(service.verify(result["target"], service.simulator)),
    }
    fixed_owners = (stream_owner, json_owner, parser_owner, types_owner, CC, CG, binding_owner, RB)
    extra_inputs, extra_dependencies = (), ()
    if type(plan) is CoherentMeasurementPlan:
        prepared = service.consumption(
            elf=Path(result["elf"]["path"]), console=_bounded_bytes(result["console"]["path"], plan.max_console_bytes)
        )["prepared_readback"]
        joined = decode_owner.join_component_decode_products(
            result_path=output / "ordinary/result.json", execution_root=output / "ordinary"
        )
        extra_inputs = (
            Path(prepared["request"]["path"]),
            joined.payload[0],
            joined.decoder_product[0],
            joined.decoder_record[0],
        )
        extra_dependencies = (Path(coherent_owner.__file__), Path(decode_owner.__file__), *plan.readback.source_paths())
    with I.observe_call(
        output / "accounting",
        stage="component_raw_measurement_accounting",
        function=collect_component_measurement,
        arguments=arguments,
        inputs=(
            output / "ordinary/result.json",
            *inputs,
            *(Path(row["path"]) for tree in original_inputs.values() for row in tree.values()),
            *(Path(row["path"]) for row in result["emission"].values()),
            *extra_inputs,
        ),
        outputs=(product,),
        dependencies=(
            Path(__file__),
            *(Path(owner.__file__) for owner in fixed_owners),
            *(Path(path) for path, _ in (*ordinary_arguments["build_service"].source_pins, *service.source_pins)),
            *extra_dependencies,
        ),
    ) as invocation:
        observed = collect_component_measurement(
            result_path=output / "ordinary/result.json",
            plan=plan,
            build_service=ordinary_arguments["build_service"],
            execution_service=service,
            native_environments=native_environments,
            original_inputs=original_inputs,
        )
        with product.open("x", encoding="utf-8") as stream:
            json.dump(observed, stream, sort_keys=True, indent=2)
            stream.write("\n")
        invocation.returned()
    return observed
