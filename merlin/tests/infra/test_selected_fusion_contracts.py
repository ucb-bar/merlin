"""Selected facts/software fusion through complete two-layer native graphs.

Public neutral test machines are independent execution evidence for this
mechanism, never qualification of a hardware target or a reference backend.
"""

import hashlib
import json
import time
from dataclasses import replace

import pytest
import yaml
from test_readout_model_fusion import EPILOGUE, IDENTITY, KEYS, OPS, SPACES, WORD, graph, inputs_and_reference, params
from test_readout_model_fusion import native as native

from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.sched.ir import Call, Const
from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction


def inputs(tmp_path, *, shape=(3, 7, 5, 4), width=1, separate=False, mutate=None):
    original = graph(*shape, width, separate)
    kernel, objects, banks, adapter, cap, codec, limits, model = original
    facts = dict(facts=dict(target="neutral", word_bits=8 * WORD.size, instruction_bits=[8] + [32] * len(KEYS),
                            offsets=[0] + [8 + 32 * i for i in range(len(KEYS))], codes=list(range(len(OPS))),
                            selectors=list(range(len(SPACES))), local_capacity=1024, external_capacity=1024,
                            zero=0, alignment=4, byte_bits=8, accumulator_bits=32, max_elements=64))

    def ref(*parts):
        return {"fact": ["facts", *parts]}

    def arg(name):
        return {"argument": name}

    def selector(name):
        return dict(argument=name, choices=[dict(value=ref("selectors", i), resource=space)
                                            for i, space in enumerate(SPACES)
                                            if name not in ("weights", "bias") or i >= 3])

    def access(space, start, dims, bits, writes=False):
        return dict(space=selector(space), start=start,
                    size=dict(elements=[arg(dim) for dim in dims], element_bits=ref(bits)), writes=writes)

    declarations, instructions = {}, []
    for code, op in enumerate(OPS):
        fields = [dict(offset=ref("offsets", i), width=ref("instruction_bits", i),
                       **(dict(fixed=ref("codes", code)) if i == 0 else dict(argument=key)))
                  for i, key in enumerate(("opcode", *KEYS))]
        operands = [dict(name=key, type=dict(kind="integer", bits=ref("instruction_bits", i + 1), signed=False),
                         unit="integer") for i, key in enumerate(KEYS)]
        if op == "dot":
            accesses = [access("src", arg("source"), ("m", "k", "width"), "byte_bits"),
                        access("weights", {"constant": 0}, ("k", "n", "width"), "byte_bits"),
                        access("dst", arg("destination"), ("m", "n"), "accumulator_bits", True)]
        else:
            accesses = [access("src", arg("source"), ("m", "n"),
                               "byte_bits" if op == "snapshot" else "accumulator_bits")]
            if op in ("finish", "fused"):
                accesses += [access("bias", {"constant": 0}, ("m", "n"), "accumulator_bits")]
            accesses += [access("dst", arg("destination"), ("m", "n"),
                                "accumulator_bits" if op == "readout" else "byte_bits", True)]
        declarations[op] = dict(accesses=accesses, completion="synchronous", token=None, waits=[],
                                exact_copy=op == "snapshot")
        instructions.append(dict(name=op, operands=operands, results=[], immediates=[], encoding=fields,
                                 semantics=dict(primitive="neutral." + op, bindings={key: key for key in KEYS}),
                                 effects=dict(reads=list(SPACES), writes=list(SPACES))))
    memory = [dict(name=space, kind="memory", unit="bytes",
                   capacity=ref("local_capacity" if i < 3 else "external_capacity")) for i, space in enumerate(SPACES)]
    storage = dict(schema="merlin.temporary_storage_contract.v1",
                   banks={b.name: dict(backing=b.backing, start=ref("zero"), size=ref("local_capacity"),
                                       alignment=ref("alignment"), ownership="invocation_private") for b in banks},
                   external={name: dict(backing=name + "_memory", start=ref("zero"), size=ref("external_capacity"),
                                         ownership="runtime_managed") for name in SPACES[3:]},
                   reservations=[], instructions=declarations)
    elementwise = {}
    for name in ("readout", "finish", "fused"):
        source = IDENTITY if name == "readout" else EPILOGUE
        stages = (["readout"] if name in ("readout", "fused") else [])
        if name != "readout":
            stages += ["bias", "clamp", "cast"]
        widths = [ref("accumulator_bits")] * 2 + ([] if name == "readout" else [ref("byte_bits")])
        elementwise[name] = dict(scalar_ir=source.decode(), elements=[arg("m"), arg("n")],
                                 element_bits=widths, stages=stages, byte_orders=["little"] * len(widths),
                                 readout=cap.readout.selector if name == "fused" else None,
                                 execution="contiguous_read_then_write")
    fusion = dict(schema="merlin.physical_fusion_contract.v1", elementwise=elementwise,
                  readouts={cap.readout.selector: dict(applies=sorted(cap.readout.applies),
                      evidence="independent neutral native primitive declaration", max_elements=ref("max_elements"),
                      routes=[dict(producer="readout", consumer="finish", result="fused")])})
    software = dict(schema="merlin.software_spec.v1", target="neutral", status="reviewed", isa=dict(
        schema="merlin.isa_spec.v1", namespace="neutral_graph", facts_sha256="0" * 64,
        word_bits=ref("word_bits"), resources=memory, instructions=instructions), scheduling=dict(
            schema="merlin.schedule_effects.v1", numerical_policy=adapter.numerical_policy_identity,
            instruction_byteorder="little", configuration={}, storage=storage, fusion=fusion))
    if mutate is not None:
        mutate(facts["facts"], software, fusion)
    fact_path, software_path = tmp_path / "facts.json", tmp_path / "software.yaml"
    fact_path.write_text(json.dumps(facts))
    software["isa"]["facts_sha256"] = hashlib.sha256(fact_path.read_bytes()).hexdigest()
    software_path.write_text(yaml.safe_dump(software))
    selected = read_scheduling_contracts(fact_path, software_path, target="neutral")
    adapter = replace(adapter, configuration_identity=selected.hardware_identity)
    def encode(kernel):
        return b"".join(encode_instruction(selected.isa, call.instr, params(call)).to_bytes(WORD.size, "little")
                        for call in kernel.body)
    def decode(image):
        rows = []
        for offset in range(0, len(image), WORD.size):
            name, values = decode_instruction(selected.isa, int.from_bytes(image[offset:offset + WORD.size], "little"))
            rows.append(Call(name, tuple((key, Const(values[key])) for key in KEYS)))
        return tuple(rows)
    codec = replace(codec, encode=encode, decode_calls=decode,
                    program_decoder=replace(codec.program_decoder, configuration_identity=selected.hardware_identity))
    return selected, kernel, objects, adapter, cap, codec, limits, model


def compile_case(model, *, enabled=True, qualified=True, emitters=True, adapter=None, budgets=None):
    selected, kernel, objects, original, cap, codec, limits, options = model
    return compile_schedule(
        kernel, objects, None, original if adapter is None else adapter, codec, limits,
        budgets=budgets or ScheduleBudgets(max_fusions=8 if enabled else 0), selected_contracts=selected,
        fusion_emitters={cap.readout.selector: cap.emit} if emitters is True else emitters if emitters else None,
        fusion_qualification_identity="independent-neutral-two-layer-v1" if qualified else None, **options,
    )


@pytest.mark.parametrize("shape,width,separate", [((3, 7, 5, 4), 1, False), ((2, 5, 11, 3), 2, True)])
def test_selected_fusion_executes_complete_graph_and_checks_realized_transfers(
    tmp_path, native, record_property, shape, width, separate
):
    model = inputs(tmp_path, shape=shape, width=width, separate=separate)
    data, expected = inputs_and_reference(*shape, width)
    results, measurements = [], []
    for enabled in (False, True):
        started = time.perf_counter()
        result = compile_case(model, enabled=enabled)
        seconds = time.perf_counter() - started
        started = time.perf_counter()
        actual, counters = native(result, data, separate)
        call_seconds = time.perf_counter() - started
        assert actual == expected
        report = result.report["final_conformance"]
        assert counters == (report["actual_commands"], report["requested_read_bytes"], report["requested_write_bytes"])
        results.append(result)
        measurements.append(dict(arm="optimized" if enabled else "baseline", compile_seconds=seconds,
                                 encoded_bytes=len(b"".join(result.program.segments)), commands=counters[0],
                                 read_bytes=counters[1], write_bytes=counters[2],
                                 native_host_call_seconds=call_seconds))
    assert results[0].report["context"]["library"] == results[1].report["context"]["library"]
    assert measurements[0]["commands"] == 7 and measurements[1]["commands"] == 5
    assert [c.instr for c in results[1].kernel.body] == ["dot", "fused", "snapshot", "dot", "fused"]
    assert len(results[1].report["storage"]["fusion"]["checked_steps"]) == 2
    m, _, n, p = shape
    assert measurements[0]["read_bytes"] - measurements[1]["read_bytes"] == 4 * m * (n + p)
    assert measurements[0]["write_bytes"] - measurements[1]["write_bytes"] == 4 * m * (n + p)
    actual, _ = native(results[1], data, separate, wrong=True)
    assert actual != expected  # Falsely qualified implementation skips i32 wrapping.
    record_property("selected_fusion_pair", json.dumps(dict(
        scope="complete neutral two-layer integer graph; no real target/model or accelerator timing qualification",
        shape=shape, element_input_bytes=width, separate_memories=separate,
        selected_contracts=results[1].report["selected_contracts"],
        shared_library_sources=results[1].report["library_sources_sha256"], measurements=measurements,
        compile_peak_memory="unmeasured", accelerator_timing="unmeasured",
    ), sort_keys=True))


@pytest.mark.parametrize("missing", ["qualification", "emitter", "capacity", "stages"])
def test_missing_capabilities_keep_the_original_materializations(tmp_path, native, missing):
    def mutate(facts, software, fusion):
        if missing == "capacity":
            fusion["readouts"]["physical-finish"]["max_elements"] = None
        if missing == "stages":
            fusion["readouts"]["physical-finish"]["applies"].remove("bias")
    model = inputs(tmp_path, mutate=mutate)
    result = compile_case(model, qualified=missing != "qualification", emitters=missing != "emitter")
    data, expected = inputs_and_reference(3, 7, 5, 4, 1)
    assert native(result, data, False)[0] == expected
    assert [c.instr for c in result.kernel.body].count("readout") == 2
    assert result.report["final_conformance"]["actual_commands"] == 7


def test_rtl_readout_capacity_boundary_selects_only_legal_pairs(tmp_path):
    for maximum, expected in ((11, 7), (12, 6), (15, 5)):
        model = inputs(tmp_path, mutate=lambda facts, sw, fusion: facts.update(max_elements=maximum))
        result = compile_case(model)
        assert result.report["final_conformance"]["actual_commands"] == expected


def test_incompatible_projection_or_physical_result_cannot_claim_selected_fusion(tmp_path):
    model = inputs(tmp_path)
    adapter = model[3]
    def wrong(call):
        effect = adapter.project(call)
        if effect.elementwise is not None:
            effect = replace(effect, elementwise=replace(effect.elementwise,
                                                         byte_orders=("big",) * len(effect.accesses)))
        return effect
    with pytest.raises(ValueError, match="projection disagrees"):
        compile_case(model, adapter=replace(adapter, project=wrong))
    def wrong_instruction(producer, consumer):
        call = model[4].emit(producer, consumer)
        return replace(call, instr="finish")
    with pytest.raises(ValueError, match="undeclared instruction composition"):
        compile_case(model, emitters={model[4].readout.selector: wrong_instruction})


def test_selected_semantics_work_without_duplicate_provider_scalar_ir(tmp_path, native):
    model = inputs(tmp_path)
    provider = model[3]
    adapter = replace(provider, project=lambda call: replace(provider.project(call), elementwise=None))
    result = compile_case(model, adapter=adapter)
    data, expected = inputs_and_reference(3, 7, 5, 4, 1)
    assert native(result, data, False)[0] == expected
    assert result.report["final_conformance"]["actual_commands"] == 5


def test_recurrent_graph_invocations_preserve_complete_state_and_mutable_weights(tmp_path, native):
    import numpy as np
    from test_readout_model_fusion import reference_outputs
    shape = (2, 5, 5, 3)
    model = inputs(tmp_path, shape=shape)
    baseline, optimized = compile_case(model, enabled=False), compile_case(model)
    initial, _ = inputs_and_reference(*shape, 1)
    state = initial["x"].copy()
    for invocation in range(3):
        data = {name: value.copy() for name, value in initial.items()}
        data["x"] = state.copy()
        data["w0"].flat[0] += invocation
        data["b0"].flat[-1] = invocation - 1
        expected = reference_outputs(data)
        actual = [native(result, data, False)[0] for result in (baseline, optimized)]
        assert actual[0] == actual[1] == expected
        state = np.frombuffer(actual[1]["hidden"], dtype="i1").reshape(shape[0], shape[1]).copy()


@pytest.mark.parametrize("fault,diagnostic", [
    ("literal_width", "fact"), ("width", "widths disagree"), ("unencoded", "encoded integer"),
    ("route", "selected scalar instructions"), ("stage_order", "ordered source stages"),
    ("readout", "one readout"), ("word_order", "byte order"), ("bits", "byte representation"),
    ("rank", "sixteen"), ("zero", "positive"), ("nonmemory", "memory-only"),
])
def test_mutated_selected_semantics_and_resources_refuse(tmp_path, fault, diagnostic):
    def mutate(facts, software, fusion):
        row = fusion["elementwise"]["fused"]
        if fault == "literal_width":
            row["element_bits"][0] = 32
        elif fault == "width":
            row["element_bits"][0] = {"fact": ["facts", "byte_bits"]}
        elif fault == "unencoded":
            row["elements"] = [{"argument": "missing"}]
        elif fault == "route":
            fusion["readouts"]["physical-finish"]["routes"][0]["result"] = "dot"
        elif fault == "stage_order":
            row["stages"] = list(reversed(row["stages"]))
        elif fault == "readout":
            row["readout"] = "different_readout"
        elif fault == "word_order":
            row["byte_orders"][0] = "unknown"
        elif fault == "bits":
            facts["accumulator_bits"] = 31
        elif fault == "rank":
            row["elements"] = [{"argument": "m"}] * 17
        elif fault == "zero":
            row["elements"] = [{"constant": 0}]
        elif fault == "nonmemory":
            software["isa"]["resources"].append(dict(name="engine", kind="execution", unit="events"))
            software["isa"]["instructions"][3]["effects"]["writes"].append("engine")
    with pytest.raises(ValueError, match=diagnostic):
        compile_case(inputs(tmp_path, mutate=mutate))


@pytest.mark.parametrize("unknown", ["execution", "floating", "budget"])
def test_unsupported_scalar_or_execution_policy_retains_original_route(tmp_path, native, unknown):
    def mutate(facts, software, fusion):
        if unknown == "execution":
            fusion["elementwise"]["fused"]["execution"] = "unknown"
        if unknown == "floating":
            fusion["elementwise"]["fused"]["scalar_ir"] = (
                "builtin.module {func.func @forward(%x:f32)->f32 {func.return %x:f32}}")
    model = inputs(tmp_path, mutate=mutate)
    budget = ScheduleBudgets(max_scalar_operations=1) if unknown == "budget" else None
    result = compile_case(model, budgets=budget)
    data, expected = inputs_and_reference(3, 7, 5, 4, 1)
    assert native(result, data, False)[0] == expected
    assert result.report["final_conformance"]["actual_commands"] == 7
    assert result.report["selected_contracts"]["fusion"]["unavailable_instructions"]


def test_one_readout_cannot_be_replaced_by_a_union_or_arbitrary_capability(tmp_path):
    model = inputs(tmp_path)
    selected, kernel, objects, adapter, cap, codec, limits, options = model
    with pytest.raises(ValueError, match="overridden"):
        compile_schedule(kernel, objects, None, adapter, codec, limits, fusion_capabilities=(cap,),
                         selected_contracts=selected, **options)
    with pytest.raises(ValueError, match="declared readouts"):
        compile_case(model, emitters={"union_of_readouts": cap.emit})
    def split(facts, software, fusion):
        readout = fusion["readouts"]["physical-finish"]
        readout["applies"] = ["readout", "bias"]
        # No individual physical capability applies the entire ordered chain.
    result = compile_case(inputs(tmp_path, mutate=split))
    assert result.report["final_conformance"]["actual_commands"] == 7


def test_stale_inputs_and_changes_during_physical_selection_reject(tmp_path):
    model = inputs(tmp_path)
    selected = model[0]
    def mutate(producer, consumer):
        selected.software_path.write_bytes(selected.software_bytes + b"\n")
        return model[4].emit(producer, consumer)
    with pytest.raises(ValueError, match="changed; reselect"):
        compile_case(model, emitters={model[4].readout.selector: mutate})
    with pytest.raises(ValueError, match="changed; reselect"):
        compile_case(model)


def test_actual_final_bytes_cannot_restore_materialization_or_omit_wrapping(tmp_path):
    model = list(inputs(tmp_path))
    original = model[5]
    def replace_fused(kernel):
        body = tuple(replace(call, instr="finish") if call.instr == "fused" else call for call in kernel.body)
        return original.encode(replace(kernel, body=body))
    model[5] = replace(original, encode=replace_fused)
    with pytest.raises(ValueError, match="actual emitted.*effects differ"):
        compile_case(model)
    # A matching provider claim cannot hide the changed actual opcode.
    def lies(image):
        calls = original.decode_calls(image)
        return tuple(replace(c, instr="fused") if c.instr == "finish" else c for c in calls)
    model[5] = replace(original, encode=replace_fused, decode_calls=lies)
    with pytest.raises(ValueError, match="disagrees with actual selected ISA words"):
        compile_case(model)


def test_failed_selected_fusion_allocation_restores_original_and_replans(tmp_path, native):
    def mutate(facts, software, fusion):
        facts.update(strict_capacity=32, strict_alignment=32, one=1)
        facts["selectors"].append(len(SPACES))
        def ref(key):
            return {"fact": ["facts", key]}
        software["isa"]["resources"].append(dict(name="strict_result", kind="memory", unit="bytes",
                                                   capacity=ref("strict_capacity")))
        storage = software["scheduling"]["storage"]
        storage["banks"]["strict_result"] = dict(backing="scratch", start=ref("zero"), size=ref("strict_capacity"),
                                                 alignment=ref("strict_alignment"), ownership="invocation_private")
        storage["reservations"] = [dict(resource="stage", start=ref("zero"), size=ref("one"), purpose="staging")]
        for instruction in software["isa"]["instructions"]:
            instruction["effects"]["writes"].append("strict_result")
            storage["instructions"][instruction["name"]]["accesses"][-1]["space"]["choices"].append(
                dict(value={"fact": ["facts", "selectors", len(SPACES)]}, resource="strict_result"))
    model = list(inputs(tmp_path, mutate=mutate))
    provider = model[3]
    def project(call):
        if params(call)["dst"] != len(SPACES):
            return provider.project(call)
        mapped = replace(call, args=tuple((k, Const(2) if k == "dst" else v) for k, v in call.args))
        effect = provider.project(mapped)
        return replace(effect, accesses=(*effect.accesses[:-1], replace(effect.accesses[-1], space="strict_result")))
    model[3] = replace(provider, project=project)
    def strict(producer, consumer):
        call = model[4].emit(producer, consumer)
        if params(call)["dst"] == 2:
            call = replace(call, args=tuple((k, Const(len(SPACES)) if k == "dst" else v) for k, v in call.args))
        return call
    emitters = {model[4].readout.selector: strict}
    result = compile_case(model, emitters=emitters)
    assert result.report["repair_rounds"] == 1
    assert result.report["storage"]["fusion"]["status"] == "allocation_fallback"
    assert all(call.instr != "fused" for call in result.kernel.body)
    data, expected = inputs_and_reference(3, 7, 5, 4, 1)
    assert native(result, data, False)[0] == expected
    with pytest.raises(ValueError, match="repair budget exhausted"):
        compile_case(model, emitters=emitters, budgets=ScheduleBudgets(max_repairs=0))


def test_scalar_equivalence_is_checked_from_selected_fused_output(tmp_path):
    def mutate(facts, software, fusion):
        row = fusion["elementwise"]["fused"]
        row["scalar_ir"] = row["scalar_ir"].replace("arith.addi", "arith.subi")
    model = inputs(tmp_path, mutate=mutate)
    provider = model[3]
    adapter = replace(provider, project=lambda call: replace(provider.project(call), elementwise=None))
    with pytest.raises(ValueError, match="actual fused arithmetic changes"):
        compile_case(model, adapter=adapter)


def test_previously_resolved_contracts_cannot_survive_resolver_changes(tmp_path, monkeypatch):
    from merlin.compile.scheduling import selected_contracts
    model = inputs(tmp_path)
    original = selected_contracts._resolver_sources
    monkeypatch.setattr(selected_contracts, "_resolver_sources", lambda: (*original(), ("modified", "0" * 64)))
    with pytest.raises(ValueError, match="resolver changed; reselect"):
        compile_case(model)


def test_resolver_changes_during_selection_are_rejected(tmp_path, monkeypatch):
    from merlin.compile.scheduling import selected_contracts
    sources, parse = selected_contracts._resolver_sources, selected_contracts._parse
    changed = False
    def current():
        return sources() + ((("modified", "0" * 64),) if changed else ())
    def mutate(*args):
        nonlocal changed
        result = parse(*args)
        changed = True
        return result
    monkeypatch.setattr(selected_contracts, "_resolver_sources", current)
    monkeypatch.setattr(selected_contracts, "_parse", mutate)
    with pytest.raises(ValueError, match="resolver changed during selection"):
        inputs(tmp_path)
