"""Selected complete-row semantics and resources through real native output."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, replace

import numpy as np
import pytest
import yaml
from test_row_softmax import (
    KINDS,
    NAMES,
    adapter_for,
    execute_native,
    exp_object,  # noqa: F401
    fixture,
    native_program,
)

from merlin.compile.scheduling.row_softmax import CausalRowMask, RowSoftmaxResources, RowSoftmaxUnavailable
from merlin.compile.scheduling.row_softmax import compile_row_softmax as compile_row
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.sched.ir import Const
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank


def inputs(tmp_path, *, dtype="f32", separate=True, columns=19, rows=5, chunk=6, mask=None,
           index_bits=64, mutate=None):
    opts = fixture(dtype=dtype, separate=separate, columns=columns, rows=rows, chunk=chunk, mask=mask)
    model, banks = opts["numerics"], opts["banks"]
    facts = dict(facts=dict(target="neutral", word_bits=8 + 12 * 64, field_bits=[8, *([64] * 12)],
                            offsets=[0, *(8 + 64 * i for i in range(12))], codes=list(range(7)),
                            selectors=list(range(len(banks))), capacities=[512, 512, 4096, 4096],
                            starts=[0, 0], sizes=[512, 512], alignments=[4, 8],
                            element_bits=32 if dtype == "f32" else 64, max_chunk=chunk,
                            granularity=1, index_bits=index_bits))
    def ref(*path):
        return {"fact": ["facts", *path]}
    role_banks = dict(zip(("maximum", "exponentials", "lanes", "normalizer"), opts["role_banks"]))
    memory = [dict(name=b.name, kind="memory", unit="bytes", capacity=ref("capacities", i))
              for i, b in enumerate(banks)]
    memory += [dict(name=name, kind="memory", unit="bytes", capacity=ref("capacities", 2 + i))
               for i, name in enumerate(("input", "output"))]
    # Effects are explicit software data. Native primitive execution below
    # observes its actual payloads independently of these resource sets.
    state, exps = role_banks["maximum"], role_banks["exponentials"]
    effects = [((), (state,)), (("input", state), (state,)), (("input", state), (exps,)),
               ((), (state,)), ((exps, state), (state,)), ((state,), (state,)),
               ((exps, state), ("output",))]
    local_roles = ((0,), (0,), (0, 1), (2,), (1, 2), (2, 3), (1, 3))
    instructions, phases = [], {}
    for phase, kind in enumerate(KINDS):
        name = "stage_" + str(phase)
        bindings = {key: key for key in ("row", "column", "count", "period", "offset")}
        if kind in ("maximum", "exponential"):
            bindings["source"] = "source"
        if kind == "normalize":
            bindings["output"] = "output"
        operands = [dict(name=key, type=dict(kind="address" if key in ("source", "output") else "integer",
                                           signed=False, bits=ref("field_bits", i + 1),
                                           **(dict(space="input" if key == "source" else "output")
                                              if key in ("source", "output") else {})),
                         unit="bytes" if key in ("source", "output", "address0", "address1") else "integer")
                    for i, key in enumerate(NAMES)]
        fields = [dict(offset=ref("offsets", 0), width=ref("field_bits", 0), fixed=ref("codes", phase))]
        fields += [dict(offset=ref("offsets", i + 1), width=ref("field_bits", i + 1), argument=key)
                   for i, key in enumerate(NAMES)]
        local, names = [], set(bindings.values())
        for i, role in enumerate(local_roles[phase]):
            bank = opts["role_banks"][role]
            bank_id = next(j for j, b in enumerate(banks) if b.name == bank)
            local.append(dict(address="address" + str(i),
                              selector=dict(argument="space" + str(i), value=ref("selectors", bank_id))))
            names.update(("address" + str(i), "space" + str(i)))
        constants = {key: None if key in ("source", "output") else 1 if key == "numeric" else 0
                     for key in NAMES if key not in names}
        phases[kind] = dict(instruction=name, bindings=bindings, local=local, constants=constants)
        instructions.append(dict(name=name, operands=operands, results=[], immediates=[], encoding=fields,
                                 semantics=dict(primitive="merlin.complete_row." + kind, bindings=bindings,
                                                parameters=dict(numerical_identity=model.identity,
                                                                completion="synchronous")),
                                 effects=dict(reads=list(dict.fromkeys(effects[phase][0])),
                                              writes=list(dict.fromkeys(effects[phase][1])))))
    storage = dict(schema="merlin.temporary_storage_contract.v1", banks={
        b.name: dict(backing=b.backing, start=ref("starts", i), size=ref("sizes", i),
                     alignment=ref("alignments", i), ownership="invocation_private") for i, b in enumerate(banks)},
        external={name: dict(backing=name + "_memory", start=ref("starts", 0), size=ref("capacities", 2 + i),
                             ownership="runtime_managed") for i, name in enumerate(("input", "output"))},
        reservations=[], instructions={})
    declaration = dict(schema="merlin.complete_row_contract.v1", algorithm=asdict(model),
                       element_bits=ref("element_bits"), limits={key: ref(fact) for key, fact in (
                           ("max_chunk_elements", "max_chunk"), ("granularity", "granularity"),
                           ("index_bits", "index_bits"))},
                       support=dict(partial_chunks=True, completion="synchronous", order="source",
                                    state="addressable", masks=["none", "causal_row"]), roles=role_banks,
                       external=dict(input="input", output="output"), instructions=phases)
    software = dict(schema="merlin.software_spec.v1", target="neutral", status="reviewed", isa=dict(
        schema="merlin.isa_spec.v1", namespace="rows_test", facts_sha256="0" * 64,
        word_bits=ref("word_bits"), resources=memory, instructions=instructions), scheduling=dict(
            schema="merlin.schedule_effects.v1", numerical_policy=model.source_policy,
            instruction_byteorder="little", configuration={}, storage=storage, row_reduction=declaration))
    if mutate:
        mutate(facts["facts"], software, declaration)
    fact_path, software_path = tmp_path / "facts.json", tmp_path / "software.yaml"
    fact_path.write_text(json.dumps(facts))
    software["isa"]["facts_sha256"] = hashlib.sha256(fact_path.read_bytes()).hexdigest()
    software_path.write_text(yaml.safe_dump(software))
    selected = read_scheduling_contracts(fact_path, software_path, target="neutral")
    windows = {w.resource: w for w in selected.storage.banks}
    banks = tuple(StorageBank(b.name, windows[b.name].backing, windows[b.name].start + windows[b.name].size,
                              "byte", windows[b.name].alignment, windows[b.name].start) for b in banks)
    opts["banks"] = banks
    opts["adapter"] = replace(adapter_for(model, banks), configuration_identity=selected.hardware_identity,
                               index_bits=index_bits)
    return selected, opts


def schedule(model, **changes):
    selected, opts = model
    controls = {**opts, "resources": None, "banks": None, "role_banks": None,
                "selected_contracts": selected, **changes}
    return compile_row(**controls)


def independent_reference(tmp_path, values, mask):
    if values.dtype == np.float64:
        result = []
        for row, score in enumerate(values):
            active = [float(x) if mask is None or c <= row % mask.period + mask.offset else -math.inf
                      for c, x in enumerate(score)]
            exps = [math.exp(x - max(active)) for x in active]
            total = exps[0]
            for value in exps[1:]:
                total += value
            result.append([value / total for value in exps])
        return np.array(result)
    from merlin.llvmlower import toolchain
    np.save(tmp_path / "source.npy", values)
    run = subprocess.run([str(toolchain.compiler_python()), "-c", """
import sys,torch,numpy as np
from pathlib import Path
if not torch.__version__.startswith('2.10.') or torch.backends.cpu.get_cpu_capability()!='AVX2':sys.exit(77)
p=Path(sys.argv[1]);x=torch.from_numpy(np.load(p/'source.npy'));period,offset=map(int,sys.argv[2:4])
if period:x=x.masked_fill(torch.arange(x.shape[1])[None,:]>torch.arange(x.shape[0])[:,None]%period+offset,float('-inf'))
np.save(p/'reference.npy',torch.softmax(x,-1).numpy())
""", str(tmp_path), str(mask.period if mask else 0), str(mask.offset if mask else 0)],
        env={**os.environ, "ATEN_CPU_CAPABILITY": "avx2"}, capture_output=True, text=True)
    if run.returncode == 77:
        pytest.skip("independent pinned Torch2.10 AVX2 unavailable")
    assert run.returncode == 0, run.stderr
    return np.load(tmp_path / "reference.npy")


@pytest.mark.parametrize("dtype,separate,index_bits,mask", [
    ("f32", False, 64, None), ("f32", True, 32, CausalRowMask(3, 2)),
    ("f64", True, 64, CausalRowMask(2)),
])
def test_selected_complete_rows_execute_full_native_outputs(
    tmp_path, exp_object, record_property, dtype, separate, index_bits, mask  # noqa: F811
):
    model = inputs(tmp_path, dtype=dtype, separate=separate, index_bits=index_bits, mask=mask)
    selected, opts = model
    results, times = [], []
    for chunk in (1, None):
        started = time.perf_counter()
        results.append(schedule(model, chunk_elements=chunk))
        times.append(time.perf_counter() - started)
    baseline, optimized = results
    versions = baseline.report["pipeline"]["library_sources_sha256"]
    assert versions == optimized.report["pipeline"]["library_sources_sha256"]
    assert optimized.report["pipeline"]["selected_contracts"]["rtl_facts"]["sha256"] == selected.hardware_identity[4:]
    values = np.random.default_rng(173).normal(0, 4, (5, 19)).astype("float32" if dtype == "f32" else "float64")
    values[1] = 1
    values[2] = np.linspace(-120, 0, 19)
    reference = independent_reference(tmp_path, values, mask)
    spaces = tuple(dict.fromkeys(bank.backing for bank in opts["banks"]))
    records, outputs = [], []
    for arm, result, elapsed in zip(("chunk_one", "default_chunk"), results, times):
        source = native_program(result, opts)
        output, counts = execute_native(tmp_path, source, values, exp_object, arm, spaces)
        for key, value in counts.items():
            assert result.report[key] == value
        outputs.append(output)
        records.append(dict(arm=arm, compile_seconds=elapsed, c_source_bytes=len(source.encode()), **counts))
        library = tmp_path / f"{arm}_{hashlib.sha256(source.encode()).hexdigest()[:16]}.so"
        records[-1]["native_library_bytes"] = library.stat().st_size
        if shutil.which("size"):
            sections = subprocess.run(["size", "-A", "-d", str(library)], check=True,
                                      capture_output=True, text=True).stdout.splitlines()
            records[-1]["native_text_bytes"] = sum(int(line.split()[1]) for line in sections
                                                   if line.split() and line.split()[0] == ".text")
    words = "u4" if dtype == "f32" else "u8"
    np.testing.assert_array_equal(outputs[0].view(words), reference.view(words))
    np.testing.assert_array_equal(outputs[1].view(words), reference.view(words))
    assert records[1]["executed_commands"] < records[0]["executed_commands"]
    assert records[1]["requested_read_bytes"] < records[0]["requested_read_bytes"]
    record_property("selected_row_pair", json.dumps(dict(
        dtype=dtype, separate=separate, index_bits=index_bits, measurements=records,
        selected_contracts=optimized.report["pipeline"]["selected_contracts"],
        shared_library_sources=optimized.report["pipeline"]["library_sources_sha256"],
        accelerator_timing="unmeasured", compile_peak_memory="unmeasured",
        scope="complete neutral rows with observed native calls; no target or machine-byte qualification")))


@pytest.mark.parametrize("fault", ["width", "effects", "bindings", "constants", "algorithm", "role", "unknown"])
def test_contradictory_selected_row_declarations_refuse(tmp_path, fault):
    def mutate(facts, software, row):
        if fault == "width":
            facts["element_bits"] = 64
        elif fault == "effects":
            software["isa"]["instructions"][2]["effects"]["reads"].remove("input")
        elif fault == "bindings":
            row["instructions"]["maximum"]["bindings"]["count"] = "row"
        elif fault == "constants":
            row["instructions"]["maximum"]["constants"]["numeric"] = -1
        elif fault == "algorithm":
            row["algorithm"]["normalization"] = "divide"
        elif fault == "role":
            row["roles"]["lanes"] = "missing"
        else:
            row["instructions"]["sum"]["instruction"] = "unknown"
    with pytest.raises(ValueError):
        inputs(tmp_path, mutate=mutate)


@pytest.mark.parametrize("fault", ["completion", "order", "state", "mask", "limit", "capacity", "ownership"])
def test_unsupported_selected_row_contracts_keep_source_route(tmp_path, fault):
    def mutate(facts, software, row):
        if fault in ("completion", "order", "state"):
            row["support"][fault] = "unknown"
            if fault == "completion":
                for inst in software["isa"]["instructions"]:
                    inst["semantics"]["parameters"]["completion"] = "unknown"
        elif fault == "mask":
            row["support"]["masks"] = ["none"]
        elif fault == "limit":
            row["limits"]["max_chunk_elements"] = None
        elif fault == "capacity":
            software["isa"]["resources"][0].pop("capacity")
        else:
            software["scheduling"]["storage"]["banks"]["state"]["ownership"] = "unknown"
    model = inputs(tmp_path, mutate=mutate, mask=CausalRowMask(3))
    with pytest.raises((RowSoftmaxUnavailable, ValueError)):
        schedule(model)


def test_resource_controls_cannot_expand_selected_limits_or_remap_state(tmp_path):
    model = inputs(tmp_path)
    result = schedule(model, resources=RowSoftmaxResources(128, 1, True))
    assert result.report["resources"]["max_chunk_elements"] == 6
    with pytest.raises(ValueError, match="roles"):
        schedule(model, role_banks=("exps", "exps", "state", "state"))
    with pytest.raises(RowSoftmaxUnavailable, match="chunk"):
        schedule(model, chunk_elements=7)
    with pytest.raises(RowSoftmaxUnavailable):
        schedule(model, reservations=(Reservation("exps", 0, 512),))


@pytest.mark.parametrize("fault", ["mode", "selector", "pointer", "count", "mask", "instruction"])
def test_actual_row_operands_do_not_inherit_provider_claims(tmp_path, fault):
    model = inputs(tmp_path, mask=CausalRowMask(3))
    selected, opts = model
    original = opts["adapter"]
    remembered = {}
    def encode(step):
        call = original.encode(step)
        if step.kind == "maximum":
            values = dict(call.args)
            key = {"mode": "numeric", "selector": "space0", "pointer": "source", "count": "count",
                   "mask": "offset"}.get(fault)
            if key:
                values[key] = Const(9)
                call = replace(call, args=tuple(values.items()))
            else:
                call = replace(call, instr="unselected")
        remembered[id(call)] = step
        return call
    adapter = replace(original, encode=encode, project=lambda call: remembered[id(call)])
    diagnostic = {"mode": "constant/mode", "selector": "selector", "instruction": "no selected phase"}
    with pytest.raises(ValueError, match=diagnostic.get(fault, "provider row projection")):
        schedule((selected, {**opts, "adapter": adapter}))


def test_selected_inputs_and_checked_result_are_current_through_emission(tmp_path, exp_object):  # noqa: F811
    model = inputs(tmp_path)
    result = schedule(model)
    with pytest.raises(ValueError, match="replan before emission"):
        native_program(replace(result, index_bits=32), model[1])
    model[0].software_path.write_bytes(model[0].software_bytes + b"\n")
    with pytest.raises(ValueError, match="changed; reselect"):
        native_program(result, model[1])


def test_changed_contract_inside_row_encoding_rejects(tmp_path):
    model = inputs(tmp_path)
    selected, opts = model
    original = opts["adapter"].encode
    def encode(step):
        selected.facts_path.write_bytes(selected.facts_bytes + b"\n")
        return original(step)
    with pytest.raises(ValueError, match="changed; reselect"):
        schedule((selected, {**opts, "adapter": replace(opts["adapter"], encode=encode)}))


def test_selected_capacity_boundary_and_alias_views_are_checked(tmp_path):
    def capacity(size):
        def mutate(facts, software, row):
            facts["capacities"][0] = facts["sizes"][0] = size
        return mutate
    for size in (112, 111):
        directory = tmp_path / str(size)
        directory.mkdir()
        model = inputs(directory, separate=False, mutate=capacity(size))
        if size == 111:
            with pytest.raises(RowSoftmaxUnavailable, match="cannot be allocated"):
                schedule(model)
        else:
            result = schedule(model)
            assert max((v.base + v.capacity) * v.unit_bytes for v in result.storage) == 112
    def alias(facts, software, row):
        banks = software["scheduling"]["storage"]["banks"]
        banks["exps"]["backing"] = banks["state"]["backing"]
        facts["selectors"][1] = 0
    directory = tmp_path / "alias"
    directory.mkdir()
    result = schedule(inputs(directory, mutate=alias), reservations=(Reservation("exps", 0, 32),))
    assert all(view.base * view.unit_bytes >= 32 for view in result.storage)
    assert len({view.backing for view in result.storage}) == 1


def test_tail_and_code_budgets_choose_a_legal_complete_domain(tmp_path):
    def mutate(facts, software, row):
        facts["granularity"] = 2
        row["support"]["partial_chunks"] = False
    model = inputs(tmp_path, columns=18, chunk=8, mutate=mutate)
    result = schedule(model, resources=RowSoftmaxResources(128, 1, True, 7))
    assert result.report["resources"]["partial_chunks"] is False
    assert result.report["static_calls"] == 7
    assert result.report["executed_commands"] == 5 * (3 + 4 * 3)
    with pytest.raises(RowSoftmaxUnavailable, match="tail"):
        schedule(model, chunk_elements=8)


def test_selected_row_source_numerics_and_index_width_cannot_be_overridden(tmp_path):
    model = inputs(tmp_path)
    for changes in (dict(numerics=replace(model[1]["numerics"], normalization="divide")),
                    dict(adapter=replace(model[1]["adapter"], index_bits=32))):
        with pytest.raises(ValueError, match="disagrees"):
            schedule(model, **changes)


def test_row_source_changes_during_encoding_reject_stale_context(tmp_path):
    model = inputs(tmp_path)
    selected, opts = model
    original = opts["adapter"].encode
    def encode(step):
        if step.kind == "normalize":
            object.__setattr__(opts["problem"], "rows", 6)
        return original(step)
    with pytest.raises(ValueError, match="inputs changed"):
        schedule((selected, {**opts, "adapter": replace(opts["adapter"], encode=encode)}))


def test_row_library_changes_during_encoding_and_before_emission_reject(tmp_path, monkeypatch):
    from merlin.compile.scheduling import schedule_pipeline
    model = inputs(tmp_path)
    result = schedule(model)
    selected, opts = model
    owner = schedule_pipeline.shared_library_sources
    changed = False
    def sources():
        return {**owner(), "mutation": "changed"} if changed else owner()
    monkeypatch.setattr(schedule_pipeline, "shared_library_sources", sources)
    original = opts["adapter"].encode
    def encode(step):
        nonlocal changed
        changed = True
        return original(step)
    with pytest.raises(ValueError, match="library changed"):
        schedule((selected, {**opts, "adapter": replace(opts["adapter"], encode=encode)}))
    # Contract revalidation shares the complete roster and now rejects the
    # expired declaration before the later row emission guard is reached.
    with pytest.raises(ValueError, match="resolver changed"):
        native_program(result, opts)


def test_selected_row_order_refuses_before_target_encoding(tmp_path):
    model = inputs(tmp_path)
    selected, opts = model
    def unreachable(step):
        raise AssertionError("invalid ordering reached the encoder")
    with pytest.raises(ValueError, match="ordering|prerequisites"):
        schedule((selected, {**opts, "adapter": replace(opts["adapter"], encode=unreachable)}),
                 pass_order=("row_tiling", "row_semantics", "row_allocation", "row_schedule"))


@pytest.mark.parametrize("fault", ["local_unit", "isa_alignment", "narrow", "encoding", "extra_effect"])
def test_unmodeled_isa_requirements_are_not_silently_discarded(tmp_path, fault):
    def mutate(facts, software, row):
        instruction = software["isa"]["instructions"][1]
        operands = {operand["name"]: operand for operand in instruction["operands"]}
        if fault == "local_unit":
            operands["address0"]["unit"] = "words"
        elif fault == "isa_alignment":
            facts["required_alignment"] = 8
            operands["source"]["alignment"] = {"fact": ["facts", "required_alignment"]}
        elif fault == "narrow":
            facts["count_bits"] = 8
            operands["count"]["type"]["bits"] = {"fact": ["facts", "count_bits"]}
        elif fault == "encoding":
            index = NAMES.index("count") + 1
            facts.update(wide_bits=64, padding_offset=facts["offsets"][index] + 8, padding_bits=56, zero=0)
            facts["field_bits"][index] = 8
            for inst in software["isa"]["instructions"]:
                next(arg for arg in inst["operands"] if arg["name"] == "count")["type"]["bits"] = {
                    "fact": ["facts", "wide_bits"]}
                inst["encoding"].append(dict(offset={"fact": ["facts", "padding_offset"]},
                                             width={"fact": ["facts", "padding_bits"]},
                                             fixed={"fact": ["facts", "zero"]}))
        else:
            software["isa"]["resources"].append(dict(name="queue", kind="execution", unit="entries"))
            instruction["effects"]["writes"].append("queue")
    with pytest.raises(ValueError, match="narrower" if fault in ("narrow", "encoding") else None):
        inputs(tmp_path, mutate=mutate)


@pytest.mark.parametrize("field", ["granularity", "index_bits"])
def test_unknown_rtl_limits_remain_unknown(tmp_path, field):
    def mutate(facts, software, row):
        row["limits"][field] = None
    model = inputs(tmp_path, mutate=mutate)
    with pytest.raises(RowSoftmaxUnavailable, match="limits are unknown"):
        schedule(model)


def test_execution_contract_cannot_change_inside_a_row_pass(tmp_path):
    model = inputs(tmp_path)
    selected, opts = model
    original = opts["adapter"].encode
    def encode(step):
        object.__setattr__(adapter, "qualification_identity", "different-evidence")
        return original(step)
    adapter = replace(opts["adapter"], encode=encode)
    with pytest.raises(ValueError, match="execution inputs changed"):
        schedule((selected, {**opts, "adapter": adapter}))


def test_changed_contract_during_c_rendering_rejects(tmp_path, monkeypatch):
    from merlin.sched import codegen
    model = inputs(tmp_path)
    result = schedule(model)
    def render(*args, **kwargs):
        model[0].software_path.write_bytes(model[0].software_bytes + b"\n")
        return "/* changed target output */"
    monkeypatch.setattr(codegen, "emit_c_function", render)
    with pytest.raises(ValueError, match="changed; reselect"):
        result.emit_c(None, symbol="changed")


def test_printable_receipt_cannot_disable_row_emission_freshness(tmp_path):
    result = schedule(inputs(tmp_path))
    result.report.pop("pipeline")
    changed = replace(result, kernel=replace(result.kernel, name="changed"))
    with pytest.raises(ValueError, match="replan before emission"):
        changed.emit_c(None, symbol="changed")
    result.report["pipeline"] = {"kernel_sha256": changed.kernel.digest()}
    with pytest.raises(ValueError, match="replan before emission"):
        changed.emit_c(None, symbol="changed")
    with pytest.raises(ValueError, match="replan before emission"):
        replace(result, passes=()).emit_c(None, symbol="changed")
