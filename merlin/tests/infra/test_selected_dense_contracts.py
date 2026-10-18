"""Two selected inputs drive existing looped tiling and native execution.

Synthetic RTL and independent byte-memory arithmetic qualify this mechanism's
test scope only. No hardware timing or real target support is established.
"""

import hashlib
import json
import time
from dataclasses import replace

import pytest
import yaml
from test_compact_dense import ADAPTER, encode, fixture, native_dense_pair

from merlin.compile.scheduling.block_schedule import Compute, Load
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, schedule_compact_dense
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.transforms.dense_planning import DenseResources


def inputs(tmp_path, *, tile=(2, 3, 4), widths=(1, 1, 4), shared=False, shape=(5, 14, 13), mutate=None):
    p, _, s = fixture(tile=tile, widths=widths, shape=shape, padded=True)
    if shared:
        s = DenseLoopStorage(replace(s.lhs, backing="unified"),
                             replace(s.rhs, backing="unified", base=32),
                             replace(s.accumulator, backing="unified", base=32))
    roles = ("lhs", "rhs", "accumulator")
    views = (s.lhs, s.rhs, s.accumulator)
    memory_names = ("local",) * 3 if shared else ("left", "right", "sums")
    capacities = {name: max((v.base + v.capacity) * v.unit_bytes
                           for v, n in zip(views, memory_names) if n == name) for name in memory_names}
    facts = {"facts": {"target": "neutral", "primitive": dict(zip(("m", "k", "n"), tile)),
                       "element_bits": dict(zip(roles, (x * 8 for x in widths))),
                       "row_bytes": {role: v.unit_bytes for role, v in zip(roles, views)},
                       "capacities": capacities, "alignment": 1, "origin": 0,
                       "word_bits": 520, "code_bits": 8, "arg_bits": 64,
                       "offsets": [8 + 64 * i for i in range(8)], "codes": list(range(5)),
                       "padding_offsets": [8 + 64 * i for i in range(9)],
                       "padding_widths": [64 * (8 - i) for i in range(9)]}}

    def ref(*parts):
        return {"fact": ["facts", *parts]}

    names = ("lhs_load", "weight_load", "preload", "compute", "store")
    arguments = (
        ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
        ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
        ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth"),
        ("lhs", "m", "k", "rows", "depth"), ("row", "col", "rows", "cols", "acc"),
    )
    left, right, acc = memory_names
    effects = ((["input_lhs"], [left]), (["input_rhs"], [right]),
               ([right, acc], ["engine"]), ([left, right, acc, "engine"], [acc]), ([acc], ["output"]))
    instructions = []
    for i, (name, args, (reads, writes)) in enumerate(zip(names, arguments, effects)):
        encoding = [dict(offset=ref("origin"), width=ref("code_bits"), fixed=ref("codes", i))]
        encoding += [dict(offset=ref("offsets", j), width=ref("arg_bits"), argument=key) for j, key in enumerate(args)]
        if len(args) < 8:
            encoding.append(dict(offset=ref("padding_offsets", len(args)),
                                 width=ref("padding_widths", len(args)), fixed=ref("origin")))
        instructions.append(dict(name=name, operands=[dict(name=key, unit="integer", type=dict(
            kind="integer", signed=True, bits=ref("arg_bits"))) for key in args], results=[], immediates=[],
            encoding=encoding, effects=dict(reads=sorted(set(reads)), writes=sorted(set(writes)))))
    resources = [dict(name=name, kind="memory", unit="bytes", capacity=ref("capacities", name))
                 for name in sorted(capacities)]
    resources += [dict(name=name, kind="memory", unit="bytes") for name in ("input_lhs", "input_rhs", "output")]
    resources.append(dict(name="engine", kind="execution", unit="commands"))
    dtypes = tuple("i" + str(8 * x) for x in widths)
    dense = dict(schema="merlin.dense_schedule.v1", primitive={k: ref("primitive", k) for k in ("m", "k", "n")},
                 formats={role: dict(dtype=dtype, bits=ref("element_bits", role))
                          for role, dtype in zip(roles, dtypes)},
                 storage={role: dict(resource=name, backing=view.backing, row_bytes=ref("row_bytes", role),
                                     origin_bytes=ref("origin"), alignment_rows=ref("alignment"))
                          for role, name, view in zip(roles, memory_names, views)},
                 external=dict(lhs="input_lhs", rhs="input_rhs", output="output"),
                 instructions=dict(zip(("lhs_load", "rhs_load", "preload", "compute", "store"), names)),
                 semantics=dict(layout="row_major", arithmetic="signed_integer_mac_modular", reduction="source_order",
                                initialization="zero", continuation="exact", completion="synchronous",
                                accumulator="addressable", ownership="invocation_private"),
                 support=dict(max_m_blocks=3, max_n_blocks=3, max_k_blocks=2,
                              partial_output_tiles=True, partial_output_blocks=True, partial_reduction_tiles=True))
    software = dict(schema="merlin.software_spec.v1", target="neutral", status="reviewed", isa=dict(
        schema="merlin.isa_spec.v1", namespace="dense_test", facts_sha256="0" * 64,
        word_bits=ref("word_bits"), resources=resources, instructions=instructions),
        scheduling=dict(schema="merlin.schedule_effects.v1", numerical_policy="ordered-integer-wrap-v1",
                        instruction_byteorder="little", configuration={}, dense=dense))
    if mutate is not None:
        mutate(facts["facts"], software, dense)
    fact_path, software_path = tmp_path / "facts.json", tmp_path / "software.yaml"
    fact_path.write_text(json.dumps(facts))
    software["isa"]["facts_sha256"] = hashlib.sha256(fact_path.read_bytes()).hexdigest()
    software_path.write_text(yaml.safe_dump(software))
    selected = read_scheduling_contracts(fact_path, software_path, target="neutral")
    adapter = replace(ADAPTER, configuration_identity=selected.hardware_identity,
                      numerical_policy_identity=selected.numerical_policy,
                      qualification_identity="independent-byte-memory-native-test-v1")
    return selected, selected.dense.problem(p.m, p.k, p.n), s, adapter


def schedule(model, *, problem=None, storage=None, adapter=None, **kwargs):
    selected, p, s, a = model
    return schedule_compact_dense(
        p if problem is None else problem, None, s if storage is None else storage, a if adapter is None else adapter,
        name="dense", dtypes=selected.dense.dtypes, static_budget=80, external_buffers_disjoint=True,
        verification_timeout_ms=20_000, selected_contracts=selected, **kwargs,
    )


@pytest.mark.parametrize("tile,widths,shared,shape", [
    ((2, 3, 4), (1, 1, 4), False, (5, 14, 13)),
    ((3, 2, 5), (2, 1, 4), True, (7, 9, 16)),
    ((2, 3, 4), (2, 2, 4), False, (5, 14, 13)),
])
def test_selected_facts_and_semantics_drive_native_paired_execution(
    tmp_path, record_property, tile, widths, shared, shape
):
    model = inputs(tmp_path, tile=tile, widths=widths, shared=shared, shape=shape)
    selected, p, s, adapter = model
    started = time.perf_counter()
    result = schedule(model)
    elapsed = time.perf_counter() - started
    assert result is not None
    receipt = result.realization["pipeline"]
    assert [p["name"] for p in receipt["passes"]] == ["admission", "blocking", "emission", "realization"]
    assert "choice" in receipt["passes"][2]["invalidated"]
    assert receipt["context"]["hardware"] == selected.hardware_identity
    assert receipt["context"]["numerical_policy"] == selected.numerical_policy
    assert receipt["selected_contracts"]["consumed_scopes"] == [
        "dense_geometry", "dense_storage_windows", "dense_numerics_and_execution", "dense_instruction_bindings"
    ]
    controls = DenseResources(1 << 20, 1 << 20, 3, 3, 2,
                             partial_output_tiles=True, partial_output_blocks=True)
    pair = native_dense_pair(tmp_path, p, controls, s, result, adapter, dtypes=selected.dense.dtypes,
                             selected_contracts=selected, overflow=widths[:2] == (2, 2), whole_graph=True)
    record_property("selected_dense_pair", json.dumps(dict(
        selected_contracts=receipt["selected_contracts"], shared_library_sources=receipt["library_sources_sha256"],
        scheduling_seconds=elapsed, **pair,
    ), sort_keys=True))


@pytest.mark.parametrize("field,value", [
    ("completion", "asynchronous"), ("completion", "unknown"), ("continuation", "rounded"),
    ("ownership", "unknown"), ("accumulator", "internal"), ("layout", "unknown"),
    ("arithmetic", "saturating"), ("reduction", "reassociated"), ("initialization", "bias"),
])
def test_unsupported_semantics_have_precise_diagnostics(tmp_path, field, value):
    model = inputs(tmp_path, mutate=lambda f, s, d: d["semantics"].update({field: value}))
    with pytest.raises(ValueError, match="requires " + field):
        schedule(model)


def test_capacity_boundaries_reservations_and_aliases(tmp_path):
    model = inputs(tmp_path)
    selected, p, s, adapter = model
    # Assigned windows end exactly at each fact-bound capacity.
    assert schedule(model) is not None
    bad = replace(s, rhs=replace(s.rhs, capacity=s.rhs.capacity + 1))
    with pytest.raises(ValueError, match="RTL window"):
        schedule(model, storage=bad)
    with pytest.raises(ValueError, match="reservations"):
        schedule(model, reservations=(AddressWindow(s.lhs.backing, s.lhs.base * s.lhs.unit_bytes, 1),))
    assert schedule(model, reservations=(AddressWindow("other", 0, 1),)) is not None
    # Software can expose several views of one physical SRAM; live overlap is
    # still forbidden even when instruction/resource names differ.
    shared = inputs(tmp_path, shared=True)
    selected, p, s, adapter = shared
    overlap = replace(s, rhs=replace(s.rhs, base=2))
    assert schedule(shared, storage=overlap) is None


def test_unknown_capacity_and_qualification_are_not_invented(tmp_path):
    model = inputs(tmp_path)
    with pytest.raises(ValueError, match="qualification"):
        schedule(model, adapter=replace(model[3], qualification_identity=None))
    with pytest.raises(ValueError, match="matching selected"):
        schedule(model, adapter=ADAPTER)

    def unknown(facts, software, dense):
        software["isa"]["resources"][0].pop("capacity")
    with pytest.raises(ValueError, match="capacity is unknown"):
        schedule(inputs(tmp_path, mutate=unknown))


@pytest.mark.parametrize("mutation,diagnostic", [
    (lambda f, s, d: d["primitive"].update(m=16), "fact"),
    (lambda f, s, d: f["element_bits"].update(lhs=16), "dtype"),
    (lambda f, s, d: s["isa"]["instructions"][3].update(effects=dict(reads=["output"], writes=[])), "effects"),
    (lambda f, s, d: s["isa"]["instructions"][3].pop("effects"), "complete ISA effects"),
    (lambda f, s, d: d["instructions"].update(compute="store"), "distinct"),
])
def test_malformed_or_contradictory_selected_inputs_refuse(tmp_path, mutation, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        inputs(tmp_path, mutate=mutation)


def test_problem_freshness_ordering_and_actual_instruction_mutations(tmp_path):
    model = inputs(tmp_path)
    with pytest.raises(ValueError, match="geometry"):
        schedule(model, problem=replace(model[1], tile_n=8))
    with pytest.raises(ValueError, match="pass order"):
        schedule(model, pass_order=[])
    with pytest.raises(ValueError, match="ordering|prerequisites"):
        schedule(model, pass_order=("admission", "blocking", "realization", "emission"))

    def wrong_instruction(op):
        call = encode(op)
        return replace(call, instr="renamed_compute") if isinstance(op, Compute) else call
    # A provider that projects an arbitrary name into the expected compute
    # cannot bypass the selected software instruction binding.
    def misleading(call):
        return ADAPTER.project(replace(call, instr="compute") if call.instr == "renamed_compute" else call)
    with pytest.raises(ValueError, match="actual instruction"):
        schedule(model, adapter=replace(model[3], encode=wrong_instruction, project=misleading))

    def corrupt(op):
        if isinstance(op, Load):
            op = replace(op, row=op.row + 1)
        return encode(op)
    with pytest.raises(ValueError):
        schedule(model, adapter=replace(model[3], encode=corrupt))

    selected = model[0]
    selected.software_path.write_bytes(selected.software_bytes + b"\n")
    with pytest.raises(ValueError, match="changed"):
        schedule(model)


def test_contract_changes_during_lowering_reject_and_result_cannot_emit_stale_inputs(tmp_path):
    model = inputs(tmp_path)
    selected = model[0]
    def mutate(op):
        selected.facts_path.write_bytes(selected.facts_bytes + b"\n")
        return encode(op)
    with pytest.raises(ValueError, match="changed"):
        schedule(model, adapter=replace(model[3], encode=mutate))
    model = inputs(tmp_path)
    result = schedule(model)
    model[0].software_path.write_bytes(model[0].software_bytes + b"\n")
    with pytest.raises(ValueError, match="changed"):
        result.emit_c(None, symbol="stale")


def test_tail_capabilities_and_no_candidate_preserve_original_route(tmp_path):
    model = inputs(tmp_path, mutate=lambda f, s, d: d["support"].update(partial_output_tiles=False))
    assert schedule(model) is None
    with pytest.raises(ValueError, match="partial reduction"):
        schedule(inputs(tmp_path, mutate=lambda f, s, d: d["support"].update(partial_reduction_tiles=False)))


def test_changed_numerical_identity_index_width_and_alignment_refuse(tmp_path):
    model = inputs(tmp_path)
    for adapter in (replace(model[3], numerical_policy_identity="another-policy"),
                    replace(model[3], configuration_identity="another-configuration")):
        with pytest.raises(ValueError, match="matching selected"):
            schedule(model, adapter=adapter)
    def coordinates32(facts, software, dense):
        facts.update(arg_bits=32, word_bits=264, offsets=[8 + 32 * i for i in range(8)],
                     padding_offsets=[8 + 32 * i for i in range(9)],
                     padding_widths=[32 * (8 - i) for i in range(9)])
    narrow_model = inputs(tmp_path, mutate=coordinates32)
    with pytest.raises(ValueError, match="narrower than"):
        schedule(narrow_model)
    assert schedule(narrow_model, adapter=replace(narrow_model[3], index_bits=32)) is not None
    with pytest.raises(ValueError, match="alignment"):
        schedule(inputs(tmp_path, mutate=lambda f, s, d: f.update(alignment=2)))

    def narrow(facts, software, dense):
        for instruction in software["isa"]["instructions"]:
            for arg in instruction["operands"]:
                arg["type"]["bits"] = 32
    with pytest.raises(ValueError, match="full-width"):
        # Encoded fields must agree with typed argument widths, independent of
        # whichever index width the adapter subsequently requests.
        inputs(tmp_path, mutate=narrow)


def test_current_library_and_checked_kernel_are_required_for_emission(tmp_path, monkeypatch):
    from merlin.compile.scheduling import compact_dense
    model = inputs(tmp_path)
    result = schedule(model)
    changed = replace(result.kernel, name="changed_after_verification")
    with pytest.raises(ValueError, match="kernel or shared library changed"):
        replace(result, kernel=changed).emit_c(None, symbol="stale")
    monkeypatch.setattr(compact_dense, "_dense_library_sources", lambda: {"modified": "0" * 64})
    with pytest.raises(ValueError, match="kernel or shared library changed"):
        result.emit_c(None, symbol="stale")


def test_library_change_inside_instruction_encoding_rejects(tmp_path, monkeypatch):
    from merlin.compile.scheduling import compact_dense
    model = inputs(tmp_path)
    original = compact_dense._dense_library_sources
    changed = False

    def sources():
        return {**original(), **({"modified": "0" * 64} if changed else {})}

    def encode_and_change(op):
        nonlocal changed
        changed = True
        return encode(op)

    monkeypatch.setattr(compact_dense, "_dense_library_sources", sources)
    with pytest.raises(ValueError, match="library changed during"):
        schedule(model, adapter=replace(model[3], encode=encode_and_change))


def test_false_continuation_qualification_fails_independent_native_reference(tmp_path):
    selected, p, s, adapter = model = inputs(tmp_path)
    result = schedule(model)
    controls = DenseResources(1 << 20, 1 << 20, 3, 3, 2,
                             partial_output_tiles=True, partial_output_blocks=True)
    # A declaration and valid projected loop cannot qualify an opaque C
    # implementation that resets its accumulator at every reduction tile.
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        native_dense_pair(tmp_path, p, controls, s, result, adapter, dtypes=selected.dense.dtypes,
                          selected_contracts=selected, lose_continuation=True)
