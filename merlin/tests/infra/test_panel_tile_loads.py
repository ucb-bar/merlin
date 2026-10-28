"""Final wide commands are expanded through every real panel epoch band."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture

from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission, schedule_dense_panel_pipeline
from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads, expand_panel_tile_loads
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw,
    TileLoadMapping,
    _linear,
    coalesce_tile_load_loops,
    propose_tile_load_coalescing,
)
from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Ptr, Var, evaluate, instances, select
from merlin.transforms.dense_planning import dense_blocks


def setup(shape=(4, 6, 4), block_n=2, require_changes=True):
    p, r, s = fixture(shape=shape, tile=(2, 2, 2))
    b = next(b for b in dense_blocks(p, replace(r, max_k_blocks=1)) if (b.m, b.n) == (2, block_n))
    admission = DensePanelOverlapAdmission(ADAPTER.configuration_identity, ADAPTER.numerical_policy_identity,
                                         "independent-test-ordered-epoch-premises", 2 * block_n, 2 + block_n, 2, 2)
    plan = make_dense_schedule_plan(p, r, s, ADAPTER, source_identity="a" * 64, dtypes=("i8", "i8", "i32"),
                                   static_budget=128, block_tiles=(2, 1, block_n), pipeline_prefetch=True)
    original = schedule_dense_panel_pipeline(p, r, s, ADAPTER, admission, name="panels", dtypes=("i8", "i8", "i32"),
                                            schedule_plan=plan, source_identity="a" * 64,
                                            external_buffers_disjoint=True)
    laws = {role: TileLoadAddressLaw(2, 2, 8, 1, pitch, 2, "test-exact-physical-tile-addresses")
            for role, pitch in [("lhs", p.k), ("weight", p.n)]}

    def project(call):
        if call.instr not in ("lhs_load", "weight_load"):
            return None
        op = ADAPTER.project(call)
        view = s.lhs if op.role == "lhs" else s.rhs
        law = laws[op.role]
        return TileLoadMapping(op.role, view.backing, op.dram_row * law.source_pitch_bytes + op.dram_col,
                               op.row, op.rows, op.cols, law)

    def encode(call, columns):
        return replace(call, args=tuple((k, Const(columns) if k == "cols" else v) for k, v in call.args))

    def band(call, mapping):
        first = project(call)
        offset, terms = _linear(mapping.source_byte)
        base, base_terms = _linear(first.source_byte)
        assert terms == base_terms
        columns = offset - base
        return replace(call, args=tuple((k, mapping.local_row if k == "dst" else mapping.columns if k == "cols"
                                         else v + columns if k == "src_col"
                                         else v + columns // mapping.law.tile_columns if k == "b" else v)
                                        for k, v in call.args))

    proposal, changes = propose_tile_load_coalescing(original.kernel, project=project, encode=encode, max_tiles=4)
    if require_changes:
        assert changes["changes"]
    return p, r, s, b, original, proposal, project, encode, band


def test_actual_wide_panel_keeps_source_products_drains_and_all_bands():
    p, r, s, b, original, proposal, project, _, band = setup()
    result = admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
    proof = result.realization["tile_load_projection"]
    assert proof["physical_load_commands"] < proof["expanded_logical_tile_bands"]
    assert proof["expanded_logical_tile_bands"] == sum(
        original.realization["source"][key] for key in ("lhs_loads", "rhs_loads"))
    assert result.realization["dynamic_epoch_drains"] == original.realization["dynamic_epoch_drains"]
    assert result.realization["source"]["output_tiles"] == original.realization["source"]["output_tiles"]
    assert result.realization["overlapped_compute_pairs"] == original.realization["overlapped_compute_pairs"] > 0
    assert any(call.arg("cols").value > p.tile_n for call, _ in instances(result.kernel)
               if call.instr == "weight_load")
    assert result.materialize_scf() is not None
    result.revalidate()


def evaluated_calls(kernel):
    def value(item, env):
        if isinstance(item, Expr):
            return evaluate(item, env)
        if type(item) is Ptr:
            return item.tensor, evaluate(item.offset, env)
        return item
    return [(call.instr, tuple((key, value(item, env)) for key, item in call.args))
            for call, env in instances(kernel)]


@pytest.mark.parametrize("columns", [4, 7, 8])
@pytest.mark.parametrize("separate_coordinate", [False, True])
def test_counted_affine_bands_preserve_every_address_extent_order_and_other_operand(columns, separate_coordinate):
    law = TileLoadAddressLaw(2, 1, 8, 1, 32, 1, "independent-counted-band-test")
    outer = Var("panel_wide_band_0")
    args = (("src", outer * columns), ("dst", outer * ((columns + 1) // 2)),
            ("rows", Const(1)), ("cols", Const(columns)), ("tag", outer if separate_coordinate else Const(0)))
    original = Kernel("physical_wide", (), (Loop(outer.name, 3, (Call("load", args),)),))

    def project(call):
        a = dict(call.args)
        return TileLoadMapping("source", "local", a["src"], a["dst"], a["rows"], a["cols"], law)

    def band(call, mapping):
        return replace(call, args=tuple((key, {"src": mapping.source_byte, "dst": mapping.local_row,
                                              "cols": mapping.columns}.get(key, value)) for key, value in call.args))

    reference, static = expand_panel_tile_loads(original, project=project, encode_band=band, counted_bands=False)
    compact, counted = expand_panel_tile_loads(original, project=project, encode_band=band)
    assert evaluated_calls(compact) == evaluated_calls(reference)
    assert counted["physical_load_commands"] == static["physical_load_commands"] == 3
    assert counted["expanded_logical_tile_bands"] == static["expanded_logical_tile_bands"]
    assert all(row["representation"] == "counted_affine_bands" for row in counted["actual_accesses"])
    assert sum(type(s) is Call for s, _ in instances(compact)) == 3 * ((columns + 1) // 2)
    if separate_coordinate:
        # The tag distinguishes the physical outer command. It must not be
        # replaced by a flattened index just because memory addresses flatten.
        assert not counted["canonical_load_runs"]


def test_mixed_radix_and_split_load_runs_recover_one_exact_source_family_without_crossing_fence():
    law = TileLoadAddressLaw(2, 1, 8, 1, 64, 1, "split-affine-band-test")
    outer, inner = Var("outer"), Var("tile")

    def call(source, local, columns):
        return Call("load", (("src", source), ("dst", local), ("rows", Const(1)), ("cols", Const(columns))))

    def project(command):
        if command.instr != "load":
            return None
        a = dict(command.args)
        return TileLoadMapping("source", "local", a["src"], a["dst"], a["rows"], a["cols"], law)

    def band(command, mapping):
        return replace(command, args=tuple((key, {"src": mapping.source_byte, "dst": mapping.local_row,
                                                 "cols": mapping.columns}.get(key, value))
                                          for key, value in command.args))

    prefix = Loop("tile", 2, (call(outer * 22 + inner * 8, outer * 11 + inner * 4, 8),))
    tail = Loop("tile", 3, (call(outer * 22 + (inner + 8) * 2, outer * 11 + inner + 8, 2),))
    original = Kernel("runs", (), (Loop("outer", 2, (prefix, tail)),))
    compact, report = expand_panel_tile_loads(original, project=project, encode_band=band)
    reference, _ = expand_panel_tile_loads(original, project=project, encode_band=band, counted_bands=False)
    assert evaluated_calls(compact) == evaluated_calls(reference)
    assert len(compact.body) == len(compact.body[0].body) == 1 and compact.body[0].extent == 22
    assert type(compact.body[0].body[0]) is Call
    assert {r["kind"] for r in report["canonical_load_runs"]} == {
        "mixed_radix_load_run", "contiguous_split_load_run"}
    separated = replace(original, body=(replace(original.body[0], body=(prefix, Call("fence", ()), tail)),))
    preserved, _ = expand_panel_tile_loads(separated, project=project, encode_band=band)
    static, _ = expand_panel_tile_loads(separated, project=project, encode_band=band, counted_bands=False)
    assert evaluated_calls(preserved) == evaluated_calls(static)
    assert [s.instr for s in preserved.body[0].body if type(s) is Call] == ["fence"]


@pytest.mark.parametrize("metadata", ["nonlinear_band", "selector", "effect"])
def test_counted_projection_preserves_unknown_metadata_and_effect_boundaries(metadata):
    law = TileLoadAddressLaw(2, 1, 8, 1, 64, 1, "metadata-band-test")
    i = Var("i")
    command = Call("load", (("src", i * 8), ("dst", i * 4),
                            ("rows", Const(1)), ("cols", Const(8)),
                            ("tag", select(i, 1, 9, 2) if metadata == "selector" else Const(0))))
    original = Kernel("metadata", (), (Loop("i", 3, (command,)),))

    def project(call):
        a = dict(call.args)
        return TileLoadMapping("source", "local", a["src"], a["dst"], a["rows"], a["cols"], law)

    def band(call, mapping):
        from merlin.compile.scheduling.tile_load_coalescing import _linear

        index = (_linear(mapping.source_byte)[0] - _linear(project(call).source_byte)[0]) // 2
        args = {"src": mapping.source_byte, "dst": mapping.local_row, "cols": mapping.columns}
        if metadata == "nonlinear_band":
            # First two bands suggest a constant tag. The third must still
            # survive completely, even though it is absent from the address law.
            args["tag"] = Const(77 if index == 2 else 0)
        result = replace(call, args=tuple((k, args.get(k, v)) for k, v in call.args))
        return replace(result, sets=(("state", Const(1)),)) if metadata == "effect" else result

    if metadata == "effect":
        with pytest.raises(ValueError, match="checking band"):
            expand_panel_tile_loads(original, project=project, encode_band=band)
    else:
        actual, report = expand_panel_tile_loads(original, project=project, encode_band=band)
        reference, _ = expand_panel_tile_loads(original, project=project, encode_band=band, counted_bands=False)
        assert evaluated_calls(actual) == evaluated_calls(reference)
        assert not report["canonical_load_runs"]
        if metadata == "nonlinear_band":
            assert report["actual_accesses"][0]["representation"] == "static_bands"


def test_address_only_mapping_cannot_enter_original_synchronous_optimizer():
    _, _, _, _, original, _, project, encode, _ = setup()
    with pytest.raises(ValueError, match="distinct selected completion"):
        coalesce_tile_load_loops(original.kernel, project=project, encode=encode, max_tiles=4)


def test_second_band_overlap_is_detected_even_when_first_band_is_safe():
    p, r, s, b, original, proposal, project, _, band = setup()

    def faulty(call):
        mapping = project(call)
        if mapping is not None:
            # Only the final physical wide command has the bad block-stride.
            # First band remains unchanged; the second reaches the active slot.
            if mapping.columns.value > mapping.law.tile_columns:
                mapping = replace(mapping, law=replace(mapping.law, local_block_stride=4))
        return mapping

    # The checking encoder must expose the same actual law for its narrowed
    # bands as the command being checked, without mutating original kernels.
    def physical_project(call):
        if call.instr.endswith("_band"):
            mapping = project(replace(call, instr=call.instr.removesuffix("_band")))
            return replace(mapping, law=replace(mapping.law, local_block_stride=4))
        return faulty(call)

    def physical_band(call, mapping):
        checked = band(call, mapping)
        return replace(checked, instr=call.instr + "_band") if call.arg("cols").value > 2 else checked

    # Use ordinary dense operand projection for this checking-only spelling.
    adapter = replace(ADAPTER, project=lambda c: ADAPTER.project(
        replace(c, instr=c.instr.removesuffix("_band"))))
    with pytest.raises(ValueError, match="overlapping in-flight"):
        admit_panel_tile_loads(original, proposal, p, b, r, s, adapter,
                              project=physical_project, encode_band=physical_band)


def test_wrong_actual_source_pitch_and_wrong_band_encoder_refuse():
    p, r, s, b, original, proposal, project, _, band = setup()

    def wrong_pitch(call):
        mapping = project(call)
        return None if mapping is None else replace(mapping, law=replace(mapping.law, source_pitch_bytes=99))

    with pytest.raises(ValueError, match="source layout"):
        admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER, project=wrong_pitch, encode_band=band)
    with pytest.raises(ValueError, match="checking band"):
        expand_panel_tile_loads(proposal, project=project, encode_band=lambda c, m: c)


def test_explicit_affine_external_view_preserves_original_bands_and_live_address_checks():
    from merlin.compile.scheduling.panel_transfers import AffineMatrixSource

    p, r, s, b, original, proposal, logical_project, _, band = setup()
    source = AffineMatrixSource("weight", (p.k, p.n), 1, 7, 21, 1, 0, 7 + (p.k - 1) * 21 + p.n)
    declaration = {"weight": source}
    actual_pitch = [21]

    def physical_project(call):
        mapping = logical_project(call)
        if mapping is None or mapping.source != "weight":
            return mapping
        operation = ADAPTER.project(call)
        return replace(mapping, source_byte=7 + operation.dram_row * actual_pitch[0] + operation.dram_col,
                       law=replace(mapping.law, source_pitch_bytes=actual_pitch[0]))

    def physical_band(call, mapping):
        offset, terms = _linear(mapping.source_byte)
        base, base_terms = _linear(physical_project(call).source_byte)
        assert terms == base_terms
        columns = offset - base
        return replace(call, args=tuple((k, mapping.local_row if k == "dst" else mapping.columns if k == "cols"
                                         else v + columns if k == "src_col"
                                         else v + columns // mapping.law.tile_columns if k == "b" else v)
                                        for k, v in call.args))

    with pytest.raises(ValueError, match="source layout"):
        admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER,
                              project=physical_project, encode_band=physical_band)
    result = admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER,
                                   project=physical_project, encode_band=physical_band, external_sources=declaration)
    assert result.realization["source"] == original.realization["source"]
    assert result.realization["dynamic_epoch_drains"] == original.realization["dynamic_epoch_drains"]
    result.revalidate()
    # Caller dictionary changes cannot alter the captured declaration. Actual
    # decoder changes remain visible and are refused during every revalidation.
    declaration.clear()
    result.revalidate()
    actual_pitch[0] = 22
    with pytest.raises(ValueError, match="source layout"):
        result.revalidate()


@pytest.mark.parametrize("mutation", ["shape", "columns", "offset"])
def test_changed_affine_binding_cannot_hide_different_source_effects(mutation):
    from merlin.compile.scheduling.panel_transfers import AffineMatrixSource

    p, r, s, b, original, proposal, project, _, band = setup()
    source = AffineMatrixSource("weight", (p.k, p.n), 1, 0, p.n, 1, 0, 128)
    changed = {"shape": replace(source, shape=(p.k + 1, p.n)),
               "columns": replace(source, column_stride_bytes=2),
               "offset": replace(source, first_byte=1)}[mutation]
    with pytest.raises(ValueError, match="external view|actual wide mapping"):
        admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER,
                              project=project, encode_band=band, external_sources={"weight": changed})


def test_actual_wide_payload_or_completion_change_cannot_use_retained_proof():
    p, r, s, b, original, proposal, project, _, band = setup()
    result = admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
    changed = replace(result, kernel=replace(result.kernel, name="different actual commands"))
    with pytest.raises(ValueError, match="changed after"):
        changed.revalidate()

    def strip(body):
        return tuple(replace(x, body=strip(x.body)) if isinstance(x, Loop) else x
                     for x in body if isinstance(x, Loop) or x.instr != "merlin_dense_panel_drain")

    with pytest.raises(ValueError, match="drained panel epoch|no explicit completion"):
        admit_panel_tile_loads(original, replace(proposal, body=strip(proposal.body)), p, b, r, s, ADAPTER,
                              project=project, encode_band=band)


def test_all_dimension_tails_keep_exact_source_and_expanded_band_counts():
    p, r, s, b, original, proposal, project, _, band = setup(shape=(3, 5, 5), block_n=3, require_changes=False)
    result = admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
    assert result.realization["source"]["output_tiles"] == 6
    projection = result.realization["tile_load_projection"]
    assert projection["physical_load_commands"] == projection["expanded_logical_tile_bands"]


def test_decoder_cannot_hide_a_final_wide_load_from_band_checks():
    p, r, s, b, original, proposal, project, _, band = setup()

    def omit_wide(call):
        mapping = project(call)
        return None if mapping is not None and mapping.columns == Const(4) else mapping

    with pytest.raises(ValueError, match="every actual panel load"):
        admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER, project=omit_wide, encode_band=band)


def test_projection_budget_bounds_all_generated_bands_before_expanding():
    _, _, _, _, _, proposal, project, _, band = setup()

    def huge(call):
        mapping = project(call)
        if mapping is None:
            return None
        return replace(mapping, columns=Const(1 << 40), law=replace(mapping.law, max_columns=1 << 40))

    with pytest.raises(ValueError, match="static bound"):
        expand_panel_tile_loads(proposal, project=huge, encode_band=band, max_static_nodes=128)


def test_readout_batching_composes_with_final_wide_loads_in_either_order():
    from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts

    p, r, s, b, original, proposal, project, encode, band = setup()
    admission = replace(original.admission, max_issued_readouts=4, pending_readouts=1,
                        readout_configuration="ordered_snapshot_before_store_issue")
    wide = admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER,
                                 project=project, encode_band=band)
    assert wide.admission is original.admission
    first_wide = batch_dense_panel_readouts(wide, p, b, r, s, ADAPTER, admission)
    assert first_wide.admission is admission
    batched = batch_dense_panel_readouts(original, p, b, r, s, ADAPTER, admission)
    after_batch, _ = propose_tile_load_coalescing(batched.kernel, project=project, encode=encode, max_tiles=4)
    first_batch = admit_panel_tile_loads(batched, after_batch, p, b, r, s, ADAPTER,
                                       project=project, encode_band=band)
    # Actual emitted commands, not receipts, are independent of this pass order.
    assert first_wide.kernel == first_batch.kernel
    assert first_wide.realization["dynamic_epoch_drains"] < wide.realization["dynamic_epoch_drains"]
    assert first_wide.realization["tile_load_projection"]["physical_load_commands"] < (
        first_wide.realization["tile_load_projection"]["expanded_logical_tile_bands"])
    diagnostic = {"solver_queries", "reused_solver_queries", "source_proof_reused"}
    assert {k: v for k, v in first_wide.realization["source"].items() if k not in diagnostic} == {
        k: v for k, v in first_batch.realization["source"].items() if k not in diagnostic
    }
    first_wide.materialize_scf().verify()
    first_wide.revalidate()
    with pytest.raises(ValueError, match="premises"):
        batch_dense_panel_readouts(wide, p, b, r, s, ADAPTER,
                                  replace(admission, max_issued_loads=admission.max_issued_loads + 1))


def typed_setup(kind):
    from test_compact_dense_bias import ADAPTER as bias_adapter
    from test_dense_integer_readout import SCALAR_IR

    from merlin.compile.scheduling.block_schedule import Store
    from merlin.compile.scheduling.compact_dense import DenseBiasSource
    from merlin.compile.scheduling.dense_contracts import (
        DenseIntegerReadout,
        DenseTiledIntegerReadout,
        TiledIntegerStore,
    )

    p, r, s, b, _, _, project, encode, band = setup()
    p = replace(p, bias=True)
    b = next(block for block in dense_blocks(p, replace(r, max_k_blocks=1)) if (block.m, block.n) == (2, 2))
    bias = DenseBiasSource(p.n, "i32")
    readout = None
    if kind == "scalar":
        readout = DenseIntegerReadout(SCALAR_IR, "i8", "store_i8", ())
    elif kind == "tiled":
        readout = DenseTiledIntegerReadout.from_column_scales([0.5, 0.5, 1.5, 1.5], tile_columns=p.tile_n)
    adapter = bias_adapter
    if readout is not None:
        def typed_encode(op):
            if type(op) is TiledIntegerStore:
                base = bias_adapter.encode(Store(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
                return replace(base, instr=readout.instruction_name,
                               args=(*base.args, ("readout_tile", op.readout_tile)))
            if type(op) is Store:
                return replace(bias_adapter.encode(replace(op, readout=None)), instr=readout.instruction_name)
            return bias_adapter.encode(op)

        def typed_project(call):
            if call.instr == readout.instruction_name:
                op = bias_adapter.project(replace(call, instr="store", args=tuple(
                    (k, v) for k, v in call.args if k != "readout_tile")))
                if kind == "tiled":
                    return TiledIntegerStore(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row,
                                             readout.identity, call.arg("readout_tile"))
                return replace(op, readout=readout.identity)
            return bias_adapter.project(call)

        adapter = replace(bias_adapter, encode=typed_encode, project=typed_project)
    admission = DensePanelOverlapAdmission(adapter.configuration_identity, adapter.numerical_policy_identity,
                                         "original-independent-panel-premises", 4, 4, 2, 2)
    plan = make_dense_schedule_plan(p, r, s, adapter, source_identity="a" * 64, dtypes=("i8", "i8", "i32"),
                                   static_budget=128, block_tiles=(2, 1, 2), pipeline_prefetch=True,
                                   output_readout=readout, bias_source=bias)
    original = schedule_dense_panel_pipeline(
        p, r, s, adapter, admission, name="typed_panels", dtypes=("i8", "i8", "i32"), schedule_plan=plan,
        source_identity="a" * 64, external_buffers_disjoint=True, output_readout=readout, bias_source=bias,
        readout_qualification_identity=None if readout is None else "original-independent-readout-premise",
    )
    proposal, _ = propose_tile_load_coalescing(original.kernel, project=project, encode=encode, max_tiles=4)
    return p, r, s, b, original, proposal, adapter, project, band


@pytest.mark.parametrize("kind", ["raw", "scalar", "tiled"])
def test_wide_bands_preserve_original_bias_and_exact_readout_contracts(kind):
    p, r, s, b, original, proposal, adapter, project, band = typed_setup(kind)
    result = admit_panel_tile_loads(original, proposal, p, b, r, s, adapter, project=project, encode_band=band)
    projection = result.realization["tile_load_projection"]
    assert projection["physical_load_commands"] < projection["expanded_logical_tile_bands"]
    assert result.realization["source"]["bias_initializations"] == 4
    assert result.realization["source"]["requested_store_bytes"] == p.m * p.n * (4 if kind == "raw" else 1)
    boundary = result.realization["original_boundary"]
    assert boundary["readout_qualification_identity"] == original.readout_qualification_identity
    result.materialize_scf().verify()
    result.revalidate()


@pytest.mark.parametrize("kind", ["raw", "scalar", "tiled"])
def test_wide_readout_batching_retains_numerical_boundaries_and_rechecks_actual_payload(kind):
    from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts

    p, r, s, b, original, proposal, adapter, project, band = typed_setup(kind)
    wide = admit_panel_tile_loads(original, proposal, p, b, r, s, adapter,
                                 project=project, encode_band=band)
    admission = replace(original.admission, max_issued_readouts=4, pending_readouts=1,
                        readout_configuration="ordered_snapshot_before_store_issue")
    if kind == "tiled":
        with pytest.raises(ValueError, match="per-tile"):
            batch_dense_panel_readouts(wide, p, b, r, s, adapter, admission)
        return
    result = batch_dense_panel_readouts(wide, p, b, r, s, adapter, admission)
    assert result.original.output_readout == original.output_readout
    assert result.original.bias_source == original.bias_source
    assert result.original.readout_qualification_identity == original.readout_qualification_identity
    assert result.realization["source"]["bias_initializations"] == 4
    assert result.realization["source"]["requested_store_bytes"] == p.m * p.n * (4 if kind == "raw" else 1)
    result.revalidate()
    with pytest.raises(ValueError, match="changed after"):
        replace(result, kernel=replace(result.kernel, name="mutated physical commands")).revalidate()


def map_panel_calls(body, rewrite):
    return tuple(replace(s, body=map_panel_calls(s.body, rewrite)) if type(s) is Loop else rewrite(s) for s in body)


@pytest.mark.parametrize("mutation", ["bias", "quantized_band"])
def test_original_bias_or_quantized_column_mutation_cannot_reuse_band_proof(mutation):
    p, r, s, b, original, proposal, adapter, project, band = typed_setup("tiled")

    def change(call):
        if mutation == "bias" and call.instr == "bias_init":
            return replace(call, args=tuple((k, v + 1 if k == "src" else v) for k, v in call.args))
        if mutation == "quantized_band" and call.instr == "store_tiled_i8":
            return replace(call, args=tuple((k, v + 1 if k == "readout_tile" else v) for k, v in call.args))
        return call

    changed = replace(proposal, body=map_panel_calls(proposal.body, change))
    assert changed != proposal
    with pytest.raises(ValueError):
        admit_panel_tile_loads(original, changed, p, b, r, s, adapter, project=project, encode_band=band)


def test_original_readout_qualification_and_bias_remain_live_after_widening():
    p, r, s, b, original, proposal, adapter, project, band = typed_setup("tiled")
    result = admit_panel_tile_loads(original, proposal, p, b, r, s, adapter, project=project, encode_band=band)
    for changed in (replace(original, bias_source=None), replace(original, readout_qualification_identity="changed")):
        with pytest.raises(ValueError, match="changed"):
            replace(result, original=changed).revalidate()
