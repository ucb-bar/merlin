"""Read-only original-source profile checks, with no SDK or optimizer calls."""

from dataclasses import replace

import pytest
from buffer_rank_one_inputs import source as buffer_source
from buffer_reuse_qualification_cases import BufferCase
from nested_configuration_hoist_inputs import actual as nested_source
from test_compiler_configuration import actual as elision_source
from test_compiler_configuration_hoist import actual as hoist_source
from test_compiler_configuration_hoist import call, selection
from xdsl.utils.exceptions import ParseError

from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer
from merlin.sched.ir import Kernel, Loop, Var
from merlin.verify import compiler_buffer_source_profile as buffer
from merlin.verify import compiler_configuration_source_profile as configuration
from merlin.verify import memory_scalar_nan_admission
from merlin.verify.compiler_profile import SourceOutsideProfile
from merlin.verify.compiler_transforms import ALLOCATOR, Options
from merlin.verify.gate_errors import GateUnavailable

OPTIONS = Options(buffer_allocator=ALLOCATOR)


def no_producers(monkeypatch):
    for path in (
        "merlin.transforms.normalization.normalize_shapes",
        "merlin.transforms.planning.plan_storage",
        "merlin.transforms.planning.reuse_storage",
        "merlin.transforms.loops.segment_loops",
        "merlin.compile.scheduling.configuration_ir._eliminate",
        "merlin.compile.scheduling.configuration_loops._hoist_loop_configuration",
    ):
        monkeypatch.setattr(path, lambda *a, **k: pytest.fail("source predicate called producer/planner"))


@pytest.mark.parametrize("extent,rank", [(1, 1), (3, 1), (65536, 1), (3, 2)])
def test_qualified_uniform_buffer_sources_are_read_only_and_target_free(extent, rank, monkeypatch):
    raw = buffer_source(extent, rank=rank)
    no_producers(monkeypatch)
    assert buffer.admit_source(raw, options=OPTIONS) == buffer.PROFILE_ID


def test_existing_rectangular_buffer_source_is_supported(monkeypatch):
    raw = BufferCase("source-only", 2, 3, 2, epochs=3).source
    no_producers(monkeypatch)
    assert buffer.admit_source(raw, options=Options(buffer_allocator=ALLOCATOR, tile_size=2)) == buffer.PROFILE_ID


@pytest.mark.parametrize("rank,computed", [(3, False), (1, True), (2, True)])
def test_computed_and_higher_rank_buffer_sources_are_known_exclusions(rank, computed, monkeypatch):
    no_producers(monkeypatch)
    with pytest.raises(SourceOutsideProfile) as failure:
        buffer.admit_source(buffer_source(3, rank=rank, computed=computed), options=OPTIONS)
    assert failure.value.profile_id == buffer.PROFILE_ID
    assert failure.value.code in {"initializer-rank", "computed-initializer"}


@pytest.mark.parametrize(
    "raw",
    [
        """builtin.module { func.func @f(%seed:f32) {
          %zero=arith.constant 0:index %one=arith.constant 1:index %n=arith.constant 3:index
          %a=memref.alloc():memref<3xf32>
          scf.for %i=%zero to %n step %one { memref.store %seed,%a[%i]:memref<3xf32> scf.yield }
          memref.dealloc %a:memref<3xf32> func.return } }""",
        """builtin.module { func.func @f(%n:index,%seed:i32) {
          %zero=arith.constant 0:index %one=arith.constant 1:index %a=memref.alloc(%n):memref<?xi32>
          scf.for %i=%zero to %n step %one { memref.store %seed,%a[%i]:memref<?xi32> scf.yield }
          memref.dealloc %a:memref<?xi32> func.return } }""",
        """builtin.module { func.func @f(%seed:i32) {
          %zero=arith.constant 0:index %one=arith.constant 1:index %n=arith.constant 3:index
          %a=memref.alloc():memref<3xi32>
          %r=scf.for %i=%zero to %n step %one iter_args(%c=%seed)->(i32) {
            memref.store %c,%a[%i]:memref<3xi32> scf.yield %c:i32 }
          memref.dealloc %a:memref<3xi32> func.return } }""",
    ],
)
def test_buffer_float_dynamic_and_completion_sources_are_excluded(raw, monkeypatch):
    no_producers(monkeypatch)
    with pytest.raises(SourceOutsideProfile):
        buffer.admit_source(raw, options=OPTIONS)


def test_nan_source_is_excluded_but_dense_resource_limit_is_not(monkeypatch):
    raw = buffer_source().replace("@kernel(", "@kernel(", 1)
    raw = raw.replace("-> (i32, index) {", "-> (i32, index) attributes {tag = 0x7F800001 : f32} {", 1)
    with pytest.raises(SourceOutsideProfile) as failure:
        buffer.admit_source(raw, options=OPTIONS)
    assert failure.value.code == "scalar-nan"
    monkeypatch.setattr(memory_scalar_nan_admission, "MAX_DENSE_FLOAT_BYTES", 4)
    raw = buffer_source().replace(
        "-> (i32, index) {", "-> (i32, index) attributes {tag = dense<0.0> : tensor<2xf32>} {", 1
    )
    with pytest.raises(GateUnavailable):
        buffer.admit_source(raw, options=OPTIONS)


def test_unexpected_parser_or_selection_errors_are_not_source_exclusions(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(buffer, "parse_memory_module", fail)
    with pytest.raises(RuntimeError, match="unexpected parser"):
        buffer.admit_source(buffer_source(), options=OPTIONS)
    with pytest.raises(ValueError) as failure:
        buffer.admit_source(buffer_source(), options=Options())
    assert not isinstance(failure.value, SourceOutsideProfile)


def test_invalid_original_options_are_not_a_source_profile_exclusion():
    invalid = Options(buffer_allocator=ALLOCATOR)
    object.__setattr__(invalid, "tile_size", 1)
    with pytest.raises(ValueError) as failure:
        buffer.admit_source(buffer_source(), options=invalid)
    assert not isinstance(failure.value, SourceOutsideProfile)


def test_descriptive_model_metadata_does_not_control_source_admission(monkeypatch):
    monkeypatch.setitem(buffer.MODEL["bounds"], "source_bytes", 0)
    monkeypatch.setitem(buffer.MODEL["bounds"], "syntax_nodes", 0)
    assert buffer.admit_source(buffer_source(), options=OPTIONS) == buffer.PROFILE_ID


def test_malformed_original_MLIR_is_not_an_unsupported_source_profile():
    with pytest.raises(ParseError):
        buffer.admit_source("builtin.module { func.func @broken( }", options=OPTIONS)


@pytest.mark.parametrize("representation", ["instruction_mlir", "schedule_kernel"])
def test_actual_configuration_elision_sources_use_only_original_inputs(tmp_path, monkeypatch, representation):
    chosen, raw, _ = elision_source(tmp_path)
    if representation == "schedule_kernel":
        raw = Kernel("commands", (), (call("set", a=3), call("set", a=3), call("work"))).text()
    no_producers(monkeypatch)
    assert (
        configuration.admit_elision_source(raw, selection=chosen, representation=representation)
        == configuration.ELISION_ID
    )


def test_configuration_source_cannot_hide_changed_selected_inputs(tmp_path):
    chosen, raw, _ = elision_source(tmp_path)
    chosen.selected_contracts.facts_path.write_text("changed")
    with pytest.raises(ValueError) as failure:
        configuration.admit_elision_source(raw, selection=chosen, representation="instruction_mlir")
    assert not isinstance(failure.value, SourceOutsideProfile)


def test_malformed_original_kernel_is_not_a_source_exclusion(tmp_path):
    chosen = selection(tmp_path)
    raw = Kernel("commands", (), (Loop("i", 0, (call("work"),)),)).text()
    with pytest.raises(ValueError) as failure:
        configuration.admit_elision_source(raw, selection=chosen, representation="schedule_kernel")
    assert not isinstance(failure.value, SourceOutsideProfile)


@pytest.mark.parametrize("case", ["inner", "outer-and-inner", "indexed", "overlap"])
def test_qualified_nested_configuration_sources_admit_without_producer_replay(tmp_path, monkeypatch, case):
    chosen, kernel, _, _ = nested_source(tmp_path, case)
    before = kernel.text()
    no_producers(monkeypatch)
    assert configuration.admit_hoist_source(kernel, selection=chosen) == configuration.HOIST_ID
    assert kernel.text() == before


def test_static_configuration_hoist_and_original_C64_wide_domain(tmp_path, monkeypatch):
    chosen, kernel, _ = hoist_source(tmp_path)
    consumer = ConfigurationCConsumer(index_bits=64)
    wide = replace(kernel, body=(replace(kernel.body[1], extent=(1 << 63) - 1),))
    no_producers(monkeypatch)
    assert configuration.admit_hoist_source(kernel, selection=chosen) == configuration.HOIST_ID
    with pytest.raises(SourceOutsideProfile, match="count"):
        configuration.admit_hoist_source(wide, selection=chosen)
    assert configuration.admit_hoist_source(wide, selection=chosen, c_consumer=consumer) == configuration.HOIST_ID


@pytest.mark.parametrize("kind", ["dynamic-setter", "completion", "elision-loop"])
def test_configuration_known_source_grammar_exclusions_precede_any_producer(tmp_path, monkeypatch, kind):
    chosen = selection(tmp_path)
    first = call("set", a=3)
    if kind == "dynamic-setter":
        first = replace(first, args=tuple((key, Var("i") if key == "a" else value) for key, value in first.args))
    if kind == "completion":
        first = replace(first, produces="done")
    kernel = Kernel("commands", (), (Loop("i", 3, (first, call("work"))),))
    no_producers(monkeypatch)
    with pytest.raises(SourceOutsideProfile):
        if kind == "elision-loop":
            configuration.admit_elision_source(kernel.text(), selection=chosen, representation="schedule_kernel")
        else:
            configuration.admit_hoist_source(kernel, selection=chosen)


@pytest.mark.parametrize("value", ["0x7F800001 : f32", "0.0 : f32"])
def test_configuration_instruction_floating_metadata_is_outside_integer_profile(tmp_path, value):
    chosen = selection(tmp_path)
    raw = f"builtin.module {{ func.func @f() attributes {{tag = {value}}} {{func.return}} }}"
    with pytest.raises(SourceOutsideProfile):
        configuration.admit_elision_source(raw, selection=chosen, representation="instruction_mlir")
