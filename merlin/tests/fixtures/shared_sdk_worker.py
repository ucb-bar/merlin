"""Run neutral native regressions with only an exported Merlin library available."""

import json
import sys
from pathlib import Path

assert sys.dont_write_bytecode, "isolated fixture execution requires -B"
config = json.loads(Path(sys.argv[1]).read_text())
bundle = Path(config["bundle"])
work = Path(config["work"])
# -I -S supplies only interpreter paths. Add explicit third-party roots without
# executing .pth files, editable hooks, user sites or a Merlin source checkout.
assert not any(name == "merlin" or name.startswith("merlin.") for name in sys.modules)
sys.path[:0] = [str(bundle), config["fixtures"], *config["dependencies"]]

import test_async_schedule_pipeline as asynchronous  # noqa: E402
import test_configured_pointwise as configured_pointwise  # noqa: E402
import test_pointwise_source as pointwise  # noqa: E402
import test_selected_dense_contracts as dense  # noqa: E402
import test_selected_fusion_contracts as fusion  # noqa: E402
import test_selected_program_contracts as program  # noqa: E402
import test_selected_row_contracts as rows  # noqa: E402
import test_selected_schedule_contracts as configuration  # noqa: E402
import test_selected_spill_contracts as spills  # noqa: E402
import test_selected_storage_contracts as storage  # noqa: E402
import test_spilled_pointwise_source as source_spills  # noqa: E402

from merlin.compile.scheduling.library_sources import shared_library_sources  # noqa: E402

observations, completed = [], []
current = None


def record(name, value):
    observations.append({"case": current, "name": name, "value": json.loads(value)})


def run(name, test, **kwargs):
    global current
    current = name
    path = work / name
    path.mkdir()
    if "native_fixture" in kwargs:
        kwargs["native"] = kwargs.pop("native_fixture").__wrapped__(path)
    test(path, **kwargs)
    completed.append(name)


run("dense_rectangular", dense.test_selected_facts_and_semantics_drive_native_paired_execution,
    record_property=record, tile=(2, 3, 4), widths=(1, 1, 4), shared=False, shape=(5, 14, 13))
run("dense_aliased", dense.test_selected_facts_and_semantics_drive_native_paired_execution,
    record_property=record, tile=(3, 2, 5), widths=(2, 1, 4), shared=True, shape=(7, 9, 16))
run("storage", storage.test_selected_allocation_realizes_reuse_and_preserves_complete_native_state,
    native_fixture=storage.native, record_property=record, width=13, separate=True, word_format=">BBBIII")
run("completion", storage.test_async_lifetimes_use_actual_waits_through_ordinary_final_emission,
    native_fixture=storage.native)
run("async_allocation", asynchronous.test_async_ordinary_allocation_executes_complete_graph_and_carried_state,
    native_fixture=storage.native, record_property=record, width=13, separate=True,
    word_format=">BBBIII", segmented=True)
run("fusion", fusion.test_selected_fusion_executes_complete_graph_and_checks_realized_transfers,
    native_fixture=fusion.native, record_property=record, shape=(2, 5, 11, 3), width=2, separate=True)
run("fusion_state", fusion.test_recurrent_graph_invocations_preserve_complete_state_and_mutable_weights,
    native_fixture=fusion.native)
run("spills", spills.test_selected_spills_execute_complete_outputs_and_carried_state,
    native_fixture=storage.native, record_property=record, width=16, word_format=">BBBIII")
path = work / "configuration"
path.mkdir()
current = "configuration"
configuration.test_two_inputs_drive_ordinary_cleanup_and_complete_native_output(
    path, configuration.native.__wrapped__(path), record, width=8, endian=">", ordered=True
)
completed.append(current)
run("program", program.test_ordinary_final_checks_use_selected_limits_and_preserve_complete_state,
    native_fixture=program.native, record_property=record, width=7, separate=False, word_format="<BBBQQQ")
# This selected source algorithm uses f64 libm, so it needs neither the f32
# exponential object nor the unexported whole-model host compiler.
execute_row = rows.execute_native
rows.execute_native = lambda *args, **kw: execute_row(*args, **kw, compiler=config["compiler"])
run("rows", rows.test_selected_complete_rows_execute_full_native_outputs,
    exp_object=None, record_property=record, dtype="f64", separate=True, index_bits=64, mask=rows.CausalRowMask(2))

def source_dense_from_export(path):
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, func, linalg, tensor
    from xdsl.parser import Parser

    from merlin.compile.scheduling.dense_source import schedule_source_dense
    context = Context()
    for dialect in (arith.Arith, builtin.Builtin, func.Func, linalg.Linalg, tensor.Tensor):
        context.load_dialect(dialect)
    source = Path(config["fixtures"]).parent / "fixtures/shared_dense_source.mlir"
    module = Parser(context, source.read_text()).parse_module()
    operation = next(op for op in module.walk() if op.name == "linalg.generic")
    selected, _, storage, adapter = dense.inputs(path, shape=(2, 3, 4))
    result = schedule_source_dense(module, operation, selected_contracts=selected, storage=storage,
        adapter=adapter, name="exported_source", static_budget=80, external_buffers_disjoint=True,
        verification_timeout_ms=20_000)
    assert result is not None
    result.revalidate()
    assert result.compiled.realization["pipeline"]["library_sources_sha256"] == shared_library_sources()


run("source_dense", source_dense_from_export)


def dense_float_io_from_export(path):
    from merlin.transforms.certificates import ir_text, parse_ir

    module = parse_ir('''builtin.module {
      %c = arith.constant dense<[0xFF800000, 0x7F800001, 0x80000000]> : tensor<3xf32>
    }''')
    expected = bytes.fromhex("000080ff0100807f00000080")
    assert module.body.block.first_op.value.data.data == expected
    assert parse_ir(ir_text(module)).body.block.first_op.value.data.data == expected


run("dense_float_io", dense_float_io_from_export)


def source_rows_from_export(path):
    from test_source_rows import case
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, func, linalg, math, scf, tensor
    from xdsl.parser import Parser

    from merlin.compile.scheduling.row_softmax import compile_source_rows
    from merlin.transforms.certificates import ir_text

    _, _, original, _, kwargs = case(path)
    context = Context()
    for dialect in (arith.Arith, builtin.Builtin, func.Func, linalg.Linalg, math.Math, scf.Scf, tensor.Tensor):
        context.load_dialect(dialect)
    module = Parser(context, ir_text(original)).parse_module()
    body = next(op for op in module.walk() if type(op) is func.FuncOp).body.block
    result = compile_source_rows(module, body.last_op.arguments[0].owner, **kwargs)
    result.revalidate()
    assert result.source.operations[0].value.data.data == bytes.fromhex("000080ff") * 2
    assert result.compiled.report["pipeline"]["library_sources_sha256"] == shared_library_sources()
    record("scope", json.dumps("compile-only typed source binding; primitive execution tested separately"))


run("source_rows_compile", source_rows_from_export)

def source_pointwise_from_export(path):
    from types import SimpleNamespace

    pointwise.test_source_arithmetic_survives_fusion_and_allocation(
        path, execute=pointwise.execute.__wrapped__(path, SimpleNamespace()), m=2, n=11, separate=True)


run("source_pointwise", source_pointwise_from_export)


def configured_pointwise_from_export(path):
    from types import SimpleNamespace

    execute = pointwise.execute.__wrapped__(
        path, SimpleNamespace(param=dict(source=configured_pointwise.EXECUTOR, counters=5))
    )
    configured_pointwise.test_configuration_cleanup_and_fusion_preserve_actual_values_and_state(
        path, execute=execute, separate=True
    )


run("configured_source_pointwise", configured_pointwise_from_export)
for name, m, n, width, word_format in (
    ("source_spills_narrow", 1, 8, 3, "<BBBQQQ"),
    ("source_spills_wide", 1, 47, 16, ">BBBIII"),
):
    run(name, source_spills.test_source_spill_repair_matches_legal_baseline_and_complete_native_state,
        native_fixture=storage.native, m=m, n=n, width=width, word_format=word_format)


def wide_source_fragments_from_export(path):
    from types import SimpleNamespace

    source_spills.test_wide_source_fragments_preserve_element_coordinates(
        path, execute=pointwise.execute.__wrapped__(path, SimpleNamespace()), misaligned=False)


run("source_fragments_i32", wide_source_fragments_from_export)

origins = {}
for name, module in tuple(sys.modules.items()):
    if name == "merlin" or name.startswith("merlin."):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(bundle), (name, path)
        origins[name] = str(path)
        assert all(Path(p).resolve().is_relative_to(bundle) for p in getattr(module, "__path__", ()))
sources = shared_library_sources()
manifest = json.loads((bundle / "shared-library.json").read_text())
assert sources == {member["module"]: member["sha256"] for member in manifest["members"] if member["module"] is not None}
result = {"schema": "merlin.shared_sdk_execution.v1", "completed": completed,
          "bytecode_disabled": sys.dont_write_bytecode,
          "library_sha256": manifest["library_sha256"], "source_owners": sources,
          "module_origins": origins, "observations": observations,
          "scope": ("neutral native regressions and named compile-only cases; "
                    "no real target or accelerator timing qualification")}
Path(config["result"]).write_text(json.dumps(result, indent=2) + "\n")
print("completed", len(completed), "isolated cases from", len(sources), "source owners")
