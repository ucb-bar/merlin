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

import test_selected_dense_contracts as dense  # noqa: E402
import test_selected_fusion_contracts as fusion  # noqa: E402
import test_selected_program_contracts as program  # noqa: E402
import test_selected_row_contracts as rows  # noqa: E402
import test_selected_schedule_contracts as configuration  # noqa: E402
import test_selected_spill_contracts as spills  # noqa: E402
import test_selected_storage_contracts as storage  # noqa: E402

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
run("completion", storage.test_async_lifetimes_use_actual_waits_and_ordinary_emission_remains_disabled,
    native_fixture=storage.native)
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
          "scope": "neutral native regressions only; no real target/model or accelerator timing qualification"}
Path(config["result"]).write_text(json.dumps(result, indent=2) + "\n")
print("completed", len(completed), "isolated cases from", len(sources), "source owners")
