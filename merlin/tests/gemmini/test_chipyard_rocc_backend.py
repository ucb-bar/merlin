"""The generic chipyard RoCC backend serves gemmini from DATA, and refuses rather than guesses.

Pure-module properties (core plugin references, the ISA-headers spec, the generated facts header)
run everywhere. Properties of the per-target backend instance need the gemmini support provider
selected on ``MERLIN_TARGET_PATH`` and skip otherwise; nothing here launches a simulator.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import selected_driver
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import isa_header_gen, isa_headers_spec, plugins

pytestmark = pytest.mark.target("gemmini")

SUPPORT = repo_root() / "examples/gemmini/support"
GENERIC = "merlin.runtime.backends.chipyard_rocc"


# --- core plugin references --------------------------------------------------------------------------
def test_a_core_module_reference_resolves_only_to_installed_merlin_source(tmp_path):
    path = plugins.core_module_path(GENERIC)
    assert path is not None and path.name == "chipyard_rocc.py" and plugins.is_core_module_source(path)
    assert plugins.core_module_path("os.path") is None  # not Merlin
    assert plugins.core_module_path("merlin._oot_backends.gemmini") is None  # synthetic namespace
    assert plugins.core_module_path("merlin/runtime/backends/chipyard_rocc.py") is None  # a path, not a module
    assert plugins.core_module_path(GENERIC, "file") is None  # a data reference never means code
    root = tmp_path / "provider"
    root.mkdir()
    assert plugins.validate({"backend": GENERIC}, root=root) == []
    assert plugins.validate({"backend": "merlin.no.such.module"}, root=root)


def test_the_template_import_registers_nothing():
    module = importlib.import_module(GENERIC)
    assert module.TARGET_NAME is None and module.GSIM_EMU_ENV is None
    with pytest.raises(module.ChipyardRoccError, match="generic chipyard RoCC template"):
        module.platform_dram_base()


# --- the ISA-headers spec ----------------------------------------------------------------------------
def _spec_tree(tmp_path: Path, *, extra: dict[str, bytes] | None = None):
    root = tmp_path / "checkout" / "tree"
    files = {"env/encoding.h": b"#define X 1\n", "common/crt.S": b"/* crt */\n", "common/test.ld": b"/* ld */\n"}
    files.update(extra or {})
    for relative, data in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(data)
    spec = {
        "schema": isa_headers_spec.SCHEMA,
        "target": "t",
        "source": {"commit": "0" * 40, "root_env": "TEST_ISA_ROOT", "path": "tree"},
        "include_roots": ["env", "common"],
        "crt": {"sources": ["common/crt.S"], "link_script": "common/test.ld", "stack_bytes_per_hart": 4096},
        "kernel_stack": {"max_static_bytes": 2048},
        "cflags": ["-O2"],
        "files": {
            k: hashlib.sha256(v).hexdigest()
            for k, v in files.items()
            if k in ("env/encoding.h", "common/crt.S", "common/test.ld")
        },
        "excluded_from_include_path": ["lib.h"],
        "evidence_files": {"params.h": "a" * 64},
    }
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path, root


def test_the_spec_resolves_by_variable_and_verifies_bytes(tmp_path, monkeypatch):
    spec, root = _spec_tree(tmp_path)
    monkeypatch.setenv("TEST_ISA_ROOT", str(root.parent))
    resolved = isa_headers_spec.load("t", spec)
    assert resolved.include_roots == (root / "env", root / "common")
    assert resolved.link_script == root / "common/test.ld"
    assert isa_headers_spec.declared_files("t", spec)["params.h"] == "a" * 64  # evidence, no checkout needed
    (root / "common/crt.S").write_bytes(b"/* different runtime */\n")
    with pytest.raises(isa_headers_spec.IsaHeadersError, match="DIFFERENT bytes"):
        isa_headers_spec.load("t", spec)


def test_an_excluded_header_on_the_include_path_is_refused(tmp_path, monkeypatch):
    spec, root = _spec_tree(tmp_path, extra={"common/lib.h": b"void vendor_routine(void);\n"})
    monkeypatch.setenv("TEST_ISA_ROOT", str(root.parent))
    with pytest.raises(isa_headers_spec.IsaHeadersError, match="excluded"):
        isa_headers_spec.load("t", spec)


def test_the_shipped_spec_keeps_the_vendor_library_off_the_include_path():
    spec = yaml.safe_load((SUPPORT / "isa_headers.yaml").read_text(encoding="utf-8"))
    roots = set(spec["include_roots"])
    assert roots.isdisjoint({".", "include", "riscv-tests"})
    assert not any(name.startswith(("include/", "rocc-software/")) for name in spec["files"])
    assert {"include/gemmini.h", "gemmini.h"} <= set(spec["excluded_from_include_path"])


# --- the generated facts header ----------------------------------------------------------------------
def _facts(legal, names, complete=False):
    return {
        "facts": {
            "target": "t",
            "interfaces": [
                {
                    "name": "funct_decode_table",
                    "legal_funct": legal,
                    "names": names,
                    "custom_opcode": 123,
                    "complete_isa": complete,
                }
            ],
            "arrays": [{"name": "mesh", "rows": 4, "cols": 4}],
            "memories": [{"name": "scratchpad.mem", "banks": 2, "depth": 8}],
            "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
        }
    }


def test_the_header_states_decoder_facts_only_and_is_deterministic():
    facts = _facts([0, 2, 126], {"0": "CONFIG_CMD", "2": "LOAD_CMD", "25": "HEADER_ONLY", "126": "COUNTER_OP"})
    contract = {
        "encoding": {"addr_len": 32, "rocc_custom_slot": 3},
        "memory_model": {"dma": {"max_transfer_bytes": 64}},
    }
    first = isa_header_gen.render("t", facts=facts, contract=contract, facts_sha256="f" * 64)
    assert first == isa_header_gen.render("t", facts=facts, contract=contract, facts_sha256="f" * 64)
    assert "#define T_FUNCT_LOAD_CMD 2" in first and "#define T_FUNCT_COUNTER_OP 126" in first
    assert "HEADER_ONLY" not in first, "a code the decoder lacks must not be emitted"
    assert "#define T_MESH_ROWS 4" in first and "typedef int8_t t_input_t;" in first
    assert "#define T_ADDR_LEN 32" in first and "#define T_DMA_MAX_TRANSFER_BYTES 64" in first
    for forbidden in ("matmul", "conv2d", "conv_"):
        assert forbidden not in first.lower()


def test_missing_decoder_facts_refuse():
    with pytest.raises(isa_header_gen.IsaHeaderError, match="funct_decode_table"):
        isa_header_gen.render("t", facts={"facts": {}}, contract={}, facts_sha256="0" * 64)


def test_the_header_is_written_once_per_content(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(_facts([0], {"0": "CONFIG_CMD"})), encoding="utf-8")
    path = isa_header_gen.materialize("t", "t_isa.h", facts_path=facts, contract={})
    assert path.is_file() and path == isa_header_gen.materialize("t", "t_isa.h", facts_path=facts, contract={})
    with pytest.raises(isa_header_gen.IsaHeaderError, match="plain .h"):
        isa_header_gen.materialize("t", "../escape.h", facts_path=facts, contract={})


# --- the per-target instance (needs the selected provider) -------------------------------------------
@pytest.fixture(scope="module")
def backend():
    selected_driver.require_support("gemmini")
    from merlin.runtime.backends import base

    module = base.get_backend("gemmini")
    if module.__file__ != str(plugins.core_module_path(GENERIC)):
        pytest.skip("the selected gemmini support is not the generic data provider")
    return module


def test_the_instance_serves_its_target_with_derived_overrides(backend):
    assert backend.__name__ == "merlin._oot_backends.gemmini" and backend.TARGET_NAME == "gemmini"
    assert (backend.GSIM_EMU_ENV, backend.VERILATOR_ENV, backend.GSIM_MAXCYCLES_ENV) == (
        "MERLIN_GEMMINI_GSIM_EMU",
        "MERLIN_GEMMINI_VERILATOR",
        "MERLIN_GEMMINI_GSIM_MAXCYCLES",
    )
    assert set(backend.ORACLE) == {"spike", "verilator", "gsim"}
    assert backend.ORACLE["gsim"]["derived_from_rtl"] and not backend.ORACLE["spike"]["derived_from_rtl"]
    assert set(backend.EXECUTION_CAPABILITIES) <= {"whole_program_kernel_abi", "warm_single_counter_region_cycles"}


def test_parse_output_reads_out_nd_and_refuses_a_short_frame(backend):
    outputs, raw = backend.parse_output("OUT_ND z 3 2 1 2 1 2 3 4\nOUT y 1 2 5 6\nMETRIC cycles 9\nMETRIC bad\nDONE\n")
    assert outputs == {"y": [[5, 6]], "z": [[[1, 2]], [[3, 4]]]} and raw == {"cycles": 9}
    with pytest.raises(backend.ChipyardRoccError, match="expected 4 values"):
        backend.parse_output("OUT_ND z 2 2 2 1 2 3\nDONE\n")
    with pytest.raises(backend.ChipyardRoccError, match="DONE"):
        backend.parse_output("OUT y 1 1 7\n")


def test_runtime_environment_pins_engines_without_touching_the_process(backend, tmp_path):
    binaries = {}
    for engine in ("gsim", "verilator"):
        binaries[engine] = tmp_path / engine
        binaries[engine].write_bytes(engine.encode())
    environment = {"KEEP": "1", "MERLIN_GEMMINI_GSIM_MAXCYCLES": "5"}
    configured = backend.runtime_environment(binaries=binaries, gsim_max_cycles=None, environment=environment)
    assert configured["MERLIN_GEMMINI_GSIM_EMU"] == str(binaries["gsim"].resolve())
    assert configured["MERLIN_GEMMINI_VERILATOR"] == str(binaries["verilator"].resolve())
    assert "MERLIN_GEMMINI_GSIM_MAXCYCLES" not in configured and configured["KEEP"] == "1"
    assert environment == {"KEEP": "1", "MERLIN_GEMMINI_GSIM_MAXCYCLES": "5"}
    with pytest.raises(ValueError, match="positive"):
        backend.runtime_environment(binaries=binaries, gsim_max_cycles=0, environment={})


def test_readout_facts_come_from_the_contract_bound_to_the_spec(backend, monkeypatch):
    abi = backend.readout_scalar_abi()
    assert abi["schema"] == "scalar_narrow_readout_contract_v1" and (abi["clamp_min"], abi["clamp_max"]) == (-128, 127)
    assert abi["provenance"]["scope"] == "contract_declared"
    assert {row["selector"] for row in backend.readout_epilogue_capability()} == {"i8", "i32"}
    real = isa_headers_spec.declared_files
    monkeypatch.setattr(
        isa_headers_spec, "declared_files", lambda *a, **k: {**real(*a, **k), "include/gemmini_params.h": "0" * 64}
    )
    with pytest.raises(backend.ChipyardRoccError, match="no longer describes"):
        backend.readout_scalar_abi()


def test_a_selected_provider_without_a_readout_hook_is_refused(backend, monkeypatch):
    from merlin.targetgen import readout_facet

    monkeypatch.delattr(backend, "readout_epilogue_capability")
    with pytest.raises(readout_facet.ReadoutSupportError, match="readout_epilogue_capability"):
        readout_facet.epilogue_readouts("gemmini")


def test_the_build_recipe_has_no_vendor_library_on_its_include_path(backend):
    try:
        recipe = backend.harness_build_recipe()
    except backend.ChipyardRoccError as exc:
        pytest.skip(f"the pinned bare-metal runtime is not available here: {exc}")
    for root in recipe.include_roots:
        assert not (root / "gemmini.h").exists() and not (root / "include" / "gemmini.h").exists(), root
    generated = backend.generated_isa_header()
    assert generated in recipe.header_dependencies and generated.parent in recipe.include_roots
    assert recipe.kernel_stack_frame.entry_symbol == "gemmini_kernel"


def test_the_console_write_is_generated_from_data_and_linked(tmp_path, monkeypatch):
    """The binary readback transport's length-taking write is generated from the spec, never copied."""
    from merlin.runtime.backends import _chipyard_console

    block = {
        "protocol": "htif_syscall",
        "write_syscall": 64,
        "host_symbols": {"request": "th", "response": "fh"},
        "symbol": "w",
    }
    text = _chipyard_console.render(block)
    assert "int w(const void *data, size_t length)" in text and "extern volatile uint64_t th;" in text
    assert "mailbox[0] = 64;" in text and '#include "' not in text
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    first = _chipyard_console.materialize("t", block)
    assert first == _chipyard_console.materialize("t", block) and first.read_text() == text
    spec = yaml.safe_load((SUPPORT / "isa_headers.yaml").read_text(encoding="utf-8"))
    assert spec["console_write"]["symbol"] == "printbuf"


def test_the_functional_trace_invocation_is_the_spike_command_with_a_commit_log(backend, monkeypatch, tmp_path):
    """The executed-command profile runs the same functional model run_elf does, plus -l --log-commits."""
    seen = {}

    class _Proc:
        returncode, stdout, stderr = 0, "DONE\n", ""

    real_run = backend.subprocess.run

    def fake_run(cmd, **kw):
        if any(str(arg).startswith("--extension=") for arg in cmd):  # only the engine launch is faked
            seen.update(cmd=list(cmd), env=kw.get("env"))
            return _Proc()
        return real_run(cmd, **kw)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    elf = tmp_path / "k.elf"
    elf.write_bytes(b"\x7fELF")
    backend.run_elf(elf, simulator="spike", timeout=5)
    argv, env = backend.functional_trace_invocation(elf)
    assert argv == [*seen["cmd"][:-1], "-l", "--log-commits", str(elf)]
    assert env["LD_LIBRARY_PATH"] == seen["env"]["LD_LIBRARY_PATH"]
    assert f"--extension={backend.SPIKE_EXTENSION_NAME}" in argv


def test_a_candidate_text_past_the_startup_branch_range_still_links(backend, tmp_path):
    """The CRT reaches _init with a +-1 MiB branch; 2 MiB of candidate code must not break the link,
    and large operand arrays land after all code. Both are properties of the generic recipe."""
    import subprocess

    try:
        recipe = backend.harness_build_recipe().with_effective_abi()
    except backend.ChipyardRoccError as exc:
        pytest.skip(f"the pinned bare-metal runtime is not available here: {exc}")
    assert [p.name for p in recipe.link_first] == ["syscalls.c"]
    (tmp_path / "harness.c").write_text(
        "#include <stdio.h>\nstatic const signed char big[3136 * 576] = {1};\n"
        "extern void kernel(const signed char *);\n"
        'int main(void) { kernel(big); printf("DONE\\n"); return 0; }\n'
    )
    (tmp_path / "kernel.c").write_text('__asm__(".text\\n.globl kernel\\nkernel:\\n.space 2097152\\nret\\n");\n')
    objects = []
    for source in recipe.ordered_link_sources([tmp_path / "harness.c", tmp_path / "kernel.c"]):
        out = tmp_path / (Path(source).stem + ".o")
        done = subprocess.run(recipe.compile_command(source=Path(source), output=out), capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-1000:]
        objects.append(out)
    elf = tmp_path / "prog.elf"
    done = subprocess.run(recipe.link_command(objects=objects, output=elf), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-1500:]
    sizes = subprocess.run(["readelf", "-SW", str(elf)], capture_output=True, text=True).stdout
    addr = {}
    for line in sizes.splitlines():
        parts = line.replace("[", " ").replace("]", " ").split()
        if len(parts) > 4 and parts[1] in (".text", ".rodata", ".bss"):
            addr[parts[1]] = int(parts[3], 16)
    assert addr[".text"] < addr[".rodata"] < addr[".bss"], addr


def test_a_gsim_cycle_cap_stop_is_a_timeout_that_keeps_its_console(backend, monkeypatch, tmp_path):
    """GSIM's +max-cycles stop (exit 124) is a budget limit, not a crash: it carries the partial console
    (with the METRIC line printed before the output frame) and says 'timed out'."""

    class _Proc:
        returncode, stdout, stderr = 124, "METRIC cycles 1234\nOUT Y0 2 2 1 2", "GSIM timeout: no RTL/TSI completion"

    real_run = backend.subprocess.run
    monkeypatch.setattr(backend, "_gsim_argv", lambda elf, **kw: ["inert-gsim", str(elf)])
    monkeypatch.setattr(
        backend.subprocess, "run", lambda cmd, **kw: _Proc() if cmd and cmd[0] == "inert-gsim" else real_run(cmd, **kw)
    )
    elf = tmp_path / "k.elf"
    elf.write_bytes(b"\x7fELF")
    with pytest.raises(backend.SimulatorTimeout, match="timed out after") as info:
        backend.run_elf(elf, simulator="gsim", timeout=5)
    assert "METRIC cycles 1234" in info.value.stdout


def test_the_caller_layout_is_the_dense_logical_interface_and_the_inspection_admits_it(backend, tmp_path):
    """A memory readback admits output storage through the provider's caller-layout projection. The
    generic backend is installed core, named by the data-only provider's contract, so the inspection
    pins its bytes as core -- and the projection is exactly the harness's dense row-major buffers."""
    from merlin_experiments.phase1.feedback.caller_layout import inspect_caller_layout

    from merlin.targetgen.contract.harness_render import explicit_whole_program, resolve
    from merlin.targetgen.rtl.facts import rtl_facts_path

    cb = {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            "W": {"shape": [8, 4], "dtype": "i8", "role": "weight"},
            "A": {"shape": [2, 3, 8], "dtype": "i8", "role": "input"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "R"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "R", "dst": "acc"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc", "dst": "Y"},
                "attributes": {"epilogue": [], "output_dtype": "i32"},
            },
            {"opcode": "EVICT", "operands": {"handle": "R"}},
        ],
    }
    wp = explicit_whole_program(cb, resolve("gemmini")[0])
    submission = tmp_path / "sub"
    submission.mkdir()
    (submission / "command_buffer.json").write_text(json.dumps(wp))
    receipt = inspect_caller_layout(
        submission=submission, command_buffer_member="command_buffer.json", target="gemmini",
        facts_path=rtl_facts_path("gemmini"),
    )
    assert receipt["policy"] == {"mode": "legacy_aligned_row_major_v1", "row_alignment_elements": 1}
    rows = {row["tensor"]: row for row in receipt["tensors"]}
    assert [row["tensor"] for row in receipt["tensors"]] == [a["tensor"] for a in wp["kernel_abi"]["args"]]
    assert rows["A"]["physical_extents"] == [6, 8] and rows["A"]["logical_strides_elements"] == [24, 8, 1]
    y_shape = wp["tensors"]["Y"]["shape"]
    assert rows["Y"]["logical_shape"] == y_shape and rows["Y"]["offset_elements"] == 0
    assert rows["Y"]["storage_elements"] == rows["Y"]["physical_extents"][0] * y_shape[-1]
    with pytest.raises(backend.ChipyardRoccError, match="dense"):
        backend.describe_caller_layout({**wp, "params": {"storage_encodings": {}}}, target="gemmini",
                                       facts={"inputs": {"target": "gemmini"}})


def test_verilator_loads_the_program_like_gsim(backend, monkeypatch, tmp_path):
    """Both RTL engines load by +loadmem: over the serial link, Verilator started the kernel window from
    different cache state (1096 vs 1112 cycles for one ELF) and spent most of a small run loading."""
    seen = {}

    class _Proc:
        returncode, stdout, stderr = 0, "METRIC cycles 1\nDONE\n", ""

    def run(cmd, **kw):
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(backend, "verilator_path", lambda: tmp_path / "simulator")
    monkeypatch.setattr(backend.subprocess, "run", run)
    elf = tmp_path / "k.elf"
    elf.write_bytes(b"\x7fELF")
    backend.run_elf(elf, simulator="verilator", timeout=5)
    assert seen["cmd"] == [str(tmp_path / "simulator"), str(elf), f"+loadmem={elf}"]
