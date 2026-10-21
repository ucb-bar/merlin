"""Ordinary multi-stage ELF building and complete independent Spike observations."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from test_session_freestanding import capture

from merlin.llvmlower import session_freestanding as F
from merlin.runtime.backends import session_build_guard as G
from merlin.runtime.backends import spike_session as B
from merlin.runtime.backends import spike_session_stage as S


def test_guard_retains_actual_implementation_and_defaults(monkeypatch):
    snapshot = G.snapshot((B, S, G), (None, None))
    G.check(snapshot, (None, None))
    monkeypatch.setattr(S, "build_stage", lambda *a, **k: None)
    with pytest.raises(ValueError, match="implementation"):
        G.check(snapshot, (None, None))


def test_stage_routes_require_exact_roster():
    with pytest.raises(ValueError, match="routing"):
        S.freeze_routes([], 0)
    with pytest.raises(ValueError, match="DeviceRouting"):
        S.freeze_routes((object(),), 1)


@pytest.mark.parametrize("mutation", [None, "host_object", "earlier_stage"])
def test_actual_two_stage_spike_elf(tmp_path, monkeypatch, mutation):
    gcc = Path(os.environ.get("MERLIN_RISCV_GCC", "/unselected-gcc"))
    spike = Path(os.environ.get("MERLIN_SPIKE", "/unselected-spike"))
    if not gcc.is_file() or not spike.is_file():
        pytest.skip("explicit original RISC-V compiler and Spike required")
    root = capture(tmp_path / "source")
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    observer = tmp_path / "observer.c"
    observer.write_text("""#include <stddef.h>
#include <stdint.h>
#include "htif.h"
int merlin_session_observe(int program,long step,int output,const void *data,size_t bytes,void *context) {
  (void)context;
  if(!data || bytes!=2*sizeof(float) || program<0 || program>1 || output<0 || output>3) return 1;
  const float *values=data;
  htif_puts("SESSION_OUT "); htif_putd(program); htif_putc(' '); htif_putd(step);
  htif_putc(' '); htif_putd(output);
  for(int i=0;i<2;i++) {
    int value=(int)values[i]; if((float)value!=values[i]) return 2;
    htif_putc(' '); htif_putd(value);
  }
  htif_putc('\\n'); return 0;
}
""")
    # Observer code is original verification-only input; never a device compiler input.
    harness = B.M._harness_dir()
    original_builder = B.D.build_prepared_device_objects

    def selected_builder(device, work, **kwargs):
        result = original_builder(device, work, **kwargs)
        if mutation == "host_object":
            (work / "model.o").write_bytes(b"rewritten after actual host compile")
        if mutation == "earlier_stage" and work.name == "stage_1":
            (work.parent / "stage_0/namespaced.o").write_bytes(b"rewritten by later stage")
        return result

    monkeypatch.setattr(B.D, "build_prepared_device_objects", selected_builder)

    def build():
        return B.build_session(
            root,
            tmp_path / "build",
            limits=F.SessionMemoryLimits(
                weights_base=0x81000000,
                address_limit=0x82000000,
                max_weights_bytes=1 << 20,
                max_static_io_bytes=1 << 20,
                max_calls=4,
            ),
            arena_bytes=1 << 20,
            stack_bytes=1 << 20,
            dram_base=0x80000000,
            dram_bytes=0x40000000,
            devices=(None, None),
            observer_sources=(observer,),
            cflags_override=[
                "-march=rv64gc",
                "-mabi=lp64d",
                "-mcmodel=medany",
                "-O2",
                "-ffreestanding",
                "-fno-builtin",
                "-I",
                str(harness),
            ],
        )

    if mutation is not None:
        with pytest.raises(ValueError, match="changed"):
            build()
        assert not (tmp_path / "build/session.elf").exists()
        return
    result = build()
    argv = [str(spike), "--isa=rv64gc", "--priv=m", "-p1", "-m0x80000000:0x40000000", result["elf"]]
    proc = subprocess.run(argv, text=True, capture_output=True, timeout=120)
    (tmp_path / "quiet.stdout").write_text(proc.stdout)
    (tmp_path / "quiet.stderr").write_text(proc.stderr)
    (tmp_path / "quiet-command.json").write_text(json.dumps(argv))
    assert proc.returncode == 0, proc.stderr
    assert "SESSION_DONE" in proc.stdout
    actual = []
    for line in proc.stdout.splitlines():
        if line.startswith("SESSION_OUT "):
            actual.append(tuple(map(int, line.split()[1:])))
    expected = []
    state = [1, 2]
    for p, frames, weights in [(0, [[3, 4]], [5, 7]), (1, [[2, 4], [6, 8], [10, 12]], [-2, 3])]:
        for step, frame in enumerate(frames):
            next_state = [state[i] + frame[i] + weights[i] for i in range(2)]
            for out, values in enumerate((next_state, frame, state, weights)):
                expected.append((p, step, out, *values))
            state = next_state
    assert actual == expected
    timing = [line for line in proc.stdout.splitlines() if line.startswith("SESSION_CYCLES ")]
    assert len(timing) == 7
    assert all(int(line.split()[-1]) > 0 for line in timing)
    assert all(p.read_bytes() == data for p, data in original.items())
    assert len(result["stages"]) == 2
    assert all(Path(row["recipe"]).is_file() for row in result["stages"])


@pytest.mark.parametrize("boundary", ["freeze", "current"])
@pytest.mark.parametrize("mutation", ["new_source", "environment", "tool", "settings", "checker"])
def test_owner_mutation_refuses_before_materialization(tmp_path, monkeypatch, boundary, mutation):
    root = capture(tmp_path / "source")
    observer = tmp_path / "observer.c"
    observer.write_text("/* verification only */")
    gcc, clang, archive = (tmp_path / name for name in ("gcc", "clang", "libm.a"))
    for path in (
        gcc,
        clang,
        archive,
        tmp_path / "riscv64-unknown-elf-ld",
        tmp_path / "llvm-nm",
        tmp_path / "llvm-objcopy",
    ):
        path.write_text("original selected tool")
    flags = ["-O2"]
    monkeypatch.setattr(B.M._spike, "gcc_path", lambda: gcc)
    monkeypatch.setattr(
        B.M,
        "selected_model_compiler_plan",
        lambda **kwargs: dict(observation=dict(compiler_resolved=str(clang)), gcc_cflags=[], clang_cflags=flags),
    )
    monkeypatch.setattr(
        B.M, "_selected_libm_archive", lambda *args: (archive, B.sha256_file(archive), B.sha256_file(gcc))
    )
    original_freeze = S.freeze_routes
    original_require = B.D.require_device_offload_selection
    calls = []

    def mutate():
        if mutation == "new_source":
            (root / "unexpected-control.json").write_text("{}")
        elif mutation == "environment":
            monkeypatch.setenv("MERLIN_CLANG", "foreign")
        elif mutation == "tool":
            clang.write_text("foreign")
        elif mutation == "checker":
            monkeypatch.setattr(G, "check", lambda *args: None)
        else:
            flags.append("-ffast-math")

    def freeze(devices, count):
        if boundary == "freeze":
            mutate()
        return original_freeze(devices, count)

    def require(device, frozen):
        if boundary == "current":
            mutate()
        return original_require(device, frozen)

    monkeypatch.setattr(S, "freeze_routes", freeze)
    monkeypatch.setattr(B.D, "require_device_offload_selection", require)
    monkeypatch.setattr(F, "generate", lambda *a, **k: calls.append("generated"))
    with pytest.raises(ValueError, match="changed"):
        B.build_session(
            root,
            tmp_path / "build",
            limits=F.SessionMemoryLimits(
                weights_base=0x81000000,
                address_limit=0x82000000,
                max_weights_bytes=1024,
                max_static_io_bytes=1024,
                max_calls=4,
            ),
            arena_bytes=1024,
            stack_bytes=1024,
            dram_base=0x80000000,
            dram_bytes=0x40000000,
            devices=(None, None),
            observer_sources=(observer,),
            cflags_override=flags,
        )
    assert not calls
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("foreign", [False, True])
def test_selected_stages_require_same_live_final_image_owner(tmp_path, foreign):
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    def evaluator(**kwargs):
        raise AssertionError("no ELF evaluation during source binding")

    import inspect

    owner_file = Path(inspect.getsourcefile(evaluator)).resolve()
    pins = ((str(owner_file), B.sha256_file(owner_file)),)
    service = LinkedElfAdmissionService("neutral", evaluator, pins)
    other = LinkedElfAdmissionService("neutral", evaluator, pins) if foreign else service
    routes = tuple(
        DeviceRouting(
            device="neutral",
            package_dir=tmp_path,
            operand_dtype="i8",
            accum_dtype="i32",
            select=lambda group: True,
            linked_elf_admission=item,
        )
        for item in (service, other)
    )
    if foreign:
        with pytest.raises(ValueError, match="exact live"):
            S.freeze_routes(routes, 2)
    else:
        selected = S.freeze_routes(routes, 2)
        assert selected[0]["admission"][1] is selected[1]["admission"][1] is service


@pytest.mark.parametrize("helpers", [False, True])
def test_existing_linux_namespace_delegate_preserves_commands(tmp_path, monkeypatch, helpers):
    from types import SimpleNamespace

    from merlin.mining import k1

    tools = tmp_path / "third_party/llvm-install/bin"
    tools.mkdir(parents=True)
    for name in ("llvm-nm", "llvm-objcopy"):
        (tools / name).write_text("selected")
    source, destination = tmp_path / "source.o", tmp_path / "output.o"
    source.write_bytes(b"actual object placeholder for command preservation")
    monkeypatch.setattr(k1, "repo_root", lambda: tmp_path)
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[0] == tools / "llvm-nm":
            names = "stage T 0 0\n_mlir_ciface_stage T 0 0\n"
            return SimpleNamespace(stdout=names + ("helper T 0 0\n" if helpers else ""), stderr="")
        destination.write_bytes(source.read_bytes())
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(k1, "_run", run)
    assert k1._namespace_stage_object(source, destination, index=2, entrypoint="stage") == destination
    assert calls[0] == [tools / "llvm-nm", "--defined-only", "--extern-only", "--format=posix", source]
    assert calls[1:] == (
        [[tools / "llvm-objcopy", "--redefine-sym=helper=merlin_stage_2__helper", source, destination]]
        if helpers
        else []
    )
    assert destination.read_bytes() == source.read_bytes()
