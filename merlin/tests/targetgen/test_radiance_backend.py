"""The SIMT package's own runtime backend: command buffer -> per-warp kernel -> its own RTL.

Two accelerators had reached a command buffer, but only the systolic one could execute it: the SIMT
package had a dialect and a toolchain and no *tile body*, so nothing turned an abstract command list
into the per-warp computation the derived scaffold runs. This suite covers the module that closes
that, and — more importantly — the boundaries it must refuse to cross.

The refusals are the substance here. A backend that quietly realizes a "resident" pack in global
memory, or rounds a requantization its own way, or picks an accumulator width by habit, produces a
kernel that builds, runs, and grades green while doing something other than what the command buffer
said. Each such case is asserted to raise with the missing derivation NAMED, so a failure attributes
to a boundary instead of reading "unsupported".

Scope of the hardware test is deliberately narrow and matches what the backend claims: the command
buffer executes bit-exact on the target's own RTL-derived model, across warps, on the base integer
ISA. It is NOT a tensor-core result and NOT a certification of the package's hand-authored dialect.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root

PACKAGE = repo_root() / "out/artifacts/targets/radiance/hand_v0"


@pytest.fixture(scope="module")
def backend():
    """The backend the CORE never names: it is discovered only via the contract's ``plugin.backend``."""
    if not PACKAGE.is_dir():
        pytest.skip(f"target package not present: {PACKAGE}")
    from merlin.runtime.backends import base

    # Scoped to this module: leaving MERLIN_TARGET_PATH set would change target discovery for every
    # later test in the session, which is how one suite silently reconfigures another.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MERLIN_TARGET_PATH", str(PACKAGE))
        patch.setattr(base, "_oot_env_seen", None)   # re-run OOT discovery with the env now set
        names = [n for n in base.list_backends() if n.startswith("radiance")]
        if not names:
            pytest.skip("the package's backend did not register (its optional deps are absent)")
        yield base.get_backend(names[0])


def _matmul_cb(m: int = 16, k: int = 32, n: int = 16, *, epilogue=None, **commit_attrs) -> dict:
    """A hand-built one-tile command buffer in the schema the staged pipeline emits.

    Hand-built on purpose: these cases are about the backend, so they must not need the kernel
    frontend installed. The hardware test below uses the pipeline's own command buffer instead.
    """
    return {
        "target": "radiance",
        "tensors": {
            "A0": {"shape": [m, k], "dtype": "i8", "role": "input"},
            "A1": {"shape": [k, n], "dtype": "i8", "role": "input"},
            "acc0": {"shape": [m, n], "dtype": "i32", "role": "accumulator"},
            "Y0": {"shape": [m, n], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "A1", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
             "attributes": {"epilogue": epilogue or [], "output_dtype": "i32", **commit_attrs}},
        ],
    }


# ------------------------------------------------------------------ the seam
def test_the_package_supplies_its_own_backend_and_the_core_names_nothing(backend):
    """``plugin.backend`` is the whole wiring: the core holds no name -> module map for this target."""
    import yaml

    contract = yaml.safe_load((PACKAGE / "contracts" / "target_contract.yaml").read_text())
    assert (contract.get("plugin") or {}).get("backend") == "backend.py", \
        "the contract must name its own backend file; that declaration is the only thing that wires it"
    assert backend.__file__.startswith(str(PACKAGE)), \
        f"the loaded backend must live in the package, not in the core tree (got {backend.__file__})"

    core_backends = repo_root() / "merlin/python/merlin/runtime/backends"
    core_text = (core_backends / "base.py").read_text(encoding="utf-8")
    assert backend.BACKEND_NAME not in core_text, \
        "the core registry must not seed this backend's name — it self-registers from the package"


# ------------------------------------------------------------------ what it refuses, and why
def test_residency_is_refused_and_names_the_missing_aperture(backend):
    """A pack realized in global memory has made nothing resident; refusing beats pretending."""
    cb = _matmul_cb()
    cb["commands"].insert(0, {"opcode": "RES_PACK", "operands": {"src": "A1", "dst": "A1_res"},
                              "attributes": {"layout": "packed_rhs"}})
    with pytest.raises(backend.EmitError) as excinfo:
        backend.emit_kernel(cb)
    message = str(excinfo.value)
    assert "aperture" in message, f"the refusal must name the missing derivation, got: {message}"
    assert "resident" in message


def test_a_resident_matmul_is_refused_for_the_same_reason(backend):
    cb = _matmul_cb()
    cb["commands"][0]["opcode"] = "MATMUL_RESIDENT"
    with pytest.raises(backend.EmitError, match="resident"):
        backend.emit_kernel(cb)


def test_requantization_is_refused_rather_than_rounded_differently(backend):
    """Merlin's integer requant rounding is not derived here, and 'close' is wrong in the low bit."""
    cb = _matmul_cb(requant_shift=4)
    with pytest.raises(backend.EmitError, match="requant"):
        backend.emit_kernel(cb)


def test_a_float_element_type_is_refused(backend):
    cb = _matmul_cb()
    cb["tensors"]["A0"]["dtype"] = "f32"
    with pytest.raises(backend.EmitError, match="integer"):
        backend.emit_kernel(cb)


def test_an_unrealizable_epilogue_stage_is_refused(backend):
    with pytest.raises(backend.EmitError, match="epilogue"):
        backend.emit_kernel(_matmul_cb(epilogue=["bias_add"]))


def test_the_warp_count_comes_from_the_contract_not_a_default(backend):
    """A guessed warp count would run correctly at any value, so nothing downstream would catch it."""
    assert backend.warps_per_core({"capabilities": {"simt": {"warps_per_core": 8}}}) == 8
    with pytest.raises(backend.EmitError, match="warps_per_core"):
        backend.warps_per_core({"capabilities": {"simt": {"lanes_per_warp": 16}}})


# ------------------------------------------------------------------ what it emits
def test_the_accumulator_width_is_computed_from_the_operands(backend):
    """i8 x i8 over K=32 needs 21 bits; habit would say i32 and be right here and wrong later."""
    assert backend.accumulator_dtype("i8", "i8", 32) == "i32"     # 8 + 8 + 5 = 21 bits
    assert backend.accumulator_dtype("i8", "i8", 1) == "i16"      # 8 + 8 + 0 = 16 bits
    assert backend.accumulator_dtype("i16", "i16", 4) == "i64"    # 16 + 16 + 2 = 34 bits
    # 32 + 32 + 2 = 66 bits fits nothing here, so it must refuse rather than wrap silently.
    with pytest.raises(backend.EmitError, match="accumulator bits"):
        backend.accumulator_dtype("i32", "i32", 4)


def test_every_command_partitions_work_the_same_way(backend):
    """The regression that matters: identical per-warp ownership is what makes no barrier safe.

    Striding a flat index by warp id gives each warp elements from every row, so the commit would read
    accumulator entries another warp is still writing — a race that can pass by luck. Neither a barrier nor
    a usable fence is available on this target (a fence transcodes but nothing after it executes on the
    oracle), so the emitter cannot repair such a schedule after the fact; it must not emit one.
    """
    source = backend.emit_kernel(_matmul_cb()).source
    assert "i += MU_NUM_WARPS" not in source, \
        "a flat-index stride crosses row ownership; partition by row like the matmul does"
    strided = source.count("m += MU_NUM_WARPS")
    assert strided > 0, "no warp-strided loop was emitted at all"
    # Counted as a ratio rather than against the command count: the derived scaffold emits the body
    # twice (the spawned workers and warp 0's own tile), and that is the scaffold's business, not this
    # assertion's. Every row loop must start at wid and step by the warp count.
    assert source.count("m = wid") == strided, \
        "every command's row loop must be both based at wid and strided by the warp count"


def test_operands_are_volatile_so_the_kernel_cannot_be_folded_away(backend):
    """Constant inputs + constant bounds let a compiler store the answer and execute no arithmetic.

    That kernel grades green and proves nothing, which is the vacuity trap this repo has hit before.
    """
    source = backend.emit_kernel(_matmul_cb()).source
    for name in ("A0", "A1", "acc0", "Y0"):
        assert f"volatile int8_t {name}[" in source or f"volatile int32_t {name}[" in source, \
            f"{name} must be volatile or the contraction can be constant-folded at build time"


def test_the_simt_control_ops_are_derived_not_spelled(backend):
    """No mnemonic for the target's own control ops: they arrive as .insn forms from the runtime ABI."""
    source = backend.emit_kernel(_matmul_cb()).source
    assert ".insn r" in source, "the SIMT control ops must be emitted as derived .insn forms"
    for invented in ("vx_tmc", "vx_wspawn", "wspawn ", "tmc "):
        assert invented not in source, f"{invented!r} is a mnemonic no stock assembler knows"


def test_a_command_buffer_with_no_commit_is_refused(backend):
    cb = _matmul_cb()
    cb["commands"] = cb["commands"][:1]
    with pytest.raises(backend.EmitError, match="commits no output"):
        backend.emit_kernel(cb)


def test_the_spawn_count_is_capped_below_the_declared_warp_slots_and_says_so(backend):
    """The cap is a workaround for a measured scaffold defect, so it must stay visible.

    Spawning the hardware's full declared warp count loses the spawning warp's own tile: measured on
    the RTL-arc model with three unrelated bodies, every one returned warp 0's element as zero at 8
    warps and was exact at 4. A cap that did not report itself would turn "used half the machine" into
    something indistinguishable from "used all of it".
    """
    emitted = backend.emit_kernel(_matmul_cb())
    assert emitted.num_warps == backend.ORACLE_SPAWN_WARPS
    assert emitted.warps_declared == backend.warps_per_core()
    assert emitted.warps_declared > emitted.num_warps, \
        "if the declared count stops exceeding the cap, re-measure rather than deleting this"
    assert emitted.warps_capped is True
    assert f"MU_NUM_WARPS {backend.ORACLE_SPAWN_WARPS}u" in emitted.source


def test_an_explicit_warp_count_overrides_the_cap(backend):
    """The cap is a default, not a ceiling — reproducing the defect must stay possible."""
    emitted = backend.emit_kernel(_matmul_cb(), num_warps=8)
    assert emitted.num_warps == 8
    assert emitted.warps_capped is False


def test_completion_is_asserted_before_any_output_is_graded(backend):
    """A budget-starved run must say so, not look like a wrong answer.

    This is not hypothetical: the first RTL run of this backend produced a bit-exact accumulator and an
    all-zero committed output, because the cycle budget ran out between the two. Without the sentinel
    that reads as a miscompile.
    """
    source = backend.emit_kernel(_matmul_cb()).source
    assert f"{backend.SENTINEL}[0] = 1;" in source, \
        "the kernel must record its own completion, and only after every warp has parked"
    # It has to be set in the manager tail — i.e. after the wait on the warp mask, not inside the body.
    tail = source.split("while(_wmask()!=1){}")[-1]
    assert f"{backend.SENTINEL}[0] = 1;" in tail, \
        "a sentinel set before the wait would mean 'warp 0 finished', not 'the kernel finished'"


# ------------------------------------------------------------------ the hardware grade
@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "the emitted kernel does not reach its completion sentinel, so its output is deliberately not graded. "
    "What IS established on RTL: the contraction is bit-exact (acc0 read back as [192,192,64,64,...], "
    "matching the reference exactly) with the operands correctly in device memory — the arithmetic and the "
    "operand plumbing are right. What blocks the whole-kernel grade is the ORACLE's visibility of a "
    "kernel's final stores: measured, a store is unreliably recovered when little or no execution follows "
    "it, and a `fence` (the standard remedy, and what this hardware's own runtime uses) transcodes but "
    "stops execution dead — nothing after one runs. Both are outside this backend; see "
    "docs/design/target_kernel_anatomy.md. Not worked around by padding the tail with filler stores until "
    "the real one becomes visible: that would pass by exploiting the race that makes the grade "
    "meaningless. strict=True so this fails loudly the moment it starts passing."))
def test_the_command_buffer_executes_bit_exact_on_the_targets_own_rtl(backend, tmp_path):
    """The result the missing slot existed to make possible.

    A small tile, so this stays inside the suite's per-test ceiling: the arc model advances roughly a
    thousand cycles a second, and the tile size is not what is under test — the emission and the
    grading path are. `test_the_pipelines_own_command_buffer...` covers the full Triton tile.

    What a pass earns: the command buffer is realizable on this hardware, across warps, bit-exact
    against an independent integer reference. What it does not earn: a tensor-core result, or any
    statement about the package's hand-authored dialect being right.
    """
    if not backend.available():
        pytest.skip("stock LLVM tools / derived ISA fact / RTL-arc model not all present")

    result = backend.run_command_buffer(_matmul_cb(m=4, k=8, n=4), workdir=tmp_path)
    assert result["oracle"]["derived_from_rtl"] is True, \
        "a grade that is not against RTL must not be reported as one"
    assert result["correct"], (
        f"device output differs from the reference\n got: {result['outputs']}\n"
        f"expected: {result['expected']}")


@pytest.mark.slow
def test_the_pipelines_own_command_buffer_executes_on_rtl(backend, tmp_path):
    """The same grade on the command buffer the staged pipeline actually emits, at its own tile size.

    Separated from the test above and gated on MERLIN_RUN_SLOW purely for wall-clock: `tl.dot` floors a
    tile at 16x32x16, which is a few hundred thousand simulated cycles at roughly a thousand a second.
    """
    import os

    if not os.environ.get("MERLIN_RUN_SLOW"):
        pytest.skip("set MERLIN_RUN_SLOW=1 to grade the full pipeline tile on the arc model")
    if not backend.available():
        pytest.skip("stock LLVM tools / derived ISA fact / RTL-arc model not all present")
    triton_kernels = pytest.importorskip("triton_kernels")
    if not triton_kernels.HAS_TRITON:
        pytest.skip("the `triton` optional extra is not installed")

    from merlin import compile_core
    from merlin.targetgen.registry import load_target
    from merlin.triton import source as triton_source
    from merlin.triton.bridge import to_linalg

    spec = triton_kernels.matmul_one_tile_spec()
    staged = compile_core.compile_core_mlir(
        to_linalg(triton_source.make_ttir(spec), spec).module,
        target_package=load_target(PACKAGE)).staged

    result = backend.run_command_buffer(staged.command_buffer, workdir=tmp_path,
                                        max_cycles=600_000, timeout=1800)
    assert result["correct"], (
        f"device output differs from the reference\n got: {result['outputs']}\n"
        f"expected: {result['expected']}")
