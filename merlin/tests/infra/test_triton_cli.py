"""``merlin-compile-kernel``: the argument grammar, and that it drives the real pipeline.

The CLI holds no compilation logic, so what is worth testing is the part that is only reachable
through it — turning command-line text into a kernel spec. That translation is where a user says
what the source cannot: shapes, effects, the grid, and the compile-time value of a runtime scalar.
Getting one of those wrong is a miscompile rather than an error, so the parser refuses anything it
cannot read instead of filling in a default.
"""
from __future__ import annotations

import json

import pytest
import triton_kernels as K

from merlin.common.paths import repo_root
from merlin.triton import cli

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")

VECTOR_ADD = str(repo_root() / "examples/triton/vector_add.py") + ":vector_add"
MATMUL = str(repo_root() / "examples/triton/matmul_simple.py") + ":repeated_rhs_matmul"
GEMMINI_PACKAGE = repo_root() / "out/artifacts/targets/gemmini/hand_v0"

ADD_ARGS = ["--arg", "x_ptr=*fp32:1025:read", "--arg", "y_ptr=*fp32:1025:read",
            "--arg", "out_ptr=*fp32:1025:write", "--arg", "n_elements=i32",
            "--assume", "n_elements=1025", "--constexpr", "BLOCK_SIZE=256", "--grid", "5"]


def test_a_pointer_argument_parses_into_shape_and_effect():
    arg = cli.parse_arg("w_ptr=*i8:32x16:read")
    assert (arg.kind, arg.dtype, arg.shape, arg.effect) == ("pointer", "i8", (32, 16), "read")


def test_a_scalar_argument_carries_no_shape():
    arg = cli.parse_arg("n=i32")
    assert (arg.kind, arg.dtype, arg.shape, arg.effect) == ("scalar", "i32", None, None)


@pytest.mark.parametrize("text", ["w_ptr=*i8:32x16", "w_ptr=*i8", "w_ptr=*i8:32x16:read:extra"])
def test_an_incomplete_pointer_spec_is_refused(text):
    """A pointer without a shape or an effect cannot be compiled, so it is not accepted."""
    with pytest.raises(SystemExit) as exc:
        cli.parse_arg(text)
    assert "DTYPE:SHAPE:EFFECT" in str(exc.value)


def test_a_bad_effect_is_refused_by_the_spec():
    with pytest.raises(Exception) as exc:
        cli.parse_arg("w_ptr=*i8:16:maybe")
    assert "effect" in str(exc.value)


def test_bindings_keep_their_types():
    """JSON-decoded, so BLOCK_SIZE=256 is an int and a name stays a string."""
    assert cli.parse_binding("BLOCK_SIZE=256") == ("BLOCK_SIZE", 256)
    assert cli.parse_binding("layout=row_major") == ("layout", "row_major")


def test_an_unknown_emit_stage_lists_the_known_ones():
    with pytest.raises(SystemExit) as exc:
        cli.resolve_emit("ttir,not-a-stage")
    assert "not-a-stage" in str(exc.value) and "command-buffer" in str(exc.value)


def test_emit_all_selects_every_stage():
    assert cli.resolve_emit("all") == list(cli.EMIT_STAGES)


def test_a_missing_target_is_refused():
    with pytest.raises(SystemExit) as exc:
        cli.main([VECTOR_ADD, *ADD_ARGS])
    assert "--target" in str(exc.value)


def test_route_only_reports_without_writing(capsys, tmp_path):
    assert cli.main([VECTOR_ADD, "--target", "saturn", *ADD_ARGS, "--route-only",
                     "--out", str(tmp_path)]) == 0
    assert "route: llvm" in capsys.readouterr().out
    assert not list(tmp_path.iterdir()), "--route-only must not write artifacts"


def test_the_vector_add_example_compiles_and_reports(tmp_path):
    """The example in the docs, run as written — so a stale example fails the suite."""
    assert cli.main([VECTOR_ADD, "--target", "saturn", *ADD_ARGS,
                     "--emit", "ttir,core-mlir,report", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["kernel"] == "vector_add"
    assert report["route"]["kind"] == "llvm"
    assert report["capability"]["grid"] == [5, 1, 1]
    assert not report["capability"]["unaccounted"]
    assert "linalg.add" in (tmp_path / "core_mlir.mlir").read_text()
    assert "tt.load" in (tmp_path / "ttir.mlir").read_text()


def test_the_matmul_example_descends_to_the_target_dialect(tmp_path):
    if not GEMMINI_PACKAGE.is_dir():
        pytest.skip("gemmini target package not present")
    assert cli.main([
        MATMUL, "--target-package", str(GEMMINI_PACKAGE),
        "--arg", "a0_ptr=*i8:16x32:read", "--arg", "a1_ptr=*i8:16x32:read",
        "--arg", "w_ptr=*i8:32x16:read",
        "--arg", "c0_ptr=*i32:16x16:write", "--arg", "c1_ptr=*i32:16x16:write",
        "--constexpr", "BM=16", "--constexpr", "BN=16", "--constexpr", "BK=32",
        "--grid", "1", "--emit", "all", "--verify", "--out", str(tmp_path)]) == 0
    command_buffer = json.loads((tmp_path / "command_buffer.json").read_text())
    assert command_buffer["target"] == "gemmini"
    assert "gemmini.matmul" in (tmp_path / "target.mlir").read_text()
    assert json.loads((tmp_path / "report.json").read_text())["route"]["kind"] == "staged"


def test_a_kernel_the_bridge_refuses_exits_nonzero_without_a_traceback(capsys, tmp_path):
    """A refusal is a diagnostic, not a crash — INV-8 as a user sees it."""
    # The same invocation with the mask bound left undeclared: the compiler cannot then check that
    # the launch stays inside the tensor, so it refuses.
    without_assumption = ["--arg", "x_ptr=*fp32:1025:read", "--arg", "y_ptr=*fp32:1025:read",
                          "--arg", "out_ptr=*fp32:1025:write", "--arg", "n_elements=i32",
                          "--constexpr", "BLOCK_SIZE=256", "--grid", "5"]
    code = cli.main([VECTOR_ADD, "--target", "saturn", *without_assumption, "--out", str(tmp_path)])
    assert code == 2
    assert "n_elements" in capsys.readouterr().err


def test_gpu_scheduling_knobs_are_recorded_as_provenance_only(tmp_path):
    """`num_warps` means nothing on a systolic array, so it is kept and never interpreted."""
    assert cli.main([VECTOR_ADD, "--target", "saturn", *ADD_ARGS, "--num-warps", "8",
                     "--num-stages", "3", "--emit", "report", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["provenance"] == {"num_warps": 8, "num_stages": 3}
    assert "num_warps" not in json.dumps(report["route"])
