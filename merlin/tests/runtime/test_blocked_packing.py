"""Explicit rank2 byte tiling is bounded, bit-exact and never an alias permission."""

import ctypes
import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import plan_deployable_chain
from merlin.llvmlower.deployable_packing import select_packing_policy, tile_policy
from merlin.runtime.outlined_chain import _views, render_outlined_chain
from merlin.runtime.staged_dispatch import StagedRoles
from merlin.tests.runtime.test_deployable_packing import SOURCE


def _plan(**kwargs):
    return plan_deployable_chain(
        parse_mlir_text(SOURCE),
        worker_slots=(0,),
        worker_count=1,
        workspace_budget_bytes=65536,
        alignment_bytes=64,
        data_preparation=True,
        **kwargs,
    )


def test_default_none_preserves_exact_plan_and_emitted_loop():
    baseline = _plan()
    explicit = _plan(packing_tile_shape=None, packing_tile_budget_bytes=None)
    assert baseline.to_dict() == explicit.to_dict()
    assert "controller_packing_policy" not in baseline.data_preparation
    view = baseline.dispatches[0]["views_before"][0]
    assert "packing_policy" not in view
    assert "r0=" not in "\n".join(_views([view], {view["input"]: 0, view["output"]: 1}))


def test_explicit_policy_records_logical_traffic_without_inferred_cache():
    p = _plan(packing_tile_shape=(2, 3), packing_tile_budget_bytes=12)
    view = p.dispatches[0]["views_before"][0]
    policy = view["packing_policy"]
    assert policy["strategy"] == "blocked_rank2"
    assert policy["effective_tile_shape"] == (2, 3)
    assert policy["logical_tile_footprint_bytes"] == 12
    assert policy["logical_read_bytes"] == policy["logical_write_bytes"] == 15
    assert policy["extra_scratch_bytes"] == 0
    assert policy["physical_traffic_bytes"] is None and not policy["physical_cache_geometry_known"]
    assert p.source_ir_sha256 == _plan().source_ir_sha256
    with pytest.raises(ValueError, match="exceeds declared budget"):
        _plan(packing_tile_shape=(2, 3), packing_tile_budget_bytes=11)


@pytest.mark.parametrize(
    "shape,budget",
    [
        (None, 64),
        ((1, 2), None),
        ([1, 2], 64),
        ((0, 1), 64),
        ((True, 2), 64),
        ((1, 2, 3), 64),
        ((1, 2), True),
        ((1, 2), 0),
        ((1 << 63, 2), 64),
    ],
)
def test_malformed_or_incomplete_explicit_policy_refuses(shape, budget):
    with pytest.raises(ValueError):
        tile_policy(shape, budget)


def test_selection_requires_source_preparation_permission():
    with pytest.raises(ValueError, match="data_preparation"):
        plan_deployable_chain(
            parse_mlir_text(SOURCE),
            worker_slots=(0,),
            worker_count=1,
            workspace_budget_bytes=65536,
            alignment_bytes=64,
            packing_tile_shape=(1, 2),
            packing_tile_budget_bytes=16,
        )


def test_rank3_fallback_is_explicit_and_not_claimed_blocked():
    view = {"kind": "pack_permutation", "source_shape": (2, 3, 4), "output_shape": (3, 2, 4), "element_bytes": 4}
    p = select_packing_policy(view, tile_policy((2, 2), 128))["packing_policy"]
    assert p["strategy"] == "flat" and "rank2" in p["reason"]
    assert "logical_tile_footprint_bytes" not in p


@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("shape,tile", [((7, 13), (4, 5)), ((1, 17), (32, 64)), ((17, 1), (3, 2)), ((3, 5), (1, 1))])
def test_blocked_native_C_full_bits_tails_and_guards(tmp_path, width, shape, tile):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    rows, cols = shape
    raw = bytes((i * 73 + 19) % 256 for i in range(rows * cols * width))
    # Distinct NaN payload, infinities, signed zero and subnormal patterns remain raw bytes.
    if len(raw) >= 16:
        raw = bytes.fromhex("000000804523c17f0000807f01000000") + raw[16:]
    view = {
        "kind": "pack_permutation",
        "input": "a",
        "output": "b",
        "source_shape": (cols, rows),
        "output_shape": shape,
        "source_strides": (1, rows),
        "element_bytes": width,
    }
    view = select_packing_policy(view, tile_policy(tile, 2 * tile[0] * tile[1] * width))
    source = (
        "#include <stddef.h>\n#include <string.h>\nstruct C{void*buffers[2];};\n"
        "void run(void*a,void*b){struct C v={{a,b}},*c=&v;\n"
    )
    source += "\n".join(_views([view], {"a": 0, "b": 1})) + "\n}\n"
    src, lib = tmp_path / f"copy_{width}_{rows}_{cols}.c", tmp_path / f"copy_{width}_{rows}_{cols}.so"
    src.write_text(source)
    subprocess.run(
        [cc, "-std=c11", "-O2", "-shared", "-fPIC", str(src), "-o", str(lib)],
        check=True,
        capture_output=True,
        timeout=20,
    )
    a = ctypes.create_string_buffer(b"\x5a" * 64 + raw + b"\x5a" * 64)
    b = ctypes.create_string_buffer(b"\x5a" * (len(raw) + 128))
    function = ctypes.CDLL(str(lib)).run
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    function(ctypes.byref(a, 64), ctypes.byref(b, 64))
    expected = b"".join(
        raw[(j * rows + i) * width : (j * rows + i + 1) * width] for i in range(rows) for j in range(cols)
    )
    assert a.raw[: len(raw) + 128] == b"\x5a" * 64 + raw + b"\x5a" * 64
    assert b.raw[: len(raw) + 128] == b"\x5a" * 64 + expected + b"\x5a" * 64


def test_production_model_init_refuses_inplace_before_packing(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    p = _plan(packing_tile_shape=(2, 3), packing_tile_budget_bytes=12)
    rendered = render_outlined_chain(
        p, ("arithmetic",), roles=StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,)), prefix="blocked"
    )
    (tmp_path / "blocked.h").write_text(rendered["header"])
    (tmp_path / "blocked.c").write_text(rendered["source"])
    (tmp_path / "main.c").write_text(r"""
#include <assert.h>
#include <string.h>
#include "blocked.c"
static unsigned hart(void*p){(void)p;return 2;}
void _mlir_ciface_arithmetic(void*a,void*b,void*c){(void)a;(void)b;(void)c;assert(0);}
int main(void){
 _Alignas(64) unsigned char arena[BLOCKED_WORKSPACE_BYTES],a[64],b[64],out[64],before[sizeof arena];
 blocked_context c={0};memset(arena,0x5a,sizeof arena);memcpy(before,arena,sizeof arena);
 c.platform_hooks.current_hart=hart;c.team.config.controller_hart=2;c.team.config.sequence=1;c.active_dispatch=0;
 c.inputs[0]=a;c.input_capacity[0]=6;c.inputs[1]=arena;c.input_capacity[1]=15;
 c.outputs[0]=out;c.output_capacity[0]=40;c.workspace=arena;c.workspace_capacity=sizeof arena;
 assert(model_init(&c)==-105);assert(!memcmp(before,arena,sizeof arena));
 c.inputs[1]=b;c.outputs[0]=b;c.team.config.sequence=2;assert(model_init(&c)==-108);
 assert(!memcmp(before,arena,sizeof arena));return 0;
}
""")
    runtime = runtime_dir() / "c"
    exe = tmp_path / "alias"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-I" + str(tmp_path),
            "-I" + str(runtime),
            str(tmp_path / "main.c"),
            str(runtime / "merlin_staged.c"),
            str(runtime / "merlin_descriptors.c"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)
