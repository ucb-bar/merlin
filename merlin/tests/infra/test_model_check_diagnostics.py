"""A wrong kernel result is localized to its element: the device-side compare, its parse, and its placement."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.model_check import diagnostics as DG
from merlin.model_check import kernel_reference as KR
from merlin.model_check import replay as RP
from merlin.model_check import report as R

TARGET = "gemmini"  # the target whose interface binding states these entries
_MATMUL = {
    "name": "g",
    "kind": "op",
    "op": "matmul",
    "M": 5,
    "K": 7,
    "N": 6,
    "epilogue": [],
    "operand_dtype": "int8",
    "scale_granularity": "tensor",
    "source_role": "model_derived",
    "source_reference": "test",
}
_HTIF_SHIM = r"""
#include <stdio.h>
#include <stdlib.h>
static void console_init(void) {}
static void htif_putc(char c) { putchar(c); }
static void htif_puts(const char *s) { fputs(s, stdout); }
static void htif_putd(long v) { printf("%ld", v); }
static void htif_puthex(unsigned long long v) { printf("0x%llx", v); }
static void htif_line_flush(int enable) { (void)enable; fflush(stdout); }
static void htif_exit(int code) { fflush(stdout); exit(code); }
"""
#: A host matmul under the kernel's logical ABI (W, A0, Y0) with two planted faults: Y[1][2] is one
#: too large and the last element is never written.
_KERNEL = r"""
#include <stdint.h>
void mc_test_kernel(void *w, void *a, void *y) {
  const int8_t *W = w, *A = a; int32_t *Y = y;
  for (int m = 0; m < 5; ++m)
    for (int n = 0; n < 6; ++n) {
      if (m == 4 && n == 5) continue;
      int32_t s = 0;
      for (int k = 0; k < 7; ++k) s += (int32_t)A[m * 7 + k] * (int32_t)W[k * 6 + n];
      Y[m * 6 + n] = s + (m == 1 && n == 2);
    }
}
"""


def _case(label: str = "visible/v00/p/sym#0", seed: int = 0):
    interface = KR.interface_for(_MATMUL, target=TARGET, operand_dtype="int8", accum_dtype="int32")
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, size=(5, 7), dtype=np.int64).astype(np.int8)
    w = rng.integers(-128, 128, size=(7, 6), dtype=np.int64).astype(np.int8)
    roles = ("input_0", "weight_0", "out_0")
    expected = KR.call_outputs(interface, [a, w, np.zeros((5, 6), np.int32)], roles)
    out = np.zeros((5, 6), np.int32)  # a call passes its destination too; its value is ignored
    case = RP.ReplayCase(symbol="sym", label=label, roles=roles, operands=(a, w, out), expected=tuple(expected))
    return case, interface


def test_choose_prefers_a_visible_failing_call_and_skips_passing_ones():
    sealed = RP.ReplayCase("k", "sealed/s0/p/k#0", ("out_0",), (), ())
    visible = RP.ReplayCase("k", "visible/v1/p/k#1", ("out_0",), (), ())
    fine = RP.ReplayCase("j", "visible/v1/p/j#0", ("out_0",), (), ())
    chosen = DG.choose([sealed, visible, fine], [{"exact": False}, {"exact": False}, {"exact": True}])
    assert chosen == {"k": visible}
    assert DG.choose([sealed], [{"exact": False}]) == {"k": sealed}


def test_map_marks_cells_none_some_and_all():
    cells = [0] * DG.MAP_CELLS
    cells[0], cells[-1] = 1, 2
    text = DG._map(cells, 2 * DG.MAP_CELLS)  # two elements per cell
    assert text[0] == "5" and text[-1] == "#" and set(text[1:-1]) == {"."}


def test_conv_position_maps_a_flat_index_to_its_output_pixel():
    entry = {"op": "conv2d", "Himg": 5, "Wimg": 4, "kh": 3, "kw": 3, "stride": [2, 2], "padding": [1, 1, 1, 1]}
    where = DG.conv_position(entry, [6, 8])  # Ho = 3, Wo = 2
    assert where(5 * 8 + 3) == {"n": 0, "oh": 2, "ow": 1, "co": 3}
    assert DG.conv_position({"op": "matmul"}, [6, 8]) is None


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a host C compiler")
def test_the_diagnostic_image_pinpoints_a_wrong_and_an_unwritten_element(tmp_path):
    """The generated image, compiled for the host with a console shim, finds both planted faults."""
    case, interface = _case()
    main_c, data_s, plan = DG.write_sources(tmp_path, [case], {"sym": "mc_test_kernel"}, {"sym": interface})
    text = main_c.read_text()
    assert "mc_compare(0, 0," in text and "mc_test_kernel(" in text
    host = text.replace('#include "htif.h"', _HTIF_SHIM).replace(
        'if (hartid) for (;;) { __asm__ volatile("wfi"); }', "(void)hartid;"
    )
    (tmp_path / "host_main.c").write_text(host)
    (tmp_path / "kernel.c").write_text(_KERNEL)
    exe = tmp_path / "image"
    subprocess.run(
        ["cc", "-O1", "-o", str(exe), str(tmp_path / "host_main.c"), str(tmp_path / "kernel.c"), str(data_s)],
        check=True,
        cwd=tmp_path,
    )
    console = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    assert "MCDONE" in console
    (found,) = DG.parse_console(console, plan, {"sym": _MATMUL})
    (result,) = found["results"]
    want = case.expected[0].reshape(5, 6)
    assert result["differing"] == 2 and result["unwritten"] == 1
    assert result["first"]["index"] == [1, 2]
    assert result["first"]["expected"] == int(want[1, 2]) and result["first"]["got"] == int(want[1, 2]) + 1
    assert result["last"]["index"] == [4, 5] and result["last"]["got"] == int(np.frombuffer(b"\xa5" * 4, "<i4")[0])
    assert result["rows_differing"] == 2 and result["cols_differing"] == 2 and result["col_range"] == [2, 5]
    assert [e["index"] for e in result["listed"]] == [[1, 2], [4, 5]]
    assert result["window"]["got"][DG.BEFORE] == result["first"]["got"]
    assert "hint" not in result  # one of the two was written, so not every difference is unwritten


def test_a_reproducible_result_and_a_silent_image_are_reported_as_such():
    case, interface = _case()
    plan = [
        {
            "case": 0,
            "symbol": "sym",
            "label": case.label,
            "outputs": [
                {
                    "role": "out_0",
                    "shape": [5, 6],
                    "dtype": "i32",
                    "elements": 30,
                    "itemsize": 4,
                    "cols": 6,
                    "expected": case.expected[0].reshape(5, 6),
                }
            ],
        }
    ]
    (silent,) = DG.parse_console("MCDONE\n", plan, {})
    assert "error" in silent["results"][0]
    console = "MCDSUM 0 0 0 0 0 0 0 0 0 0 0x0\nMCDMAP 0 0 " + " ".join(["0"] * DG.MAP_CELLS) + "\n"
    (matched,) = DG.parse_console(console, plan, {})
    assert matched["results"][0]["differing"] == 0 and "not reproducible" in matched["results"][0]["hint"]


def test_a_sealed_diagnosis_is_kept_from_the_agent_view():
    rows = [{"symbol": "a", "status": "fail"}, {"symbol": "b", "status": "fail"}]
    DG.attach(rows, {"a": {"split": "visible", "kind": "mismatch"}, "b": {"split": "sealed", "kind": "mismatch"}})
    assert rows[0]["diagnosis"]["split"] == "visible"
    assert "diagnosis" not in rows[1] and rows[1]["sealed"]["diagnosis"]["split"] == "sealed"
    report = R.new_report("check", model="m", split="all", submission={})
    report["programs"] = [{"program": "p", "kernels": rows, "runs": []}]
    view = R.agent_view(report)
    kernels = view["programs"][0]["kernels"]
    assert "diagnosis" in kernels[0] and "sealed" not in kernels[1] and "diagnosis" not in kernels[1]


def test_a_refused_kernel_is_diagnosed_with_its_form_and_refusal():
    case, interface = _case()
    catalog = SimpleNamespace(
        interfaces={"sym": interface},
        sidecar={
            "entries": {"sym": _MATMUL},
            "call_buffers": {
                "sym": [
                    {"role": "input_0", "shape": [5, 7], "dtype": "i8"},
                    {"role": "weight_0", "shape": [7, 6], "dtype": "i8"},
                    {"role": "out_0", "shape": [5, 6], "dtype": "i32"},
                ]
            },
            "routed": [{"symbol": "sym", "source_region_id": "matmul_3", "group": 3}],
        },
    )
    diagnosis = DG.refusal(catalog, "sym", "package lowering: BackendDeclined: " + "x" * 9000)
    assert diagnosis["kind"] == "refused" and diagnosis["form"] == "matmul M=5 K=7 N=6 -> i32"
    assert diagnosis["source_region_ids"] == ["matmul_3"] and "MATMUL_RESIDENT" in diagnosis["opcodes"]
    assert len(diagnosis["refusal"]) == DG.REFUSAL_CHARS
    assert DG.summary(diagnosis).startswith("matmul M=5 K=7 N=6 -> i32: refused: package lowering")
