"""Check binary32 BF16 arithmetic against the integer route and golden values."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from merlin.compiler.lower.c.float_arithmetic import emit_float_arithmetic_c
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

_FLEN_GUARD = "defined(__riscv_flen) && __riscv_flen >= 32"
_BINARY32_SUM = ('__asm__("fmv.w.x ft0,%1\\n\\tfmv.w.x ft1,%2\\n\\tfadd.s ft0,ft0,ft1,rne\\n\\tfmv.x.w %0,ft0"\n'
                 '          : "=r"(s) : "r"(x),"r"(y) : "ft0","ft1");')
_RISCV_FLAGS = ("-Os", "-march=rv64imafdc", "-mabi=lp64d", "-mcmodel=medany", "-ffunction-sections", "-fdata-sections",
                "-fno-fast-math", "-ffp-contract=off", "-frounding-math", "-Wno-unused-function")
# Exact-rational BF16 sums, written out here so that they do not come from either emitted route. None means refused.
_GOLDEN = (
    (0x0000, 0x0000, 0x0000), (0x0000, 0x8000, 0x0000), (0x8000, 0x0000, 0x0000), (0x8000, 0x8000, 0x8000),
    (0x3f80, 0xbf80, 0x0000), (0xbf80, 0x3f80, 0x0000), (0x0001, 0x8001, 0x0000),
    (0x0084, 0x8100, 0x807c), (0x8084, 0x0100, 0x007c), (0x007f, 0x0001, 0x0080), (0x0080, 0x8001, 0x007f),
    (0x3f80, 0x3b80, 0x3f80), (0x3f81, 0x3b80, 0x3f82), (0xbf80, 0xbb80, 0xbf80), (0xbf81, 0xbb80, 0xbf82),
    (0x3f80, 0x3b81, 0x3f81), (0x3f80, 0x3b7f, 0x3f80), (0x3f80, 0x0001, 0x3f80),
    (0x7f7f, 0x7f7f, 0x7f80), (0xff7f, 0xff7f, 0xff80), (0x7f7f, 0x7b00, 0x7f80), (0x7f7f, 0x7aff, 0x7f7f),
    (0x7f80, 0x3f80, 0x7f80), (0xff80, 0x3f80, 0xff80), (0x3f80, 0xff80, 0xff80), (0x7f80, 0x7f80, 0x7f80),
    (0xff80, 0xff80, 0xff80), (0x7f81, 0x3f80, None), (0x3f80, 0xffc0, None), (0x7f80, 0xff80, None),
)
_GOLDEN_CHECK = r"""
#include <stdint.h>
#include <stdio.h>
#include "route.h"
static const uint16_t rows[][3] = {ROWS};
int main(void) {
  unsigned checked = 0;
#ifdef __riscv
  for (unsigned mode = 0; mode < 5u; ++mode) {
    __asm__ volatile("fsrm %0" : : "r"(mode));
#else
  { /* The native build runs the integer routine, which has no rounding mode. */
#endif
    for (unsigned r = 0; r < sizeof rows / sizeof *rows; ++r) {
      uint16_t a[4] = {0x3f80, 0x3f80, 0x3f80, rows[r][0]}, b[4] = {0, 0, 0, (uint16_t)(rows[r][1] ^ RIGHT_SIGN)}, y[4];
      for (unsigned i = 0; i < 4u; ++i) y[i] = 0xa5a5;
      const void *in[2] = {a, b}; void *out[1] = {y};
      int status = route(0, in, out);
      if (rows[r][2] == 0xffff) {
        if (status != -2 || y[0] != 0xa5a5 || y[1] != 0xa5a5 || y[2] != 0xa5a5 || y[3] != 0xa5a5) return 10;
      } else if (status || y[0] != 0x3f80 || y[1] != 0x3f80 || y[2] != 0x3f80 || y[3] != rows[r][2]) {
        printf("row %u got %04x\n", r, y[3]);
        return 11;
      }
      ++checked;
    }
  }
  printf("GOLDEN %u\n", checked);
  return 0;
}
"""
_ROUTE_CHECK = r"""
#include <stdint.h>
#include <stdio.h>
#include "integer.c"
#include "binary32.c"
#ifndef VALUE_STEP
#define VALUE_STEP 13u
#define SAMPLES 200000u
#define MINIMUM 500000u
#endif
static const uint16_t special[] = {
  0x0000, 0x0001, 0x0002, 0x0003, 0x003f, 0x0040, 0x0041, 0x007e, 0x007f, 0x0080, 0x0081, 0x00ff, 0x0100, 0x3f7f, 0x3f80,
  0x3f81, 0x3fc0, 0x4000, 0x7eff, 0x7f00, 0x7f7e, 0x7f7f, 0x7f80};
static unsigned long compared, nonzero, differ;
static void one(uint16_t a, uint16_t b) {
  if (!INTEGER_VALID(a, b)) return;
  uint16_t want = (uint16_t)INTEGER_COMPUTE(a, b), got = (uint16_t)BINARY32_COMPUTE(a, b);
  ++compared; nonzero += want != 0; differ += want != got;
}
static void signs(uint16_t a, uint16_t b) {
  for (unsigned s = 0; s < 4; ++s) one(a ^ (uint16_t)((s & 1u) << 15), b ^ (uint16_t)((s >> 1) << 15));
}
int main(void) {
  unsigned n = sizeof special / sizeof *special;
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) signs(special[i], special[j]);
    for (uint32_t v = 0; v < 0x8000u; v += VALUE_STEP) { signs(special[i], (uint16_t)v); signs((uint16_t)v, special[i]); }
  }
  uint64_t x = UINT64_C(0x9e3779b97f4a7c15);
  for (unsigned i = 0; i < SAMPLES; ++i) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    one((uint16_t)x, (uint16_t)(x >> 16));                     /* independent exponents */
    one((uint16_t)x, (uint16_t)((x & 0xff80u) | ((x >> 32) & 0x7fu))); /* equal exponents, cancellation */
    uint16_t near = (uint16_t)((x & 0x807fu) | (((((x >> 7) & 255u) + ((x >> 40) & 15u)) & 255u) << 7));
    one((uint16_t)x, near);                                    /* close exponents, ties and carries */
  }
  printf("%lu %lu %lu\n", compared, nonzero, differ);
  return differ != 0 || compared < MINIMUM || nonzero * 2 < compared;
}
"""


def _bf16_route_files(action):
    tensor = TensorType((4,), "bf16", "raw-bit-pattern")
    operands = tuple(SemanticNode(name, "input", (), tensor, effect="input") for name in "ab")
    result = SemanticNode("y", f"bf16_{action}", ("a", "b"), tensor,
                          (("numeric_semantics", f"ieee754-bf16-{action}-rne-v1"),),
                          index_maps=pointwise_identity_maps(1, 2))
    request = KernelRequest(operands + (result,), ("y",), ("external",),
                            (("a", "external"), ("b", "external")), "test", "diagnostic")
    _, files = emit_float_arithmetic_c(request, "route")
    assert files["route.c"].count(_BINARY32_SUM) == 1, "the binary32 instruction template changed"
    assert files["route.c"].count(_FLEN_GUARD) == 1
    return files["route.h"], files["route.c"]


def _second_route(text):
    guard = text.split("#ifndef ", 1)[1].split("\n", 1)[0]
    return guard.lower(), text.replace(guard, guard + "_B").replace(guard.lower(), guard.lower() + "_b").replace("route(", "route_b(")


def _write_route_check(directory, action, integer_guard, binary32):
    header, text = _bf16_route_files(action)
    prefix, second = _second_route(binary32(text))
    directory.mkdir()
    (directory / "route.h").write_text(header + "int route_b(void *, const void *const *, void *const *);\n")
    (directory / "integer.c").write_text(text.replace(_FLEN_GUARD, integer_guard))
    (directory / "binary32.c").write_text(second)
    (directory / "check.c").write_text(_ROUTE_CHECK)
    return [f"-DINTEGER_VALID={prefix}_valid", f"-DINTEGER_COMPUTE={prefix}_compute", f"-DBINARY32_COMPUTE={prefix}_b_compute"]


def _native_binary32(text):
    # A native host has no RISC-V instruction, so this leg substitutes its default-mode binary32 sum.
    native = "{float p,q,r;__builtin_memcpy(&p,&x,4);__builtin_memcpy(&q,&y,4);r=p+q;__builtin_memcpy(&s,&r,4);}"
    return text.replace(_BINARY32_SUM, native).replace(_FLEN_GUARD, "1")


def _run_route_check(directory, action, binary32=None):
    selected = (lambda text: binary32(_native_binary32(text))) if binary32 else _native_binary32
    defines = _write_route_check(directory, action, _FLEN_GUARD, selected)
    compiler = shutil.which("cc")
    assert compiler, "a native C compiler is required"
    subprocess.run([compiler, "-std=gnu11", "-O2", "-frounding-math", "-ffp-contract=off", "-Wno-unused-function",
                    *defines, "check.c", "-o", "check"], cwd=directory, check=True, capture_output=True, text=True)
    return subprocess.run([str(directory / "check")], capture_output=True, text=True, check=False)


def _write_golden(directory, action):
    header, text = _bf16_route_files(action)
    directory.mkdir()
    rows = ",".join("{%d,%d,%d}" % (a, b, 0xffff if y is None else y) for a, b, y in _GOLDEN)
    (directory / "route.h").write_text(header)
    (directory / "route.c").write_text(text)
    (directory / "golden.c").write_text(_GOLDEN_CHECK.replace("ROWS", rows))
    return [f"-DRIGHT_SIGN={0x8000 if action == 'sub' else 0}u", "golden.c", "route.c"]


def _riscv_tools():
    compiler, simulator = shutil.which("riscv64-unknown-elf-gcc"), shutil.which("spike")
    if not compiler or not simulator:
        return None
    kernel = Path(compiler).resolve().parent.parent / "riscv64-unknown-elf" / "bin" / "pk"
    return (compiler, simulator, str(kernel)) if kernel.is_file() else None


def _on_spike(directory, tools, sources):
    compiler, simulator, kernel = tools
    subprocess.run([compiler, *_RISCV_FLAGS, *sources, "-o", "check.elf"], cwd=directory, check=True, capture_output=True, text=True)
    listing = subprocess.run([str(Path(compiler).with_name("riscv64-unknown-elf-objdump")), "-d", "check.elf"],
                             cwd=directory, check=True, capture_output=True, text=True).stdout
    assert "fadd.s\tft0,ft0,ft1,rne" in listing, "the RISC-V build lacks the statically rounded add"
    return subprocess.run([simulator, "--isa=rv64imafdc", kernel, "check.elf"], cwd=directory,
                          capture_output=True, text=True, check=False, timeout=600)


class BF16Binary32Tests(unittest.TestCase):
    def test_binary32_arithmetic(self):
        self._check_bf16_binary32_route_matches_integer_route()
        self._check_bf16_golden_values_on_the_integer_route()
        self._check_bf16_emitted_instruction_on_spike()

    def _check_bf16_binary32_route_matches_integer_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for action in ("add", "sub"):
                completed = _run_route_check(root / action, action)
                assert completed.returncode == 0, completed.stdout + completed.stderr
            for label, mutate in (
                ("round half up", lambda text: text.replace("(s+0x7fffu+((s>>16)&1u))>>16", "(s+0x8000u)>>16")),
                ("truncate", lambda text: text.replace("(s+0x7fffu+((s>>16)&1u))>>16", "s>>16")),
                ("drop right sign", lambda text: text.replace("(b^32768u)", "b")),
            ):
                action = "sub" if label == "drop right sign" else "add"
                completed = _run_route_check(root / label.replace(" ", "-"), action, mutate)
                assert completed.returncode == 1, (label, completed.stdout, completed.stderr)

    def _check_bf16_golden_values_on_the_integer_route(self):
        compiler = shutil.which("cc")
        assert compiler, "a native C compiler is required"
        with tempfile.TemporaryDirectory() as temporary:
            for action in ("add", "sub"):
                directory = Path(temporary) / action
                sources = _write_golden(directory, action)
                subprocess.run([compiler, "-std=gnu11", "-O2", "-Wno-unused-function", *sources, "-o", "golden"],
                               cwd=directory, check=True, capture_output=True, text=True)
                completed = subprocess.run([str(directory / "golden")], capture_output=True, text=True, check=False)
                assert completed.returncode == 0 and completed.stdout == f"GOLDEN {len(_GOLDEN)}\n", completed.stdout

    def _check_bf16_emitted_instruction_on_spike(self):
        tools = _riscv_tools()
        if tools is None:
            self.fail("riscv64-unknown-elf-gcc, spike, and pk are required")
        with tempfile.TemporaryDirectory() as temporary:
            for action in ("add", "sub"):
                directory = Path(temporary) / f"golden-{action}"
                completed = _on_spike(directory, tools, _write_golden(directory, action))
                assert completed.returncode == 0 and f"GOLDEN {5 * len(_GOLDEN)}" in completed.stdout, completed.stdout
                directory = Path(temporary) / f"route-{action}"
                defines = _write_route_check(directory, action, "0", lambda text: text)
                completed = _on_spike(directory, tools, [*defines, "-DVALUE_STEP=1021u", "-DSAMPLES=4000u", "-DMINIMUM=15000u", "check.c"])
                assert completed.returncode == 0, completed.stdout + completed.stderr
