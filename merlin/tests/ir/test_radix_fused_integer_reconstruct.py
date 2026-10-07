"""Completed-plane reconstruction against the original binary64 evaluation."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.llvmlower.radix_integer_reconstruct import c_fused_header, c_header
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups


def test_fused_reconstruction_rederives_complete_plan():
    plan = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=65)
    for changed in (
        replace(plan, weighted_absolute_bound=1),
        replace(plan, groups=plan.groups[:-1]),
        replace(plan, reduction_length=2049),
    ):
        with pytest.raises(ValueError):
            c_fused_header(changed)


@pytest.mark.parametrize(
    "radix_bits,digits,k",
    [
        (1, 1, 1),
        (4, 2, 65),
        (7, 3, 1),
        (7, 3, 65),
        (7, 3, 2048),
        (8, 1, 127),
    ],
)
def test_actual_c_fused_original_prefixes_bounds_aliases_and_effects(tmp_path, radix_bits, digits, k):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    plan = plan_radix_product_groups(radix_bits=radix_bits, digits=digits, reduction_length=k)
    (tmp_path / "fused.h").write_text(c_fused_header(plan))
    (tmp_path / "stream.h").write_text(c_header(plan))
    limits = ",".join(str(g.accumulator_bound) for g in plan.groups)
    weights = ",".join(str(1 << g.exponent) for g in plan.groups)
    source = r"""#include "fused.h"
#include "stream.h"
#include <fenv.h>
#include <string.h>
enum { G = MERLIN_RADIX_FUSED_INTEGER_GROUPS, N = 17 };
static const int32_t limits[G] = {LIMITS};
static const int64_t weights[G] = {WEIGHTS};
int test(void) {
  if (fesetround(FE_TONEAREST)) return 1;
  merlin_radix_integer_fused_exact_f64(0, 0, 0);
  for (unsigned scenario=0; scenario<7; ++scenario) {
    int32_t planes[G][N], saved[G][N]; const int32_t *sources[G];
    double guard[N+2], expected[N], streamed[N]; int64_t integers[N];
    for (unsigned g=0;g<G;++g) {
      for (unsigned t=0;t<N;++t) {
        int32_t value=limits[g];
        if (scenario==0) value=0;
        if (scenario==2) value=-value;
        if (scenario==3 && ((t+g)&1)) value=-value;
        if (scenario==4) value=(int32_t)((t*937+g*13)%(2*(uint64_t)limits[g]+1))-limits[g];
        if (scenario==5) value=limits[0];
        if (scenario==6 && G>1) value=g==0?(int32_t)weights[1]:g==1?-1:0;
        planes[g][t]=value;
      }
      sources[g]=scenario==5?planes[0]:planes[g];
    }
    memcpy(saved,planes,sizeof(planes));
    for (unsigned t=0;t<N;++t) {expected[t]=0.0;guard[t+1]=-19.0;}
    guard[0]=0x1.23456789abcdep+42;guard[N+1]=-0x1.abcdef1234567p-29;
    for (unsigned g=0;g<G;++g) {
      for (unsigned t=0;t<N;++t) expected[t]+=(double)sources[g][t]*(double)weights[g];
      if (!g) merlin_radix_integer_begin_from_first_group_exact_i64(integers,sources[g],N);
      else merlin_radix_integer_accumulate_exact_i64(integers,sources[g],N,g);
    }
    merlin_radix_integer_finish_exact_f64(streamed,integers,N);
    if (memcmp(expected,streamed,sizeof(expected))) return 2;
    for (unsigned sticky=0;sticky<2;++sticky) {
      feclearexcept(FE_ALL_EXCEPT);if(sticky)feraiseexcept(FE_INEXACT);
      int before=fetestexcept(FE_ALL_EXCEPT);
      merlin_radix_integer_fused_exact_f64(guard+1,sources,N);
      if (before!=fetestexcept(FE_ALL_EXCEPT)) return 3;
      if (memcmp(expected,guard+1,sizeof(expected))) return 4;
      if (guard[0]!=0x1.23456789abcdep+42||guard[N+1]!=-0x1.abcdef1234567p-29) return 5;
      if (memcmp(saved,planes,sizeof(planes))) return 6;
    }
  }
  return 0;
}
""".replace("LIMITS", limits).replace("WEIGHTS", weights)
    (tmp_path / "test.c").write_text(source)
    library = tmp_path / f"radix{radix_bits}_digits{digits}_k{k}.so"
    flags = ["-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off"]
    subprocess.run(
        [cc, *flags, "-shared", "-fPIC", str(tmp_path / "test.c"), "-o", str(library), "-lm"],
        check=True,
        capture_output=True,
    )
    assert ctypes.CDLL(str(library)).test() == 0
    (tmp_path / "runner.c").write_text("int test(void);int main(void){return test();}\n")
    executable = tmp_path / "sanitized"
    subprocess.run(
        [
            cc,
            *flags,
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            str(tmp_path / "test.c"),
            str(tmp_path / "runner.c"),
            "-o",
            str(executable),
            "-lm",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)
