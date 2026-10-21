"""Pre-encoded BF16 boundaries validate codes without rounding or partial writes."""

import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compile.c_conversions import render_bf16_codes_c
from merlin.semantic_compiler.precision_values import FloatBoundarySpec


def test_all_codes_and_late_invalid_lane(tmp_path):
    spec = FloatBoundarySpec("explicit-test-domain", "bf16", "encode")
    ranges = ((0, 0), (0x3F00, 0x4000), (0x8000, 0x8000), (0xBF00, 0xC000))
    source = tmp_path / "boundary.c"
    source.write_text(
        render_bf16_codes_c(spec, 1, "finite")
        + render_bf16_codes_c(replace(spec, admitted_code_ranges=ranges), 1, "bounded")
        + render_bf16_codes_c(spec, 3, "vector")
        + """
#include <assert.h>
#include <string.h>
int main(void) {
  unsigned admitted=0;
  for (unsigned code=0;code<65536u;++code) {
    uint8_t in[2]={(uint8_t)code,(uint8_t)(code>>8)}, out[4];
    for (unsigned domain=0;domain<2u;++domain) {
      memset(out,0xa7,sizeof(out));
      int expected=(code&0x7fffu)<=0x7f7fu;
      if (domain) expected=expected && (code==0 || code==0x8000u ||
        (code>=0x3f00u && code<=0x4000u) || (code>=0xbf00u && code<=0xc000u));
      int result=domain?bounded(in,out+1):finite(in,out+1);
      assert((result==0)==expected);
      assert(out[0]==0xa7 && out[3]==0xa7);
      if (expected) { assert(out[1]==in[0] && out[2]==in[1]); if(!domain) ++admitted; }
      else assert(out[1]==0xa7 && out[2]==0xa7);
    }
  }
  assert(admitted==65280u);
  uint8_t in[6]={0x80,0x3f,0,0x80,1,0x7f},out[6];
  memset(out,0xa7,sizeof(out)); in[4]=0x81; in[5]=0xff;
  assert(vector(in,out)==2);
  for(unsigned i=0;i<6u;++i) assert(out[i]==0xa7);
  assert(finite(0,out)==1 && finite(in,0)==1);
  return 0;
}
"""
    )
    compiler = shutil.which("cc")
    assert compiler, "a C compiler is required"
    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(source),
                    "-o", str(tmp_path / "boundary")], check=True, capture_output=True)
    subprocess.run([str(tmp_path / "boundary")], check=True)


@pytest.mark.parametrize("change", [
    {"pre_encode_scale_exponent": 1}, {"canonicalize_negative_zero": True},
    {"flush_below_normal": True}, {"format": "fp8_e4m3"}, {"direction": "decode"},
])
def test_code_copy_does_not_admit_numerical_conversion(change):
    spec = replace(FloatBoundarySpec("selected-domain", "bf16", "encode"), **change)
    with pytest.raises(ValueError, match="without numerical conversion"):
        render_bf16_codes_c(spec, 4, "boundary")
