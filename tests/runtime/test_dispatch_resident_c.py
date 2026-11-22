"""Only compiler-declared resident-only readonly ports can lack host backing."""
import shutil
import subprocess

import pytest

from merlin.compiler.lower.c.dispatch import emit_program_c
from test_dispatch_loop_c import fixture


def test_resident_port_null_dispatches_but_required_buffers_still_refuse(tmp_path):
    program, calls = fixture(count=1)
    compiler = shutil.which("cc")
    assert compiler, "generated C execution gate requires cc"
    _, ordinary = emit_program_c(program, "ordinary", calls, read_only_inputs=("x",))
    record, resident = emit_program_c(program, "resident", calls, read_only_inputs=("x",),
                                     resident_only_inputs=("x",))
    assert record["resident_only_inputs"] == ["x"]
    for key, value in {**ordinary, **resident}.items():
        (tmp_path / key).write_text(value)
    (tmp_path / "check.c").write_text('''
#include "ordinary.h"
#include "resident.h"
#include <assert.h>
#include <string.h>
static unsigned visits;
int variant0(void *context,const void *const *inputs,void *const *outputs) {
  (void)context; assert(!inputs[0]); ++visits;
  memcpy(outputs[0],inputs[1],4); return 0;
}
int main(void) {
  unsigned char arena[64],constant[4]={1,2,3,4},out[4]={0};
  void *items[1]={0}; resident_contexts contexts={items,1};
  resident_state state={0}; ordinary_state old={0};
  resident_buffer buffers[]={{0,4},{constant,4},{out,4}};
  ordinary_buffer old_buffers[]={{0,4},{constant,4},{out,4}};
  assert(ordinary_run(&old,arena,sizeof(arena),old_buffers,3,&contexts)==ORDINARY_INVALID);
  assert(!visits);
  assert(resident_run(&state,arena,sizeof(arena),buffers,3,&contexts)==RESIDENT_OK);
  assert(visits==1 && memcmp(out,constant,4)==0);
  buffers[1].data=0;
  assert(resident_run(&state,arena,sizeof(arena),buffers,3,&contexts)==RESIDENT_INVALID);
  assert(visits==1);
  return 0;
}
''')
    subprocess.run([compiler,"-std=c11","-Wall","-Wextra","-Werror","ordinary.c","resident.c",
                    "check.c","-o","check"],cwd=tmp_path,check=True,capture_output=True)
    subprocess.run([str(tmp_path/"check")],check=True,capture_output=True)


def test_resident_only_contract_requires_distinct_readonly_source_ports():
    program, calls = fixture(count=1)
    for readonly, resident in (((), ("x",)), (("x",), ("x", "x")), (("x",), ("constant",))):
        with pytest.raises(ValueError, match="resident-only"):
            emit_program_c(program, "resident", calls, read_only_inputs=readonly, resident_only_inputs=resident)
