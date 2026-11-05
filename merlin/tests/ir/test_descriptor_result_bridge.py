"""Execute the void bridge and corrupt every returned descriptor component.

The external CPU implementation is a test fixture, not a qualified device.
The bridge checks identity; it grants no memory effects or completion law.
"""

from __future__ import annotations

import subprocess

import pytest

from merlin.llvmlower.device_shim import _borrowed_result_wrapper
from merlin.llvmlower.toolchain import clang


@pytest.mark.parametrize("rank", (2, 3))
def test_borrowed_bridge_forwards_all_arguments_and_rejects_every_bad_result(tmp_path, rank):
    declarations, arguments, reconstructions = [], [], []
    for name in ("a", "b", "c"):
        fields = [("void *", "alloc"), ("void *", "aligned"), ("intptr_t ", "off")]
        fields.extend(("intptr_t ", f"s{i}") for i in range(rank))
        fields.extend(("intptr_t ", f"st{i}") for i in range(rank))
        declarations.extend(f"{typ}{name}_{field}" for typ, field in fields)
        values = [f"{name}_alloc", f"{name}_aligned", f"{name}_off"]
        values += ["{" + ",".join(f"{name}_s{i}" for i in range(rank)) + "}"]
        values += ["{" + ",".join(f"{name}_st{i}" for i in range(rank)) + "}"]
        reconstructions.append(f"D {name} = {{" + ",".join(values) + "};")
        arguments.append(f"memcmp(&{name}, &expected_{name}, sizeof(D))")
    mutations = [
        "r.allocated = other;",
        "r.aligned = other;",
        "r.offset++;",
        *(f"r.sizes[{i}]++;" for i in range(rank)),
        *(f"r.strides[{i}]++;" for i in range(rank)),
    ]
    mutate = "\n".join(f"if (mode == {i}) {{ {statement} }}" for i, statement in enumerate(mutations, 1))
    sizes = ",".join(str(i + 2) for i in range(rank))
    strides = ",".join(str(17 - i * 2) for i in range(rank))
    source = f"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
typedef struct {{ void *allocated, *aligned; intptr_t offset;
                  intptr_t sizes[{rank}], strides[{rank}]; }} merlin_memref_{rank}d;
typedef merlin_memref_{rank}d D;
static unsigned char payload_a[128], payload_b[128], payload_c[128], other[128];
static D expected_a, expected_b, expected_c;
static int calls, wrong_arguments, mode;
static D implementation({",".join(declarations)}) {{
  {" ".join(reconstructions)}
  calls++;
  if ({" || ".join(arguments)}) wrong_arguments=1;
  ((unsigned char *)c.aligned)[c.offset]=91;
  D r=c;
  {mutate}
  return r;
}}
{_borrowed_result_wrapper("implementation", rank, "borrowed")}
int main(int argc, char **argv) {{
  struct rlimit limit={{0,0}}; setrlimit(RLIMIT_CORE,&limit);
#ifdef __linux__
  if(prctl(PR_SET_DUMPABLE,0)) return 9;
#endif
  mode=argc>1 ? atoi(argv[1]) : 0;
  memset(payload_a,31,sizeof(payload_a)); memset(payload_b,43,sizeof(payload_b));
  memset(payload_c,67,sizeof(payload_c));
  expected_a=(D){{payload_a,payload_a+3,2,{{{sizes}}},{{{strides}}}}};
  expected_b=(D){{payload_b,payload_b+5,4,{{{sizes}}},{{{strides}}}}};
  expected_c=(D){{payload_c,payload_c+7,6,{{{sizes}}},{{{strides}}}}};
  _mlir_ciface_borrowed(mode==100 ? 0 : &expected_a,
                        mode==101 ? 0 : &expected_b, mode==102 ? 0 : &expected_c);
  if(calls!=1 || wrong_arguments) return 1;
  for(int i=0;i<128;i++) {{
    if(payload_a[i]!=31 || payload_b[i]!=43) return 2;
    if(payload_c[i]!=(i==13 ? 91 : 67)) return 3;
  }}
  return 0;
}}
"""
    c_path, executable = tmp_path / "bridge.c", tmp_path / "bridge"
    c_path.write_text(source)
    built = subprocess.run(
        [str(clang()), "-std=c11", "-Wall", "-Wextra", "-O2", str(c_path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert built.returncode == 0, built.stderr
    actual = subprocess.run([str(executable)], capture_output=True, timeout=5)
    assert actual.returncode == 0, (actual.stdout, actual.stderr)
    for mode in (*range(1, len(mutations) + 1), 100, 101, 102):
        bad = subprocess.run([str(executable), str(mode)], capture_output=True, timeout=5)
        assert bad.returncode < 0, (mode, bad.returncode, bad.stdout, bad.stderr)


@pytest.mark.parametrize(
    "symbol,rank,borrowed",
    [
        ("same", 2, "same"),
        ("bad-name", 2, "ok"),
        ("ok", 2, "bad();"),
        ("", 2, "ok"),
        ("ok", 1, "other"),
        ("ok", True, "other"),
        ("ok", 4, "other"),
    ],
)
def test_bridge_requires_existing_ranked_abi_and_distinct_portable_symbols(symbol, rank, borrowed):
    with pytest.raises(ValueError):
        _borrowed_result_wrapper(symbol, rank, borrowed)
