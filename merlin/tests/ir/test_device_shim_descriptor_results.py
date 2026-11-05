"""Execute actual emitted checked shims; these CPU kernels are not devices."""

import subprocess

import pytest

from merlin.llvmlower import device_shim as D
from merlin.llvmlower.toolchain import clang


@pytest.mark.parametrize("version,bits,acc", [(2, 8, 32), (1, 16, 64)])
def test_selected_emitted_forwarder_preserves_values_and_original_guards(tmp_path, monkeypatch, version, bits, acc):
    monkeypatch.setattr(D, "kernel_abi_for", lambda _: D.KernelAbi("kernel", version=version))
    shape = (3, 5, 7) if version == 2 else (2, 4, 6)
    m, n, k = shape
    kwargs = dict(tile_edge=2, argument_roles={"entry": ("lhs", "rhs", "out")})
    plain = D.emit_translation_unit(
        "neutral", {"entry": shape}, {"entry": (f"i{bits}", f"i{bits}", f"i{acc}")}, **kwargs
    )
    unit = D.emit_translation_unit(
        "neutral", {"entry": shape}, {"entry": (f"i{bits}", f"i{bits}", f"i{acc}")}, descriptor_results=True, **kwargs
    )
    assert unit.text.startswith(plain.text) and not plain.result_identities
    (identity,) = unit.result_identities
    assert identity.symbol == "entry" and identity.result_argument == 2
    assert identity.argument_access == ("read", "read", "write")
    assert [tuple(t.get_shape()) for t in identity.argument_types] == [(m, k), (k, n), (m, n)]
    assert [t.get_element_type().width.data for t in identity.argument_types] == [bits, bits, acc]
    assert D.descriptor_result_dependencies(unit) and not D.descriptor_result_dependencies(plain)
    parameters = "void *pa,void *pb,void *pc" if version == 2 else "void *pb,void *pa,void *pc"
    source = (
        unit.text
        + f"""
#include <stdlib.h>
#include <sys/resource.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
static int calls;
void kernel({parameters}) {{
  ++calls; int{bits}_t *a=pa,*b=pb; int{acc}_t *c=pc;
  for(int i=0;i<{m};i++)for(int j=0;j<{n};j++) {{
    int{acc}_t sum=0;for(int l=0;l<{k};l++)sum+=(int{acc}_t)a[i*{k}+l]*b[l*{n}+j];
    c[i*{n}+j]=sum;
  }}
}}
int main(int argc,char **argv) {{
  struct rlimit limit={{0,0}};setrlimit(RLIMIT_CORE,&limit);
#ifdef __linux__
  if(prctl(PR_SET_DUMPABLE,0))return 9;
#endif
  int mode=argc>1?atoi(argv[1]):0;
  int{bits}_t a[{m * k}],b[{k * n}]; int{acc}_t storage[{m * n + 4}];
  for(int i=0;i<{m * k};i++)a[i]=i%7-3;
  for(int i=0;i<{k * n};i++)b[i]=i%5-2;
  for(int i=0;i<{m * n + 4};i++)storage[i]=91;
  merlin_memref_2d x={{a,a,0,{{{m},{k}}},{{{k},1}}}};
  merlin_memref_2d w={{b,b,0,{{{k},{n}}},{{{n},1}}}};
  merlin_memref_2d y={{storage,storage+2,0,{{{m},{n}}},{{{n},1}}}};
  if(mode==1)x.sizes[0]++;
  if(mode==2)y.strides[1]++;
  if(mode==3)y.aligned=a;
  _mlir_ciface_{identity.borrowed_symbol}(&x,&w,&y);
  if(calls!=1)return 1;
  for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{
    int{acc}_t sum=0;for(int l=0;l<{k};l++)sum+=(int{acc}_t)((i*{k}+l)%7-3)*((l*{n}+j)%5-2);
    if(storage[2+i*{n}+j]!=sum)return 2;
  }}
  for(int i=0;i<{m * k};i++)if(a[i]!=i%7-3)return 3;
  for(int i=0;i<{k * n};i++)if(b[i]!=i%5-2)return 4;
  if(storage[0]!=91||storage[1]!=91||storage[{m * n + 2}]!=91||storage[{m * n + 3}]!=91)return 5;
  return 0;
}}
"""
    )
    path, executable = tmp_path / "actual-shim.c", tmp_path / "actual-shim"
    path.write_text(source)
    built = subprocess.run([str(clang()), "-O2", str(path), "-o", str(executable)], capture_output=True, timeout=30)
    assert built.returncode == 0, built.stderr
    assert subprocess.run([str(executable)], capture_output=True, timeout=5).returncode == 0
    for mode in (1, 2, 3):
        assert subprocess.run([str(executable), str(mode)], capture_output=True, timeout=5).returncode < 0


@pytest.mark.parametrize("shape,dtypes", [((2, 3, 4, 5), ("i8", "i8", "i32")), ((3, 5, 7), ("f32",) * 3)])
def test_unsupported_alias_domain_retains_original_entry_bytes(monkeypatch, shape, dtypes):
    monkeypatch.setattr(D, "kernel_abi_for", lambda _: D.KernelAbi("kernel", version=2))
    kwargs = dict(argument_roles={"entry": ("lhs", "rhs", "out")})
    original = D.emit_translation_unit("neutral", {"entry": shape}, {"entry": dtypes}, **kwargs)
    selected = D.emit_translation_unit(
        "neutral", {"entry": shape}, {"entry": dtypes}, descriptor_results=True, **kwargs
    )
    assert selected == original and selected.symbols == ("entry",)


def test_selected_generated_names_cannot_capture_another_entry(monkeypatch):
    monkeypatch.setattr(D, "kernel_abi_for", lambda _: D.KernelAbi("kernel", version=2))
    signatures = {"entry": (3, 5, 7), "entry_borrowed_result": (3, 5, 7)}
    with pytest.raises(ValueError, match="symbol collision"):
        D.emit_translation_unit(
            "neutral",
            signatures,
            {key: ("i8", "i8", "i32") for key in signatures},
            argument_roles={key: ("lhs", "rhs", "out") for key in signatures},
            descriptor_results=True,
        )


@pytest.mark.parametrize("selected", [1, None, "true"])
def test_malformed_selection_refuses_before_contract_callback(monkeypatch, selected):
    def forbidden(_):
        raise AssertionError("malformed original selection reached a contract callback")

    monkeypatch.setattr(D, "kernel_abi_for", forbidden)
    with pytest.raises(ValueError, match="original boolean"):
        D.emit_translation_unit("neutral", {}, {}, descriptor_results=selected)
