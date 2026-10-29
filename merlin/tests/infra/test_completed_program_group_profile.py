"""Observe real unchanged model objects through additional host counter wrappers."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common import invocation_record
from merlin.llvmlower.group_profile_sibling import completed_sibling_bindings
from merlin.llvmlower.sibling_source_route import borrowed_bridge_source
from merlin.model_check import chain_image_profile_new
from merlin.model_check.chain_image_objects import pin
from merlin.model_check.chain_image_profile import prepare, verify
from merlin.runtime.whole_model_readback import ReadbackError, parse_group_profile


def compile_c(tmp_path, name, text):
    source, obj = tmp_path / (name + ".c"), tmp_path / (name + ".o")
    source.write_text(text)
    compiler = str(Path(shutil.which("cc")).resolve())
    argv = [compiler, "-O2", "-c", str(source), "-o", str(obj)]
    subprocess.run(argv, check=True, capture_output=True)
    command = dict(requested_output=str(obj), argv=argv, inputs=[pin(source)], executable=pin(Path(compiler)))
    return obj, command


def test_completed_nonprofiled_model_object_is_reused_and_native_outputs_preserved(tmp_path, monkeypatch):
    if not shutil.which("cc") or not shutil.which("nm"):
        pytest.skip("native C toolchain unavailable")
    model, command = compile_c(tmp_path, "model", """
extern void _mlir_ciface_kernel(void *,const void *,const void *,const void *);
int invoke(int a,int b) { int output=0; _mlir_ciface_kernel(0,&a,&b,&output); return output; }
""")
    kernel, _ = compile_c(tmp_path, "kernel", """
void _mlir_ciface_kernel(void *r,const void *a,const void *b,const void *out) {
 (void)r; *(int *)out=*(const int *)a+*(const int *)b;
}
""")
    driver, driver_command = compile_c(tmp_path, "model_main", "void unused_driver(void) {}\n")
    # A producer roster can name several logical source groups sharing one
    # external callee. It supplies no actual completed-object call order.
    rows = [dict(symbol="kernel", group=37), dict(symbol="kernel", group=91)]
    routing = dict(routed=rows, signatures=dict(kernel=[2,2,2]), call_buffers=dict(kernel=[
        dict(role="input_0",shape=[2,2]), dict(role="weight_0",shape=[2,2]), dict(role="out_0",shape=[2,2])]))
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps(routing))
    receipt, recipe = tmp_path / "receipt.json", tmp_path / "recipe.json"
    receipt.write_text("{}")
    recipe.write_text("{}")
    original = [pin(model), pin(kernel), pin(driver)]
    program = dict(objects=original, by_name={Path(p["path"]).name:p for p in original},
        receipt=dict(output=dict(device_sidecar=pin(sidecar), device_artifacts=dict(kernels={}))),
        receipt_pin=pin(receipt), recipe_pin=pin(recipe), recipe=dict(commands=[command,driver_command]), root=tmp_path)
    emit = chain_image_profile_new.emit_profile

    def portable_counter(*args, **kwargs):
        source, flags = emit(*args, **kwargs)
        text = source.read_text().replace('#include <stdint.h>', '#include <stdint.h>\nstatic uint64_t test_ticks;')
        source.write_text(text.replace('  __asm__ volatile("rdcycle %0" : "=r"(c));', '  c = ++test_ticks;'))
        return source, flags

    monkeypatch.setattr(chain_image_profile_new, "emit_profile", portable_counter)
    selected, record = prepare(program, tmp_path / "profile", nm=Path(shutil.which("nm")).resolve(), timeout=10)
    assert selected["objects"][:-1] == original and program["objects"] == original
    assert selected["profile_wraps"] == ("_mlir_ciface_kernel",)
    verify(record)
    main, _ = compile_c(tmp_path, "main", """
#include <stdio.h>
extern int invoke(int,int); extern void merlin_group_profile_reset(void);
extern void merlin_group_profile_dump(unsigned long long,unsigned long long);
void htif_putc(char c) { putchar(c); } void htif_puts(const char *s) { fputs(s,stdout); }
int main(void) {
 for(int i=0;i<4;++i) { merlin_group_profile_reset();
  printf("VALUE %d\\n",invoke(i,7)); merlin_group_profile_dump(0,1000); }
}
""")
    elf = tmp_path / "run"
    subprocess.run(["cc",str(main),str(model),str(kernel),record["object"]["path"],
        "-Wl,--wrap=_mlir_ciface_kernel","-o",str(elf)], check=True,capture_output=True)
    console = subprocess.run([str(elf)],check=True,capture_output=True,text=True).stdout
    assert [line for line in console.splitlines() if line.startswith("VALUE")] == [f"VALUE {i+7}" for i in range(4)]
    blocks = console.split("VALUE ")[1:]
    assert all(parse_group_profile(block)["calls"] == 1 for block in blocks)
    assert all(parse_group_profile(block)["groups"][0]["name"] == "_mlir_ciface_kernel" for block in blocks)
    bindings = json.loads((tmp_path / "profile/group_profile_bindings.json").read_text())
    assert bindings["labels"] == "observed_callee_only"
    assert [pin(model), pin(kernel), pin(driver)] == original
    sidecar.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verify(record)


@pytest.fixture
def plural(tmp_path):
    call = [dict(role="input_0", shape=[2,4]), dict(role="out_0",shape=[2,4]), dict(role="out_1",shape=[2,4])]
    bridge = borrowed_bridge_source("device", "actual_void", call, [1,2])
    source = tmp_path / "shim.c"
    source.write_text("typedef struct { void *p; } merlin_lmemref_2d;\n"
        "typedef struct { int x; } device_results;\n"
        "void _mlir_ciface_device(device_results *r,const void*a,const void*b,const void*c) {"
        "(void)r; int v=*(int *)((const merlin_lmemref_2d*)a)->p;"
        "*(int *)((const merlin_lmemref_2d*)b)->p=v+3;"
        "*(int *)((const merlin_lmemref_2d*)c)->p=v+4;}\n" + bridge)
    obj = tmp_path / "device_shim.o"
    invocation_record.run([shutil.which("cc"),"-c",str(source),"-o",str(obj)], directory=tmp_path,
        stage="device_shim_compilation",inputs=(source,),outputs=(obj,),capture_output=True,check=True)
    observation = next(tmp_path.glob("invocations/*/invocation.json"))
    routing = dict(routed=[dict(symbol="device")],call_buffers=dict(device=call))
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps(routing))
    custody = dict(descriptor_bridge=dict(source=pin(source),object=pin(obj),compilation=pin(observation)))
    program = dict(objects=[pin(obj)],receipt=dict(output=dict(device_sidecar=pin(sidecar),
        device_artifacts=dict(kernels=dict(device=dict(custody=dict(sibling_source=custody)))))))
    return program, routing, source, obj


def test_plural_profile_abi_comes_from_complete_observed_bridge(plural):
    program, routing, _, _ = plural
    bindings, pins = completed_sibling_bindings(program,routing,{"_mlir_ciface_actual_void"})
    assert bindings == dict(device=dict(borrowed_symbol="actual_void",buffers=3,argument_ranks=[2,2,2]))
    assert pins


def test_plural_profile_preserves_every_output_pointer_in_native_call(plural, tmp_path):
    from merlin.runtime.whole_model_readback import callees, render_group_profile

    program, routing, _, obj = plural
    binding, _ = completed_sibling_bindings(program,routing,{"_mlir_ciface_actual_void"})
    selected = callees(routing['routed'],dict(device=[2,4,4]),{'_mlir_ciface_actual_void'},
                       routing['call_buffers'],binding)
    source = render_group_profile(selected, [('plural','device')])
    source = source.replace('#include <stdint.h>', '#include <stdint.h>\nstatic uint64_t test_ticks;')
    source = source.replace('  __asm__ volatile("rdcycle %0" : "=r"(c));', '  c = ++test_ticks;')
    wrapper, _ = compile_c(tmp_path,'counter',source)
    main, _ = compile_c(tmp_path,'caller', '''
typedef struct { void *p; } desc;
extern void _mlir_ciface_actual_void(const desc*,const desc*,const desc*);
void htif_puts(const char *s) {(void)s;} void htif_putc(char c) {(void)c;}
int main(void) { int a=7,b=0,c=0;desc da={&a},db={&b},dc={&c};
 _mlir_ciface_actual_void(&da,&db,&dc);return b!=10||c!=11||a!=7; }
''')
    elf=tmp_path/'plural-run'
    subprocess.run(['cc',str(main),str(obj),str(wrapper),'-Wl,--wrap=_mlir_ciface_actual_void',
                    '-o',str(elf)],check=True,capture_output=True)
    subprocess.run([str(elf)],check=True,capture_output=True)


@pytest.mark.parametrize("fault", ["source", "object", "unlinked", "reference", "outputs"])
def test_plural_profile_rejects_changed_or_incomplete_observations(plural, fault):
    program, routing, source, obj = plural
    undefined = {"_mlir_ciface_actual_void"}
    if fault == "source": source.write_text(source.read_text()+"\n")
    elif fault == "object": obj.write_bytes(b"changed")
    elif fault == "unlinked": program["objects"] = []
    elif fault == "reference": undefined = set()
    else: routing["call_buffers"]["device"][2]["role"] = "input_1"
    with pytest.raises(ReadbackError):
        completed_sibling_bindings(program,routing,undefined)
