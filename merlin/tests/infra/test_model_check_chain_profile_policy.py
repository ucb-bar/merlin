"""Real link behavior and conservative refusal; no model or FPGA execution."""

import json
import os
import shutil
from pathlib import Path

import pytest
from merlin.common import owned_process
from merlin.model_check import chain_image_objects as objects
from merlin.model_check import chain_image_profile_policy as P
from merlin.runtime import whole_model_readback as RB


def selected_tool(name, override):
    candidate = os.environ.get(override) or shutil.which(name)
    return None if candidate is None else Path(candidate).resolve()


# Cross-tool availability is explicit; pure policy and native tests still run
# when a RISC-V compiler is absent. CI can select a compiler without repo paths.
NM = selected_tool("llvm-nm", "MERLIN_TEST_LLVM_NM") or selected_tool(
    "nm", "MERLIN_TEST_NM"
)
OBJCOPY = selected_tool("llvm-objcopy", "MERLIN_TEST_LLVM_OBJCOPY") or selected_tool(
    "objcopy", "MERLIN_TEST_OBJCOPY"
)
RV = selected_tool("riscv64-unknown-elf-gcc", "MERLIN_TEST_RISCV_GCC")
CC = selected_tool("gcc", "MERLIN_TEST_CC")
ROUTE = {
    "routed": [{"group": 0, "symbol": "product"}],
    "signatures": {"product": [1, 4, 4]},
    "call_buffers": {
        "product": [
            {"role": r, "shape": [1, 4]} for r in ["input_0", "weight_0", "out_0"]
        ]
    },
}
SYMBOL = "_mlir_ciface_product"
MODEL = """extern void _mlir_ciface_product(void*,const void*,const void*,const void*);
extern int semantic_adjust(int);
void forward(int*a,int*b,int*out) { void*result=0; _mlir_ciface_product(&result,a,b,out); if(result!=out)__builtin_trap(); *out+=semantic_adjust(1); }
"""
DEVICE = "void _mlir_ciface_product(void*r,const void*a,const void*b,const void*c) { *(void**)r=(void*)c; *(int*)c=*(const int*)a**(const int*)b; }\n"
SEMANTIC = """int semantic_calls;
int semantic_adjust(int x) { return x+2; }
extern int __real_semantic_adjust(int);
int __wrap_semantic_adjust(int x) { semantic_calls++; return __real_semantic_adjust(x)+5; }
"""
MAIN = """extern void merlin_group_profile_dump(unsigned long,unsigned long);
int MERLIN_ARGS[1]={1}; void*MERLIN_INPUT_PTR[1];void*MERLIN_OUTPUT_PTR[1];
void original_driver(void){merlin_group_profile_dump(0,1);}
"""


def run(argv):
    return owned_process.run(
        list(map(str, argv)), capture_output=True, text=True, timeout=30
    )


def compile_c(directory, name, source, compiler):
    p = directory / (name + ".c")
    p.write_text(source)
    out = directory / (name + ".o")
    argv = [
        str(compiler),
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-c",
        str(p),
        "-o",
        str(out),
    ]
    result = run(argv)
    assert result.returncode == 0, result.stderr
    cmd = {
        "argv": argv,
        "requested_output": str(out),
        "inputs": [objects.pin(p)],
        "executable": objects.pin(compiler),
    }
    return objects.pin(out), cmd


def original_source():
    return RB.render_group_profile(
        RB.callees(
            ROUTE["routed"], ROUTE["signatures"], {SYMBOL}, ROUTE["call_buffers"]
        ),
        RB.group_names(ROUTE["routed"]),
    )


def fixture(directory, compiler=RV, *, host=False):
    if compiler is None:
        pytest.skip(
            "explicit fixture compiler unavailable; set MERLIN_TEST_RISCV_GCC or MERLIN_TEST_CC"
        )
    assert NM is not None and OBJCOPY is not None, "nm/objcopy test tools required"
    directory.mkdir()
    source = original_source()
    if host:
        # Host fixture ONLY: replace the RISC-V clock instruction with a
        # counted host clock; first independently recognize unchanged source.
        assert P.recognize(source, ROUTE, {SYMBOL})[0] == (SYMBOL,)
        old = """static inline uint64_t merlin_rdcycle(void) {
  uint64_t c;
  __asm__ volatile("rdcycle %0" : "=r"(c));
  return c;
}"""
        assert source.count(old) == 1
        source = source.replace(
            old,
            "extern uint64_t test_clock;\nstatic inline uint64_t merlin_rdcycle(void) { return ++test_clock; }",
            1,
        )
    products = []
    commands = []
    for name, text in [
        ("model", MODEL),
        (
            "model_call",
            "extern void forward(int*,int*,int*); void merlin_invoke(int*a,int*b,int*c){forward(a,b,c); }",
        ),
        ("model_main", MAIN),
        ("device", DEVICE),
        ("semantic", SEMANTIC),
        ("group_profile", source),
    ]:
        obj, cmd = compile_c(directory, name, text, compiler)
        products.append(obj)
        commands.append(cmd)
    sidecar = directory / "device_signatures.json"
    sidecar.write_text(json.dumps(ROUTE))
    receipt = directory / "receipt.json"
    receipt.write_text("{}")
    recipe = directory / "recipe.json"
    recipe.write_text("{}")
    return {
        "objects": products,
        "by_name": {Path(p["path"]).name: p for p in products},
        "receipt": {"output": {"device_sidecar": objects.pin(sidecar)}},
        "receipt_pin": objects.pin(receipt),
        "recipe_pin": objects.pin(recipe),
        "recipe": {"commands": commands},
        "link": {"argv": ["cc", "-Wl,--wrap=" + SYMBOL, "-Wl,--wrap=semantic_adjust"]},
    }


@pytest.mark.parametrize(
    "value,profile,expected",
    [
        ("retain", False, False),
        ("retain", True, False),
        ("bypass_verified_neutral", False, True),
    ],
)
def test_explicit_selection(value, profile, expected):
    assert P.selected(value, group_profile=profile) == expected


@pytest.mark.parametrize(
    "value,profile",
    [
        ("bypass_verified_neutral", True),
        ("unknown", False),
        (True, False),
        ("retain", 1),
    ],
)
def test_invalid_selection(value, profile):
    with pytest.raises(ValueError):
        P.selected(value, group_profile=profile)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s + "/* unknown */\n",
        lambda s: s.replace(
            "merlin_group_record(0u,", "forbidden_release(); merlin_group_record(0u,"
        ),
        lambda s: s.replace("merlin_group_count++;", "*(volatile int*)0=1;"),
    ],
)
def test_modified_generated_source_refuses(change):
    with pytest.raises(ValueError, match="exact known"):
        P.recognize(change(original_source()), ROUTE, {SYMBOL})


def test_known_resetless_form_and_both_label_modes():
    assert (
        P.recognize(original_source().replace(P.RESET, "", 1), ROUTE, {SYMBOL})[1][1]
        == "known_without_reset"
    )
    selected = RB.callees(
        ROUTE["routed"], ROUTE["signatures"], {SYMBOL}, ROUTE["call_buffers"]
    )
    assert (
        P.recognize(RB.render_group_profile(selected, []), ROUTE, {SYMBOL})[1][0]
        == "observed_callee_only"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["-Wl,--wrap,foo"],
        ["-Xlinker", "--wrap=foo"],
        ["-Wl,-z,defs,--wrap=foo"],
        ["@response"],
        ["-Wl,@response"],
        ["-Wl,--wrap=x", "-Wl,--wrap=x"],
    ],
)
def test_unsupported_wrap_forms_refuse(argv):
    with pytest.raises(ValueError):
        P.wraps(argv)


def test_real_riscv_qualification_and_revalidation(tmp_path):
    program = fixture(tmp_path / "source")
    owner = P.prepare(program, tmp_path / "qualification", nm=NM, timeout=20)
    record = owner.record()
    assert record["profiling_wraps"] == [SYMBOL]
    assert record["retained_semantic_wraps"] == ["semantic_adjust"]
    # A report is not an owner, and an owner cannot authorize modified bytes.
    with pytest.raises(ValueError, match="serialized"):
        P.rewrite([], [record], [{}], nm=NM, timeout=20)
    path = Path(program["by_name"]["group_profile.o"]["path"])
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        owner.record()


@pytest.mark.parametrize("missing,unsafe", [(True, False), (False, True)])
def test_original_wrap_or_compile_macro_mismatch_refuses(tmp_path, missing, unsafe):
    program = fixture(tmp_path / "source")
    if missing:
        program["link"]["argv"].remove("-Wl,--wrap=" + SYMBOL)
    if unsafe:
        program["recipe"]["commands"][-1]["argv"].insert(
            1, "-Dmerlin_group_record=evil"
        )
    with pytest.raises(ValueError):
        P.prepare(program, tmp_path / "qualification", nm=NM, timeout=20)


def test_direct_profile_reference_in_forward_refuses(tmp_path):
    program = fixture(tmp_path / "source")
    row, _cmd = compile_c(
        tmp_path / "source",
        "model",
        MODEL
        + "extern unsigned long merlin_group_profile_now(void); unsigned long hidden(void){return merlin_group_profile_now();}\n",
        RV,
    )
    program["by_name"]["model.o"] = row
    program["objects"][0] = row
    with pytest.raises(ValueError, match="directly references"):
        P.prepare(program, tmp_path / "qualification", nm=NM, timeout=20)


@pytest.mark.parametrize("a,b", [(-7, 3), (0, 99), (127, -128)])
def test_real_host_link_bypasses_only_trace_and_preserves_all_outputs(tmp_path, a, b):
    program = fixture(tmp_path / "host", CC, host=True)
    ns = objects.namespace(
        program, 0, tmp_path / "namespaced", nm=NM, objcopy=OBJCOPY, timeout=20
    )
    # This internal host fixture owner is deliberately NOT a production
    # qualification: only its clock body was substituted above. Real source/
    # object qualification is exercised independently with RISC-V objects.
    owner = P._ObservationOwner(
        {
            "profiling_wraps": [SYMBOL],
            "inputs": [],
            "original_objects": program["objects"],
        },
        _owner=P._OWNED,
    )
    driver = tmp_path / "driver.c"
    driver.write_text(
        f"""#include <stdio.h>\n#include <stdint.h>\nuint64_t test_clock;\nvoid htif_puts(const char*s){{(void)s;}}void htif_putc(char c){{(void)c;}}\nextern int merlin_chain_p0_semantic_calls;\nextern void merlin_chain_p0_merlin_invoke(int*,int*,int*);\nint main(void){{int a={a},b={b},guard[3]={{-111,0,222}};merlin_chain_p0_merlin_invoke(&a,&b,&guard[1]);printf("%d %d %d %d %llu\\n",guard[0],guard[1],guard[2],merlin_chain_p0_semantic_calls,(unsigned long long)test_clock);return 0;}}\n"""
    )
    driver_o = tmp_path / "driver.o"
    result = run([CC, "-c", driver, "-o", driver_o])
    assert result.returncode == 0, result.stderr
    on = tmp_path / "on"
    off = tmp_path / "off"
    base = [
        str(CC),
        str(driver_o),
        *[p["path"] for p in ns["objects"]],
        *["-Wl,--wrap=" + s for s in ns["wraps"]],
        "-o",
        str(on),
    ]
    result = run(base)
    assert result.returncode == 0, result.stderr
    argv, expected, policy = P.rewrite(
        [str(off) if s == str(on) else s for s in base],
        [owner],
        [ns],
        nm=NM,
        timeout=20,
    )
    argv[-2:-2] = ["-Wl,--trace-symbol=" + s for s in sorted(expected)]
    result = run(argv)
    assert result.returncode == 0, result.stderr
    observed = P.verify_final(
        policy,
        off,
        argv=argv,
        result=result,
        inputs=[objects.pin(driver_o), *ns["objects"]],
        nm=NM,
        timeout=20,
        directory=tmp_path,
    )
    before = run([on])
    after = run([off])
    assert before.returncode == after.returncode == 0
    assert (
        before.stdout.split()[:4]
        == after.stdout.split()[:4]
        == ["-111", str(a * b + 8), "222", "1"]
    )
    assert before.stdout.split()[4] == "2" and after.stdout.split()[4] == "0"
    assert observed["policy"]["retained_wraps"] == ["merlin_chain_p0_semantic_adjust"]
    assert observed["policy"]["removed_wraps"] == ["merlin_chain_p0_" + SYMBOL]
    # A changed final link cannot inherit this observation.
    with pytest.raises(ValueError):
        P.verify_final(
            policy,
            off,
            argv=[*argv, "-Wl,--defsym,arbitrary=0"],
            result=result,
            inputs=[],
            nm=NM,
            timeout=20,
            directory=tmp_path,
        )


def test_mutated_live_record_refuses():
    owner = P._ObservationOwner({"inputs": [], "profiling_wraps": []}, _owner=P._OWNED)
    owner._record["profiling_wraps"] = ["unqualified"]
    with pytest.raises(ValueError, match="changed"):
        owner.record()


def test_absent_standard_profile_object_with_intercepted_group_refuses(tmp_path):
    program = fixture(tmp_path / "source")
    del program["by_name"]["group_profile.o"]
    with pytest.raises(ValueError, match="no known retained"):
        P.prepare(program, tmp_path / "qualification", nm=NM, timeout=20)
