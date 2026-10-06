"""Immutable pointer-use proof, transactional refusal and actual native words."""

import ctypes
import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.immutable_llvm_base import ImmutableBaseBinding, bind_immutable_llvm_bases
from merlin.llvmlower.toolchain import clang


def source(width=7, expression="fmul"):
    values = ", ".join(f"float {float(i + 1):.6e}" for i in range(width))
    return f"""@coeff = constant [{width} x float] [{values}]
define void @map(ptr %input, ptr %output, ptr %count) {{
entry:
  %n = load i64, ptr %count
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %body]
  %more = icmp ult i64 %i, %n
  br i1 %more, label %body, label %exit
body:
  %index = urem i64 %i, {width}
  %base = getelementptr [{width} x float], ptr @coeff, i64 0, i64 %index
  %scale = load float, ptr %base
  %src = getelementptr float, ptr %input, i64 %i
  %x = load float, ptr %src
  %result = {expression} float %x, %scale
  %dst = getelementptr float, ptr %output, i64 %i
  store float %result, ptr %dst
  %next = add i64 %i, 1
  br label %loop
exit:
  ret void
}}
"""


def bind(text, **kwargs):
    return bind_immutable_llvm_bases(
        text,
        bindings=(ImmutableBaseBinding("map", "coeff", "private_map"),),
        expected_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        **kwargs,
    )


def test_default_preserves_unknown_source_bytes():
    text = "not LLVM; empty explicit selection"
    assert bind_immutable_llvm_bases(text)[0] == text


def test_binding_retains_operation_body_and_public_signature():
    text = source()
    changed, report = bind(text)
    assert "define void @map(ptr %input, ptr %output, ptr %count)" in changed
    assert (
        "define hidden void @private_map(ptr %input, ptr %output, ptr %count, ptr %immutable_base) noinline" in changed
    )
    assert "call void @private_map(ptr %input, ptr %output, ptr %count, ptr @coeff)" in changed
    original_body = text.split("{", 1)[1].rsplit("}", 1)[0]
    assert changed.rsplit("{", 1)[1].rsplit("}", 1)[0] == original_body.replace("@coeff", "%immutable_base")
    assert report["routes"][0]["closed_derived_pointer_SSA"] == ["%base"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.replace("constant", "global", 1),
        lambda x: x.replace("constant", "thread_local constant", 1),
        lambda x: x.replace(
            "%scale = load float, ptr %base", "store float 0.000000e+00, ptr %base\n  %scale = load float, ptr %base"
        ),
        lambda x: x.replace(
            "%scale = load float, ptr %base", "call void @escape(ptr %base)\n  %scale = load float, ptr %base"
        ),
        lambda x: x.replace(
            "%scale = load float, ptr %base", "%address = ptrtoint ptr %base to i64\n  %scale = load float, ptr %base"
        ),
        lambda x: x + "\ndefine void @private_map() { ret void }\n",
        lambda x: x.replace("ptr %count)", "ptr noalias %count)"),
        lambda x: x.replace("ptr %count) {", "ptr %count) alwaysinline {"),
        lambda x: x.replace("ret void", "call void @map(ptr %input, ptr %output, ptr %count)\n  ret void"),
    ],
)
def test_mutated_immutable_or_escape_contract_refuses_without_mutation(mutation):
    text = mutation(source())
    before = text
    with pytest.raises(ValueError):
        bind(text)
    assert text == before


def test_changed_source_witness_and_multiple_binding_collisions_refuse():
    text = source()
    with pytest.raises(ValueError, match="witness"):
        bind_immutable_llvm_bases(
            text, bindings=(ImmutableBaseBinding("map", "coeff", "private_map"),), expected_source_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="duplicate"):
        bind_immutable_llvm_bases(
            text,
            bindings=(ImmutableBaseBinding("map", "coeff", "private_map"),) * 2,
            expected_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )


def test_later_invalid_binding_refuses_the_entire_transaction():
    text = source() + source(3).replace("@coeff", "@other").replace("@map", "@second")
    text = text.replace("@other = constant", "@other = global")
    before = text
    with pytest.raises(ValueError, match="immutable"):
        bind_immutable_llvm_bases(
            text,
            bindings=(
                ImmutableBaseBinding("map", "coeff", "private_map"),
                ImmutableBaseBinding("second", "other", "private_second"),
            ),
            expected_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
    assert text == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.replace("ret void", "%frame = call ptr @llvm.frameaddress.p0(i32 0)\n  ret void"),
        lambda x: x.replace("ret void", "%caller = call ptr @llvm.returnaddress(i32 0)\n  ret void"),
        lambda x: x.replace("ret void", "%stack = call ptr @llvm.stacksave.p0()\n  ret void"),
        lambda x: x.replace("ret void", "musttail call void @other(ptr %input, ptr %output, ptr %count)\n  ret void"),
        lambda x: x.replace("ptr %count) {", "ptr %count) returns_twice {"),
        lambda x: x.replace("ptr %count) {", "ptr %count) prologue i8 0 {"),
        lambda x: x.replace("ptr %count) {", 'ptr %count) gc "statepoint-example" {'),
        lambda x: x.replace("ptr %count) {", 'ptr %count) "presplitcoroutine" {'),
        lambda x: x + "\n@where = constant ptr blockaddress(@map, %body)\n",
    ],
)
def test_frame_context_and_escaping_block_identity_refuse(mutation):
    text = mutation(source())
    before = text
    with pytest.raises(ValueError, match="frame|context|block identity|header contract"):
        bind(text)
    assert text == before


@pytest.mark.parametrize("width,expression", [(7, "fmul"), (13, "fadd")])
def test_actual_native_empty_tails_nonuniform_and_special_words(tmp_path, width, expression):
    text = source(width, expression)
    changed, _ = bind(text)
    functions = []
    for label, llvm in (("control", text), ("candidate", changed)):
        path = tmp_path / (label + ".ll")
        path.write_text(llvm)
        library = tmp_path / (label + ".so")
        subprocess.run(
            [
                str(clang()),
                "-O3",
                "-ffp-contract=off",
                "-fPIC",
                "-shared",
                str(path),
                "-Wl,-Bsymbolic",
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
        )
        lib = ctypes.CDLL(str(library))
        fn = lib.map
        fn.argtypes = [ctypes.c_void_p] * 3
        fn.restype = None
        functions.append((lib, fn))
    raw = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x00800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0x7F812345,
        ],
        np.uint32,
    )
    values = np.concatenate((raw, np.random.default_rng(212).uniform(-500, 500, 65).astype(np.float32).view(np.uint32)))
    original = values.copy()
    for length in (0, 1, 3, 7, 13, len(values)):
        count = ctypes.c_uint64(length)
        outputs = []
        for _, fn in functions:
            output = np.full(len(values) + 16, 0xDEADBEEF, np.uint32)
            fn(values.ctypes.data, output[8:].ctypes.data, ctypes.addressof(count))
            assert np.all(output[:8] == 0xDEADBEEF) and np.all(output[8 + length :] == 0xDEADBEEF)
            outputs.append(output)
        np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(values, original)
    # Every original pointer identity survives the wrapper. No noalias fact is
    # granted when the caller uses the same mutable input/output allocation.
    count = ctypes.c_uint64(len(values))
    aliased = []
    for _, fn in functions:
        words = original.copy()
        fn(words.ctypes.data, words.ctypes.data, ctypes.addressof(count))
        aliased.append(words)
    np.testing.assert_array_equal(aliased[0], aliased[1])
    gate_source = tmp_path / "fenv_gate.c"
    gate_source.write_text("""#include <fenv.h>
#include <stdint.h>
#include <string.h>
typedef void(*F)(void*,void*,void*);
int check(F original,F candidate,void*input,uint64_t count,unsigned mode,unsigned preset){
 uint32_t a[128],b[128];fenv_t old;
 const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 const int flags[]={0,FE_INVALID,FE_DIVBYZERO,FE_OVERFLOW,FE_UNDERFLOW,FE_INEXACT,FE_ALL_EXCEPT};
 memset(a,73,sizeof(a));memset(b,73,sizeof(b));fegetenv(&old);fesetround(modes[mode]);
 feclearexcept(FE_ALL_EXCEPT);feraiseexcept(flags[preset]);original(input,a+8,&count);
 int af=fetestexcept(FE_ALL_EXCEPT);
 feclearexcept(FE_ALL_EXCEPT);feraiseexcept(flags[preset]);candidate(input,b+8,&count);
 int bf=fetestexcept(FE_ALL_EXCEPT);fesetenv(&old);
 return af!=bf?1:memcmp(a,b,sizeof(a))?2:0;
}
""")
    gate_so = tmp_path / "fenv_gate.so"
    subprocess.run(
        [str(clang()), "-O3", "-fno-builtin", "-fPIC", "-shared", str(gate_source), "-lm", "-o", str(gate_so)],
        check=True,
        capture_output=True,
    )
    gate_library = ctypes.CDLL(str(gate_so))
    gate = gate_library.check
    gate.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint]
    gate.restype = ctypes.c_int
    for length in (0, 3, len(values)):
        for mode in range(4):
            for preset in range(7):
                assert (
                    gate(
                        *(ctypes.cast(fn, ctypes.c_void_p) for _, fn in functions),
                        values.ctypes.data,
                        length,
                        mode,
                        preset,
                    )
                    == 0
                )
