"""Native observations of forced libm and actual MLIR before/after, never a numerical proof."""

import ctypes
import json
import platform
import random
import shutil
import subprocess
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.differential import Tools, ValueType, execute_mlir


def inputs(kind):
    width, fraction, exponent = (32, 23, 8) if kind == "f32" else (64, 52, 11)
    sign, inf = 1 << (width - 1), ((1 << exponent) - 1) << fraction
    one = ((1 << (exponent - 1)) - 1) << fraction
    cases = [
        0,
        sign,
        1,
        sign | 1,
        (1 << fraction) - 1,
        1 << fraction,
        one,
        sign | one,
        one - 1,
        one + 1,
        inf,
        sign | inf,
        inf - 1,
        inf | 1,
        inf | (1 << (fraction - 1)),
        sign | inf | 7,
        inf | (1 << (fraction - 1)) | 123,
    ]
    cases += [0x3F10D742, 0x3F2FA800, 0x3F0E5874] if width == 32 else [0x3FEFB55750829A12, 0x3FED510F0375636C]
    generator = random.Random(9203)
    cases += [generator.getrandbits(width) for _ in range(2048)]
    cases += [(generator.getrandbits(fraction) | (one - (1 << fraction))) for _ in range(2048)]
    return cases


def is_nan(bits, width):
    fraction, exponent = (23, 8) if width == 32 else (52, 11)
    return bits >> fraction & ((1 << exponent) - 1) == (1 << exponent) - 1 and bits & ((1 << fraction) - 1) != 0


def direct_libm(kind, values, work):
    work.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("cc")
    if not compiler:
        raise RuntimeError("native scalar-square oracle requires a C compiler")
    source = repo_root() / "merlin/tests/fixtures/scalar_square_native.c"
    library = work / "probe.so"
    flags = [
        "-O2",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-fno-builtin",
        "-fno-fast-math",
        "-frounding-math",
        "-ffp-contract=off",
    ]
    subprocess.run([compiler, *flags, str(source), "-lm", "-o", str(library)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    assert lib.ss_round_nearest() == 0
    width = 32 if kind == "f32" else 64
    word = ctypes.c_uint32 if width == 32 else ctypes.c_uint64
    power, multiply = getattr(lib, f"ss_pow{width}"), getattr(lib, f"ss_mul{width}")
    for function in (power, multiply):
        function.argtypes, function.restype = [word], word
    records = []
    for bits in values:
        p = power(bits)
        pe, pf = lib.ss_last_errno(), lib.ss_last_exceptions()
        m = multiply(bits)
        records.append(
            {
                "input": bits,
                "pow": p,
                "mul": m,
                "pow_errno": pe,
                "pow_exceptions": pf,
                "mul_errno": lib.ss_last_errno(),
                "mul_exceptions": lib.ss_last_exceptions(),
            }
        )
    differences = [row for row in records if row["pow"] != row["mul"]]
    linked = subprocess.run(["ldd", str(library)], check=True, text=True, capture_output=True).stdout
    linked_files = []
    for line in linked.splitlines():
        for token in line.split():
            if token.startswith("/") and Path(token).is_file():
                linked_files.append(_file_ref(Path(token).resolve()))
    result = {
        "format": kind,
        "rounding": "FE_TONEAREST",
        "samples": len(values),
        "bit_differences": differences,
        "all_observations": records,
        "law_status": "refuted_for_forced_host_libm" if differences else "not_proved_by_samples",
        "source": _file_ref(source),
        "compiler": _file_ref(Path(compiler).resolve()),
        "compiler_version": subprocess.run([compiler, "--version"], check=True, text=True, capture_output=True).stdout,
        "flags": flags,
        "linked_libraries": linked_files,
        "host": platform.uname()._asdict(),
    }
    (work / "libm.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def differential(before, after, kind, arity, compiler_python, work):
    values = inputs(kind)
    direct = direct_libm(kind, values, work / "forced-libm")
    tools = Tools(compiler_python, Path("unused"), Path("unused"))
    width = 32 if kind == "f32" else 64
    mask = (1 << width) - 1
    samples = [[[bits], *([[0]] * (arity - 1))] for bits in values]
    results = {}
    for label, source in (("before", before), ("after", after)):
        path = work / f"{label}.mlir"
        path.write_text(source)
        results[label] = execute_mlir(
            path, samples, work / label, tools, inputs=[ValueType(kind)] * arity, outputs=[ValueType(kind)]
        )
    observations = []
    for bits, p, m, oracle in zip(
        values, results["before"]["results"], results["after"]["results"], direct["all_observations"], strict=True
    ):
        p, m = int(p[0]) & mask, int(m[0]) & mask
        # Returned NaN payloads remain in the record. Classification agreement is a separate,
        # explicitly weaker observation; it does not discharge the exact-bit numerical law.
        if not is_nan(m, width):
            assert m == oracle["mul"], (kind, hex(bits), hex(m), hex(oracle["mul"]))
        else:
            assert is_nan(oracle["mul"], width)
        observations.append(
            {
                "input": bits,
                "before": p,
                "after": m,
                "bits_equal": p == m,
                "both_nan": is_nan(p, width) and is_nan(m, width),
            }
        )
    result = {
        "format": kind,
        "samples": len(values),
        "observations": observations,
        "bit_mismatches": [row for row in observations if not row["bits_equal"]],
        "native_metadata": {key: {k: v for k, v in value.items() if k != "results"} for key, value in results.items()},
        "forced_libm_status": direct["law_status"],
        "numerical_law_discharged": False,
    }
    (work / "differential.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
