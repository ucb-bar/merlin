"""Versioned host conversion proofs; generation never consumes a model or its inputs.

Run in the explicitly selected reference-framework environment with --write.
The checked certificate is a required source fixture, not a case verdict cache.
"""

import functools
import hashlib
import json
import struct
from pathlib import Path

FORMATS = {"f16": (10, 5, 15), "bf16": (7, 8, 127), "f32": (23, 8, 127), "f64": (52, 11, 1023)}
VERSION = "native-ieee-storage-rne-v2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dyadic_word(integer, power, token, *, negative_zero=False):
    """Independent unbounded-integer RNE specification, including overflow."""
    fraction, exponent, bias = FORMATS[token]
    negative = integer < 0 or (integer == 0 and negative_zero)
    magnitude = abs(integer)
    sign = int(negative) << (fraction + exponent)
    if not magnitude:
        return sign
    leading = magnitude.bit_length() - 1
    drop = max(leading - fraction, 1 - bias - fraction - power)
    if drop > 0:
        quotient, remainder = divmod(magnitude, 1 << drop)
        twice = 2 * remainder
        quotient += twice > 1 << drop or (twice == 1 << drop and quotient & 1)
    else:
        quotient = magnitude << -drop
    quantum = power + drop
    if quotient >= 1 << (fraction + 1):
        quotient >>= 1
        quantum += 1
    if not quotient:
        return sign
    real_exponent = quotient.bit_length() - 1 + quantum
    if real_exponent > bias:
        return sign | (((1 << exponent) - 1) << fraction)
    field = real_exponent + bias if real_exponent >= 1 - bias else 0
    return sign | (field << fraction) | (quotient & ((1 << fraction) - 1))


def convert_word(word, source, target, nan_policy="payload_quiet"):
    sf, se, sb = FORMATS[source]
    tf, te, _ = FORMATS[target]
    sign = word >> (sf + se)
    field = word >> sf & ((1 << se) - 1)
    payload = word & ((1 << sf) - 1)
    if field == (1 << se) - 1:
        if not payload:
            return sign << (tf + te) | (((1 << te) - 1) << tf)
        if nan_policy == "positive_all_payload":
            return (((1 << te) - 1) << tf) | ((1 << tf) - 1)
        if nan_policy == "all_ones":
            return (1 << (tf + te + 1)) - 1
        if nan_policy == "positive_quiet":
            return (((1 << te) - 1) << tf) | (1 << (tf - 1))
        payload = payload << (tf - sf) if tf >= sf else payload >> (sf - tf)
        if nan_policy != "payload_preserve":
            payload |= 1 << (tf - 1)
        return (sign << (tf + te)) | (((1 << te) - 1) << tf) | payload
    significand = payload | ((1 << sf) if field else 0)
    return dyadic_word(-significand if sign else significand, max(field, 1) - sb - sf, target, negative_zero=bool(sign))


def verify_outputs(words, actual, source, target, policy, *, converter=convert_word):
    expected = [converter(int(word), source, target, policy) for word in words]
    if list(map(int, actual)) != expected:
        first = next(i for i, (a, b) in enumerate(zip(actual, expected)) if int(a) != b)
        raise ValueError(
            f"conversion proof refused: {source}->{target}/{policy}, word={int(words[first]):#x}, "
            f"CPU={int(actual[first]):#x}, proposed={expected[first]:#x}"
        )
    return hashlib.sha256(
        b"".join(int(v).to_bytes((sum(FORMATS[target][:2]) + 1) // 8, "little") for v in expected)
    ).hexdigest()


def boundary_words(target):
    """All destination rounding cells: midpoint and binary32 neighbors, both signs.

    Each cell's integer RNE preimage is monotone. These are all transitions,
    including zero/subnormal/normal seams and overflow, not sampled examples.
    """
    fraction, exponent, bias = FORMATS[target]
    infinity = ((1 << exponent) - 1) << fraction
    values = {0, 0x80000000, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF}
    for lower in range(infinity):
        lo = struct.unpack("<f", convert_word(lower, target, "f32", "payload_preserve").to_bytes(4, "little"))[0]
        if lower + 1 == infinity:
            hi = 2.0 ** (bias + 1)
        else:
            hi = struct.unpack("<f", convert_word(lower + 1, target, "f32", "payload_preserve").to_bytes(4, "little"))[
                0
            ]
        mid = struct.unpack("<I", struct.pack("<f", (lo + hi) / 2))[0]
        for word in (mid - 1, mid, mid + 1):
            values.update((word, word | 0x80000000))
    # Every sign/high-payload class and the discarded-low-bit boundary classes.
    for high in range(1 << 10):
        for low in (0, 1, (1 << 13) - 1):
            payload = (high << 13) | low
            if payload:
                values.update((0x7F800000 | payload, 0xFF800000 | payload))
    return sorted(values)


def proof_sources():
    from merlin.common.paths import python_source_dir

    root = python_source_dir()
    names = [
        "src/merlin/targetgen/native_ieee_certificate.py",
        "src/merlin/llvmlower/ieee_dyadic_ir.py",
        "src/merlin/llvmlower/significand_products.py",
        "src/merlin/llvmlower/ieee_conversion_components.py",
        "src/merlin/llvmlower/ieee_pointwise.py",
        "src/merlin/targetgen/native_ieee_ir_check.py",
        "src/merlin/llvmlower/native_sign_components.py",
        "src/merlin/llvmlower/native_cpu_layout.py",
        "src/merlin/llvmlower/integer_division_components.py",
        "src/merlin/targetgen/native_ieee_arithmetic_check.py",
        "src/merlin/targetgen/native_nan_priority_check.py",
        "src/merlin/targetgen/native_layout_check.py",
        "src/merlin/targetgen/native_widening_check.py",
        "src/merlin/targetgen/native_integer_check.py",
        "src/merlin/targetgen/native_cpu_loop_check.py",
        "src/merlin/targetgen/native_typed_semantics.py",
        "src/merlin/llvmlower/uniform_numeric_literals.py",
        "src/merlin/llvmlower/floating_component_readout.py",
        "src/merlin/llvmlower/fixed_point_ir.py",
        "src/merlin/llvmlower/_ir.py",
        "src/merlin/targetgen/closed_scalar.py",
    ]
    return {name: hashlib.sha256((root / name.removeprefix("src/")).read_bytes()).hexdigest() for name in names}


def generate():
    import numpy as np
    import torch

    from merlin.targetgen.native_ieee_ir_check import conversion_function

    torch.set_num_threads(1)
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise ValueError("this certificate checks the explicit AVX2 source profile")
    types = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    rows = []
    for source in ("f16", "bf16"):
        words = np.arange(65536, dtype=np.uint16)
        tensor = torch.from_numpy(words).view(types[source])
        policies = ("payload_quiet", "positive_all_payload") if source == "f16" else ("payload_preserve",) * 2
        for mode, policy in zip(("vector", "scalar"), policies):
            if mode == "vector":
                actual = tensor.float().view(torch.uint32).tolist()
            else:
                actual = [tensor[i : i + 1].float().view(torch.uint32).item() for i in range(65536)]
            output = verify_outputs(words, actual, source, "f32", policy)
            emitted = conversion_function(source, "f32", policy)
            if list(map(int, emitted(words))) != actual:
                raise ValueError("emitted conversion IR differs from the full CPU storage domain")
            form = dict(
                source=source, target="f32", mode=mode, nan_policy=policy, rounding="RNE", domain="all_storage_words"
            )
            rows.append(
                dict(
                    form=form,
                    key=digest(form),
                    count=65536,
                    output_sha256=output,
                    executable_ir_sha256=hashlib.sha256(emitted.source.encode()).hexdigest(),
                )
            )
    for source, target, policy in (("f16", "bf16", "positive_quiet"), ("bf16", "f16", "payload_quiet")):
        words = np.arange(65536, dtype=np.uint16)
        tensor = torch.from_numpy(words).view(types[source])
        for mode in ("vector", "scalar"):
            actual = (
                tensor.to(types[target]).view(torch.uint16).tolist()
                if mode == "vector"
                else [tensor[i : i + 1].to(types[target]).view(torch.uint16).item() for i in range(65536)]
            )
            output = verify_outputs(words, actual, source, target, policy)
            emitted = conversion_function(source, target, policy)
            if list(map(int, emitted(words))) != actual:
                raise ValueError("emitted cross-storage conversion differs on its complete domain")
            form = dict(
                source=source, target=target, mode=mode, nan_policy=policy, rounding="RNE", domain="all_storage_words"
            )
            rows.append(
                dict(
                    form=form,
                    key=digest(form),
                    count=65536,
                    output_sha256=output,
                    executable_ir_sha256=hashlib.sha256(emitted.source.encode()).hexdigest(),
                )
            )
    for target in ("f16", "bf16"):
        words = np.array(boundary_words(target), dtype=np.uint32)
        tensor = torch.from_numpy(words).view(torch.float32)
        # Keep vector probes entirely within the vector prefix, then remove padding.
        padded = torch.cat((tensor, torch.zeros((-len(tensor)) % 32)))
        policies = ("payload_quiet", "payload_quiet") if target == "f16" else ("all_ones", "all_ones")
        modes = [("vector", policies[0]), ("scalar", policies[1])]
        if target == "bf16":
            modes.append(("opmath_scalar", "positive_quiet"))
        for mode, policy in modes:
            if mode == "vector":
                actual = padded.to(types[target]).view(torch.uint16)[: len(words)].tolist()
            elif mode == "scalar":
                actual = [tensor[i : i + 1].to(types[target]).view(torch.uint16).item() for i in range(len(words))]
            else:
                actual = [
                    torch.scalar_tensor(tensor[i].item(), dtype=types[target]).view(torch.uint16).item()
                    for i in range(len(words))
                ]
            output = verify_outputs(words, actual, "f32", target, policy)
            emitted = conversion_function("f32", target, policy)
            if list(map(int, emitted(words))) != actual:
                raise ValueError("emitted narrowing IR differs at a complete RNE cell or special class")
            form = dict(
                source="f32",
                target=target,
                mode=mode,
                nan_policy=policy,
                rounding="RNE",
                domain="all_destination_cells_and_NaN_bit_classes",
            )
            rows.append(
                dict(
                    form=form,
                    key=digest(form),
                    count=len(words),
                    output_sha256=output,
                    executable_ir_sha256=hashlib.sha256(emitted.source.encode()).hexdigest(),
                )
            )
    unary = []
    for token in ("f16", "bf16"):
        fraction, exponent, _ = FORMATS[token]
        words = np.arange(65536, dtype=np.uint16)
        tensor = torch.from_numpy(words).view(types[token])
        infinity, sign = ((1 << exponent) - 1) << fraction, 1 << (fraction + exponent)
        for name, operation in (("sign_change", torch.neg), ("positive_zero_threshold", torch.relu)):
            for mode in ("vector", "scalar"):
                actual = (
                    operation(tensor).view(torch.uint16).tolist()
                    if mode == "vector"
                    else [operation(tensor[i : i + 1]).view(torch.uint16).item() for i in range(65536)]
                )
                expected = []
                for word in words:
                    word = int(word)
                    magnitude = word & (sign - 1)
                    nan = magnitude > infinity
                    if name == "sign_change":
                        value = word ^ sign
                        if nan and mode == "scalar":
                            value = 0x7FC0 if token == "bf16" else value | 1 << (fraction - 1)
                    else:
                        value = 0 if word & sign and 0 < magnitude <= infinity else word
                        if nan and mode == "vector":
                            value = 0xFFFF if token == "bf16" else word | 1 << (fraction - 1)
                    expected.append(value)
                if actual != expected:
                    raise ValueError("unary scalar/vector full-domain source theorem failed")
                form = dict(family=name, dtype=token, mode=mode, domain="all_storage_words")
                unary.append(
                    dict(
                        form=form,
                        key=digest(form),
                        count=65536,
                        output_sha256=hashlib.sha256(np.array(actual, dtype=np.uint16).tobytes()).hexdigest(),
                    )
                )
    theorem = {
        token: dict(
            precision=p,
            intermediate_precision=24,
            required_precision=2 * p + 2,
            operations=["add", "subtract", "multiply", "divide", "sqrt"],
            scope="single_operation_on_same_storage_operands; preserve_general_opmath_boundaries",
        )
        for token, p in (("f16", 11), ("bf16", 8))
    }
    library = Path(torch.__file__).parent / "lib/libtorch_cpu.so"
    from merlin.targetgen.native_cpu_loop_check import observe as observe_loop
    from merlin.targetgen.native_ieee_arithmetic_check import generate as arithmetic_checks
    from merlin.targetgen.native_integer_check import generate as integer_checks
    from merlin.targetgen.native_layout_check import generate as layout_checks
    from merlin.targetgen.native_layout_check import generate_joint
    from merlin.targetgen.native_nan_priority_check import generate as nan_checks
    from merlin.targetgen.native_widening_check import generate as widening_checks

    result = dict(
        version=VERSION,
        sources=proof_sources(),
        host=dict(
            torch=torch.__version__,
            capability="AVX2",
            library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        ),
        conversions=rows,
        unary=unary,
        arithmetic_checks=arithmetic_checks(),
        nan_priority_checks=nan_checks(),
        layout_checks=layout_checks() + generate_joint(),
        widening_checks=widening_checks(),
        integer_checks=integer_checks(),
        loop_profile=observe_loop(torch),
        double_rounding=theorem,
        finite_arithmetic="IEEE754_RNE_each_typed_boundary",
        NaN_policy="explicit_source_bit_selection; no_hardware_payload_assumption",
    )
    result["content_sha256"] = digest(result)
    return result


def _file_sha256(path):
    status = Path(path).stat()
    return _identified_sha256(path, (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns))


@functools.cache
def _identified_sha256(path, identity):
    """Hash once per file identity: a replaced, resized or rewritten library is hashed again."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def certificate():
    path = Path(__file__).with_suffix(".json")
    value = json.loads(path.read_text())
    content = dict(value)
    recorded = content.pop("content_sha256")
    if value.get("version") != VERSION or digest(content) != recorded or value["sources"] != proof_sources():
        raise ValueError("native IEEE certificate identity changed; regenerate its full-domain checks")
    if _file_sha256(value["host"]["library"]) != value["host"]["library_sha256"]:
        raise ValueError("reference CPU library changed after its full-domain proof")
    for filename, sha256 in value["loop_profile"]["source_files"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha256:
            raise ValueError("CPU loop/vector header changed after its proof")
    for row in value["conversions"]:
        if row["key"] != digest(row["form"]) or row["count"] < 65536:
            raise ValueError("incomplete or changed native conversion certificate")
    if (
        len(value["conversions"]) != 13
        or len(value.get("unary", [])) != 8
        or len(value.get("arithmetic_checks", [])) != 12
    ):
        raise ValueError("missing CPU scalar/vector form certificate")
    if (
        len(value.get("nan_priority_checks", [])) != 4
        or len(value.get("widening_checks", [])) != 2
        or len(value.get("layout_checks", [])) != 13
    ):
        raise ValueError("missing exhaustive NaN priority or widening certificate")
    for row in value["nan_priority_checks"] + value["widening_checks"] + value["layout_checks"]:
        if row["key"] != digest(row["form"]) or row["status"] != "proven":
            raise ValueError("changed CPU special-form certificate")
    if any(row["count"] != 2 * 2046 * 65536 for row in value["nan_priority_checks"]):
        raise ValueError("incomplete reduced NaN operand domain")
    if any(row["count"] != 33556972 for row in value["widening_checks"]):
        raise ValueError("incomplete binary32 widening special/subnormal domain")
    for token, row in value["double_rounding"].items():
        precision = FORMATS[token][0] + 1
        if (
            row["precision"] != precision
            or row["required_precision"] != 2 * precision + 2
            or row["intermediate_precision"] < 2 * precision + 2
        ):
            raise ValueError("unproved double rounding precision")
    return value


if __name__ == "__main__":
    import sys

    result = generate()
    path = Path(sys.argv[sys.argv.index("--write") + 1])
    compact = functools.partial(json.dumps, sort_keys=True, separators=(",", ":"))
    rows = {
        key: "[\n" + ",\n".join(map(compact, value)) + "\n]" if isinstance(value, list) else compact(value)
        for key, value in result.items()
    }
    path.write_text("{\n" + ",\n".join(f"{compact(key)}:{rows[key]}" for key in sorted(rows)) + "\n}\n")
    print(result["content_sha256"], len(result["conversions"]), "full-domain conversion forms")
