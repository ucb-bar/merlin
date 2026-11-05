"""Build and load the explicitly selected compiler-only MPFR power helper."""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

POLICY = "positive-power-correct-rne-mpfr-v1"
HELPER_ENV = "MERLIN_CONSTANT_MATH_HELPER"


class MathUnavailable(RuntimeError):
    status = "tool_unavailable"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_policy(policy):
    keys = {"schema", "policy", "helper_sha256", "dependencies", "max_precision_bits", "element_budget", "max_rounds"}
    if (
        not isinstance(policy, dict)
        or set(policy) != keys
        or policy["schema"] != "merlin.constant_math_precision.v1"
        or policy["policy"] != POLICY
        or not isinstance(policy["helper_sha256"], str)
        or len(policy["helper_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in policy["helper_sha256"])
        or type(policy["max_precision_bits"]) is not int
        or policy["max_precision_bits"] not in {64, 128, 256, 512, 1024, 2048, 4096}
        or type(policy["element_budget"]) is not int
        or not 1 <= policy["element_budget"] <= 65536
        or type(policy["max_rounds"]) is not int
        or not 1 <= policy["max_rounds"] <= 32
        or not isinstance(policy["dependencies"], list)
        or len(policy["dependencies"]) != 2
    ):
        raise ValueError("constant math precision policy has unknown or invalid fields")
    for name, row in zip(("mpfr", "gmp"), policy["dependencies"], strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "version", "sha256"}
            or row["name"] != name
            or not isinstance(row["version"], str)
            or not row["version"]
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])
        ):
            raise ValueError("constant math dependency pin is invalid")
    return json.loads(json.dumps(policy))


class MathHelper:
    """A fixed typed ABI, never an expression evaluator or a model oracle."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        digest = _sha(self.path)
        if not self.path.name.endswith(digest + ".so"):
            raise ValueError("constant math helper needs its immutable content-addressed filename")
        lib = ctypes.CDLL(str(self.path))
        lib.merlin_math_abi.restype = ctypes.c_uint
        if lib.merlin_math_abi() != 1:
            raise ValueError("constant math helper ABI differs")
        lib.merlin_math_tls.restype = ctypes.c_uint
        if lib.merlin_math_tls() != 1:
            raise MathUnavailable("constant math helper requires thread-local MPFR state")
        for name in ("merlin_math_dependency", "merlin_math_version"):
            function = getattr(lib, name)
            function.argtypes = [ctypes.c_uint]
            function.restype = ctypes.c_char_p
        dependencies = []
        for index, name in enumerate(("mpfr", "gmp")):
            raw = lib.merlin_math_dependency(index)
            if not raw:
                raise MathUnavailable("cannot identify the loaded constant math dependency")
            dependencies.append(
                {
                    "name": name,
                    "version": lib.merlin_math_version(index).decode(),
                    "sha256": _sha(Path(os.fsdecode(raw)).resolve()),
                }
            )
        if dependencies[0]["version"] != lib.merlin_math_version(2).decode():
            raise ValueError("constant math MPFR header and loaded library differ")
        lib.merlin_math_pow.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.merlin_math_pow.restype = ctypes.c_int
        self.lib = lib
        self.identity = {"helper_sha256": digest, "dependencies": dependencies}
        self._cache = {}

    @classmethod
    def selected(cls, policy):
        policy = validate_policy(policy)
        path = os.environ.get(HELPER_ENV)
        if not path:
            raise MathUnavailable(f"select a compiler helper through {HELPER_ENV}")
        try:
            digest = _sha(path)
        except OSError as error:
            raise MathUnavailable("selected constant math helper is unavailable") from error
        if digest != policy["helper_sha256"]:
            raise ValueError("constant math helper identity changed")
        helper = cls(path)
        if helper.identity != {key: policy[key] for key in ("helper_sha256", "dependencies")}:
            raise ValueError("constant math loaded dependency identity changed")
        return helper

    def policy(self, *, max_precision_bits=1024, element_budget=4096, max_rounds=8):
        return validate_policy(
            {
                "schema": "merlin.constant_math_precision.v1",
                "policy": POLICY,
                **self.identity,
                "max_precision_bits": max_precision_bits,
                "element_budget": element_budget,
                "max_rounds": max_rounds,
            }
        )

    def power(self, width, base, exponent, maximum):
        if (
            type(width) is not int
            or width not in (32, 64)
            or any(type(x) is not int or not 0 <= x < 1 << width for x in (base, exponent))
            or type(maximum) is not int
            or maximum not in {64, 128, 256, 512, 1024, 2048, 4096}
        ):
            raise ValueError("constant power needs typed source bits and a bounded precision")
        key = width, base, exponent, maximum
        if key in self._cache:
            return self._cache[key]
        precision = 64
        while precision <= maximum:
            lower, upper = ctypes.c_uint64(), ctypes.c_uint64()
            status = self.lib.merlin_math_pow(
                width, base, exponent, precision, ctypes.byref(lower), ctypes.byref(upper)
            )
            if status == 0:
                if lower.value != upper.value or lower.value >= 1 << width:
                    raise RuntimeError("constant math helper returned inconsistent bounds")
                result = lower.value, precision
                self._cache[key] = result
                return result
            if status == -3:
                raise MathUnavailable("constant math MPFR exponent range is insufficient")
            if status < 0:
                raise ValueError(f"constant power input/environment is outside the selected domain ({status})")
            precision *= 2
        raise ArithmeticError("constant power rounding remains inconclusive at the precision limit")

    def unary_reference(self, width, bits, operation, maximum=1024):
        """Independent MPFR bounds for checking a deployed math implementation.

        This does not admit a source rewrite or evaluate runtime inputs during
        compilation. Older power-only helpers remain usable by their policies.
        """
        if (
            type(width) is not int
            or width not in (32, 64)
            or type(bits) is not int
            or not 0 <= bits < 1 << width
            or operation not in ("sin", "cos")
            or type(maximum) is not int
            or maximum not in {64, 128, 256, 512, 1024, 2048, 4096}
        ):
            raise ValueError("unary reference needs finite typed inputs and a bounded precision")
        try:
            function = self.lib.merlin_math_unary
        except AttributeError as error:
            raise MathUnavailable("selected helper has no unary reference entrypoint") from error
        function.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        function.restype = ctypes.c_int
        precision = 64
        while precision <= maximum:
            lower, upper = ctypes.c_uint64(), ctypes.c_uint64()
            status = function(width, bits, int(operation == "cos"), precision, ctypes.byref(lower), ctypes.byref(upper))
            if status == 0:
                if lower.value != upper.value or lower.value >= 1 << width:
                    raise RuntimeError("unary reference returned inconsistent bounds")
                return lower.value, precision
            if status == -3:
                raise MathUnavailable("unary reference MPFR exponent range is insufficient")
            if status < 0:
                raise ValueError("unary reference input is outside its finite domain")
            precision *= 2
        raise ArithmeticError("unary reference rounding remains inconclusive at the precision limit")


def build_helper(destination, *, compiler=None):
    """Build once with provisioned MPFR/GMP development files; never install them."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "constant_math.c"
    source.write_bytes(files(__package__).joinpath("constant_math.c").read_bytes())
    compiler = compiler or shutil.which("cc")
    if not compiler:
        raise MathUnavailable("constant math helper build needs an explicitly provisioned C compiler")
    output = destination / "constant_math.so"
    command = [
        str(compiler),
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
        "-fno-fast-math",
        "-ffp-contract=off",
        str(source),
        "-lmpfr",
        "-lgmp",
        "-ldl",
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    (destination / "build.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise MathUnavailable("constant math helper build failed; inspect its build.log")
    final = output.with_name("constant_math-" + _sha(output) + ".so")
    output.rename(final)
    helper = MathHelper(final)
    manifest = {
        "schema": "merlin.constant_math_build.v1",
        "library": final.name,
        "source_sha256": _sha(source),
        **helper.identity,
        "command": command,
        "compiler_version": subprocess.check_output([str(compiler), "--version"], text=True).strip(),
        "policy": helper.policy(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return helper, manifest


def build_command(args):
    """TargetGen command with explicit failure status and a fresh build root."""
    try:
        helper, manifest = build_helper(args.out, compiler=args.cc)
        report = {"status": "built", "library": str(helper.path), "manifest": manifest}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report = {"status": getattr(error, "status", "compile_error"), "reason": str(error), "out": args.out}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "built" else 2
