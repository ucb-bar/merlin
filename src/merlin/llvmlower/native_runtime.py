"""Compile the existing numeric C runtime with its lowered model's compiler ABI.

A caller explicitly supplies the environment's header driver and model flags.
Preprocessor and compiler dependency lists bind the actual C/header bytes; this
is not a libc/link-supplier or hermetic toolchain attestation. No math source,
numerical policy, target ISA or runtime allocator is selected by target name.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import proc
from merlin.common.digest import sha256_file


class NativeRuntimeError(RuntimeError):
    pass


def compiler_command(model_compiler: Path, header_driver: Path, flags, *, target_flags=(), runner=None) -> list[str]:
    """Use the model compiler's scalar ABI and the selected environment headers.

    ``runner`` preserves the existing backend's process/timeout/error seam.
    Merely selecting the same ISA/LP64 ABI with a different C compiler does not
    establish the BF16 helper register convention.
    """
    selected = runner or (lambda argv: proc.run_checked(argv, error=NativeRuntimeError, timeout=120))
    response = selected([header_driver, "-print-sysroot"])
    text = response.stdout.strip()
    root = Path(text)
    if not text or len(text.splitlines()) != 1 or not root.is_absolute() or not (root / "include/math.h").is_file():
        raise NativeRuntimeError("GCC must report an absolute sysroot containing include/math.h")
    return [
        str(model_compiler),
        *target_flags,
        f"--sysroot={root.resolve()}",
        "-isystem",
        str((root / "include").resolve()),
        *flags,
    ]


def dependency_paths(text: str, cwd: Path) -> tuple[Path, ...]:
    """Read one compiler-generated fixed-target Make dependency rule.

    Only escaped characters, doubled literal dollars and whitespace separation
    are interpreted. No variable expansion, comments, shell or opaque callbacks
    execute. Unsupported/malformed dependency syntax refuses.
    """
    marker = "merlin_runtime:"
    if not isinstance(text, str) or not text.startswith(marker):
        raise NativeRuntimeError("runtime dependency rule has no exact fixed target")
    words, word = [], []
    index = len(marker)
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 1
            if index == len(text):
                raise NativeRuntimeError("unfinished runtime dependency escape")
            if text[index] != "\n":
                word.append(text[index])
        elif char == "$":
            index += 1
            if index == len(text) or text[index] != "$":
                raise NativeRuntimeError("runtime dependency variable expansion is unsupported")
            word.append("$")
        elif char == "#":
            raise NativeRuntimeError("runtime dependency comments are unsupported")
        elif char.isspace():
            if char == "\n" and text[index + 1 :].strip():
                raise NativeRuntimeError("multiple runtime dependency rules are unsupported")
            if word:
                words.append("".join(word))
                word = []
        else:
            word.append(char)
        index += 1
    if word:
        words.append("".join(word))
    if not words:
        raise NativeRuntimeError("runtime dependency rule has no inputs")
    paths = []
    for word in words:
        value = Path(word)
        path = (value if value.is_absolute() else cwd / value).resolve(strict=True)
        if not path.is_file():
            raise NativeRuntimeError("runtime dependency is not a regular input file")
        paths.append(path)
    return tuple(sorted(set(paths)))


def build_numeric_runtime(
    work: Path, *, model_compiler: Path, header_driver: Path, model_flags: tuple[str, ...]
) -> dict:
    """Produce a real same-compiler runtime object and its exact source closure.

    The caller links this object plus its actual platform/libm suppliers. Runtime
    entry execution, platform code/stacks, library resolution and target memory
    admission remain separate. A fresh directory preserves failed attempts.
    """
    from merlin.common import invocation_record
    from merlin.common.paths import runtime_dir

    if (
        not isinstance(model_flags, tuple)
        or not model_flags
        or any(not isinstance(f, str) or not f or "\0" in f for f in model_flags)
        or any(
            f in {"-o", "-c", "-S", "-E", "-M", "-MM", "-MD", "-MMD", "-MF", "-MT", "-MQ"}
            or f.startswith(("@", "-o", "-MF", "-MT", "-MQ", "-MJ", "--output="))
            for f in model_flags
        )
    ):
        raise NativeRuntimeError("explicit model flags without output/dependency overrides required")
    compiler, headers = (Path(p).resolve(strict=True) for p in (model_compiler, header_driver))
    if not compiler.is_file() or not headers.is_file():
        raise NativeRuntimeError("model compiler and header driver must be regular files")
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "merlin.numeric_runtime.v1",
        "status": "preparing",
        "source_math_changed": False,
        "model_compiler": str(compiler),
        "header_driver": str(headers),
        "model_flags": list(model_flags),
        "target_link_complete": False,
        "numerical_gate_performed": False,
        "toolchain_transitive_closure_complete": False,
    }
    try:
        source = runtime_dir() / "abi/mlir_runtime.c"
        owners = [
            source,
            source.with_name("portable_math.h"),
            source.with_name("portable_math_tables.h"),
            compiler,
            headers,
            Path(__file__),
        ]
        owner_pins = {str(p.resolve(strict=True)): sha256_file(p) for p in owners}
        prefix = compiler_command(compiler, headers, model_flags)
        report["command_prefix"] = prefix
        preprocessor = proc.run_checked(
            [*prefix, "-M", "-MT", "merlin_runtime", source], error=NativeRuntimeError, timeout=120, cwd=work
        )
        (work / "dependencies.before.d").write_text(preprocessor.stdout)
        before = dependency_paths(preprocessor.stdout, work)
        if source.resolve() not in before:
            raise NativeRuntimeError("runtime preprocessor omitted its source")
        input_pins = {str(p): sha256_file(p) for p in before} | owner_pins
        obj, depfile = work / "numeric_runtime.o", work / "dependencies.after.d"
        command = [
            *prefix,
            "-ffunction-sections",
            "-fdata-sections",
            "-MD",
            "-MF",
            str(depfile),
            "-MT",
            "merlin_runtime",
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        with invocation_record.observe(
            work,
            stage="object",
            argv=command,
            cwd=work,
            inputs=before,
            outputs=(obj, depfile),
            dependencies=(compiler, headers, Path(__file__)),
        ) as invocation:
            result = proc.run_checked(command, error=NativeRuntimeError, timeout=120, cwd=work)
            invocation.complete(result)
        if dependency_paths(depfile.read_text(), work) != before:
            raise NativeRuntimeError("runtime header dependency set changed during compilation")
        if any(not Path(p).is_file() or sha256_file(p) != digest for p, digest in input_pins.items()):
            raise NativeRuntimeError("runtime source/header/compiler bytes changed during compilation")
        report.update(
            status="built",
            object=str(obj),
            object_sha256=sha256_file(obj),
            input_sha256=input_pins,
            dependencies=[str(p) for p in before],
            command=command,
            dependency_semantics=(
                "actual preprocessor and compiler header sets/bytes agree; no library or execution attestation"
            ),
        )
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (work / "numeric-runtime.json").write_text(json.dumps(report, indent=2) + "\n")
