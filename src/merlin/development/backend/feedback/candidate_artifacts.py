"""Read and decode selected candidate build products and explicit object recipes."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

OWNED = ("submission", "selfcheck_out", "artifacts", "variants")


LOWERED, COMMANDS, ELF = "lowered.llvm.mlir", "command_buffer.json", "package_kernel.elf"


OBJECT, OBJECTS_AMBIGUOUS = "relocatable_object", "relocatable_objects_ambiguous"


_COMPILE_TIMEOUT_S = 300


_MAX_CASES = 128


def owned_path(workspace, value):
    """Resolve one workspace-relative owned path; refuse anything else before reading bytes."""
    if not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value:
        raise ValueError("verify_requires_owned_path")
    relative = PurePosixPath(value.rstrip("/"))
    parts = relative.parts
    if (
        relative.is_absolute()
        or not parts
        or any(part in (".", "..") for part in parts)
        or parts[0] not in OWNED
        or (parts[0] == "variants" and len(parts) < 2)
    ):
        raise ValueError("verify_requires_owned_path")
    workspace = Path(workspace).resolve()
    path = workspace.joinpath(*parts)
    chain = [path, *path.parents]
    if any(p.is_symlink() for p in chain[: len(parts)]) or not path.resolve().is_relative_to(workspace):
        raise ValueError("verify_requires_owned_path")
    if not path.exists():
        raise ValueError("verify_input_missing")
    return path, relative.as_posix()


def _read(path, label):
    from merlin.development.backend.feedback import reference_inputs as A

    raw, pin = A._read(path, label)
    return raw, pin


def case_dirs(workspace, value):
    """``{capsule: (path, label)}`` for one case directory or a directory of case directories."""
    path, label = owned_path(workspace, value)
    if path.is_file():
        return {path.parent.name: (path, label)}
    if (path / LOWERED).is_file() or (path / COMMANDS).is_file():
        return {path.name: (path, label)}
    cases = {}
    for child in sorted(path.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        if (child / LOWERED).is_file() or (child / COMMANDS).is_file():
            cases[child.name] = (child, label + "/" + child.name)
        if len(cases) > _MAX_CASES:
            raise ValueError("verify_case_bound_exceeded")
    if not cases:
        raise ValueError("verify_no_selfcheck_case_artifacts")
    return cases


def _files(path, label):
    """The build's named artifacts, plus the author's own relocatable object when exactly one is present."""
    if path.is_file():
        name = (
            LOWERED
            if path.suffix == ".mlir"
            else COMMANDS
            if path.suffix == ".json"
            else OBJECT
            if path.suffix == ".o"
            else ELF
        )
        return {name: (path, label)}
    found = {
        name: (path / name, label + "/" + name)
        for name in (LOWERED, COMMANDS, ELF)
        if (path / name).is_file() and not (path / name).is_symlink()
    }
    objects = sorted(
        child.name for child in path.iterdir() if child.suffix == ".o" and child.is_file() and not child.is_symlink()
    )
    if len(objects) == 1:
        found[OBJECT] = (path / objects[0], label + "/" + objects[0])
    elif objects:
        found[OBJECTS_AMBIGUOUS] = objects[:8]
    return found


def decode_lowered(text, decoder):
    """The host decoder's trace of emitted LLVM; ``None`` when no selected decoder exists."""
    if not decoder or decoder.get("semantics") is None:
        return None
    from merlin.targetgen.rocc.decode import _parse_module, decode_module

    module = _parse_module(text)
    if module is None:
        raise ValueError("verify_emitted_ir_unparseable")
    return decode_module(module, target=decoder["target"], semantics=decoder["semantics"])


def object_recipe(document, target=None):
    """Entry symbol and ``-march``/``-mabi`` of the selected contract's kernel build recipe.

    The entry is the contract's declared harness entry; the flags come from the toolchain block that
    declares the kernel stack frame (the grader's object recipe). Missing values stay ``None``."""

    def blocks(node):
        if isinstance(node, dict):
            if isinstance(node.get("kernel_stack_frame"), dict):
                yield node
            for value in node.values():
                yield from blocks(value)
        elif isinstance(node, list):
            for value in node:
                yield from blocks(value)

    document = document or {}
    block = next(blocks(document), {})
    flags = [str(f) for f in block.get("cflags") or () if str(f).startswith(("-march=", "-mabi=", "-mcmodel="))]
    entry = (document.get("harness_abi") or {}).get("entry_symbol") or (block.get("kernel_stack_frame") or {}).get(
        "entry_symbol"
    )
    return dict(
        entry_symbol=entry if isinstance(entry, str) and entry else None,
        flags=flags,
        target=target,
        basis="selected contract harness_abi.entry_symbol and kernel toolchain cflags",
    )


def _compiled_stream(lowered, decoder, symbol, build, *, _elf_stream):
    """Compile the build's own emitted LLVM dialect host-side (-mno-relax) and step ``symbol`` in it."""
    import subprocess
    import tempfile

    from merlin.kernels.mechanism_applicability import NO_STREAM

    source = dict(kind="host_compiled_lowered_mlir", symbol=symbol, relaxation="-mno-relax")
    flags = list((build or {}).get("flags") or ())
    if not any((f.startswith("-march=") for f in flags)) or not any((f.startswith("-mabi=") for f in flags)):
        return (None, NO_STREAM + ": the selected contract declares no kernel -march/-mabi recipe", source)
    try:
        from merlin.llvmlower import codegen, toolchain

        clang = Path(toolchain.clang())
        translate = Path(toolchain.mlir_translate())
        if not translate.is_file():
            translate = clang.with_name("mlir-translate")
        selected = tuple((f.partition("=")[0] + "=" for f in flags))
        base = [f for f in codegen.RISCV_FLAGS if not f.startswith(selected)]
        with tempfile.TemporaryDirectory(prefix="cca-verify-build-") as work:
            work = Path(work)
            (work / "kernel.llvm.mlir").write_bytes(lowered)
            argv = [
                [str(translate), "--mlir-to-llvmir", "kernel.llvm.mlir", "-o", "kernel.ll"],
                [str(clang), *base, *flags, "-mno-relax", "-c", "kernel.ll", "-o", "kernel.o"],
            ]
            for command in argv:
                done = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=_COMPILE_TIMEOUT_S)
                if done.returncode:
                    return (
                        None,
                        f"{NO_STREAM}: host compile of lowered.llvm.mlir failed ({Path(command[0]).name} rc {done.returncode}): {done.stderr[-300:]}",
                        dict(source, argv=command),
                    )
            source.update(
                argv=[[Path(c[0]).name, *c[1:]] for c in argv],
                object_sha256=hashlib.sha256((work / "kernel.o").read_bytes()).hexdigest(),
            )
            stream, reason = _elf_stream((work / "kernel.o", "host_compiled/kernel.o"), decoder, symbol)
            return (stream, reason, source)
    except (OSError, subprocess.SubprocessError, ImportError) as error:
        return (None, f"{NO_STREAM}: host compile unavailable: {type(error).__name__}: {str(error)[:200]}", source)


def _elf_stream(entry, decoder, symbol):
    """Decoded commands of a proved bounded execution of ``symbol`` in an ELF or relocatable object."""
    from merlin.kernels.mechanism_applicability import NO_STREAM, command_stream

    if entry is None or not symbol:
        return None, NO_STREAM + (
            ": no package_kernel.elf in this build" if entry is None else ": no kernel symbol named (candidate.symbol)"
        )
    from merlin.analysis.artifact_reader import PinnedELF
    from merlin.analysis.disassembly import tokenize_text
    from merlin.development.backend.feedback import reference_inputs as A

    raw, _ = A._read(*entry, allow_elf=True)
    if not isinstance(raw, PinnedELF):
        return None, NO_STREAM + f": {Path(str(entry[1])).name} is not an ELF"
    try:
        if decoder.get("host_isa") is not None:
            raw.declare_isa(decoder["host_isa"])
        rows = tuple(tokenize_text(raw.function_disassembly(symbol)["text"]))
        return command_stream(rows, decoder)
    except (ArithmeticError, LookupError, TypeError, ValueError) as error:
        return None, f"{NO_STREAM}: {type(error).__name__}: {str(error)[:200]}"
    finally:
        raw.close()


# Preserve established callable pickle identities through the explicit public exports.
for _export in (
    owned_path,
    _read,
    case_dirs,
    _files,
    decode_lowered,
    object_recipe,
    _elf_stream,
):
    _export.__module__ = "merlin.development.backend.feedback.candidate_verification"
