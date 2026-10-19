"""Explicit source-bound host/provider objects for the ordinary model link.

Device catalogs retain their own closed-object contract. This separate seam
checks imported compilation identities and typed function boundaries, records
ordered linking, and closes every required symbol in the final image. It grants
no numerical, effect, target capability or source-fallback permission.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

from merlin.common.digest import sha256_file


@dataclass(frozen=True)
class HostFunctionABI:
    symbol: str
    result: str
    arguments: tuple[str, ...]
    variadic: bool = False


@dataclass(frozen=True)
class HostProviderObject:
    path: Path
    sha256: str
    llvm_path: Path
    llvm_sha256: str
    defined: tuple[HostFunctionABI, ...]
    required: tuple[HostFunctionABI, ...]
    compilation_pins: tuple[tuple[Path, str], ...]
    compile_commands: tuple[tuple[str, ...], ...]
    compile_cwd: Path


@dataclass(frozen=True)
class HostProviderBuild:
    prepared_sha256: str
    model_llvm_sha256: str
    model_object_sha256: str
    objects: tuple[HostProviderObject, ...]
    numerical_witness_sha256: str
    effect_witness_sha256: str


@dataclass(frozen=True)
class HostProviderContext:
    prepared_path: Path
    model_llvm_path: Path
    model_object_path: Path
    workdir: Path
    compiler: Path
    compiler_flags: tuple[str, ...]
    compile: Callable


def _identity(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("provider artifacts require regular files")
    return dict(path=str(path.resolve()), sha256=sha256_file(path))


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _function_abis(source):
    """Read canonical LLVM function headers using typed tokens, never names.

    Accept only the simple C scalar/pointer ABI used by ranked descriptor and
    primitive calls. Aggregate, byval/sret, address-space, non-C conventions and
    unknown ABI attributes refuse rather than guessing a function type.
    """
    from merlin.llvmlower.late_quant_rne import _identity as name
    from merlin.llvmlower.late_quant_rne import _tokens

    scalar = {"void", "ptr", "float", "double", "i1", "i8", "i16", "i32", "i64"}
    abi = {}
    for line in source.splitlines():
        line = line.strip()
        if not line.startswith(("define ", "declare ")):
            continue
        words = [t.text for t in _tokens(line)]
        at = next((i for i, word in enumerate(words) if word.startswith("@")), None)
        if at is None or at < 2 or words[at + 1] != "(":
            raise ValueError("unsupported provider LLVM function header")
        # Other functions are irrelevant unless imported as an explicit boundary.
        # Retain a refusal sentinel for unsupported headers; lookup then refuses.
        symbol = name(words[at])
        result = words[at - 1]
        prefix = words[1 : at - 1]
        # A returned integer range constrains values, not the machine ABI.
        # Consume only one well-formed, same-type range; unknown attributes
        # still refuse below, including calling-convention/aggregate changes.
        if "range" in prefix:
            index = prefix.index("range")
            attribute = prefix[index : index + 7]
            valid_range = (
                prefix.count("range") == 1
                and len(attribute) == 7
                and attribute[1] == "("
                and attribute[2] == result
                and result in scalar - {"void", "ptr", "float", "double"}
                and attribute[4] == ","
                and attribute[6] == ")"
            )
            if valid_range:
                try:
                    bounds = (int(attribute[3]), int(attribute[5]))
                    width = int(result[1:])
                    valid_range = all(-(1 << (width - 1)) <= value < (1 << width) for value in bounds)
                except ValueError:
                    valid_range = False
            if not valid_range:
                abi[symbol] = None
                continue
            prefix = prefix[:index] + prefix[index + 7 :]
        allowed_prefix = {
            "ccc",
            "dso_local",
            "dso_preemptable",
            "internal",
            "private",
            "external",
            "hidden",
            "protected",
            "default",
            "dllimport",
            "dllexport",
            "linkonce",
            "linkonce_odr",
            "weak",
            "weak_odr",
            "available_externally",
            "noundef",
            "signext",
            "zeroext",
            "nonnull",
            "noalias",
        }
        if result not in scalar or any(word not in allowed_prefix for word in prefix):
            abi[symbol] = None
            continue
        extension = [word for word in prefix if word in ("signext", "zeroext")]
        if len(extension) > 1:
            abi[symbol] = None
            continue
        result = " ".join([*extension, result])
        segments, segment, depth = [], [], 0
        for word in words[at + 2 :]:
            if word == ")" and depth == 0:
                if segment:
                    segments.append(segment)
                break
            if word == "," and depth == 0:
                segments.append(segment)
                segment = []
                continue
            if word in ("(", "[", "{", "<"):
                depth += 1
            if word in (")", "]", "}", ">"):
                depth -= 1
            segment.append(word)
        else:
            raise ValueError("multiline or unterminated provider function ABI")
        arguments, variadic, valid = [], False, True
        for index, parameter in enumerate(segments):
            if parameter == ["..."] and index == len(segments) - 1:
                variadic = True
                continue
            if (
                not parameter
                or parameter[0] not in scalar - {"void"}
                or any(
                    word in parameter
                    for word in (
                        "byval",
                        "byref",
                        "sret",
                        "inalloca",
                        "preallocated",
                        "inreg",
                        "nest",
                        "swiftself",
                        "swifterror",
                        "addrspace",
                    )
                )
            ):
                valid = False
                break
            extension = [word for word in parameter if word in ("signext", "zeroext")]
            if len(extension) > 1:
                valid = False
                break
            arguments.append(" ".join([*extension, parameter[0]]))
        value = HostFunctionABI(symbol, result, tuple(arguments), variadic) if valid else None
        if symbol in abi:
            raise ValueError("duplicate provider LLVM function boundary")
        abi[symbol] = value
    return abi


def read_host_function_abi(path, symbol):
    """Read one explicitly named boundary from immutable companion LLVM."""
    value = _function_abis(Path(path).read_text()).get(symbol)
    if value is None:
        raise ValueError("provider function ABI is absent or unsupported")
    return value


def _symbols(path, inspector):
    if inspector is None:
        raise ValueError("provider symbol inspector is unavailable")
    result = subprocess.run(
        [str(inspector), "--format=posix", "--extern-only", str(path)], capture_output=True, text=True, check=True
    )
    defined, required = {}, set()
    for line in result.stdout.splitlines():
        words = line.split()
        if len(words) < 2 or len(words[1]) != 1:
            raise ValueError("unreadable provider object symbols")
        symbol, kind = words[:2]
        if kind in ("U", "w", "v"):
            required.add(symbol)
        else:
            defined.setdefault(symbol, []).append(kind)
    return defined, required


def prepare_host_provider(builder, context, *, inspector):
    """Invoke an explicit builder and validate every imported boundary and pin."""
    if builder is None:
        return (), None
    if type(builder) is MethodType:
        from merlin.llvmlower.shared_pointwise_routing import resolve_pointwise_provider

        builder = resolve_pointwise_provider(builder, context)
        if builder is None:
            return (), None
    if not callable(builder):
        raise ValueError("host_provider_builder must be explicitly callable")
    if inspector is None:
        raise ValueError("provider symbol inspector is unavailable")
    context.workdir.mkdir(parents=True, exist_ok=True)
    receipt_path = context.workdir / "host_provider.json"
    receipt_path.unlink(missing_ok=True)
    originals = [
        _identity(path) for path in (context.prepared_path, context.model_llvm_path, context.model_object_path)
    ]
    build = builder(context)
    if (
        not isinstance(build, HostProviderBuild)
        or not build.objects
        or any(not _pin(pin) for pin in (build.numerical_witness_sha256, build.effect_witness_sha256))
        or [build.prepared_sha256, build.model_llvm_sha256, build.model_object_sha256]
        != [identity["sha256"] for identity in originals]
        or originals
        != [_identity(path) for path in (context.prepared_path, context.model_llvm_path, context.model_object_path)]
    ):
        raise ValueError("provider lost exact source/model or explicit numerical/effect identities")
    model_abis = _function_abis(context.model_llvm_path.read_text())
    model_defined, _ = _symbols(context.model_object_path, inspector)
    all_defined, all_required, paths, pins, records = {}, {}, [], {}, []
    for artifact in build.objects:
        if not isinstance(artifact, HostProviderObject):
            raise ValueError("explicit pinned provider objects required")
        object_id, llvm_id = _identity(artifact.path), _identity(artifact.llvm_path)
        if (
            object_id["sha256"] != artifact.sha256
            or llvm_id["sha256"] != artifact.llvm_sha256
            or object_id["path"] in paths
            or not artifact.compilation_pins
        ):
            raise ValueError("provider object/LLVM/imported compilation identity changed")
        paths.append(object_id["path"])
        for path, pin in (
            *artifact.compilation_pins,
            (artifact.path, artifact.sha256),
            (artifact.llvm_path, artifact.llvm_sha256),
        ):
            identity = _identity(path)
            if identity["sha256"] != pin or not _pin(pin):
                raise ValueError("provider compilation dependency changed")
            if identity["path"] in pins and pins[identity["path"]] != pin:
                raise ValueError("ambiguous provider compilation dependency")
            pins[identity["path"]] = pin
        # Imported recipes are explicit evidence, not inferred from an object.
        # Pin both emissions and the actual compiler, while retaining full argv.
        if not artifact.compile_commands or not Path(artifact.compile_cwd).is_dir():
            raise ValueError("provider requires explicit compilation commands and working directory")
        outputs = set()
        for command in artifact.compile_commands:
            executable = shutil.which(str(command[0])) if command else None
            if executable is None:
                raise ValueError("provider compilation command has no executable")
            compiler = _identity(Path(executable).resolve())
            if pins.get(compiler["path"]) != compiler["sha256"]:
                raise ValueError("provider compilation executable is not pinned")
            if "-o" not in command or command.count("-o") != 1:
                raise ValueError("provider compilation command requires one explicit output")
            output_index = command.index("-o") + 1
            if output_index == len(command):
                raise ValueError("provider compilation output is absent")
            output = Path(command[output_index])
            outputs.add(str((Path(artifact.compile_cwd) / output).resolve()))
        if not {object_id["path"], llvm_id["path"]}.issubset(outputs):
            raise ValueError("provider object and companion LLVM emissions are not both declared")
        if any(not isinstance(abi, HostFunctionABI) for abi in (*artifact.defined, *artifact.required)):
            raise ValueError("provider requires explicit typed function boundaries")
        local = _function_abis(artifact.llvm_path.read_text())
        defined, required = _symbols(artifact.path, inspector)
        expected_def = {abi.symbol: abi for abi in artifact.defined}
        expected_req = {abi.symbol: abi for abi in artifact.required}
        if (
            len(expected_def) != len(artifact.defined)
            or len(expected_req) != len(artifact.required)
            or set(defined) != set(expected_def)
            or required != set(expected_req)
            or any(kinds != ["T"] for kinds in defined.values())
            or set(expected_def).intersection(model_defined)
        ):
            raise ValueError("provider has missing, extra, weak, data or ambiguous object symbols")
        for kind, contracts, combined in [
            ("defined", expected_def, all_defined),
            ("required", expected_req, all_required),
        ]:
            for symbol, abi in contracts.items():
                if (
                    local.get(symbol) != abi
                    or symbol in model_abis
                    and model_abis[symbol] != abi
                    or symbol in combined
                    and (kind == "defined" or combined[symbol] != abi)
                ):
                    raise ValueError("provider typed function ABI differs or is ambiguous")
                combined[symbol] = abi
        records.append(
            dict(
                object=object_id,
                llvm=llvm_id,
                defined=[asdict(abi) for abi in artifact.defined],
                required=[asdict(abi) for abi in artifact.required],
                compile_commands=[list(map(str, cmd)) for cmd in artifact.compile_commands],
                compile_cwd=str(Path(artifact.compile_cwd).resolve()),
            )
        )
    if any(symbol in all_defined and all_defined[symbol] != abi for symbol, abi in all_required.items()):
        raise ValueError("provider-to-provider typed function ABI differs")
    record = dict(
        schema="source_bound_host_provider_v1",
        status="prepared",
        source_model=originals,
        objects=records,
        compilation_pins=pins,
        numerical_witness_sha256=build.numerical_witness_sha256,
        effect_witness_sha256=build.effect_witness_sha256,
        required_symbols=sorted(all_required),
        scope="Imported compilation/typed ABI evidence; no numerical/effect/target qualification inferred",
        receipt_path=str(receipt_path.resolve()),
    )
    receipt_path.write_text(json.dumps(record, indent=2) + "\n")
    return tuple(Path(path) for path in paths), record


def close_host_provider(record, objects, executable, *, inspector, link_flags=()):
    """Recheck immutable imports and final ordered object/symbol resolution."""
    if record is None:
        return
    for identity in record["source_model"]:
        if _identity(identity["path"]) != identity:
            raise ValueError("provider source/model changed before final link closure")
    for path, pin in record["compilation_pins"].items():
        if _identity(path)["sha256"] != pin:
            raise ValueError("provider import changed before final link closure")
    ordered = [_identity(path) for path in objects]
    declared = [row["object"] for row in record["objects"]]
    if any(ordered.count(identity) != 1 for identity in declared):
        raise ValueError("provider object missing or duplicated in ordered final link")
    wrappers = {}
    for flag in link_flags:
        prefix = "-Wl,--wrap="
        if str(flag).startswith(prefix):
            symbol = str(flag)[len(prefix) :]
            wrappers[symbol] = "__wrap_" + symbol
    defined, unresolved = _symbols(executable, inspector)
    resolved = {symbol: wrappers.get(symbol, symbol) for symbol in record["required_symbols"]}
    exported = [abi["symbol"] for row in record["objects"] for abi in row["defined"]]
    if unresolved or any(len(defined.get(symbol, ())) != 1 for symbol in [*exported, *resolved.values()]):
        raise ValueError("final provider symbol resolution is incomplete or ambiguous")
    record.update(
        status="completed",
        ordered_link_objects=ordered,
        actual_link_flags=list(map(str, link_flags)),
        resolved_required_symbols=resolved,
        executable=_identity(executable),
    )
    Path(record["receipt_path"]).write_text(json.dumps(record, indent=2) + "\n")
