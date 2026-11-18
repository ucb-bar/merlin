"""Explicit ordered host LLVM transforms with source and emission witnesses.

The chain enforces selected-stage composition, not the stages' semantic theorems.
Trusted synchronous stage verifiers own those theorems and validate actual emitted
IR. Toolchain/dependency closure and final object/link qualification remain separate.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from string import ascii_letters, digits
from typing import TYPE_CHECKING

from merlin.common.digest import is_sha256, sha256_file, write_pretty_json

# ---- host_transform_chain ----------------------------------------------------------------------

FILENAME = "host_transform_chain.json"


@dataclass(frozen=True)
class HostLLVMArtifact:
    path: Path
    sha256: str
    bytes: int

    @classmethod
    def capture(cls, path: Path) -> HostLLVMArtifact:
        path = Path(path).resolve()
        return cls(path, sha256_file(path), path.stat().st_size)

    def validate(self) -> None:
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or not is_sha256(self.sha256)
            or type(self.bytes) is not int
            or self.bytes < 0
            or HostLLVMArtifact.capture(self.path) != self
        ):
            raise ValueError("host transform artifact identity changed or is invalid")

    def to_dict(self) -> dict:
        return {"path": str(self.path), "sha256": self.sha256, "bytes": self.bytes}


def _artifacts(values: tuple[HostLLVMArtifact, ...], label: str) -> None:
    if type(values) is not tuple or not values or any(type(value) is not HostLLVMArtifact for value in values):
        raise TypeError(f"{label} requires nonempty immutable typed artifact witnesses")
    if len({value.path for value in values}) != len(values):
        raise ValueError(f"{label} has duplicate artifact paths")
    for value in values:
        value.validate()


@dataclass(frozen=True)
class HostLLVMStageContract:
    name: str
    sources: tuple[HostLLVMArtifact, ...]
    semantic_proofs: tuple[HostLLVMArtifact, ...]

    def validate(self) -> None:
        if type(self.name) is not str or not self.name.isascii() or not self.name.isidentifier():
            raise ValueError("host transform stage requires an explicit identifier")
        if self.name == "legacy_terminal":
            raise ValueError("legacy_terminal is reserved for the existing single hook")
        _artifacts(self.sources, "stage sources")
        _artifacts(self.semantic_proofs, "stage semantic proofs")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sources": [value.to_dict() for value in self.sources],
            "semantic_proofs": [value.to_dict() for value in self.semantic_proofs],
        }


@dataclass(frozen=True)
class HostLLVMEmission:
    source: HostLLVMArtifact
    selected: HostLLVMArtifact
    evidence: tuple[HostLLVMArtifact, ...]

    def validate(self, source: Path, selected: Path, workdir: Path) -> None:
        if type(self.source) is not HostLLVMArtifact or type(self.selected) is not HostLLVMArtifact:
            raise TypeError("host emission requires typed input and output witnesses")
        if self.source != HostLLVMArtifact.capture(source) or self.selected != HostLLVMArtifact.capture(selected):
            raise ValueError("host emission does not bind the actual stage input and output")
        _artifacts(self.evidence, "stage emission evidence")
        if any(not value.path.is_relative_to(workdir.resolve()) for value in self.evidence):
            raise ValueError("stage emission evidence must be retained in its stage workdir")


@dataclass(frozen=True)
class HostLLVMTransformStage:
    contract: HostLLVMStageContract
    transform: Callable[[Path, Path], Path]
    verify_emission: Callable[[Path, Path, Path], HostLLVMEmission]

    def validate(self) -> None:
        if type(self.contract) is not HostLLVMStageContract:
            raise TypeError("typed host stage contract required")
        self.contract.validate()
        for callback in (self.transform, self.verify_emission):
            if not callable(callback):
                raise TypeError("host transform and actual emission verifier must be callable")
            try:
                owner = inspect.getsourcefile(inspect.unwrap(callback))
            except TypeError as exc:
                raise ValueError("host stage callback lacks a pinned Python source owner") from exc
            if owner is None or Path(owner).resolve() not in {source.path for source in self.contract.sources}:
                raise ValueError("host stage callback implementation is not among its pinned sources")


@dataclass(frozen=True)
class HostLLVMTransformChain:
    promised_stages: tuple[HostLLVMStageContract, ...]
    stages: tuple[HostLLVMTransformStage, ...]

    def validate(self) -> None:
        if type(self.promised_stages) is not tuple or not self.promised_stages:
            raise TypeError("host chain requires an explicit nonempty immutable promised order")
        if type(self.stages) is not tuple or any(type(stage) is not HostLLVMTransformStage for stage in self.stages):
            raise TypeError("host chain requires immutable typed selected stages")
        if any(type(stage) is not HostLLVMStageContract for stage in self.promised_stages):
            raise TypeError("host chain promises require typed stage contracts")
        if tuple(stage.contract for stage in self.stages) != self.promised_stages:
            raise ValueError("host transform stages do not satisfy the complete promised order")
        if len({stage.name for stage in self.promised_stages}) != len(self.promised_stages):
            raise ValueError("host transform stage names must be unique")
        for stage in self.stages:
            stage.validate()


def _selected(source: Path, selected: Path, workdir: Path) -> Path:
    selected = Path(selected).resolve()
    if selected != source.resolve() and not selected.is_relative_to(workdir.resolve()):
        raise ValueError("host transform output must be retained in its stage workdir")
    if not selected.is_file():
        raise ValueError("host transform returned no LLVM IR file")
    return selected


def apply_host_transform_chain(
    source: Path,
    workdir: Path,
    chain: HostLLVMTransformChain,
    *,
    terminal: Callable[[Path, Path], tuple[Path, dict | None]] | None = None,
) -> tuple[Path, dict]:
    """Execute every promised stage and the separately selected legacy terminal.

    Stage verifiers must inspect actual emissions; static names or an invocation
    receipt do not fulfill a selected semantic obligation. No semantic fact is
    inferred by core from a callback's name or a proof file's contents.
    """
    if type(chain) is not HostLLVMTransformChain:
        raise TypeError("typed host transform chain required")
    source, workdir = Path(source).resolve(), Path(workdir).resolve()
    original = HostLLVMArtifact.capture(source)
    occupied = workdir.exists() and any(workdir.iterdir())
    workdir.mkdir(parents=True, exist_ok=True)
    receipt = workdir / FILENAME
    record = {
        "schema": "merlin.host_transform_chain.v1",
        "status": "prepared",
        "scope": "ordered host LLVM emissions; semantic theorems provider-owned; no object/link certification",
        "source": original.to_dict(),
        "producer": HostLLVMArtifact.capture(Path(__file__)).to_dict(),
        "stages": [],
    }
    write_pretty_json(receipt, record)
    retained = [original]
    try:
        if occupied:
            raise ValueError("host transform chain needs a fresh owned workdir")
        chain.validate()
        record["promised_order"] = [stage.to_dict() for stage in chain.promised_stages]
        write_pretty_json(receipt, record)
        selected = source
        for index, stage in enumerate(chain.stages):
            chain.validate()
            for artifact in retained:
                artifact.validate()
            snapshot = workdir / f"input_{index}.ll"
            snapshot.write_bytes(selected.read_bytes())
            incoming = HostLLVMArtifact.capture(snapshot)
            retained.append(incoming)
            stage_work = workdir / f"stage_{index}_{stage.contract.name}"
            stage_work.mkdir()
            row = {
                "contract": stage.contract.to_dict(),
                "input": incoming.to_dict(),
                "status": "invoked",
                "callbacks": {
                    "transform": stage.transform.__module__ + ":" + stage.transform.__qualname__,
                    "verify_emission": stage.verify_emission.__module__ + ":" + stage.verify_emission.__qualname__,
                },
            }
            record["stages"].append(row)
            write_pretty_json(receipt, record)
            selected = _selected(snapshot, stage.transform(snapshot, stage_work), stage_work)
            incoming.validate()
            outgoing = HostLLVMArtifact.capture(selected)
            emission = stage.verify_emission(snapshot, selected, stage_work)
            if type(emission) is not HostLLVMEmission:
                raise TypeError("host stage verifier must return a typed actual emission witness")
            emission.validate(snapshot, selected, stage_work)
            outgoing.validate()
            incoming.validate()
            retained.extend((outgoing, *emission.evidence))
            row.update(
                status="verified", output=outgoing.to_dict(), evidence=[item.to_dict() for item in emission.evidence]
            )
            write_pretty_json(receipt, record)
        if terminal is not None:
            incoming = HostLLVMArtifact.capture(selected)
            selected, legacy = terminal(selected, workdir / "legacy_terminal")
            selected = _selected(incoming.path, selected, workdir / "legacy_terminal")
            incoming.validate()
            outgoing = HostLLVMArtifact.capture(selected)
            retained.extend((incoming, outgoing))
            record["legacy_terminal"] = {"input": incoming.to_dict(), "output": outgoing.to_dict(), "receipt": legacy}
        chain.validate()
        for artifact in retained:
            artifact.validate()
        record.update(status="completed", selected=HostLLVMArtifact.capture(selected).to_dict())
        write_pretty_json(receipt, record)
        return selected, {
            "source_path": str(source),
            "source_sha256": original.sha256,
            "selected_path": str(selected),
            "selected_sha256": record["selected"]["sha256"],
            "chain": HostLLVMArtifact.capture(receipt).to_dict(),
        }
    except Exception as exc:
        record.update(status="refused", failure={"type": type(exc).__name__, "reason": str(exc)})
        write_pretty_json(receipt, record)
        raise


def recheck_host_transform_chain(receipt: Path, *, expected_chain: HostLLVMTransformChain | None = None) -> dict:
    """Reclose retained source/proof/IR bytes and ordered handoffs before codegen completion."""
    record = json.loads(Path(receipt).read_text())
    if record.get("schema") != "merlin.host_transform_chain.v1" or record.get("status") != "completed":
        raise ValueError("host transform chain has no completed emission")
    if expected_chain is not None:
        if type(expected_chain) is not HostLLVMTransformChain:
            raise TypeError("typed expected host transform chain required")
        expected_chain.validate()
        if record.get("promised_order") != [stage.to_dict() for stage in expected_chain.promised_stages]:
            raise ValueError("host chain receipt differs from the externally selected promised order")

    def artifact(row):
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"} or type(row["path"]) is not str:
            raise ValueError("malformed host transform artifact witness")
        value = HostLLVMArtifact(Path(row["path"]), row["sha256"], row["bytes"])
        value.validate()
        return value

    artifact(record["producer"])
    previous = artifact(record["source"])
    stages, promised = record.get("stages"), record.get("promised_order")
    if not isinstance(stages, list) or not stages or not isinstance(promised, list) or len(stages) != len(promised):
        raise ValueError("host chain does not retain every promised stage")
    names = []
    for expected, stage in zip(promised, stages, strict=True):
        if stage.get("status") != "verified" or stage.get("contract") != expected:
            raise ValueError("host chain stage is unverified or differs from the promised order")
        contract = HostLLVMStageContract(
            expected["name"],
            tuple(artifact(row) for row in expected["sources"]),
            tuple(artifact(row) for row in expected["semantic_proofs"]),
        )
        contract.validate()
        names.append(contract.name)
        incoming, outgoing = artifact(stage["input"]), artifact(stage["output"])
        if (incoming.sha256, incoming.bytes) != (previous.sha256, previous.bytes):
            raise ValueError("host chain stage input does not consume the previous emission")
        if not stage.get("evidence"):
            raise ValueError("host chain stage lacks actual emission evidence")
        for row in stage["evidence"]:
            artifact(row)
        previous = outgoing
    if len(set(names)) != len(names):
        raise ValueError("host chain repeats a stage")
    if "legacy_terminal" in record:
        terminal = record["legacy_terminal"]
        if artifact(terminal["input"]) != previous:
            raise ValueError("legacy terminal did not consume the completed chain")
        previous = artifact(terminal["output"])
    if artifact(record["selected"]) != previous:
        raise ValueError("selected host IR differs from the complete chain emission")
    return record


# ---- host_transform_dispatch -------------------------------------------------------------------
# Independent host LLVM transform dispatch with immutable source ownership.
#
# No target backend, numerical oracle, corpus or measured cost is selected here.
# The selected callback or chain owns its semantic evidence.

if TYPE_CHECKING:
    pass


class HostLLVMTransformError(RuntimeError):
    pass


def _transform_host_ir(
    source: Path,
    workdir: Path,
    transform: Callable[[Path, Path], Path] | None,
    *,
    chain: HostLLVMTransformChain | None = None,
) -> tuple[Path, dict | None]:
    """Keep target-selected late legalization inside the normal object build.

    The source lowering artifact remains immutable. Clang verifies the selected
    LLVM IR, and its compiled object participates in the existing build hash.
    The optional callback owns its semantic proof and supporting artifacts.
    """
    if chain is not None:
        return apply_host_transform_chain(
            source,
            workdir,
            chain,
            terminal=(lambda current, work: _transform_host_ir(current, work, transform))
            if transform is not None
            else None,
        )
    if transform is None:
        return source, None
    import hashlib

    before = source.read_bytes()
    workdir.mkdir(parents=True, exist_ok=True)
    selected = Path(transform(source, workdir)).resolve()
    if source.read_bytes() != before:
        raise HostLLVMTransformError("host LLVM transform modified its source lowering artifact")
    if selected != source.resolve() and not selected.is_relative_to(workdir.resolve()):
        raise HostLLVMTransformError("host LLVM transform output must be retained in its workdir")
    if not selected.is_file():
        raise HostLLVMTransformError("host LLVM transform returned no LLVM IR file")
    return selected, {
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(before).hexdigest(),
        "selected_path": str(selected),
        "selected_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }


# ---- ciface_inline -----------------------------------------------------------------------------
# Keep exceptionally wide MLIR C-interface boundaries out of target backends.
#
# ``llvm.emit_c_interface`` deliberately exposes one descriptor pointer per memref, but its
# generated wrapper expands those pointers into the flattened implementation ABI.  Large models can
# therefore contain a wrapper call with hundreds of scalar arguments.  Once the implementation gains
# an external call, LLVM may stop inlining that boundary and some targets cannot legally materialize
# the enormous outgoing frame (LLVM's RISC-V backend fails in register scavenging, for example).
#
# This repair is ABI-driven, not model- or target-driven: mark only implementation functions that
# have a matching ``_mlir_ciface_<name>`` wrapper and exceed the declared flattened-argument limit.
# Inlining removes the artificial wrapper-to-implementation call while preserving both public
# symbols and all runtime-visible argument semantics.

DEFAULT_MAX_FLATTENED_ARGUMENTS = 256

#: LLVM identifier characters, per the grammar this module accepts.  The leading character may be
#: neither a digit nor a dash, which ``_define_name`` enforces separately.
_NAME_CHARS = frozenset(ascii_letters + digits + "$._-")
_NAME_HEAD = frozenset(ascii_letters + "$._")
#: Word characters, so an attribute lookup matches whole words only.
_WORD_CHARS = frozenset(ascii_letters + digits + "_")


def _define_name(line: str) -> tuple[str, int] | None:
    """The symbol a ``define`` line declares, and the offset its name starts at.

    Parsed structurally rather than by pattern.  A too-narrow pattern silently drops a
    valid-but-differently-spelled definition, which produces a wrong ABI repair instead of a loud
    failure.  Quoted symbol names (``@"with space"``) are deliberately not accepted, matching this
    module's rule of never guessing at LLVM syntax.
    """
    if not line.startswith("define"):
        return None
    if not line[len("define") :][:1].isspace():
        return None
    at = line.find("@")
    while at != -1:
        start = end = at + 1
        while end < len(line) and line[end] in _NAME_CHARS:
            end += 1
        name = line[start:end]
        if name and name[0] in _NAME_HEAD and line[end : end + 1] == "(":
            return name, start
        at = line.find("@", at + 1)
    return None


def _ends_with_value_name(part: str) -> bool:
    """Whether an argument fragment ends in a ``%ssa`` value name.

    A parameter carrying a name is a value argument; a bare type is not.  Scanned back to the final
    ``%`` rather than matched, for the same reason as ``_define_name``.
    """
    text = part.rstrip()
    at = text.rfind("%")
    if at == -1 or at == len(text) - 1:
        return False
    return all(character in _NAME_CHARS for character in text[at + 1 :])


def _has_attribute(attributes: str, wanted: str) -> bool:
    """Whether a whole-word attribute appears in a function-attribute fragment.

    Whole-word, so ``alwaysinlinehint`` does not count as ``alwaysinline``.
    """
    token = ""
    for character in attributes:
        if character in _WORD_CHARS:
            token += character
            continue
        if token == wanted:
            return True
        token = ""
    return token == wanted


@dataclass(frozen=True)
class InlinedCInterfaceBoundary:
    implementation: str
    wrapper: str
    flattened_arguments: int


def _signature_close(line: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(line)):
        token = line[index]
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _value_argument_count(arguments: str) -> int:
    """Count top-level LLVM arguments without splitting aggregate/function-pointer types."""
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    closes = {")": "(", "]": "[", "}": "{", ">": "<"}
    quoted = False
    escaped = False
    for index, token in enumerate(arguments):
        if quoted:
            if escaped:
                escaped = False
            elif token == "\\":
                escaped = True
            elif token == '"':
                quoted = False
            continue
        if token == '"':
            quoted = True
        elif token in depths:
            depths[token] += 1
        elif token in closes:
            depths[closes[token]] -= 1
        elif token == "," and not any(depths.values()):
            parts.append(arguments[start:index])
            start = index + 1
    if arguments.strip():
        parts.append(arguments[start:])
    return sum(_ends_with_value_name(part.strip()) for part in parts)


def inline_wide_ciface_implementations(
    llvm_ir: str, *, max_flattened_arguments: int = DEFAULT_MAX_FLATTENED_ARGUMENTS
) -> tuple[str, dict]:
    """Add ``alwaysinline`` to wide implementations that have an MLIR C wrapper.

    The LLVM translator currently prints each function signature on one line.  Refuse malformed or
    multiline definitions by leaving them untouched; this helper must never guess at LLVM syntax.
    The returned report makes the automatic ABI repair auditable in lowering receipts.
    """
    if max_flattened_arguments < 1:
        raise ValueError("max_flattened_arguments must be positive")
    lines = llvm_ir.splitlines(keepends=True)
    definitions: dict[str, tuple[int, int, int, int]] = {}
    for line_index, line in enumerate(lines):
        declared = _define_name(line)
        if declared is None:
            continue
        name, name_start = declared
        open_index = line.find("(", name_start + len(name))
        close_index = _signature_close(line, open_index)
        if close_index is None or "{" not in line[close_index + 1 :]:
            continue
        arguments = line[open_index + 1 : close_index]
        argument_count = _value_argument_count(arguments)
        definitions[name] = (line_index, open_index, close_index, argument_count)

    repairs: list[InlinedCInterfaceBoundary] = []
    for name, (line_index, _open, close, argument_count) in definitions.items():
        if name.startswith("_mlir_ciface_") or argument_count <= max_flattened_arguments:
            continue
        wrapper = f"_mlir_ciface_{name}"
        if wrapper not in definitions:
            continue
        line = lines[line_index]
        brace = line.find("{", close + 1)
        attributes = line[close + 1 : brace]
        if _has_attribute(attributes, "alwaysinline"):
            continue
        lines[line_index] = f"{line[:brace].rstrip()} alwaysinline {line[brace:]}"
        repairs.append(InlinedCInterfaceBoundary(name, wrapper, argument_count))

    return "".join(lines), {
        "schema": "wide_ciface_inline_v1",
        "max_flattened_arguments": max_flattened_arguments,
        "inlined": [asdict(repair) for repair in repairs],
        "count": len(repairs),
    }
