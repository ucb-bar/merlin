"""Explicit ordered host LLVM transforms with source and emission witnesses.

The chain enforces selected-stage composition, not the stages' semantic theorems.
Trusted synchronous stage verifiers own those theorems and validate actual emitted
IR. Toolchain/dependency closure and final object/link qualification remain separate.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_file
from merlin.common.jsonio import write_pretty_json

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
