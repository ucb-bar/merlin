"""Join the fixed conditional copy theorem to actual ordinary compiler products.

Only a live original ABI selection can supply storage preconditions. The
source roster, lowering, stock translation, actual selected object and linked
ELF are reopened; compiler metadata or a supplied proof callback is not proof.
Unproved resources/physical/runtime roles retain their original denominator.
"""

from __future__ import annotations

import json
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record
from merlin.llvmlower.counted_copy_check import check_counted_copy
from merlin.llvmlower.layout_observation import LLVMLayoutObservation, observe_compiled_layout
from merlin.targetgen.compile_only_execution import verify_compile_only_report
from merlin_experiments.phase2 import contracts as C

from .component_pointer_storage import IndependentPointerStorageSelection

_ISSUED = weakref.WeakKeyDictionary()


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": C.sha256_file(path)}


def _joined_artifacts(report_path, report):
    root = report_path.parent
    original = Path(report["inputs"]["source"]["path"])
    emitted, translated, obj = (root / "build" / name for name in ("kernel.llvm.mlir", "kernel.ll", "kernel.o"))
    if emitted.read_bytes() != (root / "generated/lowered.llvm.mlir").read_bytes():
        raise C.StageGateError("static checker cannot join the actual emitted and translated LLVM source")
    records = [(Path(row["path"]), invocation_record.verify(Path(row["path"]))) for row in report["invocations"]]
    object_records = [
        path
        for path, row in records
        if row["stage"] == "object" and row["inputs"] == [_pin(translated)] and _pin(obj) in row["outputs"]
    ]
    translators = [
        row
        for _, row in records
        if row["stage"] == "llvm_translation"
        and row["kind"] == "subprocess"
        and row["inputs"] == [_pin(emitted)]
        and row["outputs"] == [_pin(translated)]
        and "--mlir-to-llvmir" in row["argv"]
    ]
    links = [
        row
        for _, row in records
        if row["stage"] == "elf" and _pin(obj) in row["inputs"] and report["elf"] in row["outputs"]
    ]
    lowerings = [
        row
        for _, row in records
        if row["stage"] == "compile_only_source_lowering"
        and row["inputs"] == [_pin(original)]
        and _pin(root / "generated/lowered.llvm.mlir") in row["outputs"]
    ]
    if len(object_records) != 1 or len(translators) != 1 or len(links) != 1 or len(lowerings) != 1:
        raise C.StageGateError(
            "static checker needs the exact original lowering/translation/object/link join; transforms remain UNKNOWN"
        )
    return original, emitted, object_records[0]


@dataclass(frozen=True, eq=False)
class ComponentCopyProof:
    selection: IndependentPointerStorageSelection
    member: object
    build_service: object
    instruction_check: object
    transport_report: Path
    observation: LLVMLayoutObservation
    receipt: Path
    result_json: bytes
    pins: tuple[tuple[str, str], ...]
    compiler_library: object = None
    compiler_library_root: Path | None = None

    def _identity(self):
        return (
            id(self.selection),
            id(self.member),
            id(self.build_service),
            id(self.instruction_check),
            self.transport_report,
            id(self.observation),
            self.receipt,
            self.result_json,
            self.pins,
            id(self.compiler_library),
            self.compiler_library_root,
        )

    def verify(self):
        if _ISSUED.get(self) != self._identity():
            raise C.StageGateError("static copy proof requires an actual fixed checker invocation")
        for path, sha in self.pins:
            if _pin(path) != {"path": path, "sha256": sha}:
                raise C.StageGateError("static copy proof source or actual product changed")
        actual = tuple(
            sorted(
                (str(path.resolve()), C.sha256_file(path)) for path in self.receipt.parent.rglob("*") if path.is_file()
            )
        )
        if actual != self.pins:
            raise C.StageGateError("static copy proof lost its complete actual artifact membership")
        if self.receipt.read_bytes() != self.result_json:
            raise C.StageGateError("static copy proof receipt changed")
        storage = self.selection.bind_member(self.member)
        report = verify_compile_only_report(
            self.transport_report,
            build_service=self.build_service,
            elf_admission=self.instruction_check.admission_service(),
            compiler_library=self.compiler_library,
            compiler_library_root=self.compiler_library_root,
        )
        original, emitted, object_record = _joined_artifacts(self.transport_report, report)
        if (
            report["inputs"]["source"] != _pin(self.member.source)
            or report["inputs"]["original_abi"] != self.member.original_abi.record()
            or self.observation.object_record != (str(object_record), C.sha256_file(object_record))
        ):
            raise C.StageGateError("static copy proof lost its exact original source/ABI/object selection")
        storage.bind_candidate(C.mapping_file(self.transport_report.parent / "generated/command_buffer.json"))
        self.observation.verify()
        result = check_counted_copy(
            original_source=original.read_text(),
            emitted_llvm=emitted.read_text(),
            storage=storage,
            layout_observation=self.observation,
            entry_symbol=self.build_service.recipe.require_kernel_stack_frame().entry_symbol,
        )
        if C.canonical_json(result) != self.result_json:
            raise C.StageGateError("static copy proof no longer derives from its actual source/code/layout")
        records = tuple(self.receipt.parent.rglob("invocation.json"))
        checker = [
            invocation_record.verify(path)
            for path in records
            if json.loads(path.read_bytes())["stage"] == "component_counted_copy_check"
        ]
        if len(checker) != 1 or checker[0]["inputs"] != sorted(
            (_pin(original), _pin(emitted)), key=lambda row: row["path"]
        ):
            raise C.StageGateError("static copy proof lacks the actual original/code checker invocation")
        return result


def prove_component_copy(
    *,
    selection,
    member,
    transport_report,
    build_service,
    instruction_check,
    output_root,
    timeout_s,
    compiler_library=None,
    compiler_library_root=None,
):
    """Run a fixed emitted-code checker, not an operator or candidate callback."""
    if type(selection) is not IndependentPointerStorageSelection:
        raise C.StageGateError("static copy proof has no independent original pointer storage selection")
    storage = selection.bind_member(member)
    report_path = Path(transport_report).resolve(strict=True)
    report = verify_compile_only_report(
        report_path,
        build_service=build_service,
        elf_admission=instruction_check.admission_service(),
        compiler_library=compiler_library,
        compiler_library_root=compiler_library_root,
    )
    if (
        report["inputs"]["source"] != _pin(member.source)
        or report["inputs"]["original_abi"] != member.original_abi.record()
    ):
        raise C.StageGateError("static copy proof lost its original complete source/ABI")
    original, emitted, object_record = _joined_artifacts(report_path, report)
    storage.bind_candidate(C.mapping_file(report_path.parent / "generated/command_buffer.json"))
    root = Path(output_root).absolute()
    if root.exists() or root.resolve() != root or root.is_relative_to(report_path.parent):
        raise C.StageGateError("static proof requires a fresh separate retained artifact owner")
    root.mkdir(parents=True, mode=0o700)
    observation = observe_compiled_layout(
        object_record=object_record,
        integer_bits=storage.slots[0].element_bits,
        native_compiler=selection.native_compiler,
        llvm_config=selection.llvm_config,
        output_root=root / "layout",
        timeout_s=timeout_s,
    )
    receipt = root / "copy_check.json"
    dependencies = tuple(Path(path) for path, _ in observation.source_pins) + tuple(
        Path(pin.path) for pin in selection.source_pins
    )
    with invocation_record.observe_call(
        root,
        stage="component_counted_copy_check",
        function=check_counted_copy,
        arguments={"original_pointer_selection_sha256": selection.sha256, "member": member.name},
        inputs=(original, emitted),
        outputs=(receipt,),
        dependencies=(*dependencies, Path(__file__)),
    ) as call:
        result = check_counted_copy(
            original_source=original.read_text(),
            emitted_llvm=emitted.read_text(),
            storage=storage,
            layout_observation=observation,
            entry_symbol=build_service.recipe.require_kernel_stack_frame().entry_symbol,
        )
        receipt.write_bytes(C.canonical_json(result))
        call.returned()
    pins = tuple(sorted((str(path.resolve()), C.sha256_file(path)) for path in root.rglob("*") if path.is_file()))
    proof = ComponentCopyProof(
        selection,
        member,
        build_service,
        instruction_check,
        report_path,
        observation,
        receipt,
        C.canonical_json(result),
        pins,
        compiler_library,
        compiler_library_root,
    )
    _ISSUED[proof] = proof._identity()
    proof.verify()
    return proof
