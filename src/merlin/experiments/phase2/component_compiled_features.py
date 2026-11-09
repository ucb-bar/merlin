"""Static features from the ordinary consumed compile-only artifacts.

This fixed observer reopens source membership and the actual native producer
chain. Its data cannot qualify source equivalence, compiler isolation, ISA,
runtime, physical traffic or calibrated feedback. No execution is requested.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import execution_deadline
from merlin.common import invocation_record as I
from merlin.common.strict_json import loads
from merlin.llvmlower import codegen
from merlin.perf import compiled_static_features as S
from merlin.perf import component_source_demand as D
from merlin.perf.component_cost import ComponentCostScope
from merlin.targetgen import capsule_common, component_program
from merlin.targetgen import compile_only_execution as CO
from merlin.targetgen.contract import compile as compile_owner
from merlin.targetgen.contract import elf_admission as admission_owner
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

from . import corpus as C
from .contracts import StageGateError, document_sha256, exact_tree_record, mapping_file, sha256_file
from .feedback_protocol import FeedbackValueLimits, encode_feedback_value


def _plain(path, *, file=False):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(part.is_symlink() for part in (path, *path.parents))
        or file
        and not stat.S_ISREG(path.stat().st_mode)
    ):
        raise StageGateError("compile features require canonical unlinked regular members before reading")
    return path


def _pin(path):
    return {"path": str(_plain(path, file=True)), "sha256": sha256_file(path)}


def _tree(root, *, selection):
    root = _plain(root)
    if not root.is_dir():
        raise StageGateError("compile feature source/product owner is absent")
    pending, members, count, size = [root], [], 0, 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                count += 1
                info = entry.stat(follow_symlinks=False)
                if count > selection.max_members:
                    raise StageGateError("compile feature whole member budget exceeded")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    members.append(Path(entry.path))
                    size += info.st_size
                    if size > selection.max_tree_bytes:
                        raise StageGateError("compile feature whole byte budget exceeded")
                else:
                    raise StageGateError("compile feature owner contains an alias or special file")
    return tuple(sorted(members))


def _json(path, limits):
    path = _plain(path, file=True)
    if path.stat().st_size > limits.max_bytes:
        raise StageGateError("compile feature metadata exceeds its byte budget")
    value = loads(path.read_text())
    encode_feedback_value(value, limits=limits)
    return value


@dataclass(frozen=True)
class CompileFeatureSelection:
    """Explicit tools and bounds; neither a grant nor a qualification owner."""

    build_service: BuildOnlyService
    elf_admission: LinkedElfAdmissionService
    translator: tuple[str, str]
    object_compiler: tuple[str, str]
    readelf: tuple[str, str]
    package_tools: tuple[tuple[str, str], ...]
    pointer_bits: int
    static_limits: S.StaticFeatureLimits
    source_limits: D.SourceDemandLimits
    metadata_limits: FeedbackValueLimits
    max_members: int
    max_tree_bytes: int

    def tools(self):
        return self.translator, self.object_compiler, self.readelf, *self.package_tools

    def record(self):
        return {
            "tools": self.tools(),
            "pointer_bits": self.pointer_bits,
            "static_limits": asdict(self.static_limits),
            "source_limits": asdict(self.source_limits),
            "metadata_limits": asdict(self.metadata_limits),
            "max_members": self.max_members,
            "max_tree_bytes": self.max_tree_bytes,
            "build_source_pins": self.build_service.source_pins,
            "instruction_source_pins": self.elf_admission.source_pins,
        }

    def verify_limits(self):
        if (
            type(self.build_service) is not BuildOnlyService
            or type(self.elf_admission) is not LinkedElfAdmissionService
            or type(self.static_limits) is not S.StaticFeatureLimits
            or type(self.source_limits) is not D.SourceDemandLimits
            or type(self.metadata_limits) is not FeedbackValueLimits
            or type(self.pointer_bits) is not int
            or not 1 <= self.pointer_bits <= 256
            or type(self.max_members) is not int
            or not 0 < self.max_members <= 100000
            or type(self.max_tree_bytes) is not int
            or not 0 < self.max_tree_bytes <= 1 << 30
            or type(self.package_tools) is not tuple
            or len(self.package_tools) > self.max_members
        ):
            raise StageGateError("compile features require exact explicit services/tools and bounded selection")
        self.static_limits.verify()
        self.source_limits.verify()
        self.metadata_limits.verify()

    def verify(self, target):
        self.verify_limits()
        for pin in self.tools():
            if type(pin) is not tuple or len(pin) != 2 or _pin(Path(pin[0])) != {"path": pin[0], "sha256": pin[1]}:
                raise StageGateError("compile feature selected native tool changed")
        self.build_service.verify(target)
        self.elf_admission.verify(target)


def _producer(rows, *, stage, tool, inputs, output=None, argv=None):
    matches = [row for row in rows if row[1]["stage"] == stage]
    if len(matches) != 1:
        raise StageGateError("compile features lack a unique original native producer: " + stage)
    path, row = matches[0]
    if (
        row.get("kind") != "subprocess"
        or row["executable"] != tool
        or any(pin not in row["inputs"] for pin in inputs)
        or output is not None
        and output not in row["outputs"]
        or any(row["argv"].count(pin["path"]) != 1 for pin in (*inputs, *((output,) if output else ())))
        or argv is not None
        and row["argv"] != argv
    ):
        raise StageGateError("compile features do not join actual tool/argv/input/product consumption: " + stage)
    return path, row


def observe_component_compiled_features(
    *, report_path, package_dir, contract_root, corpus, member, scope, selection, evidence_root
):
    """Reopen exact original source and actual LLVM/object/linked-ELF products.

    This first explicit version supports checked integer component programs and
    the unrepaired ordinary native translation/object route. Other source forms
    and transformed object routes refuse; they cannot produce optimistic zeroes.
    A live corpus view is source membership only, not performance admission.
    """
    if (
        type(selection) is not CompileFeatureSelection
        or type(corpus) is not C.FrozenPerformanceCorpus
        or type(member) is not C.PerformanceCapsule
        or type(scope) is not ComponentCostScope
        or not any(value is member for value in corpus.capsules)
    ):
        raise StageGateError("compile features require exact original corpus/member/scope selections")
    path = _plain(report_path, file=True)
    selection.verify_limits()
    if not corpus.capsules or len(corpus.capsules) > selection.max_members:
        raise StageGateError("compile feature complete source roster exceeds its member budget")
    root, package, contract = path.parent, _plain(package_dir), _plain(contract_root)
    owners = (root, package, contract, _plain(corpus.root))
    members = tuple(member for owner in owners for member in _tree(owner, selection=selection))
    if len(members) > selection.max_members or sum(p.stat().st_size for p in members) > selection.max_tree_bytes:
        raise StageGateError("compile feature combined source/product roster exceeds its budget")
    if (
        _plain(corpus.manifest_path, file=True) not in members
        or not _plain(corpus.capsules_root).is_relative_to(corpus.root)
        or not _plain(member.source_dir).is_relative_to(corpus.capsules_root)
    ):
        raise StageGateError("compile feature corpus/source membership escapes the original owner")
    before = tuple(_pin(p) for p in sorted(set(members)))
    report = _json(path, selection.metadata_limits)
    target = report.get("target")
    for selected in (
        *selection.build_service.source_pins,
        *selection.elf_admission.source_pins,
        *selection.tools(),
    ):
        if _plain(selected[0], file=True).stat().st_size > selection.max_tree_bytes:
            raise StageGateError("compile feature selected source/tool exceeds its byte budget")
    selection.verify(target)
    manifest = _json(corpus.manifest_path, selection.metadata_limits)
    identities = [(row.family, row.capsule) for row in corpus.capsules]
    declared = [(row["family"], row["capsule"]) for row in manifest["capsules"]]
    if (
        len(set(identities)) != len(identities)
        or len(set(declared)) != len(declared)
        or set(identities) != set(declared)
    ):
        raise StageGateError("compile feature original corpus member roster is incomplete or repeated")
    encode_feedback_value([row.descriptor for row in corpus.capsules], limits=selection.metadata_limits)
    original_views = document_sha256(
        [(row.family, row.capsule, row.source_sha256, row.descriptor) for row in corpus.capsules]
    )
    for capsule in corpus.capsules:
        source_root = _plain(capsule.source_dir)
        if not source_root.is_relative_to(corpus.capsules_root):
            raise StageGateError("compile feature corpus member escapes its original source owner")
        encode_feedback_value(capsule.descriptor, limits=selection.metadata_limits)
        actual = exact_tree_record(source_root)
        if (actual["sha256"], actual["n_files"], actual["n_bytes"]) != (
            capsule.source_sha256,
            capsule.n_files,
            capsule.n_bytes,
        ):
            raise StageGateError("compile feature corpus member differs from actual source bytes")
        if (source_root / "capsule.yaml").stat().st_size > selection.metadata_limits.max_bytes:
            raise StageGateError("compile feature original descriptor exceeds its metadata budget")
    C.verify_frozen_performance_corpus(corpus)
    encode_feedback_value(member.descriptor, limits=selection.metadata_limits)
    descriptor = mapping_file(member.source_dir / "capsule.yaml", yaml_file=True)
    if (
        document_sha256(descriptor) != document_sha256(member.descriptor)
        or descriptor.get("operation", {}).get("op") != "component_program"
    ):
        raise StageGateError("compile features have no checked original integer component source")
    program = descriptor["operation"]["attributes"]["program"]
    storage = descriptor["component_program"]["selected_storage"]
    demand = D.derive_source_demand(
        program=program,
        operand_dtype=storage["operand"],
        accumulator_dtype=storage["accumulator"],
        schedule=tuple(row["name"] for row in program["nodes"]),
        limits=selection.source_limits,
    )
    if demand["status"] != "observed":
        raise StageGateError("compile feature original source grammar/arithmetic is UNKNOWN")
    typed, original_text = component_program.render(
        program, operand_dtype=storage["operand"], accumulator_dtype=storage["accumulator"]
    )
    name = descriptor.get("interface_mlir")
    if type(name) is not str or Path(name).name != name:
        raise StageGateError("compile feature original source is not a direct member")
    original = _plain(member.source_dir / name, file=True)
    abi = CompileOnlySourceAbi(
        *(
            tuple(CompileOnlyTensor(row["name"], tuple(row["shape"]), row["dtype"]) for row in typed[role])
            for role in ("inputs", "outputs")
        )
    )
    if (
        original.read_text() != original_text
        or document_sha256(typed) != document_sha256(descriptor["component_program"])
        or report["inputs"]["source"] != _pin(original)
        or report["inputs"]["original_abi"] != abi.record()
        or report["inputs"]["package_root"] != str(package)
        or report["inputs"]["contract_root"] != str(contract)
        or report["inputs"]["readelf"] != dict(zip(("path", "sha256"), selection.readelf, strict=True))
    ):
        raise StageGateError("compile features changed the original source/ABI/compiler/contract/tool membership")
    # Validate every scanned record and every referenced file before the normal
    # verifier reads it. Saved records cannot add a foreign dependency/alias.
    source_owners = (I, CO, capsule_common, compile_owner, codegen, execution_deadline, admission_owner, S, D)
    selected_paths = set(members) | {Path(module.__file__).resolve() for module in source_owners}
    selected_paths |= {
        Path(pin[0])
        for pin in (
            *selection.build_service.source_pins,
            *selection.elf_admission.source_pins,
            *selection.tools(),
        )
        if type(pin) is tuple
    }
    recipe = selection.build_service.recipe
    selected_paths |= {recipe.compiler, recipe.link_script, *recipe.support_sources, *recipe.header_dependencies}
    rows = []
    for record in (p for p in members if p.is_relative_to(root) and p.name == "invocation.json"):
        row = _json(record, selection.metadata_limits)
        for pin in (
            row["executable"],
            row["stdout"],
            row["stderr"],
            *row["inputs"],
            *row["outputs"],
            *row["dependencies"],
        ):
            file = _plain(pin["path"], file=True)
            if file not in selected_paths or file.stat().st_size > selection.max_tree_bytes:
                raise StageGateError("compile feature record introduces an unselected or excessive member")
        rows.append((record, I.verify(record)))
    if Path(report["elf"]["path"]) not in members or Path(report["instruction_policy"]["report_path"]) not in members:
        raise StageGateError("compile feature report introduces a foreign linked product")
    CO.verify_compile_only_report(path, build_service=selection.build_service, elf_admission=selection.elf_admission)
    generated, build = root / "generated", root / "build"
    llvm, staged, translated, obj = (
        generated / "lowered.llvm.mlir",
        build / "kernel.llvm.mlir",
        build / "kernel.ll",
        build / "kernel.o",
    )
    elf = _plain(report["elf"]["path"], file=True)
    if any(p not in members for p in (llvm, staged, translated, obj, elf)) or llvm.read_bytes() != staged.read_bytes():
        raise StageGateError("compile features do not bind the actual emitted and consumed LLVM bytes")
    if (
        llvm.stat().st_size > selection.static_limits.max_source_bytes
        or elf.stat().st_size > selection.static_limits.max_elf_bytes
    ):
        raise StageGateError("compile feature emitted source or ELF exceeds its selected byte limit")
    lowering = [row for _, row in rows if row["stage"] == "compile_only_source_lowering"]
    if (
        len(lowering) != 1
        or lowering[0]["kind"] != "python_call"
        or _pin(original) not in lowering[0]["inputs"]
        or _pin(llvm) not in lowering[0]["outputs"]
        or lowering[0]["stdout"]["sha256"] != sha256_file(llvm)
    ):
        raise StageGateError("compile features lack the original source to actual emitted LLVM join")
    translation = _producer(
        rows,
        stage="llvm_translation",
        tool=_pin(Path(selection.translator[0])),
        inputs=(_pin(staged),),
        output=_pin(translated),
    )
    object_record = _producer(
        rows,
        stage="object",
        tool=_pin(Path(selection.object_compiler[0])),
        inputs=(_pin(translated),),
        output=_pin(obj),
    )
    link = _producer(rows, stage="elf", tool=_pin(recipe.compiler), inputs=(_pin(obj),), output=_pin(elf))
    entry_record = _producer(
        rows,
        stage="compile_only_linked_entry",
        tool=_pin(Path(selection.readelf[0])),
        inputs=(_pin(elf),),
        argv=[selection.readelf[0], "--wide", "--symbols", str(elf)],
    )
    entry_symbol = recipe.require_kernel_stack_frame().entry_symbol
    definitions = set()
    for line in Path(entry_record[1]["stdout"]["path"]).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == "FUNC" and fields[6] != "UND" and fields[7] == entry_symbol:
            definitions.add((int(fields[1], 16), int(fields[2])))
    if len(definitions) != 1:
        raise StageGateError("compile features lack one actual native retained entry definition")
    address, size = next(iter(definitions))
    retained = {"symbol": entry_symbol, "address": address, "size_bytes": size, "scope": "static linked definition"}
    if retained != report["retained_entry"]:
        raise StageGateError("compile feature entry extent differs from actual native symbol output")
    features = S.derive_compiled_static_features(
        llvm_mlir=llvm.read_text(),
        elf_bytes=elf.read_bytes(),
        entry_symbol=entry_symbol,
        pointer_bits=selection.pointer_bits,
        retained_entry=retained,
        limits=selection.static_limits,
    )
    evidence = _plain(evidence_root)
    if evidence.exists() or any(evidence.is_relative_to(p) or p.is_relative_to(evidence) for p in owners):
        raise StageGateError("compile features need a fresh disjoint observation owner")
    evidence.mkdir(parents=True, mode=0o700)
    product = evidence / "static_features.json"
    inputs = tuple(sorted(set(members)))
    with I.observe_call(
        evidence,
        stage="compiled_static_features",
        function=observe_component_compiled_features,
        arguments={"selection": selection.record(), "scope_sha256": scope.sha256},
        inputs=inputs,
        outputs=(product,),
        dependencies=tuple(sorted(selected_paths - set(members))) + (Path(__file__).resolve(),),
    ) as call:
        result = {
            "schema": "merlin.component_compiled_features.v1",
            "features": features,
            "source_membership": {
                "manifest_sha256": corpus.manifest_sha256,
                "corpus_sha256": corpus.capsules_sha256,
                "member_sha256": member.source_sha256,
                "descriptor_sha256": document_sha256(member.descriptor),
                "family": member.family,
                "capsule": member.capsule,
            },
            "scope_sha256": scope.sha256,
            "target": target,
            "selection": selection.record(),
            "report": _pin(path),
            "products": [_pin(p) for p in (original, llvm, staged, translated, obj, elf)],
            "producer_records": [_pin(row[0]) for row in (translation, object_record, link, entry_record)],
            "static_obligations": report["static_obligations"],
            "authority": "none",
            "execution": "not_attempted",
            "numeric": "not_attempted",
            "scope": "actual compile consumption and static sites only; no physical/performance qualification",
        }
        text = json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
        with product.open("x") as stream:
            stream.write(text)
        call.returned(stdout=text)
    I.verify(call.path)
    selection.verify(target)
    C.verify_frozen_performance_corpus(corpus)
    CO.verify_compile_only_report(path, build_service=selection.build_service, elf_admission=selection.elf_admission)
    if before != tuple(_pin(p) for p in sorted(set(members))):
        raise StageGateError("compile feature original source/product bytes changed during observation")
    if original_views != document_sha256(
        [(row.family, row.capsule, row.source_sha256, row.descriptor) for row in corpus.capsules]
    ):
        raise StageGateError("compile feature original source views changed during observation")
    return result
