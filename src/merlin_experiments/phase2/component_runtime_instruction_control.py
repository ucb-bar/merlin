"""Private linked-instruction mutations for the ordinary independent grader.

Only a live independently decoded source-policy word is injected. The private
GNU assembler data directive uses its observed length and the actual assembler's
byte order. This support never enters author inputs or grants physical effects.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from merlin.common import invocation_record
from merlin.targetgen.contract.build_service import BuildOnlyService

from .component_instruction_audit import IndependentLinkedInstructionCheck
from .component_runtime_qualification import RuntimeControlRefusal
from .contracts import StageGateError, document_sha256, mapping_file, sha256_file, write_json

_SYMBOL = "merlin_private_prohibited_instruction_control"


def prepare_mutation(*, root, check):
    if type(check) is not IndependentLinkedInstructionCheck:
        raise StageGateError("instruction control requires a live source-bound native decoder selection")
    check.verify()
    facts = check.accessor.public_facts()
    symbol, selector = check.policy.selectors[0]
    field = facts["fields"][check.selection.selector_member]
    word = facts["source_constants"][check.selection.match_constant] | (selector << field["offset"])
    native = check.accessor.decode_words((word,), root / "mutation_native_proof")
    row = native["rows"][0]
    length = row["length"]
    if (
        row["word"] != word
        or row[check.selection.selector_member] != selector
        or type(length) is not int
        or length not in (1, 2, 4, 8)
    ):
        raise StageGateError("instruction mutation has no supported actual native word/assembler data extent")
    directive = ".byte" if length == 1 else "." + str(length) + "byte"
    source = root / "private_instruction_mutation.S"
    source.write_text(
        '.section .text.private_instruction_control,"ax",@progbits\n'
        + ".globl "
        + _SYMBOL
        + "\n"
        + _SYMBOL
        + ":\n"
        + directive
        + " "
        + str(word)
        + "\n"
    )
    write_json(
        root / "instruction_mutation.json",
        {
            "schema": "merlin.private_native_instruction_mutation.v1",
            "scope": "evaluator-only executable-section policy defect; physical effects unqualified",
            "source": str(source),
            "source_sha256": sha256_file(source),
            "symbol": _SYMBOL,
            "word": word,
            "length": length,
            "source_symbol": symbol,
            "selector": selector,
            "policy_sha256": check.policy.sha256,
            "check_sha256": check.verify(),
            "native_receipt": native["receipt_path"],
            "native_receipt_sha256": native["receipt_sha256"],
        },
    )
    return mutation_identity(root, check)


def mutation_identity(root, check):
    record_path = root / "instruction_mutation.json"
    if not record_path.is_file():
        return None
    record = mapping_file(record_path)
    source, receipt = Path(record["source"]), Path(record["native_receipt"])
    for path in (source, receipt):
        if (
            not path.is_file()
            or not path.is_relative_to(root)
            or any(part.is_symlink() for part in (path, *path.parents))
        ):
            raise StageGateError("private instruction mutation escaped its exact evaluator workspace")
    if (
        record["check_sha256"] != check.verify()
        or record["policy_sha256"] != check.policy.sha256
        or sha256_file(source) != record["source_sha256"]
        or sha256_file(receipt) != record["native_receipt_sha256"]
    ):
        raise StageGateError("private instruction source, native proof or policy changed")
    for path in (root / "mutation_native_proof").rglob("invocation.json"):
        invocation_record.verify(path)
    return document_sha256({"record_sha256": sha256_file(record_path), "record": record})


def scoped_build_service(*, build, fixture, check):
    if fixture is None or fixture.case_id != "instruction_audit.negative":
        return build
    mutation_identity(fixture.evidence_root, check)
    record = mapping_file(fixture.evidence_root / "instruction_mutation.json")
    source = Path(record["source"])
    # The selected ordinary compiler/assembler chooses endianness. Retain the
    # exact private symbol even when the original recipe removes dead sections.
    recipe = replace(
        build.recipe,
        support_sources=(*build.recipe.support_sources, source),
        ldflags=(*build.recipe.ldflags, "-Wl,--undefined=" + record["symbol"]),
    )
    pins = tuple(
        sorted(
            set(
                (
                    *build.source_pins,
                    (str(source), record["source_sha256"]),
                    (str(Path(__file__)), sha256_file(Path(__file__))),
                )
            )
        )
    )
    selected = BuildOnlyService(build.target, recipe, build.renderer, pins)
    selected.verify(build.target)
    return selected


def attribute_refusal(*, fixture, check, result):
    mutation_identity(fixture.evidence_root, check)
    mutation_path = fixture.evidence_root / "instruction_mutation.json"
    mutation = mapping_file(mutation_path)
    report = result.get("elf_admission", {})
    artifact = Path(result.get("elf", ""))
    if (
        result.get("status") != "refused_before_execution"
        or result.get("execution") != "not_attempted"
        or report.get("status") != "refused"
        or not artifact.is_relative_to(fixture.evidence_root)
    ):
        raise StageGateError("instruction negative did not produce an actual pre-execution policy rejection")
    check.admission_service().revalidate(elf=artifact, result=report, target=check.policy.target)
    if not any(
        all(hit.get(key) == mutation[key] for key in ("word", "length", "source_symbol", "selector"))
        for hit in report.get("prohibited_hits", ())
    ):
        raise StageGateError("instruction rejection does not observe the original private injected source word")
    records = [invocation_record.verify(path) for path in fixture.evidence_root.rglob("invocation.json")]
    if any(record["stage"] == "execution" for record in records):
        raise StageGateError("prohibited instruction negative reached execution before rejection")
    evidence = tuple(
        (path, sha256_file(path))
        for path in (
            artifact,
            Path(report["report_path"]),
            mutation_path,
            Path(mutation["source"]),
            Path(mutation["native_receipt"]),
        )
    )
    raise RuntimeControlRefusal(case_id=fixture.case_id, mechanism="instruction_audit", evidence_files=evidence)
