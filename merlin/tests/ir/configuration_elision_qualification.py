"""Independent actual configuration-pass corpus; no saved-report proof authority."""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from configuration_elision_qualification_cases import QUALIFICATION, TARGET, boundary_cases, cases, declarations
from configuration_elision_qualification_process import run, save

from merlin.common.paths import repo_root


def corpus(args):
    root = repo_root()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = root / "merlin/tests/ir/configuration_elision_qualification_worker.py"
    environment = {
        **os.environ,
        "PYTHONPATH": str(args.reference.resolve() / "src"),
        "MERLIN_REPO_ROOT": str(args.reference.resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    rows = list(cases())
    save(output / "EXPECTED-POSITIVES.json", [dataclasses.asdict(row) for row in rows])
    for bits in (32, 64):
        folder = output / f"schema-i{bits}"
        folder.mkdir()
        facts, software = declarations(bits)
        (folder / "facts.json").write_bytes(facts)
        (folder / "software.yaml").write_bytes(software)
        save(
            folder / "POLICY-STATUS.json",
            dict(
                human_approved=False,
                hardware_qualified=False,
                parser_required_reviewed_label="SYNTHETIC TEST DATA ONLY",
                qualification_identity=QUALIFICATION,
            ),
        )

    def request(bits):
        schema = output / f"schema-i{bits}"
        return dict(
            reference=str(args.reference.resolve()),
            facts=str(schema / "facts.json"),
            software=str(schema / "software.yaml"),
            target=TARGET,
            qualification=QUALIFICATION,
        )

    positives, prepared, files = [], {}, []

    def produce(name, source, bits, expected, *, boundary=False):
        folder = output / ("boundaries" if boundary else "positives") / name
        folder.mkdir(parents=True)
        (folder / "authored.mlir").write_text(source)
        actual = run(
            [sys.executable, worker, "produce"],
            folder / "producer",
            environment=environment,
            request={**request(bits), "source": source},
        )
        before, after = actual.pop("before"), actual.pop("after")
        (folder / "stage0.mlir").write_text(before)
        (folder / "stage1.mlir").write_text(after)
        save(folder / "PRODUCER.json", actual)
        removed = actual["producer_report"]["verification"]["removed_operations"]
        if removed != expected or before == after:
            raise ValueError("actual producer did not make the declared independent fixture changes")
        prepared[name] = dict(before=before, after=after, bits=bits)
        files.extend([folder / "authored.mlir", folder / "stage0.mlir", folder / "stage1.mlir"])
        row = dict(
            name=name,
            bits=bits,
            removed=removed,
            status="ACTUAL_PRODUCER_CHANGED_UNCOMPILED",
            human_approved=False,
            source_decoder_compared=True,
        )
        positives.append(row)
        save(output / "POSITIVES.json", positives)
        return actual

    for row in rows:
        produce(row.name, row.source, row.bits, row.removed)
    for name, source in boundary_cases().items():
        produce(name, source, 32, 2 if name == "independent_invocations" else 1, boundary=True)

    mutations = [
        ("multifield", name)
        for name in (
            "drop_first_write",
            "drop_observer",
            "wrong_retained_value",
            "wrong_same_typed_binding",
            "changed_constant",
            "changed_visibility",
            "changed_symbol",
            "changed_input_type",
            "changed_signature",
            "changed_metadata",
            "reordered_observer",
            "inserted_observer",
            "drop_unused_constant",
        )
    ]
    mutations += [
        (case, name)
        for case, name in (
            ("single_fields", "wrong_retained_field"),
            ("partial_clobber", "drop_reset"),
            ("partial_clobber", "drop_required_after_clobber"),
            ("unknown_effect", "drop_unknown"),
            ("unordered_effect", "drop_unordered"),
            ("pulse_retained", "drop_pulse"),
            ("ssa_results", "wrong_token_binding"),
            ("partially_changed_value", "drop_changed_pair"),
        )
    ]
    save(output / "EXPECTED-MUTATIONS.json", [dict(source_case=case, mutation=name) for case, name in mutations])
    negative = []
    for source_case, name in mutations:
        original = prepared[source_case]
        folder = output / "negatives" / name
        folder.mkdir(parents=True)
        actual = run(
            [sys.executable, worker, "mutate"],
            folder / "producer",
            environment=environment,
            request={**request(original["bits"]), **original, "mutation": name},
        )
        (folder / "stage0.mlir").write_text(original["before"])
        (folder / "stage1.mlir").write_text(actual.pop("after"))
        save(folder / "MUTATION.json", actual)
        files.extend([folder / "stage0.mlir", folder / "stage1.mlir"])
        negative.append(
            dict(
                name=name,
                source_case=source_case,
                status="XDSL_VALID_AWAITING_AUTHORITATIVE_COMPILER",
                original_production_conformance=actual["original_conformance"]["status"],
            )
        )
        save(output / "NEGATIVES.json", negative)
    oracle = run(
        [args.compiler_python, worker, "verify"],
        output / "compiler-byte-verification",
        environment=os.environ.copy(),
        request=dict(files=list(map(str, files))),
    )
    save(output / "COMPILER.json", oracle)
    invalid = [row for row in oracle["files"] if not row["valid"]]
    if invalid:
        raise ValueError("qualification fixture not authoritative compiler-valid: " + str(invalid))
    result = dict(
        status="ACTUAL_PRODUCER_CORPUS_COMPILER_VALID_UNCOMPILED_UNQUALIFIED",
        expected_changed_core_cases=len(rows),
        additional_changed_boundary_cases=len(positives) - len(rows),
        valid_mutant_pairs=len(negative),
        compiler_valid_complete_buffers=len(files),
        human_approved=False,
        formal_checker_executed=False,
        effect_model_hardware_fidelity=False,
    )
    save(output / "RESULT.json", result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--compiler-python", required=True, type=Path)
    corpus(parser.parse_args())


if __name__ == "__main__":
    main()
