"""Distinct finite obligations, explicitly separate from full buffer acceptance.

WholeEpoch's retained120s timeout prevents a gate completion. These tests use
only F's successful111project/first-three-query objects. No W objects, failed
module import, source-success assumption or gate authority is substituted.
"""

import argparse
import json
import os
from pathlib import Path

from buffer_reuse_qualification import check_cut, run, save
from buffer_reuse_qualification_classifier import EXPECTED, classify

from merlin.verify.buffer_reuse_gate import _ordered_sources
from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare, prepare_chain
from merlin.verify.loop_transform_intake import file_ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "corpus", "baseline-objects", "q-source-cut", "lean"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--label", action="append", help="explicit distinct targeted source-fix cases only")
    args = parser.parse_args()
    output, corpus, parent = args.output.resolve(), args.corpus.resolve(), args.baseline_objects.resolve()
    output.mkdir(parents=True, exist_ok=False)
    check_cut(args.q_source_cut)
    original = tuple((corpus / "positives/retained_two_epochs" / f"stage{i}.mlir").read_text() for i in range(4))
    baseline = prepare_chain(original, allocator=ALLOCATOR)
    hint = json.loads(baseline.hint_json)
    objects = []
    for index, (name, source) in enumerate(_ordered_sources(baseline)):
        process = parent / f"import-{index}.process.json"
        if json.loads(process.read_text())["returncode"] != 0:
            raise ValueError("component import was not actually compiled successfully")
        objects.append(
            {
                "source": file_ref(source),
                "object": file_ref(parent / "lib" / (name.replace(".", "/") + ".olean")),
                "process": file_ref(process),
            }
        )
    for index, part in enumerate(json.loads(baseline.query_parts_json)[:3]):
        source = parent / (part["module"] + ".lean")
        process = parent / f"query-{index}-{part['module']}.process.json"
        if source.read_text() != part["source"] or json.loads(process.read_text())["returncode"] != 0:
            raise ValueError("successful baseline component source/command differs")
        objects.append(
            {
                "source": file_ref(source),
                "object": file_ref(parent / "lib" / (part["module"] + ".olean")),
                "process": file_ref(process),
            }
        )
    save(
        output / "INTENT.json",
        {
            "status": "FINITE_COMPONENT_ONLY_NO_GATE_AUTHORITY",
            "parent_full_gate": "WholeEpoch120s timeout/UNAVAILABLE",
            "objects": objects,
            "expected_roots": EXPECTED,
            "runner": file_ref(Path(__file__)),
        },
    )
    results = []
    for row in json.loads((corpus / "NEGATIVES.json").read_text()):
        label = row["name"]
        if args.label and label not in args.label:
            continue
        if row["status"].startswith("HOST_"):
            results.append(
                {
                    "name": label,
                    "status": row["status"],
                    "reason": row.get("component_host_refusal", row.get("raw4_host_refusal")),
                }
            )
            save(output / "RESULTS.json", results)
            continue
        if label not in EXPECTED:
            raise ValueError("missing expected obligation before native launch: " + label)
        changed = tuple((corpus / "negatives" / label / f"stage{i}.mlir").read_text() for i in range(4))
        candidate = (
            prepare(changed[0], changed[3], hint, allocator=ALLOCATOR)
            if "baseline_UNTRUSTED" in row["component_route"]
            else prepare_chain(changed, allocator=ALLOCATOR)
        )
        if candidate.query != (corpus / "negatives" / label / "CanonicalBufferQuery.lean").read_text():
            raise ValueError("fresh component differs from retained corpus")
        parts = {part["module"]: part for part in json.loads(candidate.query_parts_json)}
        expected = EXPECTED[label]
        if expected == "GraphMemoryAfterWhole.source_term_exact":
            source = parts["WholeFinal"]["source"]
            definitions = [line for line in source.splitlines(True) if line.startswith("def independentSource :")]
            theorem = [line for line in source.splitlines(True) if line.startswith("theorem source_term_exact :")]
            if len(definitions) != 1 or len(theorem) != 1:
                raise ValueError("closed actual source finite binding required")
            options = [line for line in source.splitlines(True) if line.startswith("set_option ")]
            if options != [
                "set_option autoImplicit false\n",
                "set_option maxRecDepth 16384\n",
                "set_option maxHeartbeats 8000000\n",
            ]:
                raise ValueError("exact original source-component options required")
            query = (
                "import WholeShape\n"
                + "".join(options)
                + "open Merlin Merlin.GraphLoopsShape\nnamespace GraphMemoryAfterWhole\n"
                + definitions[0]
                + theorem[0]
                + "end GraphMemoryAfterWhole\n"
            )
            delta = (
                "Exact original closed independentSource definition and source_term_exact theorem only; "
                "original scope options preserved, no whole-chain assertion."
            )
        elif expected == "GraphMemorySCFObligation.valid":
            source = parts["WholeMemory"]["source"]
            query = source
            delta = "Unchanged actual WholeMemory query, using own compiled WholeReuse dependency."
        else:
            source = parts["WholeMaterialization"]["source"]
            if not source.startswith("import WholeEpoch\n"):
                raise ValueError("unknown materialization header")
            query = "import Merlin.GraphLoopsMemorySCFAfterEntriesWitness\n" + source[len("import WholeEpoch\n") :]
            delta = (
                "Only import WholeEpoch replaced with existing AfterEntriesWitness; "
                "every closed definition/theorem byte unchanged."
            )
        folder = output / label
        folder.mkdir()
        path = folder / "Component.lean"
        path.write_text(query)
        save(
            folder / "SOURCE-DELTA.json",
            {"description": delta, "original_source": source, "component": file_ref(path), "expected_root": expected},
        )
        process, stdout = run(
            [args.lean, "-j1", path], folder / "lean", environment={**os.environ, "LEAN_PATH": str(parent / "lib")}
        )
        classified = classify(
            label, process["returncode"], stdout, (folder / "lean/stderr.txt").read_text(), query, path
        )
        results.append({"name": label, **classified, "component_only": True, "gate_completion": False})
        save(output / "RESULTS.json", results)
    for row in objects:
        for ref in row.values():
            if file_ref(Path(ref["path"])) != ref:
                raise ValueError("successful baseline dependency changed")
    check_cut(args.q_source_cut)
    unavailable = [row["name"] for row in results if row["status"] == "UNAVAILABLE"]
    save(
        output / "FINAL.json",
        {
            "status": "FINITE_COMPONENT_BATCH_PASSED_FULL_GATE_UNAVAILABLE"
            if not unavailable
            else "PARTIAL_UNAVAILABLE",
            "ordinary_refusals": sum(row["status"] == "EXPECTED_COMPONENT_OBLIGATION_REFUSAL" for row in results),
            "host_refusals": sum(row["status"].startswith("HOST_") for row in results),
            "unavailable": unavailable,
            "full_gate_checked": False,
            "human_approved": False,
            "selected_labels": args.label,
        },
    )
    if unavailable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
