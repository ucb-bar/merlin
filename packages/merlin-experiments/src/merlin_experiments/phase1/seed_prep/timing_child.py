"""Time one public-scalability case with a frozen compiler, using an export's own grader.

Run as a script with ``python -P`` and the EXPORT's source roots on ``PYTHONPATH``,
so every grading and timing module comes from the export the runs used and none
from this checkout.  It imports nothing from ``seed_prep``.

    timing_child.py --export E --descriptor D --suite-root R --submission S
                    --case C --out O --timeout T [--emit-only]

``--emit-only`` compiles the case and reports the digests of the emitted
artifacts without running Spike; it is how an operator tells whether a
variant changes a case's executable bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _digest_tree(root: Path, suffixes: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--descriptor", required=True)
    ap.add_argument("--suite-root", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--result", required=True, help="JSON file this child writes")
    ap.add_argument("--compiler-library", help="reviewed library dir holding contract.json and python/")
    ap.add_argument("--rtl-facts", help="frozen RTL facts the runs select for host-side grading")
    a = ap.parse_args(argv)
    a.out = str(Path(a.out).absolute())
    a.submission = str(Path(a.submission).absolute())

    from merlin_experiments.corpus import public_scalability as suite
    from merlin_experiments.phase1.context import load_context
    from merlin_experiments.phase1.feedback import public_scalability as grader

    started = time.monotonic()
    doc: dict = {"case": a.case, "submission": a.submission}
    try:
        ctx = load_context(a.descriptor, repo=a.export)
        if a.rtl_facts:
            # The runs' broker restores the selected facts the same way (frozen_facts.select).
            os.environ["MERLIN_RTL_FACTS"] = str(Path(a.rtl_facts).absolute())
        root = Path(a.suite_root)
        record = suite.verify(root, target=ctx.target)
        def run(_argv=None):
            return grader.grade(a.submission, root=root, record=record, context=ctx,
                                output=Path(a.out), timeout=a.timeout, names={a.case}, max_workers=0)

        if a.compiler_library:
            # The runs grade inside the reviewed library selection; enter it the same way.
            from merlin_experiments.phase1.compiler_library import selected_child
            from merlin_experiments.phase2 import compiler_library_state as state

            library = Path(a.compiler_library)
            contract = library / "contract.json"
            digest = state._contract(json.loads(contract.read_bytes())).sha256
            result = selected_child(run)([
                "--compiler-library-root", str(library / "python"),
                "--compiler-library-contract", str(contract),
                "--compiler-library-contract-sha256", digest,
            ])
        else:
            result = run()
        observed = (result.get("observed") or {}).get(a.case) or {}
        doc["observed"] = observed
        doc["result_keys"] = sorted(result)
    except BaseException as exc:  # noqa: BLE001 - a child reports every failure as data
        doc["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    doc["wall_s"] = round(time.monotonic() - started, 2)
    doc["emitted"] = _digest_tree(Path(a.out), (".ll", ".elf", ".o", ".bin"))
    Path(a.result).write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    return 0 if "error" not in doc else 1


if __name__ == "__main__":
    sys.exit(main())
