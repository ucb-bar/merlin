"""Built-in checks the speculative gate's worker can run.

Each takes the request dictionary (``directory``, ``before``, ``after``, ``params``) and returns a
detail dictionary on acceptance or raises a :mod:`~merlin.verify.gate_errors` error.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

THEOREMS = ("MerlinVerify.Check.TypedModule.check_sound", "MerlinVerify.Check.supportCheck_sound")


def _text(path: str | None, what: str) -> str:
    if not path:
        raise GateUnsupported(f"the check needs the exact {what} text")
    return Path(path).read_text()


def _stage(directory: Path, label: str, before: str, after: str, name: str = "stage") -> Path:
    """Write one boundary in the recorder's layout (``merlin.lean.stage.v1``)."""
    stage = directory / name
    stage.mkdir()
    (stage / "before.mlir").write_text(before)
    (stage / "after.mlir").write_text(after)
    meta = {
        "schema": "merlin.lean.stage.v1",
        "boundary": label,
        "pass": label,
        "run": uuid.uuid4().hex,
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "changed": before != after,
    }
    (stage / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return stage


def _tensor_refinement(job: dict[str, Any]) -> dict[str, Any]:
    """Prove a pass's output refines its input with Merlin's tensor checker.

    The boundary must verify under ``MerlinVerify.Check.TypedModule.check_sound``, every function of
    the input must verify under ``supportCheck_sound`` (so the refinement is strict, not conditional),
    and ``compose_chain`` must validate and compose both. Two optional strengthenings are attempted and reported,
    never required: the universal boundary theorem (``moduleCheck_sound``, for all inputs) and
    source definedness on typed, nonpoison inputs (no UB or poison), which
    ``compose_chain`` then folds into its verdict.
    """
    params = job["params"]
    sdk_bin, python = Path(params["sdk_bin"]), Path(params["compiler_python"])
    bits = int(params.get("index_bits", 64))
    if not (sdk_bin / "lake").is_file():
        raise GateUnavailable(f"no lake in the selected Lean SDK: {sdk_bin}")
    if not python.is_file():
        raise GateUnavailable(f"compiler Python not found: {python}")
    from .lean import check

    if not check.lean_exe().is_file():
        raise GateUnavailable(f"the merlin-lean checker is not built: {check.lean_exe()}")
    # check.py runs lake by name; expose exactly the selected SDK.
    os.environ["PATH"] = f"{sdk_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    directory = Path(job["directory"])
    stage = _stage(directory, job.get("label") or "pass", _text(job["before"], "input"), _text(job["after"], "output"))
    typed = check.check_typed_stage(stage, m2m_python=python, index_bits=bits)
    if typed["verdict"] != "verified":
        reason = typed.get("reason", typed["verdict"])
        if typed["verdict"] == "unmeasured":
            raise GateUnavailable(f"the typed boundary check could not run: {reason}")
        if typed.get("exporter", {}).get("round_trip") == "different":
            raise GateUnsupported(f"the authoritative MLIR exporter refused the IR: {reason}")
        raise GateRejected(f"{THEOREMS[0]} did not verify the pass: {reason}")
    strengthenings = {"universal_boundary": "not attempted", "definedness": "not attempted"}
    universal = check.check_stage(
        _stage(
            directory,
            job.get("label") or "pass",
            (stage / "before.mlir").read_text(),
            (stage / "after.mlir").read_text(),
            name="universal",
        ),
        m2m_python=python,
    )
    strengthenings["universal_boundary"] = universal["verdict"]
    defined_everywhere = True
    chains = {}
    for function in typed["functions"]:
        support = check.check_support(
            stage / "before.mlir", function, directory / f"support-{function}", m2m_python=python, index_bits=bits
        )
        if support["verdict"] != "verified":
            raise GateUnsupported(f"source support of @{function} is not established: {support.get('reason')}")
        defined = check.check_definedness(
            stage / "before.mlir", function, directory / f"defined-{function}", m2m_python=python, index_bits=bits
        )
        chain = check.compose_chain(
            [stage],
            directory / f"support-{function}" / "receipt.json",
            function,
            directory / f"chain-{function}.json",
            m2m_python=python,
            definedness=directory / f"defined-{function}" / "receipt.json"
            if defined["verdict"] == "verified"
            else None,
        )
        if chain.get("refinement_verdict") != "verified":
            raise GateRejected(f"the refinement of @{function} did not compose: {chain.get('reason')}")
        defined_everywhere = defined_everywhere and chain.get("verdict") == "verified"
        chains[function] = str(directory / f"chain-{function}.json")
    strengthenings["definedness"] = "verified" if defined_everywhere else "abstracted"
    established = [name for name, verdict in strengthenings.items() if verdict == "verified"]
    return {
        # Universal conditional refinement does not widen the typed domain on
        # which support established NotStuck, required for strict refinement.
        "claim": "strict refinement on inputs of the declared types; "
        + (
            "defined on typed, nonpoison inputs (no UB or poison)"
            if "definedness" in established
            else "definedness not established"
        ),
        "established": established,
        "strengthenings": strengthenings,
        "theorems": THEOREMS,
        "functions": typed["functions"],
        "axioms": typed["axioms"],
        "receipt": str(stage / "receipt.json"),
        "chains": chains,
    }


def tensor_refinement(job: dict[str, Any]) -> dict[str, Any]:
    """Run the real proof checks, reusing only their live results within this job."""
    from .lean.receipt_session import reuse_receipts

    with reuse_receipts():
        return _tensor_refinement(job)


def compiler_loop_refinement(job: dict[str, Any]) -> dict[str, Any]:
    from .compiler_transforms import loop_check

    return loop_check(job)


def compiler_buffer_refinement(job: dict[str, Any]) -> dict[str, Any]:
    from .compiler_transforms import buffer_check

    return buffer_check(job)


def compiler_macc_contract(job: dict[str, Any]) -> dict[str, Any]:
    from .compiler_macc import check

    return check(job)
