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


def _stage(directory: Path, label: str, before: str, after: str) -> Path:
    """Write one boundary in the recorder's layout (``merlin.lean.stage.v1``)."""
    stage = directory / "stage"
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


def tensor_refinement(job: dict[str, Any]) -> dict[str, Any]:
    """Prove a pass's output refines its input with Merlin's tensor checker.

    The boundary must verify under ``MerlinVerify.Check.TypedModule.check_sound``, every function of
    the input must verify under ``supportCheck_sound`` (so the refinement is strict, not conditional),
    and ``compose_chain`` must replay both. Definedness (no UB or poison) is not claimed.
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
    chains = {}
    for function in typed["functions"]:
        support = check.check_support(
            stage / "before.mlir", function, directory / f"support-{function}", m2m_python=python, index_bits=bits
        )
        if support["verdict"] != "verified":
            raise GateUnsupported(f"source support of @{function} is not established: {support.get('reason')}")
        chain = check.compose_chain(
            [stage],
            directory / f"support-{function}" / "receipt.json",
            function,
            directory / f"chain-{function}.json",
            m2m_python=python,
        )
        if chain.get("refinement_verdict") != "verified":
            raise GateRejected(f"the refinement of @{function} did not compose: {chain.get('reason')}")
        chains[function] = str(directory / f"chain-{function}.json")
    return {
        "claim": "strict refinement on inputs of the declared types; definedness not established",
        "theorems": THEOREMS,
        "functions": typed["functions"],
        "axioms": typed["axioms"],
        "receipt": str(stage / "receipt.json"),
        "chains": chains,
    }
