"""A Phase 2 gSIM measurement that reads its outputs back as one digest, verified by Spike.

Reading a large output back over gSIM's simulated serial link costs hours: a 3136x64 i32 output
was still printing after 2 h as text, as a binary frame, and as a coherent memory dump. The cycles of
a measurement cell come from the ``METRIC`` window and are unaffected by how outputs come back, but
the cell must still show that the gSIM run produced the right outputs.

So the grader-rendered harness of the gSIM run prints one XXH64 per output (``out_digest_v1``, never
candidate code), and that digest is tied to full values by these checks:

1. the program's full-value build runs on Spike, where reading values back is cheap: values ``V``;
2. the digest build -- the SAME kernel object behind the digest harness -- runs on Spike, and its digest
   must be the digest of ``V``'s container bytes (the digest program holds exactly ``V`` on Spike);
3. the SAME ELF bytes (sha256 checked) run on gSIM, and its digest must equal the Spike digest.

Then gSIM held ``V`` (up to an accidental 64-bit collision), and ``V`` is what the cell reports --
labelled as values verified by digest rather than read from gSIM. Any disagreement falls back to the
historical full gSIM readback, so a numerical verdict never rests on the shortcut being right.
``MERLIN_PHASE2_GSIM_READBACK=full`` keeps the full readback for every cell.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_file
from merlin.targetgen.contract import compile as OOT
from merlin.targetgen.contract import readback_policy as RB

MODE_ENV = "MERLIN_PHASE2_GSIM_READBACK"
SCHEMA = "merlin_gsim_digest_readback_v1"


def mode() -> str:
    """``digest`` (default) or ``full``: how Phase 2 gSIM runs read their outputs back."""
    selected = os.environ.get(MODE_ENV, "digest").strip().lower()
    if selected not in ("digest", "full"):
        raise ValueError(f"{MODE_ENV} must be 'digest' or 'full', got {selected!r}")
    return selected


def enabled() -> bool:
    return mode() == "digest"


def run_gsim(
    cb: dict[str, Any],
    llvm_text: str,
    *,
    target: str,
    workdir: str | Path,
    timeout: int,
    oracle: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The gSIM oracle result for ``cb``: digest-verified when the checks hold, else a full readback."""
    oracle = OOT.run_on_oracle if oracle is None else oracle
    workdir = Path(workdir)

    def full_gsim(reason: str) -> dict[str, Any]:
        result = oracle(cb, llvm_text, simulator="gsim", target=target, workdir=workdir, timeout=timeout)
        result["readback"] = {"schema": SCHEMA, "mode": "full", "reason": reason}
        return result

    if not enabled():
        return full_gsim(f"{MODE_ENV}=full")
    policy = RB.ReadbackPolicy(RB.OUT_DIGEST_V1)
    digest_dir = workdir / "digest_readback"
    values_dir = workdir / "spike_full_values"
    digest_dir.mkdir(parents=True, exist_ok=True)
    values_dir.mkdir(parents=True, exist_ok=True)
    try:
        values = oracle(cb, llvm_text, simulator="spike", target=target, workdir=values_dir, timeout=timeout)
        on_spike = oracle(
            cb,
            llvm_text,
            simulator="spike",
            target=target,
            workdir=digest_dir,
            timeout=timeout,
            readback_policy=policy,
        )
    except Exception as exc:  # noqa: BLE001 -- the shortcut is optional; the full readback decides
        return full_gsim(f"spike digest verification unavailable: {type(exc).__name__}: {exc}"[:400])
    roster = on_spike.get("output_digests") or {}
    if not roster:
        return full_gsim("the digest program reported no output digests on Spike")
    spike_elf = sha256_file(Path(str(on_spike["elf"])))
    try:
        mismatched = RB.digest_mismatches(roster, values["outputs"])
    except (KeyError, ValueError) as exc:
        return full_gsim(f"spike values cannot be packed for their digest: {exc}"[:400])
    if mismatched:
        return full_gsim(f"the digest program's Spike output differs from the full-value build: {mismatched}")
    on_gsim = oracle(
        cb, llvm_text, simulator="gsim", target=target, workdir=digest_dir, timeout=timeout, readback_policy=policy
    )
    gsim_elf = sha256_file(Path(str(on_gsim["elf"])))
    if gsim_elf != spike_elf:
        return full_gsim("the gSIM digest build is not the ELF Spike verified")
    observed = {name: row["digest"] for name, row in (on_gsim.get("output_digests") or {}).items()}
    expected = {name: row["digest"] for name, row in roster.items()}
    if observed != expected:
        differing = sorted(n for n in set(observed) | set(expected) if observed.get(n) != expected.get(n))
        return full_gsim(f"gSIM digests differ from the Spike-verified values: {differing}")
    result = dict(on_gsim)
    result["outputs"] = values["outputs"]
    result["readback"] = {
        "schema": SCHEMA,
        "mode": "digest",
        "policy": policy.record(),
        "elf_sha256": gsim_elf,
        "spike_full_values_elf_sha256": sha256_file(Path(str(values["elf"]))),
        "digests": expected,
        "basis": (
            "outputs are the Spike full-value readback; the same digest ELF produced identical XXH64 "
            "digests of every output on Spike and on gSIM, and those equal the digest of these values"
        ),
    }
    return result
