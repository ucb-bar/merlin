"""Verify a decoded RoCC instruction trace against a capsule's expected coverage.

Given a trace from :mod:`merlin.targetgen.rocc.decode` and a capsule's ``expected`` block, assert:

* all required instruction classes appear; all forbidden classes are absent;
* the legal ordering invariants hold (FLUSH/FENCE bracketing, config-before-use,
  preload/compute pairing);
* the declared *modes* are actually exercised (i8 readout, relu activation bits, non-identity
  acc_scale, K-accumulation, resident reuse, movement-only);
* (optionally) decoded tile counts are consistent with the command-buffer tensor shapes.

Returns ``{"status": "pass"|"fail", "violations": [...]}``. Never raises on a mere mismatch — a
mismatch is data; it raises only on a malformed trace argument.
"""
from __future__ import annotations

from typing import Any

_COMPUTE = {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"}
_CONFIG = {"CONFIG_EX", "CONFIG_LD", "CONFIG_ST"}
_MVIN = {"MVIN", "MVIN2", "MVIN3"}


def _classes(trace: dict) -> list[str]:
    return [i["class"] for i in trace.get("instructions", [])]


def _first_index(classes: list[str], target: set[str] | str) -> int | None:
    tgt = {target} if isinstance(target, str) else target
    for i, c in enumerate(classes):
        if c in tgt:
            return i
    return None


def drives_accelerator(trace: dict) -> bool:
    """The one oracle-INDEPENDENT anti-cheese floor: did the kernel actually drive the accelerator?

    A custom-opcode instruction is one the target's RTL decoder claimed — recorded with a non-null
    ``funct`` (a plain memory ``fence`` and any non-custom asm carry ``funct=None``). This counts even
    instructions we could not sub-classify (``UNKNOWN`` with a ``funct``): the point is only that the
    kernel emitted the accelerator's ISA rather than computing on the host and moving a result. It is
    fully derived (no class-name literals) and stays true for any target — the *only* thing ``trace_check``
    gates on; every other finding is advisory (correctness is the numeric + RTL oracle, and the hidden
    golden precludes faking an answer)."""
    return any(i.get("funct") is not None for i in trace.get("instructions", []))


def dram_address_findings(trace: dict, address_model: str) -> list[str]:
    """Advisory DRAM-address provenance findings, parameterized by the HARNESS'S address model so it is
    correct for any target (never a per-target literal):

    * ``pointer_args`` — the harness passes each operand buffer as a POINTER argument (e.g. a RoCC
      bare-metal harness), so a memory-movement instruction's DRAM address MUST derive from a kernel
      argument (the decoder resolves it to ``argbase``); a baked literal (``const``) will not match the
      runtime buffer the harness allocated. Flagged.
    * ``fixed_preload`` — the harness PRELOADS each operand at a declared canonical base, so the correct
      DRAM address IS that constant; a baked ``const`` is expected, not an error. (Not flagged here; a
      value-vs-declared-base check belongs to that oracle's own path.)

    ``dram`` is the decoder's DERIVED memory-address operand (present exactly on the instructions the
    target's semantic roles mark as memory movement), and ``kind`` is the decoder's existing operand
    provenance — so this reads only the agent's OWN emitted operands (no golden), and generalizes by the
    address model, not by hardcoding an instruction class or a target."""
    if address_model != "pointer_args":
        return []
    out: list[str] = []
    for i in trace.get("instructions", []):
        dram = (i.get("decoded") or {}).get("dram")
        if isinstance(dram, dict) and dram.get("kind") == "const":
            out.append(
                f"instruction #{i.get('index')} ({i.get('class')}) uses a BAKED DRAM address "
                f"({dram.get('raw')}): the harness passes each operand as a POINTER argument, so derive "
                f"the DRAM address from the matching kernel argument (ptrtoint of the arg) — a baked "
                f"literal cannot match the buffer the runtime allocated.")
    return out


def check(trace: dict, expected: dict, cb: dict | None = None,
          address_model: str | None = None) -> dict:
    """Validate ``trace`` against capsule ``expected`` (+ optional command buffer).

    The returned ``violations`` are ADVISORY diagnostics — instruction-class coverage, ordering, and
    declared-mode checks that help the author, but do NOT decide pass/fail. The verdict is the oracle
    (numerics + L2/L3 RTL, which execute the actual emitted stream); an instruction we cannot classify
    (``UNKNOWN``) is our decoder's limit, not the backend's defect, so it is reported, never gated on.
    The sole gating signal derived here is :func:`drives_accelerator` (anti-cheese)."""
    violations: list[str] = []
    ins = trace.get("instructions", [])
    classes = _classes(trace)
    present = set(classes)

    if "UNKNOWN" in present:
        n = classes.count("UNKNOWN")
        violations.append(f"trace contains {n} UNKNOWN instruction(s) (fail-closed decode)")

    # 1. required classes present
    for req in expected.get("instruction_classes", []):
        if req not in present:
            violations.append(f"required instruction class missing: {req}")

    # 2. forbidden classes absent
    for forb in expected.get("forbidden_classes", []):
        if forb in present:
            violations.append(f"forbidden instruction class present: {forb}")

    # 3. ordering invariants
    if ins:
        if classes[0] != "FENCE":
            violations.append("trace does not open with a FENCE")
        if classes[-1] != "FENCE":
            violations.append("trace does not close with a FENCE")
    if "FLUSH" in present:
        flush_i = classes.index("FLUSH")
        first_work = _first_index(classes, _COMPUTE | _MVIN | {"MVOUT"})
        if first_work is not None and flush_i > first_work:
            violations.append("FLUSH appears after the first MVIN/MVOUT/COMPUTE")

    def _before(cfg: str, use: set[str], label: str) -> None:
        if cfg in present:
            ci = classes.index(cfg)
            ui = _first_index(classes, use)
            if ui is not None and ci > ui:
                violations.append(f"{cfg} appears after first {label}")

    _before("CONFIG_EX", _COMPUTE | {"PRELOAD"}, "PRELOAD/COMPUTE")
    _before("CONFIG_LD", _MVIN, "MVIN")
    _before("CONFIG_ST", {"MVOUT"}, "MVOUT")

    # preload/compute pairing: every COMPUTE must be immediately preceded by a PRELOAD
    n_pre = classes.count("PRELOAD")
    n_cmp = sum(classes.count(c) for c in _COMPUTE)
    if n_cmp and n_pre != n_cmp:
        violations.append(f"PRELOAD count ({n_pre}) != COMPUTE count ({n_cmp})")
    for i, c in enumerate(classes):
        if c in _COMPUTE and (i == 0 or classes[i - 1] != "PRELOAD"):
            violations.append(f"COMPUTE at #{i} is not immediately preceded by PRELOAD")
            break

    # 4. mode checks
    modes = expected.get("modes", {}) or {}
    decoded = [i.get("decoded", {}) for i in ins]

    def _any(pred) -> bool:
        return any(pred(i) for i in ins)

    if modes.get("i8"):
        if not _any(lambda i: i["class"] == "MVOUT" and i.get("decoded", {}).get("readout") == "i8"):
            violations.append("mode i8 declared but no MVOUT has i8 readout")
    if modes.get("relu"):
        if not _any(lambda i: i["class"] == "CONFIG_ST" and i.get("decoded", {}).get("relu") is True):
            violations.append("mode relu declared but no CONFIG_ST sets relu activation")
    if modes.get("acc_scale"):
        ok = _any(lambda i: i["class"] == "CONFIG_ST"
                  and (i.get("decoded", {}).get("acc_scale") not in (None, 1.0)))
        if not ok:
            violations.append("mode acc_scale declared but no CONFIG_ST has a non-identity scale")
    if modes.get("k_accumulate"):
        ok = ("COMPUTE_ACCUMULATE" in present) or _any(
            lambda i: i["class"] == "PRELOAD" and i.get("decoded", {}).get("accumulate") is True)
        if not ok:
            violations.append("mode k_accumulate declared but no accumulate-onto PRELOAD / "
                              "COMPUTE_ACCUMULATE found")
    if modes.get("resident_reuse"):
        # reuse = >=2 compute groups (MVOUTs) but weights loaded into the resident region once.
        n_mvout = classes.count("MVOUT")
        n_cfg_ex = classes.count("CONFIG_EX")
        if n_mvout < 2:
            violations.append("mode resident_reuse declared but <2 output commits (no reuse visible)")
        if n_cfg_ex != 1:
            violations.append(f"mode resident_reuse: expected a single weight-stationary config, "
                              f"saw {n_cfg_ex} CONFIG_EX")
    if modes.get("movement"):
        bad = present & _COMPUTE | (present & {"PRELOAD"})
        if bad:
            violations.append(f"mode movement declared but compute instructions present: {sorted(bad)}")
        if not present & _MVIN or "MVOUT" not in present:
            violations.append("mode movement declared but trace lacks MVIN/MVOUT")

    # 5. optional cross-validation against the command buffer tile geometry
    if cb is not None:
        try:
            _check_tiles(classes, cb, violations)
        except Exception as e:  # never let cross-check crash the verifier
            violations.append(f"tile cross-check error (non-fatal): {e}")

    # advisory DRAM-address provenance (parameterized by the harness address model; no-op if unknown)
    if address_model:
        violations += dram_address_findings(trace, address_model)

    return {"status": "pass" if not violations else "fail", "violations": violations}


def _ceil16(x: int) -> int:
    return ((x + 15) // 16) * 16


def _check_tiles(classes: list[str], cb: dict, violations: list[str]) -> None:
    """For a single resident matmul, MVOUT count should equal Mt*Nt over padded dims."""
    tensors = cb.get("tensors", {})
    cmds = cb.get("commands", [])
    matmuls = [c for c in cmds if c.get("opcode") in ("MATMUL_RESIDENT", "MATMUL")]
    if len(matmuls) != 1:
        return  # multi-matmul / movement: skip the simple geometry check
    mm = matmuls[0]
    lhs = tensors.get(mm.get("operands", {}).get("lhs"))
    # resident weight shape lives on the RES_PACK source
    packs = [c for c in cmds if c.get("opcode") == "RES_PACK"]
    if not lhs or not packs:
        return
    wsrc = tensors.get(packs[0].get("operands", {}).get("src"))
    if not wsrc:
        return
    M = lhs["shape"][0]
    N = wsrc["shape"][1]
    Mt, Nt = _ceil16(M) // 16, _ceil16(N) // 16
    # A fused pooling store retains all Mt row tiles as one spatial plane, then issues one MVOUT for
    # each channel tile. Counting one store per compute tile would diagnose the required retained-plane
    # schedule as missing stores (GP1 is Mt=2, Nt=2 but correctly has two, not four, MVOUTs).
    commits = [c for c in cmds if c.get("opcode") == "COMMIT"]
    pooled = len(commits) == 1 and "maxpool" in (
        (commits[0].get("attributes") or {}).get("epilogue") or [])
    exp_mvout = Nt if pooled else Mt * Nt
    got = classes.count("MVOUT")
    if got != exp_mvout:
        basis = "Nt for retained-plane maxpool" if pooled else "Mt*Nt"
        violations.append(f"MVOUT count {got} != expected {basis}={exp_mvout} (M={M},N={N})")
