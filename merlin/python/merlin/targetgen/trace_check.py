"""Verify a decoded RoCC instruction trace against a capsule's expected coverage.

Given a trace from :mod:`merlin.targetgen.rocc_decode` and a capsule's ``expected`` block, assert:

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


def _classes(trace: dict) -> list[str]:
    return [i["class"] for i in trace.get("instructions", [])]


def _first_index(classes: list[str], target: set[str] | str) -> int | None:
    tgt = {target} if isinstance(target, str) else target
    for i, c in enumerate(classes):
        if c in tgt:
            return i
    return None


def check(trace: dict, expected: dict, cb: dict | None = None) -> dict:
    """Validate ``trace`` against capsule ``expected`` (+ optional command buffer)."""
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
        first_work = _first_index(classes, _COMPUTE | {"MVIN", "MVOUT"})
        if first_work is not None and flush_i > first_work:
            violations.append("FLUSH appears after the first MVIN/MVOUT/COMPUTE")

    def _before(cfg: str, use: set[str], label: str) -> None:
        if cfg in present:
            ci = classes.index(cfg)
            ui = _first_index(classes, use)
            if ui is not None and ci > ui:
                violations.append(f"{cfg} appears after first {label}")

    _before("CONFIG_EX", _COMPUTE | {"PRELOAD"}, "PRELOAD/COMPUTE")
    _before("CONFIG_LD", {"MVIN"}, "MVIN")
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
        if "MVIN" not in present or "MVOUT" not in present:
            violations.append("mode movement declared but trace lacks MVIN/MVOUT")

    # 5. optional cross-validation against the command buffer tile geometry
    if cb is not None:
        try:
            _check_tiles(classes, cb, violations)
        except Exception as e:  # never let cross-check crash the verifier
            violations.append(f"tile cross-check error (non-fatal): {e}")

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
    exp_mvout = Mt * Nt
    got = classes.count("MVOUT")
    if got != exp_mvout:
        violations.append(f"MVOUT count {got} != expected Mt*Nt={exp_mvout} (M={M},N={N})")
