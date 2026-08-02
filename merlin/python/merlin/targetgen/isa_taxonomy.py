"""DERIVE a self-hosted-ISA target's instruction taxonomy + per-op required classes from the repo's ISA
docs — so the capsule corpus's ``expected.instruction_classes`` and the trace-check are DISCOVERED, never
hardcoded (see the atlas-isa-grounding finding). For a self-hosted core, mlc's behavioural role probe is
RoCC-only, so the authoritative taxonomy comes from introspecting the shipped ISA definition
(``isa_definition.py``) + the shipped worked ``example_kernel`` — both curated inputs already in the
target's hwbringup bundle. Nothing here holds an opcode/class table: it all falls out of the model's own
ISA definition (via the model-venv helper :mod:`oracle_helpers.isa_introspect`).
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from merlin.common.paths import merlin_dir, repo_root

_HELPER = Path(__file__).resolve().parent / "oracle_helpers" / "isa_introspect.py"
_CACHE: dict[str, dict] = {}


def _resolve(rel: str) -> Path:
    """Resolve a descriptor-relative path. The ``experiments/…`` bundle-convention paths are
    ``merlin/``-relative; a few refs are repo-root-relative — try merlin/ first, then repo root."""
    for base in (merlin_dir(), repo_root()):
        p = base / rel
        if p.exists():
            return p
    return merlin_dir() / rel


def _isa_def_path(te) -> Path | None:
    """The target's shipped ISA-definition module (``isa_definition.py``), from the descriptor's ISA
    headers, resolved absolute."""
    for h in getattr(te, "isa_headers", []) or []:
        if str(h).endswith("isa_definition.py"):
            return _resolve(str(h))
    return None


def _example_kernels(te) -> list[Path]:
    """The shipped worked example kernels (``<hwbringup>/example_kernel/*.S``) — the reference instruction
    SEQUENCES a real program uses, from which per-op required classes are derived."""
    hw = getattr(te, "hwbringup_set", None)
    if not hw:
        return []
    d = _resolve(str(hw)) / "example_kernel"
    return sorted(d.glob("*.S")) if d.is_dir() else []


def derive_isa_taxonomy(te, *, model_ext: str | None = None, timeout: int = 120) -> dict[str, Any]:
    """Introspect the target's ISA definition in the MODEL venv → {by_class, by_mnemonic, asm_mnemonics}.
    Returns an empty taxonomy (``{}``) when the target ships no ISA definition (e.g. a RoCC/command-ISA
    target whose classes come from mlc discovery instead) — callers fall back to their existing path."""
    isa = _isa_def_path(te)
    if isa is None or not isa.is_file():
        return {}
    key = str(isa)
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key])
    from .program_oracle import _model_venv_python  # reuse the model-venv resolver
    from merlin.common.paths import ext_path
    mext = model_ext
    if not mext:
        # resolve from the target's capability manifest runner block (same path as capsule_runner)
        try:
            from .target_experiment import load_capability_manifest
            m = load_capability_manifest(te.target)
            mext = (m.contract.get("runner") or {}).get("model_ext") \
                or (m.contract.get("toolchain") or {}).get("model")
        except Exception:  # noqa: BLE001
            mext = None
    if not mext:
        return {}
    py = _model_venv_python(mext)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "taxonomy.json"
        cmd = [str(py), str(_HELPER), "--isa-module", str(isa), "--out", str(out)]
        p = subprocess.run(cmd, cwd=str(ext_path(mext)), capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0 or not out.is_file():
            return {}
        tax = json.loads(out.read_text())
    _CACHE[key] = tax
    return copy.deepcopy(tax)


def classes_from_kernel(kernel_text: str, taxonomy: dict) -> list[str]:
    """Map an example kernel's instruction lines → their SEMANTIC classes (ordered, deduped) using the
    derived taxonomy's assembler-mnemonic map. This is how a capsule's required instruction classes are
    derived from the shipped worked example (the real programming sequence), rather than hardcoded."""
    asm = taxonomy.get("asm_mnemonics", {}) or {}
    by_mnem = taxonomy.get("by_mnemonic", {}) or {}
    seen, out = set(), []
    for raw in kernel_text.splitlines():
        line = raw.split("#", 1)[0].split("//", 1)[0].strip()
        if not line or line.startswith("."):
            continue
        tok = line.split()[0]                         # the mnemonic (e.g. VMATMUL.MXU0 / LI)
        cls = asm.get(tok) or asm.get(tok.lower()) or asm.get(tok.upper())
        sem = by_mnem.get(cls, {}).get("class") if cls else None
        if sem and sem not in seen:
            seen.add(sem)
            out.append(sem)
    return out


# The instruction classes a capsule must exercise are selected by the DERIVED semantic ROLE each class
# carries (isa_introspect attaches a structural role to every class from its operand datapath), never by
# a hardcoded pattern name. A matmul exercises the systolic datapath (weight push -> matmul -> acc
# readout) plus operand load; movement is a tensor load/store copy; a relu epilogue adds the vector unit.


def _classes_by_role(taxonomy: dict) -> dict[str, list[str]]:
    """{role: [semantic class names present with that role]} from the derived taxonomy — the target's own
    structural roles, in discovery order. Empty for a taxonomy whose entries carry no role."""
    out: dict[str, list[str]] = {}
    for cls, ents in (taxonomy.get("by_class") or {}).items():
        role = next((e.get("role") for e in ents if e.get("role")), None)
        if role:
            out.setdefault(role, [])
            if cls not in out[role]:
                out[role].append(cls)
    return out


def required_classes_for_op(taxonomy: dict, *, op: str = "matmul", output_dtype: str | None = None,
                            epilogue: tuple[str, ...] = (), movement: bool = False) -> list[str]:
    """Derive the instruction classes a capsule of this op MUST exercise, selected by DERIVED semantic
    ROLE from the target's own taxonomy (never a hardcoded list). Empty if the taxonomy has no systolic
    (matmul) role — a non-systolic target."""
    return required_classes_from_roles(_classes_by_role(taxonomy), op=op, output_dtype=output_dtype,
                                       epilogue=epilogue, movement=movement)


def required_role_slots(*, op: str = "matmul", output_dtype: str | None = None,
                        epilogue: tuple[str, ...] = (), movement: bool = False) -> list[tuple[str, ...]]:
    """The ordered semantic-role SLOTS a kernel for ``op`` must exercise — each slot a tuple of acceptable
    roles (the first that resolves wins). Pure op->role semantics expressed in the DERIVED role vocabulary:
    no target data and no class names here, so it is fully target-agnostic. Shared by
    :func:`required_classes_from_roles` (which maps each slot to a concrete class from the target's own role
    map) and the static linter (which checks slot presence by ROLE — robust to a target having several
    classes per role). A movement/copy op needs only a memory op; a matmul needs load, stationary-weight
    push, the systolic multiply, and the accumulator read-out, plus any epilogue role."""
    if movement or op in ("movement", "copy"):
        return [("memory",)]                              # dequant load + store, no MXU
    slots: list[tuple[str, ...]] = [
        ("memory",),                                      # load operands
        ("weight_load",),                                 # push stationary weight
        ("matmul",),                                      # systolic multiply-accumulate
    ]
    # readout pop: fp8 output uses the scaled (exponent) pop; else the plain pop — pick by dtype+role.
    if output_dtype and "fp8" in output_dtype:
        slots.append(("acc_readout_scaled", "acc_readout"))
    else:
        slots.append(("acc_readout", "acc_readout_scaled"))
    if "relu" in epilogue:
        slots.append(("tensor_compute_unary",))           # the vector-unary (VRELU) epilogue
    return slots


def required_classes_from_roles(by_role: dict[str, list[str]], *, op: str = "matmul",
                                output_dtype: str | None = None, epilogue: tuple[str, ...] = (),
                                movement: bool = False) -> list[str]:
    """The role-selected required classes, taking the ``{role: [classes]}`` map directly — so a caller that
    already holds the derived roles (e.g. an :class:`~merlin.targetgen.isa_model.IsaModel`) reuses the exact
    same selection logic without re-deriving the whole taxonomy. Each :func:`required_role_slots` slot maps
    to the first present class of the first role that resolves."""
    req: list[str] = []
    for slot in required_role_slots(op=op, output_dtype=output_dtype, epilogue=epilogue, movement=movement):
        for r in slot:
            cs = by_role.get(r) or []
            if cs and cs[0] not in req:
                req.append(cs[0])
                break
    return req


def taxonomy_for_target(target: str, *, timeout: int = 120) -> dict[str, Any]:
    """Convenience: derive the ISA taxonomy for a target by NAME, resolving its descriptor from the
    standard capsule-bench location. Returns {} if the target ships no descriptor / ISA definition
    (callers then skip the taxonomy-powered checks). Cached via :func:`derive_isa_taxonomy`."""
    from .target_experiment import load_target_experiment
    p = merlin_dir() / "experiments" / "capsule_bench" / "targets" / target / "target_experiment.yaml"
    if not p.is_file():
        return {}
    try:
        return derive_isa_taxonomy(load_target_experiment(p), timeout=timeout)
    except Exception:  # noqa: BLE001 — model venv / ISA def absent -> no taxonomy, caller falls back
        return {}


def classify(word: int, taxonomy: dict) -> list[tuple[str, int]]:
    """Classify one emitted word → list of (class, fixed_mask) for each matching op decode-signature. The
    fixed_mask lets a caller isolate the OPERAND payload (``word & ~fixed_mask``) for field-sanity checks
    (e.g. a memory op whose address operand is all-zero). Usually one match; a list surfaces ambiguity."""
    out: list[tuple[str, int]] = []
    for ent in (taxonomy.get("by_mnemonic") or {}).values():
        m, v, cls = ent.get("fixed_mask"), ent.get("fixed_value"), ent.get("class")
        if m is None or v is None or not cls:
            continue
        if (word & m) == v and (cls, m) not in out:
            out.append((cls, m))
    return out


def role_classes(taxonomy: dict) -> dict[str, str | None]:
    """The tile-producing COMPUTE class and the MEMORY (load/store) class for structural checks, selected
    by DERIVED semantic role from the target's own taxonomy (the ``matmul`` and ``memory`` roles). None
    when the target has no such role (the corresponding tiling / field-sanity check is then skipped,
    honestly) — so this is "derive from this target's roles, else drop", never a per-target hardcode."""
    by_role = _classes_by_role(taxonomy)
    compute = (by_role.get("matmul") or [None])[0]
    memory = (by_role.get("memory") or [None])[0]
    return {"compute": compute, "memory": memory}


def decode_word(word: int, taxonomy: dict) -> list[str]:
    """Classify one emitted instruction word into its semantic class(es) using the DERIVED per-op decode
    signatures (fixed_mask/fixed_value from the ISA def's own encoder). Returns the matching classes
    (usually exactly one; a list so an ambiguous/overlapping encoding surfaces rather than hides). Empty
    if no op matches — i.e. the word decodes to nothing the ISA defines (an illegal/garbage instruction)."""
    hits: list[str] = []
    for ent in (taxonomy.get("by_mnemonic") or {}).values():
        m, v = ent.get("fixed_mask"), ent.get("fixed_value")
        if m is None or v is None:
            continue
        if (word & m) == v:
            cls = ent.get("class")
            if cls and cls not in hits:
                hits.append(cls)
    return hits


def clear_cache() -> None:
    _CACHE.clear()
