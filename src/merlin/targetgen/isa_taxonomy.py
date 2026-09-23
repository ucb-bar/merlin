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

from merlin.common.paths import merlin_dir, repo_root

from .rtl.facts import UNKNOWN_KEY  # ONE vocabulary for "could not be derived" (see rtl/facts.py)

_HELPER = Path(__file__).resolve().parent / "oracle_helpers" / "isa_introspect.py"
_CACHE: dict[str, dict] = {}

# --------------------------------------------------------------------------- fail-closed emptiness
# An EMPTY taxonomy used to mean two incompatible things at once: "this target declares no ISA
# definition" (a RoCC/command-ISA target whose classes come from mlc discovery instead — a capability
# it genuinely does not have) and "we could not determine its taxonomy" (no descriptor resolved, the
# model venv was unreachable, the introspect helper crashed or timed out — a MISSING INPUT). Both came
# back as ``{}`` behind a bare ``except``, and every consumer skipped its check identically, so a
# target nobody could examine reported exactly like a target that needs no examination.
#
# The vocabulary is deliberately the one :mod:`merlin.targetgen.rtl.facts` already uses for the RTL
# facts artifact: reasons live under :data:`UNKNOWN_KEY`, OUTSIDE the body, so a record that grounded
# nothing still tests as EMPTY.
NOT_APPLICABLE_KEY = "not_applicable"
STATUS_KEY = "status"
STATUS_DERIVED = "derived"
STATUS_NOT_APPLICABLE = NOT_APPLICABLE_KEY
STATUS_UNKNOWN = UNKNOWN_KEY

#: The keys that actually carry derived content. Anything else in the record is provenance.
_BODY_KEYS = ("by_class", "by_mnemonic", "asm_mnemonics")


class TaxonomyUnknown(RuntimeError):
    """The taxonomy could not be DETERMINED — a missing input, never a statement about the hardware."""


class Taxonomy(dict):
    """A derived ISA taxonomy, or a fail-closed record of why there is none.

    ``bool(tax)`` is True only when something was actually DERIVED: an ungrounded record carries its
    reason but still tests as empty, exactly as ``facts["facts"] == {}`` does in the RTL-facts
    artifact. That is deliberate — a consumer that has not been taught the new state keeps SKIPPING,
    which is honest, instead of screening a target against a taxonomy with zero instruction classes,
    which is a vacuous pass. Consumers that must tell the two apart read :func:`status_of`,
    :func:`is_unknown`, :func:`reasons_of`, or :func:`require_taxonomy`.
    """

    __slots__ = ()

    def __bool__(self) -> bool:  # noqa: D105 — see the class docstring
        return any(self.get(k) for k in _BODY_KEYS)


def _empty_record(status: str, subject: str, why: str) -> Taxonomy:
    """An empty taxonomy that RECORDS why it is empty, in the rtl.facts shape."""
    rec = Taxonomy({k: {} for k in _BODY_KEYS})
    rec[STATUS_KEY] = status
    rec["subject"] = subject
    rec[UNKNOWN_KEY if status == STATUS_UNKNOWN else NOT_APPLICABLE_KEY] = {"taxonomy": why}
    return rec


def _unknown(subject: str, why: str) -> Taxonomy:
    """ "We could not determine this target's taxonomy" — a MISSING INPUT, worth fixing."""
    return _empty_record(STATUS_UNKNOWN, subject, why)


def _not_applicable(subject: str, why: str) -> Taxonomy:
    """ "This target declares no ISA definition" — a capability it does not have, nothing to fix."""
    return _empty_record(STATUS_NOT_APPLICABLE, subject, why)


def status_of(taxonomy: dict | None) -> str:
    """``derived`` | ``not_applicable`` | ``unknown`` for any taxonomy value, including a legacy ``{}``.

    FAIL CLOSED: an emptiness that records NOTHING is ``unknown``, never ``not_applicable``. A bare
    ``{}`` (what every one of these call sites used to return, and what a caller may still hand in)
    makes no claim about the hardware, so it must not be read as one."""
    if not isinstance(taxonomy, dict):
        return STATUS_UNKNOWN
    if any(taxonomy.get(k) for k in _BODY_KEYS):
        return STATUS_DERIVED
    declared = taxonomy.get(STATUS_KEY)
    if declared in (STATUS_DERIVED, STATUS_NOT_APPLICABLE, STATUS_UNKNOWN):
        return STATUS_NOT_APPLICABLE if declared == STATUS_NOT_APPLICABLE else STATUS_UNKNOWN
    return STATUS_UNKNOWN


def is_unknown(taxonomy: dict | None) -> bool:
    """True when the taxonomy could not be DETERMINED (so no absence here is evidence)."""
    return status_of(taxonomy) == STATUS_UNKNOWN


def declares_isa(taxonomy: dict | None) -> bool:
    """True unless the target positively declares NO ISA definition.

    An UNKNOWN record answers True: we do not know that it declares none, and answering False would
    re-create the conflation this record exists to end."""
    return status_of(taxonomy) != STATUS_NOT_APPLICABLE


def reasons_of(taxonomy: dict | None) -> dict[str, str]:
    """The record's own reasons ``{what -> why}``, for either kind of emptiness. Empty when derived."""
    if not isinstance(taxonomy, dict):
        return {}
    for key in (UNKNOWN_KEY, NOT_APPLICABLE_KEY):
        rec = taxonomy.get(key)
        if isinstance(rec, dict) and rec:
            return {str(k): str(v) for k, v in rec.items()}
    return {}


def require_taxonomy(taxonomy: dict | None, subject: str, *, needs: str) -> dict:
    """The taxonomy body, or a clear refusal — for a consumer that must not proceed on an absence.

    Raises :class:`TaxonomyUnknown` when the taxonomy is UNDETERMINED (fix the input) and
    :class:`NotImplementedError` when the target genuinely declares no ISA definition (this generator
    does not apply to that class of target). Mirrors :func:`merlin.targetgen.rtl.facts.decode_body`,
    which makes the same two-way distinction for the RTL facts artifact."""
    status = status_of(taxonomy)
    if status == STATUS_DERIVED:
        return dict(taxonomy)  # type: ignore[arg-type]
    why = "; ".join(f"{k}: {v}" for k, v in sorted(reasons_of(taxonomy).items())) or (
        "the record carries NO reason, which is itself a defect: an empty taxonomy must say why"
    )
    if status == STATUS_NOT_APPLICABLE:
        raise NotImplementedError(
            f"{subject}: declares no ISA definition, so {needs} cannot be derived from a taxonomy — a "
            f"capability this class of target does not have, not a missing input. {why}"
        )
    raise TaxonomyUnknown(
        f"{subject}: the ISA taxonomy could not be DETERMINED, so {needs} is UNKNOWN — this is a "
        f"MISSING INPUT, not a statement that the target has no ISA. {why}"
    )


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


def derive_isa_taxonomy(te, *, model_ext: str | None = None, timeout: int = 120) -> Taxonomy:
    """Introspect the target's ISA definition in the MODEL venv → {by_class, by_mnemonic, asm_mnemonics}.

    Returns a :class:`Taxonomy` that is EMPTY but SELF-DESCRIBING when nothing could be derived, and
    the two kinds of emptiness are kept apart:

    * ``not_applicable`` — the target declares no ``isa_definition.py`` at all (a RoCC/command-ISA
      target whose classes come from mlc discovery instead). Callers fall back, correctly.
    * ``unknown`` — a taxonomy was supposed to exist and we could not get it: a declared ISA file that
      is not on disk, an introspect helper that exited non-zero, timed out, or wrote unreadable JSON.
      An absence here is NOT evidence about the hardware, and a check that skips on it is UNCHECKED.
    """
    subject = f"target {getattr(te, 'target', '?')!r}"
    declared = [str(h) for h in (getattr(te, "isa_headers", []) or [])]
    isa = _isa_def_path(te)
    if isa is None:
        return _not_applicable(
            subject,
            f"the descriptor declares no isa_definition.py among its ISA headers "
            f"({declared or 'none declared'}), so this target has no self-hosted ISA taxonomy to derive",
        )
    if not isa.is_file():
        return _unknown(
            subject,
            f"the descriptor declares the ISA definition {str(isa)!r} but no such file exists, so the "
            f"taxonomy could not be derived (a MISSING INPUT, not a target without an ISA)",
        )
    key = str(isa)
    if key in _CACHE:
        return Taxonomy(copy.deepcopy(_CACHE[key]))
    from merlin.common.paths import ext_path

    from .program_engine_policy import selected_model_venv_python

    mext = model_ext
    if not mext:
        # resolve from the target's capability manifest runner block (same path as capsule_runner)
        try:
            from .target_experiment import load_capability_manifest

            m = load_capability_manifest(te.target)
            mext = (m.contract.get("runner") or {}).get("model_ext") or (m.contract.get("toolchain") or {}).get("model")
        except Exception:  # noqa: BLE001
            mext = None
    import os
    import sys
    import tempfile

    if mext:
        # A target backed by a model package: introspect its ISA definition in that model's own venv
        # (the ISA def imports the model package), cwd = the model project root.
        py = selected_model_venv_python(mext)
        run_cwd = str(ext_path(mext))
        run_env = None
    else:
        # A target that ships a SELF-CONTAINED ISA definition (no model package to import) — introspect it
        # in-process, with the definition's own directory importable so its sibling ``isa_patterns`` module
        # resolves. Target-agnostic: any target shipping a self-contained ISA doc gets the tools without
        # registering a model venv (fail-closed — a bad import just yields the empty taxonomy below).
        py = sys.executable
        run_cwd = str(isa.parent)
        run_env = dict(os.environ)
        run_env["PYTHONPATH"] = str(isa.parent) + os.pathsep + run_env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "taxonomy.json"
        cmd = [str(py), str(_HELPER), "--isa-module", str(isa), "--out", str(out)]
        try:
            p = subprocess.run(cmd, cwd=run_cwd, capture_output=True, text=True, timeout=timeout, env=run_env)
        except subprocess.TimeoutExpired:
            return _unknown(
                subject,
                f"the isa_introspect helper timed out after {timeout}s on "
                f"{key!r}; the taxonomy is UNDETERMINED, not absent",
            )
        except OSError as e:
            # A helper that cannot even be LAUNCHED (no interpreter at that path, no such cwd) used to
            # propagate to a bare except one frame up and come back as "this target has no ISA".
            return _unknown(
                subject,
                f"could not launch the isa_introspect helper for {key!r} "
                f"({type(e).__name__}: {e}); the taxonomy is UNDETERMINED",
            )
        if p.returncode != 0 or not out.is_file():
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
            return _unknown(
                subject,
                f"the isa_introspect helper exited {p.returncode} for {key!r} and wrote "
                f"{'no' if not out.is_file() else 'an'} output; the taxonomy is UNDETERMINED "
                f"(stderr tail: {' | '.join(tail) if tail else 'empty'})",
            )
        try:
            tax = json.loads(out.read_text())
        except ValueError as e:
            return _unknown(
                subject,
                f"the isa_introspect helper wrote unreadable JSON for {key!r} "
                f"({type(e).__name__}: {e}); the taxonomy is UNDETERMINED",
            )
    if not isinstance(tax, dict) or not any(tax.get(k) for k in _BODY_KEYS):
        # The helper SUCCEEDED and grounded nothing. That is still not "this target has no ISA": the
        # target declares one, so an empty result means the introspection did not reach it.
        return _unknown(
            subject,
            f"the isa_introspect helper ran for {key!r} and grounded NOTHING "
            f"(no by_class/by_mnemonic/asm_mnemonics entries)",
        )
    _CACHE[key] = tax
    return Taxonomy(copy.deepcopy(tax))


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
        tok = line.split()[0]  # the mnemonic (e.g. VMATMUL.MXU0 / LI)
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


def roles_of(taxonomy: dict) -> frozenset[str]:
    """Every structural role the target's ISA definition evidences. The role census in set form — the
    input to :func:`families_of`."""
    return frozenset(_classes_by_role(taxonomy))


def families_of(taxonomy: dict) -> frozenset[str]:
    """The canonical semantic families a target's own ISA EVIDENCES, derived from its role census.

    This is how a self-hosted-ISA target — which ships an ``isa_definition.py`` instead of an
    ``encoding.semantic_class`` map — gets a family vocabulary at all. Roles come from the instructions'
    typed operands (:func:`oracle_helpers.isa_introspect._role_for_pattern`), and
    :func:`semantic_families.from_isa_role` pins each to a family, so nothing here reads a mnemonic.

    ⚠️ This is a LOWER BOUND on capability, and one family is structurally invisible to it: a reduction
    and a per-element map compile to the same tensor->tensor instructions, so the census cannot evidence
    ``reduction``. Treat its absence as UNKNOWN, never as "this target cannot reduce"."""
    from . import semantic_families

    return semantic_families.families_from_roles(roles_of(taxonomy))


def families_for_target(target: str, *, timeout: int = 120) -> frozenset[str]:
    """:func:`families_of` for a target by name; empty frozenset when no ISA definition resolves.

    An empty result here is a LOWER BOUND either way, and it does not say which of the two reasons
    produced it. A caller that needs to tell "this target evidences no family" from "we could not
    look" must ask :func:`taxonomy_for_target` and read :func:`status_of` — this signature cannot
    carry the distinction."""
    try:
        return families_of(taxonomy_for_target(target, timeout=timeout))
    except Exception:  # noqa: BLE001 — no derivable ISA -> honestly empty, never guessed
        return frozenset()


def required_classes_for_op(
    taxonomy: dict,
    *,
    op: str = "matmul",
    output_dtype: str | None = None,
    epilogue: tuple[str, ...] = (),
    movement: bool = False,
) -> list[str]:
    """Derive the instruction classes a capsule of this op MUST exercise, selected by DERIVED semantic
    ROLE from the target's own taxonomy (never a hardcoded list). Empty if the taxonomy has no systolic
    (matmul) role — a non-systolic target."""
    return required_classes_from_roles(
        _classes_by_role(taxonomy), op=op, output_dtype=output_dtype, epilogue=epilogue, movement=movement
    )


def required_role_slots(
    *, op: str = "matmul", output_dtype: str | None = None, epilogue: tuple[str, ...] = (), movement: bool = False
) -> list[tuple[str, ...]]:
    """The ordered semantic-role SLOTS a kernel for ``op`` must exercise — each slot a tuple of acceptable
    roles (the first that resolves wins). Pure op->role semantics expressed in the DERIVED role vocabulary:
    no target data and no class names here, so it is fully target-agnostic. Shared by
    :func:`required_classes_from_roles` (which maps each slot to a concrete class from the target's own role
    map) and the static linter (which checks slot presence by ROLE — robust to a target having several
    classes per role). A movement/copy op needs only a memory op; a matmul needs load, stationary-weight
    push, the systolic multiply, and the accumulator read-out, plus any epilogue role."""
    from . import semantic_families as _sf

    family = _sf.from_op(op)
    if movement or family == "movement":
        return [("memory",)]  # dequant load + store, no MXU
    # FAIL CLOSED unless the op's family actually CONTAINS a contraction. This used to fall straight
    # through to the systolic sequence for EVERY op, so a softmax/rmsnorm/gelu capsule was told it must
    # exercise MXUMatMul -- a fabricated requirement, and the direction of error that running it cannot
    # catch (the kernel is simply marked non-conformant). Membership comes from the closed vocabulary's
    # own decomposition, not a list here: `attention` decomposes to (contraction, reduction,
    # elementwise_map), so an attention capsule DOES owe the systolic sequence, while `softmax` and
    # `normalization` decompose to (reduction, elementwise_map) and owe nothing. An op the vocabulary
    # does not recognise owes nothing either -- the caller records that rather than inventing a demand.
    prims = _sf.primitives_of(family or "")
    if not prims:
        return []  # an op the closed vocabulary does not recognise owes nothing (recorded, not invented)
    if "contraction" not in prims:
        # NOT nothing. This used to return an empty list for every non-contraction op, and an empty
        # requirement is satisfied by emitting anything at all -- the same unfalsifiable shape as a
        # coverage expectation written too coarsely. Measured on the atlas corpus: 13 elementwise and
        # reduction capsules owed no instruction whatsoever, so no static check could observe whether
        # the backend had used the machine's vector units or ignored them.
        #
        # What such an op DOES owe is expressed in the same derived role vocabulary: its operands have to
        # be moved, and its arithmetic has to happen on a compute unit. Which compute role is target-
        # dependent (one target files its reductions under the vector-unary role), so both are offered
        # and the first that resolves wins -- and a target declaring neither role adds nothing rather
        # than inventing a demand it cannot meet.
        slots = [("memory",)]
        if "reduction" in prims:
            slots.append(("tensor_compute_unary", "tensor_compute_binary"))
        if "elementwise_map" in prims:
            slots.append(("tensor_compute_binary", "tensor_compute_unary"))
        return slots
    slots: list[tuple[str, ...]] = [
        ("memory",),  # load operands
        ("weight_load",),  # push stationary weight
        ("matmul",),  # systolic multiply-accumulate
    ]
    # readout pop: fp8 output uses the scaled (exponent) pop; else the plain pop — pick by dtype+role.
    if output_dtype and "fp8" in output_dtype:
        slots.append(("acc_readout_scaled", "acc_readout"))
    else:
        slots.append(("acc_readout", "acc_readout_scaled"))
    if "relu" in epilogue:
        slots.append(("tensor_compute_unary",))  # the vector-unary (VRELU) epilogue
    if "bias_add" in epilogue or "bias" in epilogue:
        # A bias epilogue adds a VECTOR to the accumulator, so it is a BINARY tensor op, not the unary
        # one relu resolves to -- offering unary as the fallback would let a target with only a unary
        # role satisfy a two-operand stage with a one-operand instruction.
        #
        # Offered, not demanded: a target that folds the bias into its accumulator read-out has no
        # separate class for it, `required_classes_from_roles` finds no class for the slot, and the slot
        # contributes nothing. That is the derive-or-drop this function is built on -- the alternative
        # is demanding an instruction the datapath does not have, which marks a conformant backend
        # non-conformant, and it is the same fabricated requirement the contraction check above exists
        # to avoid.
        slots.append(("tensor_compute_binary",))
    return slots


def required_classes_from_roles(
    by_role: dict[str, list[str]],
    *,
    op: str = "matmul",
    output_dtype: str | None = None,
    epilogue: tuple[str, ...] = (),
    movement: bool = False,
) -> list[str]:
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


def taxonomy_for_target(target: str, *, timeout: int = 120) -> Taxonomy:
    """Convenience: derive the ISA taxonomy for a target by NAME, resolving its descriptor from the
    standard capsule-bench location. Cached via :func:`derive_isa_taxonomy`.

    Returns the same self-describing :class:`Taxonomy` as :func:`derive_isa_taxonomy`. A target with
    NO DESCRIPTOR is ``unknown``, not ``not_applicable``: nothing was read, so nothing is known about
    whether it has an ISA. Measured 2026-09-21, three of the registered targets were in exactly that
    state while ~10 consumers silently skipped their taxonomy-powered checks on it."""
    from .corpora import descriptor_path
    from .target_experiment import load_target_experiment

    # Through `corpora`, which honors MERLIN_TARGET_EXPERIMENT; the convention path built here by hand
    # read the in-tree descriptor even when a caller had pointed the run at another one.
    subject = f"target {target!r}"
    p = descriptor_path(target)
    if not p.is_file():
        return _unknown(
            subject,
            f"no target descriptor resolves at {str(p)!r}, so nothing was read — "
            f"whether this target has an ISA definition is UNDETERMINED",
        )
    try:
        te = load_target_experiment(p)
    except Exception as e:  # noqa: BLE001 — an unreadable descriptor is a missing input, not a verdict
        return _unknown(
            subject,
            f"the descriptor {str(p)!r} could not be loaded ({type(e).__name__}: {e}); the taxonomy is UNDETERMINED",
        )
    try:
        return derive_isa_taxonomy(te, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — record WHAT failed; never report it as "no ISA"
        return _unknown(
            subject,
            f"deriving the taxonomy from {str(p)!r} raised {type(e).__name__}: {e}; the taxonomy is UNDETERMINED",
        )


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


# --------------------------------------------------------------------------- matrix-extension bridge
# A matrix extension reached through the VECTOR opcode has no shipped ``isa_definition.py`` and no RoCC
# ``encoding`` map, so neither regime above sees it and its class list came out EMPTY — which makes a
# corpus's coverage expectation unfalsifiable (an empty required set is satisfied by emitting nothing).
# Its encodings ARE derived, just by a different reader (:mod:`targetgen.rtl.opu_isa`, cross-checked
# against the unit's own header), so this bridges that derivation into the same role-slot vocabulary the
# taxonomy path uses.
#
# The class NAMES are the derived instruction names — whatever the unit's own RTL calls them — so nothing
# here invents a vocabulary. What IS stated is the alignment between the two role vocabularies we own: the
# microkernel's roles (``matrix_units.yaml``'s ``kernel_roles``, deliberately declared because which
# instruction plays which role is a property of the kernel's structure) and the capsule role slots
# (:func:`required_role_slots`, pure op semantics). A hardware fact would have to be derived; a mapping
# between two of our own declarations is exactly the kind of thing that belongs written down.
KERNEL_ROLE_TO_CAPSULE_ROLE: dict[str, str] = {
    "operand_load": "memory",  # move an operand into the unit's register file
    "broadcast": "weight_load",  # push the stationary operand across the array
    "accumulate": "matmul",  # the multiply-accumulate itself
    "readout": "acc_readout",  # read the accumulator back out
}


def matrix_unit_role_classes(
    unit: str, *, support_target: str, contract_path: "str | Path | None" = None
) -> dict[str, list[str]]:
    """``{capsule role: [derived instruction name]}`` for a declared matrix extension.

    Fails closed, loudly, in three ways, because each silent version of it produces a corpus that cannot
    fail: an undeclared unit, a derivation that is not ``ok`` (a gap or a cross-check disagreement), and a
    ``kernel_roles`` entry naming an instruction the derivation does not contain.
    """
    from merlin.targetgen.plugins import load_declared

    provider = load_declared(support_target, "matrix_lowering")
    uc = provider.load_contract(unit, path=contract_path)
    derivation = provider.derive_encodings(uc)
    if not derivation.ok:
        raise ValueError(
            f"matrix unit {unit!r}: encodings are not fully derived (gaps={list(derivation.gaps)}, "
            f"crosschecks={[c.get('agrees') for c in derivation.crosschecks]}) — refusing to build a "
            f"coverage expectation from an ungrounded derivation"
        )
    out: dict[str, list[str]] = {}
    for kernel_role, insn in sorted(uc.kernel_roles.items()):
        capsule_role = KERNEL_ROLE_TO_CAPSULE_ROLE.get(kernel_role)
        if capsule_role is None:
            continue  # a role the capsule slots do not consume (not an error)
        if insn not in derivation.encodings:
            raise ValueError(
                f"matrix unit {unit!r}: kernel_roles.{kernel_role} names {insn!r}, which the derivation "
                f"does not contain (derived: {sorted(derivation.encodings)}) — the contract and the RTL "
                f"disagree about what this unit implements"
            )
        out.setdefault(capsule_role, []).append(insn)
    return out


def matrix_unit_classes_for(unit: str, *, support_target: str, contract_path: "str | Path | None" = None):
    """A ``classes_for(op=, output_dtype=, epilogue=, movement=)`` callable over a matrix extension's
    derived encodings, shaped exactly like the taxonomy and RoCC regimes so ``CorpusBinding`` cannot tell
    them apart. Raises if the derivation yields no usable role — an empty class list is the failure this
    bridge exists to prevent."""
    by_role = matrix_unit_role_classes(unit, support_target=support_target, contract_path=contract_path)
    if not by_role:
        raise ValueError(
            f"matrix unit {unit!r}: no declared kernel role maps to a capsule role slot "
            f"(kernel roles must cover at least one of {sorted(KERNEL_ROLE_TO_CAPSULE_ROLE)})"
        )

    def _from_matrix_unit(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
        classes = required_classes_from_roles(
            by_role, op=op, output_dtype=output_dtype, epilogue=tuple(epilogue), movement=movement
        )
        if not classes:
            raise ValueError(
                f"matrix unit {unit!r}: op={op!r} resolved to NO instruction classes from roles "
                f"{sorted(by_role)} — a capsule whose coverage expectation is empty cannot fail, so this "
                f"is refused rather than recorded"
            )
        return classes

    return _from_matrix_unit
