"""Bind the closed role vocabulary to each target's OWN derived encoding table.

``merlin/contract/compute_endpoints.yaml`` says WHERE a target's encoding comes from and, for a target
whose derived table uses its own names, which of those names carries which role. This module resolves
that declaration against the live derivation and reports what it could not bind.

The reporting is the point. A role whose declared name is absent from the target's derived table is
recorded as MISSING, never dropped: a rename in the RTL then surfaces as a missing role instead of a
decoder that silently stops recognizing an instruction. That silent-drop failure is the recorded
`rocc_decode` shape, where a too-narrow matcher quietly mis-measured a conformant backend.

Nothing here contains an opcode, a funct value or a field position. Those are derived per target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import merlin_dir
from merlin.kernels import roles as _roles

_SPEC = "contract/compute_endpoints.yaml"


def _spec() -> dict[str, Any]:
    path = merlin_dir() / _SPEC
    if not path.is_file():
        return {"endpoints": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"endpoints": {}}


@dataclass(frozen=True)
class _EndpointStub:
    """Just enough of an Endpoint for the intrinsics reader, which needs only the declaration block.

    Exists so name RESOLUTION can consult the header before the Endpoint that will carry the resolved
    names has been built — the alternative is a two-pass load whose halves can disagree.
    """

    block: dict
    name: str = ""


@dataclass(frozen=True)
class Endpoint:
    """One compute endpoint, with its roles resolved against the target's derived table."""

    name: str
    target: str
    engine: str
    exposure: str
    source: str
    #: role -> the derived names carrying it, as ACTUALLY FOUND in the target's own table.
    roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: role -> names the spec declared that the derived table does NOT contain. Never silently dropped.
    missing: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Derived names carrying no declared role — the other half of the same honesty.
    unmapped: tuple[str, ...] = ()
    levels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    crosscheck: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def roles_of(self, name: str) -> tuple[str, ...]:
        """EVERY role ``name`` carries, in declaration order.

        One instruction can do more than one thing, and collapsing that loses a real check: gemmini's
        MVOUT both drains the accumulator (``readout``) and makes the result architecturally visible
        (``commit``). Recording only the first made a complete contraction report a missing readout --
        and "did the accumulator ever get drained" is precisely the question a past audit failed to ask.
        """
        return tuple(role for role, names in self.roles.items() if name in names)

    def role_of(self, name: str) -> str | None:
        """The first role ``name`` carries (convenience; prefer :meth:`roles_of`)."""
        got = self.roles_of(name)
        return got[0] if got else None

    def engines_evidenced(self) -> frozenset[str]:
        """Facets the roles this endpoint ACTUALLY binds evidence — not what it declares."""
        return frozenset(e for r in self.roles if (e := _roles.engine_of(r)))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "target": self.target, "engine": self.engine,
                "exposure": self.exposure, "source": self.source,
                "roles": {k: list(v) for k, v in sorted(self.roles.items())},
                "missing": {k: list(v) for k, v in sorted(self.missing.items())},
                "unmapped": list(self.unmapped),
                "levels": {k: list(v) for k, v in sorted(self.levels.items())},
                "crosscheck": dict(self.crosscheck)}


def endpoint_names(target: str | None = None) -> tuple[str, ...]:
    eps = _spec().get("endpoints") or {}
    return tuple(sorted(n for n, b in eps.items()
                        if target is None or (b or {}).get("target") == target))


def _derived_names(target: str, block: dict) -> tuple[set[str], str]:
    """Every instruction name the target's own derivation yields, and how it was obtained.

    Returns an EMPTY set when the derivation is unavailable — an honest "cannot check", distinct from
    "the table has no such name". The caller keeps the declared roles in that case rather than reporting
    every one of them missing, because an absent toolchain is not evidence about the hardware.
    """
    enc = block.get("encoding") or {}
    source = str(enc.get("source") or "")
    if source == "rtl_facts":
        from merlin.targetgen.rtl import facts as _F
        body = (_F.load_facts(target) or {}).get("facts") or {}
        table = next((i for i in body.get("interfaces", ())
                      if i.get("name") == enc.get("table")), {})
        return set((table.get(enc.get("names_from") or "names") or {}).values()), "rtl_facts"
    if source == "isa_model":
        from merlin.targetgen import isa_model as _IM
        try:
            model = _IM.isa_model_for_target(target)
        except Exception:  # noqa: BLE001
            return set(), "isa_model(unavailable)"
        # The model's ROLE table is keyed by instruction CLASS, and classes are what a role is declared
        # over. Reading `by_mnemonic` instead compares class names against mnemonics -- two different
        # name spaces -- so every role comes back missing while 127 mnemonics come back unmapped, a
        # report that is wrong in both directions at once.
        return {n for names in (model.roles or {}).values() for n in names}, "isa_model"
    if source == "matrix_units":
        import yaml as _y
        path = merlin_dir() / "contract" / "matrix_units.yaml"
        units = (_y.safe_load(path.read_text(encoding="utf-8")) or {}).get("units") or {}
        unit = units.get(enc.get("unit")) or {}
        return set((unit.get(enc.get("roles_from") or "kernel_roles") or {}).values()), "matrix_units"
    if source == "funct_header":
        # A RoCC funct table declared by the target's own C ISA header, in the `#define k_<NAME> <N>`
        # convention. The same derivation the RTL introspector already uses for a target that ships
        # headers -- reused rather than restated, so the two cannot drift.
        from merlin.common import provenance as _prov
        from merlin.targetgen.rtl.circt_introspect import _functs_from_headers
        try:
            root = Path(_prov.verify(str(enc.get("pin"))).observed.path)
        except (KeyError, OSError, ValueError) as exc:
            # NARROW on purpose. A blanket except here swallowed a NameError from a missing import and
            # reported it as "the pin will not verify" — a wrong diagnosis that sends the reader to
            # the checkout instead of to the code.
            return set(), f"funct_header(pin unresolved: {type(exc).__name__})"
        by_code = _functs_from_headers([root / str(enc.get("path"))])
        return set(by_code.values()), f"funct_header ({len(by_code)} codes)"
    if source == "mnemonic_grammar":
        # A standard ISA has no per-target decode table to verify against: its GRAMMAR is the
        # vocabulary. Returning the declared set means every role binds, and the loader stamps the
        # result UNVERIFIED so nobody mistakes "declared" for "confirmed against silicon". The asm
        # audit then reports every observed mnemonic no role covers, so the gap stays visible.
        return set(), "mnemonic_grammar [declared, not verifiable against a decode table]"
    if source == "isa_encoding":
        from merlin.targetgen.rtl import mlc_bridge as _mb
        got = _mb.isa_encoding_for(target) or {}
        names = set(got.get("opcodes") or {})
        how = "isa_encoding"
        # A target may ALSO declare an intrinsics header, which names the individual instructions
        # inside a custom opcode space. Both are the target's own vocabulary at different resolutions:
        # the decoder tells us which space a word is in, the header tells us which operation.
        intr = (block.get("encoding") or {}).get("intrinsics") or {}
        if intr:
            from merlin.kernels.decode import insn_header as _ih
            table, problems = _ih.table_for(target, _EndpointStub(block))
            if table:
                names |= set(table.values())
                how = f"{how} + intrinsics header"
            if problems:
                how = f"{how} [header problems: {len(problems)}]"
        return names, how
    return set(), f"{source or 'none'}(unsupported)"


def _roles_from_isa_model(target: str) -> dict[str, tuple[str, ...]]:
    """Roles for a self-hosted-ISA target, bridged from its DERIVED IsaModel role table."""
    from merlin.targetgen import isa_model as _IM
    try:
        model = _IM.isa_model_for_target(target)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, list[str]] = {}
    for isa_role, names in (model.roles or {}).items():
        role = _roles.from_isa_role(isa_role)
        if role:
            out.setdefault(role, []).extend(sorted(names))
    return {k: tuple(v) for k, v in out.items()}


def _roles_from_matrix_units(block: dict) -> dict[str, tuple[str, ...]]:
    """Roles already declared under this repo's vocabulary in ``matrix_units.yaml``."""
    import yaml as _y
    enc = block.get("encoding") or {}
    path = merlin_dir() / "contract" / "matrix_units.yaml"
    units = (_y.safe_load(path.read_text(encoding="utf-8")) or {}).get("units") or {}
    declared = (units.get(enc.get("unit")) or {}).get(enc.get("roles_from") or "kernel_roles") or {}
    _roles.check_roles(declared)
    return {str(r): (str(n),) for r, n in declared.items()}


def load_endpoint(name: str) -> Endpoint:
    """Resolve one endpoint's declaration against its target's live derivation."""
    block = ((_spec().get("endpoints") or {}).get(name)) or {}
    if not block:
        raise KeyError(f"no compute endpoint {name!r}; known: {list(endpoint_names())}")
    target = str(block.get("target") or "")
    engine = str(block.get("engine") or "")
    declared = {str(r): tuple(v) for r, v in (block.get("roles") or {}).items()}
    _roles.check_roles(declared)

    enc_source = str((block.get("encoding") or {}).get("source") or "")
    if not declared:
        declared = (_roles_from_isa_model(target) if enc_source == "isa_model"
                    else _roles_from_matrix_units(block) if enc_source == "matrix_units" else {})
        # Two endpoints can share one ISA (an array and a lane engine on the same self-hosted ISA), so
        # a model-derived role table must be split by which engine each role evidences. Roles that
        # evidence NO engine (operand_load, dma, config, sync, commit) belong to both: every endpoint
        # loads operands and moves data, and assigning them to one would make the other look inert.
        if engine:
            declared = {r: v for r, v in declared.items()
                        if _roles.engine_of(r) in (engine, None)}

    names, how = _derived_names(target, block)
    bound: dict[str, tuple[str, ...]] = {}
    missing: dict[str, tuple[str, ...]] = {}
    if not names:
        # Derivation unavailable. Keep the declaration and say so, rather than reporting every role
        # missing -- an absent toolchain is not evidence about the hardware.
        bound = dict(declared)
        how = f"{how} [unverified: no derived table available]"
    else:
        for role, want in declared.items():
            have = tuple(n for n in want if n in names)
            gone = tuple(n for n in want if n not in names)
            if have:
                bound[role] = have
            if gone:
                missing[role] = gone
    claimed = {n for v in bound.values() for n in v}
    return Endpoint(name=name, target=target, engine=engine,
                    exposure=str(block.get("exposure") or ""), source=how,
                    roles=bound, missing=missing,
                    unmapped=tuple(sorted(names - claimed)) if names else (),
                    levels={str(k): tuple(v) for k, v in (block.get("levels") or {}).items()},
                    crosscheck=_crosscheck_with_pairs(block),
                    notes=str(block.get("notes") or ""))


def _crosscheck_with_pairs(block: dict) -> dict:
    """The endpoint's crosscheck block, plus the RTL-name -> header-name pairs its source declares.

    The two vocabularies differ (the array's own name for an instruction versus the macro an expert
    header defines for it), and consumers that look for one spelling in a file written in the other
    find nothing. The correspondence is already declared for the encoding crosscheck, so it is surfaced
    here rather than restated.
    """
    out = dict(block.get("crosscheck") or {})
    enc = block.get("encoding") or {}
    if enc.get("source") == "matrix_units" and enc.get("unit"):
        import yaml as _y
        path = merlin_dir() / "contract" / "matrix_units.yaml"
        try:
            units = (_y.safe_load(path.read_text(encoding="utf-8")) or {}).get("units") or {}
            pairs = ((units.get(enc["unit"]) or {}).get("declarations") or {}).get("crosscheck_pairs")
            if pairs:
                out["pairs"] = dict(pairs)
        except OSError:
            pass
    return out


def endpoints_for(target: str) -> tuple[Endpoint, ...]:
    return tuple(load_endpoint(n) for n in endpoint_names(target))
