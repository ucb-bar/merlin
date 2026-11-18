"""Compile a target's DECLARED compiler obligations into runnable lit/FileCheck pass tests.

The sibling of :mod:`merlin.development.targetgen.checks.rtl_check_compiler`, and deliberately the same shape: derive
every literal from the target's own sources, **omit a check that cannot be grounded rather than
defaulting it**, and carry per-check provenance so a reader can see which checks are evidence and
which are convention.

The input is a triple, all three of which already exist and are derived:

* the capability manifest's ``compiler_obligations`` — what THIS target's compiler must do. Measured
  2026-09-04: this field is required by the manifest validator and had no consumer anywhere in the
  repo. These tests are its first one.
* the conformance cell / tile geometry — the shape class the obligation is quantified over.
* the pass catalog's obligation vocabulary — which pass owes the obligation.

**What is target-specific here, stated precisely.** The obligations demanded and the geometry
constants are the target's own. The module the checks run over is on the target-INDEPENDENT
``interface`` plane, because that plane's op names are defined in-tree and therefore have a
derivation source. Checking a generated target dialect's invented mnemonics was tried, corroborated
over 383 runs, and removed; the emitted instruction stream is covered by the RTL check compiler
instead. So this layer checks "did the compiler discharge the obligation structurally", and the
decoded-stream layer checks "did it emit legal instructions for this ISA".
"""

from __future__ import annotations

import argparse
import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- lit_check_compiler ------------------------------------------------------------------------

#: The interface-plane op vocabulary the checks refer to. In-tree, bare namespace, fixed.
_PACK = "interface.resident_pack"
_MATMUL = "interface.matmul"
_COMMIT = "interface.commit"
_EVICT = "interface.resident_evict"
_PLACE = "schedule.place"


@dataclass
class Check:
    """One emitted check family, with the evidence that grounds it."""

    obligation: str
    lines: list[str]
    grounded_by: str
    derived: bool
    prefix: str = "CHECK"


@dataclass
class Omission:
    """A check that was NOT emitted, and why. Reported; never silently dropped."""

    obligation: str
    reason: str


@dataclass
class Compiled:
    target: str
    checks: list[Check] = field(default_factory=list)
    omissions: list[Omission] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "obligations_declared": len(self.checks) + len(self.omissions),
            "emitted": len(self.checks),
            "omitted": len(self.omissions),
            "omission_reasons": [{"obligation": o.obligation, "reason": o.reason} for o in self.omissions],
            "checks": [
                {
                    "obligation": c.obligation,
                    "grounded_by": c.grounded_by,
                    "derived": c.derived,
                    "n_lines": len(c.lines),
                }
                for c in self.checks
            ],
        }


@functools.lru_cache(maxsize=None)
def _manifest(target: str) -> dict[str, Any] | None:
    from merlin.target.capabilities import manifest as capability_manifests

    try:
        return capability_manifests.manifest_for(target)
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _facts(target: str) -> dict[str, Any]:
    """Derived RTL facts as ``{field: {value, derived, source}}``; empty on any failure.

    Memoized: fact derivation shells out to the RTL introspection bridge, which is slow enough that
    re-deriving per obligation (six targets x several obligations) dominated generation time. The
    bridge is already content-addressed, so caching per process changes no result.
    """
    try:
        from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_1
        from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

        return _merlin_mlc_discovery_1.target_fact_bundle(target).get("fields", {}) or {}
    except Exception:
        return {}


def _grounded(facts: dict[str, Any], name: str):
    """A fact's value only when it is actually DERIVED. A present-but-underived field is not a fact."""
    rec = facts.get(name)
    if not isinstance(rec, dict) or not rec.get("derived"):
        return None
    return rec.get("value")


# --- one builder per obligation --------------------------------------------------------------------
# Each returns a Check, or an Omission carrying the reason it could not be grounded.


def _tile_to_mesh(target: str, facts: dict[str, Any], shape: tuple[int, int, int]):
    """Tiling to the mesh is NOT checkable on the interface plane, and the reason matters.

    Measured 2026-09-04: `interface.accumulate` has no producer anywhere in the in-tree lowering, and
    there is no K-splitting in any staged pass. The tiling this obligation demands is emitted by the
    generated out-of-tree backend, which differs per run — so the interface plane simply does not
    contain the structure to assert a tile count over, whether or not the mesh edge is derivable.

    The mesh edge is still recorded as evidence when derived, because the obligation becomes
    checkable here the moment tiling moves in-tree, and it is already checkable TODAY on the decoded
    instruction stream (rtl_check_compiler asserts ceil(M/DIM)*ceil(N/DIM) mvouts there).
    """
    dim = _grounded(facts, "mesh_dim")
    if dim in (None, 0):
        return Omission(
            "must_tile_to_mesh_shape",
            "the mesh edge is not derivable from this target's RTL facts (empty facts "
            "block or no extracted mesh), AND in-tree lowering does not tile; a software "
            "default is not a hardware boundary, so nothing is asserted",
        )
    return Omission(
        "must_tile_to_mesh_shape",
        f"mesh edge IS derived (mesh_dim={dim}) but in-tree lowering does not tile: "
        "no staged pass splits K and interface.accumulate has no producer. The tiling is "
        "emitted by the out-of-tree backend and is checked on the decoded instruction "
        "stream instead",
    )


def _commit_before_reuse(target: str, facts: dict[str, Any], shape):
    # A grammar invariant, not a hardware fact: the interface grammar admits exactly one commit per
    # committed tensor. Emitted for every target that declares the obligation, and marked underived
    # so nobody cites it as RTL evidence.
    return Check(
        "must_commit_accumulator_before_reuse",
        [f"// CHECK: {_MATMUL}", f"// CHECK: {_COMMIT}", f"// CHECK-NOT: {_COMMIT}", f"// CHECK: {_MATMUL}"],
        grounded_by="merlin_iface grammar (one commit per committed tensor)",
        derived=False,
    )


def _rhs_immutable_residency(target: str, facts: dict[str, Any], shape):
    """Pack once, use, evict -- and NOTHING may use the weight after the evict.

    The trailing ``CHECK-NOT`` is the half that makes this a lifetime check rather than a presence
    check. Without it the sequence is satisfied by a program that evicts the resident weight and then
    keeps matmul-ing against it: a use-after-evict, which is the exact defect the obligation names.
    Measured 2026-09-04: the generated form lacking this line PASSED the ``evict_before_last_use``
    mutation while the hand-written check in tests/data/lit/core/ caught it, i.e. the derived
    check was strictly weaker than the one it was modelled on.
    """
    return Check(
        "must_prove_rhs_immutable_for_residency",
        [
            f"// CHECK: {_PACK}",
            f"// CHECK-NOT: {_PACK}",
            f"// CHECK: {_MATMUL}",
            f"// CHECK: {_EVICT}",
            f"// CHECK-NOT: {_MATMUL}",
        ],
        grounded_by="merlin_iface grammar (pack once, use, evict after LAST use)",
        derived=False,
    )


def _block_scales(target: str, facts: dict[str, Any], shape):
    quantum = None
    try:
        from merlin.development.targetgen.corpus import binding as corpus_spec

        quantum = getattr(corpus_spec, "shape_quantum", None)
    except Exception:
        quantum = None
    return Omission(
        "must_supply_e8m0_block_scales",
        "the block-scale group size is not derivable from this target's facts "
        "(scale-group quantum unavailable), so no per-group scale-operand count is "
        "asserted"
        if quantum is None
        else "scale operands are not representable on the interface plane; this obligation "
        "is checked on the decoded stream instead",
    )


def _map_to_warps(target: str, facts: dict[str, Any], shape: tuple[int, int, int]):
    """Placement onto warps needs an ARITY to assert, not just a unit kind.

    A manifest can declare a unit whose kind is SIMT without stating how many lanes or warps it has.
    Presence of a placement op is then not evidence of anything — the residency obligation already
    forces one — so asserting it would manufacture coverage. The reason distinguishes the two cases,
    because "the manifest says nothing" and "the manifest says the kind but not the arity" are
    different gaps with different fixes.
    """
    manifest = _manifest(target) or {}
    units = manifest.get("compute_units") or []
    geometry = None
    kinds = set()
    for unit in units:
        kind = unit.get("kind")
        if kind:
            kinds.add(str(kind))
        for key in ("warp_size", "warps", "lanes", "n_lanes", "threads_per_warp"):
            if key in unit:
                geometry = (key, unit[key])
                break
    if geometry is None:
        detail = (
            f"unit kind(s) {sorted(kinds)} ARE declared but no lane/warp arity is"
            if kinds
            else "no compute-unit geometry is"
        )
        return Omission(
            "must_map_to_warps",
            f"{detail} declared in the capability manifest, so there is no arity to "
            "assert; checking merely that a placement op exists would manufacture "
            "coverage, since the residency obligation already forces one",
        )
    return Check(
        "must_map_to_warps",
        [f"// CHECK: {_PLACE}"],
        grounded_by=f"{geometry[0]}={geometry[1]} (capability manifest)",
        derived=True,
    )


def _scratchpad_capacity(target: str, facts: dict[str, Any], shape):
    return Omission(
        "must_respect_scratchpad_capacity",
        "a numeric bound, not a structural property: FileCheck cannot express it. "
        "It is checked by the RTL numeric screen (rtl_checks), not here",
    )


_BUILDERS = {
    "must_tile_to_mesh_shape": _tile_to_mesh,
    "must_commit_accumulator_before_reuse": _commit_before_reuse,
    "must_prove_rhs_immutable_for_residency": _rhs_immutable_residency,
    "must_supply_e8m0_block_scales": _block_scales,
    "must_map_to_warps": _map_to_warps,
    "must_respect_scratchpad_capacity": _scratchpad_capacity,
}


def compile_checks(target: str, *, shape: tuple[int, int, int] = (16, 16, 16)) -> Compiled:
    """Compile every obligation this target declares into checks or recorded omissions."""
    out = Compiled(target=target)
    manifest = _manifest(target)
    if manifest is None:
        out.omissions.append(
            Omission(
                "<all>", "no capability manifest for this target (fails closed); nothing to derive obligations from"
            )
        )
        return out
    facts = _facts(target)
    out.facts = {
        k: {"derived": bool(v.get("derived")), "value": v.get("value")} for k, v in facts.items() if isinstance(v, dict)
    }

    obligations = list(manifest.get("compiler_obligations") or [])
    if not obligations:
        out.omissions.append(Omission("<all>", "the capability manifest declares no compiler_obligations"))
        return out

    for ob in obligations:
        builder = _BUILDERS.get(ob)
        if builder is None:
            out.omissions.append(
                Omission(
                    ob,
                    "no check builder knows how to express this "
                    "obligation structurally; recorded rather than "
                    "silently skipped",
                )
            )
            continue
        result = builder(target, facts, shape)
        (out.checks if isinstance(result, Check) else out.omissions).append(result)
    return out


# ---- lit_suite ---------------------------------------------------------------------------------
# Emit the derived per-target lit suite, plus the coverage ledger that says what it could not derive.
#
# Generated output, so it lands under the ``out/`` root (never tracked): a target-named ``.mlir`` in the
# source tree would go stale against the facts cache and is exactly the overfit the repo's register
# exists to shrink. The tracked seed suite in ``tests/data/lit`` is target-INDEPENDENT and stays
# hand-authored.
#
# ``coverage.json`` is the deliverable for a target that can derive nothing. For a target with no
# capability manifest, "0 emitted, here is why" is the honest result and is more useful than an empty
# directory.
#
# CLI::
#
#     python -m merlin.development.targetgen.checks.lit_suite --target <t> --write
#     python -m merlin.development.targetgen.checks.lit_suite --target <first> --target <second>

#: Shape the generated module is built at when the target's tile edge is not derivable.
_FALLBACK_SHAPE = (16, 16, 16)


def _shape_for(compiled: Compiled) -> tuple[tuple[int, int, int], str]:
    """The shape to build at, and the SOURCE of that shape.

    Returned together because a shape whose provenance is unknown is not evidence — and the derived
    edge can coincide with the fallback, so equality is not a usable test of where it came from.
    """
    rec = compiled.facts.get("mesh_dim") or {}
    dim = rec.get("value") if rec.get("derived") else None
    if not dim:
        return _FALLBACK_SHAPE, "fallback: tile edge not derivable from RTL facts"
    d = int(dim)
    return (d, d, d), f"mesh_dim={d} (derived from RTL facts)"


def _interface_text(shape: tuple[int, int, int], reuse: int = 2) -> str:
    """The schedule-plane module the generated test feeds to merlin-opt."""
    from merlin.compiler.dialects import xdsl_context as _common
    from merlin.compiler.dialects.lowering.analyses import lower_to_contract
    from merlin.compiler.dialects.lowering.interface_stage import lower_to_schedule
    from merlin.compiler.dialects.lowering.pipeline import build_input_module, load_curated_contract

    m, k, n = shape
    tc = load_curated_contract("toy_npu")
    mod = build_input_module(reuse=reuse, m=m, k=k, n=n)
    return _common.text(lower_to_schedule(lower_to_contract(mod, tc)))


def render_test(compiled: Compiled, shape: tuple[int, int, int]) -> str:
    """A runnable lit test: ONE RUN line per obligation, each with its own check prefix.

    FileCheck applies directives sequentially over the whole file, so several obligations sharing the
    default CHECK prefix would be read as one ordered sequence and fail spuriously. A prefix per
    obligation is the standard MLIR idiom and keeps each obligation independently attributable — a
    failure names the obligation it broke.
    """
    head = [
        "// GENERATED by merlin.development.targetgen.checks.lit_suite -- do not edit; regenerate.",
        f"// target: {compiled.target}   shape: {shape[0]}x{shape[1]}x{shape[2]}",
        "//",
        "// Each RUN line below discharges one obligation this target's capability manifest",
        "// DECLARES. Checks run on the target-independent `interface` plane, whose op names are",
        "// defined in-tree; a generated target dialect's invented mnemonics are never matched.",
        "",
    ]
    for i, c in enumerate(compiled.checks):
        prefix = f"OB{i}"
        head.append(f"// RUN: %merlin-opt %s -p merlin-materialize-interface | %filecheck %s --check-prefix={prefix}")
    head.append("")
    for i, c in enumerate(compiled.checks):
        prefix = f"OB{i}"
        head.append(f"// obligation: {c.obligation}")
        head.append(f"// grounded by: {c.grounded_by} (derived={c.derived})")
        for line in c.lines:
            head.append(line.replace("// CHECK", f"// {prefix}"))
        head.append("")
    for o in compiled.omissions:
        head.append(f"// OMITTED {o.obligation}: {o.reason}")
    head.append("")
    return "\n".join(head) + _interface_text(shape)


def emit(target: str, *, write: bool = False) -> dict[str, Any]:
    compiled = compile_checks(target)
    shape, shape_source = _shape_for(compiled)
    coverage = dict(compiled.coverage)
    coverage["shape"] = {"m": shape[0], "k": shape[1], "n": shape[2]}
    coverage["shape_source"] = shape_source

    body = render_test(compiled, shape) if compiled.checks else ""
    coverage["test_emitted"] = bool(body)
    if not body:
        coverage["test_omitted_reason"] = (
            "no obligation could be expressed structurally for this target; see omission_reasons"
        )

    if write:
        from merlin.common.artifacts import artifacts_dir

        out = Path(artifacts_dir()) / "verify" / "lit" / target
        out.mkdir(parents=True, exist_ok=True)
        (out / "coverage.json").write_text(json.dumps(coverage, indent=1), encoding="utf-8")
        test_path = out / "obligations.mlir"
        if body:
            test_path.write_text(body, encoding="utf-8")
        elif test_path.exists():
            # A target that used to emit checks and now cannot must not leave a stale test behind
            # passing on last run's evidence.
            test_path.unlink()
        coverage["written_to"] = str(out)
        _install_lit_cfg(out.parent)
    return coverage


def _install_lit_cfg(root: Path) -> None:
    """Give the generated tree its own lit config, copied from the tracked one.

    The tracked config locates the repo by walking up for ``pyproject.toml``, and the generated tree
    lives inside the repo, so the copy works unchanged — no second implementation to drift.
    """
    from merlin.common.paths import repo_root

    src = repo_root() / "tests" / "data" / "lit" / "lit.cfg.py"
    if not src.is_file():
        return
    root.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8").replace(
        'config.name = "merlin-pass-tests"', 'config.name = "merlin-derived-pass-tests"'
    )
    (root / "lit.cfg.py").write_text(text, encoding="utf-8")


#: The pass the derived checks run, and the requirement class they discharge. Named explicitly: a
#: verdict attributed to the wrong pass is worse than no verdict.
CHECKED_PASS = "merlin-materialize-interface"
CHECKED_CLASS = "host-seam"


def record_verdicts(target: str, compiled, *, lit_passed: bool | None) -> None:
    """Write this target's obligations to the shared verification log the obligation gate reads.

    Three outcomes, and the middle one is why the ledger exists:

    ``verified``    a check was emitted AND the suite ran green
    ``unmeasured``  a check was emitted but the suite has not run — emitting is not verifying
    ``abstracted``  the obligation could not be expressed at all; the recorded reason travels with it

    A no-op unless ``MERLIN_VERIFY_LOG`` is set, exactly like invocation recording.
    """
    try:
        from merlin.compiler.dialects.lowering import passes as P
    except Exception:
        return
    for check in compiled.checks:
        if lit_passed is None:
            verdict = "unmeasured"
        else:
            verdict = "verified" if lit_passed else "refuted"
        try:
            P.record_verification(
                CHECKED_PASS,
                requirement_class=CHECKED_CLASS,
                method="filecheck",
                verdict=verdict,
                target=target,
                evidence={"obligation": check.obligation, "grounded_by": check.grounded_by, "derived": check.derived},
                provenance={"source": "merlin.development.targetgen.checks.lit_suite"},
            )
        except Exception:
            pass
    for omission in compiled.omissions:
        try:
            P.record_verification(
                CHECKED_PASS,
                requirement_class=CHECKED_CLASS,
                method="filecheck",
                verdict="abstracted",
                target=target,
                evidence={"obligation": omission.obligation, "reason": omission.reason},
                provenance={"source": "merlin.development.targetgen.checks.lit_suite"},
            )
        except Exception:
            pass


def core_suite_dir() -> Path:
    """The hand-written core lit suite (tracked), as opposed to the derived per-target one."""
    from merlin.common.paths import repo_root

    return repo_root() / "tests" / "data" / "lit" / "core"


def passes_exercised_by(path: Path) -> list[str]:
    """The pass names a lit file's RUN lines drive, read from the file itself.

    Parsed STRUCTURALLY (token walk, no regex, no hardcoded pass list) so the mapping cannot drift
    from what the test actually runs: a file that stops exercising a pass stops crediting it. A RUN
    line with no ``-p`` drives no catalogued pass and contributes nothing.
    """
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        head, sep, rest = line.partition("RUN:")
        if not sep or not head.strip().startswith("//"):
            continue
        toks = rest.split()
        for i, tok in enumerate(toks):
            if tok == "-p" and i + 1 < len(toks):
                out.append(toks[i + 1].strip("\"'"))
            elif tok.startswith("-p="):
                out.append(tok[3:].strip("\"'"))
    return out


def record_core_verdicts(*, lit_passed: bool | None) -> dict[str, str]:
    """Record a static verdict for every pass the CORE suite exercises, production ones included.

    The derived per-target suite can only ever credit ``merlin-materialize-interface`` — a
    PROTOTYPE-catalog pass — so the obligation gate rightly refused to count it toward production and
    reported ``0 / 4 verified``. The core suite drives production passes too, and nothing was reading
    it. This closes that: the pass under test comes from the RUN line, and its requirement class from
    the catalog, so neither is asserted here.

    ``None`` for ``lit_passed`` means the suite did not run — recorded as ``unmeasured``, never as a
    pass. A no-op unless ``MERLIN_VERIFY_LOG`` is set.
    """
    from merlin.compiler.dialects.lowering import passes as P

    verdict = "unmeasured" if lit_passed is None else ("verified" if lit_passed else "refuted")
    by_name = {i.name: i for i in P.all_catalogs()}
    out: dict[str, str] = {}
    root = core_suite_dir()
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.mlir")):
        for name in passes_exercised_by(path):
            info = by_name.get(name)
            if info is None:
                # A RUN line naming a pass the catalog does not define is a broken test, not a
                # verdict; surfaced rather than silently skipped.
                out[name] = "unknown-pass"
                continue
            out[name] = verdict
            try:
                P.record_verification(
                    name,
                    requirement_class=info.obligation,
                    method="filecheck",
                    verdict=verdict,
                    evidence={"check_file": path.name, "suite": "core"},
                    provenance={"source": "merlin.development.targetgen.checks.lit_suite.record_core_verdicts"},
                )
            except Exception:
                pass
    return out


def run_core_suite() -> bool | None:
    """Run the tracked core lit suite. ``None`` when llvm-lit is unavailable."""
    import subprocess

    from merlin.verification.translation.smt_encoding import find_lit

    lit = find_lit()
    root = core_suite_dir()
    if lit is None or not root.is_dir():
        return None
    r = subprocess.run([lit, "-s", str(root)], capture_output=True, text=True, timeout=600)
    return r.returncode == 0


def run_suite(target: str) -> bool | None:
    """Run the derived suite for one target through llvm-lit. ``None`` when it cannot run."""
    import subprocess

    from merlin.common.artifacts import artifacts_dir
    from merlin.verification.translation.smt_encoding import find_lit

    lit = find_lit()
    root = Path(artifacts_dir()) / "verify" / "lit" / target
    if lit is None or not (root / "obligations.mlir").is_file():
        return None
    r = subprocess.run([lit, "-s", str(root)], capture_output=True, text=True, timeout=600)
    return r.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--target", action="append", required=True, help="explicit target; repeat for multiple targets")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--run", action="store_true", help="run the emitted suite through llvm-lit and record the verdicts")
    args = ap.parse_args(argv)

    records = []
    for t in args.target:
        rec = emit(t, write=args.write)
        if args.run:
            passed = run_suite(t) if args.write else None
            rec["suite_passed"] = passed
            record_verdicts(t, compile_checks(t), lit_passed=passed)
        records.append(rec)
    if args.json:
        print(json.dumps(records, indent=1))
    else:
        print(f"{'target':16s} {'declared':>8} {'emitted':>8} {'omitted':>8}  test")
        for r in records:
            print(
                f"{r['target']:16s} {r['obligations_declared']:>8} {r['emitted']:>8} "
                f"{r['omitted']:>8}  {'yes' if r['test_emitted'] else 'no'}"
            )
        total_d = sum(r["obligations_declared"] for r in records)
        total_e = sum(r["emitted"] for r in records)
        print(
            f"\n{total_e} of {total_d} declared obligations are now checked "
            f"({total_d - total_e} omitted with recorded reasons)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
