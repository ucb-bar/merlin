"""Host-owned Phase 0 golden_cache implementation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from merlin.development.targetgen.corpus import binding as CS  # noqa: E402
from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1
from merlin.development.targetgen.corpus.materialization import write_description
from merlin.development.targetgen.grading import golden as CG  # noqa: E402
from merlin.development.targetgen.grading import golden_inputs as NF  # noqa: E402
from merlin.development.targetgen.grading import golden_store as GS  # noqa: E402
from merlin.development.testgen.generation.goldens import (
    _float_golden,
    _simt_golden,
    float_semantics,
    specir_oracle_source_identity,
)
from merlin.development.testgen.generation.microscaling_goldens import (
    _mx_attention_golden,
    _mx_gemv_batched_golden,
    _mx_golden,
)
from merlin.development.testgen.generation.provenance import _document_digest
from merlin.development.testgen.sealed_capture.generation import capture_source

# ---- golden_cache ------------------------------------------------------------------------------

#: MEMOIZE THE GOLDEN ENGINES. A golden is DERIVED and fully deterministic -- operands come from a
#: name-salted fill with no RNG -- so identical inputs always produce an identical answer and
#: recomputing them is pure waste. The engines are deliberately slow: `fp_reduce` accumulates in the
#: device's own order, one step at a time, in pure Python, because a numpy dot product would round
#: differently from the hardware. Measured 2026-09-05: an atlas regeneration spends ~90 minutes there
#: and a radiance one ~40, nearly all of it recomputing capsules that nothing changed.
#:
#: ⚠️ THE KEY INCLUDES THE ENGINE'S OWN SOURCE, not just the entry. On this same date a conv2d branch
#: was added to the SIMT engine and an attention statement to the composed micro model. A cache keyed
#: on the entry alone would have served the pre-change goldens straight through both edits -- and a
#: stale golden does not fail loudly, it grades a backend against the wrong answer, which is precisely
#: the failure class this corpus exists to catch. The key binds the entry, binding, engine, operand
#: synthesis, selected external oracle source, Phase 0 source closure, and device facts.
#:
#: The store is a PURGEABLE cache namespace: deleting it costs time, never correctness.
_GOLDEN_CACHE_DISABLED = os.environ.get("MERLIN_NO_GOLDEN_CACHE", "").strip() not in ("", "0")


def source_files() -> tuple[Path, ...]:
    """The complete source closure formerly carried by the single generator file.

    Ignore interpreter caches, never source indirection. Required owners cannot
    disappear silently, and adding another Python helper extends the identity.
    """
    root = Path(__file__).parent
    required = {
        "__init__.py",
        "__main__.py",
        "profiles.py",
        "numerics.py",
        "golden_cache.py",
        "writer.py",
        "sweeps.py",
        "provenance.py",
        "generation.py",
    }
    sources = []
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise OSError("Phase 0 source closure contains a symlink")
        if path.suffix == ".py":
            if not path.is_file():
                raise OSError("Phase 0 source closure contains a nonregular Python source")
            sources.append(path)
    if not required <= {path.relative_to(root).as_posix() for path in sources}:
        raise OSError("Phase 0 source closure is incomplete")
    return tuple(sources)


def source_digest() -> str:
    """Conservatively invalidate cached answers after any derivation source edit."""
    root = Path(__file__).parent
    try:
        from merlin.common.paths import module_source_path

        record = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files()
        }
        for name in ("merlin.common.external_tools", "merlin.target.spec.software"):
            helper = module_source_path(name)
            if helper.is_symlink() or not helper.is_file():
                return "unresolvable"
            record[name] = hashlib.sha256(helper.read_bytes()).hexdigest()
    except (ImportError, ValueError, OSError):
        return "unresolvable"
    return _document_digest(record)


def _source_digest_of(obj) -> str:
    """Digest of the SOURCE FILE that defines ``obj`` -- the bytes actually on disk.

    A dirty working tree changes this even when the commit does not, which is the property that makes
    an edit-in-progress invalidate its own cached results.
    """
    import inspect

    try:
        return hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
    except (TypeError, OSError):  # no resolvable source -> never cache against an unknown engine
        return "unresolvable"


def _golden_cache_key(fn, entry, binding, facts_sha: str = "", *, oracle_source: dict | None = None) -> str:
    """Digest of everything that determines the answer.

    ``facts_sha`` is the derived performance-document digest the corpus manifest records,
    not a hash of raw CIRCT extraction bytes. The
    goldens are deliberately INDEPENDENT of the RTL (an oracle derived from the device would be the
    device grading itself), but the RTL still reaches them INDIRECTLY: the binding's tile edge, dtypes
    and subnormal handling are derived from the capability manifest, and an entry's extents come from
    facts like memory capacity and array geometry. Keying on the facts digest invalidates
    conservatively -- more often than strictly required, never less -- so a changed device can never be
    answered from a cache built against the previous one. An empty digest means the caller could not
    establish which device this is, and is carried as its own distinct key rather than treated as
    "no change".
    """
    from merlin.development.targetgen.corpus import operands as CO

    h = hashlib.sha256()
    h.update(_document_digest(entry).encode("utf-8"))
    h.update(repr(binding).encode("utf-8"))  # frozen dataclass -> stable repr
    h.update(fn.__name__.encode("utf-8"))
    h.update(_source_digest_of(fn).encode("utf-8"))  # the engine that will answer
    h.update(_source_digest_of(CO).encode("utf-8"))  # how its operands are synthesized
    h.update(source_digest().encode("utf-8"))  # all extracted generation helpers, not just the numerical engine
    h.update(str(facts_sha).encode("utf-8"))  # WHICH DEVICE this corpus is about
    h.update(_document_digest(oracle_source or {}).encode("utf-8"))  # external numerical implementation, if selected
    return h.hexdigest()


def _golden_cached(fn, entry, binding, facts_sha: str = "", *, oracle_source: dict | None = None):
    """``fn(entry, binding)``, answered from the cache when every input digest matches."""
    from merlin.development.testgen.generation.goldens import _float_golden

    if fn is _float_golden and (
        not isinstance(oracle_source, dict)
        or oracle_source.get("schema") != "merlin.specir_oracle_source.v1"
        or set((oracle_source.get("modules") or {}).keys()) != {"specir.oracle.dtypes", "specir.oracle.refmodel"}
        or any(
            not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in (oracle_source.get("modules") or {}).values()
        )
    ):
        raise ValueError("selected external SpecIR oracle source identity is required for float golden cache")
    if _GOLDEN_CACHE_DISABLED or _source_digest_of(fn) == "unresolvable" or source_digest() == "unresolvable":
        return fn(entry, binding)
    from merlin.common.artifacts import cache_dir

    key = _golden_cache_key(fn, entry, binding, facts_sha, oracle_source=oracle_source)
    path = Path(cache_dir("capsule_goldens")) / key[:2] / f"{key}.json"
    if path.is_file():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("oracle_source") == oracle_source:
                return rec["outputs"], rec["prov"]
        except Exception:  # noqa: BLE001 -- a damaged entry is a MISS, never a wrong answer
            pass
    outputs, prov = fn(entry, binding)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"outputs": outputs, "prov": prov, "oracle_source": oracle_source}), encoding="utf-8")
        tmp.replace(path)  # atomic: a concurrent reader never sees a half-written entry
    except OSError:  # an unwritable cache must not fail a generation
        pass
    return outputs, prov


# ---- writer ------------------------------------------------------------------------------------
# Host-owned Phase 0 writer implementation.

INTEGER_CONTRACTION_BOUND_OPS = frozenset(
    {
        "matmul",
        "linear",
        "matmul_bias",
        "fused_matmul_bias",
        "resident_reuse",
        "host_island_seam",
        "residual_seam",
        "sibling_feed",
        "conv2d",
        "scope_chain",
        "attention_qk",
    }
)


def _m2m_unavailable_reason() -> str:
    if "MERLIN_PHASE0_FROZEN_SOURCE_MAP" in os.environ:
        if os.environ.get("MERLIN_PHASE0_M2M_REQUIRED") == "1":
            return "selected Model2MLIR capture runtime is unavailable"
        return "frozen Phase 0 has no selected Model2MLIR capture runtime"
    return "model2MLIR capture runtime unavailable (set MERLIN_M2M_PYTHON)"


def _with_attestations(written, source, *, start: int):
    """Record on the capsule the sealed-runner attestations of the captures it was built from."""
    attestations = list(getattr(source, "attestations", ()) or ())[start:]
    if written is None or not attestations:
        return written
    path = Path(written) / "capsule.yaml"
    capsule = yaml.safe_load(path.read_bytes())
    capsule["capture_execution_attestations"] = attestations
    path.write_text(yaml.safe_dump(capsule, sort_keys=False))
    return written


def _skip_or_require_m2m(entry: dict) -> None:
    reason = _m2m_unavailable_reason()
    required = os.environ.get("MERLIN_PHASE0_M2M_REQUIRED") == "1"
    verified = os.environ.get("MERLIN_PHASE0_EVIDENCE_MODE") == "verified"
    if required or verified:
        raise ValueError(f"{entry['name']}: {reason}; requested frontend capsule cannot be omitted")
    print(f"  [skip] {entry['name']}: {reason}")


def _is_source_backed(entry: dict) -> bool:
    return bool(
        entry.get("kind") == "model"
        or entry.get("op") == "model"
        or entry.get("source") in {"pytorch", "spec"}
        or entry.get("pytorch_ref")
        or entry.get("spec_ref")
    )


def _write_capsule(entry, binding, out_root, facts_sha: str = "", *, capture=None, component_only: bool = False):
    """Write one capsule, then GUARANTEE it carries its generalization-intent block.

    The stamp is a post-step rather than something each writer does, because there are four writers
    (direct-MLIR, pytorch-sourced, spec-sourced, whole-model) and three of them build their capsule dict
    themselves and return early. Stamping inside ``corpus_spec.build`` alone left 14 of atlas's 33
    capsules unannotated -- exactly the silent-gap failure mode this block exists to close -- so it is
    applied here, at the one point every path must pass through.
    """
    written = _write_capsule_source(entry, binding, out_root, facts_sha, capture=capture, component_only=component_only)
    return _finish_capsule(entry, binding, out_root, written, component_only=component_only)


def _write_capsule_source(entry, binding, out_root, facts_sha="", *, capture=None, component_only=False):
    """Capture and write local source bytes; no sibling qualification or publication."""
    if type(component_only) is not bool:
        raise TypeError("component_only must be an explicit Boolean")
    if entry.get("public_source_role") is not None:
        from merlin.development.testgen.public_sources.public_source_roles import require_entry

        require_entry(entry)
    return _write_capsule_inner(
        entry, binding, out_root, facts_sha, **({"capture": capture} if capture is not None else {})
    )


def _finish_capsule(entry, binding, out_root, written, *, component_only=False, diagnostic_source_protocol=None):
    """All ordinary postchecks, including model qualification, in the caller's order."""
    from merlin.development.testgen.generation.reference_bounds import _source_integer_reference_bound

    if not written:
        return written
    d = Path(written) if not isinstance(written, Path) else written
    capf = d / "capsule.yaml" if d.is_dir() else None
    if capf is None or not capf.exists():
        return written
    cap = yaml.safe_load(capf.read_text()) or {}
    dirty = False
    if entry.get("public_boundary"):
        from merlin.development.testgen.public_sources.boundaries import verify_written

        cap["public_boundary"] = copy.deepcopy(entry["public_boundary"])
        source_evaluation = verify_written(cap, d)
        if "stages" in source_evaluation:
            cap["functional_source_evaluation"] = source_evaluation
        dirty = True
    regime, _ = CS.entry_binding(entry, binding)
    model_capsule = entry.get("kind") == "model" or entry.get("op") == "model"
    if diagnostic_source_protocol is not None:
        from merlin.development.testgen.public_sources.public_source_frontier import arithmetic

        if getattr(binding, "target", None) != "source_only" or regime != "int" or not model_capsule:
            raise ValueError("source frontier cannot skip qualification under a target-bound writer")
        cap["public_source_frontier"] = arithmetic(entry, cap, d, protocol=diagnostic_source_protocol)
        dirty = True
    from merlin.development.targetgen.applications.native_storage import native_storage_source_spec

    # An integer storage type does not make a cast or permutation a mesh MAC.
    # This exception is limited to the validated nonaccelerating native source
    # declaration. Its complete captured ABI and values are checked by the
    # source writer; ordinary written-program placement checks still apply.
    native_storage = (
        native_storage_source_spec(entry) if entry.get("source") == "pytorch" or entry.get("pytorch_ref") else None
    )
    if (
        diagnostic_source_protocol is None
        and regime == "int"
        and _is_source_backed(entry)
        and (model_capsule or native_storage is None)
    ):
        if model_capsule:
            # A whole model's contractions read internal tensors, so it is qualified by the isolated
            # verification of every device group it forms; without that evidence it is refused
            # exactly as before (see model_qualification).
            from merlin.development.testgen.generation.model_capsules import qualify

            bound = qualify(entry, cap, d, binding, Path(out_root))
            cap["model_qualification"] = bound
        else:
            bound = _source_integer_reference_bound(entry, cap, d)
        golden = GS.load_golden(d)
        source = golden.get("golden_source") if isinstance(golden, dict) else None
        if source != "host_torch_eager" and not (isinstance(source, str) and source.startswith("specir_program_")):
            raise ValueError("source-backed integer bound requires its independently captured golden")
        cap["integer_partial_sum_bound"] = bound
        golden["integer_partial_sum_bound"] = bound
        golden["qualification"] = (
            "complete connected integer source arithmetic and selected executable correspondence verified; "
            "full-model accuracy and timing unverified"
            if model_capsule
            and bound.get("method") in {"checked_native_integer_v1", "checked_native_integer_affine_copy_v2"}
            else "source-backed integer model qualified by isolated per-group verification; "
            "target execution and full-model numerics unverified"
            if model_capsule
            else "source-backed integer contraction with concrete operand-stream internal-width bounds; "
            "target execution and full-mesh ordering unverified"
        )
        GS.write_golden(d, golden)
        dirty = True
    if not (cap.get("semantic") or {}).get("generalization_axis"):
        _, eb = CS.entry_binding(entry, binding)
        cap["semantic"] = CS._semantic_block(entry, eb)
        dirty = True
    dirty = _backfill_required_classes(cap, binding) or dirty
    _validate_lane_declaration(entry, binding)
    dirty = _carry_declared_blocks(entry, cap) or dirty
    # AFTER the declared blocks are carried, because `lanes` reaches the capsule THERE. Checking before
    # it read an empty lanes block and passed everything -- a verification that cannot see what it
    # verifies is worse than none, because it reports the assertion as checked.
    _verify_a_forbidden_lane_is_provable(d, cap, getattr(binding, "target", None))
    dirty = _cap_oracle_tiers(entry, cap) or dirty
    if not component_only:
        dirty = _stamp_member_geometry(cap, binding) or dirty
    # THE TOLERANCE MUST BE FALSIFIABLE AT THIS GOLDEN'S SCALE, and here is the first point at which
    # both the capsule and its golden exist for EVERY writer -- the same reason the generalization stamp
    # lives here. A profile declares ONE absolute tolerance for a whole target, which is the right shape
    # for a datapath error budget and the wrong shape for a small-magnitude output: a softmax capsule
    # whose golden spans 0.0139..0.1523 was graded at `atol: 0.25`, so zeros, the mean and the midrange
    # all passed it. It reported a numeric pass and proved nothing.
    if (d / GS.DOCUMENT).is_file() and (cap.get("numeric_policy") or {}).get("atol") is not None:
        _gdoc = GS.load_golden(d) or {}
        _pol, _prov = NF.falsifiable_policy(
            cap["numeric_policy"], _gdoc.get("outputs") or {}, name=str(entry.get("name") or d.name)
        )
        if cap.get("numeric_policy") != _pol or cap.get("numeric_falsifiability") != _prov:
            cap["numeric_policy"] = _pol
            cap["numeric_falsifiability"] = _prov
            dirty = True
    if dirty:
        capf.write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    if entry.get("public_source_role") is not None:
        from merlin.development.testgen.public_sources.public_source_roles import stamp_written

        stamp_written(entry, cap, d)
        capf.write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    write_description(entry, cap, d)
    return written


_DECLARED_BLOCKS = (
    "component_coverage",
    "input_palette",
    "performance",
    "comparison_group",
    "pass_requirements",
    "lanes",
    # The oracle-tier ceiling and the sibling a capped member rests on. Declared
    # once in a profile and carried onto every member derived from it, so the
    # link between a screened member and the capsule that certifies it is
    # machine-readable rather than prose. See merlin.development.targetgen.oracles.tier_policy.
    "max_oracle_tier",
    "max_timing_tier",
    "extends",
    # A case's role in performance scoring and the derived demand it represents
    # (merlin.experiments.phase0.case_redundancy / performance_demand).
    "performance_scope",
    "performance_demand",
    # The chain a connected case was derived from (merlin.experiments.phase0.chain_demand).
    "chain_demand",
)


def _carry_declared_blocks(entry: dict, cap: dict) -> bool:
    """Copy the profile entry's declared intent blocks onto the capsule. Never overwrites one already
    there (a hand-authored capsule is the source of record), and never invents one."""
    dirty = False
    for key in _DECLARED_BLOCKS:
        value = entry.get(key)
        if value is None or cap.get(key) is not None:
            continue
        cap[key] = dict(value) if isinstance(value, dict) else value
        dirty = True
    return dirty


def _cap_oracle_tiers(entry: dict, cap: dict) -> bool:
    """Trim a capsule's required tiers to the deepest one its SIZE can afford, and say what it rests on.

    ``corpus_spec.build`` gives every capsule the target's full tier list, which is right for a
    capsule sized to the tile edge and wrong for one sized to an application: a shape too large to
    simulate cycle-accurately cannot demand the cycle-accurate tier, and demanding it anyway makes
    the whole corpus unrunnable rather than making the capsule affordable.

    ``extends`` is carried onto the capsule for the same reason it exists at all -- an L2-only
    capsule is admissible only as an extension of a sibling that WAS certified, so the thing it rests
    on has to be readable from the capsule itself rather than inferred from a naming convention.
    """
    # The cap may come from the profile entry or already sit on a carried capsule (a hand-authored or
    # materialized member). Either way a written capsule must never demand a tier above its own cap:
    # `required_oracle_tiers: [..., L3]` beside `max_oracle_tier: L2` is unrunnable as a corpus.
    cap_to = str(entry.get("max_oracle_tier") or cap.get("max_oracle_tier") or "")
    if not cap_to:
        return False
    tiers = [str(t) for t in (cap.get("required_oracle_tiers") or ())]
    if cap_to not in tiers:
        raise ValueError(
            f"{cap.get('name')!r} caps its oracle tier at {cap_to!r}, which is not among the tiers "
            f"this target declares ({tiers}); a cap onto a tier that does not exist would silently "
            f"leave the capsule demanding everything"
        )
    trimmed = tiers[: tiers.index(cap_to) + 1]
    changed = trimmed != tiers or cap.get("max_oracle_tier") != cap_to
    cap["required_oracle_tiers"] = trimmed
    cap["max_oracle_tier"] = cap_to
    if entry.get("extends"):
        cap["extends"] = str(entry["extends"])
        changed = True
    return changed


SYNTH_ROLE = "derived_sweep"


class UnprovableForbid(ValueError):
    """A capsule forbids the mesh on a program the target would legitimately accelerate.

    A distinct type because the right response depends on who wrote the capsule. A HAND-AUTHORED one
    is a contradiction its author must resolve, and aborting is how they find out. A SYNTHESIZED one
    is not: synthesis is pure -- it derives entries from the requirement without building or
    classifying anything -- so the axis genuinely cannot know that `normalization` decomposes into
    regions this target admits. The generator is the first place that fact exists, and the honest
    response there is to drop the capsule and REPORT the family as uncovered, which is the same
    fail-closed shape as `host_only_unsynthesizable`: a requirement that produced no capsule stays
    visible, and nothing can pass in its place.
    """


def _verify_a_forbidden_lane_is_provable(d: Path, cap: dict, target: str | None) -> None:
    """A capsule may only forbid the mesh if its own program has nothing the mesh may legitimately take.

    CLASSIFIED, not predicted. Whether a capsule is host-only is a property of the regions its written
    interface contains, and the only honest way to know is to ask the classifier the coverage gate asks
    (`boundary.profile_capsule`). Deriving it from the family instead is nearly right and not right
    enough: `normalization` decomposes into a reduction and an elementwise map, so the family-level rule
    catches a target that admits either -- and still passed a target admitting NEITHER whose rmsnorm
    program turned out to contain an eligible region anyway.

    Why it must raise rather than quietly drop the assertion. `forbid: [on_mesh]` says the submission
    must NOT accelerate this; on a program containing admitted work that is a demand to leave
    performance on the table, and a compiler doing the right thing is recorded as violating a lane. The
    capsule is wrong, not the compiler, and the generator is where that is still cheap to fix.
    """
    if not target:
        return
    forbid = {str(x) for x in ((cap.get("lanes") or {}).get("forbid") or ())}
    if "on_mesh" not in forbid:
        return
    from merlin.development.targetgen.grading import boundary as BD

    prof = BD.profile_capsule(d, str(target))
    if prof.kind == BD.HOST_ONLY:
        return
    raise UnprovableForbid(
        f"{cap.get('name')!r} forbids `on_mesh`, but {str(target)!r} classifies its program as "
        f"{prof.kind!r} rather than host-only: it contains region(s) the manifest admits, so the "
        f"assertion demands the compiler decline work it is entitled to do. Choose a family whose "
        f"decomposition this target admits nothing of, or drop the forbid"
    )


def _validate_lane_declaration(entry: dict, binding) -> None:
    """Refuse an unreachable or self-contradictory lane declaration AT GENERATION TIME.

    The whole-model writer already ran ``_checked_lanes``; the other writers did not, because they never
    carried lanes at all. Now that every writer does, the check has to move with it -- a bar the target's
    declared units make unreachable is not a capability test, it is a wall, and the place to catch it is
    where an author can still fix it.
    """
    lanes = entry.get("lanes") or {}
    if not lanes:
        return
    from merlin.experiments.corpus.capsules.model_inputs import _checked_lanes

    _checked_lanes(entry, binding)  # raises on an unreachable `require`
    forbid = [str(x) for x in (lanes.get("forbid") or ())]
    both = sorted(set(str(x) for x in (lanes.get("require") or ())) & set(forbid))
    if both:
        raise ValueError(
            f"{entry.get('name')!r}: lane(s) {both} are both required and forbidden; one "
            f"of the two assertions can never hold"
        )
    target = getattr(binding, "target", None)
    if forbid and target:
        from merlin.target.capabilities.routing import reachable_lanes

        unreachable = sorted(set(forbid) - reachable_lanes(target))
        if unreachable:
            raise ValueError(
                f"{entry.get('name')!r}: forbids lane(s) {unreachable} that {target!r} cannot populate "
                f"anyway, so the assertion is vacuously true and tests nothing"
            )


def _stamp_member_geometry(cap: dict, binding) -> bool:
    """Record which shape class an OBJECTIVE member occupies, and whether real models present it.

    A perf member exists to make generated code faster on shapes that matter, and nothing in a
    generated capsule said which shapes those were. MEASURED on this repo's corpus: 27 of the 29
    classifiable OBJECTIVE members sit in a geometric class the target's own census -- derived from
    real captures -- does not contain, and every reachable class in that census has no members. That
    was invisible from every artifact and answerable only by running a script.

    Stamped here for the same reason the generalization block is: four writers build their capsule
    dict themselves, so a key handled in one of them is absent from three quarters of the corpus.

    DELIBERATELY NOT A GATE. The census's mass-carrying class is recorded as unbuildable on this
    target, so refusing off-census members would emit an empty corpus. Recording the placement makes
    the hole readable from the tracked capsule; deciding what to do about it is the corpus's job, not
    the writer's.
    """
    perf = cap.get("performance")
    if not isinstance(perf, dict) or perf.get("member_class") != "OBJECTIVE":
        return False
    target = str(getattr(binding, "target", "") or "")
    if not target:
        return False
    from merlin.analysis.programs.member_geometry import stamp_for

    block = stamp_for(cap, target=target)
    # None and a block saying `in_census: false` are different answers: the first is "this member's
    # geometry is unreadable here", the second is "it was read and no capture presents it". Writing
    # the first as the second would turn an unpriced op into a coverage claim.
    if block is None:
        return False
    # THE GRANULARITY THE EXTENTS WERE MINTED AGAINST, recorded beside them. Sweep axes are written as
    # multiples of the target's own tile, so an extent alone does not say how many array row blocks it
    # spans -- and that count decides, for the array-side stream families, whether a member can
    # exercise their lever at all or is their control. A record, never a bound: nothing fails on it.
    tile = getattr(binding, "tile_dim", None)
    if isinstance(tile, int) and not isinstance(tile, bool) and tile >= 1:
        block = {**block, "row_block": int(tile)}
    if perf.get("shape_geometry") == block:
        return False
    perf["shape_geometry"] = block
    return True


def _backfill_required_classes(cap: dict, binding) -> bool:
    """Fill an EMPTY ``expected.instruction_classes`` from the target's own derived taxonomy.

    The source-backed writers (pytorch / spec / model) build their capsule dict themselves and leave this
    empty, so a contraction authored in PyTorch shipped with no coverage requirement at all while the
    direct-MLIR twin next to it carried the full systolic sequence -- the L1 coverage assertion silently
    did not apply to exactly the frontend-faithful capsules the generalization corpus is made of.

    Derived, never hardcoded: the slots come from the op's family in the closed vocabulary and are mapped
    to class names through THIS target's role census. Fail-closed at every step -- an op that owes no
    contraction, an undecidable taxonomy, or a role the target does not have all leave the list empty
    rather than inventing a demand. Only ever fills an empty list; never edits an authored one."""
    exp = cap.get("expected")
    if not isinstance(exp, dict) or exp.get("instruction_classes"):
        return False
    op = (cap.get("operation") or {}).get("op")
    if not op:
        return False
    attrs = (cap.get("operation") or {}).get("attributes", {}) or {}
    modes = exp.get("modes", {}) or {}
    from merlin.target.isa import taxonomy as IT

    tax = IT.taxonomy_for_target(binding.target)  # {} when the target ships no ISA definition
    if not tax or not tax.get("by_class"):
        return False
    want = IT.required_classes_for_op(
        tax,
        op=op,
        output_dtype=attrs.get("output_dtype") or (cap.get("numeric_policy") or {}).get("dtype"),
        epilogue=tuple(attrs.get("epilogue", []) or []),
        movement=op in ("movement", "copy") or bool(modes.get("movement")),
    )
    if not want:
        return False
    exp["instruction_classes"] = list(want)
    return True


def _roster_captures() -> dict:
    """Captured bundles available as derivation evidence, keyed by model name.

    Same store and key normalisation as `check_conformance_coverage._captures`, so a micro model is
    derived from exactly the captures the requirement was derived from.
    """
    from merlin.common.paths import artifacts_dir

    root = artifacts_dir() / "recaptures"
    if not root.is_dir():
        return {}
    out = {}
    for d in sorted(root.iterdir()):
        m = d / "model.mlir"
        if m.is_file():
            out[d.name.replace("_fp32_consistent", "").replace("_consistent", "")] = m
    return out


def _emit_micro_model_loader(entry: dict, target: str, out_root, *, capture_dtype: str | None = None) -> bool:
    """Write the derived micro model's loader into its capsule directory, or say why not.

    `micro_model.spec` selects operations whose standalone placement is supported by the selected
    hardware and software contracts, and puts the others on the host. The run passes its frozen capture
    and software-spec snapshots, so the loader cannot drift with an ambient target file.
    """
    from merlin.development.targetgen.corpus import derived_models as MM

    # Derived-only execution passes exact run-owned snapshots. Pop the internal
    # selector before the entry becomes a public capsule or provenance record.
    captures = entry.pop("_frozen_application_captures", None)
    software_spec = entry.pop("_frozen_software_spec", None)
    if captures is None:
        captures = _roster_captures()
    if not captures:
        print(f"  [skip] {entry['name']}: no captured model is available to derive the inventory from")
        return False
    try:
        spec = MM.spec(target, captures, software_spec=software_spec, capture_dtype=capture_dtype)
        src = MM.emit_pytorch(spec)
    except MM.UnwritableLayer as exc:
        print(f"  [skip] {entry['name']}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 -- an underivable spec is not a crash
        print(f"  [skip] {entry['name']}: micro-model spec unavailable: {type(exc).__name__}: {exc}")
        return False
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    loader = d / "capsule.pytorch.py"
    loader.write_text(src, encoding="utf-8")
    entry["loader"] = str(loader)
    entry.setdefault("model", entry["name"])
    print(f"  [micro] {entry['name']}: {spec.composition()} over {len(spec.layers)} derived layer(s)")
    return True


def _write_capsule_inner(entry, binding, out_root, facts_sha: str = "", *, capture=None):
    from merlin.development.testgen.generation.reference_bounds import _integer_reference_bound

    regime, eb = CS.entry_binding(entry, binding)
    if _merlin_model_builders_1.entry_component_program(entry, eb) is not None:
        from merlin.development.testgen.components.component_integer_bounds import preflight_entry

        preflight_entry(entry, binding=binding)
    if entry.get("input_palette") is not None and regime == "mx":
        raise ValueError("component input palettes require separately declared typed block-scale inputs on MX")
    # Whole-model capsule: a small representative network lowered end-to-end via model2MLIR, graded vs its
    # host torch-eager output, GATED so it runs only after the op suite proves itself. Additive: skipped
    # (loudly) when the m2m venv is absent.
    if entry.get("kind") == "model" or entry.get("op") == "model":
        from merlin.capture import artifacts as source_artifacts
        from merlin.capture import spec as source_spec
        from merlin.experiments.corpus.capsules import framework_sources as source_pytorch_sources
        from merlin.experiments.corpus.capsules import framework_sources as source_spec_sources
        from merlin.experiments.corpus.capsules import model_sources as source_model_sources

        if entry.get("materialized_capture"):
            artifact = source_artifacts.materialized_model_artifacts(entry["materialized_capture"])
            written = source_model_sources.write_model_capsule(entry, eb, out_root, artifact=artifact)
            from merlin.development.targetgen.applications.materialized_integer_capture import selected

            if selected(entry["materialized_capture"]):
                from merlin.development.testgen.sealed_capture.attestation import require_verified_execution

                attestations = entry.get("capture_execution_attestations")
                if not isinstance(attestations, list) or len(attestations) != 1:
                    raise ValueError("integer public writer needs its one original sealed execution")
                require_verified_execution(attestations[0])
                path = Path(written) / "capsule.yaml"
                cap = yaml.safe_load(path.read_bytes())
                cap["capture_execution_attestations"] = copy.deepcopy(attestations)
                path.write_text(yaml.safe_dump(cap, sort_keys=False))
            return written
        src = capture if capture is not None else capture_source()
        if not src.available():
            _skip_or_require_m2m(entry)
            return None
        # A DERIVED micro model writes its own loader first. Without this the entry names a loader that
        # does not exist, and the capsule that the composition axis exists to produce cannot be built.
        if entry.get("micro_model") and not _emit_micro_model_loader(
            entry, eb.target, out_root, capture_dtype=entry.get("capture_dtype") or eb.operand_dtype
        ):
            return None
        start = len(getattr(src, "attestations", ()) or ())
        return _with_attestations(
            source_model_sources.write_model_capsule(entry, eb, out_root, source=src), src, start=start
        )
    # PREFERRED source: a capsule defined in PyTorch (frontend-faithful), lowered to linalg via model2MLIR
    # with a host torch-eager golden. Opt in per entry (``source: pytorch``). Restricted to the float
    # regime: a host-eager float reference is graded with tolerance, matching the merlin_iface float
    # interface; int/MX datapaths keep the direct-MLIR engines below (the endorsed fallback for the
    # dtypes torch/torchAO does not faithfully model, e.g. int8xint8 systolic or block-scaled MX).
    if entry.get("source") == "pytorch" or entry.get("pytorch_ref"):
        from merlin.development.targetgen.applications.inventory import int_mm_source_is_qualified
        from merlin.development.targetgen.corpus.declarations import native_storage_source_spec

        # AN ENTRY THAT NAMES A QUANTIZATION SCHEME has said which arithmetic its program must contain,
        # so the float-regime restriction below does not apply to it. The restriction exists because a
        # host-eager float reference cannot grade an int/MX datapath -- but that is a statement about
        # the DEFAULT weight-only capture, which emits a float matmul over dequantized weights. A W8A8
        # scheme emits `aten._int_mm` accumulating in i32, which IS the mesh's arithmetic, and torch
        # eager then computes the same quantized math, so the golden is right by construction.
        # For operation-derived slices, the source's exact digest-bound integer arithmetic is checked
        # independently. Static and dynamic whole-model quantization remain distinct source schemes.
        if (
            native_storage_source_spec(entry) is not None
            or entry.get("quant_scheme")
            or (
                entry.get("capture_op") == "int_matmul"
                and int_mm_source_is_qualified(entry.get("application_signature_match") or {})
            )
        ):
            pass
        elif regime != "simt":
            raise ValueError(
                f"pytorch source for capsule {entry['name']!r} needs a float dtype "
                f"(got regime {regime!r} for {eb.operand_dtype!r}); author int/MX capsules "
                f"via the direct-MLIR engine"
            )
        from merlin.capture import artifacts as source_artifacts
        from merlin.capture import spec as source_spec
        from merlin.experiments.corpus.capsules import framework_sources as source_pytorch_sources
        from merlin.experiments.corpus.capsules import framework_sources as source_spec_sources
        from merlin.experiments.corpus.capsules import model_sources as source_model_sources

        src = capture if capture is not None else capture_source()
        if not src.available():
            # Diagnostic derivation without a selected runtime may skip a
            # frontend capsule. Selected or verified runs must fail closed.
            _skip_or_require_m2m(entry)
            return None
        start = len(getattr(src, "attestations", ()) or ())
        return _with_attestations(
            source_pytorch_sources.write_pytorch_capsule(entry, eb, out_root, source=src), src, start=start
        )
    # Spec source: a capsule whose PROGRAM + bit-exact golden come from the specir verification spec itself
    # (``spec_ref: '<gen>:op.<name>'``). Additive: a gen without a specir program emitter (or no specir) is
    # skipped loudly rather than sinking the target.
    if entry.get("source") == "spec" or entry.get("spec_ref"):
        from merlin.capture import artifacts as source_artifacts
        from merlin.capture import spec as source_spec
        from merlin.experiments.corpus.capsules import framework_sources as source_pytorch_sources
        from merlin.experiments.corpus.capsules import framework_sources as source_spec_sources
        from merlin.experiments.corpus.capsules import model_sources as source_model_sources

        src = source_spec.SpecRefSource()
        if not src.available():
            print(f"  [skip] {entry['name']}: spec source needs specir (set SPECIR_ROOT)")
            return None
        try:
            return source_spec_sources.write_spec_capsule(entry, eb, out_root, source=src)
        except source_spec.SpecProgramUnavailable as e:
            print(f"  [skip] {entry['name']}: {e}")
            return None
    cap, mlir = _merlin_model_builders_1.build(entry, eb)
    if cap["operation"]["op"] == "component_program":
        cap["numerical_semantics"] = copy.deepcopy(entry.get("numerical_semantics") or {})
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8"
    )
    if regime == "int":
        bound = _integer_reference_bound(entry, cap)
        cap["integer_partial_sum_bound"] = bound
        (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
        if cap["operation"]["op"] == "component_program":
            from merlin.development.testgen.components.component_numerics import evaluate

            outputs = evaluate(cap)
            source = "merlin_tensor_component_program"
        else:
            outputs = CG.golden({**cap, "__dir__": ""})
            source = "merlin_tensor_int"
        GS.write_golden(
            d,
            {
                "golden_source": source,
                "integer_partial_sum_bound": bound,
                "qualification": "mathematical reference; target execution and full-mesh ordering unverified",
                "outputs": outputs,
            },
        )
    elif regime == "specir":
        selected_semantics = float_semantics(entry, eb)
        oracle_source = specir_oracle_source_identity(selected_semantics)
        outputs, prov = _golden_cached(_float_golden, entry, eb, facts_sha, oracle_source=oracle_source)
        if specir_oracle_source_identity(selected_semantics) != oracle_source:
            raise OSError("selected SpecIR oracle source changed while computing the golden")
        GS.write_golden(
            d,
            {
                "golden_source": (
                    "specir_refmodel_float"
                    if selected_semantics["selection_status"] == "explicit"
                    else "specir_refmodel_fp8_e4m3_bf16"
                ),
                "oracle_provenance": {
                    "engine": "specir.oracle.dtypes + specir.oracle.refmodel.fp_reduce",
                    "source_identity": oracle_source,
                    "datapath": selected_semantics,
                    # How the datapath decodes an operand code. A unit that admits only normal operands
                    # reads a zero exponent field as zero; the golden decodes it the same way, so the two
                    # references implement ONE datapath (see the target's profile ``datapath`` block).
                    "operand_decode": ("subnormal_flush_to_zero" if eb.subnormal_operand_flush else "exact"),
                    "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                    "accum_dtype": eb.cap_dtype(eb.accum_dtype),
                    "output_dtype": selected_semantics["readout_dtype"],
                    "note": "INDEPENDENT of the target RTL (not self-oracle); specir refmodel is the reference.",
                    "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                    "inputs": prov,
                },
                "outputs": outputs,
            },
        )
    elif regime == "mx":
        # matmul/linear -> the single MX GEMM golden; attention_mx -> the fused flash-attention composition
        # (two MX GEMMs + a bf16 softmax), both over the SAME validated mx_ref engine.
        if entry.get("op") == "attention_mx":
            outputs, prov = _golden_cached(_mx_attention_golden, entry, eb, facts_sha)
            engine = (
                "mlc.validate.mx_ref.mx_matmul x2 (QK & PV, the engine's transcription of the MX "
                "host golden lib/golden/mx_golden.cpp) + numpy bf16 row-softmax; P requantized to mxfp8 per-row"
            )
            datapath = (
                "O = mx_matmul(softmax(mx_matmul(Q,K^T)/sqrt(H) [+softcap]), V); E8M0 per 32-elt "
                "K group; bf16 accumulate + bf16 softmax"
            )
        elif entry.get("op") == "gemv_batched":
            outputs, prov = _golden_cached(_mx_gemv_batched_golden, entry, eb, facts_sha)
            engine = "mlc.validate.mx_ref.mx_matmul x B (independent batched MX GEMMs stacked row-major)"
            datapath = "B x [M,H]@[H,N] on the mx_pe; one E8M0 scale per 32-elt K group; bf16 accumulate"
        else:
            outputs, prov = _golden_cached(_mx_golden, entry, eb, facts_sha)
            engine = (
                "mlc.validate.mx_ref.mx_matmul (the engine's transcription of the MX host golden "
                "lib/golden/{mx_fp_math.h,mx_golden.cpp}; mirrors the RTL, bit-exact vs spike)"
            )
            datapath = (
                "16-deep systolic per-column acc schedule (ACC_E/ACC_M); one E8M0 scale per "
                "32-elt K group; bf16 accumulate"
            )
        GS.write_golden(
            d,
            {
                "golden_source": "mlc_mx_ref_hardware_semantics",
                "oracle_provenance": {
                    "engine": engine,
                    "datapath": datapath,
                    "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                    "block_scale": "e8m0",
                    "output_dtype": "bf16",
                    "note": (
                        "NOT specir (the SpecIR float refmodel is a different datapath); "
                        "MX is a distinct block-scaled datapath."
                    ),
                    "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                    "inputs": prov,
                },
                "outputs": outputs,
            },
        )
    else:  # simt (IEEE fp16/bf16/f32)
        outputs, prov = _golden_cached(_simt_golden, entry, eb, facts_sha)
        GS.write_golden(
            d,
            {
                "golden_source": "ieee_simt_f32_accumulate",
                "oracle_provenance": {
                    "engine": "numpy IEEE float (CVFPU fp32 accumulate; format-rounded operands)",
                    "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                    "accum_dtype": "f32",
                    "output_dtype": "f32",
                    "note": "SIMT cores do ordinary IEEE math; reference is independent of any accelerator model.",
                    "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                    "inputs": prov,
                },
                "outputs": outputs,
            },
        )
    return d
