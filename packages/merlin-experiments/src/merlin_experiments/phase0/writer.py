"""Host-owned Phase 0 writer implementation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from merlin.targetgen import capsule_golden as CG  # noqa: E402
from merlin.targetgen import corpus_spec as CS  # noqa: E402
from merlin.targetgen import numeric_falsifiability as NF  # noqa: E402

from .golden_cache import _golden_cached
from .numerics import _float_golden, _mx_attention_golden, _mx_gemv_batched_golden, _mx_golden, _simt_golden


def _entry_regime(entry, binding):
    """Route an entry to its numeric regime + return a per-entry binding (operand/accum overridden). ``int``
    (gemmini), ``specir`` (atlas fp8), ``mx`` (microscaling block-scaled FP), ``simt`` (IEEE fp16/bf16/f32).
    Routed purely by the entry's operand dtype token — no target name."""
    tok = entry.get("operand_dtype") or binding.operand_dtype
    # ONE definition of the routing, in corpus_spec, so the synthesizer can ask the same question this
    # answers. A second copy here drifted from the synthesizer's view and let entries be emitted that no
    # writer could materialize.
    regime = CS.regime_for_dtype(tok)
    acc = {"mx": "bf16", "simt": "f32"}.get(regime, binding.accum_dtype)
    eb = dataclasses.replace(
        binding,
        operand_dtype=tok,
        accum_dtype=acc,
        integer=(regime == "int"),
        compare=("exact_int" if regime == "int" else "tolerance_float"),
    )
    return regime, eb


# ------------------------------------------------------------------------------------------------
def _write_capsule(entry, binding, out_root, facts_sha: str = ""):
    """Write one capsule, then GUARANTEE it carries its generalization-intent block.

    The stamp is a post-step rather than something each writer does, because there are four writers
    (direct-MLIR, pytorch-sourced, spec-sourced, whole-model) and three of them build their capsule dict
    themselves and return early. Stamping inside ``corpus_spec.build`` alone left 14 of atlas's 33
    capsules unannotated -- exactly the silent-gap failure mode this block exists to close -- so it is
    applied here, at the one point every path must pass through.
    """
    written = _write_capsule_inner(entry, binding, out_root, facts_sha)
    if not written:
        return written
    d = Path(written) if not isinstance(written, Path) else written
    capf = d / "capsule.yaml" if d.is_dir() else None
    if capf is None or not capf.exists():
        return written
    cap = yaml.safe_load(capf.read_text()) or {}
    dirty = False
    if not (cap.get("semantic") or {}).get("generalization_axis"):
        _, eb = _entry_regime(entry, binding)
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
    dirty = _stamp_member_geometry(cap, binding) or dirty
    # THE TOLERANCE MUST BE FALSIFIABLE AT THIS GOLDEN'S SCALE, and here is the first point at which
    # both the capsule and its golden exist for EVERY writer -- the same reason the generalization stamp
    # lives here. A profile declares ONE absolute tolerance for a whole target, which is the right shape
    # for a datapath error budget and the wrong shape for a small-magnitude output: a softmax capsule
    # whose golden spans 0.0139..0.1523 was graded at `atol: 0.25`, so zeros, the mean and the midrange
    # all passed it. It reported a numeric pass and proved nothing.
    _gp = d / "golden.yaml"
    if _gp.is_file() and (cap.get("numeric_policy") or {}).get("atol") is not None:
        _gdoc = yaml.safe_load(_gp.read_text(encoding="utf-8")) or {}
        _pol, _prov = NF.falsifiable_policy(
            cap["numeric_policy"], _gdoc.get("outputs") or {}, name=str(entry.get("name") or d.name)
        )
        if cap.get("numeric_policy") != _pol or cap.get("numeric_falsifiability") != _prov:
            cap["numeric_policy"] = _pol
            cap["numeric_falsifiability"] = _prov
            dirty = True
    if dirty:
        capf.write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    _write_capsule_readme(entry, cap, d)
    return written


#: Profile-entry keys that describe what a capsule is FOR rather than what it computes, and which every
#: writer must carry through untouched. They are stamped in the same post-step as the generalization
#: block, and for the same reason: three of the four writers build their capsule dict themselves, so a
#: key handled in only one of them is silently absent from two thirds of the corpus.
#:
#: ``performance``      which optimization level the capsule exercises and which schedule lever its cycle
#:                      count can see. A capsule is otherwise mute about this, so a perf corpus and a
#:                      functional corpus are indistinguishable once generated.
#: ``comparison_group`` the capsule's place in a set whose cycle counts are comparable to one another --
#:                      a fused implementation against the parts it replaces. The field has been declared
#:                      on four capsules since they were written and consumed by nothing, which is the
#:                      same thing as not existing.
#: ``pass_requirements`` the compiler-obligation classes a capsule demands, which is the ONLY link
#:                      between a catalogued pass and a concrete capsule that requires it
#:                      (``check_pass_obligations.py`` rejects a pass no capsule obliges). It was
#:                      hand-written onto two capsules and unknown to this generator, so every
#:                      regeneration silently deleted the corpus's only pass obligations.
#: ``lanes``               the interop/negative-lane contract: which execution lanes must have carried
#:                      work, and (``forbid``) which must have carried none. Only the whole-model writer
#:                      emitted it, so a model_slice capsule declaring lanes silently lost them -- which
#:                      is how the first host-only capsule generated with `lanes: None` and asserted
#:                      nothing at all.
_DECLARED_BLOCKS = (
    "performance",
    "comparison_group",
    "pass_requirements",
    "lanes",
    # The oracle-tier ceiling and the sibling a capped member rests on. Declared
    # once in a profile and carried onto every member derived from it, so the
    # link between a screened member and the capsule that certifies it is
    # machine-readable rather than prose. See merlin.targetgen.tier_policy.
    "max_oracle_tier",
    "max_timing_tier",
    "extends",
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
    cap_to = str(entry.get("max_oracle_tier") or "")
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
    changed = trimmed != tiers
    cap["required_oracle_tiers"] = trimmed
    if entry.get("extends"):
        cap["extends"] = str(entry["extends"])
        changed = True
    return changed


#: ``source_role`` the corpus synthesizer stamps on every entry it derives. Mirrors
#: ``corpus_synth.SOURCE_ROLE``; compared as data so a hand-authored capsule and a derived
#: one can be told apart where the two need different handling.
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
    from merlin.targetgen import boundary as BD

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
    from merlin.targetgen.capsule_source import _checked_lanes

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
        from merlin.targetgen.routing import reachable_lanes

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
    from merlin.perf.member_geometry import stamp_for

    block = stamp_for(cap, target=target)
    # None and a block saying `in_census: false` are different answers: the first is "this member's
    # geometry is unreadable here", the second is "it was read and no capture presents it". Writing
    # the first as the second would turn an unpriced op into a coverage claim.
    if block is None or perf.get("shape_geometry") == block:
        return False
    perf["shape_geometry"] = block
    return True


def _write_capsule_readme(entry: dict, cap: dict, d: Path) -> None:
    """Write the capsule's ``README.md`` -- the 5th of the five files a capsule is DEFINED to have.

    The generator only ever emitted four of them, so every generated capsule was incomplete by the
    corpus's own definition and the materialized public view failed its own completeness check the moment
    a capsule arrived without a hand-written README. Derived from the profile entry and the capsule, so it
    cannot go stale: the prose is the entry's ``comment`` when it has one, otherwise a sentence built from
    the op, the source it was authored from, and the operand shapes/dtypes. Never overwrites a README that
    is already there -- the hand-written ones are the frozen source-of-record."""
    rd = d / "README.md"
    if rd.exists():
        return
    name = cap.get("name") or entry.get("name", "")
    prose = (entry.get("comment") or "").strip()
    if not prose:
        op = (cap.get("operation") or {}).get("op") or entry.get("op") or "unknown"
        ops = ", ".join(
            f"{i.get('name')}{list(i.get('shape') or [])}:{i.get('dtype')}"
            for i in (cap.get("inputs") or [])
            if i.get("name")
        )
        src = cap.get("source_reference") or entry.get("source_reference") or ""
        prose = f"{name}: {op}" + (f" over {ops}" if ops else "")
        prose += f", authored from {src}." if src else "."
    line = " ".join(
        f"{k}={v}"
        for k, v in (
            ("kind", cap.get("kind") or entry.get("kind")),
            ("label", cap.get("label") or entry.get("label")),
            ("op", (cap.get("operation") or {}).get("op") or entry.get("op")),
            ("modes", (cap.get("expected") or {}).get("modes", {})),
        )
        if v is not None
    )
    rd.write_text(f"# {name}\n\n{prose}\n\n{line}\n", encoding="utf-8")


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
    from merlin.targetgen import isa_taxonomy as IT

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


def _emit_micro_model_loader(entry: dict, target: str, out_root) -> bool:
    """Write the derived micro model's loader into its capsule directory, or say why not.

    `micro_model.spec` states what a target's minimal whole-model capsule must contain -- one layer per
    admitted family, one per family real captures contain that the manifest does not admit, sized to the
    target's own tile edge, host layers interleaved into the INTERIOR. `emit_pytorch` turns that into the
    loader. Doing it here rather than in `corpus_synth` is deliberate: the spec needs the captures, which
    is I/O, and the synthesizer is pure.
    """
    from merlin.targetgen import micro_model as MM

    captures = _roster_captures()
    if not captures:
        print(f"  [skip] {entry['name']}: no captured model is available to derive the inventory from")
        return False
    try:
        spec = MM.spec(target, captures)
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


def _write_capsule_inner(entry, binding, out_root, facts_sha: str = ""):
    regime, eb = _entry_regime(entry, binding)
    # Whole-model capsule: a small representative network lowered end-to-end via model2MLIR, graded vs its
    # host torch-eager output, GATED so it runs only after the op suite proves itself. Additive: skipped
    # (loudly) when the m2m venv is absent.
    if entry.get("kind") == "model" or entry.get("op") == "model":
        from merlin.targetgen import capsule_source as CSRC

        src = CSRC.PytorchRefSource()
        if not src.available():
            print(f"  [skip] {entry['name']}: model capsule needs the m2m venv (set MERLIN_M2M_PYTHON)")
            return None
        # A DERIVED micro model writes its own loader first. Without this the entry names a loader that
        # does not exist, and the capsule that the composition axis exists to produce cannot be built.
        if entry.get("micro_model") and not _emit_micro_model_loader(entry, eb.target, out_root):
            return None
        return CSRC.write_model_capsule(entry, eb, out_root, source=src)
    # PREFERRED source: a capsule defined in PyTorch (frontend-faithful), lowered to linalg via model2MLIR
    # with a host torch-eager golden. Opt in per entry (``source: pytorch``). Restricted to the float
    # regime: a host-eager float reference is graded with tolerance, matching the merlin_iface float
    # interface; int/MX datapaths keep the direct-MLIR engines below (the endorsed fallback for the
    # dtypes torch/torchAO does not faithfully model, e.g. int8xint8 systolic or block-scaled MX).
    if entry.get("source") == "pytorch" or entry.get("pytorch_ref"):
        # AN ENTRY THAT NAMES A QUANTIZATION SCHEME has said which arithmetic its program must contain,
        # so the float-regime restriction below does not apply to it. The restriction exists because a
        # host-eager float reference cannot grade an int/MX datapath -- but that is a statement about
        # the DEFAULT weight-only capture, which emits a float matmul over dequantized weights. A W8A8
        # scheme emits `aten._int_mm` accumulating in i32, which IS the mesh's arithmetic, and torch
        # eager then computes the same quantized math, so the golden is right by construction.
        if entry.get("quant_scheme"):
            pass
        elif regime != "simt":
            raise ValueError(
                f"pytorch source for capsule {entry['name']!r} needs a float dtype "
                f"(got regime {regime!r} for {eb.operand_dtype!r}); author int/MX capsules "
                f"via the direct-MLIR engine"
            )
        from merlin.targetgen import capsule_source as CSRC

        src = CSRC.PytorchRefSource()
        if not src.available():
            # A pytorch capsule needs the m2m venv (torch) at generation time. It is additive: skip it
            # (loudly) rather than sink the whole target, so a checkout without the venv still regenerates
            # the direct-MLIR corpus. A capture that STARTS but fails (opaque/crash) still raises.
            print(f"  [skip] {entry['name']}: pytorch source needs the m2m venv (set MERLIN_M2M_PYTHON)")
            return None
        return CSRC.write_pytorch_capsule(entry, eb, out_root, source=src)
    # Spec source: a capsule whose PROGRAM + bit-exact golden come from the specir verification spec itself
    # (``spec_ref: '<gen>:op.<name>'``). Additive: a gen without a specir program emitter (or no specir) is
    # skipped loudly rather than sinking the target.
    if entry.get("source") == "spec" or entry.get("spec_ref"):
        from merlin.targetgen import capsule_source as CSRC

        src = CSRC.SpecRefSource()
        if not src.available():
            print(f"  [skip] {entry['name']}: spec source needs specir (set SPECIR_ROOT)")
            return None
        try:
            return CSRC.write_spec_capsule(entry, eb, out_root, source=src)
        except CSRC.SpecProgramUnavailable as e:
            print(f"  [skip] {entry['name']}: {e}")
            return None
    cap, mlir = CS.build(entry, eb)
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8"
    )
    if regime == "int":
        (d / "golden.yaml").write_text(
            yaml.safe_dump(
                {"golden_source": "merlin_tensor_int", "outputs": CG.golden({**cap, "__dir__": ""})}, sort_keys=False
            ),
            encoding="utf-8",
        )
    elif regime == "specir":
        outputs, prov = _golden_cached(_float_golden, entry, eb, facts_sha)
        (d / "golden.yaml").write_text(
            yaml.safe_dump(
                {
                    "golden_source": "specir_refmodel_fp8_bf16",
                    "oracle_provenance": {
                        "engine": "specir.oracle.dtypes + specir.oracle.refmodel.fp_reduce",
                        "datapath": "acc <- round_bf16(acc + round_bf16(a*w)); k index_sequential; per_step; rne",
                        # How the datapath decodes an operand code. A unit that admits only normal operands
                        # reads a zero exponent field as zero; the golden decodes it the same way, so the two
                        # references implement ONE datapath (see the target's profile ``datapath`` block).
                        "operand_decode": ("subnormal_flush_to_zero" if eb.subnormal_operand_flush else "exact"),
                        "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                        "accum_dtype": eb.cap_dtype(eb.accum_dtype),
                        "output_dtype": "bf16",
                        "note": "INDEPENDENT of the target RTL (not self-oracle); specir refmodel is the reference.",
                        "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                        "inputs": prov,
                    },
                    "outputs": outputs,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    elif regime == "mx":
        # matmul/linear -> the single MX GEMM golden; attention_mx -> the fused flash-attention composition
        # (two MX GEMMs + a bf16 softmax), both over the SAME validated mx_ref engine.
        if entry.get("op") == "attention_mx":
            outputs, prov = _golden_cached(_mx_attention_golden, entry, eb, facts_sha)
            engine = (
                "mlc.validate.mx_ref.mx_matmul x2 (QK & PV, transcribed from "
                "radiance-kernels "  # target-ok: reference-source provenance, not control flow
                "lib/golden/mx_golden.cpp) + numpy bf16 row-softmax; P requantized to mxfp8 per-row"
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
                "mlc.validate.mx_ref.mx_matmul (transcribed from "
                "radiance-kernels "  # target-ok: reference-source provenance, not control flow
                "lib/golden/{mx_fp_math.h,mx_golden.cpp}; mirrors the RTL, bit-exact vs spike)"
            )
            datapath = (
                "16-deep systolic per-column acc schedule (ACC_E/ACC_M); one E8M0 scale per "
                "32-elt K group; bf16 accumulate"
            )
        (d / "golden.yaml").write_text(
            yaml.safe_dump(
                {
                    "golden_source": "mlc_mx_ref_hardware_semantics",
                    "oracle_provenance": {
                        "engine": engine,
                        "datapath": datapath,
                        "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                        "block_scale": "e8m0",
                        "output_dtype": "bf16",
                        "note": (
                            "NOT specir (specir is atlas fp8); "  # target-ok: descriptive numeric-regime contrast
                            "MX is a distinct block-scaled datapath."
                        ),
                        "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                        "inputs": prov,
                    },
                    "outputs": outputs,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    else:  # simt (IEEE fp16/bf16/f32)
        outputs, prov = _golden_cached(_simt_golden, entry, eb, facts_sha)
        (d / "golden.yaml").write_text(
            yaml.safe_dump(
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
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    return d
