"""Foundations for Target-Conditioned Semantic Coverage (Acceleratable Region Recall):
the canonical family vocabulary, the ``semantic_capabilities`` contract block, and the independent
eligibility oracle (the ARR denominator)."""
from __future__ import annotations

from merlin.common.paths import repo_root
from merlin.targetgen import compute_units as cu
from merlin.targetgen import coverage_certificate as cert
from merlin.targetgen import coverage_report as cov
from merlin.targetgen import capability_probes as cp
from merlin.targetgen import eligibility as el
from merlin.targetgen import generalization_splits as gs
from merlin.targetgen import routing as rt
from merlin.targetgen import semantic_families as sf
from merlin.targetgen import semantic_fuzzer as fz


# --- A1: canonical semantic-family vocabulary ---------------------------------------------------

def test_family_vocabulary_is_consistent():
    assert sf.check() == []
    assert set(sf.PRIMITIVES) <= sf.FAMILIES
    # composites decompose only to primitives
    for comp, parts in sf.COMPOSITES.items():
        assert all(p in sf.PRIMITIVES for p in parts)


def test_family_mapping_from_prov_and_op():
    assert sf.from_prov("contraction") == "contraction"
    assert sf.from_prov("normalization") == "normalization"
    assert sf.from_prov("elementwise") == "elementwise_map"
    assert sf.from_op("matmul") == "contraction"
    assert sf.from_op("softmax") == "softmax"
    assert sf.from_op("transpose") == "movement"
    # fail-closed on the unknown
    assert sf.from_prov(None) is None
    assert sf.from_op("gather_scatter") is None
    assert sf.primitives_of("attention") == ("contraction", "reduction", "elementwise_map")
    assert sf.primitives_of("contraction") == ("contraction",)


# --- A2: semantic_capabilities in the capability contract ---------------------------------------

def _contract() -> dict:
    return {"compute_units": [
        {"name": "simt", "kind": "simt", "dtypes": ["fp16", "fp32"],
         "ops": ["matmul", "elementwise"], "contains": ["mx"],
         "semantic_capabilities": [
             {"family": "contraction", "dtypes": ["fp16", "fp32"], "ranks": [2, 3]},
             {"family": "reduction", "dtypes": ["fp16", "fp32"]},
             {"family": "elementwise_map", "dtypes": ["fp16", "fp32"]},
             {"family": "movement", "dtypes": ["fp16", "fp32"]},
         ]},
        {"name": "mx", "kind": "systolic", "dtypes": ["mxfp8", "bf16"], "ops": ["matmul"],
         "semantic_capabilities": [{"family": "contraction", "dtypes": ["mxfp8", "bf16"]}]},
    ]}


def test_semantic_capabilities_parse_and_fold():
    units = cu.compute_units(_contract())
    # composition folds the mx PE's contraction dtypes up into the simt cluster
    eff = cu.effective(units[0], units)
    fams = {c.family for c in eff.semantic_capabilities}
    assert fams == {"contraction", "reduction", "elementwise_map", "movement"}
    cmap = cu.semantic_capability_map(units)
    # merged contraction is the union across both units
    assert set(cmap["contraction"].dtypes) == {"fp16", "fp32", "mxfp8", "bf16"}


def test_backward_compat_no_semantic_block():
    units = cu.compute_units({"compute_units": [{"name": "m", "kind": "systolic", "dtypes": ["int8"]}]})
    assert units[0].semantic_capabilities == ()
    assert cu.semantic_capability_map(units) == {}


# --- A3: independent eligibility oracle (the ARR denominator) ------------------------------------

def test_eligibility_truth_table():
    cm = el.capability_map_from_contract(_contract())
    D = el.RegionDescriptor

    assert el.is_eligible(D(op="matmul", in_dtype="bf16", rank=2), cm).eligible          # contraction
    assert el.is_eligible(D(op="softmax", in_dtype="fp16"), cm).eligible                 # composite via prims
    assert el.is_eligible(D(op="reduce", in_dtype="fp32"), cm).eligible

    # ineligible cases, each with a reason
    v = el.is_eligible(D(op="matmul", in_dtype="int8", rank=2), cm)
    assert not v.eligible and "int8" in v.reason                                         # dtype not supported
    v = el.is_eligible(D(op="matmul", in_dtype="bf16", rank=5), cm)
    assert not v.eligible and "rank" in v.reason
    v = el.is_eligible(D(op="gather", in_dtype="fp16"), cm)
    assert not v.eligible and v.family is None                                           # fail-closed


# --- B1: coverage certificate (numerator=routing decision, denominator=eligibility) ------------

def test_coverage_certificate_surfaces_false_fallback_and_ineligible():
    # Mesh routing binds only bf16/int8; the eligibility oracle declares contraction eligible for
    # bf16 AND fp16. So an fp16 matmul is eligible-but-not-accelerated (false_fallback = compiler gap),
    # and an int8 matmul routes to the mesh but the oracle rejects it (accelerated_ineligible).
    contract = {"compute_units": [
        {"name": "mesh", "kind": "systolic", "dtypes": ["bf16", "int8"], "ops": ["matmul"],
         "semantic_capabilities": [{"family": "contraction", "dtypes": ["bf16", "fp16"]}]},
    ]}
    units = cu.compute_units(contract)
    cap_map = el.capability_map_from_contract(contract)
    demands = [
        rt.OpDemand(op="matmul", in_fmt="bf16", weight_fmt="bf16", site="L0", m=16, k=16, n=16),  # accel+eligible
        rt.OpDemand(op="matmul", in_fmt="fp16", weight_fmt="fp16", site="L1", m=16, k=16, n=16),  # eligible, gapped
        rt.OpDemand(op="matmul", in_fmt="int8", weight_fmt="int8", site="L2", m=16, k=16, n=16),  # accel, ineligible
    ]
    plan = rt.route_plan_on(demands, units)
    c = cert.build(plan, cap_map, target="synthetic")

    assert c["n_regions"] == 3
    assert c["n_eligible"] == 2                       # bf16 + fp16 contraction
    assert c["n_accelerated"] == 2                    # bf16 + int8 routed to mesh
    assert c["n_eligible_accelerated"] == 1           # only bf16 is both
    assert c["false_fallback_count"] == 1             # fp16: hw-capable but compiler left it off-mesh
    assert c["accelerated_ineligible_count"] == 1     # int8: accelerated but oracle rejects
    m = c["metrics"]
    assert m["acceleratable_region_recall"] == 0.5    # 1 of 2 eligible accelerated
    assert m["acceleration_precision"] == 0.5         # 1 of 2 accelerated were eligible


# --- B2: suite-level ARR in the coverage aggregate ----------------------------------------------

def test_suite_acceleratable_coverage():
    # Three capsules with author-eligibility overrides (target=None -> oracle empty, overrides drive it).
    # A2 accelerated (passed L2/spike); A7 eligible but only L1 passed = false_fallback; SORT ineligible.
    caps = [
        {"name": "A2", "operation": {"op": "matmul"}, "numeric_policy": {"dtype": "bf16"},
         "semantic": {"eligible": True, "must_accelerate": True}},
        {"name": "A7", "operation": {"op": "matmul"}, "numeric_policy": {"dtype": "bf16"},
         "semantic": {"eligible": True, "must_accelerate": True}},
        {"name": "SORT", "operation": {"op": "reduce"}, "numeric_policy": {"dtype": "fp32"},
         "semantic": {"eligible": False, "fallback_allowed": True}},
    ]
    results = [
        {"capsule": "A2", "kind": "isa", "label": "public", "status": "pass",
         "tiers": {"L2": {"status": "pass"}}},
        {"capsule": "A7", "kind": "isa", "label": "public", "status": "fail",
         "tiers": {"L1": {"status": "pass"}, "L2": {"status": "fail"}}},
        {"capsule": "SORT", "kind": "isa", "label": "public", "status": "pass",
         "tiers": {"L0": {"status": "pass"}}},
    ]
    cap_by_name = {c["name"]: c for c in caps}
    ac = cov._acceleratable_coverage(results, cap_by_name, target=None)
    assert ac["n_eligible"] == 2
    assert ac["n_eligible_accelerated"] == 1
    assert ac["false_fallback"] == ["A7"]
    assert ac["acceleratable_region_recall"] == 0.5
    # A7 is eligible + must_accelerate but did not accelerate -> a must_accelerate violation.
    assert ac["must_accelerate_violations"] == ["A7"]
    assert ac["must_accelerate_pass"] is False
    # full aggregate surfaces it too
    full = cov.aggregate(results, capsules=caps, target=None)
    assert full["acceleratable_coverage"]["false_fallback"] == ["A7"]


# --- C1: generalization-axis (G0-G5) breakdown --------------------------------------------------

def test_generalization_axis_breakdown():
    caps = [
        {"name": "G0", "operation": {"op": "matmul"}, "numeric_policy": {"dtype": "bf16"},
         "semantic": {"eligible": True, "generalization_axis": "seen"}},
        {"name": "G1", "operation": {"op": "matmul"}, "numeric_policy": {"dtype": "bf16"},
         "semantic": {"eligible": True, "generalization_axis": "shape"}},
        {"name": "G1b", "operation": {"op": "matmul"}, "numeric_policy": {"dtype": "bf16"},
         "semantic": {"eligible": True, "generalization_axis": "shape"}},
    ]
    results = [
        {"capsule": "G0", "kind": "isa", "label": "public", "status": "pass",
         "tiers": {"L2": {"status": "pass"}}},                                   # seen: accelerated
        {"capsule": "G1", "kind": "isa", "label": "public", "status": "pass",
         "tiers": {"L2": {"status": "pass"}}},                                   # shape: accelerated
        {"capsule": "G1b", "kind": "isa", "label": "public", "status": "fail",
         "tiers": {"L2": {"status": "fail"}}},                                   # shape: NOT accelerated
    ]
    ac = cov._acceleratable_coverage(results, {c["name"]: c for c in caps}, target=None)
    axes = ac["by_generalization_axis"]
    assert axes["seen"]["recall"] == 1.0
    assert axes["shape"]["n_eligible"] == 2 and axes["shape"]["recall"] == 0.5


# --- cross-target: the denominator DERIVES per target, it is not one shape ----------------------

def _residual_caps(relpath: str) -> dict:
    from merlin.common.yaml import load_yaml
    res = load_yaml(repo_root() / relpath)
    return cu.semantic_capability_map(cu.compute_units(res))


def test_semantic_capabilities_differ_across_targets():
    # Each target's declared hardware capability follows its silicon — a GEMM mesh, a matmul-only MX
    # tile, and a full SIMT tensor target expose progressively larger semantic surfaces. This is the
    # derive-not-overfit property: the ARR denominator is not a single hardcoded shape.
    gem = _residual_caps("merlin/targets/gemmini/contracts/residual.yaml")
    mx = _residual_caps("out/artifacts/targets/mx_gemmini/contracts/residual.yaml")
    rad = _residual_caps("out/artifacts/targets/radiance/contracts/residual.yaml")

    assert set(gem) == {"contraction", "elementwise_map", "movement"}          # int8 GEMM accelerator
    assert gem["contraction"].dtypes == ("int8",)
    assert set(mx) == {"contraction"}                                          # matmul-only MX tile
    assert "mxfp8" in mx["contraction"].dtypes
    assert {"attention", "softmax", "normalization", "reduction"} <= set(rad)  # full SIMT tensor target
    # value scales with the resource-legality surface: strictly growing family sets
    assert len(gem) < len(rad) and len(mx) < len(rad)


# --- C2: leave-one-family-out splits ------------------------------------------------------------

def test_leave_one_family_out_splits():
    caps = [
        {"name": "A", "operation": {"op": "matmul"}, "semantic": {"semantic_family": "contraction"}},
        {"name": "B", "operation": {"op": "matmul"}, "semantic": {"semantic_family": "contraction"}},
        {"name": "C", "operation": {"op": "softmax"}, "semantic": {"semantic_family": "softmax"}},
        {"name": "M", "kind": "model", "operation": {"op": "model"}},   # whole-model: never held out
    ]
    assert gs.families_present(caps) == ["contraction", "softmax"]
    s = gs.partition_by_family(caps, "softmax")
    assert s["holdout"] == ["C"]
    assert "M" in s["dev"] and "A" in s["dev"]            # model stays in dev as an integration test
    splits = gs.leave_one_family_out_splits(caps)
    assert {x["held_out_family"] for x in splits} == {"contraction", "softmax"}
    # every split withholds something and keeps something
    assert all(x["holdout"] and x["dev"] for x in splits)


# --- D1: capability-derived property-test synthesis ---------------------------------------------

def _rad_like_capmap():
    contract = {"compute_units": [
        {"name": "simt", "kind": "simt", "dtypes": ["fp16", "fp32", "bf16"],
         "ops": ["matmul", "elementwise"],
         "semantic_capabilities": [
             {"family": "contraction", "dtypes": ["fp16", "fp32", "bf16"], "ranks": []},
             {"family": "reduction", "dtypes": ["fp16", "fp32", "bf16"]},
             {"family": "elementwise_map", "dtypes": ["fp16", "fp32", "bf16"]},
             {"family": "softmax", "dtypes": ["fp16", "fp32", "bf16"]},
         ]},
    ]}
    return el.capability_map_from_contract(contract)


def test_probes_are_derived_and_eligible_by_construction():
    cm = _rad_like_capmap()
    probes = cp.synthesize(cm)
    assert probes, "closure should synthesize probes"
    # a probe exists for every declared family, and spans multiple generalization axes
    fams = {p.descriptor.family for p in probes}
    assert {"contraction", "reduction", "elementwise_map", "softmax"} <= fams
    assert {"shape", "dtype"} <= {p.axis for p in probes}
    # THE self-consistency property: every derived probe is drawn from the closure, so is eligible
    for p in probes:
        assert el.is_eligible(p.descriptor, cm).eligible, f"probe {p.name} not eligible: {p.descriptor}"
    # deterministic: same contract -> same probe set
    assert [p.name for p in cp.synthesize(cm)] == [p.name for p in probes]


# --- D2: constrained-random semantic fuzzing ----------------------------------------------------

def test_fuzzer_is_deterministic_and_in_closure():
    cm = _rad_like_capmap()
    a = fz.fuzz_corpus(cm, 20, base_seed=0)
    b = fz.fuzz_corpus(cm, 20, base_seed=0)
    # deterministic per seed
    assert [p.name for p in a] == [p.name for p in b]
    assert [tuple(r.source for r in p.regions) for p in a] == \
           [tuple(r.source for r in p.regions) for p in b]
    # every fuzzed region is legal (in-closure -> eligible)
    for prog in a:
        assert prog.regions
        for r in prog.regions:
            assert el.is_eligible(r, cm).eligible, f"fuzzed region not eligible: {r}"


def test_eligibility_does_not_depend_on_routing():
    # The denominator must be independent of the compiler's routing/lowering. Walk the AST and assert
    # no import (module- or function-level) pulls in routing — so a future edit can't silently create
    # the circularity that would make ARR trivially 1.0. (The docstring may *mention* routing.)
    import ast

    src = (repo_root() / "merlin" / "python" / "merlin" / "targetgen" / "eligibility.py").read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.add(base)
            imported.update(f"{base}.{a.name}" for a in node.names)
    assert not any("routing" in m for m in imported), \
        f"eligibility oracle must not import routing (ARR would be trivially 1.0); imports={sorted(imported)}"


# --- the ARR denominator reads OPERAND dtypes, never the accumulator ------------------------------

def _cap(rel: str) -> dict:
    import yaml
    return yaml.safe_load((repo_root() / "merlin" / "contract" / "capsules" / rel / "capsule.yaml").read_text())


def test_capsule_region_reads_operand_dtype_not_the_accumulator():
    """``numeric_policy.dtype`` is the ACCUMULATOR dtype. Reading it as the operand dtype disabled ARR
    on every target: a gemmini matmul presented ``i32`` against a contract declaring ``int8``, so every
    gemmini capsule was ineligible and ARR reported ``None``; radiance escaped only because it
    accumulates in f32 and also declares fp32 as an operand format, so its fp16/MX capsules were judged
    as f32/bf16 and the dtype axis was unenforced everywhere."""
    gem = _cap("isa/A2_single_tile_matmul")
    assert gem["numeric_policy"]["dtype"] == "i32", "fixture drifted: this capsule accumulates in i32"
    r = cov._capsule_region(gem)
    assert r.in_dtype == "i8" and r.weight_dtype == "i8", (r.in_dtype, r.weight_dtype)

    mx = _cap("radiance/isa/R5_mx_tile_mxfp8")
    assert cov._capsule_region(mx).in_dtype == "mxfp8"          # was reported as the bf16 accumulator
    assert cov._capsule_region(_cap("radiance/isa/R1_gemm_fp16")).in_dtype == "fp16"   # was f32


def test_capsule_region_populates_rank_and_batch():
    """``SemanticCapability`` declares ``ranks`` and ``batch``; a descriptor that leaves them unset can
    never exercise them, so gemmini's ``ranks: [2]`` was dead on the corpus path."""
    r = cov._capsule_region(_cap("isa/A2_single_tile_matmul"))
    assert r.rank == 2 and r.batch == 1


def test_gemmini_denominator_is_not_empty():
    """The regression that matters: an ARR of ``None`` is a metric that measures nothing, and that is
    what gemmini reported for every capsule it ships."""
    caps = {}
    for sub in ("isa", "layers", "model_slices"):
        for f in sorted((repo_root() / "merlin" / "contract" / "capsules" / sub).rglob("capsule.yaml")):
            import yaml
            c = yaml.safe_load(f.read_text())
            caps[c["name"]] = c
    results = [{"capsule": n, "tiers": {"L2": {"status": "pass"}}} for n in caps]
    out = cov._acceleratable_coverage(results, caps, "gemmini")
    assert out["n_eligible"] > 0, "gemmini declares contraction+movement; its denominator cannot be empty"
    assert out["acceleratable_region_recall"] is not None


# --- capability DERIVATION: the denominator is evidence, not assertion ---------------------------

def test_isa_class_vocabulary_is_shared_and_refuses_mnemonics():
    """``from_isa_class`` maps the SHARED, closed class vocabulary a contract declares. It must NOT be
    a place to name-match a target's own mnemonics: atlas calls its reduction ``VREDSUM_BF``, and
    classifying that by its letters is exactly the string-matching this repo exists to avoid."""
    assert sf.from_isa_class("MVIN") == "movement"
    assert sf.from_isa_class("COMPUTE_ACCUMULATE") == "contraction"
    assert sf.from_isa_class("FENCE") == "synchronization"
    # plumbing licenses nothing -- configuring a datapath is not a computation
    assert sf.from_isa_class("CONFIG") is None and sf.from_isa_class("FLUSH") is None
    # a target-specific mnemonic must fall through, not be guessed at
    assert sf.from_isa_class("VREDSUM_BF") is None
    assert sf.check() == []


def test_composed_with_blocks_standalone_and_intersects_on_merge():
    """A systolic mesh whose only elementwise hardware is the readout requant can fuse it onto a
    contraction and cannot run a standalone gelu. Declaring it standalone gives the target a permanent
    false_fallback no compiler change can clear."""
    epi = cu.SemanticCapability(family="elementwise_map", dtypes=("int8",),
                                composed_with=("contraction",))
    v = el.is_eligible(el.RegionDescriptor(op="gelu", in_dtype="int8"), {"elementwise_map": epi})
    assert not v.eligible and "only fused" in v.reason

    # ...but a unit that runs it standalone wins the merge: composed_with INTERSECTS, it does not union,
    # or one unit's limitation would constrain another unit's freedom.
    free = cu.SemanticCapability(family="elementwise_map", dtypes=("int8",))
    merged = cu._merge_caps(epi, free)
    assert merged.composed_with == ()
    assert el.is_eligible(el.RegionDescriptor(op="gelu", in_dtype="int8"),
                          {"elementwise_map": merged}).eligible


def test_undetermined_is_neither_eligible_nor_ineligible():
    """A family no evidence source could decide must not be scored either way: counting it ineligible
    shrinks the denominator and flatters recall; counting it eligible demands work the hardware may not
    support. It is reported as unmeasured."""
    r = el.RegionDescriptor(op="rmsnorm", in_dtype="int8")
    plain = el.is_eligible(r, {})
    assert not plain.eligible and not plain.undetermined

    undet = el.is_eligible(r, {}, undetermined={"normalization"})
    assert not undet.eligible and undet.undetermined
    assert "UNDETERMINED" in undet.reason


def test_derivation_grounds_each_target_in_its_own_evidence():
    """Every declared family should trace to something the target itself carries. The deriver is the
    auditor, not the author -- it never rewrites the declaration."""
    from merlin.targetgen import capability_derive as cd
    from merlin.targetgen import target_registry as trg

    for target in ("gemmini", "atlas"):
        contract = trg.load_contract(target)
        derived = cd.derive(target, contract, {})
        assert derived.families(), f"{target}: no family derived from its own evidence"
        for fam, ev in derived.supported.items():
            assert ev.evidence, f"{target}/{fam}: derived without recording what was observed"
            assert ev.source in ("isa_role", "isa_class", "rtl_facts", "unit_intent"), ev.source
        # the declaration and the evidence must agree -- drift is an error, not a note
        drift = cd.reconcile(el.capability_map_from_contract(contract), derived)
        assert not [d for d in drift if d["kind"] == "missing_declaration"], \
            f"{target}: hardware evidence the contract hides SHRINKS the ARR denominator: {drift}"


def test_generated_capsules_carry_their_generalization_block():
    """A capsule with no ``semantic`` block can never raise a ``must_accelerate`` violation, so its
    coverage certificate passes vacuously. It was hand-authored (and a regeneration deleted it); it is
    now stamped at the one point every writer passes through."""
    import yaml
    caps = repo_root() / "merlin" / "contract" / "capsules"
    for sub in ("isa", "atlas", "radiance"):
        for f in sorted((caps / sub).rglob("capsule.yaml")):
            c = yaml.safe_load(f.read_text()) or {}
            sem = c.get("semantic") or {}
            assert sem.get("generalization_axis"), f"{c.get('name')}: no generalization block"
            fam, op = sem.get("semantic_family"), (c.get("operation") or {}).get("op")
            derived = sf.from_op(op)
            if fam and derived:
                assert fam == derived, f"{c.get('name')}: declares {fam!r}, op derives {derived!r}"


# --- a headline recall must say WHICH families it is a claim about --------------------------------

def test_recall_reports_its_per_family_denominator():
    """``ARR = 1.000`` reads as "the compiler covers this device". On a target whose only standalone
    families are contraction and movement it is a claim about those two, and the report has to say so.

    Measured on the shipped corpus: 22 eligible regions = 21 contraction + 1 movement, while 4
    elementwise, 2 attention, 3 reduction, 1 normalization and 1 softmax region sit outside the
    denominator entirely. Every one of those exclusions is correct; quoting the ratio without them is
    not."""
    import yaml
    caps = {}
    for sub in ("isa", "layers", "model_slices"):
        d = repo_root() / "merlin" / "contract" / "capsules" / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("capsule.yaml")):
            c = yaml.safe_load(f.read_text())
            caps[c["name"]] = c
    results = [{"capsule": n, "tiers": {"L2": {"status": "pass"}}} for n in caps]
    arr = cov._acceleratable_coverage(results, caps, "gemmini")

    by_fam = arr["by_family"]
    assert by_fam, "a recall with no per-family breakdown cannot be read for what it claims"
    # the denominator is exactly the families that contributed to it
    contributing = {f for f, b in by_fam.items() if b["n_eligible"]}
    assert sum(by_fam[f]["n_eligible"] for f in contributing) == arr["n_eligible"]
    assert "contraction" in contributing
    # families present in the corpus but outside the denominator must still appear, at zero
    assert set(by_fam) - contributing, "excluded families must be visible, not omitted"


def test_a_fused_only_family_is_named_as_a_hardware_exclusion():
    """A family the device runs ONLY as an epilogue is real hardware whose standalone regions are
    correctly ineligible. That exclusion is invisible in the ratio, so it is reported by name — the
    alternative is a reader inferring the compiler covers an elementwise lane that does not exist."""
    import yaml
    caps = {}
    for sub in ("isa", "layers", "model_slices"):
        d = repo_root() / "merlin" / "contract" / "capsules" / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("capsule.yaml")):
            c = yaml.safe_load(f.read_text())
            caps[c["name"]] = c
    results = [{"capsule": n, "tiers": {"L2": {"status": "pass"}}} for n in caps]
    arr = cov._acceleratable_coverage(results, caps, "gemmini")
    assert "elementwise_map" in arr["fused_only_families"], \
        "the mesh's readout epilogue is not a standalone elementwise engine; the contract says so"
    assert arr["n_fused_only_ineligible"] >= 1, \
        "regions excluded by that hardware fact must be counted, not merely implied"


def test_the_rendered_report_carries_the_family_table_and_the_caveat():
    """coverage.json is not what a human reads. The markdown is."""
    from merlin.targetgen import coverage_report as cr
    doc = cr.render_markdown({
        "total": 1, "by_kind": {}, "by_label": {}, "by_tier_reached": {},
        "instruction_class_coverage": {}, "mode_coverage": {},
        "unavailable": {"vcs": 0, "firesim": 0},
        "acceleratable_coverage": {
            "n_eligible": 3, "n_eligible_accelerated": 3,
            "acceleratable_region_recall": 1.0, "acceleration_precision": 1.0,
            "n_undetermined": 0, "n_unclassified": 0,
            "by_family": {
                "contraction": {"n_regions": 3, "n_eligible": 3, "n_eligible_accelerated": 3,
                                "recall": 1.0},
                "elementwise_map": {"n_regions": 2, "n_eligible": 0, "n_eligible_accelerated": 0,
                                    "recall": None},
            },
            "fused_only_families": ["elementwise_map"],
            "n_fused_only_ineligible": 2,
        },
    }, [])
    assert "| semantic family | regions | eligible | accelerated | recall |" in doc
    assert "| elementwise_map | 2 | 0 | 0 | n/a |" in doc
    assert "ONLY fused behind another family" in doc
    assert "not about everything the device can compute" in doc
