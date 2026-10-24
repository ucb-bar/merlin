"""Which classes a target DECLARES, against which ones a capsule can ever be required to issue.

``corpus_spec._classes_source`` now reads the target's authored ``encoding.corpus_issue_order``.
The shared code must neither name nor silently filter out an instruction class. The target may still
declare more alternatives than its current baseline corpus demands, and this test keeps those
declarations visible until operation-specific capsules cover them.

This is a RATCHET, not a fix. It pins the gap so it is visible and cannot grow, and it fails if a
declared class starts being dropped that was reachable before. Closing it means deriving the
requirement from the declared role rather than extending the literal -- and deciding subsumption, since
a capsule owing LOOP_WS must not ALSO owe the PRELOAD/COMPUTE pair the loop performs itself.
"""

from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("gemmini")

#: Classes these targets DECLARE that no capsule can be required to issue. Measured 2026-09-18. This
#: may only SHRINK: an entry removed means the requirement became reachable, which is the fix.
_UNREACHABLE: dict[str, set[str]] = {
    "gemmini": {"COMPUTE_ACCUMULATE", "LOOP_CONV", "LOOP_WS", "MVIN2"},
}


def _contract(target: str) -> dict:
    return yaml.safe_load((repo_root() / "examples" / target / "target/contracts/target_contract.yaml").read_text())


def _declared_pool(target: str) -> set[str]:
    """What the target's own encoding defines, exactly as `_classes_source` computes it."""
    contract = _contract(target)
    enc = contract.get("encoding") or {}
    semantic = set((enc.get("semantic_class") or {}).values())
    subtypes = set((enc.get("config_subtype") or {}).values())
    return semantic | subtypes


def _issue_order(target: str) -> set[str]:
    return set((_contract(target).get("encoding") or {}).get("corpus_issue_order") or [])


def test_the_declared_but_unreachable_set_is_exactly_what_was_measured():
    """If this fails because the set SHRANK, delete the entry -- the requirement became reachable."""
    for target, expected in _UNREACHABLE.items():
        pool = _declared_pool(target)
        assert pool, f"{target} declares no semantic classes at all"
        unreachable = pool - _issue_order(target) - {"CONFIG"}  # CONFIG has only declared subtypes here
        assert unreachable == expected, (
            f"{target}: declared-but-unrequirable classes changed from {sorted(expected)} to "
            f"{sorted(unreachable)}. A class that became reachable is the fix — remove it here. A NEW "
            f"one is a regression: the corpus can no longer demand a capability the target declares."
        )


def test_the_device_loop_family_is_among_the_unreachable():
    """Named explicitly because it is the one this cost: measured on ResNet-50, the program used 8 of
    the 25 functs the RTL declares, and the 17 it never emitted are this family."""
    unreachable = _declared_pool("gemmini") - _issue_order("gemmini") - {"CONFIG"}
    assert {"LOOP_WS", "LOOP_CONV"} <= unreachable, (
        "the loop family is now requirable — close the ratchet entry and make sure a capsule owing "
        "LOOP_WS does not ALSO owe the PRELOAD/COMPUTE pair the loop issues itself"
    )


def test_every_class_the_contract_requires_is_actually_declared():
    """The other direction: a demanded class the target does not declare is fabricated coverage."""
    pool = _declared_pool("gemmini")
    order = _issue_order("gemmini")
    assert order <= pool, f"the issue order lists classes gemmini does not declare: {sorted(order - pool)}"


def test_a_stage_scoped_class_is_demanded_by_a_capsule_carrying_its_stage():
    """A class the contract scopes to epilogue stages (``encoding.stage_scoped_classes``) is reachable
    only if some capsule carries one of those stages and therefore owes it. Listing it in the issue order
    without such a capsule would make it requirable in name and demanded by nothing."""
    from merlin.common.paths import merlin_dir

    scoped = (_contract("gemmini").get("encoding") or {}).get("stage_scoped_classes") or {}
    assert scoped, "the reviewed contract scopes no class to a stage"
    assert set(scoped) <= _issue_order("gemmini")
    demanded: dict[str, list[str]] = {}
    for path in sorted((merlin_dir() / "contract" / "capsules").glob("*/*/capsule.yaml")):
        cap = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        owed = set((cap.get("expected") or {}).get("instruction_classes") or ())
        stages = set(((cap.get("operation") or {}).get("attributes") or {}).get("epilogue") or ())
        for cls, scope in scoped.items():
            if cls in owed:
                assert stages & set(scope), f"{cap.get('name')} owes {cls} without carrying any of {scope}"
                demanded.setdefault(cls, []).append(str(cap.get("name")))
    assert set(demanded) == set(scoped), (
        f"stage-scoped classes no capsule demands: {sorted(set(scoped) - set(demanded))}"
    )


def _bias_trace(seed_class: str, *, seed_to_accumulator: bool) -> dict:
    """A well-formed bias kernel whose seed is issued as ``seed_class`` (decoded flags as RTL would)."""
    rows = [
        ("FENCE", {}), ("FLUSH", {}), ("CONFIG_EX", {}), ("CONFIG_LD", {}),
        (seed_class, {"is_acc_addr": seed_to_accumulator}),
        ("MVIN", {"is_acc_addr": False}), ("MVIN", {"is_acc_addr": False}),
        ("CONFIG_ST", {}), ("PRELOAD", {}), ("COMPUTE_PRELOADED", {}), ("MVOUT", {}), ("FENCE", {}),
    ]  # fmt: skip
    return {"instructions": [{"index": i, "class": c, "decoded": d} for i, (c, d) in enumerate(rows)]}


@pytest.mark.parametrize("op", ["matmul", "fused_matmul_bias"])
def test_either_seed_form_satisfies_a_derived_bias_capsule(op):
    """What a REGENERATED bias capsule (the epilogue-axis bias capsule, or a fused-bias perf capsule)
    would owe, checked against both RTL-legal seeds: the full-width seed load, and a plain load into an
    accumulator address (the in-repo codegen's form). A kernel with neither is refused."""
    from types import SimpleNamespace

    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen import trace_check as TCK

    contract = _contract("gemmini")
    enc = contract["encoding"]
    classes = CS._classes_source(SimpleNamespace(target="gemmini"), contract, taxonomy={})
    expected = {"instruction_classes": classes(op=op, output_dtype="i8", epilogue=("bias_add",))}
    alternatives = TCK.parse_alternatives(enc.get("class_alternatives") or {})
    scoped = set(enc.get("stage_scoped_classes") or ())
    assert scoped and scoped <= set(expected["instruction_classes"]) and scoped <= set(alternatives)
    for seed in sorted(scoped):
        for klass, fields in alternatives[seed]:
            alt = _bias_trace(klass, seed_to_accumulator=bool(fields.get("is_acc_addr")))
            dedicated = _bias_trace(seed, seed_to_accumulator=True)
            for trace in (dedicated, alt):
                res = TCK.check(trace, expected, alternatives=alternatives)
                assert not [v for v in res["violations"] if "missing" in v], res["violations"]
            neither = _bias_trace(klass, seed_to_accumulator=False)
            res = TCK.check(neither, expected, alternatives=alternatives)
            assert f"required instruction class missing: {seed}" in res["violations"]


def test_the_declared_alternative_names_a_field_the_contract_decodes():
    """An alternative is identified by a decoded field; one the contract never decodes could never match,
    so a typo would silently turn the alternative off."""
    contract = _contract("gemmini")
    alts = (contract.get("encoding") or {}).get("class_alternatives") or {}
    roles = (contract.get("rocc_operand_roles") or {}).get("classes") or {}
    assert alts
    for forms in alts.values():
        for form in forms:
            names = {f.get("name") for f in (roles.get(form["class"]) or {}).get("fields") or ()}
            assert set(form.get("decoded") or {}) <= names, (form, sorted(n for n in names if n))


def test_the_seed_capsule_keeps_the_dedicated_form_requirable():
    """The alternative must not leave the dedicated seed class demanded by nothing: one tracked capsule
    demands it in its own form."""
    from merlin.common.paths import merlin_dir

    alts = set((_contract("gemmini").get("encoding") or {}).get("class_alternatives") or ())
    strict = set()
    for path in sorted((merlin_dir() / "contract" / "capsules").glob("*/*/capsule.yaml")):
        expected = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("expected") or {}
        strict |= set(expected.get("strict_classes") or ()) & set(expected.get("instruction_classes") or ())
    assert alts <= strict, (
        f"classes with an alternative that no capsule demands in their own form: {sorted(alts - strict)}"
    )


def test_a_regenerated_bias_capsule_is_covered_by_the_move_in_to_accumulator_seed():
    """The declared-versus-emitted comparison, exercised directly. A bias capsule regenerated under the
    current contract declares the dedicated seed class; a kernel that seeds with a move-in to an
    accumulator address (the in-repo codegen's form) covers it, every emitted class is still declared,
    and listing the seed class as strict -- or seeding nowhere -- leaves it uncovered."""
    from types import SimpleNamespace

    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen import trace_check as TCK

    contract = _contract("gemmini")
    alternatives = TCK.parse_alternatives((contract.get("encoding") or {}).get("class_alternatives") or {})
    classes = CS._classes_source(SimpleNamespace(target="gemmini"), contract, taxonomy={})
    expected = {"instruction_classes": classes(op="matmul", output_dtype="i8", epilogue=("bias_add",))}
    seeds = set(alternatives) & set(expected["instruction_classes"])
    assert seeds, "a regenerated bias capsule declares no class the contract gives an alternative"
    for seed in sorted(seeds):
        (klass, fields), *_ = alternatives[seed]
        instrs = _bias_trace(klass, seed_to_accumulator=bool(fields.get("is_acc_addr")))["instructions"]
        emitted = {row["class"] for row in instrs}
        missing, how = TCK.declared_class_coverage(expected, instrs, alternatives=alternatives)
        assert not missing and any(seed in note for note in how), (missing, how)
        assert emitted <= set(expected["instruction_classes"]) | {"FENCE"}
        strict = {**expected, "strict_classes": [seed]}
        assert TCK.declared_class_coverage(strict, instrs, alternatives=alternatives)[0] == [seed]
        unseeded = _bias_trace(klass, seed_to_accumulator=False)["instructions"]
        assert TCK.declared_class_coverage(expected, unseeded, alternatives=alternatives)[0] == [seed]
