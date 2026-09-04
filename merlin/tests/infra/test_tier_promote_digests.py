"""Per-component digests: an edit must requeue only the capsules that ride on what it touched.

Phase F proves a compiler functionally complete; Phase P then forks it and edits it continuously. Every
edit re-hashes the submission, and while that hash is one number, EVERY certificate dies with it -- so the
cycle-accurate tier (minutes per capsule) is re-bought for the whole corpus per edit and the round never
finishes.

These tests pin the fix and, more importantly, its failure directions:

  * a capsule that declares `depends_on` survives an edit to a component it does not name (the saving);
  * a capsule that declares NOTHING is invalidated by any edit at all (today's behaviour, fail closed --
    an undeclared dependency set means "depends on everything", never "depends on nothing");
  * bytes no component claims invalidate every capsule, so narrowing the declared attribution cannot
    quietly keep stale certificates alive;
  * a verdict recorded before the decomposition existed, or naming a component this submission does not
    have, comes back UNDETERMINABLE and re-runs -- never silently "still fresh".

The falsifier that matters is `test_invalidation_matrix`: it touches each component in turn and asserts
the exact set of capsules that lost their certificate. A matrix where every touch invalidates everything
(or nothing) would prove nothing, so the assertion is on the whole matrix, not on a count.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.oracle_schedule import (
    CHANGED, NO_VERDICT, PASS, UNATTRIBUTED, UNDETERMINABLE, UNKNOWN, WHOLE_SUBMISSION,
    CapsuleState, Verdict, explain, schedule,
)

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"

# The four entrypoints manifest.schema.json REQUIRES of every package. Spelled out here (a test may name
# the contract it is testing) but derived, never listed, in the library under test.
CMDS = ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")

MANIFEST = """\
artifact_type: mlir_oot_target_backend
target: t
language: python
authoring: {mode: hand_curated}
integrity_exempt: false
entrypoints: {tool: mlir_oot/opt}
commands:
  parse: {argv: ["{tool}", "{input_mlir}"]}
  lower_interface_to_target: {argv: ["{tool}", "--to-target", "{input_mlir}"]}
  emit_command_buffer: {argv: ["{tool}", "--cb={output_json}", "{input_mlir}"]}
  lower_target_to_llvm: {argv: ["{tool}", "--artifact", "{input_mlir}"]}
components:
  parse: [mlir_oot/parse.py]
  lower_interface_to_target: [mlir_oot/lowering/]
  emit_command_buffer: [mlir_oot/cmdbuf.py]
  lower_target_to_llvm: [mlir_oot/codegen.py]
"""

FILES = {
    "manifest.yaml": MANIFEST,
    "mlir_oot/opt": "#!/usr/bin/env python3\n",          # unattributed on purpose: nothing claims it
    "mlir_oot/parse.py": "parse v1\n",
    "mlir_oot/lowering/tile.py": "tile v1\n",
    "mlir_oot/lowering/emit.py": "emit v1\n",
    "mlir_oot/cmdbuf.py": "cb v1\n",
    "mlir_oot/codegen.py": "cg v1\n",
}


def _mod():
    """Import the harness promotion module by path -- it is a script, not an installed package."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("tier_promote", HARNESS / "tier_promote.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"tier_promote not importable here: {type(e).__name__}: {e}")
    return m


def _ws(tmp_path, files=None, name="ws"):
    ws = tmp_path / name
    for rel, body in (files or FILES).items():
        f = ws / "submission" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    return ws


# ---------------------------------------------------------------------------------------------
# 1. the vocabulary is DERIVED, not written down
# ---------------------------------------------------------------------------------------------
def test_vocabulary_comes_from_the_manifests_own_command_keys(tmp_path):
    B = _mod()
    vocab = B.component_vocabulary(_ws(tmp_path))
    assert set(CMDS) <= vocab
    # the renamed 4th entrypoint resolves under either spelling (oot_runner's own alias map)
    assert "emit_target_artifact" in vocab


def test_a_target_that_declares_a_fifth_command_gets_a_fifth_component(tmp_path):
    """No code change buys a new target its own components -- that is the point of deriving them."""
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST.replace(
        "components:\n", "  optimize_target: {argv: [\"{tool}\", \"-O2\", \"{input_mlir}\"]}\ncomponents:\n"
    ).replace("  parse: [mlir_oot/parse.py]", "  parse: [mlir_oot/parse.py]\n  optimize_target: [mlir_oot/opt.py]")
    files["mlir_oot/opt.py"] = "opt v1\n"
    ws = _ws(tmp_path, files)
    assert "optimize_target" in B.component_vocabulary(ws)
    assert "optimize_target" in B.submission_digests(ws)[1]


def test_an_unreadable_manifest_is_undeterminable_not_empty(tmp_path):
    """No manifest must not read as 'this submission declares no components' -- that would silently make
    every declared dependency unmatchable and hand back today's whole-submission behaviour with no signal.
    """
    B = _mod()
    ws = _ws(tmp_path, {"mlir_oot/parse.py": "x\n"})
    assert B.component_vocabulary(ws) is None
    assert B.component_paths(ws) is None
    assert B.submission_digests(ws)[1] == {}


def test_a_component_outside_the_vocabulary_is_rejected_and_reported(tmp_path):
    """A component with no legal name would hold no bytes, so it could never change -- and every
    certificate depending on it would live forever."""
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST + "  tiling_pass: [mlir_oot/lowering/tile.py]\n"
    ws = _ws(tmp_path, files)
    _, comps, rejected = B.submission_digests(ws)
    assert rejected == ["tiling_pass"]
    assert "tiling_pass" not in comps


def test_an_all_rejected_components_block_is_still_reported(tmp_path):
    """"every name was a typo" must not read as "no components declared" -- both fall through to the
    harness-derived decomposition, but only one of them is a mistake somebody has to see."""
    B = _mod()
    files = dict(FILES)
    head, _, _ = MANIFEST.partition("components:")
    files["manifest.yaml"] = head + "components:\n  tiling_pass: [mlir_oot/lowering/]\n"
    ws = _ws(tmp_path, files)
    _, comps, rejected = B.submission_digests(ws)
    assert rejected == ["tiling_pass"]
    assert "tiling_pass" not in comps
    # the block bought nothing, so the DERIVED decomposition takes over -- never a silent fall back to
    # the whole-submission digest, which is what made the rejection invisible in the first place
    assert B.decomposition(ws)["source"] == "derived"


# ---------------------------------------------------------------------------------------------
# 2. the decomposition itself
# ---------------------------------------------------------------------------------------------
def test_the_whole_digest_is_unchanged_by_adding_components(tmp_path):
    """Turning the feature on must not invalidate a single existing certificate."""
    B = _mod()
    plain = dict(FILES)
    plain["manifest.yaml"] = MANIFEST.split("components:")[0]
    with_comps = _ws(tmp_path, FILES, name="a")
    without = _ws(tmp_path, plain, name="b")
    # the manifests differ, so compare the LEGACY entry point against the new one on the same tree
    assert B._submission_digest(with_comps) == B.submission_digests(with_comps)[0]
    # and the tree WITHOUT a declared block is decomposed by the harness rather than left whole -- the
    # declared names are gone, so the components are the derived ones
    d = B.decomposition(without)
    assert d["source"] == "derived"
    assert set(B.submission_digests(without)[1]) >= set(CMDS)


def test_every_declared_component_and_the_residual_get_a_digest(tmp_path):
    B = _mod()
    _, comps, _ = B.submission_digests(_ws(tmp_path))
    assert set(comps) == set(CMDS) | {UNATTRIBUTED}


def test_the_longest_declared_prefix_owns_a_file(tmp_path):
    """A nested grant must not be swallowed by its parent, or a whole subtree collapses to one component."""
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST.replace("  parse: [mlir_oot/parse.py]",
                                              "  parse: [mlir_oot/]")
    base = B.submission_digests(_ws(tmp_path, files, name="a"))[1]
    files2 = dict(files)
    files2["mlir_oot/lowering/tile.py"] = "tile v2\n"
    moved = B.submission_digests(_ws(tmp_path, files2, name="b"))[1]
    # tile.py is under BOTH `mlir_oot/` (parse) and `mlir_oot/lowering/` -- the longer prefix wins
    assert moved["lower_interface_to_target"] != base["lower_interface_to_target"]
    assert moved["parse"] == base["parse"]


def test_a_file_two_components_claim_equally_falls_to_the_residual(tmp_path):
    """A tie broken by manifest key order would let a cosmetic reordering re-attribute a file, and a
    certificate would then survive an edit it should not have. Ambiguous ownership means everyone's."""
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST.replace("  parse: [mlir_oot/parse.py]", "  parse: [mlir_oot/shared/]") \
                                     .replace("  emit_command_buffer: [mlir_oot/cmdbuf.py]",
                                              "  emit_command_buffer: [mlir_oot/shared/]")
    files["mlir_oot/shared/util.py"] = "util v1\n"
    base = B.submission_digests(_ws(tmp_path, files, name="a"))[1]
    files2 = dict(files)
    files2["mlir_oot/shared/util.py"] = "util v2\n"
    moved = B.submission_digests(_ws(tmp_path, files2, name="b"))[1]
    assert moved[UNATTRIBUTED] != base[UNATTRIBUTED]
    assert moved["parse"] == base["parse"] and moved["emit_command_buffer"] == base["emit_command_buffer"]


# ---------------------------------------------------------------------------------------------
# 3. THE FALSIFIER -- touch one component, see exactly who loses a certificate
# ---------------------------------------------------------------------------------------------
def _states(before, after, declared):
    """One CapsuleState per capsule, certified against `before`, now living on `after`."""
    (dw, cw, _), (dn, cn, _) = before, after
    return [CapsuleState(name=n, digest=dn, components=cn, depends_on=dep,
                         verdicts={"L2": Verdict(PASS, dw, cw), "L3": Verdict(PASS, dw, cw)})
            for n, dep in declared.items()]


# capsule -> what it declares. `whole_model` declares nothing on purpose: it is the fail-closed control.
DECLARED = {
    "isa_parse_smoke": ("parse",),
    "matmul_tile": ("lower_interface_to_target",),
    "matmul_cb": ("emit_command_buffer",),
    "conv_codegen": ("lower_target_to_llvm",),
    "fused_tile_cb": ("lower_interface_to_target", "emit_command_buffer"),
    "whole_model": None,
}

# which file to touch -> which capsules MUST lose their certificate.
TOUCHES = {
    "mlir_oot/parse.py": {"isa_parse_smoke", "whole_model"},
    "mlir_oot/lowering/tile.py": {"matmul_tile", "fused_tile_cb", "whole_model"},
    "mlir_oot/cmdbuf.py": {"matmul_cb", "fused_tile_cb", "whole_model"},
    "mlir_oot/codegen.py": {"conv_codegen", "whole_model"},
    "mlir_oot/opt": set(DECLARED),          # unattributed: a dependency of EVERY capsule
}


@pytest.mark.parametrize("touched", sorted(TOUCHES))
def test_invalidation_matrix(tmp_path, touched):
    """The measurement this whole change exists for: an edit invalidates exactly its dependents.

    Asserted as an exact set, not a count. A matrix where every touch invalidates everything is the bug
    this replaces; one where nothing is invalidated is a stale certificate presented as valid, which is
    strictly worse -- so both are failures here.
    """
    B = _mod()
    before = B.submission_digests(_ws(tmp_path, FILES, name="before"))
    files = dict(FILES)
    files[touched] = FILES[touched] + "# edited\n"
    after = B.submission_digests(_ws(tmp_path, files, name="after"))

    states = _states(before, after, DECLARED)
    lost = {s.name for s in states if s.known("L3") == UNKNOWN}
    assert lost == TOUCHES[touched]
    # and the survivors are still CERTIFIED, not merely "not requeued"
    for s in states:
        if s.name not in lost:
            assert s.known("L3") == PASS and s.known("L2") == PASS


def test_the_matrix_is_discriminating(tmp_path):
    """Guard on the guard: the five touches must not all produce the same answer, or the matrix above
    would pass against an implementation that invalidates everything (or nothing)."""
    B = _mod()
    before = B.submission_digests(_ws(tmp_path, FILES, name="before"))
    seen = set()
    for i, touched in enumerate(sorted(TOUCHES)):
        files = dict(FILES)
        files[touched] = FILES[touched] + "# edited\n"
        after = B.submission_digests(_ws(tmp_path, files, name=f"a{i}"))
        seen.add(frozenset(s.name for s in _states(before, after, DECLARED)
                           if s.known("L3") == UNKNOWN))
    assert len(seen) == 5, f"touches are not discriminating: {seen}"


def test_no_edit_at_all_keeps_every_certificate(tmp_path):
    B = _mod()
    d = B.submission_digests(_ws(tmp_path))
    assert all(s.known("L3") == PASS for s in _states(d, d, DECLARED))


# ---------------------------------------------------------------------------------------------
# 4. fail closed -- the direction that makes a stale certificate look valid
# ---------------------------------------------------------------------------------------------
def test_an_undeclared_dependency_set_means_everything():
    for dep in (None, ()):
        s = CapsuleState("A", digest="d2", depends_on=dep, components={"parse": "p1"},
                         verdicts={"L2": Verdict(PASS, "d1", {"parse": "p1"})})
        # `parse` did NOT move, but the capsule declared nothing, so the whole submission decides
        assert s.known("L2") == UNKNOWN
        assert [x.component for x in s.invalidated_by("L2")] == [WHOLE_SUBMISSION]


def test_a_verdict_predating_the_decomposition_is_undeterminable():
    """Not-yet-known and undeterminable are different states and must not collapse: a verdict with no
    component map is evidence of nothing, so it re-runs."""
    s = CapsuleState("A", digest="d1", depends_on=("parse",), components={"parse": "p1"},
                     verdicts={"L2": Verdict(PASS, "d1")})       # legacy row: no components
    assert s.known("L2") == UNKNOWN
    assert [(x.component, x.reason) for x in s.invalidated_by("L2")] == [("parse", UNDETERMINABLE)]


def test_a_dependency_this_submission_does_not_have_is_undeterminable():
    s = CapsuleState("A", digest="d1", depends_on=("no_such_stage",),
                     components={"parse": "p1", UNATTRIBUTED: "u1"},
                     verdicts={"L2": Verdict(PASS, "d1", {"parse": "p1", UNATTRIBUTED: "u1"})})
    assert s.known("L2") == UNKNOWN
    assert ("no_such_stage", UNDETERMINABLE) in [(x.component, x.reason) for x in s.invalidated_by("L2")]


def test_no_verdict_is_reported_as_not_yet_known_not_as_changed():
    s = CapsuleState("A", digest="d1", depends_on=("parse",), components={"parse": "p1"})
    assert [(x.component, x.reason) for x in s.invalidated_by("L3")] == [(WHOLE_SUBMISSION, NO_VERDICT)]


# ---------------------------------------------------------------------------------------------
# 5. the report: WHICH component requeued a capsule
# ---------------------------------------------------------------------------------------------
def test_the_queue_names_the_component_that_requeued_the_capsule():
    s = CapsuleState("matmul_tile", digest="d2", depends_on=("lower_interface_to_target",),
                     components={"lower_interface_to_target": "l2", UNATTRIBUTED: "u1"},
                     verdicts={"L2": Verdict(PASS, "d1", {"lower_interface_to_target": "l1",
                                                          UNATTRIBUTED: "u1"})})
    q = schedule([s], tier_order=["L2", "L3"], cert_tiers=("L3",), cost_s={"L2": 2.5, "L3": 300.0})
    assert len(q) == 1 and q[0].tier == "L2"
    assert "lower_interface_to_target" in q[0].reason and CHANGED in q[0].reason


def test_explain_reports_the_invalidating_component_per_capsule():
    fresh = CapsuleState("A", digest="d2", depends_on=("parse",),
                         components={"parse": "p1", UNATTRIBUTED: "u1"},
                         verdicts={"L2": Verdict(PASS, "d1", {"parse": "p1", UNATTRIBUTED: "u1"}),
                                   "L3": Verdict(PASS, "d1", {"parse": "p1", UNATTRIBUTED: "u1"})})
    moved = CapsuleState("B", digest="d2", depends_on=("emit_command_buffer",),
                         components={"emit_command_buffer": "c2", UNATTRIBUTED: "u1"},
                         verdicts={"L2": Verdict(PASS, "d1", {"emit_command_buffer": "c1",
                                                              UNATTRIBUTED: "u1"})})
    rep = explain([fresh, moved], tier_order=["L2", "L3"], cert_tiers=("L3",),
                  cost_s={"L2": 2.5, "L3": 300.0})
    assert rep["unchanged"] == ["A"]
    assert rep["invalidated_by"] == [{"capsule": "B", "tier": "L2",
                                      "component": "emit_command_buffer", "reason": CHANGED}]


# ---------------------------------------------------------------------------------------------
# 6. end to end through the broker
# ---------------------------------------------------------------------------------------------
CORPUS = {"matmul_tile": "lower_interface_to_target", "conv_codegen": "lower_target_to_llvm"}


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    for name, dep in CORPUS.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "capsule.yaml").write_text(
            f"name: {name}\nkind: isa\nlabel: public\ndepends_on: [{dep}]\n")
    (root / "undeclared").mkdir(parents=True, exist_ok=True)
    (root / "undeclared" / "capsule.yaml").write_text("name: undeclared\nkind: isa\nlabel: public\n")
    return root


def test_capsule_dependencies_reads_each_capsules_own_declaration(tmp_path):
    B = _mod()
    got = B.capsule_dependencies([_corpus(tmp_path)])
    assert got == {n: (d,) for n, d in CORPUS.items()}
    # a capsule that declares nothing gets NO entry, so `depends_on` stays None -> whole submission
    assert "undeclared" not in got


def test_promote_requeues_only_the_dependent_capsule_after_an_edit(tmp_path, monkeypatch, capsys):
    """The plumbing, not just the policy: two real `promote()` calls across one edit.

    The first certifies both capsules; the edit touches the tiling component only; the second must
    re-enqueue the cert tier for `matmul_tile` alone and say which component did it. Under the old
    whole-submission digest both would requeue -- which at minutes per capsule is the cost this change
    exists to remove.
    """
    B = _mod()
    import json

    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),))
    v = {"per_capsule": [{"capsule": n, "pass": True} for n in CORPUS]}

    assert sorted(B.promote(ws, ch, v, "L2", "L3", None, sys.stderr)) == ["conv_codegen", "matmul_tile"]
    st = json.loads((ws / "qa" / "tier_state.json").read_text())
    assert set(st["matmul_tile"]["L2"]["components"]) == set(CMDS) | {UNATTRIBUTED}
    for f in ch.glob("simreq_*.json"):
        f.unlink()

    (ws / "submission" / "mlir_oot" / "lowering" / "tile.py").write_text("tile v2\n")
    assert B.promote(ws, ch, v, "L2", "L3", None, sys.stderr) == ["matmul_tile"]
    assert [json.loads(f.read_text())["capsules"] for f in ch.glob("simreq_*.json")] == ["matmul_tile"]
    err = capsys.readouterr().err
    assert "matmul_tile L3 invalidated by lower_interface_to_target (changed)" in err
    assert "conv_codegen L3 invalidated" not in err
