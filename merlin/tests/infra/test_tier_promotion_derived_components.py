"""The HARNESS-DERIVED component decomposition, and the reconciliation it exists to make possible.

A certificate is earned at time T by the bytes that were on disk at time T. The agent keeps editing after
that, so the question every round has to answer is: does this edit invalidate that certificate? Until now
the answer was always yes -- the only content address was the whole-submission digest, so every byte in
the tree was a dependency of every capsule. Measured on a live arm-4 round: 19 capsules cleared the loop
tier, 3 held a cert, and the promotion log was almost entirely
``invalidated by <whole-submission> (changed)``.

`tier_promote` already preferred an agent-declared ``components:`` block; when none was declared it fell
straight back to the whole submission. These tests pin the SECOND source -- the decomposition the harness
derives from the submission's own declared surface -- and, much more importantly, pin the direction that
would be a silent disaster: a certificate that survives an edit it should not have.

The falsifier is `test_reconciliation_matrix`. It edits one file at a time and asserts the EXACT set of
capsules that lose their cert. A matrix where everything invalidates everything is the bug being replaced;
one where nothing invalidates anything is a stale certificate presented as valid, which is strictly worse.
Both fail here. `test_the_matrix_discriminates` guards the guard.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.oracle_schedule import PASS, UNATTRIBUTED, UNKNOWN, CapsuleState, Verdict

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"

CMDS = ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")

# A package shaped the way the ABI describes: one entrypoint script per command, each importing what it
# needs from inside the submission. This is the shape the derivation can actually decompose -- and the
# shape the task prompt now asks for -- so the fixture is written that way on purpose.
MANIFEST = """\
artifact_type: mlir_oot_target_backend
target: t
package_id: p
language: python
authoring: {mode: hand_curated}
integrity_exempt: false
entrypoints: {tool: mlir_oot/parse_main.py}
commands:
  parse: {argv: ["mlir_oot/parse_main.py", "{input_mlir}"]}
  lower_interface_to_target: {argv: ["mlir_oot/lower_main.py", "{input_mlir}"]}
  emit_command_buffer: {argv: ["mlir_oot/cb_main.py", "--cb={output_json}", "{input_mlir}"]}
  lower_target_to_llvm: {argv: ["mlir_oot/artifact_main.py", "{input_mlir}"]}
"""

FILES = {
    "manifest.yaml": MANIFEST,
    # one entrypoint per command
    "mlir_oot/parse_main.py": "from mlir_oot import shared\nfrom mlir_oot import frontend\n",
    "mlir_oot/lower_main.py": "from mlir_oot import shared\nfrom mlir_oot.lowering import tile\n",
    "mlir_oot/cb_main.py": "from mlir_oot import shared\nfrom mlir_oot import cmdbuf\n",
    "mlir_oot/artifact_main.py": "from mlir_oot import shared\nfrom mlir_oot import codegen\n",
    "mlir_oot/__init__.py": "",
    "mlir_oot/shared.py": "TILE = 16\n",                     # imported by all four
    "mlir_oot/frontend.py": "def parse(t): return t\n",
    "mlir_oot/lowering/__init__.py": "",
    "mlir_oot/lowering/tile.py": "def tile(x): return x\n",
    "mlir_oot/cmdbuf.py": "def cb(x): return {}\n",
    "mlir_oot/codegen.py": "def emit(x): return ''\n",
    # a data file the codegen opens at a fixed path: reached, though nothing imports it
    "mlir_oot/tables/encodings.json": "{}\n",
    # not reached, but the same KIND of thing as reached code: a dead module today, a plugin tomorrow
    "mlir_oot/experimental_pass.py": "# not imported by anything\n",
    # not reached and not that kind of thing: the agent's own prose and a scratch listing
    "REPORT.md": "# report\n",
    "docs/iteration_notes.md": "round 1\n",
    "scratch_ops.S": ".word 0x0\n",
}
FILES["mlir_oot/codegen.py"] += 'TABLES = open("mlir_oot/tables/encodings.json").read()\n'


def _mod():
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
# 1. declare, else derive -- and never silently
# ---------------------------------------------------------------------------------------------
def test_a_submission_with_no_components_block_is_decomposed_by_the_harness(tmp_path):
    """The whole point: no declaration must no longer mean "every byte is every capsule's dependency"."""
    B = _mod()
    d = B.decomposition(_ws(tmp_path))
    assert d["source"] == "derived"
    assert set(d["names"]) >= set(CMDS)


def test_a_declared_block_is_preferred_over_the_derivation(tmp_path):
    """"Both: declare, else derive" -- an agent that says where its code lives is believed."""
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST + (
        "components:\n"
        "  parse: [mlir_oot/frontend.py]\n"
        "  lower_interface_to_target: [mlir_oot/lowering/]\n"
        "  emit_command_buffer: [mlir_oot/cmdbuf.py]\n"
        "  lower_target_to_llvm: [mlir_oot/codegen.py]\n")
    d = B.decomposition(_ws(tmp_path, files))
    assert d["source"] == "declared"
    # under the DECLARATION the entrypoint scripts belong to nobody, so they are unattributed; under the
    # derivation each belongs to its own command. The two answers differ, which is how we know which ran.
    assert d["owners"]["mlir_oot/parse_main.py"] == {UNATTRIBUTED}


def test_an_unknown_component_name_is_reported_and_never_silently_honoured(tmp_path):
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST + "components:\n  parse: [mlir_oot/frontend.py]\n  tiler: [mlir_oot/lowering/]\n"
    _, comps, rejected = B.submission_digests(_ws(tmp_path, files))
    assert rejected == ["tiler"] and "tiler" not in comps


# ---------------------------------------------------------------------------------------------
# 2. what the derivation attributes, and to whom
# ---------------------------------------------------------------------------------------------
def test_each_commands_entrypoint_and_imports_are_its_own(tmp_path):
    B = _mod()
    own = B.decomposition(_ws(tmp_path))["owners"]
    assert own["mlir_oot/lower_main.py"] == {"lower_interface_to_target"}
    assert own["mlir_oot/lowering/tile.py"] == {"lower_interface_to_target"}
    assert own["mlir_oot/cmdbuf.py"] == {"emit_command_buffer"}


def test_a_module_every_command_imports_belongs_to_every_command(tmp_path):
    """Shared code is a real dependency of each command that reaches it, not of an arbitrary one."""
    B = _mod()
    assert B.decomposition(_ws(tmp_path))["owners"]["mlir_oot/shared.py"] == set(CMDS)


def test_the_manifest_itself_is_a_dependency_of_every_command(tmp_path):
    """It declares the argv. Editing it can change what every command does, so it can never be inert."""
    B = _mod()
    assert B.decomposition(_ws(tmp_path))["owners"]["manifest.yaml"] == set(CMDS)


def test_a_data_file_opened_at_a_literal_path_is_attributed_not_inert(tmp_path):
    """Nothing imports it, so only the path literal in the reading module places it. Missing that would
    let an encoding table change under a certificate that depends on it."""
    B = _mod()
    own = B.decomposition(_ws(tmp_path))["owners"]
    assert own["mlir_oot/tables/encodings.json"] == {"lower_target_to_llvm"}


def test_an_unreached_module_is_unattributed_not_inert(tmp_path):
    """FAIL CLOSED. A dead module and a plugin loaded by a path this could not resolve look identical from
    a static read, so anything of the same KIND as reached code is a dependency of every capsule."""
    B = _mod()
    d = B.decomposition(_ws(tmp_path))
    assert d["owners"]["mlir_oot/experimental_pass.py"] == {UNATTRIBUTED}
    assert "mlir_oot/experimental_pass.py" not in d["inert"]


def test_only_files_outside_every_commands_reach_are_inert(tmp_path):
    B = _mod()
    d = B.decomposition(_ws(tmp_path))
    assert set(d["inert"]) == {"REPORT.md", "docs/iteration_notes.md", "scratch_ops.S"}


# ---------------------------------------------------------------------------------------------
# 3. the derivation REFUSES rather than guesses
# ---------------------------------------------------------------------------------------------
def test_a_language_whose_imports_cannot_be_traced_gets_no_decomposition(tmp_path):
    """A compiled tool's reads are invisible from here. Refusing returns exactly today's behaviour."""
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST.replace("language: python", "language: cpp")
    ws = _ws(tmp_path, files)
    assert B.derived_components(ws) is None
    assert B.decomposition(ws)["source"] is None
    assert B.submission_digests(ws)[1] == {}


def test_a_command_whose_argv_names_no_submission_file_gets_no_decomposition(tmp_path):
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST.replace('"mlir_oot/cb_main.py"', '"/usr/bin/true"')
    assert B.derived_components(_ws(tmp_path, files)) is None


def test_an_unparseable_file_in_the_surface_gets_no_decomposition(tmp_path):
    """Unknown content is unknown reads. A partial answer here is how a stale certificate survives."""
    B = _mod()
    files = dict(FILES)
    files["mlir_oot/shared.py"] = "def broken(:\n"
    assert B.derived_components(_ws(tmp_path, files)) is None


def test_refusing_leaves_the_whole_digest_byte_identical(tmp_path):
    """Adding this feature must not, by itself, invalidate one existing certificate."""
    B = _mod()
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST.replace("language: python", "language: cpp")
    a = _ws(tmp_path, files, name="a")
    b = _ws(tmp_path, files, name="b")
    assert B.submission_digests(a)[0] == B.submission_digests(b)[0] == B._submission_digest(a)


# ---------------------------------------------------------------------------------------------
# 4. THE FALSIFIER -- reconciliation: who loses a certificate when one file moves
# ---------------------------------------------------------------------------------------------
# capsule -> the dependency set the SCHEDULER is given. These are `oracle_schedule` policy fixtures,
# not capsule declarations: `depends_on` is no longer a capsule.yaml field (see `tier_promote`), and
# `promote()` supplies every capsule the full component set. The narrow sets are kept here because the
# scheduler must still discriminate per component -- that is what the derived decomposition buys.
# `whole_model` is given None: the control for "no dependency set means the whole submission".
DECLARED = {
    "isa_parse_smoke": ("parse",),
    "matmul_tile": ("lower_interface_to_target",),
    "matmul_cb": ("emit_command_buffer",),
    "conv_codegen": ("lower_target_to_llvm",),
    "whole_model": None,
}

# file touched -> capsules that MUST lose their certificate.
TOUCHES = {
    # inert: the agent's notes, its report, a scratch listing. NOBODY loses a certificate.
    "REPORT.md": set(),
    "docs/iteration_notes.md": set(),
    "scratch_ops.S": set(),
    # one command's own code
    "mlir_oot/lowering/tile.py": {"matmul_tile", "whole_model"},
    "mlir_oot/cmdbuf.py": {"matmul_cb", "whole_model"},
    "mlir_oot/tables/encodings.json": {"conv_codegen", "whole_model"},
    # shared code, the manifest, and an unreached module: EVERYONE loses it
    "mlir_oot/shared.py": set(DECLARED),
    "manifest.yaml": set(DECLARED),
    "mlir_oot/experimental_pass.py": set(DECLARED),
}


def _states(B, before, after, declared):
    (dw, cw, _), (dn, cn, _) = before, after
    default = tuple(sorted(cn)) or None
    return [CapsuleState(name=n, digest=dn, components=cn, depends_on=dep or default,
                         verdicts={"L2": Verdict(PASS, dw, cw), "L3": Verdict(PASS, dw, cw)})
            for n, dep in declared.items()]


def _touch(B, tmp_path, path, name):
    files = dict(FILES)
    files[path] = FILES[path] + ("\n# edited\n" if path.endswith(".py") else "\nedited\n")
    return B.submission_digests(_ws(tmp_path, files, name=name))


@pytest.mark.parametrize("touched", sorted(TOUCHES))
def test_reconciliation_matrix(tmp_path, touched):
    """One edit, and the exact set of certificates it may kill.

    The empty rows are the ones this change exists for; the full rows are the ones that keep it honest. A
    certificate that survives an edit to code it rides on is a wrong hardware verdict presented as a right
    one, so those rows are asserted just as hard as the savings.
    """
    B = _mod()
    before = B.submission_digests(_ws(tmp_path, FILES, name="before"))
    after = _touch(B, tmp_path, touched, "after")
    states = _states(B, before, after, DECLARED)
    lost = {s.name for s in states if s.known("L3") == UNKNOWN}
    assert lost == TOUCHES[touched]
    for s in states:                       # survivors are CERTIFIED, not merely "not requeued"
        if s.name not in lost:
            assert s.known("L3") == PASS and s.known("L2") == PASS


def test_the_matrix_discriminates(tmp_path):
    """Guard on the guard: the rows must not all give the same answer, or the matrix above would pass
    against an implementation that invalidates everything -- which is the bug being replaced."""
    B = _mod()
    before = B.submission_digests(_ws(tmp_path, FILES, name="before"))
    seen = set()
    for i, touched in enumerate(sorted(TOUCHES)):
        after = _touch(B, tmp_path, touched, f"a{i}")
        seen.add(frozenset(s.name for s in _states(B, before, after, DECLARED)
                           if s.known("L3") == UNKNOWN))
    assert len(seen) >= 4, f"rows are not discriminating: {seen}"


def test_no_edit_keeps_every_certificate(tmp_path):
    B = _mod()
    d = B.submission_digests(_ws(tmp_path))
    assert all(s.known("L3") == PASS for s in _states(_mod(), d, d, DECLARED))


# ---------------------------------------------------------------------------------------------
# 5. the same thing through the real `promote()` plumbing
# ---------------------------------------------------------------------------------------------
CORPUS = {"matmul_tile": "lower_interface_to_target", "conv_codegen": "lower_target_to_llvm"}


def _corpus(tmp_path):
    """A corpus whose capsules still carry the RETIRED `depends_on` key, which must buy them nothing."""
    root = tmp_path / "corpus"
    for name, dep in CORPUS.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "capsule.yaml").write_text(f"name: {name}\nkind: isa\nlabel: public\ndepends_on: [{dep}]\n")
    d = root / "whole_model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text("name: whole_model\nkind: model\nlabel: public\n")
    return root


def _promote_once(B, ws, ch, log):
    for f in ch.glob("simreq_*.json"):
        f.unlink()
    v = {"per_capsule": [{"capsule": n, "pass": True} for n in (*CORPUS, "whole_model")]}
    return B.promote(ws, ch, v, "L2", "L3", None, log)


def test_an_inert_edit_requeues_nothing_through_promote(tmp_path, monkeypatch, capsys):
    """The saving, end to end: the agent updates its notes and no cert job is bought."""
    B = _mod()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),), raising=False)

    assert sorted(_promote_once(B, ws, ch, sys.stderr)) == ["conv_codegen", "matmul_tile", "whole_model"]
    (ws / "submission" / "docs" / "iteration_notes.md").write_text("round 2\n")
    (ws / "submission" / "REPORT.md").write_text("# report v2\n")
    assert _promote_once(B, ws, ch, sys.stderr) == []
    assert list(ch.glob("simreq_*.json")) == []
    assert "invalidated by" not in capsys.readouterr().err


def test_a_live_edit_requeues_every_capsule_not_only_the_declared_dependents(tmp_path, monkeypatch,
                                                                             capsys):
    """The negative half of the same run, and the shape of the retired `depends_on`.

    The edit lands on the tiling pass. `matmul_tile` still carries a stale `depends_on` naming that
    command and `conv_codegen` carries one naming a different command -- and BOTH must lose their
    certificate anyway, because the grader runs every command for every capsule
    (`capsule_common.run_entrypoints`), so an edit to any of them can flip any verdict. Against the code
    that honoured the declaration, `conv_codegen` survived here, holding a certificate its current bytes
    had not earned.

    The saving is still real and still asserted -- by the sibling test above, where an edit to bytes no
    command can READ requeues nothing at all. That is the axis on which certificates survive; the
    capsule's own opinion never was.
    """
    B = _mod()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),), raising=False)

    _promote_once(B, ws, ch, sys.stderr)
    capsys.readouterr()
    (ws / "submission" / "mlir_oot" / "lowering" / "tile.py").write_text("def tile(x): return x + 1\n")
    assert sorted(_promote_once(B, ws, ch, sys.stderr)) == ["conv_codegen", "matmul_tile", "whole_model"]
    assert sorted(json.loads(f.read_text())["capsules"] for f in ch.glob("simreq_*.json")) == \
        ["conv_codegen", "matmul_tile", "whole_model"]
    err = capsys.readouterr().err
    for name in ("matmul_tile", "conv_codegen", "whole_model"):
        assert f"{name} L3 invalidated by lower_interface_to_target (changed)" in err


def test_promote_says_where_the_decomposition_came_from(tmp_path, monkeypatch, capsys):
    """A run whose promotion is silently running on the whole-submission fallback looks exactly like one
    that is not. The source is stated once per promote, so the log can be read."""
    B = _mod()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),), raising=False)
    _promote_once(B, ws, ch, sys.stderr)
    assert "[promote] components: derived" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# 6. the invariant the whole scheme rests on: a cert belongs to the bytes that earned it
# ---------------------------------------------------------------------------------------------
def test_a_completed_cert_is_recorded_against_the_pending_bytes_not_the_current_ones(tmp_path,
                                                                                     monkeypatch):
    """`record_cert` must never re-hash. The job was enqueued for the bytes that were pending; crediting
    the result to whatever is on disk when it lands is exactly the attribution bug the run must not have.
    """
    B = _mod()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),), raising=False)
    _promote_once(B, ws, ch, sys.stderr)
    pending = json.loads((ws / "qa" / "tier_state.json").read_text())["matmul_tile"]["L3"]
    assert pending["status"] == "pending"

    # the agent edits while the cert job runs
    (ws / "submission" / "mlir_oot" / "lowering" / "tile.py").write_text("def tile(x): return x + 2\n")
    B.record_cert(ws, {"per_capsule": [{"capsule": "matmul_tile", "pass": True}]}, "L3", sys.stderr)
    entry = json.loads((ws / "qa" / "tier_state.json").read_text())["matmul_tile"]["L3"]
    assert entry["status"] == "pass"
    assert entry["digest"] == pending["digest"] and entry["components"] == pending["components"]
    # and that recorded cert is NOT credited to the edited bytes
    now, comps, _ = B.submission_digests(ws)
    s = CapsuleState(name="matmul_tile", digest=now, components=comps,
                     depends_on=("lower_interface_to_target",),
                     verdicts={"L3": Verdict(entry["status"], entry["digest"], entry["components"])})
    assert s.known("L3") == UNKNOWN


def test_an_unattributable_cert_result_is_not_recorded(tmp_path):
    """No pending entry means we cannot say which bytes earned it. Recording nothing is the answer."""
    B = _mod()
    ws = _ws(tmp_path)
    (ws / "qa").mkdir(parents=True, exist_ok=True)
    assert B.record_cert(ws, {"per_capsule": [{"capsule": "matmul_tile", "pass": True}]}, "L3", None) == []
    assert not (ws / "qa" / "tier_state.json").exists()
