"""``check_inert_capabilities`` must keep detecting the four defects that motivated it.

An inert capability is one that is DECLARED and cannot fire, and its whole signature is silence.
The gate that finds them is itself a capability that can go inert -- a detector whose predicate
stops matching reports "clean" exactly like a detector that found nothing. So every detector here
is pinned by a MUTATION PAIR: a fixture in the broken shape, which must be reported, and the same
fixture in the fixed shape, which must not be. A detector that stops working fails the first half;
a detector that has become a rubber stamp fails the second.

The fixtures are written here rather than pointed at the live tree on purpose. Three of the four
motivating defects are already fixed, and the fourth is one commit from being; a test that asserted
against the real source would go green when somebody repaired the code and take the detector's
coverage with it.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest

from merlin.common import paths

CHECKER = paths.repo_root() / "build_tools" / "scripts" / "check_inert_capabilities.py"


def _load_checker():
    """Import the gate as a module (it lives in ``build_tools/scripts``, which is not a package)."""
    spec = importlib.util.spec_from_file_location("check_inert_capabilities", CHECKER)
    assert spec and spec.loader, f"cannot load {CHECKER}"
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE executing: the gate defines dataclasses, and `dataclasses` resolves a
    # field's annotations through `sys.modules[cls.__module__]`. Without this every test errors
    # in collection with an AttributeError that says nothing about the gate.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def _write(root, rel: str, body: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _run(checker, root, *, kinds: str, extra: list[str] | None = None) -> list[dict]:
    """Run the gate over a fixture tree and return its findings."""
    argv = ["--repo-root", str(root), "--scan-root", "lib", "--witness-root", "lib",
            "--mention-root", "docs", "--kinds", kinds, "--no-imports", "--no-ratchet", "--json"]
    argv += extra or []
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        checker.main(argv)
    return json.loads(buf.getvalue())["findings"]


# ---------------------------------------------------------------------------------------------
# 1. lower_conv_int8 -- a pass whose predicate no earlier stage can satisfy.
#    Not statically decidable (it depends on what a frontend OUTSIDE this repo emits), so the
#    detector is the MEASURED one: a healthy candidate population, zero transformed.
# ---------------------------------------------------------------------------------------------

#: The real counters ``passes_quant_int.lower_conv_int8`` reports on a deepjscc int8 capture: 190
#: ops carry convolution provenance, four generics have a compound map (all broadcasts), and the
#: pass lowers none of them, because the frontend already expanded every conv into im2col+matmul.
CONV_REPORT_INERT = {"lower_conv_int8": {"generics_scanned": 280, "compound_map_generics": 4,
                                         "windowed_map_generics": 0, "conv_prov_ops": 190,
                                         "lowered": 0}}
#: The same pass on a frontend that hands it a fused conv. Same population, non-zero work.
CONV_REPORT_LIVE = dict(CONV_REPORT_INERT)
CONV_REPORT_LIVE["lower_conv_int8"] = dict(CONV_REPORT_INERT["lower_conv_int8"], lowered=190)


def test_runtime_inert_flags_a_pass_that_saw_candidates_and_transformed_none(checker, tmp_path):
    rp = tmp_path / "quant_report.json"
    rp.write_text(json.dumps(CONV_REPORT_INERT), encoding="utf-8")
    found, notes = checker.detect_runtime_inert([rp])
    assert [f.ident for f in found] == ["lower_conv_int8"], (found, notes)
    assert "190" in found[0].blocked_by or "280" in found[0].blocked_by


def test_runtime_inert_is_silent_when_the_pass_actually_fires(checker, tmp_path):
    """MUTATION: same population, ``lowered`` non-zero. The detector must NOT report it."""
    rp = tmp_path / "quant_report.json"
    rp.write_text(json.dumps(CONV_REPORT_LIVE), encoding="utf-8")
    found, _ = checker.detect_runtime_inert([rp])
    assert found == []


def test_runtime_inert_refuses_to_call_a_missing_report_clean(checker):
    """"We did not look" must never read as "nothing is inert" -- the failure this repo keeps
    hitting. With no report the detector returns a NOTE and zero findings, and the note says so."""
    found, notes = checker.detect_runtime_inert([])
    assert found == []
    assert notes and "NOT evaluated" in notes[0]


# ---------------------------------------------------------------------------------------------
# 2. interface.CommitOp -- a five-member epilogue vocabulary whose only builder wrote ArrayAttr([])
# ---------------------------------------------------------------------------------------------

_EPILOGUE_DIALECT = '''
    KNOWN_EPILOGUE = {"bias", "bias_add", "requant", "relu", "maxpool"}

    class CommitOp:
        def verify_(self):
            for stage in self.properties["epilogue"]:
                if stage not in KNOWN_EPILOGUE:
                    raise VerifyException("unknown epilogue stage")
'''

_EPILOGUE_LOWERING_BROKEN = '''
    def lower(op, acc_type, out_t):
        commit = CommitOp(operands=[op], result_types=[out_t], properties={
            "epilogue": ArrayAttr([]),
            "output_dtype": StringAttr("i8")})
        return commit
'''

_EPILOGUE_LOWERING_FIXED = '''
    def lower(op, acc_type, out_t, fused):
        stages = [StringAttr("bias_add")] if fused is not None else []
        commit = CommitOp(operands=[op], result_types=[out_t], properties={
            "epilogue": ArrayAttr(stages),
            "output_dtype": StringAttr("i8")})
        return commit
'''


def _epilogue_tree(tmp_path, lowering: str):
    root = tmp_path / "repo"
    _write(root, "lib/interface.py", _EPILOGUE_DIALECT)
    _write(root, "lib/interface_lowering.py", lowering)
    return root


def test_always_empty_field_flags_the_hardcoded_empty_epilogue(checker, tmp_path):
    root = _epilogue_tree(tmp_path, _EPILOGUE_LOWERING_BROKEN)
    ids = [f["id"] for f in _run(checker, root, kinds="always-empty-field")]
    assert any(i.endswith(":epilogue") for i in ids), ids


def test_always_empty_field_clears_once_a_stage_can_be_built(checker, tmp_path):
    """MUTATION: the builder now computes ``stages``, so the field is no longer decidably empty."""
    root = _epilogue_tree(tmp_path, _EPILOGUE_LOWERING_FIXED)
    ids = [f["id"] for f in _run(checker, root, kinds="always-empty-field")]
    assert not any(i.endswith(":epilogue") for i in ids), ids


def test_unproduced_member_flags_a_stage_nothing_constructs(checker, tmp_path):
    """The second half of the same defect: even with ``bias_add`` reachable, ``maxpool`` is
    validated for and built by nothing. The verifier admits a value the compiler cannot emit."""
    root = _epilogue_tree(tmp_path, _EPILOGUE_LOWERING_FIXED)
    ids = [f["id"] for f in _run(checker, root, kinds="unproduced-member")]
    assert any(i.endswith("KNOWN_EPILOGUE:maxpool") for i in ids), ids
    assert not any(i.endswith("KNOWN_EPILOGUE:bias_add") for i in ids), ids


def test_unproduced_member_accepts_a_member_produced_only_by_data(checker, tmp_path):
    """MUTATION: a capsule yaml that emits the stage. A value a DATA file can produce is not
    inert, and a census that reads only Python would manufacture the finding."""
    root = _epilogue_tree(tmp_path, _EPILOGUE_LOWERING_FIXED)
    _write(root, "lib/capsule.yaml", "epilogue: [maxpool]\n")
    ids = [f["id"] for f in _run(checker, root, kinds="unproduced-member")]
    assert not any(i.endswith("KNOWN_EPILOGUE:maxpool") for i in ids), ids


# ---------------------------------------------------------------------------------------------
# 3. bundle_footprint -- a fit check whose total was computed non-recursively
# ---------------------------------------------------------------------------------------------

_FOOTPRINT_BROKEN = '''
    def bundle_footprint(root, budget_bytes=0):
        total = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        return {"resident_bytes": total, "fits": total <= budget_bytes}
'''
_FOOTPRINT_FIXED = '''
    def bundle_footprint(root, budget_bytes=0):
        total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        return {"resident_bytes": total, "fits": total <= budget_bytes}
'''


def test_nonrecursive_aggregate_flags_a_total_over_a_flat_listing(checker, tmp_path):
    root = tmp_path / "repo"
    _write(root, "lib/et_campaign.py", _FOOTPRINT_BROKEN)
    ids = [f["id"] for f in _run(checker, root, kinds="nonrecursive-aggregate")]
    assert any(i.endswith(":bundle_footprint") for i in ids), ids


def test_nonrecursive_aggregate_clears_once_the_walk_recurses(checker, tmp_path):
    """MUTATION: ``iterdir()`` -> ``rglob("*")``. The 1.83 GB one directory down now counts."""
    root = tmp_path / "repo"
    _write(root, "lib/et_campaign.py", _FOOTPRINT_FIXED)
    assert _run(checker, root, kinds="nonrecursive-aggregate") == []


# ---------------------------------------------------------------------------------------------
# 4. a gate that cannot fail -- the constant-golden shape, generalised
# ---------------------------------------------------------------------------------------------

_GATE_BROKEN = '''
    def check_outputs_match(run, golden):
        if golden is None:
            return True
        for a, b in zip(run, golden):
            cos = _cosine(a, b)
        return True
'''
_GATE_FIXED = '''
    def check_outputs_match(run, golden):
        if golden is None:
            return False
        for a, b in zip(run, golden):
            if _cosine(a, b) < 0.999:
                return False
        return True
'''


def test_tautological_gate_flags_a_verdict_with_no_failure_path(checker, tmp_path):
    root = tmp_path / "repo"
    _write(root, "lib/grade.py", _GATE_BROKEN)
    ids = [f["id"] for f in _run(checker, root, kinds="tautological-gate")]
    assert any(i.endswith(":check_outputs_match") for i in ids), ids


def test_tautological_gate_clears_once_a_falsy_return_exists(checker, tmp_path):
    """MUTATION: one reachable ``return False`` is enough to make it a real gate."""
    root = tmp_path / "repo"
    _write(root, "lib/grade.py", _GATE_FIXED)
    assert _run(checker, root, kinds="tautological-gate") == []


def test_self_comparison_flags_a_tautological_conjunct(checker, tmp_path):
    root = tmp_path / "repo"
    _write(root, "lib/consistency.py", '''
        def audit(inventory, names):
            chk(all("exists" in r for r in inventory) and (names == names),
                "inventory records exists for every artifact")
    ''')
    ids = [f["id"] for f in _run(checker, root, kinds="self-comparison")]
    assert any("names == names" in i for i in ids), ids


def test_self_comparison_spares_the_nan_idiom(checker, tmp_path):
    """MUTATION: ``v == v`` beside a finiteness test is the NaN idiom, not a defect."""
    root = tmp_path / "repo"
    _write(root, "lib/codec.py", '''
        def table(decode):
            out = {}
            for c in range(256):
                v = decode(c)
                if v == v and abs(v) != float("inf"):
                    out[v] = c
            return out
    ''')
    assert _run(checker, root, kinds="self-comparison") == []


# ---------------------------------------------------------------------------------------------
# 5. a gate whose work list comes from a command whose exit status it never reads
# ---------------------------------------------------------------------------------------------

_STAGED_BROKEN = '''
    def _iter_targets(staged):
        out = subprocess.run(["git", "diff", "--cached", "--name-only"],
                             capture_output=True, text=True).stdout
        return [ln for ln in out.splitlines() if ln.strip()]
'''
_STAGED_FIXED = '''
    def _iter_targets(staged):
        got = subprocess.run(["git", "diff", "--cached", "--name-only"],
                             capture_output=True, text=True)
        if got.returncode != 0:
            raise SystemExit("could not read the index; refusing to report a clean tree")
        return [ln for ln in got.stdout.splitlines() if ln.strip()]
'''


def test_unchecked_subprocess_input_flags_a_worklist_from_an_unchecked_command(checker, tmp_path):
    root = tmp_path / "repo"
    _write(root, "lib/gate.py", _STAGED_BROKEN)
    ids = [f["id"] for f in _run(checker, root, kinds="unchecked-subprocess-input")]
    assert any(i.endswith(":_iter_targets") for i in ids), ids


def test_unchecked_subprocess_input_clears_once_the_status_is_read(checker, tmp_path):
    """MUTATION: the sibling gate ``check_no_answer_keys`` already fails closed this way."""
    root = tmp_path / "repo"
    _write(root, "lib/gate.py", _STAGED_FIXED)
    assert _run(checker, root, kinds="unchecked-subprocess-input") == []


# ---------------------------------------------------------------------------------------------
# 6. registered-but-unranked, in BOTH directions
# ---------------------------------------------------------------------------------------------

def test_registry_asymmetry_reports_both_directions(checker, tmp_path, monkeypatch):
    """A name the search offers but the registry cannot resolve is INVISIBLE (the composition
    check swallows the KeyError); a name the registry holds that nothing offers is never proposed.
    Both are silent, so both must be reported."""
    pkg = tmp_path / "regfix"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "producer.py").write_text(
        "def known():\n    return ['ranked_and_registered', 'registered_only']\n", encoding="utf-8")
    (pkg / "consumer.py").write_text(
        "RANKED = [('ranked_and_registered', False), ('ranked_only', True)]\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(checker, "REGISTRY_PAIRS", [
        ("fixture", ("regfix.producer", "known", "call"),
         ("regfix.consumer", "RANKED", "first-of-pairs"))])
    corpus = checker._Corpus(root=tmp_path)
    found, notes = checker.detect_registry_asymmetry(corpus, enabled=True)
    ids = {f.ident for f in found}
    assert "fixture:consumed-not-registered:ranked_only" in ids, (ids, notes)
    assert "fixture:registered-not-consumed:registered_only" in ids, (ids, notes)
    assert not any("ranked_and_registered" in i for i in ids), ids


def test_registry_asymmetry_refuses_to_decide_without_imports(checker):
    """Registration here is an import side effect. Reading the tables statically would under-count
    the registry and invent findings, so with ``--no-imports`` the axis reports UNDECIDED."""
    found, notes = checker.detect_registry_asymmetry(checker._Corpus(), enabled=False)
    assert found == []
    assert notes and "SKIPPED" in notes[0]


# ---------------------------------------------------------------------------------------------
# 7. dead config
# ---------------------------------------------------------------------------------------------

def test_dead_env_knob_flags_a_documented_knob_nothing_reads(checker, tmp_path):
    root = tmp_path / "repo"
    _write(root, "lib/toolchain.py", 'CLANG = os.environ.get("MERLIN_CLANG")\n')
    _write(root, "docs/guide.md", "Set `MERLIN_CLANG` and `MERLIN_PHANTOM_KNOB` before building.\n")
    ids = [f["id"] for f in _run(checker, root, kinds="dead-env-knob")]
    assert ids == ["MERLIN_PHANTOM_KNOB"], ids


def test_dead_env_knob_honours_a_composed_read(checker, tmp_path):
    """MUTATION: ``os.environ.get(f"MERLIN_{source.upper()}_REPO")`` really does read
    MERLIN_TRITON_REPO, which appears as a literal nowhere. Head AND tail must match, so the bare
    ``MERLIN_`` head does not excuse every knob in the namespace."""
    root = tmp_path / "repo"
    _write(root, "lib/index.py", '''
        def repo_for(source):
            return os.environ.get(f"MERLIN_{source.upper()}_REPO")
    ''')
    _write(root, "docs/guide.md", "Set `MERLIN_TRITON_REPO`, and `MERLIN_PHANTOM_KNOB`.\n")
    ids = [f["id"] for f in _run(checker, root, kinds="dead-env-knob")]
    assert ids == ["MERLIN_PHANTOM_KNOB"], ids


def test_unreferenced_def_flags_a_private_helper_nothing_names(checker, tmp_path):
    root = tmp_path / "repo"
    _write(root, "lib/mod.py", '''
        def _used():
            return 1

        def _orphan():
            return 2

        def entry():
            return _used()
    ''')
    ids = [f["id"] for f in _run(checker, root, kinds="unreferenced-def")]
    assert [i.rsplit(":", 1)[-1] for i in ids] == ["_orphan"], ids


def test_unreferenced_def_spares_a_name_reached_by_dynamic_dispatch(checker, tmp_path):
    """MUTATION: a name that appears as a STRING is reachable through ``getattr``/a registry, and
    calling it dead would be a guess."""
    root = tmp_path / "repo"
    _write(root, "lib/mod.py", '''
        def _orphan():
            return 2

        ENTRIES = {"handler": "_orphan"}
    ''')
    assert _run(checker, root, kinds="unreferenced-def") == []


# ---------------------------------------------------------------------------------------------
# 8. the ratchet: pre-existing debt is carried, anything new fails, and the count may only fall
# ---------------------------------------------------------------------------------------------

def _seeded_tree(tmp_path):
    root = tmp_path / "repo"
    _write(root, "lib/grade.py", _GATE_BROKEN)
    return root


def test_ratcheted_finding_passes_and_a_new_one_fails(checker, tmp_path):
    root = _seeded_tree(tmp_path)
    ratchet = tmp_path / "ratchet.txt"
    base = ["--repo-root", str(root), "--scan-root", "lib", "--witness-root", "lib",
            "--kinds", "tautological-gate", "--no-imports", "--ratchet", str(ratchet)]

    assert checker.main(base + ["--write-ratchet"]) == 0
    assert "tautological-gate" in ratchet.read_text(encoding="utf-8")
    assert checker.main(base) == 0, "a ratcheted finding must not fail the gate"

    _write(root, "lib/grade2.py", _GATE_BROKEN.replace("check_outputs_match", "verify_shapes"))
    assert checker.main(base) == 1, "a finding outside the ratchet must fail the gate"


def test_ratchet_reports_entries_that_no_longer_reproduce(checker, tmp_path):
    """The ratchet may only SHRINK, so a fixed defect must be surfaced as a line to delete rather
    than sit in the file forever pretending to be debt."""
    root = _seeded_tree(tmp_path)
    ratchet = tmp_path / "ratchet.txt"
    ratchet.write_text("tautological-gate lib/gone.py:check_vanished\n", encoding="utf-8")
    findings, _ = checker.run([root / "lib"], [root / "lib"], root=root,
                              kinds=["tautological-gate"], report_paths=[], imports=False,
                              env_prefix="MERLIN_", mention_roots=[])
    stale = set(checker.load_ratchet(ratchet)) - {f.key for f in findings}
    assert stale == {"tautological-gate lib/gone.py:check_vanished"}


def test_ratchet_ids_are_line_independent(checker, tmp_path):
    """A ratchet keyed on a line number would go stale on every unrelated edit above the defect,
    and the debt file would grow by re-reporting what it already carries."""
    root = _seeded_tree(tmp_path)
    before = [f["id"] for f in _run(checker, root, kinds="tautological-gate")]
    # Dedent BEFORE prepending: a flush first line would make `textwrap.dedent` a no-op and
    # leave the rest indented, so the fixture would not parse and the test would pass vacuously.
    _write(root, "lib/grade.py", "# a new comment line\n" + textwrap.dedent(_GATE_BROKEN))
    after = [f["id"] for f in _run(checker, root, kinds="tautological-gate")]
    assert before == after and before, (before, after)


# ---------------------------------------------------------------------------------------------
# 9. the gate's own hygiene
# ---------------------------------------------------------------------------------------------

def test_every_declared_kind_is_wired(checker):
    """A detector named in ``KINDS`` but never dispatched would be an inert capability inside the
    inert-capability gate: ``--kinds`` would accept it and it would find nothing, forever."""
    import ast
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    dispatched = {n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.In)
                  and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str)
                  and isinstance(n.comparators[0], ast.Name)
                  and n.comparators[0].id == "kinds"
                  for n in [n.left]}
    assert set(checker.KINDS) == dispatched, set(checker.KINDS) ^ dispatched
    assert set(checker.KINDS) == set(checker.RANK), "every kind needs a consequence rank"


def test_checker_uses_no_regex(checker):
    """This tree forbids regex in library and tooling code; the gate must parse structurally."""
    import ast
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name != "re" for a in node.names), "check_inert_capabilities imports re"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "re", "check_inert_capabilities imports from re"


def test_cli_runs_over_the_real_tree_and_emits_a_machine_readable_count():
    """End to end: the gate must actually run here, and print a count CI can assert never rises."""
    out = subprocess.run([sys.executable, str(CHECKER), "--json"],
                         capture_output=True, text=True, cwd=str(paths.repo_root()), timeout=1800)
    payload = json.loads(out.stdout)
    for key in ("ratchet_count", "finding_count", "new_count", "by_kind", "findings", "notes"):
        assert key in payload, key
    assert payload["new_count"] == 0, (
        "new inert capabilities outside the ratchet: " + ", ".join(payload["new"]))
    assert any("NOT evaluated" in n for n in payload["notes"]), (
        "with no --report the gate must SAY the measured axis was not evaluated; a clean static "
        "run is not evidence that no pass is inert")
