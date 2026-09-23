"""Metadata extraction preserves evaluator semantics without importing its engines."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_golden, golden_provenance


@pytest.fixture(params=[golden_provenance, capsule_golden], ids=["core", "legacy"])
def queries(request):
    return request.param


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ("", "merlin_tensor_int"),
        ("null", "merlin_tensor_int"),
        ("{}", "merlin_tensor_int"),
        ("[]", "merlin_tensor_int"),
        ("0", "merlin_tensor_int"),
        ("false", "merlin_tensor_int"),
        ("''", "merlin_tensor_int"),
        ("golden_source: null", "merlin_tensor_int"),
        ("golden_source: ''", "merlin_tensor_int"),
        ("golden_source: 0", "merlin_tensor_int"),
        ("golden_source: false", "merlin_tensor_int"),
        ("golden_source: merlin_tensor_int", "merlin_tensor_int"),
        ("golden_source: independent_fixture", "independent_fixture"),
        ("golden_source: 7", 7),
        ("golden_source: [independent_fixture]", ["independent_fixture"]),
    ],
)
def test_source_projection_preserves_yaml_values(queries, tmp_path, document, expected):
    (tmp_path / "golden.yaml").write_text(document, encoding="utf-8")
    assert queries.golden_source({}, tmp_path) == expected


@pytest.mark.parametrize("document", ["[nonempty]", "scalar", "7", "true"])
def test_truthy_non_mapping_yaml_is_not_silently_normalized(queries, tmp_path, document):
    (tmp_path / "golden.yaml").write_text(document, encoding="utf-8")
    with pytest.raises(AttributeError, match="has no attribute 'get'"):
        queries.golden_source({}, tmp_path)


def test_parser_error_preserves_exact_string_input_diagnostic(queries, tmp_path):
    document = "golden_source: [unterminated"
    (tmp_path / "golden.yaml").write_text(document, encoding="utf-8")
    with pytest.raises(yaml.YAMLError) as original:
        yaml.safe_load(document)
    with pytest.raises(type(original.value)) as actual:
        queries.golden_source({}, tmp_path)
    assert str(actual.value) == str(original.value)
    assert "<unicode string>" in str(actual.value)


def test_read_error_still_propagates(queries, tmp_path, monkeypatch):
    (tmp_path / "golden.yaml").touch()

    def denied_read(self, **kwargs):
        raise PermissionError("synthetic permission failure")

    monkeypatch.setattr(Path, "read_text", denied_read)
    with pytest.raises(PermissionError, match="synthetic permission failure"):
        queries.golden_source({}, tmp_path)


def test_only_none_directory_uses_capsule_fallback(queries, tmp_path):
    (tmp_path / "golden.yaml").write_text("golden_source: independent_fixture", encoding="utf-8")
    capsule = {"__dir__": str(tmp_path)}
    assert queries.golden_source(capsule) == "independent_fixture"
    assert queries.golden_source(capsule, None) == "independent_fixture"
    assert queries.golden_source(capsule, "") == "merlin_tensor_int"
    assert queries.golden_source(capsule, tmp_path / "absent") == "merlin_tensor_int"
    assert queries.golden_source({}) == "merlin_tensor_int"
    other = tmp_path / "other"
    other.mkdir()
    (other / "golden.yaml").write_text("golden_source: explicit_fixture", encoding="utf-8")
    assert queries.golden_source(capsule, other) == "explicit_fixture"


@pytest.mark.parametrize("policy", [None, {}, {"compare": "exact_int"}, {"compare": "exact"}])
def test_integer_policy_never_queries_source(queries, monkeypatch, policy):
    def forbidden_source(*args, **kwargs):
        raise AssertionError("integer policy must short-circuit")

    monkeypatch.setattr(queries, "golden_source", forbidden_source)
    assert queries.is_independent_float_golden({"numeric_policy": policy}) is False


@pytest.mark.parametrize("compare", ["tolerance_float", "unrecognized", None, 0])
def test_non_integer_policy_keeps_original_source_lookup(queries, monkeypatch, compare):
    capsule = {"numeric_policy": {"compare": compare}, "__dir__": "unused"}
    seen = []

    def source(cap, directory):
        seen.append((cap, directory))
        return "independent_fixture"

    monkeypatch.setattr(queries, "golden_source", source)
    assert queries.is_independent_float_golden(capsule) is True
    assert seen == [(capsule, None)]
    monkeypatch.setattr(queries, "golden_source", lambda *args: "merlin_tensor_int")
    assert queries.is_independent_float_golden(capsule) is False


def test_legacy_loader_override_controls_source_and_classifier(monkeypatch):
    seen = []

    def overridden_loader(directory):
        seen.append(directory)
        return {"golden_source": "overridden_fixture", "outputs": {"Y": "PRIVATE_SENTINEL"}}

    monkeypatch.setattr(capsule_golden, "_load_golden_yaml", overridden_loader)
    capsule = {"__dir__": "fixture", "numeric_policy": {"compare": "tolerance_float"}}
    assert capsule_golden.golden_source(capsule) == "overridden_fixture"
    assert capsule_golden.is_independent_float_golden(capsule)
    assert seen == ["fixture", "fixture"]
    monkeypatch.setattr(capsule_golden, "_load_golden_yaml", lambda directory: None)
    assert not capsule_golden.is_independent_float_golden(capsule)


@pytest.mark.parametrize("override", ["loader", "source", "classifier"])
def test_loaded_evaluator_overrides_still_control_prompt_regime(monkeypatch, tmp_path, override):
    from merlin.targetgen.generate_prompt import (
        _corpus_golden_regimes,
        _corpus_uses_independent_float_goldens,
        _grading_model,
    )

    directory = tmp_path / "capsule"
    directory.mkdir()
    capsule = {"numeric_policy": {"compare": "tolerance_float"}}
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule), encoding="utf-8")
    (directory / "golden.yaml").write_text("golden_source: independent_fixture", encoding="utf-8")
    te = SimpleNamespace(capsule_corpus=tmp_path, corpus_siblings=lambda: [])
    assert _corpus_golden_regimes(te) == (True, False)
    if override == "loader":
        monkeypatch.setattr(capsule_golden, "_load_golden_yaml", lambda *args: None)
    elif override == "source":
        monkeypatch.setattr(capsule_golden, "golden_source", lambda *args: "merlin_tensor_int")
    else:
        monkeypatch.setattr(capsule_golden, "is_independent_float_golden", lambda *args: False)
    assert not golden_provenance.selected_independent_float_golden(capsule, directory)
    assert _corpus_golden_regimes(te) == (False, True)
    assert not _corpus_uses_independent_float_goldens(te)
    assert "exact-integer" in _grading_model(te)
    if override != "classifier":
        assert golden_provenance.selected_golden_source(capsule, directory) == "merlin_tensor_int"


def test_full_classifier_override_applies_even_to_integer_prompt(monkeypatch, tmp_path):
    from merlin.targetgen.generate_prompt import _corpus_golden_regimes

    directory = tmp_path / "capsule"
    directory.mkdir()
    (directory / "capsule.yaml").write_text("numeric_policy: {compare: exact_int}", encoding="utf-8")
    monkeypatch.setattr(capsule_golden, "is_independent_float_golden", lambda *args: True)
    te = SimpleNamespace(capsule_corpus=tmp_path, corpus_siblings=lambda: [])
    assert _corpus_golden_regimes(te) == (True, False)


def test_core_does_not_export_or_print_answer_payload(tmp_path, capsys):
    (tmp_path / "golden.yaml").write_text(
        "golden_source: independent_fixture\n"
        "outputs: {Y: PRIVATE_ANSWER_SENTINEL}\n"
        "oracle_provenance: PRIVATE_PROVENANCE_SENTINEL\n",
        encoding="utf-8",
    )
    assert golden_provenance.__all__ == [
        "golden_source",
        "is_independent_float_golden",
        "selected_golden_source",
        "selected_independent_float_golden",
    ]
    assert not hasattr(golden_provenance, "golden")
    assert not hasattr(golden_provenance, "_load_golden_yaml")
    assert golden_provenance.golden_source({}, tmp_path) == "independent_fixture"
    assert golden_provenance.is_independent_float_golden({"numeric_policy": {"compare": "tolerance_float"}}, tmp_path)
    assert capsys.readouterr() == ("", "")


def test_provenance_is_not_cached_across_source_changes(queries, tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text("golden_source: independent_fixture", encoding="utf-8")
    assert queries.golden_source({}, tmp_path) == "independent_fixture"
    path.write_text("golden_source: merlin_tensor_int", encoding="utf-8")
    assert queries.golden_source({}, tmp_path) == "merlin_tensor_int"


def test_synthetic_independent_golden_still_drives_numerical_grading(tmp_path):
    expected = {"Y": [[1.25, -2.5]]}
    (tmp_path / "golden.yaml").write_text(
        yaml.safe_dump({"golden_source": "independent_fixture", "outputs": expected}), encoding="utf-8"
    )
    policy = {"compare": "tolerance_float", "atol": 0.01, "rtol": 0.0}
    capsule = {"__dir__": str(tmp_path), "numeric_policy": policy}
    actual = capsule_golden.golden(capsule)
    assert actual == expected
    source = capsule_golden.golden_source(capsule)
    assert capsule_golden.compare(actual, expected, policy, golden_source=source)["status"] == "pass"
    mismatched = {"Y": [[1.25, -1.5]]}
    assert capsule_golden.compare(actual, mismatched, policy, golden_source=source)["status"] == "fail"


def test_full_prompt_core_only_uses_metadata_without_evaluator_or_answers(tmp_path):
    corpus = tmp_path / "isa"
    for name, compare in (("integer", "exact_int"), ("floating", "tolerance_float")):
        capsule = corpus / name
        capsule.mkdir(parents=True)
        (capsule / "capsule.yaml").write_text(
            yaml.safe_dump({"numeric_policy": {"compare": compare}}), encoding="utf-8"
        )
        (capsule / "golden.yaml").write_text(
            "golden_source: independent_fixture\noutputs: {Y: PRIVATE_ANSWER_SENTINEL}\n",
            encoding="utf-8",
        )
    private = tmp_path / "hidden" / "PRIVATE_CAPSULE_NAME"
    private.mkdir(parents=True)
    (private / "golden.yaml").write_text("PRIVATE_HOLDOUT_SENTINEL", encoding="utf-8")
    script = """
import importlib.abc
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
sys.path[:0] = sys.argv[1:3]
class RefuseEvaluator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        evaluators = ('merlin.targetgen.capsule_golden', 'merlin.targetgen.capsule_runner')
        optional = ('merlin_experiments', 'aet', 'torch')
        if fullname in evaluators or fullname.split('.')[0] in optional:
            raise AssertionError('optional evaluator imported: ' + fullname)
sys.meta_path.insert(0, RefuseEvaluator())
from merlin.targetgen import golden_provenance, generate_prompt
assert golden_provenance.is_independent_float_golden({}) is False
assert golden_provenance.selected_independent_float_golden({}) is False
assert golden_provenance.selected_golden_source({}) == 'merlin_tensor_int'
# Keep target discovery independent of the metadata seam under test. Prompt slots,
# corpus classification and the complete shared template remain production code.
bridge = ModuleType('merlin.targetgen.rtl.mlc_bridge')
bridge.fact_bundle_for = lambda target: {'fields': {}}
bridge.render_fact_bundle_for = lambda target, bundle: '# synthetic ISA'
sys.modules[bridge.__name__] = bridge
isa = ModuleType('merlin.targetgen.isa_model')
isa.isa_model_for_target = lambda target: SimpleNamespace(inst_width=32)
sys.modules[isa.__name__] = isa
from merlin.targetgen import operation_capabilities
operation_capabilities.operation_contract_for_target = lambda target, manifest: {}
te = SimpleNamespace(target='fixture', capsule_corpus=Path(sys.argv[3]),
    corpus_siblings=lambda: [], corpus_rel=lambda: 'fixture/isa/',
    prior_backends=[], isa_headers=[], hwbringup_set='')
manifest = SimpleNamespace(endpoint_kind='inline_asm_insn', contract={},
    tier_sim={'L2': 'fixture'}, fourth_output_name='')
assert generate_prompt._corpus_golden_regimes(te) == (True, True)
assert generate_prompt._corpus_uses_independent_float_goldens(te)
prompt = generate_prompt.render_prompt(te, manifest)
assert 'PER CAPSULE' in prompt
assert 'fixture-opt' in prompt
assert 'PRIVATE_' not in prompt
assert 'independent_fixture' not in prompt
print('core-only prompt verified')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(repo_root() / "src"),
            str(Path(yaml.__file__).parents[1]),
            str(corpus),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "core-only prompt verified\n"
    assert not result.stderr
