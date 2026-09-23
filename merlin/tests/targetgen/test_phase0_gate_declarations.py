"""Profile relocation cannot erase gate inventory or change runtime/corpus aliases."""

import importlib.util
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.spec import SpecError

from merlin.common.paths import repo_root


def _script(name):
    spec = importlib.util.spec_from_file_location(name, repo_root() / "build_tools/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gates(tmp_path, monkeypatch):
    modules = [_script(name) for name in ("check_phase_split", "check_defect_reach", "check_cert_affordability")]
    declarations = [
        SimpleNamespace(profile="root-alias", target="root-runtime"),
        SimpleNamespace(profile="short", target="qualified-runtime"),
    ]
    root = tmp_path / "contract/capsules"
    for relative, name in (("isa/base", "base"), ("short/isa/nested", "nested")):
        member = root / relative
        member.mkdir(parents=True)
        (member / "capsule.yaml").write_text(f"name: {name}\n")
    for module in modules:
        monkeypatch.setattr(module, "merlin_dir", lambda: tmp_path)
        monkeypatch.setattr(module, "all_declarations", lambda: tuple(declarations))
        monkeypatch.setattr(
            module,
            "for_target",
            lambda selected: next(row for row in declarations if selected in (row.profile, row.target)),
        )
    # Affordability's helper has a local paths import retained for compatibility.
    import merlin.common.paths

    monkeypatch.setattr(merlin.common.paths, "merlin_dir", lambda: tmp_path)
    return modules, declarations, root


def test_declared_inventory_survives_missing_profiles_directory(gates):
    (phase, defect, affordability), _, root = gates
    assert not (root / "profiles").exists()
    assert phase._targets(root) == ["root-alias", "short"]
    assert [p.parent.name for p in phase._capsules_for(root, "root-alias", {"short"})] == ["base"]
    assert [p.parent.name for p in phase._capsules_for(root, "short", {"short"})] == ["nested"]
    assert [row["name"] for row in defect._capsule_docs(root, "root-alias", {"short"})] == ["base"]
    assert affordability._default_corpus_target() == "root-alias"
    assert affordability._resolved_target("root-alias") == "root-runtime"
    assert affordability._resolved_target("short") == "qualified-runtime"
    assert affordability._target_label_for(root / "isa/base/capsule.yaml") == "root-alias"
    assert affordability._target_label_for(root / "short/isa/nested/capsule.yaml") == "short"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_empty_declarations_never_produce_empty_successful_gate(gates, monkeypatch, index):
    modules, _, _ = gates
    gate = modules[index]
    monkeypatch.setattr(gate, "all_declarations", lambda: ())
    with pytest.raises(ValueError, match="nonempty Phase 0 declaration inventory"):
        gate.main(["--json"])


def test_ambiguous_selector_refuses_before_fit_or_registry(gates, monkeypatch):
    (_, _, affordability), declarations, _ = gates
    declarations.append(SimpleNamespace(profile="short", target="different-runtime"))

    def ambiguous(_):
        raise SpecError("selector has two declarations")

    def forbidden(*args, **kwargs):
        pytest.fail("ambiguous selection must not query fits or a fallback registry")

    from merlin.targetgen import target_registry

    monkeypatch.setattr(affordability, "for_target", ambiguous)
    monkeypatch.setattr(affordability.CA, "fits_for", forbidden)
    monkeypatch.setattr(target_registry, "declared_target_for", forbidden)
    with pytest.raises(SpecError, match="two declarations"):
        affordability._resolved_target("short")
    with pytest.raises(SpecError, match="two declarations"):
        affordability._engine_fit("short", {})
    with pytest.raises(SpecError, match="two declarations"):
        affordability._selected_l3_engine("short")


def test_malformed_declaration_inventory_does_not_fall_back(gates, monkeypatch):
    (_, _, affordability), _, _ = gates

    def malformed():
        raise SpecError("invalid catalog")

    monkeypatch.setattr(affordability, "all_declarations", malformed)
    with pytest.raises(SpecError, match="invalid catalog"):
        affordability._resolved_target("unknown")


def test_absent_derivation_selector_keeps_historical_registry_resolution(gates, monkeypatch):
    from merlin.targetgen import target_registry

    (_, _, affordability), _, _ = gates
    monkeypatch.setattr(target_registry, "declared_target_for", lambda label: "historical-runtime")
    assert affordability._resolved_target("not-a-derivation") == "historical-runtime"


@pytest.mark.parametrize("selector", ["short", "qualified-runtime"])
def test_defect_gate_reports_alias_but_queries_runtime_target(gates, monkeypatch, capsys, selector):
    (_, defect, _), _, _ = gates
    calls = []
    monkeypatch.setattr(
        defect,
        "store_overflow_reach",
        lambda caps, *, target: (
            calls.append((target, [cap["name"] for cap in caps])) or {"capacity_elements": None, "over": []}
        ),
    )
    assert defect.main(["--target", selector, "--json"]) == 0
    report, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    assert list(report) == ["short"]
    assert report["short"]["n_capsules"] == 1
    assert calls == [("qualified-runtime", ["nested"])]


@pytest.mark.parametrize("selector", ["short", "qualified-runtime"])
def test_phase_gate_passes_runtime_identity_to_all_scientific_services(gates, monkeypatch, capsys, selector):
    (phase, _, _), _, _ = gates
    calls = []
    monkeypatch.setattr(phase.CC, "fit_for", lambda target: calls.append(("fit", target)))
    monkeypatch.setattr(phase.PP, "cycle_accurate_seen", lambda target: calls.append(("cycles", target)) or False)

    def split(caps, *, target, **kwargs):
        calls.append(("split", target))
        assert [row["name"] for row in caps] == ["nested"]
        return {"counts": {phase.PP.UNDETERMINED: 0}, "n_capsules": 1, "single_phase_reasons": {}, "verdicts": []}

    def anchors(caps, *, target, **kwargs):
        calls.append(("anchors", target))
        return {"n_obligations": 0, "n_paired": 0, "n_orphaned": 0}

    monkeypatch.setattr(phase.PP, "split_report", split)
    monkeypatch.setattr(phase.PP, "anchors", anchors)
    monkeypatch.setattr(phase.PP, "lever_reach_report", lambda caps: {"unreachable": {}})
    assert phase.main(["--target", selector, "--json"]) == 0
    report, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    assert "short" in report
    assert calls == [(role, "qualified-runtime") for role in ("fit", "cycles", "split", "anchors")]
