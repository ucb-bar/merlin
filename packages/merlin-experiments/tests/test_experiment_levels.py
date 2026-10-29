"""The displayed experiment level must not silently relabel a frozen treatment."""

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.adapters import PHASE1_MODULE
from merlin_experiments.cli import main
from merlin_experiments.phase1.__main__ import main as phase1_main
from merlin_experiments.phase1.levels import LEVELS, level_for_phase1
from merlin_experiments.runner import _verify_rtlcheck_support
from merlin_experiments.spec import SpecError, catalog, load_spec


def test_original_comparison_and_optional_branches() -> None:
    selections = (
        ({"arm": "raw_baseline", "treatment": "baseline", "bundle": "raw_baseline_public_v0"}, "EL1"),
        ({"arm": "cpp_merlininfra", "treatment": "baseline", "bundle": "cpp_merlininfra_public_v0"}, "EL2"),
        ({"arm": "merlin_assisted", "treatment": "baseline", "bundle": "merlin_assisted_public_v0"}, "EL3"),
        ({"arm": "merlin_assisted", "treatment": "rtlchecks", "bundle": "merlin_assisted_rtlchecks_public_v0"}, "EL4"),
        ({"arm": "merlin_assisted", "treatment": "baseline", "bundle": "merlin_assisted_eqsat_public_v0"}, "EL3-E"),
        ({"arm": "merlin_assisted", "treatment": "baseline", "bundle": "merlin_assisted_verify_public_v0"}, "EL4-V"),
        (
            {"arm": "merlin_assisted", "treatment": "rtlchecks", "bundle": "merlin_assisted_pass_catalog_public_v0"},
            "EL4-P",
        ),
    )
    assert len({entry["id"] for entry in LEVELS}) == len(LEVELS)
    for config, expected in selections:
        assert level_for_phase1(config) == expected


def test_contradictory_selection_has_no_level() -> None:
    assert level_for_phase1({"arm": "raw_baseline", "treatment": "rtlchecks"}) is None
    assert (
        level_for_phase1(
            {"arm": "merlin_assisted", "treatment": "baseline", "bundle": "merlin_assisted_rtlchecks_public_v0"}
        )
        is None
    )


def test_catalog_levels_materialize_exact_execution_keys() -> None:
    for path in catalog().values():
        spec = load_spec(path)
        phase = spec.document["phases"].get("1")
        if phase is None or phase["adapter"] != "capsule_bench":
            continue
        config = phase["config"]
        if config["level"] in ("EL4", "EL4-P"):
            assert (config["arm"], config["treatment"]) == ("merlin_assisted", "rtlchecks")
        elif config["level"] == "EL1":
            assert (config["arm"], config["treatment"]) == ("raw_baseline", "baseline")
        else:
            pytest.fail(f"unexpected authored level {config['level']}")
        assert level_for_phase1(config) == config["level"]


def test_level_refuses_conflicting_legacy_selection() -> None:
    from merlin_experiments.adapters import ADAPTERS

    adapter = ADAPTERS["capsule_bench"]
    with pytest.raises(SpecError, match="EL4 requires arm"):
        adapter.validate({"level": "EL4", "arm": "raw_baseline"})
    with pytest.raises(SpecError, match="contradicts the selected bundle"):
        adapter.validate({"level": "EL4", "bundle": "merlin_assisted_public_v0"})


def test_installed_phase1_level_preserves_legacy_execution_keys(tmp_path, monkeypatch) -> None:
    from merlin_experiments.phase1 import context, controller
    from merlin_experiments.phase1.feedback import rtlchecks

    manifest = tmp_path / "input_bundle_manifest.yaml"
    manifest.write_text("bundle_id: merlin_assisted_rtlchecks_public_v0\narm: merlin_rtlchecks\n")
    monkeypatch.setattr(context, "load_context", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rtlchecks, "treatment", lambda _context: object())
    observed = {}

    def capture(_context, options, **kwargs):
        observed.update(options=options, **kwargs)
        return 0

    monkeypatch.setattr(controller, "run", capture)
    assert (
        phase1_main(
            [
                "--descriptor",
                str(tmp_path / "descriptor.yaml"),
                "--repo",
                str(tmp_path),
                "--bundle-manifest",
                str(manifest),
                "--oracle-timing",
                str(tmp_path / "timing.json"),
                "--run-id",
                "example",
                "--bundle",
                "merlin_assisted_rtlchecks_public_v0",
                "--level",
                "EL4",
            ]
        )
        == 0
    )
    assert observed["options"].arm == "merlin_assisted"
    argv = observed["launcher_argv"]
    assert argv[argv.index("--level") + 1] == "EL4"
    assert argv[argv.index("--arm") + 1] == "merlin_assisted"
    assert argv[argv.index("--treatment") + 1] == "rtlchecks"


def test_cli_exposes_levels_and_labels_gemmini(capsys) -> None:
    assert main(["levels"]) == 0
    assert json.loads(capsys.readouterr().out)[3]["id"] == "EL4"
    assert main(["list"]) == 0
    definitions = json.loads(capsys.readouterr().out)
    gemmini = next(row for row in definitions if row["id"] == "gemmini-functional")
    assert gemmini["phase1_level"] == "EL4"


def test_el4_preflight_does_not_treat_metadata_as_executable_support(monkeypatch) -> None:
    monkeypatch.setattr("merlin.targetgen.plugins.resolve_support", lambda _: SimpleNamespace(plugin=lambda: {}))
    plan = {
        "target": "sample",
        "phases": {"1": {"module": PHASE1_MODULE, "argv": ["python", "--treatment", "rtlchecks"], "env": {}}},
    }
    assert "plugin.backend" in _verify_rtlcheck_support(plan)[0]


def test_el4_preflight_requires_explicit_support_selection(monkeypatch) -> None:
    from merlin.targetgen.plugins import PluginError

    def refuse(_target):
        raise PluginError("executable support requires explicit MERLIN_TARGET_PATH selection")

    monkeypatch.setattr("merlin.targetgen.plugins.resolve_support", refuse)
    plan = {
        "target": "sample",
        "phases": {"1": {"module": PHASE1_MODULE, "argv": ["python", "--treatment", "rtlchecks"], "env": {}}},
    }
    assert "explicit MERLIN_TARGET_PATH" in _verify_rtlcheck_support(plan)[0]
