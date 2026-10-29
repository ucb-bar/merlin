"""The bundle generator retains historic grants except retired broad and stale paths."""

from __future__ import annotations

import json

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.generate_bundles import (
    _materialize_prompt_and_grants,
    generate_bundles,
)
from merlin.targetgen.target_experiment import load_target_experiment


def _te():
    return load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    )


def _sets(m):
    return ({e["path"] for e in m.get("allowed", [])}, {e["path"] for e in m.get("denied", [])})


def test_generates_every_registered_arm():
    """One bundle per registered arm. Asserted against the registry rather than a written-out set, so
    adding an arm cannot leave this test asserting the old count while the launcher runs a new one."""
    from merlin.targetgen.generate_bundles import _ARMS

    gen = generate_bundles(_te())
    assert set(gen) == {f"{stem}_hwbringup_v0" for stem in _ARMS.values()}
    # the arms that exist today, named so a deletion is also visible
    assert {
        "raw_baseline_hwbringup_v0",
        "cpp_merlininfra_hwbringup_v0",
        "merlin_assisted_hwbringup_v0",
        "merlin_assisted_rtlchecks_hwbringup_v0",
        "merlin_assisted_eqsat_hwbringup_v0",
    } <= set(gen)


def _norm(paths):
    # normalize the stale bare 'artifacts/' (hand-authored) vs the correct 'out/artifacts/' (generated)
    return {p[4:] if p.startswith("out/artifacts/") else p for p in paths}


def test_declared_numeric_profile_is_frozen_without_a_candidate_grant():
    te = _te()
    assert te.numeric_profile
    for manifest in generate_bundles(te).values():
        assert str(te.path) in {entry["path"] for entry in manifest["host_inputs"]}
        assert str(te.path) not in {entry["path"] for entry in manifest["allowed"]}
        assert te.numeric_profile in {entry["path"] for entry in manifest["host_inputs"]}
        assert te.numeric_profile not in {entry["path"] for entry in manifest["allowed"]}


def test_selected_phase0_facts_replace_cache_grants_and_feed_prompt(tmp_path, monkeypatch):
    from merlin.targetgen import generate_bundles as generator
    from merlin.targetgen.rtl import facts

    te = _te()
    selected = tmp_path / "facts"
    selected.mkdir()
    document = {"facts": {"target": te.target, "selection_marker": "phase0"}}
    (selected / "facts.json").write_text(json.dumps(document))
    bundles = generate_bundles(te, arms=("merlin_assisted", "merlin_rtlchecks"), rtl_facts_root=selected)
    selected_path = str(selected) + "/"
    assert selected_path in _sets(bundles["merlin_assisted_rtlchecks_hwbringup_v0"])[0]
    assert selected_path in _sets(bundles["merlin_assisted_hwbringup_v0"])[1]
    assert te.rtl_facts_pin not in _sets(bundles["merlin_assisted_rtlchecks_hwbringup_v0"])[0]
    assert bundles["merlin_assisted_rtlchecks_hwbringup_v0"]["selected_rtl_facts_file"] == str(selected / "facts.json")
    assert "selected_rtl_facts_file" not in bundles["merlin_assisted_hwbringup_v0"]

    from merlin.targetgen.sandbox.bwrap import base_argv

    permitted = base_argv(
        tmp_path / "workspace",
        {"allowed": [{"path": selected_path}], "selected_rtl_facts_file": str(selected / "facts.json")},
        _policy_test_live_inputs=True,
    )
    denied = base_argv(tmp_path / "workspace", {"allowed": []}, _policy_test_live_inputs=True)
    selected_index = permitted.index("MERLIN_RTL_FACTS")
    assert permitted[selected_index - 1 : selected_index + 2] == [
        "--setenv",
        "MERLIN_RTL_FACTS",
        str(selected / "facts.json"),
    ]
    assert ["--unsetenv", "MERLIN_RTL_FACTS"] == denied[
        denied.index("MERLIN_RTL_FACTS") - 1 : denied.index("MERLIN_RTL_FACTS") + 1
    ]
    with pytest.raises(RuntimeError, match="exact allowed directory grant"):
        base_argv(
            tmp_path / "workspace",
            {"allowed": [], "selected_rtl_facts_file": str(selected / "facts.json")},
            _policy_test_live_inputs=True,
        )

    observed = []

    def prompt(_te, _directory, _bundle_id, _variant, _manifest, _cap, _written, **_kwargs):
        observed.append(facts.load_facts(te.target)["facts"]["selection_marker"])

    monkeypatch.setattr(generator, "_materialize_prompt_and_grants", prompt)
    generator.materialize_bundles(te, tmp_path / "bundles", arms=("merlin_rtlchecks",), rtl_facts_root=selected)
    assert observed == ["phase0"]
    (selected / "facts.json").write_text(json.dumps({"facts": {"target": "another"}}))
    with pytest.raises(ValueError, match="descriptor target"):
        generate_bundles(te, rtl_facts_root=selected)
    (selected / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly one ordinary facts.json"):
        generate_bundles(te, rtl_facts_root=selected)


def test_retains_historical_grants_without_reintroducing_broad_contract_access():
    """Old manifests remain frozen; new bundles narrow the public ABI and use example-owned tasks."""
    gen = generate_bundles(_te())
    B = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/input_bundles"
    for bid, gm in gen.items():
        hand = yaml.safe_load((B / bid / "input_bundle_manifest.yaml").read_text())
        ga, gd = _sets(gm)
        ha, hd = _sets(hand)
        retired = {"merlin/contract/", "experiments/capsule_bench/targets/gemmini/task/"}
        assert ha - retired <= ga, f"{bid} lost a historical non-retired grant: {ha - retired - ga}"
        assert "merlin/contract/" not in ga
        assert "merlin/contract/schemas/" in ga
        assert "examples/gemmini/phase1/task/" in ga
        # The old shared hidden path no longer exists in this checkout; a
        # release descriptor supplies its own hidden root and deny rule.
        historical_hidden = {"merlin/contract/capsules/hidden/"}
        assert _norm(hd - historical_hidden) <= _norm(gd), (
            f"{bid} deny missing from generated: {_norm(hd - historical_hidden) - _norm(gd)}"
        )


def test_agnostic_tool_blocks_have_no_target_name():
    """The per-rung tool blocks are literal merlin/python paths — no target name, for any target. They
    now live in the tool registry, which is also where an ablation cell picks them up from."""
    literal = [p for t in TR.TOOLS.values() for p in t.bundle_paths]
    assert literal, "the registry grants nothing — the rungs would all be empty"
    for p in literal:
        assert p.startswith(("merlin/python/merlin/", "merlin/contract/")) and "gemmini" not in p


def test_increasing_help_gradient():
    """arm1 ⊂ arm2/arm3 tools; arm4 = arm3 tools + the CIRCT rtl generators + the rtl_facts pin."""
    gen = generate_bundles(_te())
    raw_a, _ = _sets(gen["raw_baseline_hwbringup_v0"])
    cpp_a, _ = _sets(gen["cpp_merlininfra_hwbringup_v0"])
    mer_a, _ = _sets(gen["merlin_assisted_hwbringup_v0"])
    rtl_a, _ = _sets(gen["merlin_assisted_rtlchecks_hwbringup_v0"])
    py = "merlin/python/merlin/"
    assert not any(p.startswith(py) for p in raw_a)  # arm1: no merlin tools
    assert f"{py}targetgen/generate/mlir_scaffold.py" in cpp_a  # arm2: C++ generators
    assert f"{py}kernels/cca_contract.py" in mer_a and f"{py}targetgen/rtl_backend.py" in mer_a  # arm3: CCA spine
    assert f"{py}targetgen/rtl/" in rtl_a and rtl_a > mer_a  # arm4: + CIRCT rtl, superset of arm3


def test_target_specific_paths_come_from_descriptor():
    te = _te()
    gen = generate_bundles(te)
    mer_a, _ = _sets(gen["merlin_assisted_hwbringup_v0"])
    assert te.corpus_rel() in mer_a and all(h in mer_a for h in te.isa_headers)
    _, rtl_d = _sets(gen["merlin_assisted_hwbringup_v0"])
    assert te.rtl_facts_pin in rtl_d and f"merlin/targets/{te.target}/" in te.rtl_facts_pin


def test_radiance_information_treatments_are_structurally_distinct():
    te = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"
    )
    example = "examples/radiance/phase1/contracts/hwbringup_radiance_v0/example_kernel/"
    library = "out/artifacts/targets/radiance/kernel_library_pr1_v1/"
    kernels = next(iter(generate_bundles(te, variant="hwbringup_v0").values()))
    none = next(iter(generate_bundles(te, variant="hwbringup_nokernel_v0").values()))
    full = next(iter(generate_bundles(te, variant="hwbringup_kernellibrary_v0").values()))

    assert (kernels["condition"], none["condition"], full["condition"]) == ("kernels", "no-kernels", "kernel-library")
    assert example not in _sets(kernels)[1] and example in _sets(none)[1]
    assert library not in _sets(kernels)[0] and library in _sets(full)[0]
    assert full["source_pins"] == ["radiance_kernels"]
    assert full["variant"] == "hwbringup_kernellibrary_v0"


def test_assisted_tool_doc_is_regenerated_from_manifest_without_static_sandbox_claim(tmp_path):
    manifest = {
        "bundle_id": "merlin_assisted_rtlchecks_hwbringup_v0",
        "arm": "merlin_rtlchecks",
        "allowed": [{"path": "merlin/python/merlin/targetgen/rtl/", "note": "RTL generators"}],
        "denied": [{"path": "merlin/python/merlin/runtime/reference.py", "reason": "oracle route"}],
    }
    written = []

    _materialize_prompt_and_grants(_te(), tmp_path, manifest["bundle_id"], "hwbringup_v0", manifest, None, written)

    doc = (tmp_path / "ALLOWED_MERLIN_TOOLS.md").read_text()
    assert "does **not** select a sandbox" in doc
    assert "TASK.md" in doc and "environment.yaml" in doc
    assert "launcher's real `--sandbox` argument" in doc
    assert "scored, trusted run requires deny-by-default `bwrap`" in doc
    assert "explicit `none` run is diagnostic only" in doc
    assert "bwrap crashes" not in doc and "the mode both arms run" not in doc
    assert "merlin/python/merlin/targetgen/rtl/" in doc
    assert "merlin/python/merlin/runtime/reference.py" in doc
    assert tmp_path / "ALLOWED_MERLIN_TOOLS.md" in written


def test_non_assisted_bundle_does_not_claim_merlin_tooling(tmp_path):
    manifest = {"bundle_id": "raw_baseline_hwbringup_v0", "arm": "raw_baseline", "allowed": [], "denied": []}
    _materialize_prompt_and_grants(_te(), tmp_path, manifest["bundle_id"], "hwbringup_v0", manifest, None, [])
    assert not (tmp_path / "ALLOWED_MERLIN_TOOLS.md").exists()


def test_installed_assisted_tool_selection_tracks_actual_grants_and_ablation(tmp_path, monkeypatch):
    source = tmp_path / "installed"
    (source / "merlin").mkdir(parents=True)
    te = _te()
    base = next(iter(generate_bundles(te, arms=("merlin_rtlchecks",), python_source_root=source).values()))
    granted = TR.selected_bundle_tool_paths(
        {entry["path"] for entry in base["allowed"]}, base["tools"], python_source_root=source
    )
    assert str(source / "merlin/targetgen/rtl") + "/" in granted
    assert not any(path.startswith("merlin/python/merlin/") for path in granted)
    rendered = []

    def render(_target, _capability, _experiment, _arm, *, granted_tools):
        rendered.append(granted_tools)
        return "generated prompt\n"

    monkeypatch.setattr("merlin.targetgen.generate_prompt.render_prompt", render)
    (tmp_path / "bundle").mkdir()
    _materialize_prompt_and_grants(
        te,
        tmp_path / "bundle",
        base["bundle_id"],
        "hwbringup_v0",
        base,
        object(),
        [],
        python_source_root=source,
    )
    assert rendered == [granted]
    without_rtl = next(
        iter(
            generate_bundles(
                te, arms=("merlin_rtlchecks",), drop_tools=("rtl_generators",), python_source_root=source
            ).values()
        )
    )
    selected = TR.selected_bundle_tool_paths(
        {entry["path"] for entry in without_rtl["allowed"]},
        without_rtl["tools"],
        python_source_root=source,
    )
    assert not any("/targetgen/rtl/" in path for path in selected)


def test_new_bundle_can_pin_external_llvm_without_changing_legacy_default(tmp_path):
    llvm = tmp_path / "llvm"
    (llvm / "bin").mkdir(parents=True)
    for tool in ("clang-23", "mlir-opt"):
        (llvm / "bin" / tool).write_text("fixture\n")
    default = next(iter(generate_bundles(_te(), arms=("merlin_rtlchecks",)).values()))
    selected = next(iter(generate_bundles(_te(), arms=("merlin_rtlchecks",), llvm_toolchain_root=llvm).values()))
    default_paths = {row["path"] for row in default["allowed"]}
    selected_paths = {row["path"] for row in selected["allowed"]}
    assert "third_party/llvm-install/" in default_paths
    assert "third_party/llvm-install/" not in selected_paths
    assert str(llvm) + "/" in selected_paths
    assert default_paths - {"third_party/llvm-install/"} == selected_paths - {str(llvm) + "/"}



@pytest.mark.parametrize("version", (16, 17))
def test_new_common_substrate_attaches_once_after_source_selection(monkeypatch, version):
    from merlin.targetgen import tool_profile
    from merlin.targetgen.sandbox.bwrap import _host_input_paths

    calls = []
    def attach(bundle, **kwargs):
        calls.append(kwargs)
        bundle['host_inputs'].append({'path': '/selected/host-owned-formal.py', 'note': 'host-owned'})
    monkeypatch.setattr(tool_profile, 'attach_common_substrate', attach)
    generated = generate_bundles(_te(), arms=('merlin_rtlchecks',), python_source_root=repo_root() / 'src',
                                 common_substrate=getattr(tool_profile, f"COMMON_SUBSTRATE_V{version}"),
                                 host_index_bits=64)
    assert len(calls) == 1
    assert calls[0]['software_spec_path'] is not None
    assert calls[0]['target'] == _te().target
    [manifest] = generated.values()
    paths = _host_input_paths(manifest)  # actual strict reader, no duplicate waiver
    assert paths.count('/selected/host-owned-formal.py') == 1
