"""Two modules that were implemented, tested, reviewed -- and called from nothing.

`build_tools/scripts/check_wiring.py` refuses a library module that only its own test imports, for
the reason this repo keeps paying for: a mechanism nobody calls is indistinguishable from one that
works. Two such modules sit on the phase-2 evidence path.

`merlin.perf.design_identity` resolves a run's design keys to a DEVICE, by the queue hw-config the
run was submitted under confirmed against the artifact digest the record carries. It exists because
two registered bitstreams in this repo elaborate the same configuration string onto the same board
and are different machines -- so `cost_plane`, which refuses to compare two cycle counts unless
every design key matches, held an identity the reference ledger could not read. It is now called
where a measured cycle count is attributed to hardware: the iteration cost plane.

`merlin.perf.phase2_edit_contract` turns a reviewed declaration into the contract object
`compiler_edit_scope` already takes, and adds the one check that module cannot make -- that the
symbol DECIDING the thing under study was named. A declaration omitting it validates, runs,
produces rounds and a verdict, and was unwinnable the whole time. Two such declarations are tracked
under `merlin/contract/phase2_edit_contracts/gemmini/` and reached the enforcement without ever
passing the loader. It is now called where the edit authority is frozen.

Both tests below fail if the CALL is removed, not merely if the module is.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
from merlin_experiments.phase2 import emission_diagnostics as ED

from merlin.common.paths import repo_root
from merlin.common.provenance import load_artifacts
from merlin.perf.gate_phase import configured_phase

SCRIPTS = repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"


def _load(name: str):
    """Import one of the phase-2 stage scripts by path (they are scripts, not a package)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage():
    from merlin_experiments.phase2 import authoring

    return authoring


@pytest.fixture(scope="module")
def runner():
    return _load("run_global_perf_experiment")


@pytest.fixture(scope="module")
def derivable_target() -> str:
    """A target whose array geometry and decode table this checkout really derives.

    Derived, never named: a target string here would make the test about this checkout instead of
    about the join it is checking.
    """
    from merlin.kernels.decode.rocc import funct_table_for
    from merlin.targetgen.target_registry import list_targets

    for name in sorted(list_targets()):
        table = funct_table_for(name)
        if table.get("names") and table.get("custom_opcode") is not None:
            return name
    pytest.skip("no target in this checkout carries a derived funct_decode_table")


def _timing_capsule(tier: str = "L3") -> dict:
    return {"performance": {"acceptance": {"evidence": {"timing_tier": tier}}}}


def _registered_bitstream():
    """One registered device and the hw-config it declares, taken from the pin registry itself.

    Derived, never named: the test is about the join, and writing a device name here would make it
    about this checkout's registry instead.
    """
    from merlin.common.provenance import load_artifacts
    from merlin.perf.design_identity import hw_config_index

    artifacts = load_artifacts()
    for hw_config, names in sorted(hw_config_index(artifacts).items()):
        artifact = artifacts[names[0]]
        if len(names) == 1 and getattr(artifact, "digest", "") and getattr(artifact, "config", ""):
            return hw_config, artifact
    pytest.skip("no registered bitstream declares a unique hw-config with a digest and a config")


def _freeze_holder(out):
    """Exercise the native delegation with a real installed authority and no iterations."""
    from types import SimpleNamespace

    from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
    from merlin_experiments.phase2.revision_journal import RevisionJournal

    out.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(revisions=RevisionJournal(out), edit_authority=FrozenEditAuthority(out))


def _tier_record(**extra):
    return {"substrate": "s", "engine": "e", **extra}


def test_a_measured_count_names_the_device_not_the_configuration(stage, derivable_target) -> None:
    """A cycle count is about a DEVICE, and two registered bitstreams share one configuration string.

    MUTATION: drop `verdict["device"] = resolved_device(...)` from `iteration_cost_plane`, or resolve
    by configuration name instead of confirming the digest, and this fails.
    """
    hw_config, artifact = _registered_bitstream()
    plane = ED.iteration_cost_plane(
        _timing_capsule(),
        target=derivable_target,
        arms={"candidate": {"macs": 10_000_000}},
        tiers={"L3": _tier_record(cycles=10_000_000, hw_config=hw_config, hwdb_config_artifact_sha256=artifact.digest)},
        phase=configured_phase("cost_plane"),
        artifacts=load_artifacts(),
    )
    assert plane["device"]["config"] == artifact.config
    assert plane["device"]["name"], "the plane named a configuration and no device"
    assert "digest" in plane["device"]["confirmed_by"], (
        "the device was accepted on its hw-config name alone; the digest is what makes it a device"
    )


def test_a_device_the_registry_cannot_confirm_is_unknown_not_a_mismatch(stage, derivable_target) -> None:
    """Three states, never two: an unresolved device means do not score the claim, not "they differ".

    MUTATION: fall back to the configuration name when the digest disagrees and this fails.
    """
    from types import SimpleNamespace

    hw_config = "fixture_hw_config"
    artifacts = {
        "fixture_bitstream": SimpleNamespace(
            role="firesim_bitstream",
            hw_configs=(hw_config,),
            config="fixture_device",
            digest="b" * 64,
            hwdb_digest="a" * 64,
        )
    }
    plane = ED.iteration_cost_plane(
        _timing_capsule(),
        target=derivable_target,
        arms={"candidate": {"macs": 10_000_000}},
        tiers={"L3": _tier_record(cycles=10_000_000, hw_config=hw_config, hwdb_config_artifact_sha256="f" * 64)},
        phase=configured_phase("cost_plane"),
        artifacts=artifacts,
    )
    assert plane["device"]["config"] is None
    assert plane["device"]["name"] is None
    assert "BYTES" in plane["device"]["reason"] or "digest" in plane["device"]["reason"]


def test_an_unwinnable_edit_declaration_is_refused_when_the_scope_is_frozen(runner, tmp_path) -> None:
    """A declaration that omits the symbol DECIDING the thing under study validates, runs, produces
    rounds and a verdict, and was unwinnable the whole time.

    MUTATION: stop routing declaration-shaped contracts through `phase2_edit_contract.load` in
    `freeze_edit_scope` and this fails -- the refusal becomes a generic identity error, or (for the
    accepted case below) the declaration is rejected for carrying no digest of its own.
    """

    from merlin.perf.phase2_edit_contract import Phase2EditContractError

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text("def sym():\n    pass\n", encoding="utf-8")

    def declaration(owner: str) -> dict:
        return {
            "schema": "compiler_edit_contract_v1",
            "target": "a_target_directory",
            "package_id": "a_package",
            "existing_symbols": [{"surface_id": "s0", "path": "mod.py", "symbol": "sym"}],
            "required_decisions": [{"decision": "where the epilogue is placed", "owner": owner}],
        }

    def freeze(contract, out):
        holder = _freeze_holder(out)
        return runner.GlobalPerfExperiment.freeze_edit_scope(holder, package, contract)

    with pytest.raises(Phase2EditContractError) as refused:
        freeze(declaration("mod.py:someone_else"), tmp_path / "bad")
    assert "where the epilogue is placed" in str(refused.value)
    assert "someone_else" in str(refused.value)

    binding = freeze(declaration("mod.py:sym"), tmp_path / "good")
    assert binding["contract"]["existing_symbols"][0]["symbol"] == "sym"
    assert binding["contract"]["sha256"], "the accepted declaration was sealed with no identity"


def test_an_inventory_built_contract_is_not_routed_through_the_declaration_loader(runner, tmp_path) -> None:
    """The automatic path builds a contract with no declaration annotations and no
    `required_decisions`; refusing it here would break every run that declares no surface by hand.

    MUTATION: route every contract through the loader and this fails.
    """

    from merlin.perf.agent_guidance import (
        EditSurface,
        PackageOptimizationInventory,
        SourceSymbol,
        build_compiler_edit_contract,
    )

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text("def sym():\n    pass\n", encoding="utf-8")
    inventory = PackageOptimizationInventory(
        symbols=(SourceSymbol(path="mod.py", symbol="sym", kind="function", line=1, commands=()),),
        surfaces=(
            EditSurface(
                id="s0",
                scope="pass",
                path="mod.py",
                symbol="sym",
                line=1,
                effects=(),
                cca_axes=(),
                cca_axis_status=(),
                mechanism="m",
                emitted_delta="d",
                validation="v",
                abandonment="a",
            ),
        ),
    )
    contract = build_compiler_edit_contract(inventory)
    assert "required_decisions" not in contract
    holder = _freeze_holder(tmp_path / "auto")
    binding = runner.GlobalPerfExperiment.freeze_edit_scope(holder, package, contract)
    assert binding["contract"]["sha256"] == contract["sha256"]
