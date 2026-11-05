"""Layered support providers: a v2 provider ``extends`` a base and authors only an overlay."""

from __future__ import annotations

import os
import subprocess

import pytest

from merlin.common.yaml import safe_load_text, write_yaml
from merlin.targetgen import provider_compose as pc
from merlin.targetgen import target_registry as registry
from merlin.targetgen.providers import ProviderError, read_provider

SHA = "0123456789abcdef0123456789abcdef01234567"

BASE_CONTRACT = {
    "name": "base_isa",
    "features": ["vector", "multicore"],
    "compute_units": [{"name": "vector", "dtypes": ["int8", "fp32"], "ops": ["matmul"]}],
    "runtime": {"backends": ["zephyr"], "default_backend": "zephyr"},
    "provenance": "measured on some other board",
    "plugin": {"backend": "backend"},
}


def base_provider(root, contract=None, target="base_isa"):
    write_yaml(root / "contracts/target_contract.yaml", {**(contract or BASE_CONTRACT), "name": target})
    write_yaml(
        root / "provider.yaml",
        {"schema": "merlin.provider.v1", "id": f"{target}-support", "target": target, "role": "support"},
    )
    return root


def layer(root, overlay, *, target="chip_a", base="base_isa", commit=None, contract=True):
    extends = {"target": base}
    if commit:
        extends["commit"] = commit
    write_yaml(
        root / "provider.yaml",
        {
            "schema": "merlin.provider.v2",
            "id": f"{target}-support",
            "target": target,
            "role": "support",
            "extends": extends,
        },
    )
    write_yaml(root / "contracts/overlay.yaml", overlay)
    return root


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(registry, "in_repo_support", lambda: {})


# ── merge rules ──────────────────────────────────────────────────────────────────────────────────


def test_overlay_extends_without_restating_the_base():
    eff = pc.compose(
        BASE_CONTRACT,
        {
            "features": ["matrix"],
            "compute_units": [{"name": "vector", "ops": ["requant"]}, {"name": "matrix", "ops": ["matmul"]}],
            "runtime": {"default_backend": "baremetal"},
            "matrix_contract": "contracts/matrix_units.yaml",
        },
        target="chip_a",
    )
    assert eff["name"] == "chip_a"
    assert eff["features"] == ["vector", "multicore", "matrix"]
    units = {u["name"]: u for u in eff["compute_units"]}
    assert units["vector"]["dtypes"] == ["int8", "fp32"] and units["vector"]["ops"] == ["matmul", "requant"]
    assert units["matrix"]["ops"] == ["matmul"]
    assert eff["runtime"] == {"backends": ["zephyr"], "default_backend": "baremetal"}
    assert eff["matrix_contract"] == "contracts/matrix_units.yaml"


def test_base_plugin_is_never_inherited_but_a_layer_may_declare_its_own():
    assert "plugin" not in pc.compose(BASE_CONTRACT, {}, target="chip_a")
    eff = pc.compose(BASE_CONTRACT, {"plugin": {"matrix_lowering": "m.py"}}, target="chip_a")
    assert eff["plugin"] == {"matrix_lowering": "m.py"}


def test_replace_and_remove_directives():
    eff = pc.compose(
        BASE_CONTRACT,
        {
            "replace": ["features"],
            "remove": ["provenance"],
            "features": ["scalar"],
        },
        target="chip_a",
    )
    assert eff["features"] == ["scalar"]
    assert "provenance" not in eff and "replace" not in eff and "remove" not in eff
    with pytest.raises(pc.ComposeError, match="does not exist"):
        pc.compose(BASE_CONTRACT, {"remove": ["nope.deeper"]}, target="chip_a")
    with pytest.raises(pc.ComposeError, match="does not set"):
        pc.compose(BASE_CONTRACT, {"replace": ["features"]}, target="chip_a")


def test_overlay_cannot_rename_the_layer():
    with pytest.raises(pc.ComposeError, match="declares 'chip_a'"):
        pc.compose(BASE_CONTRACT, {"name": "other"}, target="chip_a")


# ── provider declaration ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "extends, match",
    [
        ({"target": "chip_a"}, "its own target"),
        ({"target": "base_isa", "commit": "abc123"}, "40-char"),
        ({"target": "base_isa", "branch": "main"}, "unknown fields"),
        ("base_isa", "must be a mapping"),
    ],
)
def test_invalid_extends_is_refused(tmp_path, extends, match):
    root = tmp_path / "chip"
    write_yaml(
        root / "provider.yaml",
        {"schema": "merlin.provider.v2", "id": "x", "target": "chip_a", "role": "support", "extends": extends},
    )
    write_yaml(root / "contracts/overlay.yaml", {})
    write_yaml(root / "contracts/target_contract.yaml", {"name": "chip_a"})
    with pytest.raises(ProviderError, match=match):
        read_provider(root)


def test_v1_cannot_declare_extends(tmp_path):
    root = tmp_path / "chip"
    write_yaml(
        root / "provider.yaml",
        {
            "schema": "merlin.provider.v1",
            "id": "x",
            "target": "chip_a",
            "role": "support",
            "extends": {"target": "base_isa"},
        },
    )
    with pytest.raises(ProviderError, match="unknown fields"):
        read_provider(root)


def test_unmaterialized_layer_names_the_command(tmp_path):
    root = layer(tmp_path / "chip", {})
    with pytest.raises(ProviderError, match="merlin-target-compose"):
        read_provider(root)


# ── materialization ──────────────────────────────────────────────────────────────────────────────


def test_materialize_check_and_drift(tmp_path, monkeypatch):
    base = base_provider(tmp_path / "base")
    chip = layer(tmp_path / "chip", {"features": ["matrix"]})
    monkeypatch.setenv("MERLIN_TARGET_PATH", os.pathsep.join(map(str, (chip, base))))

    fresh, comp = pc.materialize(chip)
    assert not fresh and comp.base.root == base.resolve()
    eff = safe_load_text((chip / "contracts/target_contract.yaml").read_text())
    assert eff["name"] == "chip_a" and eff["features"] == ["vector", "multicore", "matrix"]
    assert (chip / "contracts/target_contract.yaml").read_text().startswith("# GENERATED")
    lock = safe_load_text((chip / pc.LOCK_FILE).read_text())
    assert lock["base"]["target"] == "base_isa" and lock["base"]["revision"] is None
    assert pc.materialize(chip, check=True)[0]
    assert pc.main([str(chip), "--check"]) == 0

    # The layered provider now resolves like any flat one, with the composed view.
    info = registry.resolve("chip_a")
    assert info.base == chip.resolve() and info.load_contract()["runtime"]["default_backend"] == "zephyr"
    assert info.plugin() == {"path": str(chip.resolve())}

    write_yaml(base / "contracts/target_contract.yaml", {**BASE_CONTRACT, "features": ["vector"]})
    assert pc.main([str(chip), "--check"]) == 1  # the base moved
    pc.materialize(chip)
    write_yaml(chip / "contracts/overlay.yaml", {"features": ["matrix", "mx"]})
    assert not pc.materialize(chip, check=True)[0]  # the overlay moved


def test_missing_base_points_at_fetch(tmp_path, monkeypatch):
    chip = layer(tmp_path / "chip", {})
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(chip))
    with pytest.raises(pc.ComposeError, match="merlin-target-fetch base_isa"):
        pc.materialize(chip)


def test_pinned_revision_from_vendored_source_record(tmp_path, monkeypatch):
    base = base_provider(tmp_path / "vendored" / "support")
    write_yaml(tmp_path / "vendored" / "SOURCE.yaml", {"path": "support", "source": {"commit": SHA}})
    assert pc.provider_revision(base) == SHA
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(base))
    good = layer(tmp_path / "good", {}, commit=SHA)
    assert pc.materialize(good)[1].base_revision == SHA
    bad = layer(tmp_path / "bad", {}, commit="f" * 40)
    with pytest.raises(pc.ComposeError, match="pinned at f+"):
        pc.materialize(bad)


def test_revision_of_a_cloned_base_is_its_head_but_not_an_enclosing_repo(tmp_path):
    repo = tmp_path / "base-repo"
    base_provider(repo)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert pc.provider_revision(repo) == head
    nested = base_provider(repo / "nested")
    assert pc.provider_revision(nested) is None


def test_cycles_are_refused(tmp_path, monkeypatch):
    a = layer(tmp_path / "a", {}, target="chip_a", base="chip_b")
    b = layer(tmp_path / "b", {}, target="chip_b", base="chip_a")
    for root, name in ((a, "chip_a"), (b, "chip_b")):
        write_yaml(root / "contracts/target_contract.yaml", {"name": name})
    monkeypatch.setenv("MERLIN_TARGET_PATH", os.pathsep.join(map(str, (a, b))))
    with pytest.raises(pc.ComposeError, match="cycle"):
        pc.materialize(a)


def test_two_level_chain_composes_from_the_materialized_middle(tmp_path, monkeypatch):
    base = base_provider(tmp_path / "base")
    mid = layer(tmp_path / "mid", {"features": ["matrix"]}, target="chip_family")
    leaf = layer(tmp_path / "leaf", {"features": ["mx"]}, target="chip_rev2", base="chip_family")
    monkeypatch.setenv("MERLIN_TARGET_PATH", os.pathsep.join(map(str, (leaf, mid, base))))
    pc.materialize(mid)
    pc.materialize(leaf)
    assert [p.target for p in pc.chain(read_provider(leaf))] == ["base_isa", "chip_family", "chip_rev2"]
    eff = safe_load_text((leaf / "contracts/target_contract.yaml").read_text())
    assert eff["features"] == ["vector", "multicore", "matrix", "mx"]
