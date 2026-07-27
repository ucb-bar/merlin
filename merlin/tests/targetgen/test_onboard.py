"""Onboarding spine: `merlin-onboard` brings a target up from ONE descriptor, target-agnostically.

Covers the two deltas that close the gap to the one-YAML ideal:
  * the additive, backward-compatible ``rtl.repo`` descriptor field, and
  * the single :func:`merlin.targetgen.onboard.onboard` entrypoint that grounds the RTL pointer,
    regenerates the capability manifest, and validates it routes through the spine — failing honestly
    (never fabricating a manifest) when a step cannot be grounded.

Hermetic: OOT contracts are regenerated into ``tmp_path`` from the generator (the source of truth),
never depending on the gitignored dev working tree; mlc is never required (its facts are reported, not
gated).
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import families
from merlin.targetgen.onboard import OnboardError, onboard
from merlin.targetgen.target_experiment import load_target_experiment


def _real_desc(target: str) -> str:
    return str(repo_root() / "merlin" / "experiments" / f"{target}_capsule_bench_v0"
               / "target_experiment.yaml")


def _write_desc(tmp_path, body: str):
    p = tmp_path / "target_experiment.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- Delta 1: rtl.repo schema
def test_existing_descriptors_load_with_rtl_repo_none():
    """Backward-compat proof: every committed descriptor still loads, and ``rtl_repo`` defaults to None."""
    for t in ("gemmini", "radiance", "atlas", "mx_gemmini"):
        te = load_target_experiment(_real_desc(t))
        assert te.rtl_repo is None
        assert te.rtl_via == "mlc"          # legacy field unchanged


def test_rtl_repo_field_is_parsed_when_present(tmp_path):
    desc = _write_desc(tmp_path, "target: acme\nrtl:\n  via: mlc\n  repo: /opt/rtl/acme\n")
    assert load_target_experiment(desc).rtl_repo == "/opt/rtl/acme"


def test_capsule_bench_targets_contract_lives_outside_the_answer_surface():
    """A capsule-bench target's contract must NOT live inside ``out/artifacts/targets/<t>`` — that dir is
    the champion/answer-surface tree the launcher ``chmod 000``-locks before any spend, and the launcher
    reads the contract to build each arm's prompt. If the contract sits inside it (the failure mode when a
    target is 'generated' rather than a first-class 'reference' target), every arm dies with a
    PermissionError at round 0. Both capsule-bench targets must resolve as reference targets."""
    from merlin.common.paths import artifacts_dir
    from merlin.targetgen import target_registry
    for t in ("gemmini", "atlas"):
        info = target_registry.resolve(t)
        assert info.kind == "reference", f"{t} resolved as {info.kind!r}, not a first-class reference target"
        answer_surface = artifacts_dir() / "targets" / t
        assert answer_surface not in info.contract_path.parents, (
            f"{t} contract {info.contract_path} lives INSIDE its answer-surface {answer_surface} — "
            "it would be chmod-000-locked out from under the launcher's prompt build")


# --------------------------------------------------------------------------- Delta 2: onboard flow
@pytest.mark.parametrize("target,kind,endpoint,mesh_key", [
    ("radiance", "simt", "inline_asm_insn", None),
    ("atlas", "systolic", "external_backend", "rows"),   # self-hosted ISA (kernel.S), not RoCC .insn
    ("mx_gemmini", "systolic", "inline_asm_insn", "rows"),
])
def test_onboard_regenerates_manifest_and_routes(tmp_path, monkeypatch, target, kind, endpoint, mesh_key):
    """The same target-agnostic flow onboards a SIMT and two systolic targets — kind/endpoint are DERIVED
    from the regenerated manifest via the family registry, never a per-target branch."""
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    res = onboard(_real_desc(target), oot_root=tmp_path / target)
    assert res.target == target
    assert res.regenerated is True
    assert (res.oot_root / "contracts" / "target_contract.yaml").is_file()
    assert res.manifest.kind == kind and res.manifest.kind in families.known_kinds()
    assert res.manifest.endpoint_kind == endpoint
    assert res.dtypes                                     # a non-empty derived dtype set
    if mesh_key is None:
        assert res.mesh is None or "dim" in res.mesh      # SIMT: no static mesh
    else:
        assert res.mesh and res.mesh.get("rows")          # systolic mesh geometry present
    assert "OK — the target routes through the capability spine." in __import__(
        "merlin.targetgen.onboard", fromlist=["render"]).render(res)


def test_onboard_fails_honestly_on_unresolvable_rtl_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    desc = _write_desc(tmp_path, "target: radiance\nrtl:\n  via: mlc\n  repo: ./no/such/rtl/here\n")
    with pytest.raises(OnboardError) as e:
        onboard(desc, oot_root=tmp_path / "out")
    assert "does not resolve" in str(e.value)


def test_onboard_fails_honestly_when_no_manifest_can_be_grounded(tmp_path, monkeypatch):
    """A target with neither a generator entry nor a committed contract fails closed — no fabrication."""
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    desc = _write_desc(tmp_path, "target: totally_new_accel\nrtl:\n  via: mlc\n")
    with pytest.raises(OnboardError) as e:
        onboard(desc, oot_root=tmp_path / "out")
    assert "Refusing to fabricate" in str(e.value)


def test_onboard_accepts_remote_url_pointer_and_emits_registration_step(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    desc = _write_desc(
        tmp_path, "target: radiance\nrtl:\n  via: mlc\n  repo: https://example.com/acme/radiance.git\n")
    res = onboard(desc, oot_root=tmp_path / "out")
    assert res.manifest.kind == "simt"
    assert any("remote URL" in n for n in res.rtl_notes)
    assert any("circt-arc/radiance" in n for n in res.rtl_notes)   # exact, honest mlc step


def test_onboard_rejects_malformed_rtl_pointer(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    desc = _write_desc(tmp_path, "target: radiance\nrtl:\n  via: mlc\n  repo: 'weird://\\x00'\n")
    with pytest.raises(OnboardError):
        onboard(desc, oot_root=tmp_path / "out")
