"""Resolving a target's compute-unit KIND fails CLOSED — it never quietly becomes systolic.

`_resolve_kind` wrapped `load_capability_manifest` in a bare `except Exception`, so every way a
manifest can fail came back as `kind=None`, and `None` is the documented cue to route to the generic
static extractor. Two very different states — "nothing has declared this target's class yet" and "this
target's declaration is BROKEN" — arrived at the same caller as the same answer, and the second one was
invisible.

It was live. `k1_cpu`'s contract declares `int16`/`int32` on its compute units, the quant-format
registry knew neither, `compute_units._unit` raised, the raise was swallowed, and a general-purpose
vector CPU with no mesh anywhere in it was extracted by the systolic reader. Nothing anywhere recorded
that a declaration had failed to load.

So there are two halves here, and both matter: the RESOLVER now refuses instead of defaulting, and a
sweep over every in-tree capability manifest makes the class of failure that caused it impossible to
keep secret.
"""

from __future__ import annotations

import json

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import families
from merlin.targetgen.rtl import facts as F
from merlin.targetgen.rtl import mlc_bridge as B


def _in_tree_contracts() -> list[tuple[str, object]]:
    """Every in-tree target that ships a capability contract, by directory name.

    Located structurally (a contract file under `merlin/targets/<t>/contracts/`), so a target added to
    the tree is swept without editing this test — which is the point, since the defect below reached
    production precisely because nothing enumerated the roster.
    """
    return [
        (p.parent.parent.name, p) for p in sorted((merlin_dir() / "targets").glob("*/contracts/target_contract.yaml"))
    ]


class TestEveryInTreeTargetManifestLoads:
    """The sweep. This is the test whose absence let a broken contract sit in the tree unnoticed."""

    def test_the_roster_is_not_empty(self) -> None:
        assert _in_tree_contracts(), "no in-tree target contracts were found; every check below is vacuous"

    @pytest.mark.parametrize("target", [t for t, _ in _in_tree_contracts()])
    def test_it_loads_through_the_capability_spine(self, target: str) -> None:
        """A contract that does not load is a target whose class nothing can resolve.

        NOT tolerated with a skip: the reason this failure was expensive is that every caller tolerated
        it. The failure it caught, spelled out so a future breakage reads as the same thing: a compute
        unit declaring a numeric format the quant-format registry does not carry raises here, and
        every consumer downstream silently substituted a datapath class for it.
        """
        from merlin.targetgen.target_experiment import load_capability_manifest

        manifest = load_capability_manifest(target)
        assert manifest.kind in families.known_kinds(), f"{target}: kind {manifest.kind!r} is not a known kind"

    @pytest.mark.parametrize("target", [t for t, _ in _in_tree_contracts()])
    def test_its_declared_formats_are_all_in_the_quant_registry(self, target: str) -> None:
        """The specific failure, asserted directly so the message names the format rather than the raise.

        A format is referenced by NAME here and defined once in the registry; a contract naming one the
        registry does not carry is either a missing registry entry or a wrong contract, and both are
        fixable only if someone is told which format it is.
        """
        from merlin.common import quant_formats as qf
        from merlin.targetgen.compute_units import compute_units
        from merlin.targetgen.target_registry import resolve

        for unit in compute_units(resolve(target).load_contract()):
            unknown = [d for d in unit.dtypes if not qf.has(d)]
            assert not unknown, f"{target}/{unit.name}: formats {unknown} are not in the quant-format registry"

    @pytest.mark.parametrize("target", [t for t, _ in _in_tree_contracts()])
    def test_its_kind_set_resolves_without_a_refusal(self, target: str) -> None:
        """The reading path itself, on the real roster: no in-tree target may be unresolvable."""
        assert B._resolve_kinds(target), f"{target}: no compute-unit kind resolved"


class TestAnUnresolvableKindRefusesInsteadOfDefaulting:
    """MUTATION: break one manifest and watch the routing refuse rather than pick a class."""

    @staticmethod
    def _break_the_manifest(monkeypatch) -> None:
        from merlin.targetgen import target_experiment

        def _boom(target, **_kw):
            raise ValueError(f"compute unit 'x': unknown quant formats ['int16'] for {target}")

        monkeypatch.setattr(target_experiment, "load_capability_manifest", _boom)

    def test_the_kind_set_raises_rather_than_coming_back_empty(self, monkeypatch) -> None:
        self._break_the_manifest(monkeypatch)
        with pytest.raises(B.KindUnresolved) as excinfo:
            B._resolve_kinds("a_target_whose_contract_is_broken")
        assert "int16" in str(excinfo.value), "the underlying reason must survive into the refusal"

    def test_the_extractor_choice_raises_too(self, monkeypatch) -> None:
        """`_extractors_for` is the seam production and reading share; it must not absorb the refusal."""
        self._break_the_manifest(monkeypatch)
        with pytest.raises(B.KindUnresolved):
            B._extractors_for("a_target_whose_contract_is_broken")

    def test_the_systolic_extractor_is_never_reached(self, monkeypatch, tmp_path) -> None:
        """The defect itself, as a property: an unresolved kind must not become the static reader.

        The previous behaviour is exactly what this asserts against — `kind=None` routed to
        `circt_static`, which is the systolic/vector/scalar reader, so a target whose declaration had
        failed was extracted as though it were one of those.
        """
        from merlin.targetgen.rtl import circt_introspect, spatial_introspect

        self._break_the_manifest(monkeypatch)
        for mod, name in ((circt_introspect, "dump_facts"), (spatial_introspect, "spatial_facts")):

            def _must_not_run(*_a, **_kw):
                raise AssertionError("an extractor ran for a target whose kind did not resolve")

            monkeypatch.setattr(mod, name, _must_not_run, raising=False)
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_target_whose_contract_is_broken")
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["facts"] == {}, "nothing was extracted, so nothing may be claimed"
        reasons = F.unknown_reasons(doc)
        assert "compute_unit_kind" in reasons, f"the refusal was not recorded: {sorted(reasons)}"
        assert "int16" in reasons["compute_unit_kind"]

    def test_the_recorded_refusal_is_a_schema_valid_artifact(self, monkeypatch, tmp_path) -> None:
        """A refusal nobody can read is a crash with extra steps."""
        from merlin.targetgen.rtl import circt_introspect

        self._break_the_manifest(monkeypatch)
        monkeypatch.setattr(circt_introspect, "dump_facts", lambda *a, **k: None)
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_target_whose_contract_is_broken")
        doc = json.loads(out.read_text(encoding="utf-8"))
        doc.setdefault("family", "circt_static")
        assert F.validate_facts(doc) == []


class TestAMissingManifestStillDegradesTheWayItAlwaysDid:
    """The other half of fail-closed: a target with NO contract is not a broken one."""

    def test_no_contract_anywhere_resolves_to_no_kind(self, monkeypatch) -> None:
        from merlin.targetgen import target_experiment

        def _absent(target, **_kw):
            raise FileNotFoundError(f"{target}: no capability contract")

        monkeypatch.setattr(target_experiment, "load_capability_manifest", _absent)
        monkeypatch.setattr(B, "_arc_model_kind", lambda _t: None)
        assert B._resolve_kinds("a_target_nobody_has_described") == ()
        assert B._resolve_kind("a_target_nobody_has_described") is None
        assert B._extractors_for("a_target_nobody_has_described") == ("circt_static",)

    def test_an_arc_declared_kind_is_still_honoured(self, monkeypatch) -> None:
        """The registry is the next declaration to ask, not a fallback invented here."""
        from merlin.targetgen import target_experiment

        monkeypatch.setattr(
            target_experiment,
            "load_capability_manifest",
            lambda target, **_kw: (_ for _ in ()).throw(FileNotFoundError(target)),
        )
        monkeypatch.setattr(B, "_arc_model_kind", lambda _t: "spatial")
        assert B._resolve_kinds("an_arc_only_model") == ("spatial",)
        assert B._extractors_for("an_arc_only_model") == ("opu",)

    def test_an_undeclared_class_is_recorded_rather_than_assumed(self, monkeypatch, tmp_path) -> None:
        """A default is a decision, so it is written down.

        The generic static reader running for a target nobody has classified is a reasonable default and
        an unreasonable silence: the artifact it produces is not attributable to a datapath class, and
        saying so is the difference between "we never decided what this device is" and "we extracted it".
        """
        from merlin.targetgen.rtl import circt_introspect

        monkeypatch.setattr(B, "_resolve_kinds", lambda _t: ())
        monkeypatch.setattr(
            circt_introspect,
            "dump_facts",
            lambda out, **kw: F.write_facts_guarded(
                out, {"schema_version": "2.0", "inputs": kw, "facts": {"target": kw["target"], "source": "a probe"}}
            ),
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_target_nobody_has_classified")
        reasons = F.unknown_reasons(json.loads(out.read_text(encoding="utf-8")))
        assert "compute_unit_kind" in reasons
        assert "no compute-unit kind is declared" in reasons["compute_unit_kind"]
