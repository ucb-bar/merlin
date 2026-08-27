"""The mining CLI must reach the expert's ASSEMBLY, not a summary of it.

`expert_cca_from_policies` summarizes what the mined POLICIES say, which is a reconstruction of the
expert's intent rather than a reading of the expert's code. Every shipped mining run therefore recorded
`expert_cca level: policy` against `our_cca level: asm` — comparing two different kinds of evidence,
which is the comparison this loop exists to avoid making.

Separately, the CLI lifted every target through the vector lifter. That does not fail on an
accelerator's command stream: it returns an EMPTY vector facet, and an empty facet compares equal to
anything, so the pipeline reported no divergence for a kernel it had never read.
"""
from __future__ import annotations

import pytest

from merlin.mining import mine


class TestTheTargetSelectsTheLifter:
    def test_an_accelerator_target_resolves_its_endpoint(self):
        ep = mine.endpoint_for_target("gemmini")
        if ep is None:
            pytest.skip("gemmini endpoint not resolvable in this checkout")
        assert ep.name == "gemmini_rocc" and "accumulate" in ep.roles

    def test_a_lane_target_now_resolves_its_own_endpoint(self):
        """The base ISA used to resolve None and fall through to the mnemonic-counting lifter, which
        made the one target the loop worked on the one whose assembly carried no declared meaning.
        It has a declared lane endpoint now, so it is measured like every other target."""
        ep = mine.endpoint_for_target("rvv")
        if ep is None:
            pytest.skip("rvv endpoint not resolvable in this checkout")
        assert ep.engine == "vector" and "elementwise" in ep.roles

    def test_a_target_with_no_endpoint_at_all_still_resolves_none(self):
        """None remains a real answer, not a failure — it means the vector lifter, honestly."""
        assert mine.endpoint_for_target("a-target-that-does-not-exist") is None

    def test_an_unknown_target_does_not_invent_an_endpoint(self):
        assert mine.endpoint_for_target("a-target-that-does-not-exist") is None

    def test_the_endpoint_preferred_is_the_one_that_contracts(self):
        # A target with several endpoints (an array and a lane engine on one ISA) must lift a mined
        # matmul through the one that accumulates.
        ep = mine.endpoint_for_target("atlas")
        if ep is None:
            pytest.skip("atlas endpoints not resolvable")
        assert "accumulate" in ep.roles


class TestGeometryIsDerivedNotGuessed:
    def test_geometry_comes_from_the_rtl_facts(self):
        geo = mine._geometry_for("gemmini")
        if not geo:
            pytest.skip("gemmini RTL facts unavailable")
        assert geo["pe_rows"] == geo["pe_cols"] == 16

    def test_an_uncorroborated_array_contributes_no_geometry(self, monkeypatch):
        """An uncorroborated reading is a ranking, not a fact — the measured case being a 17x17 mesh
        that was 289 flip-flops in a divide/sqrt unit. Feeding that into a facet publishes it."""
        from merlin.targetgen.rtl import facts as F
        monkeypatch.setattr(F, "load_facts", lambda t: {
            "facts": {"arrays": [{"name": "mesh", "rows": 17, "cols": 17}]}})
        assert mine._geometry_for("anything") == {}

    def test_an_absent_fact_bundle_yields_no_geometry(self, monkeypatch):
        from merlin.targetgen.rtl import facts as F
        monkeypatch.setattr(F, "load_facts", lambda t: None)
        assert mine._geometry_for("anything") == {}


class TestNothingDecodedIsNotACleanResult:
    def test_a_stream_with_no_endpoint_roles_lifts_to_none(self, monkeypatch, tmp_path):
        """The load-bearing one. An empty CCA compares equal to anything on the facets it did not
        fill, so returning one for a stream the decoder never read reports NO divergence — a clean
        result for a kernel nobody looked at."""
        ep = mine.endpoint_for_target("gemmini")
        if ep is None:
            pytest.skip("gemmini endpoint not resolvable")

        class _S:
            insns = ()

            def spans_reliable(self):
                return False

            def loop_spans(self):
                return []

        from merlin.kernels.decode import rvv as rvv_decode
        monkeypatch.setattr(rvv_decode, "decode", lambda p: _S())
        obj = tmp_path / "model.o"
        obj.write_bytes(b"")
        assert mine.cca_from_object(obj, "gemmini", "matmul", "t", endpoint=ep) is None

    def test_expert_cca_from_asm_returns_none_rather_than_a_blank(self, monkeypatch):
        monkeypatch.setattr(mine, "cca_from_object", lambda *a, **k: None)
        cca, src = mine.expert_cca_from_asm(["/no/such.o"], "gemmini", "matmul")
        assert cca is None and src == ""

    def test_it_takes_the_first_object_that_actually_decodes(self, monkeypatch):
        seen = []

        def _fake(obj, *a, **k):
            seen.append(str(obj))
            return None if "bad" in str(obj) else "CCA"

        monkeypatch.setattr(mine, "cca_from_object", _fake)
        cca, src = mine.expert_cca_from_asm(["/bad.o", "/good.o"], "gemmini", "matmul")
        assert cca == "CCA" and src == "/good.o" and len(seen) == 2


class TestTheRunRecordsWhichLevelItReached:
    def test_the_manifest_declares_both_levels(self):
        """A run that compared a policy summary against emitted asm is a WEAKER result than one that
        compared two streams, and the difference has to be legible in the artifact rather than
        inferred from which fields happen to be filled."""
        import inspect
        src = inspect.getsource(mine.mine_run)
        assert '"expert_level"' in src and '"our_level"' in src
        assert '"endpoint"' in src

    def test_asm_is_preferred_over_the_policy_summary(self):
        import inspect
        src = inspect.getsource(mine.mine_run)
        assert src.index("expert_cca_from_asm") < src.index("expert_cca_from_policies"), (
            "the policy summary must be the FALLBACK, not the first choice")
