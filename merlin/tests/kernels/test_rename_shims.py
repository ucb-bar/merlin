"""The de-RVV rename, and the two traps in it.

`rvvgen` was named for the one backend it could handle. The loop mines expert kernels for any endpoint
now — the lifter keys on instruction ROLES — so a name claiming one ISA had become a claim about scope
that is no longer true. The rename is only safe if the old names keep resolving and if nothing quietly
attributes a kernel to the wrong endpoint.
"""
from __future__ import annotations

import pytest


class TestTheOldNamesStillResolve:
    def test_the_package_alias_is_the_same_module_object(self):
        """ALIASED, not re-imported. Sharing __path__ would execute the module a second time under a
        second name, and this package's neighbours hold registries (routes, seams, backends) — a
        duplicate would register everything twice and each half would see only its own."""
        import merlin.mining.beam as new
        import merlin.rvvgen.beam as old
        assert old is new

    def test_attribute_access_forwards_too(self):
        import merlin.mining.mine as new
        import merlin.rvvgen as shim
        assert shim.mine is new

    def test_a_function_imported_through_either_name_is_one_function(self):
        from merlin.mining.mine import main as m_new
        from merlin.rvvgen.mine import main as m_old
        assert m_old is m_new

    def test_the_knobs_module_alias_resolves(self):
        from merlin.kernels import knobs, rvv_knobs
        assert rvv_knobs.__name__ != knobs.__name__
        for name in [n for n in dir(knobs) if not n.startswith("_")][:5]:
            assert getattr(rvv_knobs, name) is getattr(knobs, name)

    def test_a_genuinely_absent_submodule_still_raises(self):
        # The shim must not turn a typo into a silent success.
        with pytest.raises(ImportError):
            __import__("merlin.rvvgen.no_such_module")


class TestTheIngestAttributionTrap:
    """Measured: a shipped matrix-extension kernel is named `..._ukernel_16x4v__rvv` and its body issues
    the outer-product unit's own instructions. Attributing by ISA suffix files it as a plain vector
    kernel and makes every matrix-unit decision in it INVISIBLE — the corpus would then teach the loop
    that the expert used lanes where the expert used an array."""

    def _markers(self):
        from merlin.kernels.ingest import xnnpack as X
        m = X.endpoint_markers()
        if not m:
            pytest.skip("no compute endpoints resolvable in this checkout")
        return m

    def test_a_kernel_driving_an_endpoint_is_attributed_to_it(self):
        from merlin.kernels.ingest import xnnpack as X
        m = self._markers()
        src = ("void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x4v__rvv(void)"
               "{ VOPACC(md, v1, v2); VMV_VR(a, b, c); }")
        target, hit = X.attribute_target(src, "rvv", m)
        assert target != "rvv", "the ISA suffix won over the instructions actually issued"
        assert hit, "the attribution must record WHICH markers matched, so it can be checked"

    def test_a_plain_vector_kernel_keeps_its_isa_target(self):
        from merlin.kernels.ingest import xnnpack as X
        target, hit = X.attribute_target("void xnn_f32_gemm_ukernel_4x4v__rvv(void){ vfmacc(); }",
                                         "rvv", self._markers())
        assert target == "rvv" and hit == ()

    def test_markers_carry_both_vocabularies(self):
        """The RTL's name for an instruction and the expert HEADER's macro for it differ, and it is the
        header spelling that appears in a kernel's source. Using only the RTL names finds nothing in
        the very file this attribution exists to catch."""
        from merlin.kernels import endpoints as EP
        ep = EP.load_endpoint("saturn_opu")
        pairs = ep.crosscheck.get("pairs") or {}
        if not pairs:
            pytest.skip("no crosscheck pairs declared")
        markers = self._markers().get(ep.target, ())
        assert any(v in markers for v in pairs.values()), "header spellings are missing from the markers"
        assert any(k in markers for k in pairs), "RTL spellings are missing from the markers"

    def test_the_isa_suffix_is_a_parameter_not_a_target(self):
        from merlin.kernels.ingest import xnnpack as X
        assert X.DEFAULT_ISA == "rvv"
        # the base ISA is threaded in, never baked: the same symbol grammar parses any suffix
        assert X.find_ukernel_symbol("xnn_f32_gemm_ukernel_2x2__foo(", "foo") == ("f32_gemm", "2x2")
        assert X.find_ukernel_symbol("xnn_f32_gemm_ukernel_2x2__foo(", "rvv") is None


class TestTheMiningCliNamesItsTarget:
    def test_target_is_required_rather_than_defaulted(self):
        """A default of one target silently mislabels every run for another: the artifacts are written
        under <target>/ and the CCA is compared against that target's expert corpus, so a mislabelled
        run compares the wrong things without saying so."""
        import argparse

        from merlin.mining import mine

        parser = None
        for attr in dir(mine):
            fn = getattr(mine, attr)
            if attr == "main" and callable(fn):
                break
        # Build the parser the same way main() does, by inspecting the source for the requirement.
        import inspect
        src = inspect.getsource(mine)
        assert '"--target", required=True' in src, "--target must not default to one target"
        assert 'default="rvv"' not in src.split("--target")[1][:80]
        del argparse, parser
