"""Route on the exact registry format name, not the compile-mode token.

A whole model routed 0 of its 15 contractions to a mesh that supports every one of them, because the
capsule's compile_dtype ("fp8", an RVV compile mode) was fed to the router while the target declares its
datapath as "fp8_e4m3". The capsule carries BOTH tokens; the router must get the exact one.
"""

from __future__ import annotations

import pathlib

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_source as CSRC
from merlin.targetgen import routing as R


def _bundle_mlir(name: str) -> str:
    for d in (pathlib.Path(repo_root()) / "out/artifacts/recaptures").glob(f"**/{name}"):
        for m in d.rglob("*.mlir"):
            return m.read_text()
    pytest.skip(f"bundle {name} not on disk")


def _mesh(target: str, bundle: str, tok: str) -> int:
    plan = R.route_plan(CSRC.model_op_demands(_bundle_mlir(bundle), tok), target)
    return len(plan.get("mesh") or [])


def test_the_compile_token_routes_nothing_but_the_exact_name_routes_everything():
    assert _mesh("atlas", "small_llama_fp8_consistent", "fp8") == 0          # the bug
    assert _mesh("atlas", "small_llama_fp8_consistent", "fp8_e4m3") == 15    # the fix


def test_gemmini_is_unaffected():
    """Its compile token and its declared format are the same string, so it never had the bug."""
    assert _mesh("gemmini", "small_llama_int8_consistent", "int8") == 15


def test_e5m2_never_routes_onto_an_e4m3_unit():
    """The negative guard. An `fp8 -> fp8_e4m3` alias would have made this route, silently computing
    with the wrong exponent bias -- which is why the registry omits that alias and we thread the exact
    name instead."""
    d = [R.OpDemand(op="matmul", in_fmt="fp8_e5m2", weight_fmt="fp8_e5m2",
                    site="matmul", m=8, n=128, k=128)]
    assert len(R.route_plan(d, "atlas").get("mesh") or []) == 0


def test_a_capsule_declares_both_tokens():
    """The fix relies on the exact name being present in the capsule; assert it is."""
    import yaml
    p = (pathlib.Path(repo_root())
         / "merlin/contract/capsules/atlas/model/M0_small_llama_atlas/capsule.yaml")
    if not p.is_file():
        pytest.skip("atlas model capsule not on disk")
    attrs = (yaml.safe_load(p.read_text())["operation"] or {}).get("attributes") or {}
    assert attrs.get("dtype") == "fp8_e4m3"      # exact registry name -> routing
    assert attrs.get("compile_dtype") == "fp8"   # compile mode -> compile_rvv
