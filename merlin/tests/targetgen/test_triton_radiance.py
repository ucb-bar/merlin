"""The third target: the same Triton kernel on a SIMT tensor-core cluster.

Two accelerators that share their whole shape below the interface dialect would prove nothing, so
Radiance is deliberately not a renamed Gemmini. A weight-stationary systolic array packs an operand
into a feed and streams against it; a SIMT cluster **stages** it into shared scratchpad and has warps
cooperate on the tile. Radiance's contract makes that difference binding: it carries
``compiler_obligations: [must_map_to_warps]`` with ``capabilities.simt.lanes_per_warp``, and its
target dialect therefore *requires* a warp width on every matmul where Gemmini's has no such
property at all.

That requirement is what turns the grid question into a decidable one. The bridge normalizes the SPMD
grid away entirely — no loop, no lanes, no warps — so the parallelism decision is still unmade when
Merlin takes over. From that one byte-identical module, Gemmini emits no warp mapping and Radiance
emits `lanes_per_warp = 16` derived from its own contract. Had the frontend chosen either, one of the
two arms would be impossible.

Scope is honest and stops at the command-buffer tier: outputs are gated against an independent
integer reference, not against Radiance RTL. Its own contract is `status: prototype` with
`requires_human_review: true`, and the SIMT emitter its cyclotron/Verilator oracles need is not on
this branch.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import triton_kernels as K

from merlin.common.paths import repo_root
from merlin.runtime import reference_outputs, simulate
from merlin.triton import source
from merlin.triton.bridge import to_linalg
from merlin.xdsl_dialects._common import text

RADIANCE_PACKAGE = repo_root() / "out/artifacts/targets/radiance/hand_v0"
GEMMINI_PACKAGE = repo_root() / "out/artifacts/targets/gemmini/hand_v0"

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")


def _package(path):
    from merlin.targetgen.registry import load_target

    if not path.is_dir():
        pytest.skip(f"target package not present: {path}")
    return load_target(path)


@pytest.fixture(scope="module")
def arms():
    """The SAME kernel spec, descended once per accelerator. Nothing differs but the package."""
    from merlin import compile_core

    spec = K.repeated_rhs_matmul_spec()
    bridged = to_linalg(source.make_ttir(spec), spec)
    out = {"bridged": bridged}
    for name, path in (("radiance", RADIANCE_PACKAGE), ("gemmini", GEMMINI_PACKAGE)):
        out[name] = compile_core.compile_core_mlir(
            bridged.module, target_package=_package(path)).staged
    return out


# ------------------------------------------------------------------ M5a: the descent works


def test_the_package_loads_in_isolation_and_derives_its_warp_width():
    """The fact is read from the package's own contract, never defaulted."""
    package = _package(RADIANCE_PACKAGE)
    assert package.name == "radiance"
    lanes = package.spec.extra("matmul")["lanes_per_warp"]
    assert lanes.value.data == 16
    capabilities = package.contract["capabilities"]
    # This assertion used to read `1 << 19`, taken from the kernel headers' SMEM_LOG_SIZE. That is the
    # shared-memory ADDRESS WINDOW, not the capacity — the same header derives IO_BASE_ADDR from it —
    # so it overstated the scratchpad 4x. Capacity now comes from the RTL config the hardware is
    # elaborated from, and the window is kept under its own name. See docs/design/target_kernel_anatomy.
    assert capabilities["resident_storage_bytes"] == 128 * 1024
    assert capabilities["smem_aperture_bytes"] == 1 << 19
    assert capabilities["resident_storage_bytes"] < capabilities["smem_aperture_bytes"], \
        "a capacity equal to its address window means the two facts have been conflated again"


def test_an_incoherent_contract_is_refused_rather_than_defaulted():
    """`must_map_to_warps` with no declared warp width must fail, not pick a number."""
    package = _package(RADIANCE_PACKAGE)
    derive = package.dialect_module.op_properties
    with pytest.raises(ValueError) as exc:
        derive({"compiler_obligations": ["must_map_to_warps"], "capabilities": {}})
    assert "lanes_per_warp" in str(exc.value)


def test_all_six_stage_modules_verify(arms):
    modules = list(arms["radiance"].modules())
    assert len(modules) == 6
    for module in modules:
        module.verify()


def test_it_reaches_the_radiance_dialect(arms):
    ops = {op.name for op in arms["radiance"].target_module.walk()}
    ops -= {"builtin.module", "func.func", "func.return"}
    assert ops == {"radiance.stage", "radiance.matmul", "radiance.commit", "radiance.release"}


def test_the_command_buffer_names_radiance_and_matches_an_independent_reference(arms):
    from merlin.runtime.commandbuffer import materialize_inputs

    cb = arms["radiance"].command_buffer
    assert cb["target"] == "radiance"
    outputs = simulate(cb)["outputs"]
    assert outputs == reference_outputs(cb)

    tensors = materialize_inputs(cb)
    packed = {c["operands"]["dst"]: c["operands"]["src"]
              for c in cb["commands"] if c["opcode"] == "RES_PACK"}
    lhs_of = {c["operands"]["dst"]: (c["operands"]["lhs"], c["operands"]["rhs"])
              for c in cb["commands"] if c["opcode"].startswith("MATMUL")}
    for commit in [c for c in cb["commands"] if c["opcode"] == "COMMIT"]:
        lhs, rhs = lhs_of[commit["operands"]["src"]]
        activation = np.array(tensors[lhs].to_list(), dtype=np.int64)
        weight = np.array(tensors[packed.get(rhs, rhs)].to_list(), dtype=np.int64)
        got = np.array(outputs[commit["operands"]["dst"]], dtype=np.int64)
        assert np.array_equal(got, activation @ weight), commit["operands"]["dst"]


# ------------------------------------------------- M5a: the portability number, and M5b: the grid


def test_the_frontend_contributed_nothing_target_specific(arms):
    """RQ2: target #3 costs zero frontend lines. Asserted, not claimed.

    The frontend may reach the target *registry* — resolving "what does this target's contract say"
    is the generic seam it is supposed to use, and the CLI has to accept a package path from its
    caller. What it may not do is know which target it is talking to.
    """
    import io
    import pathlib
    import token as token_mod
    import tokenize

    from merlin.common.paths import merlin_dir

    # CODE only — comments and docstrings are stripped. Explaining in prose why a GPU knob means
    # nothing on a systolic array is exactly the sort of reasoning that should be written down; what
    # must not exist is code that acts on it.
    code = []
    prose = []
    for path in sorted(pathlib.Path(merlin_dir() / "python" / "merlin" / "triton").rglob("*.py")):
        body = path.read_text(encoding="utf-8")
        prose.append(body)
        for tok in tokenize.generate_tokens(io.StringIO(body).readline):
            if tok.type not in (token_mod.COMMENT, token_mod.STRING):
                code.append(tok.string)
    code_text = " ".join(code).lower()

    assert "radiance" not in "".join(prose).lower(), "the frontend now mentions this target"
    # `num_warps` is Triton's OWN vocabulary, accepted precisely so it can be recorded and ignored;
    # these are the target's structure, which must stay below the convergence point.
    for word in ("lanes_per_warp", "simt", "scratchpad", "systolic", "shared_tensor"):
        assert word not in code_text, (
            f"the frontend now has code naming {word!r} — that is target structure, and it belongs "
            "below the convergence point")


def test_both_accelerators_descend_from_one_identical_module(arms):
    """The input to both descents is the same object — so any difference below is Merlin's."""
    bridged = arms["bridged"]
    again = to_linalg(source.make_ttir(K.repeated_rhs_matmul_spec()),
                      K.repeated_rhs_matmul_spec())
    assert again.text == bridged.text
    assert text(arms["radiance"].input_module) == text(arms["gemmini"].input_module)
    assert text(arms["radiance"].interface_module) == text(arms["gemmini"].interface_module)


def test_the_grid_was_normalized_not_lowered(arms):
    """No parallelism decision survives the frontend: no loop, no lanes, no warps."""
    core = arms["bridged"].text
    for token in ("scf.for", "scf.parallel", "affine.for", "gpu.", "warp", "lane", "program_id"):
        assert token not in core, f"the bridge baked {token!r} into its output"
    assert arms["bridged"].report.grid == (1, 1, 1)


def test_the_warp_mapping_is_the_targets_decision_not_the_frontends(arms):
    """The same module: Radiance records a warp width, Gemmini records none.

    This is the claim M5b exists for. Had the bridge chosen 'grid -> threads', Gemmini would be
    broken; had it chosen 'grid -> sequential loop', Radiance could never reach warps. It chose
    neither, so both are expressible.
    """
    radiance = text(arms["radiance"].target_module)
    gemmini = text(arms["gemmini"].target_module)
    assert "lanes_per_warp = 16" in radiance
    assert "lanes_per_warp" not in gemmini
    # And the width came from Radiance's contract, not from the kernel or the grid.
    package = _package(RADIANCE_PACKAGE)
    declared = package.contract["capabilities"]["simt"]["lanes_per_warp"]
    assert f"lanes_per_warp = {declared}" in radiance


def test_the_two_accelerators_really_are_different_shapes(arms):
    """Otherwise the portability claim is proven against a mirror."""
    radiance = {op.name for op in arms["radiance"].target_module.walk()}
    gemmini = {op.name for op in arms["gemmini"].target_module.walk()}
    assert not (radiance & gemmini) - {"builtin.module", "func.func", "func.return"}
    # Same target-independent ABI out the far end, though — that is what makes them comparable.
    assert ([c["opcode"] for c in arms["radiance"].command_buffer["commands"]]
            == [c["opcode"] for c in arms["gemmini"].command_buffer["commands"]])


def test_the_two_command_buffers_differ_only_in_the_target_name(arms):
    """The runtime ABI is target-independent, so the payload must be identical."""
    def normalized(cb):
        out = json.loads(json.dumps(cb, sort_keys=True))
        out.pop("target", None)
        out.pop("backend", None)
        return out

    assert normalized(arms["radiance"].command_buffer) == normalized(
        arms["gemmini"].command_buffer)


def test_the_elementwise_arm_reaches_radiances_vector_lanes():
    """Radiance declares an accelerated elementwise unit, and can now actually be handed one.

    This is what the interface.elementwise work was for: the capability was declared in Radiance's
    own contract long before anything could reach it. Gemmini's plan does not declare it, so the same
    kernel still routes to the generic path there — coverage stays per target.
    """
    from merlin import compile_core
    from merlin.runtime import reference_outputs, simulate

    spec = K.vector_add_i32_spec()
    bridged = to_linalg(source.make_ttir(spec), spec)

    radiance = compile_core.compile_core_mlir(
        bridged.module, target_package=_package(RADIANCE_PACKAGE))
    assert radiance.route.kind == "staged"
    for module in radiance.staged.modules():
        module.verify()
    assert "radiance.elementwise" in text(radiance.staged.target_module)
    # The warp obligation applies to every dispatch, not only to contractions.
    assert "lanes_per_warp = 16" in text(radiance.staged.target_module)
    cb = radiance.staged.command_buffer
    assert [c["opcode"] for c in cb["commands"]] == ["VECTOR_MAP"]
    assert simulate(cb)["outputs"] == reference_outputs(cb)

    gemmini_route = compile_core.choose_route(
        bridged.module, target_package=_package(GEMMINI_PACKAGE))
    assert gemmini_route.kind == "llvm", "coverage stopped being read per target"
