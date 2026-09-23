"""Two provenance lists read off one module must describe the SAME operations.

``linalg_summary`` returns per-op lists -- the op name, its family, the operation carrying the tag, and
whether that operation reduces -- and ``model_op_demands`` indexes them all by one counter. That is only
sound if they are derived by one construction. ``prov.op`` and ``prov.family`` are separate tags and are
NOT in one-to-one correspondence: measured on `M1_lstmnetvit_fp32`, 732 op tags against 723 family tags,
so nine operations state an op and no family. Split separately and indexed positionally, every family
from tag 662 onward belonged to the next operation. Five capsules carry that drift, the whole lstmnetvit
family, nine each.

The second half is what the aligned family is FOR. ``prov.family`` is stamped on every op a captured
region produced, so an im2col gather -- pure movement, every iterator parallel -- is tagged
``contraction`` because the convolution it belongs to is one. Excluding the region's `tensor.*` ops by
requiring a `linalg.` carrier does not exclude it: a parallel-only `linalg.generic` passes that test and
is just as much not a contraction. Measured: 13 capsules, 215 such generics, 54 of them in
`SY_model_resnet50` alone, whose real contractions are its 53 named `linalg.matmul` ops.
"""

from __future__ import annotations

from merlin.common.paths import merlin_dir
from merlin.targetgen.capsule_source import linalg_summary, model_op_demands

_GENERIC = (
    "%0 = linalg.generic {{indexing_maps = [], iterator_types = [{iters}]}} "
    "ins(%a : tensor<4x4xi8>) outs(%b : tensor<4x4xi8>) attrs = {{{tags}}} {{ ^bb0: }}\n"
)


def _module(*ops: str) -> str:
    return "module {\n" + "".join(ops) + "}\n"


def _op(iters: str, op: str, family: str | None) -> str:
    tags = f'prov.op = "{op}"' + (f', prov.family = "{family}"' if family is not None else "")
    return _GENERIC.format(iters=iters, tags=tags)


def test_the_four_per_op_lists_are_the_same_length():
    text = _module(
        _op('"parallel", "reduction"', "matmul", "contraction"),
        _op('"parallel"', "relu", None),  # states an op and NO family
        _op('"parallel", "reduction"', "matmul", "contraction"),
    )
    s = linalg_summary(text)
    assert len({len(s[k]) for k in ("prov_ops", "prov_families", "carrying_ops", "carrier_reduces")}) == 1


def test_a_missing_family_does_not_shift_every_family_after_it():
    """The defect itself: with a positional split the third op would inherit the second's family."""
    text = _module(
        _op('"parallel", "reduction"', "matmul", "contraction"),
        _op('"parallel"', "relu", None),
        _op('"parallel"', "sigmoid", "elementwise_map"),
    )
    s = linalg_summary(text)
    assert s["prov_ops"] == ["matmul", "relu", "sigmoid"]
    assert s["prov_families"] == ["contraction", "", "elementwise_map"]


def test_the_reduction_flag_is_read_per_tag():
    text = _module(
        _op('"parallel", "parallel"', "convolution_im2col_matmul", "contraction"),  # the gather
        _op('"parallel", "reduction"', "convolution_im2col_matmul", "contraction"),  # the contraction
    )
    assert linalg_summary(text)["carrier_reduces"] == [False, True]


def test_a_parallel_only_generic_is_not_routed_as_a_contraction():
    text = _module(
        _op('"parallel", "parallel"', "convolution_im2col_matmul", "contraction"),
        _op('"parallel", "reduction"', "convolution_im2col_matmul", "contraction"),
    )
    families = [d.family for d in model_op_demands(text, "int8")]
    assert families == [None, "contraction"], "the gather must not claim the region's family"


def test_the_real_capture_has_both_shapes_under_one_tag():
    """`M2_microvit_gemmini` is the capsule this was found on: three generics tagged
    `convolution_im2col_matmul`, of which exactly one reduces."""
    path = merlin_dir() / "contract/capsules/model/M2_microvit_gemmini/capsule.interface.mlir"
    s = linalg_summary(path.read_text(encoding="utf-8"))
    rows = [
        r
        for o, f, c, r in zip(s["prov_ops"], s["prov_families"], s["carrying_ops"], s["carrier_reduces"])
        if c == "linalg.generic" and f == "contraction"
    ]
    assert rows == [False, True, False], rows


def test_no_capsule_gains_mesh_work_from_the_correction():
    """The direction that matters: an op may leave the mesh, never join it. If this ever fails, the
    family is being WIDENED somewhere, which is how a host op gets asked of the accelerator."""
    path = merlin_dir() / "contract/capsules/model/M2_microvit_gemmini/capsule.interface.mlir"
    demands = model_op_demands(path.read_text(encoding="utf-8"), "int8")
    contractions = [d for d in demands if d.family == "contraction"]
    assert len(contractions) <= len(demands)
    # the two gathers are gone from the contraction set, the reducing one remains
    convs = [d for d in contractions if d.op == "convolution_im2col_matmul"]
    assert len(convs) == 1, f"exactly one of the three conv-tagged generics contracts, got {len(convs)}"


def test_no_affected_capsule_is_left_with_nothing_on_the_mesh():
    """The failure mode worth guarding: an op may leave the mesh, but a capsule whose point is mesh
    execution must not end up with zero mesh work and fail a drives-accelerator check for a reason
    nobody intended. Measured over the whole corpus when this landed: 145 capsules, 17 whose mesh count
    changes, none reduced to zero."""
    from merlin.targetgen import routing

    for name in ("SY_model_resnet50", "M2_microvit_gemmini", "M1_lstmnetvit_gemmini", "GC1_depthwise_bf16_pt"):
        path = next(merlin_dir().joinpath("contract/capsules").rglob(f"{name}/capsule.interface.mlir"), None)
        if path is None:
            continue
        demands = model_op_demands(path.read_text(encoding="utf-8"), "int8")
        mesh = routing.route_plan(demands, "gemmini").get("mesh") or []
        assert mesh, f"{name} has no mesh work left after the correction"
