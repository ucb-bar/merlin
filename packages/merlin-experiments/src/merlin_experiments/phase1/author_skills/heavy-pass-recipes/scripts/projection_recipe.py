"""Counted shared-input projections with separate original tensor pointers.

    python projection_recipe.py [--shape M,K,N1,N2] [--facts FILE] [--library DIR]

The neutral host check supplies no target encoding, overlap or completion law.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _support():
    if "recipe_support" not in sys.modules:
        spec = importlib.util.spec_from_file_location("recipe_support", Path(__file__).with_name("recipe_support.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["recipe_support"] = module
        spec.loader.exec_module(module)
    return sys.modules["recipe_support"]


def _source(m, k, widths):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    arguments = [f"%a:tensor<{m}x{k}xi8>"] + [f"%w{i}:tensor<{k}x{n}xi8>" for i, n in enumerate(widths)]
    operations = []
    for i, n in enumerate(widths):
        ty = f"tensor<{m}x{n}xi32>"
        operations.append(f"""%z{i}=arith.constant 0:i32
%init{i}="tensor.splat"(%z{i}):(i32)->{ty}
%r{i}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]}} ins(%a,%w{i}:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init{i}:{ty}) {{
^bb0(%x:i8,%w:i8,%s:i32):
%xx=arith.extsi %x:i8 to i32
%ww=arith.extsi %w:i8 to i32
%p=arith.muli %xx,%ww:i32
%v=arith.addi %s,%p:i32
linalg.yield %v:i32
}}->{ty}""")
    types = [f"tensor<{m}x{n}xi32>" for n in widths]
    return parse_mlir_text(
        "module {func.func @siblings("
        + ",".join(arguments)
        + ")->("
        + ",".join(types)
        + "){\n"
        + "\n".join(operations)
        + "\nreturn "
        + ",".join(f"%r{i}" for i in range(len(widths)))
        + ":"
        + ",".join(types)
        + "\n}}"
    )


def recipe(facts=None, shape=None):
    from dataclasses import replace

    from merlin.compile.scheduling.projection_dense import bind_dense_projection
    from merlin.compile.scheduling.projection_views import plan_shared_projection_views
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
    from merlin.sched.ir import Loop
    from merlin.transforms.certificates import ir_digest

    S = _support()
    caps = S.capacities(facts)
    edge_k, edge_n = caps["array"]
    m, k, *widths = shape or (2 * edge_n + 1, 2 * edge_k - 1, 2 * edge_n, 4 * edge_n)
    source = _source(m, k, widths)
    (group,) = find_shared_input_groups(source)
    views = plan_shared_projection_views(
        source, group, weight_element_bytes=1, output_element_bytes=(4,) * len(widths), tile_columns=edge_n
    )
    p, storage, resources = S.dense_view((m, k, sum(widths)), caps)
    a = S.adapter()
    block = S.pick(
        S.blocks(p, storage, resources),
        lambda b: b.k == p.tiles[1] and b.n in views.aligned_panel_tiles(resources.max_n_blocks),
    )
    S.SOURCE = ir_digest(source)
    original, _ = S.retained_pipeline(p, storage, resources, a, block, name="projections")
    paths = []

    def discover(body, path=()):
        for statement in body:
            if type(statement) is Loop:
                if statement.var == "outer_n":
                    paths.append((*path, statement.var))
                discover(statement.body, (*path, statement.var))

    discover(original.kernel.body)
    bound = bind_dense_projection(
        original.kernel, views, a, source_module=source, column_loop_paths=tuple(paths), panel_tiles=block.n
    )
    pointer_checks = 0

    def check_pointer(tag, fields):
        nonlocal pointer_checks
        if "projection_member" not in fields:
            return
        role = "weight" if tag == "weight_load" else "output"
        rectangle = views.rectangle(
            role,
            fields["src_row"] if role == "weight" else fields["row"],
            fields["src_col"] if role == "weight" else fields["col"],
            fields["rows"],
            fields["cols"],
        )
        assert rectangle.member == fields["projection_member"]
        assert rectangle.column == fields["projection_local_column"]
        assert rectangle.row_stride_bytes == fields["projection_row_columns"] * (1 if role == "weight" else 4)
        pointer_checks += 1

    # The neutral adapter's original field spellings stay unchanged. The
    # checked pointer operands are additional, not reconstructed tensor bytes.
    original.revalidate()
    bound.emit_index_program(S.Integers(), check_pointer, complete_epoch=lambda: None)
    checked_adapter = replace(
        a,
        project=lambda call: a.project(
            replace(call, args=tuple((name, value) for name, value in call.args if not name.startswith("projection_")))
        ),
    )
    numerics = S.execute_dense(bound.kernel, checked_adapter, p)
    original.revalidate()
    return dict(
        schema="neutral.counted_projection_recipe.v1",
        facts=caps["source"],
        shape=[m, k, list(widths)],
        source=ir_digest(source),
        block=[block.m, block.k, block.n],
        pointer_checks=pointer_checks,
        numerics=numerics,
        mapping=bound.report,
        shared_input_requested_bytes=original.realization["source"].get("requested_lhs_bytes"),
        original_output_boundaries=list(views.boundaries),
        timing="UNKNOWN",
        qualification="neutral host check and test-only laws; original target ABI/effects/completion still required",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts")
    parser.add_argument("--shape", help="M,K,N1,N2 (each N must contain whole primitive column tiles)")
    parser.add_argument("--library")
    args = parser.parse_args(argv)
    S = _support()
    S.use_library(args.library)
    shape = tuple(int(v) for v in args.shape.split(",")) if args.shape else None
    if shape is not None and (not 4 <= len(shape) <= 18 or min(shape) <= 0):
        parser.error("shape must contain M,K and 2..16 positive member widths")
    print(json.dumps(recipe(args.facts, shape), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
