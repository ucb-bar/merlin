"""Private byte views permit value copies while the atomic stays in place."""

import pytest
from test_loop_tensor_destinations import rewrite

from merlin.llvmlower.toolchain import available


def source(*, private=True, fresh=True, no_inline=True, extra=False, float_body=False, dead=False):
    raw = "%raw" if private else "%external"
    allocation = "%raw=memref.alloc():memref<16xi8>" if private else ""
    start = "%seed" if fresh else "%input"
    boundary = "no_inline" if no_inline else ""
    additional = "memref.store %one,%view[%zero]:memref<?xi32>" if extra else ""
    update = "%xored=arith.xori %current,%x:i32\n%sum=arith.addi %xored,%one:i32"
    if float_body:
        update = "%fp=arith.sitofp %current:i32 to f32\n%sum=arith.fptosi %fp:f32 to i32"
    if dead:
        update += "\n%dead=arith.addi %current,%current:i32"
    return f"""module {{
      func.func @forward(%input:tensor<4xi32>,%external:memref<16xi8>,%bound:index)->tensor<4xi32> {{
        %zero=arith.constant 0:index
        %step=arith.constant 1:index
        %four=arith.constant 4:index
        %one=arith.constant 1:i32
        %empty=tensor.empty():tensor<4xi32>
        %seed=linalg.fill ins(%one:i32) outs(%empty:tensor<4xi32>)->tensor<4xi32>
        {allocation}
        %view=memref.view {raw}[%zero][%four]:memref<16xi8> to memref<?xi32>
        %result=scf.for %iv=%zero to %bound step %step iter_args(%old={start})->tensor<4xi32> {{
          %extent=memref.dim %view,%zero:memref<?xi32>
          %event=scf.execute_region ->tensor<4xi32> {boundary} {{
            {additional}
            %prior=linalg.generic {{
                indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],
                iterator_types=["parallel"]}}
                ins(%old:tensor<4xi32>) outs(%empty:tensor<4xi32>) {{
              ^bb0(%x:i32,%unused:i32):
                %j=linalg.index 0:index
                %valid=arith.cmpi slt,%j,%extent:index
                cf.assert %valid,"complete word bound"
                %v=memref.generic_atomic_rmw %view[%j]:memref<?xi32> {{
                ^bb1(%current:i32):
                  {update}
                  memref.atomic_yield %sum:i32
                }}
                linalg.yield %v:i32
            }} ->tensor<4xi32>
            scf.yield %prior:tensor<4xi32>
          }}
          %next=linalg.generic {{
                indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>],
                iterator_types=["parallel"]}}
              ins(%old,%event:tensor<4xi32>,tensor<4xi32>) outs(%empty:tensor<4xi32>) {{
            ^bb2(%a:i32,%b:i32,%unused:i32):
              %sum=arith.addi %a,%b:i32
              linalg.yield %sum:i32
          }} ->tensor<4xi32>
          scf.yield %next:tensor<4xi32>
        }}
        return %result:tensor<4xi32>
      }}
    }}"""


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize(
    "options,changed",
    (
        ({}, 1),
        ({"private": False}, 0),
        ({"fresh": False}, 0),
        ({"no_inline": False}, 0),
        ({"extra": True}, 0),
        ({"float_body": True}, 0),
        ({"dead": True}, 0),
    ),
)
def test_byte_word_snapshots_require_actual_private_root_and_qualified_body(tmp_path, options, changed):
    original = source(**options)
    report, rewritten = rewrite(tmp_path, original, atomic_support=True)
    assert f"OK loop_tensor_destinations {changed}" in report
    assert rewritten.count("memref.generic_atomic_rmw") == 1
    assert rewritten.count("cf.assert") == 1
    assert rewritten.count("scf.execute_region") == 1


def nested_source(*, private=True, fresh=True, no_inline=True, extra=False, opaque=False, shared_inner=False):
    allocation = "%raw=memref.alloc():memref<16xi8>" if private else ""
    root = "%raw" if private else "%external"
    start = "%seed" if fresh else "%input"
    boundary = "no_inline" if no_inline else ""
    effect = "memref.store %one,%view[%zero]:memref<?xi32>" if extra else ""
    declaration = "func.func private @unknown(tensor<4xi32>)->tensor<4xi32>" if opaque else ""
    captured = (
        "%read=func.call @unknown(%old):(tensor<4xi32>)->tensor<4xi32>"
        if opaque
        else """
      %read=linalg.generic {indexing_maps=[affine_map<(i)->(i)>],iterator_types=["parallel"]}
          outs(%empty:tensor<4xi32>) {
        ^bb0(%unused:i32):
          %j=linalg.index 0:index
          %v=memref.load %view[%j]:memref<?xi32>
          linalg.yield %v:i32
      } ->tensor<4xi32>"""
    )
    inner_seed = (
        "%unit=tensor.extract_slice %input[0][1][1]:tensor<4xi32> to tensor<1xi32>"
        if shared_inner
        else "%unit=tensor.empty():tensor<1xi32>"
    )
    return f"""module {{ {declaration}
      func.func @forward(%input:tensor<4xi32>,%external:memref<16xi8>,%bound:index)->tensor<4xi32> {{
        %zero=arith.constant 0:index
        %step=arith.constant 1:index
        %four=arith.constant 4:index
        %one=arith.constant 1:i32
        %empty=tensor.empty():tensor<4xi32>
        %seed=linalg.fill ins(%one:i32) outs(%empty:tensor<4xi32>)->tensor<4xi32>
        {allocation}
        %view=memref.view {root}[%zero][%four]:memref<16xi8> to memref<?xi32>
        %result=scf.for %iv=%zero to %bound step %step iter_args(%old={start})->tensor<4xi32> {{
          {captured}
          %event=scf.execute_region ->tensor<4xi32> no_inline {{
            %prior=linalg.generic {{
                indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],
                iterator_types=["parallel"]}}
                ins(%old:tensor<4xi32>) outs(%empty:tensor<4xi32>) {{
              ^bb0(%x:i32,%unused:i32):
                %j=linalg.index 0:index
                %before=tensor.extract %read[%j]:tensor<4xi32>
                %operand=arith.xori %x,%before:i32
                %values=tensor.from_elements %operand:tensor<1xi32>
                %indices=tensor.from_elements %j:tensor<1xindex>
                {inner_seed}
                %cast=memref.reinterpret_cast %view to offset:[0],sizes:[%four],strides:[1]
                    :memref<?xi32> to memref<?xi32>
                %dim=memref.dim %cast,%zero:memref<?xi32>
                scf.for %byte=%zero to %step step %step {{
                  %valid=arith.cmpi slt,%j,%dim:index
                  cf.assert %valid,"word bounds"
                  {effect}
                }}
                %nested=scf.execute_region ->tensor<1xi32> {boundary} {{
                  %old_word=linalg.generic {{
                indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>],
                iterator_types=["parallel"]}}
                      ins(%values,%indices:tensor<1xi32>,tensor<1xindex>) outs(%unit:tensor<1xi32>) {{
                    ^bb1(%value:i32,%index:index,%unused2:i32):
                      %v=memref.generic_atomic_rmw %cast[%index]:memref<?xi32> {{
                      ^bb2(%current:i32):
                        %sum=arith.addi %current,%value:i32
                        memref.atomic_yield %sum:i32
                      }}
                      linalg.yield %v:i32
                  }} ->tensor<1xi32>
                  scf.yield %old_word:tensor<1xi32>
                }}
                %word=tensor.extract %nested[%zero]:tensor<1xi32>
                linalg.yield %word:i32
            }} ->tensor<4xi32>
            scf.yield %prior:tensor<4xi32>
          }}
          scf.yield %event:tensor<4xi32>
        }}
        return %result:tensor<4xi32>
      }}
    }}"""


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize(
    "options,changed",
    (
        ({}, 1),
        ({"private": False}, 0),
        ({"fresh": False}, 0),
        ({"no_inline": False}, 0),
        ({"extra": True}, 0),
        ({"opaque": True}, 0),
        ({"shared_inner": True}, 0),
    ),
)
def test_nested_snapshots_close_views_tensor_reads_guards_and_effects(tmp_path, options, changed):
    report, rewritten = rewrite(tmp_path, nested_source(**options), atomic_support=True)
    assert f"OK loop_tensor_destinations {changed}" in report
    assert rewritten.count("memref.generic_atomic_rmw") == 1
    assert rewritten.count("cf.assert") == 1
    assert rewritten.count("scf.execute_region") == 2
    assert rewritten.count("scf.for") == 2
