"""Complete destination proofs at the real post-fusion native seam."""

import os
import subprocess

import pytest

from merlin.llvmlower.loop_tensor_destinations import FEATURE, MARKER, PRELUDE, bind_runner, edit_pipeline
from merlin.llvmlower.toolchain import available, compiler_python


def source(*, shape="4", init_read=False, opaque=False, later_use=False, reverse=False):
    empty = "%size" if shape == "?" else ""
    value = "%init" if init_read else "%one"
    payload = f"%v=arith.addf %in,{value}:f32"
    declaration = ""
    if opaque:
        declaration = "func.func private @unknown(f32)->f32"
        payload = "%v=func.call @unknown(%in):(f32)->f32"
    type_ = f"tensor<{shape}xf32>"
    input_map = "affine_map<(i)->(3-i)>" if reverse else "affine_map<(i)->(i)>"
    after = f"%observed=tensor.extract %old[%c0]:{type_}\nmemref.store %observed,%sink[%c0]:memref<4xf32>" if later_use else ""
    return f"""module {{ {declaration}
      func.func @forward(%x:{type_},%bound:index,%sink:memref<4xf32>) -> {type_} {{
        %c0=arith.constant 0:index
        %c1=arith.constant 1:index
        %one=arith.constant 1.0:f32
        %seven=arith.constant 7.0:f32
        %size=tensor.dim %x,%c0:{type_}
        %e=tensor.empty({empty}):{type_}
        %seed=linalg.fill ins(%seven:f32) outs(%e:{type_})->{type_}
        %result=scf.for %iv=%c0 to %bound step %c1 iter_args(%old=%x)->{type_} {{
          %next=linalg.generic {{indexing_maps=[{input_map},affine_map<(i)->(i)>],iterator_types=["parallel"]}}
            ins(%old:{type_}) outs(%seed:{type_}) {{
            ^bb0(%in:f32,%init:f32):
              {payload}
              linalg.yield %v:f32
          }} -> {type_}
          {after}
          scf.yield %next:{type_}
        }}
        return %result:{type_}
      }}
    }}"""


def rewrite(tmp_path, text):
    original, output, script = (tmp_path / name for name in ("original.mlir", "rewritten.mlir", "rewrite.py"))
    original.write_text(text)
    script.write_text("from torch_mlir import ir\nimport sys\n_run_stages = None\n" + PRELUDE + """
with ir.Context() as ctx:
    module = ir.Module.parse(open(sys.argv[1]).read())
    module.operation.verify()
    _ltd_reuse(ctx, module)
    open(sys.argv[2], 'w').write(str(module))
""")
    result = subprocess.run([str(compiler_python()), str(script), str(original), str(output)],
        check=True, capture_output=True, text=True, timeout=20,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    return result.stdout, output.read_text()


def atomic_source(*, private=True, fresh=True, no_inline=True, unknown=False, init_read=False, extra=False):
    target = "%private" if private else "%external"
    allocation = "%private=memref.alloc():memref<4xi32>" if private else ""
    seed = "%seed" if fresh else "%input"
    boundary = "no_inline" if no_inline else ""
    effect = "memref.store %one,%external[%zero]:memref<4xi32>" if extra else ""
    payload = ("%v=func.call @unknown(%x):(i32)->i32" if unknown else
               f"%v=memref.atomic_rmw addi %one,{target}[%j]:(i32,memref<4xi32>)->i32")
    value = "%unused" if init_read else "%v"
    declaration = "func.func private @unknown(i32)->i32" if unknown else ""
    return f'''module {{ {declaration}
      func.func @forward(%input:tensor<4xi32>,%external:memref<4xi32>,%bound:index)->tensor<4xi32> {{
        %zero=arith.constant 0:index
        %step=arith.constant 1:index
        %one=arith.constant 1:i32
        %empty=tensor.empty():tensor<4xi32>
        %seed=linalg.fill ins(%one:i32) outs(%empty:tensor<4xi32>)->tensor<4xi32>
        {allocation}
        %result=scf.for %iv=%zero to %bound step %step iter_args(%old={seed})->tensor<4xi32> {{
          %event=scf.execute_region ->tensor<4xi32> {boundary} {{
            {effect}
            %prior=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
                ins(%old:tensor<4xi32>) outs(%empty:tensor<4xi32>) {{
              ^bb0(%x:i32,%unused:i32):
                %j=linalg.index 0:index
                {payload}
                linalg.yield {value}:i32
            }} ->tensor<4xi32>
            scf.yield %prior:tensor<4xi32>
          }}
          %next=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
              ins(%old,%event:tensor<4xi32>,tensor<4xi32>) outs(%empty:tensor<4xi32>) {{
            ^bb0(%a:i32,%b:i32,%unused:i32):
              %sum=arith.addi %a,%b:i32
              linalg.yield %sum:i32
          }} ->tensor<4xi32>
          scf.yield %next:tensor<4xi32>
        }}
        return %result:tensor<4xi32>
      }}
    }}'''


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,changed", [({}, 1), ({"private": False}, 0), ({"fresh": False}, 0),
    ({"no_inline": False}, 0), ({"unknown": True}, 0), ({"init_read": True}, 0), ({"extra": True}, 0)])
def test_atomic_result_reuse_requires_closed_private_exactly_once_snapshot(tmp_path, options, changed):
    source = atomic_source(**options)
    report, rewritten = rewrite(tmp_path, source)
    assert f"OK loop_tensor_destinations {changed}" in report
    assert rewritten.count("memref.atomic_rmw") == source.count("memref.atomic_rmw")
    assert rewritten.count("scf.execute_region") == 1
    if options.get("no_inline", True):
        assert "no_inline" in rewritten


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("dtype,kind", [("f32", "assign"), ("f64", "assign"), ("f64", "addf")])
@pytest.mark.parametrize("private,changed", [(True, 1), (False, 0)])
def test_floating_atomic_snapshot_keeps_private_effect_and_old_value(tmp_path, dtype, kind, private, changed):
    text = atomic_source(private=private).replace("i32", dtype).replace("memref.atomic_rmw addi", f"memref.atomic_rmw {kind}")
    text = text.replace(f"arith.constant 1:{dtype}", f"arith.constant 1.0:{dtype}")
    text = text.replace(f"arith.addi", "arith.addf")
    report, rewritten = rewrite(tmp_path, text)
    assert f"OK loop_tensor_destinations {changed}" in report
    assert rewritten.count("memref.atomic_rmw") == 1
    assert "no_inline" in rewritten


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("shape,init_read,opaque,changed", [
    ("4", False, False, 1), ("17", False, False, 1),
    ("4", True, False, 1), ("0", False, False, 0),
    ("?", False, False, 0), ("4", False, True, 0),
])
def test_complete_destinations_or_value_copies_preserve_original_producers(tmp_path, shape, init_read, opaque, changed):
    stdout, text = rewrite(tmp_path, source(shape=shape, init_read=init_read, opaque=opaque))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "scf.for" in text
    assert ('prov.op = "copy-selected-state"' in text) == init_read


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("dtype", ["f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ"])
@pytest.mark.parametrize("opaque", [False, True])
def test_fp8_destination_proofs_remain_closed_over_numeric_values(tmp_path, dtype, opaque):
    stdout, text = rewrite(tmp_path, source(opaque=opaque).replace("f32", dtype))
    assert f"OK loop_tensor_destinations {int(not opaque)}" in stdout
    assert "scf.for" in text
    assert dtype in text


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options", [{"later_use": True}, {"reverse": True}])
def test_old_state_reads_and_permutations_finish_before_destination_updates(tmp_path, options):
    stdout, output = rewrite(tmp_path, source(**options))
    assert "OK loop_tensor_destinations 1" in stdout
    assert 'prov.op = "copy-selected-state"' in output
    if options.get("later_use"):
        assert output.index("memref.store") < output.index('prov.op = "copy-selected-state"')


def read_source(*, fresh=True, readonly=True, generalize=False, conditional=True, write=False, opaque=False, constant=False, while_state=None, outside_read=False, private=False, unknown_owner=False, tensor_read=False, tensor_view=False, read_seed=False, seed_init_read=False):
    marker = "read_only" if readonly else ""
    seed = "%seed" if fresh else "%x"
    declaration = "func.func private @observe()" if opaque else ""
    if unknown_owner:
        declaration += "func.func private @allocate()->memref<4xf32>"
    extra = "func.call @observe() : () -> ()" if opaque else ""
    store = "memref.store %v, %mutable[%i] : memref<4xf32>" if write else ""
    read = "%v = memref.load %view[%i] : memref<?xf32>"
    hoisted = "%v=memref.load %view[%c0]:memref<?xf32>" if outside_read else ""
    if outside_read:
        read = ""
    if conditional and not outside_read:
        read = f"""%yes=arith.cmpi slt,%i,%c2:index
          %v=scf.if %yes -> f32 {{
            %r=memref.load %view[%i]:memref<?xf32>
            scf.yield %r:f32
          }} else {{ {extra} scf.yield %zero:f32 }}"""
    tensor_import = ""
    if tensor_read:
        tensor_source, tensor_type = "%x", "tensor<4xf32>"
        if private or unknown_owner:
            tensor_source = "%snapshot"
            tensor_import = "%snapshot=bufferization.to_tensor %base restrict:memref<4xf32> to tensor<4xf32>"
        if tensor_view:
            tensor_import += f"\n%tensor_view=tensor.cast {tensor_source}:{tensor_type} to tensor<?xf32>"
            tensor_source, tensor_type = "%tensor_view", "tensor<?xf32>"
        read = read.replace("memref.load %view[%i] : memref<?xf32>", f"tensor.extract {tensor_source}[%i] : {tensor_type}")
        read = read.replace("memref.load %view[%i]:memref<?xf32>", f"tensor.extract {tensor_source}[%i]:{tensor_type}")
    fill = "%seed=linalg.fill ins(%zero:f32) outs(%empty:tensor<4xf32>)->tensor<4xf32>"
    if generalize:
        fill = """%seed=linalg.generic {indexing_maps=[affine_map<(i)->()>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
          ins(%zero:f32) outs(%empty:tensor<4xf32>) {^bb0(%v:f32,%init:f32):linalg.yield %v:f32}->tensor<4xf32>"""
    if constant:
        fill = "%seed=arith.constant dense<0.0>:tensor<4xf32>"
    if read_seed:
        yielded = "%value"
        read_init = ""
        if seed_init_read:
            read_init, yielded = "%with_init=arith.addf %value,%init:f32", "%with_init"
        fill = f'''%seed=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
          ins(%x:tensor<4xf32>) outs(%empty:tensor<4xf32>) {{^bb0(%a:f32,%init:f32):
            %i=linalg.index 0:index
            %loaded=memref.load %view[%i]:memref<?xf32>
            %value=arith.addf %a,%loaded:f32
            {read_init}
            linalg.yield {yielded}:f32
          }}->tensor<4xf32>'''
    imported = f"%base=bufferization.to_buffer %x {marker}:tensor<4xf32> to memref<4xf32>"
    if private:
        imported = f"""%source=bufferization.to_buffer %x {marker}:tensor<4xf32> to memref<4xf32>
          %base=memref.alloc():memref<4xf32>
          memref.copy %source,%base:memref<4xf32> to memref<4xf32>"""
    if unknown_owner:
        imported = "%base=func.call @allocate():()->memref<4xf32>"
    header = f"%result=scf.for %iv=%c0 to %bound step %c1 iter_args(%old={seed})->tensor<4xf32> {{"
    tail = "scf.yield %next:tensor<4xf32>"
    returned = "%result"
    if while_state is not None:
        forwarded = {"identity": "%before,%other_before", "computed": "%x,%other_before",
                     "permuted": "%other_before,%before"}[while_state]
        header = f"""%other_seed=arith.constant dense<9.0>:tensor<4xf32>
        %result:3=scf.while (%before={seed},%other_before=%other_seed,%iv_before=%c0)
          :(tensor<4xf32>,tensor<4xf32>,index)->(tensor<4xf32>,tensor<4xf32>,index) {{
          %continue=arith.cmpi slt,%iv_before,%bound:index
          scf.condition(%continue) {forwarded},%iv_before:tensor<4xf32>,tensor<4xf32>,index
        }} do {{ ^bb0(%old:tensor<4xf32>,%other:tensor<4xf32>,%iv:index):"""
        tail = "%next_iv=arith.addi %iv,%c1:index\nscf.yield %next,%other,%next_iv:tensor<4xf32>,tensor<4xf32>,index"
        returned = "%result#0"
    return f"""module {{ {declaration}
      func.func @forward(%x:tensor<4xf32>,%bound:index,%mutable:memref<4xf32>)->tensor<4xf32> {{
        %c0=arith.constant 0:index
        %c1=arith.constant 1:index
        %c2=arith.constant 2:index
        %zero=arith.constant 0.0:f32
        {imported}
        %view=memref.cast %base:memref<4xf32> to memref<?xf32>
        {tensor_import}
        %empty=tensor.empty():tensor<4xf32>
        {fill}
        {header}
          {hoisted}
          %next=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
            ins(%old:tensor<4xf32>) outs(%empty:tensor<4xf32>) {{^bb0(%in:f32,%init:f32):
              %i=linalg.index 0:index
              {read}
              {store}
              %sum=arith.addf %in,%v:f32
              linalg.yield %sum:f32
            }}->tensor<4xf32>
          {tail}
        }}
        return {returned}:tensor<4xf32>
      }}
    }}"""


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,changed", [({}, 1), ({"generalize": True}, 1), ({"constant": True}, 1), ({"conditional": False}, 1),
    ({"fresh": False}, 0), ({"readonly": False}, 0), ({"write": True}, 0), ({"opaque": True}, 0)])
def test_read_capture_requires_fresh_seed_and_closed_readonly_entry_origin(tmp_path, options, changed):
    stdout, output = rewrite(tmp_path, read_source(**options))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "memref.load" in output and "scf.for" in output
    if options.get("conditional", True):
        assert "scf.if" in output


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,changed", [({}, 1), ({"fresh": False}, 0), ({"readonly": False}, 0)])
def test_scalar_read_captures_close_their_loop_local_producer_graph(tmp_path, options, changed):
    stdout, output = rewrite(tmp_path, read_source(conditional=False, outside_read=True, **options))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "memref.load" in output


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,changed", [({}, 1), ({"private": True}, 1), ({"fresh": False}, 0),
    ({"readonly": False}, 0), ({"unknown_owner": True}, 0), ({"seed_init_read": True}, 0)])
def test_read_computed_seeds_require_independent_unused_output_storage(tmp_path, options, changed):
    stdout, output = rewrite(tmp_path, read_source(read_seed=True, **options))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "memref.load" in output and "scf.if" in output


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,changed", [
    ({"private": True}, 1), ({"private": True, "readonly": False}, 1),
    ({"private": True, "fresh": False}, 0), ({"unknown_owner": True}, 0),
])
def test_private_read_roots_require_actual_allocations_and_independent_seeds(tmp_path, options, changed):
    stdout, _ = rewrite(tmp_path, read_source(**options))
    assert f"OK loop_tensor_destinations {changed}" in stdout


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,changed", [
    ({}, 1), ({"tensor_view": True}, 1), ({"conditional": False}, 1),
    ({"private": True}, 1), ({"fresh": False}, 0), ({"unknown_owner": True}, 0),
])
def test_tensor_reads_require_entry_values_or_separate_private_snapshots(tmp_path, options, changed):
    stdout, output = rewrite(tmp_path, read_source(tensor_read=True, **options))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "tensor.extract" in output


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,restricted,mixed,changed", [
    ({}, True, False, 2), ({}, True, True, 2), ({}, False, False, 0),
    ({"unknown_owner": True}, True, False, 0), ({"opaque": True}, True, False, 0),
    ({"write": True}, True, False, 0),
])
def test_private_imported_carried_seed_is_copied_only_after_complete_proof(tmp_path, options, restricted, mixed, changed):
    program = read_source(private=True, tensor_read=True, fresh=False, **options)
    program = program.replace("iter_args(%old=%x)", "iter_args(%old=%snapshot)")
    if not restricted:
        program = program.replace("%snapshot=bufferization.to_tensor %base restrict:",
                                  "%snapshot=bufferization.to_tensor %base:")
    if mixed:
        program = program.replace("%sum=arith.addf %in,%v:f32", """%wide=arith.extf %in:f32 to f64
              %other=arith.extf %v:f32 to f64
              %wide_sum=arith.addf %wide,%other:f64
              %sum=arith.truncf %wide_sum:f64 to f32""")
    stdout, output = rewrite(tmp_path, program)
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert ('prov.op = "seed-private-carried-value"' in output) == bool(changed)
    assert "bufferization.to_tensor" in output and "tensor.extract" in output


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("state,changed", [("identity", 2), ("computed", 0), ("permuted", 0)])
def test_private_imported_while_seed_requires_unchanged_condition_state(tmp_path, state, changed):
    program = read_source(private=True, tensor_read=True, fresh=False, while_state=state)
    program = program.replace("(%before=%x,", "(%before=%snapshot,")
    stdout, output = rewrite(tmp_path, program)
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert ('prov.op = "seed-private-carried-value"' in output) == bool(changed)


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("state,options,changed", [
    ("identity", {}, 1), ("identity", {"generalize": True}, 1),
    ("identity", {"constant": True}, 1), ("identity", {"fresh": False}, 0),
    ("identity", {"readonly": False}, 0), ("identity", {"write": True}, 0),
    ("identity", {"opaque": True}, 0), ("computed", {}, 0), ("permuted", {}, 0),
])
def test_while_destinations_require_unmodified_condition_state(tmp_path, state, options, changed):
    stdout, output = rewrite(tmp_path, read_source(while_state=state, **options))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "scf.while" in output and "scf.condition" in output and "memref.load" in output


def branch_source(*, while_loop=True, nested=False, passthrough=False, opaque=False, multiple=False, inside=False, reduction=False, later_use=False):
    type_ = "tensor<4xf32>"
    returned = "%result#0" if while_loop else "%result"
    header = f"""%result:2=scf.while (%before=%seed,%step_before=%zero_index)
        :({type_},index)->({type_},index) {{
        %continue=arith.cmpi slt,%step_before,%bound:index
        scf.condition(%continue) %before,%step_before:{type_},index
      }} do {{ ^bb0(%old:{type_},%step:index):"""
    tail = f"%next_step=arith.addi %step,%one_index:index\nscf.yield %chosen,%next_step:{type_},index"
    if not while_loop:
        header = f"%result=scf.for %step=%zero_index to %bound step %one_index iter_args(%old=%seed)->{type_} {{"
        tail = f"scf.yield %chosen:{type_}"
    declaration = "func.func private @unknown(f32)->f32" if opaque else ""
    arithmetic = "%v=func.call @unknown(%in):(f32)->f32" if opaque else "%v=arith.addf %in,%one:f32"
    value = "%old" if passthrough else "%inside" if inside else "%capture"
    extra = f"%inside=arith.constant dense<2.0>:{type_}" if inside else ""
    after = f"%observed=tensor.extract %old[%zero_index]:{type_}\nmemref.store %observed,%sink[%zero_index]:memref<4xf32>" if later_use else ""
    computed = f"""%next=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
      ins(%old:{type_}) outs(%empty:{type_}) {{^bb0(%in:f32,%init:f32):
      {arithmetic}
      linalg.yield %v:f32
    }}->{type_}"""
    if reduction:
        computed = f"""%next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>],iterator_types=["parallel","reduction"]}}
          ins(%matrix:tensor<4x3xf32>) outs(%seed:{type_}) {{^bb0(%in:f32,%init:f32):
          %v=arith.addf %in,%init:f32
          linalg.yield %v:f32
        }}->{type_}"""
    if nested:
        computed = f"""%next=scf.if %choose -> {type_} {{
          scf.yield %old:{type_}
        }} else {{ {computed} scf.yield %next:{type_} }}"""
    if multiple:
        branch = f"""%chosen_pair:2=scf.if %choose -> (index,{type_}) {{
          scf.yield %zero_index,{value}:index,{type_}
        }} else {{ {computed} scf.yield %one_index,%next:index,{type_} }}"""
        tail = tail.replace("%chosen", "%chosen_pair#1")
    else:
        branch = f"""%chosen=scf.if %choose -> {type_} {{
          scf.yield {value}:{type_}
        }} else {{ {computed} scf.yield %next:{type_} }}"""
    return f"""module {{ {declaration}
      func.func @forward(%capture:{type_},%matrix:tensor<4x3xf32>,%bound:index,%choose:i1,%sink:memref<4xf32>)->{type_} {{
        %zero_index=arith.constant 0:index
        %one_index=arith.constant 1:index
        %zero=arith.constant 0.0:f32
        %one=arith.constant 1.0:f32
        %empty=tensor.empty():{type_}
        %seed=linalg.fill ins(%zero:f32) outs(%empty:{type_})->{type_}
        {header}
          {extra}
          {branch}
          {after}
          {tail}
        }}
        return {returned}:{type_}
      }}
    }}"""


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,changed", [
    ({}, 2), ({"while_loop": False}, 2), ({"nested": True}, 2),
    ({"passthrough": True}, 1), ({"multiple": True}, 2),
    ({"opaque": True}, 0), ({"inside": True}, 2), ({"reduction": True}, 2),
    ({"later_use": True}, 1), ({"later_use": True, "opaque": True}, 0),
])
def test_branch_state_qualifies_every_path_before_reusing_destinations(tmp_path, options, changed):
    stdout, output = rewrite(tmp_path, branch_source(**options))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "scf.if" in output and ("scf.for" if options.get("while_loop") is False else "scf.while") in output
    assert ('prov.op = "copy-selected-state"' in output) == (changed == 2 or changed == 1 and options.get("later_use", False))
    if options.get("multiple"):
        assert "index, tensor<4xf32>" in output


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("tensor_read", [False, True])
def test_function_cfg_phi_is_not_an_independent_entry_tensor(tmp_path, tensor_read):
    source = '''module {
      func.func @forward(%bound:index)->tensor<4xf32> {
        %zero=arith.constant 0.0:f32
        %c0=arith.constant 0:index
        %c1=arith.constant 1:index
        %empty=tensor.empty():tensor<4xf32>
        %seed=linalg.fill ins(%zero:f32) outs(%empty:tensor<4xf32>)->tensor<4xf32>
        cf.br ^other(%seed:tensor<4xf32>)
      ^other(%alias:tensor<4xf32>):
        %base=bufferization.to_buffer %alias read_only:tensor<4xf32> to memref<4xf32>
        %result=scf.for %iv=%c0 to %bound step %c1 iter_args(%old=%seed)->tensor<4xf32> {
          %next=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
            ins(%old:tensor<4xf32>) outs(%empty:tensor<4xf32>) {^bb0(%in:f32,%init:f32):
              %i=linalg.index 0:index
              %read=memref.load %base[%i]:memref<4xf32>
              %sum=arith.addf %in,%read:f32
              linalg.yield %sum:f32
            }->tensor<4xf32>
          scf.yield %next:tensor<4xf32>
        }
        return %result:tensor<4xf32>
      }
    }'''
    if tensor_read:
        source = source.replace("memref.load %base[%i]:memref<4xf32>", "tensor.extract %alias[%i]:tensor<4xf32>")
    stdout, _ = rewrite(tmp_path, source)
    assert "OK loop_tensor_destinations 0" in stdout


def test_explicit_feature_places_rewrite_after_fusion_before_bufferization():
    passes = ["canonicalize", "func.func(linalg-fuse-elementwise-ops)", "one-shot-bufferize{bufferize-function-boundaries}", "canonicalize"]
    edited = edit_pipeline(passes)
    assert edited[2:4] == [MARKER, passes[2]]
    original = "def _run_stages(*args): pass\n_run_stages(ctx, module, pipeline, 0)\n"
    assert bind_runner(original, selected=False) == original
    assert "_ltd_reuse" in bind_runner(original, selected=True)
    with pytest.raises(ValueError, match="one native stage runner call"):
        bind_runner("pass\n", selected=True)
    with pytest.raises(ValueError, match="one unmodified bufferization seam"):
        edit_pipeline([*passes, passes[2]])
    from merlin.llvmlower.impr_features import known
    assert FEATURE in known()
    from merlin.llvmlower.pipeline import _upstream_pipeline
    assert MARKER in _upstream_pipeline(frozenset({FEATURE}))


@pytest.mark.parametrize("pipeline", [
    "one-shot-bufferize{}", MARKER + ",canonicalize,one-shot-bufferize{}",
    MARKER + "," + MARKER + ",one-shot-bufferize{}",
    "one-shot-bufferize{}," + MARKER + ",one-shot-bufferize{}",
])
def test_missing_duplicate_or_moved_rewrite_seam_refuses(pipeline):
    scope = {"_run_stages": lambda *args: pytest.fail("unsafe pipeline reached native lowering")}
    exec(PRELUDE, scope)
    with pytest.raises(ValueError, match="marker|immediately precede"):
        scope["_run_stages"](None, None, pipeline, 0)
