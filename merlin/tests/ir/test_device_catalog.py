"""Compiled ABI and source-binding checks for an external dense device catalog."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from merlin.llvmlower.device_catalog import build_catalog_objects
from merlin.llvmlower.device_shim import emit_dense_translation_unit

_CC = shutil.which("cc") or shutil.which("clang")
pytestmark = pytest.mark.target("gemmini")


def test_catalog_selects_an_exact_source_region(monkeypatch):
    from xdsl.dialects.builtin import StringAttr

    from merlin.common import mlir_query as mq
    from merlin.llvmlower.device_offload import rewrite_contractions_to_device
    from merlin.system import offload

    monkeypatch.setattr(offload, "device_dtype_triples", lambda _device: (("i8", "i8", "i32"),))
    module = mq.parse("""
module {
  func.func @forward(%a: tensor<3x5xi8>, %b: tensor<5x7xi8>) -> tensor<3x7xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<3x7xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<3x7xi32>) -> tensor<3x7xi32>
    %o = linalg.matmul ins(%a, %b : tensor<3x5xi8>, tensor<5x7xi8>)
                       outs(%f : tensor<3x7xi32>) -> tensor<3x7xi32>
    return %o : tensor<3x7xi32>
  }
}
""")
    op = next(mq.walk(module, "linalg.matmul"))
    op.attributes["prov.region_id"] = StringAttr("matmul_0")
    result = rewrite_contractions_to_device(
        module,
        "gemmini",
        catalog_selection={"matmul_0": ("tensor<3x5xi8>", "tensor<5x7xi8>", "tensor<3x7xi32>")},
        model_sha256="prepared-model-sha",
    )
    assert result.moved == 1
    assert result.routed[0].source_region == "matmul_0"
    assert result.routed[0].tensor_types == ("tensor<3x5xi8>", "tensor<5x7xi8>", "tensor<3x7xi32>")
    assert result.model_sha256 == "prepared-model-sha"


def test_catalog_explicit_source_conversion_routes_f32_without_claiming_native_f32(monkeypatch):
    from xdsl.dialects.builtin import StringAttr

    from merlin.common import mlir_query as mq
    from merlin.llvmlower.device_catalog import catalog_source_conversion
    from merlin.llvmlower.device_offload import rewrite_contractions_to_device
    from merlin.system import offload

    monkeypatch.setattr(offload, "device_dtype_triples", lambda _device: (("i8", "i8", "i32"),))
    source = """
module {
  func.func @forward(%a: tensor<3x5xf32>, %b: tensor<5x7xf32>) -> tensor<3x7xf32> {
    %z = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<3x7xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<3x7xf32>) -> tensor<3x7xf32>
    %o = linalg.matmul ins(%a, %b : tensor<3x5xf32>, tensor<5x7xf32>)
                       outs(%f : tensor<3x7xf32>) -> tensor<3x7xf32>
    return %o : tensor<3x7xf32>
  }
}
"""
    types = ("tensor<3x5xf32>", "tensor<5x7xf32>", "tensor<3x7xf32>")
    policy = {"format": "declared_by_test_provider", "rounding": "rne"}
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    conversion = {
        "schema": "target_owned_dtype_conversion_v1",
        "source_dtypes": ["f32", "f32", "f32"],
        "target_dtypes": [["i8", "i8", "i32"]],
        "policy": policy,
        "policy_sha256": digest,
    }

    def prepared():
        module = mq.parse(source)
        op = next(mq.walk(module, "linalg.matmul"))
        op.attributes["prov.region_id"] = StringAttr("matmul_0")
        ordinal = next(index for index, entry in enumerate(module.walk()) if entry is op)
        row = {"region": "matmul_0", "symbol": "converted_kernel",
               "source_operation_ordinal": ordinal, "tensor_types": list(types),
               "target_dtypes": ["i8", "i8", "i32"]}
        manifest = {"abi": {"dtypes": ["f32", "f32", "f32"]},
                    "bindings": [row], "source_conversion": conversion}
        return module, row, manifest

    module, row, manifest = prepared()
    assert not offload.offloadable_contractions(module, "gemmini")
    assert catalog_source_conversion(manifest, "gemmini") == ("f32", "f32", "f32")
    result = rewrite_contractions_to_device(
        module, "gemmini", catalog_selection={"matmul_0": types},
        catalog_bindings={"matmul_0": row}, catalog_manifest_data=manifest,
    )
    assert result.moved == 1
    assert result.routed[0].dtypes == ("f32", "f32", "f32")

    _, row, manifest = prepared()
    manifest["source_conversion"] = {**conversion, "policy_sha256": "0" * 64}
    with pytest.raises(ValueError, match="policy hash"):
        catalog_source_conversion(manifest, "gemmini")
    manifest["source_conversion"] = {**conversion, "target_dtypes": [["f32", "f32", "f32"]]}
    with pytest.raises(ValueError, match="native target datapaths"):
        catalog_source_conversion(manifest, "gemmini")

    module, row, manifest = prepared()
    row["target_dtypes"] = ["f32", "f32", "f32"]
    with pytest.raises(ValueError, match="unbound target format"):
        rewrite_contractions_to_device(module, "gemmini", catalog_selection={"matmul_0": types},
                                       catalog_bindings={"matmul_0": row}, catalog_manifest_data=manifest)

    nonzero = source.replace("constant 0.0", "constant 1.0")
    module = mq.parse(nonzero)
    op = next(mq.walk(module, "linalg.matmul"))
    op.attributes["prov.region_id"] = StringAttr("matmul_0")
    with pytest.raises(ValueError, match="absent or ambiguous"):
        rewrite_contractions_to_device(module, "gemmini", catalog_selection={"matmul_0": types},
                                       catalog_manifest_data={"abi": {"dtypes": ["f32", "f32", "f32"]},
                                                              "bindings": [prepared()[1]],
                                                              "source_conversion": conversion})


@pytest.mark.skipif(_CC is None, reason="no C compiler")
def test_dense_rank2_and_whole_batch_abi_runs_numerically(tmp_path):
    unit = emit_dense_translation_unit(
        "demo",
        {"two": (3, 7, 5), "three": (2, 3, 7, 5)},
        {"two": ("i8", "i8", "i32"), "three": ("i8", "i8", "i32")},
        kernel_symbol_for={"two": "kernel_two", "three": "kernel_three"}.__getitem__,
    )
    assert set(unit.symbols) == {"two", "three"}
    (tmp_path / "shim.c").write_text(unit.text)
    (tmp_path / "driver.c").write_text("""
#include <stdint.h>
typedef struct { void *alloc, *aligned; intptr_t off, size[2], stride[2]; } mr2;
typedef struct { void *alloc, *aligned; intptr_t off, size[3], stride[3]; } mr3;
extern mr2 two(void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
               void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
               void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t);
extern mr3 three(void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                 void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                 void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t);
static void gemm(const int8_t *a, const int8_t *b, int32_t *c, int batches) {
  for (int h=0; h<batches; ++h)
    for (int i=0; i<3; ++i) for (int j=0; j<7; ++j) {
      int32_t sum=0;
      for (int p=0; p<5; ++p) sum += a[h*15+i*5+p]*b[h*35+p*7+j];
      c[h*21+i*7+j]=sum;
    }
}
void kernel_two(void *a, void *b, void *c) { gemm(a,b,c,1); }
void kernel_three(void *a, void *b, void *c) { gemm(a,b,c,2); }
int main(void) {
  int8_t a[31]={0}, b[71]={0}; int32_t c[43]={0};
  for (int i=0;i<30;++i) a[i+1]=(i%9)-4;
  for (int i=0;i<70;++i) b[i+1]=(i%7)-3;
  mr2 r2=two(a,a,1,3,5,5,1,b,b,1,5,7,7,1,c,c,1,3,7,7,1);
  if (r2.aligned!=c || r2.off!=1 || r2.size[1]!=7 || r2.stride[0]!=7) return 1;
  for (int i=0;i<3;++i) for (int j=0;j<7;++j) {
    int32_t sum=0; for (int p=0;p<5;++p) sum+=a[1+i*5+p]*b[1+p*7+j];
    if (c[1+i*7+j]!=sum) return 2;
  }
  mr2 bad=two(a,a,1,4,5,5,1,b,b,1,5,7,7,1,c,c,1,3,7,7,1);
  if (bad.aligned || bad.size[0]) return 3;
  mr3 r3=three(a,a,1,2,3,5,15,5,1,b,b,1,2,5,7,35,7,1,
               c,c,1,2,3,7,21,7,1);
  if (r3.aligned!=c || r3.size[0]!=2 || r3.stride[0]!=21) return 4;
  for (int h=0;h<2;++h) for (int i=0;i<3;++i) for (int j=0;j<7;++j) {
    int32_t sum=0; for (int p=0;p<5;++p) sum+=a[1+h*15+i*5+p]*b[1+h*35+p*7+j];
    if (c[1+h*21+i*7+j]!=sum) return 5;
  }
  return 0;
}
""")
    exe = tmp_path / "driver"
    build = subprocess.run(
        [_CC, "-Wall", "-Wextra", "-Werror", str(tmp_path / "shim.c"), str(tmp_path / "driver.c"), "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    assert subprocess.run([exe], capture_output=True).returncode == 0


@pytest.mark.skipif(_CC is None, reason="no C compiler")
def test_catalog_build_rejects_source_object_and_operation_drift(tmp_path, monkeypatch):
    from merlin.llvmlower import device_catalog

    monkeypatch.setattr(device_catalog, "clang", lambda: _CC)
    src = tmp_path / "kernel.c"
    src.write_text("void selected_kernel(void *a, void *b, void *c) {(void)a;(void)b;(void)c;}\n")
    obj = tmp_path / "kernel.o"
    subprocess.run([_CC, "-c", str(src), "-o", str(obj)], check=True)
    digest = hashlib.sha256(obj.read_bytes()).hexdigest()
    manifest = {
        "abi": {
            "argument_order": ["lhs", "rhs", "out"],
            "pointee_layout": "dense_row_major",
            "batch_call": "whole_batch",
            "dtypes": ["i8", "i8", "i32"],
        },
        "source_sha256": "prepared-model-sha",
        "coverage_complete": True,
        "matched_contractions": 1,
        "covered_contractions": 1,
        "unique_kernels": 1,
        "kernels": [
            {"symbol": "selected_kernel", "batched": False, "dimensions": {"batch": 1, "m": 3, "n": 7, "k": 5}}
        ],
        "bindings": [
            {
                "source_operation_ordinal": 12,
                "symbol": "selected_kernel",
                "region": "matmul_0",
                "tensor_types": ["tensor<3x5xi8>", "tensor<5x7xi8>", "tensor<3x7xi32>"],
            }
        ],
        "compilation": {"object_sha256": digest},
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(manifest))
    args = dict(
        device="demo",
        signatures={"entry": (3, 7, 5)},
        dtypes={"entry": ("i8", "i8", "i32")},
        manifest_path=path,
        object_path=obj,
        source_sha256="prepared-model-sha",
        routed=[
            {
                "source_operation_ordinal": 12,
                "symbol": "entry",
                "source_region": "matmul_0",
                "tensor_types": ["tensor<3x5xi8>", "tensor<5x7xi8>", "tensor<3x7xi32>"],
                "dtypes": ["i8", "i8", "i32"],
            }
        ],
        workdir=tmp_path / "build",
        codegen_target="x86",
        cflags=["-O2"],
    )
    built = build_catalog_objects(**args)
    assert built.ok and built.kernels == {"entry": "selected_kernel"}
    receipt = json.loads((tmp_path / "build/catalog_binding.json").read_text())
    assert receipt["source_sha256"] == "prepared-model-sha"
    assert receipt["catalog_object_sha256"] == digest
    assert receipt["entry_to_kernel"] == {"entry": "selected_kernel"}
    with pytest.raises(ValueError, match="different prepared model"):
        build_catalog_objects(**{**args, "source_sha256": "wrong"})
    with pytest.raises(ValueError, match="source catalog binding"):
        build_catalog_objects(**{**args, "routed": [{**args["routed"][0], "source_region": "matmul_1"}]})
    obj.write_bytes(obj.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="object hash"):
        build_catalog_objects(**args)


def test_equal_shapes_share_only_an_identical_source_bound_implementation(monkeypatch):
    from xdsl.dialects.builtin import StringAttr

    from merlin.common import mlir_query as mq
    from merlin.llvmlower.device_offload import rewrite_contractions_to_device
    from merlin.system import offload

    monkeypatch.setattr(offload, "device_dtype_triples", lambda _: (("i8", "i8", "i32"),))
    source = """module {
      func.func @forward(%a:tensor<3x5xi8>,%b:tensor<5x7xi8>) -> tensor<3x7xi32> {
        %z = arith.constant 0 : i32
        %e = tensor.empty() : tensor<3x7xi32>
        %f = linalg.fill ins(%z:i32) outs(%e:tensor<3x7xi32>) -> tensor<3x7xi32>
        %r0 = linalg.matmul ins(%a,%b:tensor<3x5xi8>,tensor<5x7xi8>) outs(%f:tensor<3x7xi32>) -> tensor<3x7xi32>
        %r1 = linalg.matmul ins(%a,%b:tensor<3x5xi8>,tensor<5x7xi8>) outs(%f:tensor<3x7xi32>) -> tensor<3x7xi32>
        %r2 = linalg.matmul ins(%a,%b:tensor<3x5xi8>,tensor<5x7xi8>) outs(%f:tensor<3x7xi32>) -> tensor<3x7xi32>
        return %r2:tensor<3x7xi32>
      }
    }"""

    def fixture():
        module = mq.parse(source)
        ordinals = {op: i for i, op in enumerate(module.walk())}
        bindings = {}
        types = ("tensor<3x5xi8>", "tensor<5x7xi8>", "tensor<3x7xi32>")
        for index, op in enumerate(mq.walk(module, "linalg.matmul")):
            region = f"contraction_{index}"
            op.attributes["prov.region_id"] = StringAttr(region)
            bindings[region] = dict(
                region=region,
                source_operation_ordinal=ordinals[op],
                tensor_types=types,
                symbol="selected" if index == 0 else "default",
            )
        return module, bindings

    module, bindings = fixture()
    selection = {region: row["tensor_types"] for region, row in bindings.items()}
    result = rewrite_contractions_to_device(
        module, "fixture", catalog_selection=selection, catalog_bindings=bindings, model_sha256="source-sha"
    )
    assert result.moved == 3 and len(result.signatures) == 2
    assert result.routed[0].symbol != result.routed[1].symbol
    assert result.routed[1].symbol == result.routed[2].symbol
    assert all(shape == (3, 7, 5) for shape in result.signatures.values())

    module, bindings = fixture()
    bindings["contraction_0"]["source_operation_ordinal"] += 1
    with pytest.raises(ValueError, match="source operation ordinal"):
        rewrite_contractions_to_device(module, "fixture", catalog_selection=selection, catalog_bindings=bindings)
    assert len(list(mq.walk(module, "linalg.matmul"))) == 3

    # Ordinary equal implementations keep the previous declaration/call bytes.
    legacy, _ = fixture()
    rewrite_contractions_to_device(legacy, "fixture", catalog_selection=selection)
    current, bindings = fixture()
    for binding in bindings.values():
        binding["symbol"] = "default"
    ordinary = rewrite_contractions_to_device(
        current, "fixture", catalog_selection=selection, catalog_bindings=bindings
    )
    assert len(ordinary.signatures) == 1 and str(current) == str(legacy)


@pytest.mark.skipif(_CC is None, reason="no C compiler")
def test_catalog_dispatches_distinct_implementations_for_equal_shapes(tmp_path, monkeypatch):
    from merlin.llvmlower import device_catalog

    monkeypatch.setattr(device_catalog, "clang", lambda: _CC)
    source = tmp_path / "kernels.c"
    source.write_text(
        "void selected(void*a,void*b,void*c){(void)a;(void)b;(void)c;}\n"
        "void normal(void*a,void*b,void*c){(void)a;(void)b;(void)c;}\n"
    )
    obj = tmp_path / "kernels.o"
    subprocess.run([_CC, "-c", str(source), "-o", str(obj)], check=True)
    types = ["tensor<3x5xi8>", "tensor<5x7xi8>", "tensor<3x7xi32>"]
    bindings = [
        dict(region=f"contraction_{i}", symbol=kernel, source_operation_ordinal=12 + i, tensor_types=types)
        for i, kernel in enumerate(("selected", "normal", "normal"))
    ]
    manifest = dict(
        abi=dict(
            argument_order=["lhs", "rhs", "out"],
            pointee_layout="dense_row_major",
            batch_call="whole_batch",
            dtypes=["i8", "i8", "i32"],
        ),
        source_sha256="source-sha",
        coverage_complete=True,
        matched_contractions=3,
        covered_contractions=3,
        unique_kernels=2,
        bindings=bindings,
        kernels=[
            dict(symbol=kernel, batched=False, dimensions=dict(batch=1, m=3, n=7, k=5))
            for kernel in ("selected", "normal")
        ],
        compilation=dict(object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest()),
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(manifest))
    routed = [
        dict(
            symbol="entry_selected" if i == 0 else "entry_normal",
            source_region=row["region"],
            source_operation_ordinal=row["source_operation_ordinal"],
            tensor_types=types,
            dtypes=["i8", "i8", "i32"],
        )
        for i, row in enumerate(bindings)
    ]
    args = dict(
        device="fixture",
        signatures={"entry_selected": (3, 7, 5), "entry_normal": (3, 7, 5)},
        dtypes={entry: ("i8", "i8", "i32") for entry in ("entry_selected", "entry_normal")},
        manifest_path=path,
        object_path=obj,
        source_sha256="source-sha",
        routed=routed,
        workdir=tmp_path / "build",
        codegen_target="x86",
        cflags=["-O2"],
    )
    result = build_catalog_objects(**args)
    assert result.ok and result.kernels == {"entry_selected": "selected", "entry_normal": "normal"}
    with pytest.raises(ValueError, match="different source-bound implementations"):
        build_catalog_objects(**{**args, "routed": [{**routed[0], "symbol": "entry_normal"}, *routed[1:]]})
    with pytest.raises(ValueError, match="source catalog binding"):
        build_catalog_objects(**{**args, "routed": [{**routed[0], "source_operation_ordinal": 99}, *routed[1:]]})
    with pytest.raises(ValueError, match="source catalog binding"):
        build_catalog_objects(**{**args, "signatures": {**args["signatures"], "entry_selected": (3, 7, 6)}})
