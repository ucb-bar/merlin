"""Actual imported provider compilation, typed linkage and immutable closures."""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from merlin.common.digest import sha256_file
from merlin.runtime.host_provider import (
    HostProviderBuild,
    HostProviderContext,
    HostProviderObject,
    close_host_provider,
    prepare_host_provider,
    read_host_function_abi,
    revalidate_host_provider,
)


def _tools():
    from merlin.llvmlower import toolchain

    clang = toolchain.clang()
    nm = shutil.which("llvm-nm") or shutil.which("nm")
    if not clang.is_file() or nm is None:
        pytest.skip("compiler and symbol inspector unavailable")
    return clang, nm


def _compile(clang, source, directory, stem, optimization="-O2", extra=()):
    llvm, obj = directory / (stem + ".ll"), directory / (stem + ".o")
    commands = []
    for mode, output in [(("-S", "-emit-llvm"), llvm), (("-c",), obj)]:
        argv = tuple(map(str, [clang, optimization, "-fPIC", *extra, *mode, source, "-o", output]))
        subprocess.run(argv, check=True, capture_output=True)
        commands.append(argv)
    return llvm, obj, tuple(commands)


def _fixture(directory, optimization="-O2", provider_text=None):
    clang, nm = _tools()
    prepared = directory / "model.mlir"
    prepared.write_text("module { /* explicit fixture source identity */ }\n")
    model = directory / "model.c"
    model.write_text("""extern void provider(const float*, float*);
void source_fallback(const float *in, float *out) {
  for (int i=0; i<5; ++i) out[i] = in[i] + 2.0f;
}
void entry(const float *in, float *out) { provider(in, out); }
""")
    source = directory / "provider.c"
    source.write_text(
        provider_text
        or """extern void source_fallback(const float*, float*);
void provider(const float *in, float *out) { source_fallback(in, out); }
"""
    )
    model_llvm, model_obj, _ = _compile(clang, model, directory, "model", optimization)
    llvm, obj, commands = _compile(clang, source, directory, "provider", optimization)
    pins = tuple((path, sha256_file(path)) for path in [source, clang.resolve()])
    artifact = HostProviderObject(
        obj,
        sha256_file(obj),
        llvm,
        sha256_file(llvm),
        (read_host_function_abi(llvm, "provider"),),
        (read_host_function_abi(llvm, "source_fallback"),),
        pins,
        commands,
        directory,
    )
    context = HostProviderContext(
        prepared,
        model_llvm,
        model_obj,
        directory / "host_provider",
        clang,
        (optimization, "-fPIC"),
        lambda *args, **kwargs: None,
    )
    build = HostProviderBuild(
        sha256_file(prepared), sha256_file(model_llvm), sha256_file(model_obj), (artifact,), "1" * 64, "2" * 64
    )
    return context, build, nm


def _link(context, paths, output):
    subprocess.run(
        [
            str(context.compiler),
            "-shared",
            "-nostdlib",
            str(context.model_object_path),
            *map(str, paths),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def test_default_hook_is_inert(tmp_path):
    context = HostProviderContext(
        tmp_path / "absent",
        tmp_path / "absent.ll",
        tmp_path / "absent.o",
        tmp_path / "no_directory",
        tmp_path / "absent_cc",
        (),
        None,
    )
    assert prepare_host_provider(None, context, inspector=None) == ((), None)
    close_host_provider(None, (), tmp_path / "absent.elf", inspector=None)
    revalidate_host_provider(None)
    assert not context.workdir.exists()


@pytest.mark.parametrize("mutation", ["prepared", "dependency", "object", "llvm", "executable", "receipt", "symlink"])
def test_completed_provider_custody_expires_after_real_file_changes(tmp_path, mutation):
    context, build, nm = _fixture(tmp_path)
    paths, receipt = prepare_host_provider(lambda ctx: build, context, inspector=nm)
    with pytest.raises(ValueError, match="completed link"):
        revalidate_host_provider(receipt)
    executable = tmp_path / "linked.so"
    _link(context, paths, executable)
    close_host_provider(receipt, [context.model_object_path, *paths], executable, inspector=nm)
    revalidate_host_provider(receipt)
    if mutation == "receipt":
        record = json.loads(Path(receipt["receipt_path"]).read_text())
        record["numerical_witness_sha256"] = "3" * 64
        Path(receipt["receipt_path"]).write_text(json.dumps(record))
    else:
        path = {
            "prepared": context.prepared_path, "dependency": tmp_path / "provider.c",
            "object": paths[0], "llvm": build.objects[0].llvm_path,
            "executable": executable, "symlink": context.prepared_path,
        }[mutation]
        if mutation == "symlink":
            backup = path.with_suffix(".original")
            path.rename(backup)
            path.symlink_to(backup)
        else:
            path.write_bytes(path.read_bytes() + b"\nchanged after link\n")
    with pytest.raises(ValueError, match="changed|regular"):
        revalidate_host_provider(receipt)


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_actual_provider_calls_source_fallback_and_closes_link(tmp_path, optimization):
    context, build, nm = _fixture(tmp_path, optimization)
    paths, receipt = prepare_host_provider(lambda ctx: build, context, inspector=nm)
    executable = tmp_path / "linked.so"
    _link(context, paths, executable)
    close_host_provider(receipt, [context.model_object_path, *paths], executable, inspector=nm)
    revalidate_host_provider(receipt)
    library = ctypes.CDLL(str(executable))
    library.entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    original = np.array([-5, -0.75, -0.0, 1.25, 7], dtype=np.float32)
    values = original.copy()
    destination = np.full(7, np.float32(-947.25))
    library.entry(values.ctypes.data, destination[1:6].ctypes.data)
    np.testing.assert_array_equal(values.view(np.uint32), original.view(np.uint32))
    np.testing.assert_array_equal(destination[1:6].view(np.uint32), (original + np.float32(2)).view(np.uint32))
    assert destination[[0, 6]].tolist() == [-947.25, -947.25]
    record = json.loads(Path(receipt["receipt_path"]).read_text())
    assert record["status"] == "completed"
    assert record["resolved_required_symbols"] == {"source_fallback": "source_fallback"}
    assert record["executable"]["sha256"] == sha256_file(executable)
    assert [row["path"] for row in record["ordered_link_objects"]] == [
        str(context.model_object_path.resolve()),
        str(paths[0].resolve()),
    ]
    assert record["objects"][0]["compile_commands"] == [list(cmd) for cmd in build.objects[0].compile_commands]


@pytest.mark.parametrize(
    "mutation",
    ["source", "object", "llvm", "compiler_pin", "missing_commands", "bad_abi", "missing_required", "extra_export"],
)
def test_import_refusals_have_no_completed_receipt(tmp_path, mutation):
    context, build, nm = _fixture(tmp_path)
    artifact = build.objects[0]
    if mutation == "source":
        context.prepared_path.write_text("changed source\n")
    elif mutation == "object":
        artifact.path.write_bytes(artifact.path.read_bytes() + b"changed")
    elif mutation == "llvm":
        artifact.llvm_path.write_text(artifact.llvm_path.read_text() + "; changed\n")
    elif mutation == "compiler_pin":
        artifact = replace(artifact, compilation_pins=artifact.compilation_pins[:1])
    elif mutation == "missing_commands":
        artifact = replace(artifact, compile_commands=artifact.compile_commands[:1])
    elif mutation == "bad_abi":
        artifact = replace(artifact, required=(replace(artifact.required[0], arguments=("ptr",)),))
    elif mutation == "missing_required":
        artifact = replace(artifact, required=())
    elif mutation == "extra_export":
        source = tmp_path / "provider.c"
        source.write_text(source.read_text() + "void extra(void) {}\n")
        llvm, obj, commands = _compile(context.compiler, source, tmp_path, "provider")
        artifact = replace(
            artifact,
            sha256=sha256_file(obj),
            llvm_sha256=sha256_file(llvm),
            compile_commands=commands,
            compilation_pins=tuple((p, sha256_file(p)) for p in [source, context.compiler.resolve()]),
        )
    build = replace(build, objects=(artifact,))
    with pytest.raises(ValueError):
        prepare_host_provider(lambda ctx: build, context, inspector=nm)
    assert not (context.workdir / "host_provider.json").exists()


@pytest.mark.parametrize("mutation", ["source", "dependency", "missing_object", "duplicate_object"])
def test_final_closure_rechecks_imports_and_order(tmp_path, mutation):
    context, build, nm = _fixture(tmp_path)
    paths, receipt = prepare_host_provider(lambda ctx: build, context, inspector=nm)
    executable = tmp_path / "linked.so"
    _link(context, paths, executable)
    objects = [context.model_object_path, *paths]
    if mutation == "source":
        context.prepared_path.write_text("changed source\n")
    elif mutation == "dependency":
        (tmp_path / "provider.c").write_text("changed dependency\n")
    elif mutation == "missing_object":
        objects = objects[:1]
    else:
        objects += list(paths)
    with pytest.raises(ValueError):
        close_host_provider(receipt, objects, executable, inspector=nm)
    assert json.loads(Path(receipt["receipt_path"]).read_text())["status"] == "prepared"


@pytest.mark.parametrize(
    "header",
    [
        "declare fastcc void @boundary(ptr)",
        "declare void @boundary(ptr byval(i32))",
        "declare {i32,i32} @boundary(ptr)",
        "declare void @boundary(ptr addrspace(1))",
    ],
)
def test_unsupported_abi_refuses(tmp_path, header):
    llvm = tmp_path / "bad.ll"
    llvm.write_text(header + "\n")
    with pytest.raises(ValueError, match="absent or unsupported"):
        read_host_function_abi(llvm, "boundary")


def test_inspector_absence_refuses_before_builder(tmp_path):
    called = []
    context = HostProviderContext(
        tmp_path / "none", tmp_path / "none", tmp_path / "none", tmp_path / "unused", tmp_path / "none", (), None
    )
    with pytest.raises(ValueError, match="inspector"):
        prepare_host_provider(lambda ctx: called.append(ctx), context, inspector=None)
    assert called == [] and not context.workdir.exists()


def test_normal_upstream_model_links_ranked_provider_and_runs_stock_isa(tmp_path):
    from xdsl.dialects import arith, builtin, memref, scf, tensor
    from xdsl.parser import Parser

    from merlin.llvmlower import toolchain
    from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
    from merlin.runtime.backends import spike, spike_model
    from merlin.xdsl_dialects._common import make_context, text

    if not spike.available() or not toolchain.m2m_python().is_file():
        pytest.skip("upstream and target toolchain unavailable")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    module = Parser(
        make_context(tensor.Tensor, arith.Arith, memref.MemRef, scf.Scf),
        """module {
      func.func @forward(%a:tensor<5xf32>)->tensor<5xf32> attributes {llvm.emit_c_interface} {
        %out=tensor.empty():tensor<5xf32>
        %r=func.call @writer(%a,%out):(tensor<5xf32>,tensor<5xf32>)->tensor<5xf32>
        return %r:tensor<5xf32>
      }
      func.func @source_fallback(%a:memref<5xf32>,%out:memref<5xf32>) attributes {llvm.emit_c_interface} {
        %zero=arith.constant 0:index
        %one=arith.constant 1:index
        %five=arith.constant 5:index
        %two=arith.constant 2.0:f32
        scf.for %i=%zero to %five step %one {
          %v=memref.load %a[%i]:memref<5xf32>
          %sum=arith.addf %v,%two:f32
          memref.store %sum,%out[%i]:memref<5xf32>
        }
        return
      }
      func.func private @writer(tensor<5xf32>,tensor<5xf32>)->tensor<5xf32> attributes {llvm.emit_c_interface}
    }""",
    ).parse_module()
    module.body.block.last_op.properties["arg_attrs"] = builtin.ArrayAttr(
        [builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr(access)}) for access in ("read", "write")]
    )
    rewrite_fresh_tensor_writers(module, [FreshTensorWriterContract("writer", 1, (1,), "borrowed", 32)])
    (bundle / "model.mlir").write_text(text(module, generic=True))
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    values = np.array([-5, -0.75, -0.0, 1.25, 7], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=values)

    def builder(context):
        source = context.workdir / "provider.c"
        source.write_text("""#include <stdint.h>
typedef struct {void *allocated,*aligned;int64_t offset,size[1],stride[1];} M;
void _mlir_ciface_source_fallback(M*,M*);
void _mlir_ciface_borrowed(M*a,M*out){_mlir_ciface_source_fallback(a,out);}
""")
        llvm = context.workdir / "provider.ll"
        obj = context.workdir / "provider.o"
        commands = []
        for mode, output in [(("-S", "-emit-llvm"), llvm), (("-c",), obj)]:
            argv = tuple(map(str, [context.compiler, *context.compiler_flags, *mode, source, "-o", output]))
            context.compile(argv, inputs=[source, context.compiler.resolve()], output=output)
            commands.append(argv)
        artifact = HostProviderObject(
            obj,
            sha256_file(obj),
            llvm,
            sha256_file(llvm),
            (read_host_function_abi(llvm, "_mlir_ciface_borrowed"),),
            (read_host_function_abi(llvm, "_mlir_ciface_source_fallback"),),
            tuple((p, sha256_file(p)) for p in (source, context.compiler.resolve())),
            tuple(commands),
            Path.cwd(),
        )
        return HostProviderBuild(
            sha256_file(context.prepared_path),
            sha256_file(context.model_llvm_path),
            sha256_file(context.model_object_path),
            (artifact,),
            "3" * 64,
            "4" * 64,
        )

    flags = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]
    built = spike_model.build(
        bundle, tmp_path / "build", arena_mb=1, cflags_override=flags, host_provider_builder=builder
    )
    record = built["host_provider"]
    assert record["status"] == "completed"
    assert record["resolved_required_symbols"] == {"_mlir_ciface_source_fallback": "_mlir_ciface_source_fallback"}
    compilation = json.loads((tmp_path / "build/compilation_recipe.json").read_text())
    assert any(Path(row["output"]["path"]).name == "provider.ll" for row in compilation["commands"])
    assert any(Path(row["output"]["path"]).name == "provider.o" for row in compilation["commands"])
    assert compilation["commands"][-1]["output"]["sha256"] == sha256_file(built["elf"])
    result = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
    np.testing.assert_array_equal(result["outputs"].view(np.uint32), (values + np.float32(2)).view(np.uint32))
    assert result["metrics"]["memref_rank_mismatch"] == 0
    assert "DONE" in result["console"]
    # The recorded final command reproduces the exact qualified executable.
    command = compilation["commands"][-1]
    subprocess.run(command["argv"], cwd=command["cwd"], check=True, capture_output=True)
    assert sha256_file(built["elf"]) == record["executable"]["sha256"]


def test_compiler_integer_return_range_preserves_machine_abi(tmp_path):
    clang, _ = _tools()
    source = tmp_path / "range.c"
    source.write_text("int provider(const void *p) { return p != 0; }\n")
    llvm, _, _ = _compile(
        clang, source, tmp_path, "range", extra=("--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d")
    )
    abi = read_host_function_abi(llvm, "provider")
    assert abi.result == "signext i32"
    assert abi.arguments == ("ptr",)


@pytest.mark.parametrize(
    "attribute",
    [
        "range(i32 0, 2)",
        "range(i64 -1, 2)",
        "range(i32 0, 2) range(i32 0, 2)",
    ],
)
def test_return_range_structure_refuses_wrong_type_or_duplicate(tmp_path, attribute):
    llvm = tmp_path / "abi.ll"
    llvm.write_text(f"declare {attribute} i64 @provider(ptr)\n")
    if attribute == "range(i64 -1, 2)":
        assert read_host_function_abi(llvm, "provider").result == "i64"
    else:
        with pytest.raises(ValueError, match="absent or unsupported"):
            read_host_function_abi(llvm, "provider")
