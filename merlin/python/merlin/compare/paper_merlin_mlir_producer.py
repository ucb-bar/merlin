"""Closed v4 producer for genuine Merlin-lowered whole-session objects.

The public closure contains MLIR, the semantic MRLNSES2 descriptor, and the
generic Merlin runtime only.  Buffer ABI is derived from parsed MLIR function
types.  Session values enter exclusively through the common session ABI after
the independently replayed build barrier has been verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root, runtime_dir
from merlin.llvmlower.codegen import RISCV_FLAGS
from merlin.llvmlower.toolchain import clang, m2m_python

from .paper_build_bundle import (
    MultiToolchainAuthority,
    PublicBuildBundle,
    TargetABI,
    VerifiedBuildBarrier,
    _elf_identity,
    _elf_matches_target,
    issue_verified_build_barrier,
    load_multi_toolchain_authority,
    snapshot_public_build_bundle,
    verify_public_build_bundle,
    write_public_resource_roles,
)
from .paper_session_abi import (
    InputEndpoint,
    InputFrame,
    SessionDescriptor,
    decode_response,
    descriptor_from_contract,
    descriptor_from_dict,
    encode_request,
)
from .paper_session_tracer import ENTRYPOINT, render_runner_source

RECIPE_ID = "merlin_mlir_model_object_v1"
PRODUCER_SCHEMA = "merlin.paper.merlin-mlir-public/v1"
RECEIPT_SCHEMA = "merlin.paper.merlin-mlir-build/v1"
LOWERING_TOOL_ROLE = "mlir_lowering_python"

_DTYPE_BYTES = {
    "f64": 8,
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "i64": 8,
    "i32": 4,
    "i16": 2,
    "i8": 1,
    "i1": 1,
}

K1_TARGET_ABI = TargetABI(
    "k1_rvv_lp64d", "riscv64-unknown-elf", "rv64gcv", "lp64d", ("c", "g", "v"), 64, 243, 0, 0xFFFFFFFF, 0x5
)


@dataclass(frozen=True)
class TensorABI:
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class ProgramABI:
    id: int
    name: str
    entrypoint: str
    resource: str
    inputs: tuple[TensorABI, ...]
    outputs: tuple[TensorABI, ...]


@dataclass(frozen=True)
class MerlinMLIRBuild:
    receipt: Path
    composite_object: Path
    runner: Path
    descriptor: SessionDescriptor


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_identity() -> dict[str, str]:
    root = repo_root().resolve()
    paths = (
        Path(__file__).resolve(),
        root / "merlin/python/merlin/__init__.py",
        root / "merlin/python/merlin/common/__init__.py",
        root / "merlin/python/merlin/common/ir_lock.py",
        root / "merlin/python/merlin/common/mlir_query.py",
        root / "merlin/python/merlin/frontends/__init__.py",
        root / "merlin/python/merlin/frontends/linalg_mlir.py",
        root / "merlin/python/merlin/llvmlower/codegen.py",
        root / "merlin/python/merlin/llvmlower/lower.py",
        root / "merlin/python/merlin/llvmlower/passes_xdsl.py",
        root / "merlin/python/merlin/llvmlower/pipeline.py",
        root / "merlin/python/merlin/llvmlower/session_bundle.py",
        root / "merlin/python/merlin/compare/paper_merlin_lower_worker.py",
        root / "merlin/python/merlin/compare/paper_build_bundle.py",
        root / "merlin/python/merlin/compare/paper_session_abi.py",
        root / "merlin/python/merlin/compare/paper_session_tracer.py",
    )
    return {path.relative_to(root).as_posix(): _sha(path) for path in paths}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _load_json(path: Path, where: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{where} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    return value


def _closed(value: Mapping[str, Any], keys: set[str], where: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{where} differs from its closed schema")


def synthetic_merlin_mlir_descriptor() -> SessionDescriptor:
    root = {
        "version": 2,
        "kind": "synthetic_merlin_recurrent",
        "paper_ready": True,
        "stages": ["prefill", "decode"],
        "stage_schedule": [
            {"name": "prefill", "steps": 1, "execution": "compiled", "timed": True},
            {"name": "decode", "steps": 3, "execution": "compiled_recurrent", "timed": True},
        ],
        "programs": [
            {"name": "prefill", "bundle": "programs/0", "steps": 1},
            {"name": "decode", "bundle": "programs/1", "steps": 3},
        ],
        "bindings": [
            {
                "name": "state_seed",
                "from": {"program": "prefill", "output_index": 0},
                "to": {"program": "decode", "input_arg": 1},
            }
        ],
        "states": ["state"],
        "streams": [],
        "quality": {"scope": "trajectory", "program": "decode"},
    }

    def child(name: str, steps: int, *, state: bool) -> dict[str, Any]:
        return {
            "version": 1,
            "kind": "synthetic_merlin_recurrent",
            "paper_ready": True,
            "stages": [name],
            "steps": steps,
            "stage_schedule": [
                {
                    "name": name,
                    "steps": steps,
                    "execution": "compiled_recurrent" if state else "compiled",
                    "timed": True,
                }
            ],
            "streams": [{"name": "value", "input_arg": 0, "key": "value"}],
            "states": ([{"name": "state", "input_arg": 1, "output_index": 0}] if state else []),
            "quality": {"scope": "trajectory", "output_index": 0},
        }

    return descriptor_from_contract(
        root,
        child_contracts={
            "prefill": child("prefill", 1, state=False),
            "decode": child("decode", 3, state=True),
        },
    )


def _prefill_mlir(constant: int) -> str:
    return f"""builtin.module {{
  func.func @forward(%seed: tensor<1xi64>) -> tensor<1xi64> {{
    %empty = tensor.empty() : tensor<1xi64>
    %out = linalg.generic {{
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]
    }} ins(%seed : tensor<1xi64>) outs(%empty : tensor<1xi64>) {{
    ^bb0(%s: i64, %old: i64):
      %c = arith.constant {constant} : i64
      %v = arith.addi %s, %c : i64
      linalg.yield %v : i64
    }} -> tensor<1xi64>
    func.return %out : tensor<1xi64>
  }}
}}
"""


def _decode_mlir() -> str:
    return """builtin.module {
  func.func @forward(%delta: tensor<1xi64>, %state: tensor<1xi64>)
      -> tensor<1xi64> {
    %empty0 = tensor.empty() : tensor<1xi64>
    %out0 = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                       affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]
    } ins(%delta, %state : tensor<1xi64>, tensor<1xi64>)
      outs(%empty0 : tensor<1xi64>) {
    ^bb0(%d: i64, %s: i64, %old: i64):
      %three = arith.constant 3 : i64
      %scaled = arith.muli %s, %three : i64
      %v = arith.addi %scaled, %d : i64
      linalg.yield %v : i64
    } -> tensor<1xi64>
    func.return %out0 : tensor<1xi64>
  }
}
"""


def materialize_synthetic_merlin_mlir_public_closure(root: str | Path, *, prefill_constant: int = 2) -> Path:
    """Write public MLIR/runtime resources; never session frames or references."""
    root = Path(root).resolve()
    descriptor = synthetic_merlin_mlir_descriptor()
    for directory in ("descriptor", "programs/0", "programs/1", "runtime", "sources", "lib"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "descriptor/session_descriptor.json").write_bytes(descriptor.canonical_bytes + b"\n")
    (root / "programs/0/model.mlir").write_text(_prefill_mlir(prefill_constant), encoding="utf-8")
    (root / "programs/1/model.mlir").write_text(_decode_mlir(), encoding="utf-8")
    shutil.copyfile(runtime_dir() / "c/merlin_model.c", root / "runtime/merlin_model.c")
    shutil.copyfile(runtime_dir() / "c/merlin_model.h", root / "runtime/merlin_model.h")
    (root / "sources/runner.c").write_text(render_runner_source(), encoding="utf-8")
    (root / "lib/libpublic_anchor.a").write_bytes(b"!<arch>\n")
    programs = [
        {
            "id": index,
            "name": name,
            "resource": f"programs/{index}/model.mlir",
            "entrypoint": f"merlin_paper_program_{index}",
        }
        for index, name in enumerate(("prefill", "decode"))
    ]
    _write_json(
        root / "producer_manifest.json",
        {
            "schema": PRODUCER_SCHEMA,
            "recipe": RECIPE_ID,
            "programs": programs,
            "runtime": {"source_resource": "runtime/merlin_model.c", "header_resource": "runtime/merlin_model.h"},
            "runner_resource": "sources/runner.c",
        },
    )
    write_public_resource_roles(
        root,
        {
            "descriptor/session_descriptor.json": "session_descriptor",
            "lib/libpublic_anchor.a": "static_library",
            "producer_manifest.json": "producer_manifest",
            "programs/0/model.mlir": "mlir",
            "programs/1/model.mlir": "mlir",
            "runtime/merlin_model.c": "c_source",
            "runtime/merlin_model.h": "header",
            "sources/runner.c": "c_source",
        },
    )
    return root


def _tensor_abi(raw: tuple[list[int], str], where: str) -> TensorABI:
    shape, dtype = raw
    width = _DTYPE_BYTES.get(dtype)
    if width is None or not shape or len(shape) > 8 or any(dim <= 0 for dim in shape):
        raise ValueError(
            f"{where}: existing Merlin session runtime lacks a static ranked tensor ABI "
            f"for shape={shape!r} dtype={dtype!r}"
        )
    elements = 1
    for dim in shape:
        elements *= dim
    return TensorABI(tuple(shape), dtype, elements * width)


def _load_recipe(
    public: PublicBuildBundle, authority: MultiToolchainAuthority
) -> tuple[SessionDescriptor, tuple[ProgramABI, ...]]:
    descriptor = descriptor_from_dict(
        _load_json(public.closure_root / "descriptor/session_descriptor.json", "session descriptor")
    )
    expected_resources = {
        "descriptor/session_descriptor.json",
        "lib/libpublic_anchor.a",
        "producer_manifest.json",
        "resource_roles.json",
        "runtime/merlin_model.c",
        "runtime/merlin_model.h",
        "sources/runner.c",
        *(f"programs/{program.id}/model.mlir" for program in descriptor.programs),
    }
    if {str(row["path"]) for row in public.files} != expected_resources:
        raise ValueError("Merlin producer public resources differ from its exact path-role graph")
    for relative in ("c/merlin_model.c", "c/merlin_model.h"):
        public_path = public.closure_root / "runtime" / Path(relative).name
        if public_path.read_bytes() != (runtime_dir() / relative).read_bytes():
            raise ValueError(f"Merlin producer runtime/{Path(relative).name} is not the bound runtime")
    if (public.closure_root / "sources/runner.c").read_text(encoding="utf-8") != render_runner_source():
        raise ValueError("Merlin producer runner differs from deterministic producer output")
    manifest = _load_json(public.closure_root / "producer_manifest.json", "Merlin producer manifest")
    _closed(manifest, {"schema", "recipe", "programs", "runtime", "runner_resource"}, "Merlin producer manifest")
    if manifest.get("schema") != PRODUCER_SCHEMA or manifest.get("recipe") != RECIPE_ID:
        raise ValueError("Merlin producer manifest schema or recipe differs")
    rows = manifest.get("programs")
    if not isinstance(rows, list) or len(rows) != len(descriptor.programs):
        raise ValueError("Merlin producer omits a compiled session program")
    result: list[ProgramABI] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("Merlin producer program row must be a mapping")
        _closed(raw, {"id", "name", "resource", "entrypoint"}, f"program {index}")
        expected = descriptor.programs[index]
        entrypoint = f"merlin_paper_program_{index}"
        if raw.get("id") != index or raw.get("name") != expected.name or raw.get("entrypoint") != entrypoint:
            raise ValueError("Merlin producer program order/identity differs from the session")
        resource = str(raw.get("resource", ""))
        expected_resource = f"programs/{index}/model.mlir"
        if resource != expected_resource:
            raise ValueError("Merlin producer program resource differs from the closed recipe")
        path = (public.closure_root / resource).resolve()
        if not path.is_relative_to(public.closure_root) or path.is_symlink() or not path.is_file():
            raise ValueError("Merlin producer program resource escapes or is absent")
        worker = repo_root() / "merlin/python/merlin/compare/paper_merlin_lower_worker.py"
        completed = _run_capture(
            [str(authority.tool(LOWERING_TOOL_ROLE).path), str(worker), "--signature", str(path)],
            f"program {index} signature extraction",
            authority,
        )
        try:
            signature = json.loads(completed.stdout.decode("ascii"))
            inputs, outputs = signature["inputs"], signature["outputs"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"program {index}: sealed signature worker output is malformed") from exc
        result.append(
            ProgramABI(
                index,
                expected.name,
                entrypoint,
                resource,
                tuple(_tensor_abi(value, f"program {index} input {at}") for at, value in enumerate(inputs)),
                tuple(_tensor_abi(value, f"program {index} output {at}") for at, value in enumerate(outputs)),
            )
        )
    _validate_session_route(descriptor, tuple(result))
    return descriptor, tuple(result)


def _validate_session_route(descriptor: SessionDescriptor, programs: tuple[ProgramABI, ...]) -> None:
    external = {(row.endpoint.program, row.endpoint.input) for row in descriptor.inputs}
    route_targets = {(row.target_program, row.target_input) for row in descriptor.routes}
    state_inputs = {(row.program, row.input) for row in descriptor.states}
    for program in programs:
        covered = {index for owner, index in external | route_targets | state_inputs if owner == program.id}
        if covered != set(range(len(program.inputs))):
            raise ValueError(
                f"program {program.id}: public MLIR inputs lack a complete stream/route/state ABI; "
                f"covered={sorted(covered)} inputs={len(program.inputs)}"
            )
    for row in descriptor.inputs:
        if row.endpoint.input >= len(programs[row.endpoint.program].inputs):
            raise ValueError("session input endpoint is outside the parsed MLIR signature")
    for route in descriptor.routes:
        source, target = programs[route.source_program], programs[route.target_program]
        if route.source_output >= len(source.outputs) or route.target_input >= len(target.inputs):
            raise ValueError("session route is outside the parsed MLIR signatures")
        if source.outputs[route.source_output] != target.inputs[route.target_input]:
            raise ValueError("existing Merlin runtime cannot route state with unequal tensor ABIs")
    for state in descriptor.states:
        program = programs[state.program]
        if state.input >= len(program.inputs) or state.output >= len(program.outputs):
            raise ValueError("carried state is outside the parsed MLIR signature")
        if program.inputs[state.input] != program.outputs[state.output]:
            raise ValueError("existing Merlin runtime cannot commit state with unequal tensor ABIs")
    output = descriptor.output
    if output.output >= len(programs[output.program].outputs):
        raise ValueError("quality output is outside the parsed MLIR signature")


def _dims(tensor: TensorABI) -> str:
    return "{" + ",".join([*(str(dim) for dim in tensor.shape), *("0" for _ in range(8 - len(tensor.shape)))]) + "}"


def _render_adapter(descriptor: SessionDescriptor, programs: tuple[ProgramABI, ...]) -> str:
    descriptor_bytes = ",".join(str(byte) for byte in descriptor.canonical_bytes)
    call_rows = ",".join(f"{{{call.program}U,{call.step}U}}" for call in descriptor.calls)
    external = {(row.endpoint.program, row.endpoint.input): row for row in descriptor.inputs}
    states_by_program = {
        program.id: [state for state in descriptor.states if state.program == program.id] for program in programs
    }
    lines = [
        f"""/* Generated only from public MLIR signatures and MRLNSES2. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "merlin_model.h"
typedef unsigned char u8; typedef unsigned int u32; typedef unsigned long long u64;
static const u8 MAGIC[8] = {{77,82,76,78,83,69,83,50}};
static const u8 DESCRIPTOR[] = {{{descriptor_bytes}}};
static const u32 CALLS[][2] = {{{call_rows}}};
static int same(const u8*a,const u8*b,size_t n){{size_t i;for(i=0;i<n;i++)if(a[i]!=b[i])return 0;return 1;}}
static int get32(const u8*p,size_t n,size_t*a,u32*v){{size_t i=*a;
if(i>n||n-i<4)return-1;*v=((u32)p[i]<<24)|((u32)p[i+1]<<16)|((u32)p[i+2]<<8)|p[i+3];
*a=i+4;return 0;}}
static int get64(const u8*p,size_t n,size_t*a,u64*v){{size_t i;u64 x=0;
if(*a>n||n-*a<8)return-1;for(i=0;i<8;i++)x=(x<<8)|p[*a+i];*a+=8;*v=x;return 0;}}
static int put32(u8*p,size_t n,size_t*a,u32 v){{size_t i=*a;if(i>n||n-i<4)return-1;
p[i]=(u8)(v>>24);p[i+1]=(u8)(v>>16);p[i+2]=(u8)(v>>8);p[i+3]=(u8)v;*a=i+4;return 0;}}
static int put64(u8*p,size_t n,size_t*a,u64 v){{size_t i;if(*a>n||n-*a<8)return-1;
for(i=0;i<8;i++)p[*a+7-i]=(u8)(v>>(i*8));*a+=8;return 0;}}
static int frame(const u8*r,size_t n,size_t*a,u32 p,u32 in,u32 step,u8*dst,u64 need){{
u32 gp,gi,gs;u64 bytes;
if(get32(r,n,a,&gp)||get32(r,n,a,&gi)||get32(r,n,a,&gs)||get64(r,n,a,&bytes))return-1;
if(gp!=p||gi!=in||gs!=step||bytes!=need||*a>n||n-*a<bytes)return-1;
memcpy(dst,r+*a,(size_t)bytes);*a+=(size_t)bytes;return 0;}}
"""
    ]
    for program in programs:
        for index, tensor in enumerate(program.inputs):
            row = external.get((program.id, index))
            if row is not None and row.role == "stream":
                lines.append(f"static u8 EXT_p{program.id}_i{index}[{row.frames}][{tensor.nbytes}];\n")
            else:
                lines.append(f"static u8 IN_p{program.id}_i{index}[{tensor.nbytes}];\n")
        for index, tensor in enumerate(program.outputs):
            lines.append(f"static u8 OUT_p{program.id}_o{index}[{tensor.nbytes}];\n")
        args = [f"{{MERLIN_INPUT,0,{len(t.shape)},{_dims(t)},{_DTYPE_BYTES[t.dtype]}}}" for t in program.inputs] + [
            f"{{MERLIN_OUTPUT,0,{len(t.shape)},{_dims(t)},{_DTYPE_BYTES[t.dtype]}}}" for t in program.outputs
        ]
        lines.append(
            f"static const merlin_arg_t ARGS_p{program.id}[]={{{','.join(args)}}};\n"
            f"static merlin_descriptor_t DESCS_p{program.id}[{len(args)}];\n"
            f"static void *INPUTS_p{program.id}[{len(args)}];\n"
            f"static void *OUTPUTS_p{program.id}[{len(program.outputs)}];\n"
        )
        decl = ",".join("void *" for _ in args)
        call = ",".join(f"d[{index}]" for index in range(len(args)))
        lines.append(
            f"extern void _mlir_ciface_{program.entrypoint}({decl});\n"
            f"static void invoke_p{program.id}(void **d){{_mlir_ciface_{program.entrypoint}({call});}}\n"
        )
        setup = []
        for index in range(len(program.inputs)):
            row = external.get((program.id, index))
            value = (
                f"EXT_p{program.id}_i{index}[step]"
                if row is not None and row.role == "stream"
                else f"IN_p{program.id}_i{index}"
            )
            setup.append(f"INPUTS_p{program.id}[{index}]={value};")
        for index in range(len(program.outputs)):
            setup.append(f"OUTPUTS_p{program.id}[{index}]=OUT_p{program.id}_o{index};")
        lines.append(
            f"static void run_p{program.id}(u32 step){{{''.join(setup)}"
            f"merlin_run_multi_with(ARGS_p{program.id},{len(args)},0,INPUTS_p{program.id},"
            f"OUTPUTS_p{program.id},DESCS_p{program.id},invoke_p{program.id});}}\n"
        )
    lines.append("void merlin_invoke(void **d){invoke_p0(d);}\n")
    quality = programs[descriptor.output.program].outputs[descriptor.output.output]
    lines.append(f"static u8 QUALITY[{descriptor.output.frames}][{quality.nbytes}];\n")
    parse_frames = []
    for row in descriptor.inputs:
        tensor = programs[row.endpoint.program].inputs[row.endpoint.input]
        for step in range(row.frames):
            destination = (
                f"EXT_p{row.endpoint.program}_i{row.endpoint.input}[{step}]"
                if row.role == "stream"
                else f"IN_p{row.endpoint.program}_i{row.endpoint.input}"
            )
            parse_frames.append(
                f"if(frame(request,request_size,&at,{row.endpoint.program}U,"
                f"{row.endpoint.input}U,{step}U,{destination},{tensor.nbytes}U))return 15;"
            )
    execution = []
    for call in descriptor.calls:
        execution.append(f"run_p{call.program}({call.step}U);")
        program = programs[call.program]
        if call.program == descriptor.output.program:
            execution.append(
                f"memcpy(QUALITY[{call.step}],OUT_p{call.program}_o{descriptor.output.output},{quality.nbytes}U);"
            )
        if call.step == descriptor.programs[call.program].steps - 1:
            for route in descriptor.routes:
                if route.source_program == call.program:
                    nbytes = program.outputs[route.source_output].nbytes
                    execution.append(
                        f"memcpy(IN_p{route.target_program}_i{route.target_input},"
                        f"OUT_p{route.source_program}_o{route.source_output},{nbytes}U);"
                    )
        states = states_by_program[call.program]
        if states:
            input_args = ",".join(str(state.input) for state in states)
            output_indices = ",".join(str(state.output) for state in states)
            execution.append(
                f"{{static const int si[]={{{input_args}}};static const int so[]={{{output_indices}}};"
                f"if(merlin_commit_state(ARGS_p{call.program},{len(program.inputs) + len(program.outputs)},"
                f"INPUTS_p{call.program},OUTPUTS_p{call.program},{len(states)},si,so))return 30;}}"
            )
    output_frames = []
    for step in range(descriptor.output.frames):
        output_frames.append(
            f"if(put32(response,response_capacity,&out,{descriptor.output.program}U)||"
            f"put32(response,response_capacity,&out,{descriptor.output.output}U)||"
            f"put32(response,response_capacity,&out,{step}U)||"
            f"put64(response,response_capacity,&out,{quality.nbytes}U)||"
            f"out>response_capacity||response_capacity-out<{quality.nbytes}U)return 23;"
            f"memcpy(response+out,QUALITY[{step}],{quality.nbytes}U);out+={quality.nbytes}U;"
        )
    lines.append(f"""
int {ENTRYPOINT}(const char *runtime_root,const u8 *request,size_t request_size,
                 u8 *response,size_t response_capacity,size_t *response_size){{
  size_t at=0,out=0,i;u32 descriptor_size,frame_count;(void)runtime_root;
  if(!request||!response||!response_size||request_size<13)return 10;
  if(!same(request,MAGIC,8)||request[8]!=1)return 11;at=9;
  if(get32(request,request_size,&at,&descriptor_size)||descriptor_size!=sizeof(DESCRIPTOR)||
     at>request_size||request_size-at<descriptor_size||
     !same(request+at,DESCRIPTOR,descriptor_size))return 12;at+=descriptor_size;
  if(get32(request,request_size,&at,&frame_count)||frame_count!={len(descriptor.required_input_keys)}U)return 13;
  {"".join(parse_frames)} if(at!=request_size)return 16;
  {"".join(execution)}
  if(response_capacity<13U+sizeof(DESCRIPTOR))return 20;
  for(i=0;i<8;i++)response[out++]=MAGIC[i];response[out++]=2;
  if(put32(response,response_capacity,&out,(u32)sizeof(DESCRIPTOR)))return 21;
  if(out>response_capacity||response_capacity-out<sizeof(DESCRIPTOR))return 21;
  memcpy(response+out,DESCRIPTOR,sizeof(DESCRIPTOR));out+=sizeof(DESCRIPTOR);
  if(put32(response,response_capacity,&out,{len(descriptor.calls)}U))return 22;
  for(i=0;i<{len(descriptor.calls)}U;i++)if(put32(response,response_capacity,&out,CALLS[i][0])||
     put32(response,response_capacity,&out,CALLS[i][1]))return 22;
  if(put32(response,response_capacity,&out,{descriptor.output.frames}U))return 23;
  {"".join(output_frames)} *response_size=out;return 0;
}}
""")
    return "".join(lines)


def _response_capacity(descriptor: SessionDescriptor, programs: tuple[ProgramABI, ...]) -> int:
    """Return the exact number of bytes emitted by the closed response ABI."""
    quality = programs[descriptor.output.program].outputs[descriptor.output.output]
    return (
        13
        + len(descriptor.canonical_bytes)
        + 4
        + 8 * len(descriptor.calls)
        + 4
        + descriptor.output.frames * (20 + quality.nbytes)
    )


def _producer_env(authority: MultiToolchainAuthority) -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "SOURCE_DATE_EPOCH": "0",
        "PATH": os.pathsep.join(sorted({str(tool.path.parent) for tool in authority.tools})),
        "PYTHONPATH": str(repo_root() / "merlin/python"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MERLIN_CLANG": str(authority.tool("c_compiler").path),
        "MERLIN_M2M_VENV": str(authority.tool(LOWERING_TOOL_ROLE).path.parent.parent),
    }


def _run_capture(
    argv: Sequence[str], where: str, authority: MultiToolchainAuthority
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        timeout=180,
        env=_producer_env(authority),
        cwd=authority.sysroot,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{where} failed ({completed.returncode}): {detail}")
    return completed


def _run(argv: Sequence[str], where: str, authority: MultiToolchainAuthority) -> None:
    _run_capture(argv, where, authority)


def _validate_authority(public: PublicBuildBundle, authority: MultiToolchainAuthority) -> None:
    if authority.tool("c_compiler").path != Path(os.path.abspath(clang())):
        raise ValueError("Merlin MLIR producer requires the pinned actual Merlin clang")
    if authority.tool(LOWERING_TOOL_ROLE).path != Path(os.path.abspath(m2m_python())):
        raise ValueError("Merlin MLIR producer requires the pinned actual lowering Python")
    if authority.target_abi != K1_TARGET_ABI:
        raise ValueError("Merlin production objects require the K1 ELF64 EM_RISCV target ABI")
    query_env = {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PATH": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    compiler_query = subprocess.run(
        [str(authority.tool("c_compiler").path), "--print-resource-dir"],
        capture_output=True,
        timeout=30,
        cwd=authority.sysroot,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        env=query_env,
    )
    if compiler_query.returncode:
        raise ValueError("bound Merlin compiler resource-directory query failed")
    compiler_resource = Path(compiler_query.stdout.decode("utf-8", errors="strict").strip()).resolve()
    python_query = subprocess.run(
        [
            str(authority.tool(LOWERING_TOOL_ROLE).path),
            "-I",
            "-c",
            "import json,pathlib,sys,sysconfig;"
            "P=pathlib.Path;site=P(sysconfig.get_path('purelib'));"
            "trees={'lowering_stdlib':str(P(sysconfig.get_path('stdlib')).resolve()),"
            "'lowering_numpy':str((site/'numpy').resolve()),"
            "'lowering_torch_mlir':str((site/'torch_mlir').resolve()),"
            "'lowering_yaml':str((site/'yaml').resolve()),"
            "'lowering_xdsl':str((site/'xdsl').resolve()),"
            "'lowering_immutabledict':str((site/'immutabledict').resolve()),"
            "'lowering_distutils_hack':str((site/'_distutils_hack').resolve())};"
            "base=[site/'typing_extensions.py',site/'_virtualenv.py',"
            "site/'_cuda_bindings_redirector.py',P(sys.prefix)/'pyvenv.cfg'];"
            "more=sorted(site.glob('*.pth'))+sorted(site.glob('__editable__*.py'));"
            "files={('lowering_config_%03d'%i):str(p.resolve()) for i,p in enumerate(base+more)};"
            "print(json.dumps({'trees':trees,'files':files},sort_keys=True))",
        ],
        capture_output=True,
        timeout=30,
        cwd=authority.sysroot,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        env=query_env,
    )
    if python_query.returncode:
        raise ValueError("bound lowering-environment resource query failed")
    try:
        queried = json.loads(python_query.stdout.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bound lowering-environment resource query is malformed") from exc
    expected_resources = {
        "compiler_resource_dir": compiler_resource,
        **{name: Path(value).resolve() for name, value in queried["trees"].items()},
    }
    if {resource.name: resource.path for resource in authority.tree_resources} != expected_resources:
        raise ValueError("Merlin producer transitive compiler/lowering resource closure differs")
    expected_files = {name: Path(value).resolve() for name, value in queried["files"].items()}
    if {resource.name: resource.path for resource in authority.file_resources} != expected_files:
        raise ValueError("Merlin producer transitive lowering config-file closure differs")
    expected = (public.closure_root / "lib/libpublic_anchor.a").resolve()
    if (
        len(authority.static_libraries) != 1
        or authority.static_libraries[0][0] != "public_anchor"
        or authority.static_libraries[0][1] != expected
    ):
        raise ValueError("Merlin MLIR producer requires the exact public_anchor library")


def _execute(
    public: PublicBuildBundle, authority: MultiToolchainAuthority, output_root: Path
) -> tuple[SessionDescriptor, dict[str, list[str]], dict[str, Path]]:
    _validate_authority(public, authority)
    descriptor, programs = _load_recipe(public, authority)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    recipe: dict[str, list[str]] = {}
    host_model_objects: list[Path] = []
    riscv_model_objects: list[Path] = []
    worker = repo_root() / "merlin/python/merlin/compare/paper_merlin_lower_worker.py"
    compiler_resource = authority.tree_resource("compiler_resource_dir").path
    for program in programs:
        program_root = output_root / f"program-{program.id}"
        command = [
            str(authority.tool(LOWERING_TOOL_ROLE).path),
            str(worker),
            str(public.closure_root / program.resource),
            program.entrypoint,
            str(program_root),
            str(compiler_resource),
            authority.target_abi.target_triple,
            authority.target_abi.march,
            authority.target_abi.mabi,
            ",".join(authority.target_abi.features),
        ]
        _run(command, f"lower program {program.id}", authority)
        host_object = program_root / "model_host.o"
        riscv_object = program_root / "model_riscv.o"
        host_model_objects.append(host_object)
        riscv_model_objects.append(riscv_object)
        outputs[f"program_{program.id}_source_mlir"] = program_root / "model.mlir"
        outputs[f"program_{program.id}_lowering_script"] = program_root / "run_lowering.py"
        outputs[f"program_{program.id}_upstream_mlir"] = program_root / "model.upstream.mlir"
        outputs[f"program_{program.id}_llvm_ir"] = program_root / "model.ll"
        outputs[f"program_{program.id}_host_object"] = host_object
        outputs[f"program_{program.id}_riscv_object"] = riscv_object
        recipe[f"lower_program_{program.id}"] = command
    adapter = output_root / "session_adapter.c"
    adapter.write_text(_render_adapter(descriptor, programs), encoding="utf-8")
    outputs["session_adapter_source"] = adapter
    runner_source = output_root / "session_runner.c"
    runner_source.write_text(
        render_runner_source(response_capacity=_response_capacity(descriptor, programs)),
        encoding="utf-8",
    )
    outputs["session_runner_source"] = runner_source
    c_driver = str(authority.tool("cxx_compiler").path)
    include = str(public.closure_root / "runtime")
    runtime_source = public.closure_root / "runtime/merlin_model.c"
    runtime_object = output_root / "merlin_model_host.o"
    adapter_object = output_root / "session_adapter_host.o"
    runtime_riscv_object = output_root / "merlin_model_riscv.o"
    adapter_riscv_object = output_root / "session_adapter_riscv.o"
    runner_object = output_root / "runner.o"
    commands = {
        "compile_runtime": [
            c_driver,
            "-x",
            "c",
            "-std=c11",
            "-O2",
            "-fPIC",
            "-I",
            include,
            "-c",
            str(runtime_source),
            "-o",
            str(runtime_object),
        ],
        "compile_adapter": [
            c_driver,
            "-x",
            "c",
            "-std=c11",
            "-O2",
            "-fPIC",
            "-I",
            include,
            "-c",
            str(adapter),
            "-o",
            str(adapter_object),
        ],
        "compile_runner": [c_driver, "-x", "c", "-std=c11", "-O2", "-c", str(runner_source), "-o", str(runner_object)],
        "compile_runtime_riscv": [
            str(authority.tool("c_compiler").path),
            *RISCV_FLAGS,
            f"-resource-dir={authority.tree_resource('compiler_resource_dir').path}",
            f"--sysroot={authority.sysroot}",
            "-I",
            include,
            "-c",
            str(runtime_source),
            "-o",
            str(runtime_riscv_object),
        ],
        "compile_adapter_riscv": [
            str(authority.tool("c_compiler").path),
            *RISCV_FLAGS,
            f"-resource-dir={authority.tree_resource('compiler_resource_dir').path}",
            f"--sysroot={authority.sysroot}",
            "-I",
            include,
            "-c",
            str(adapter),
            "-o",
            str(adapter_riscv_object),
        ],
    }
    for name, argv in commands.items():
        _run(argv, name, authority)
        recipe[name] = argv
    host_composite = output_root / "merlin_session_host.o"
    host_partial = [
        str(authority.tool("host_linker").path),
        "-r",
        *(str(path) for path in host_model_objects),
        str(runtime_object),
        str(adapter_object),
        "-o",
        str(host_composite),
    ]
    _run(host_partial, "host partial link", authority)
    recipe["host_partial_link"] = host_partial
    composite = output_root / "merlin_session_riscv.o"
    partial = [
        str(authority.tool("linker").path),
        "-m",
        "elf64lriscv",
        "-r",
        *(str(path) for path in riscv_model_objects),
        str(runtime_riscv_object),
        str(adapter_riscv_object),
        *(str(row[1]) for row in authority.static_libraries),
        "-o",
        str(composite),
    ]
    _run(partial, "RISC-V partial link", authority)
    recipe["riscv_partial_link"] = partial
    runner = output_root / "runner"
    link = [
        c_driver,
        str(runner_object),
        str(host_composite),
        *(str(row[1]) for row in authority.static_libraries),
        "-o",
        str(runner),
    ]
    _run(link, "runner link", authority)
    recipe["link_runner"] = link
    outputs.update(
        {
            "runtime_object": runtime_object,
            "session_adapter_object": adapter_object,
            "runtime_riscv_object": runtime_riscv_object,
            "session_adapter_riscv_object": adapter_riscv_object,
            "runner_object": runner_object,
            "host_composite_object": host_composite,
            "composite_object": composite,
            "runner": runner,
        }
    )
    return descriptor, recipe, outputs


def build_merlin_mlir_model_object(
    public_manifest: str | Path, authority_path: str | Path, output_root: str | Path
) -> MerlinMLIRBuild:
    public = verify_public_build_bundle(public_manifest)
    authority = load_multi_toolchain_authority(authority_path)
    output_root = Path(output_root).resolve()
    descriptor, recipe, outputs = _execute(public, authority, output_root)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "recipe_id": RECIPE_ID,
        "entrypoint": ENTRYPOINT,
        "target_abi": authority.target_abi.to_dict(),
        "host_liveness_abi": {"name": "host_x86_64_non_measurement_tracer", "elf_class": 64, "elf_machine": 62},
        "producer_implementation": _implementation_identity(),
        "public_manifest_sha256": public.manifest_sha256,
        "public_tree_sha256": public.tree_sha256,
        "toolchain_authority_sha256": authority.sha256,
        "descriptor_sha256": descriptor.sha256,
        "recipe": _normalize_recipe(recipe, output_root, public, authority),
        "outputs": {
            name: {"path": path.relative_to(output_root).as_posix(), "sha256": _sha(path), "size": path.stat().st_size}
            for name, path in sorted(outputs.items())
        },
    }
    receipt_path = output_root / "merlin_mlir_build_receipt.json"
    _write_json(receipt_path, receipt)
    _verify_exact_output_graph(output_root, outputs, receipt=receipt_path)
    return MerlinMLIRBuild(receipt_path, outputs["composite_object"], outputs["runner"], descriptor)


def _resolve_outputs(receipt_path: Path, rows: object) -> dict[str, Path]:
    if not isinstance(rows, Mapping):
        raise ValueError("Merlin build outputs must be a mapping")
    result: dict[str, Path] = {}
    for name, raw in rows.items():
        if not isinstance(raw, Mapping):
            raise ValueError("Merlin build output row must be a mapping")
        _closed(raw, {"path", "sha256", "size"}, f"Merlin output {name}")
        relative = Path(str(raw["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Merlin build output path traverses its receipt")
        path = (receipt_path.parent / relative).resolve()
        if (
            not path.is_relative_to(receipt_path.parent)
            or path.is_symlink()
            or not path.is_file()
            or _sha(path) != raw["sha256"]
            or path.stat().st_size != raw["size"]
        ):
            raise ValueError(f"Merlin build output identity differs for {name}")
        result[str(name)] = path
    return result


def _verify_exact_output_graph(root: Path, outputs: Mapping[str, Path], *, receipt: Path | None) -> None:
    root = root.resolve()
    expected_files = {path.resolve().relative_to(root).as_posix() for path in outputs.values()}
    if receipt is not None:
        expected_files.add(receipt.resolve().relative_to(root).as_posix())
    expected_dirs = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in [*names, *files]:
            if (directory_path / name).is_symlink():
                raise ValueError("Merlin build output graph contains a symlink")
        actual_dirs.update((directory_path / name).relative_to(root).as_posix() for name in names)
        actual_files.update((directory_path / name).relative_to(root).as_posix() for name in files)
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError(
            "Merlin build output graph has omitted or extra paths: "
            f"missing_files={sorted(expected_files - actual_files)} "
            f"extra_files={sorted(actual_files - expected_files)} "
            f"extra_dirs={sorted(actual_dirs - expected_dirs)}"
        )


def _normalize_recipe(
    value: object, output_root: Path, public: PublicBuildBundle, authority: MultiToolchainAuthority
) -> object:
    replacements = {
        str(output_root.resolve()): "<OUTPUT_ROOT>",
        str(public.closure_root.resolve()): "<PUBLIC_ROOT>",
        str(public.manifest_path.resolve()): "<PUBLIC_MANIFEST>",
        str(authority.path.resolve()): "<TOOLCHAIN_AUTHORITY>",
        str(authority.sysroot.resolve()): "<SYSROOT>",
        **{str(tool.path.resolve()): f"<TOOL:{tool.role}>" for tool in authority.tools},
        **{str(resource.path.resolve()): f"<TREE:{resource.name}>" for resource in authority.tree_resources},
        **{str(resource.path.resolve()): f"<FILE:{resource.name}>" for resource in authority.file_resources},
        **{str(path.resolve()): f"<LIB:{name}>" for name, path, _sha256, _size in authority.static_libraries},
    }
    if isinstance(value, str):
        result = value
        for source, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            result = result.replace(source, replacement)
        return result
    if isinstance(value, list):
        return [_normalize_recipe(item, output_root, public, authority) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_recipe(child, output_root, public, authority) for key, child in value.items()}
    raise ValueError("Merlin build recipe contains a non-string command value")


def _synthetic_request(descriptor: SessionDescriptor) -> bytes:
    values = (2, 1, 4, 5)
    return encode_request(
        descriptor,
        [
            InputFrame(InputEndpoint(program, input_index), step, struct.pack("=q", value))
            for (program, input_index, step), value in zip(descriptor.required_input_keys, values, strict=True)
        ],
    )


def _run_and_decode(runner: Path, descriptor: SessionDescriptor) -> tuple[int, ...]:
    completed = subprocess.run(
        [str(runner), str(runner.parent)],
        input=_synthetic_request(descriptor),
        capture_output=True,
        timeout=30,
        cwd=runner.parent,
        close_fds=True,
        env={"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": ""},
    )
    if completed.returncode:
        raise ValueError(f"Merlin session runner failed public liveness ({completed.returncode})")
    response = decode_response(completed.stdout, expected_descriptor=descriptor)
    return tuple(struct.unpack("=q", frame.payload)[0] for frame in response.outputs)


def verify_merlin_mlir_build_barrier(
    public_manifest: str | Path, authority_path: str | Path, receipt_path: str | Path
) -> VerifiedBuildBarrier:
    public = verify_public_build_bundle(public_manifest)
    authority = load_multi_toolchain_authority(authority_path)
    receipt_path = Path(receipt_path).resolve()
    receipt = _load_json(receipt_path, "Merlin MLIR build receipt")
    _closed(
        receipt,
        {
            "schema",
            "recipe_id",
            "entrypoint",
            "public_manifest_sha256",
            "public_tree_sha256",
            "toolchain_authority_sha256",
            "descriptor_sha256",
            "target_abi",
            "host_liveness_abi",
            "producer_implementation",
            "recipe",
            "outputs",
        },
        "Merlin MLIR build receipt",
    )
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("recipe_id") != RECIPE_ID
        or receipt.get("entrypoint") != ENTRYPOINT
        or receipt.get("public_manifest_sha256") != public.manifest_sha256
        or receipt.get("public_tree_sha256") != public.tree_sha256
        or receipt.get("toolchain_authority_sha256") != authority.sha256
    ):
        raise ValueError("Merlin MLIR receipt input/recipe identities differ")
    if receipt.get("target_abi") != authority.target_abi.to_dict():
        raise ValueError("Merlin MLIR receipt target ABI differs")
    if receipt.get("host_liveness_abi") != {
        "name": "host_x86_64_non_measurement_tracer",
        "elf_class": 64,
        "elf_machine": 62,
    }:
        raise ValueError("Merlin MLIR host liveness ABI differs")
    if receipt.get("producer_implementation") != _implementation_identity():
        raise ValueError("Merlin MLIR producer implementation identity differs")
    _validate_authority(public, authority)
    descriptor, _programs = _load_recipe(public, authority)
    if receipt.get("descriptor_sha256") != descriptor.sha256:
        raise ValueError("Merlin MLIR receipt descriptor identity differs")
    outputs = _resolve_outputs(receipt_path, receipt.get("outputs"))
    _verify_exact_output_graph(receipt_path.parent, outputs, receipt=receipt_path)
    production_elf = _elf_identity(outputs["composite_object"])
    if (
        production_elf["type"] != 1
        or not _elf_matches_target(production_elf, authority.target_abi)
        or production_elf["global_definitions"].count(ENTRYPOINT) != 1
    ):
        raise ValueError("Merlin production object is not ELF64 EM_RISCV with one session export")
    with tempfile.TemporaryDirectory(prefix="merlin-mlir-closed-replay-") as raw_root:
        replay_root = Path(raw_root).resolve()
        replay_descriptor, replay_recipe, replay_outputs = _execute(public, authority, replay_root)
        _verify_exact_output_graph(replay_root, replay_outputs, receipt=None)
        if replay_descriptor != descriptor:
            raise ValueError("Merlin MLIR clean replay descriptor differs")
        if _normalize_recipe(receipt.get("recipe"), receipt_path.parent, public, authority) != _normalize_recipe(
            replay_recipe, replay_root, public, authority
        ):
            raise ValueError("Merlin MLIR clean replay recipe differs")
        if set(outputs) != set(replay_outputs):
            raise ValueError("Merlin MLIR receipt omits a generated program/object")
        for name, replay_path in replay_outputs.items():
            if _sha(outputs[name]) != _sha(replay_path):
                raise ValueError(f"Merlin MLIR output differs from independent clean replay for {name}")
        replay_values = _run_and_decode(replay_outputs["runner"], descriptor)
    values = _run_and_decode(outputs["runner"], descriptor)
    if values != replay_values:
        raise ValueError("Merlin MLIR runner trajectory differs from independent clean replay")
    return issue_verified_build_barrier(
        public_manifest=public.manifest_path,
        authority_path=authority.path,
        receipt_path=receipt_path,
        runner=outputs["runner"],
        composite_object=outputs["composite_object"],
        descriptor=descriptor,
        verifier=verify_merlin_mlir_build_barrier,
    )


def snapshot_synthetic_merlin_mlir_bundle(
    root: str | Path, manifest_path: str | Path, *, prefill_constant: int = 2
) -> PublicBuildBundle:
    closure = materialize_synthetic_merlin_mlir_public_closure(root, prefill_constant=prefill_constant)
    return snapshot_public_build_bundle(closure, manifest_path)
