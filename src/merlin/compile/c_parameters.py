"""Bind a staged model's immutable parameters in C or a checked weight blob.

Runtime arguments keep their captured order. Mutable model buffers require an
explicit state interface and cannot become static constants through this API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from math import prod
from pathlib import Path

import numpy as np

from merlin.frontends.compile_inputs import _WEIGHT_DTYPES, load_compile_inputs, load_lifted_argument_bytes
from merlin.llvmlower.c_runtime import DT_BYTES, embed_array
from merlin.llvmlower.weights_pack import DTYPE_BYTES, WeightEntry, checked_payload, pack


@dataclass(frozen=True)
class FrozenCArguments:
    declarations: str
    arguments: tuple[str, ...]
    runtime_inputs: tuple[dict, ...]
    parameters: tuple[dict, ...]
    weights_sha256: str | None
    constants_sha256: str | None = None


@dataclass(frozen=True)
class ApplicationParameters:
    """Checked immutable weight-blob bindings outside the semantic graph.

    The path is invocation-owned intake, never part of serialized search state.
    Public records retain byte identities and the existing weight argument table.
    """

    source_request_digest: str
    staged_program_sha256: str
    archive_sha256: str
    payload_sha256: str
    payload_bytes: int
    entries: tuple[WeightEntry, ...]
    archive_path: Path = field(compare=False, repr=False)

    @property
    def node_ids(self):
        return tuple(f"arg{entry.arg_index}" for entry in self.entries)

    def check(self, source):
        if self.source_request_digest != source.digest() or not self.entries:
            raise ValueError("model parameters differ from their semantic source")
        for digest in (self.staged_program_sha256, self.archive_sha256, self.payload_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("model parameter identity is not canonical SHA256")
        if type(self.payload_bytes) is not int or not 0 < self.payload_bytes < 1 << 61:
            raise ValueError("model parameter payload exceeds its byte/hash domain")
        if len(set(self.node_ids)) != len(self.entries):
            raise ValueError("model parameter ports are repeated")
        for key, entry in zip(self.node_ids, self.entries, strict=True):
            node = source.node(key)
            width = DT_BYTES.get(node.type.dtype)
            if (
                node.op != "input" or node.effect != "input" or key not in dict(source.input_storages)
                or type(entry.arg_index) is not int or entry.arg_index < 0
                or tuple(entry.shape) != node.type.shape or width is None or entry.dtype not in DTYPE_BYTES
                or not any(dtype[0] == node.type.dtype and dtype[1] == entry.dtype for dtype in _WEIGHT_DTYPES.values())
                or type(entry.offset) is not int or entry.offset < 0 or entry.offset % width
                or type(entry.nbytes) is not int or entry.nbytes != prod(node.type.shape) * width
                or entry.offset + entry.nbytes > self.payload_bytes
            ):
                raise ValueError("model parameter offsets, types or source ports differ")
        for i, entry in enumerate(self.entries):
            for other in self.entries[:i]:
                if entry.offset < other.offset + other.nbytes and other.offset < entry.offset + entry.nbytes:
                    if (entry.name, entry.offset, entry.nbytes, entry.dtype, entry.shape) != (
                        other.name, other.offset, other.nbytes, other.dtype, other.shape
                    ):
                        raise ValueError("model parameter spans overlap without an identical stored tensor")

    def record(self):
        return {
            "schema": "merlin.application_parameters.v1",
            "source_request_digest": self.source_request_digest,
            "staged_program_sha256": self.staged_program_sha256,
            "archive_sha256": self.archive_sha256,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "path": "handoff/parameters.bin",
            "ports": [{"node_id": key, **vars(entry)}
                      for key, entry in zip(self.node_ids, self.entries, strict=True)],
            "ownership": "caller-owned immutable bytes through completion and recovery",
            "validation": "SHA256 at bind; immutable lifetime required after bind",
        }

    def write_payload(self, destination):
        info = checked_payload(self.archive_path, archive_sha256=self.archive_sha256, out_blob=destination)
        if (info["payload_sha256"], info["payload_bytes"]) != (self.payload_sha256, self.payload_bytes):
            raise ValueError("model parameter payload changed during publication")

    def emit_binding(self, name):
        from merlin.common.paths import runtime_dir

        # Reuse the installed byte SHA256 implementation. Its output-specific
        # float helper is referenced only to avoid an unused-static warning.
        checksum = (runtime_dir() / "baremetal/spike/output_sha256.h").read_text()
        tag = int(self.payload_sha256[:16], 16)
        alignment = max(DTYPE_BYTES[entry.dtype] for entry in self.entries)
        expected = ",".join(str(byte) for byte in bytes.fromhex(self.payload_sha256))
        declarations = f"""
typedef struct {{ const unsigned char *data; size_t bytes; uintptr_t seal; }} {name}_parameters;
#define {name.upper()}_PARAMETER_BYTES {self.payload_bytes}u
#define {name.upper()}_PARAMETER_ALIGNMENT {alignment}u
/* Bind once while idle. Keep the descriptor at this address and the bytes
 * immutable until all calls/recovery finish.
 * Failed binding leaves the previous descriptor intact. No allocation is performed. */
int {name}_parameters_bind({name}_parameters *,const void *,size_t);
"""
        implementation = f"""
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "parameter tensors use little-endian codes");
static int {name}_parameters_ready(const {name}_parameters *p) {{
  return {name}_valid(p,sizeof(*p)) && p->bytes=={self.payload_bytes}u &&
    !((uintptr_t)p->data%{alignment}u) &&
    {name}_valid(p->data,p->bytes) &&
    p->seal==((uintptr_t)p^(uintptr_t)p->data^(uintptr_t)p->bytes^(uintptr_t)UINT64_C({tag}));
}}
int {name}_parameters_bind({name}_parameters *p,const void *data,size_t bytes) {{
  if (!{name}_valid(p,sizeof(*p)) || bytes!={self.payload_bytes}u ||
      (uintptr_t)data%{alignment}u ||
      !{name}_valid(data,bytes) || {name}_overlap(p,sizeof(*p),data,bytes)) return -1;
  (void)merlin_output_sha256;
  static const unsigned char expected[32]={{{expected}}};
  unsigned char actual[32];merlin_sha256 hash;
  merlin_sha256_init(&hash);merlin_sha256_update(&hash,data,bytes);merlin_sha256_final(&hash,actual);
  unsigned mismatch=0;for(unsigned i=0;i<32;++i) mismatch|=actual[i]^expected[i];
  if (mismatch) return -2;
  *p=({name}_parameters){{data,bytes,(uintptr_t)p^(uintptr_t)data^(uintptr_t)bytes^(uintptr_t)UINT64_C({tag})}};
  return 0;
}}
"""
        return declarations, implementation, {"parameter_sha256.h": checksum}


def bind_application_parameters(directory, staged, source):
    """Bind all declared parameters to the existing safetensors weight table."""
    declared = {i: row for i, row in enumerate(staged.argument_bindings) if row["kind"] == "param"}
    if not declared:
        return None
    root = Path(directory)
    entries = tuple(entry for entry in pack(root / "weights.safetensors.manifest.json", root / "weights.safetensors")
                    if entry.arg_index in declared)
    if {entry.arg_index for entry in entries} != set(declared):
        raise ValueError("weight argument table omits a declared model parameter")
    for entry in entries:
        row = declared[entry.arg_index]
        if (entry.name != row["weight"] or entry.shape != row["shape"]
                or source.node(f"arg{entry.arg_index}").type.dtype != row["dtype"]):
            raise ValueError("weight argument table differs from the staged parameter signature")
    info = checked_payload(root / "weights.safetensors", archive_sha256=staged.weights_sha256)
    result = ApplicationParameters(source.digest(), staged.program_sha256, staged.weights_sha256,
                                   info["payload_sha256"], info["payload_bytes"], entries,
                                   root / "weights.safetensors")
    result.check(source)
    return result


def parameter_source_identity(source_identity, staged):
    """Seal the declared weights and their argument roles into compilation identity."""
    value = [source_identity, staged.weights_sha256, staged.argument_bindings]
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_c_arguments(
    directory: str | Path,
    *,
    name: str,
    source_sha256: str,
    max_embedded_bytes: int = 64 * 1024 * 1024,
) -> FrozenCArguments:
    """Embed declared f32 parameters, returning expressions in function order.

    The source identity must name the exact staged IR used by the caller's
    compiler. The checked staging API reopens IR, signature and weights; runtime
    samples and reference outputs are not compiler inputs. The byte budget
    limits source generation and is not a device-memory capacity declaration.
    """
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not name or name[0] not in letters or any(char not in letters + "0123456789_" for char in name):
        raise ValueError("parameter namespace must be a C identifier")
    if type(max_embedded_bytes) is not int or max_embedded_bytes < 0:
        raise ValueError("parameter source budget must be a nonnegative byte count")
    staged = load_compile_inputs(directory)
    if staged.program_sha256 != source_sha256:
        raise ValueError("parameter source differs from compiled IR")
    if not staged.argument_bindings:
        raise ValueError("parameter export requires a staged argument signature")
    if any(row["kind"] == "buffer" for row in staged.argument_bindings):
        raise ValueError("mutable model buffers require an explicit state interface")
    if any(row["dtype"] != "f32" for row in staged.argument_bindings):
        raise ValueError("C parameter export currently requires f32 source arguments")
    payloads = load_lifted_argument_bytes(directory, staged, max_bytes=max_embedded_bytes)
    declarations, arguments, parameters = [], [], []
    symbols: dict[tuple[str, str], str] = {}
    runtime: dict[int, dict] = {}
    byte_count = 0
    for index, row in enumerate(staged.argument_bindings):
        if row["kind"] == "input":
            ordinal = row["input_index"]
            runtime[ordinal] = dict(row)
            arguments.append(f"inputs[{ordinal}]")
            continue
        weight = (row.get("file", "weights.safetensors"), row["weight"])
        if weight not in symbols:
            payload = payloads[index]
            byte_count += len(payload)
            if byte_count > max_embedded_bytes:
                raise ValueError("embedded parameters exceed the C source byte budget")
            values = np.frombuffer(payload, dtype="<f4")
            if not values.size:
                raise ValueError("C parameter export requires nonempty parameter tensors")
            symbol = f"{name}_parameter_{len(symbols)}"
            symbols[weight] = symbol
            declarations.append(f"static const float {symbol}[{values.size}] = {{{embed_array(values, 'f32')}}};")
            parameters.append({**row, "symbol": symbol, "bytes": len(payload)})
        arguments.append(symbols[weight])
    return FrozenCArguments(
        "\n".join(declarations),
        tuple(arguments),
        tuple(runtime[index] for index in range(len(runtime))),
        tuple(parameters),
        staged.weights_sha256,
        staged.constants_sha256,
    )
