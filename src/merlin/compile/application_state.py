"""Bind captured model state to explicit tensor inputs and returned updates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import prod

from merlin.frontends.compile_inputs import StagedCompileInputs
from merlin.llvmlower.c_runtime import DT_BYTES
from merlin.semantic_compiler.model import KernelRequest, TensorType

from .c_sources import c_source_pair


@dataclass(frozen=True)
class StatePort:
    argument_index: int
    name: str
    mode: str
    output_index: int | None
    initialization: str


@dataclass(frozen=True)
class StateDeclaration:
    source_program_sha256: str
    ports: tuple[StatePort, ...]

    @classmethod
    def from_record(cls, record: dict, staged: StagedCompileInputs):
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "source_program_sha256", "ports"}
            or record["schema"] != "merlin.application_state.v1"
            or record["source_program_sha256"] != staged.program_sha256
            or not isinstance(record["ports"], list)
            or not record["ports"]
        ):
            raise ValueError("state declaration must name the exact staged program and its ports")
        ports = []
        for row in record["ports"]:
            if not isinstance(row, dict) or set(row) != {
                "argument_index",
                "name",
                "mode",
                "output_index",
                "initialization",
            }:
                raise ValueError("state port fields differ")
            index, output = row["argument_index"], row["output_index"]
            if any(not isinstance(row[field], str) or not row[field] for field in ("name", "mode", "initialization")):
                raise ValueError("state names, modes and initialization must be explicit strings")
            if type(index) is not int or not 0 <= index < len(staged.argument_bindings):
                raise ValueError("state argument index is outside the staged signature")
            argument = staged.argument_bindings[index]
            if (
                argument["kind"] not in {"buffer", "input"}
                or row["name"] != argument.get("weight", argument.get("name"))
                or row["mode"] not in {"read_only", "updated"}
                or (row["mode"] == "read_only" and output is not None)
                or (row["mode"] == "updated" and (type(output) is not int or output < 0))
                or row["initialization"] not in {"captured", "caller"}
                or (row["initialization"] == "captured" and argument["kind"] != "buffer")
            ):
                raise ValueError("state port role, name, update or initialization differs from its argument")
            ports.append(StatePort(**row))
        indices = {port.argument_index for port in ports}
        updates = [port.output_index for port in ports if port.mode == "updated"]
        if (
            len(indices) != len(ports)
            or len({port.name for port in ports}) != len(ports)
            or len(updates) != len(set(updates))
            or any(
                row["kind"] == "buffer" and index not in indices for index, row in enumerate(staged.argument_bindings)
            )
        ):
            raise ValueError("state declarations must account for every buffer once, with distinct updates")
        return cls(staged.program_sha256, tuple(ports))

    def record(self):
        return {
            "schema": "merlin.application_state.v1",
            "source_program_sha256": self.source_program_sha256,
            "ports": [vars(port) for port in self.ports],
        }

    def digest(self):
        return sha256(json.dumps(self.record(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def compilation_identity(self, argument_bytes):
        initializers = [
            (port.argument_index, sha256(argument_bytes[port.argument_index]).hexdigest())
            for port in self.ports
            if port.initialization == "captured"
        ]
        return sha256(json.dumps([self.digest(), initializers], separators=(",", ":")).encode()).hexdigest()

    def bind(self, source: KernelRequest, argument_bytes: dict[int, bytes]):
        inputs = [key for key, _ in source.input_storages]
        bound = []
        for port in self.ports:
            key = f"arg{port.argument_index}"
            node = source.node(key)
            if (
                key not in inputs
                or node.op != "input"
                or node.effect != "input"
                or (node.type.dtype, node.type.numerical_policy)
                not in {
                    ("f32", "raw-bit-pattern"), ("f64", "raw-bit-pattern"),
                    ("bf16", "raw-bit-pattern"), ("fp8_e4m3", "raw-bit-pattern"),
                    ("i64", "signed-i64"),
                }
            ):
                raise ValueError("state port must remain an explicit supported tensor input")
            if port.mode == "updated" and (
                port.output_index >= len(source.outputs)
                or source.node(source.outputs[port.output_index]).type != node.type
            ):
                raise ValueError("state update output shape or type differs from its input")
            initial = argument_bytes.get(port.argument_index) if port.initialization == "captured" else None
            if port.initialization == "captured" and (
                type(initial) is not bytes or len(initial) != prod(node.type.shape) * DT_BYTES[node.type.dtype]
            ):
                raise ValueError("captured state initialization bytes differ from the tensor signature")
            bound.append(BoundStatePort(port, key, inputs.index(key), node.type, initial))
        return ApplicationState(source.digest(), self.digest(), tuple(bound))


@dataclass(frozen=True)
class BoundStatePort:
    port: StatePort
    node_id: str
    input_index: int
    type: TensorType
    initial: bytes | None


@dataclass(frozen=True)
class ApplicationState:
    source_request_digest: str
    declaration_sha256: str
    ports: tuple[BoundStatePort, ...]

    def check(self, source):
        if self.source_request_digest != source.digest() or not self.ports:
            raise ValueError("application state interface differs from its source")
        declaration = StateDeclaration("", tuple(item.port for item in self.ports))
        initial = {item.port.argument_index: item.initial for item in self.ports if item.initial is not None}
        rebound = declaration.bind(source, initial)
        updates = [item.port.output_index for item in self.ports if item.port.mode == "updated"]
        if (
            rebound.ports != self.ports
            or len({item.node_id for item in self.ports}) != len(self.ports)
            or len({item.port.name for item in self.ports}) != len(self.ports)
            or len(updates) != len(set(updates))
        ):
            raise ValueError("application state ports differ from the source input/output interface")

    def record(self):
        return {
            "schema": "merlin.bound_application_state.v1",
            "source_request_digest": self.source_request_digest,
            "declaration_sha256": self.declaration_sha256,
            "update_publication": "explicit outputs; caller advances state after success",
            "in_place_updates": False,
            "ports": [
                {
                    **vars(item.port),
                    "node_id": item.node_id,
                    "input_index": item.input_index,
                    "type": item.type.record(),
                    "initial_sha256": sha256(item.initial).hexdigest() if item.initial is not None else None,
                    "initial_bytes": len(item.initial) if item.initial is not None else None,
                }
                for item in self.ports
            ],
        }

    def emit_initializers(self, name, *, max_bytes=64 * 1024 * 1024, input_order=None):
        """Copy exact captured initial values; never infer zeros for caller state."""
        initialized = [item for item in self.ports if item.initial is not None]

        def input_index(item):
            return item.input_index if input_order is None else input_order.index(item.node_id)

        port_declarations = [
            "#include <stddef.h>",
            f"#define {name.upper()}_MODEL_STATE_COUNT {len(self.ports)}u",
            f"#define {name.upper()}_CAPTURED_STATE_COUNT {len(initialized)}u",
            "/* State stays caller-owned. Input and update output buffers are separate.",
            " * Advance to returned state only after invoke succeeds; after failure,",
            " * retain the old state and drain/reset outstanding device work. */",
        ]
        for i, item in enumerate(self.ports):
            prefix = f"{name.upper()}_STATE_{i}"
            shape = ",".join(map(str, item.type.shape))
            port_declarations.extend(
                (
                    f"/* State {i}: dense {item.type.dtype}[{shape}], {item.port.mode}, "
                    f"{item.port.initialization} initialization. */",
                    f"#define {prefix}_INPUT_INDEX {input_index(item)}u",
                    f"#define {prefix}_BYTES {prod(item.type.shape) * DT_BYTES[item.type.dtype]}u",
                )
            )
            if item.port.mode == "updated":
                port_declarations.append(f"#define {prefix}_OUTPUT_INDEX {item.port.output_index}u")
        if not initialized:
            return {"captured_initializers": []}, c_source_pair(
                name + "_state",
                public_headers={},
                private_headers={},
                sources={"state.c": "/* All model state is initialized by the caller. */"},
                declarations="\n".join(port_declarations) + "\n",
            )
        if sum(len(item.initial) for item in initialized) > max_bytes:
            raise ValueError("captured state initializers exceed the embedded byte budget")
        for i, item in enumerate(initialized):
            prefix = f"{name.upper()}_CAPTURED_STATE_{i}"
            port_declarations.extend(
                (
                    f"#define {prefix}_PORT_INDEX {self.ports.index(item)}u",
                    f"#define {prefix}_INPUT_INDEX {input_index(item)}u",
                    f"#define {prefix}_BYTES {len(item.initial)}u",
                )
            )
        declarations = f"""/* Buffers and capacities follow the captured-state indices above.
 * Call only when no invocation is using these buffers. Caller-initialized state
 * is not written. All ranges are checked before any initial value is copied. */
int {name}_initialize_captured_state(void *const *buffers,const size_t *capacities);
"""
        arrays, checks, copies = [], [], []
        for i, item in enumerate(initialized):
            size = len(item.initial)
            arrays.append(
                f"static const unsigned char {name}_state_{i}[{size}]={{{','.join(map(str, item.initial))}}};"
            )
            checks.append(f"!buffers[{i}] || capacities[{i}]<{size}u || (uintptr_t)buffers[{i}]>UINTPTR_MAX-{size}u")
            for j, previous in enumerate(initialized[:i]):
                checks.append(
                    f"((uintptr_t)buffers[{j}]<(uintptr_t)buffers[{i}]+{size}u && "
                    f"(uintptr_t)buffers[{i}]<(uintptr_t)buffers[{j}]+{len(previous.initial)}u)"
                )
            copies.append(f"__builtin_memcpy(destinations[{i}],{name}_state_{i},{size}u);")
        body = (
            """#include <stddef.h>
#include <stdint.h>
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "captured state uses little-endian tensors");
"""
            + "\n".join(arrays)
            + f"""
int {name}_initialize_captured_state(void *const *buffers,const size_t *capacities) {{
  if (!buffers || !capacities) return -1;
  void *destinations[{len(initialized)}]={{{",".join(f"buffers[{i}]" for i in range(len(initialized)))}}};
  if ({" || ".join(checks)}) return -1;
  {" ".join(copies)}
  return 0;
}}
"""
        )
        metadata = {
            "entry": name + "_initialize_captured_state",
            "captured_initializers": [
                {"state_port": self.ports.index(item), "input_index": input_index(item), "bytes": len(item.initial)}
                for item in initialized
            ],
        }
        return metadata, c_source_pair(
            name + "_state",
            public_headers={},
            private_headers={},
            sources={"state.c": body},
            declarations="\n".join(port_declarations) + "\n" + declarations,
        )
