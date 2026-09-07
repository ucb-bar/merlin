"""Out-of-band host authority for prepacking one exact captured constant payload.

The objects here are created inside the trusted host, after capture-byte and
normalization-identity verification. They are never deserialized from candidate
JSON, inferred from a tensor role, or accepted as a performance certificate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from merlin.frontends.argument_identity import ArgumentIdentityBridge
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.perf.storage_encoding import GroupedAxesStorage
from .captured_constants import CapturedConstant


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class HostPrepackAuthorization:
    tensor: str
    command_buffer_digest: str
    encoding: GroupedAxesStorage
    constant: CapturedConstant
    argument_bridge: ArgumentIdentityBridge

    def validate_binding(self, cb: Mapping, tensor: str, encoding: GroupedAxesStorage) -> None:
        if (tensor != self.tensor or _digest(cb) != self.command_buffer_digest
                or encoding != self.encoding):
            raise ValueError("prepack authority does not bind this command buffer/tensor/encoding")
        abi = cb.get("kernel_abi")
        args = abi.get("args") if isinstance(abi, Mapping) else None
        if not isinstance(args, list):
            raise ValueError("authorized prepacking requires the declared readonly pointer ABI")
        matching = [arg for arg in args if isinstance(arg, Mapping) and arg.get("tensor") == tensor]
        if len(matching) != 1 or matching[0].get("access") != "read":
            raise ValueError("authorized prepacking requires an exactly readonly input")

    def validate_payload(self, actual_bytes: bytes) -> None:
        if type(actual_bytes) is not bytes or actual_bytes != self.constant.logical_payload:
            raise ValueError("actual initializer differs from the authorized captured constant payload")

    def to_evidence(self) -> dict:
        return {"schema": "host_capture_prepack_authorization_v1", "tensor": self.tensor,
                "command_buffer_digest": self.command_buffer_digest,
                "encoding_digest": _digest(self.encoding.to_dict()),
                "capture": self.constant.to_evidence(),
                "argument_bridge": self.argument_bridge.to_evidence(),
                "prepack_authorized": True,
                "actual_initializer_payload": "must be checked again by the consuming caller",
                "scope": "offline format preparation of this fixed captured payload only",
                "numerical_equivalence": "UNPROVEN", "performance_promotion": False}


def authorize_capture_prepack(*, constant: CapturedConstant, bridge: ArgumentIdentityBridge,
                             raw_source_text: str, normalized_source_text: str,
                             command_buffer: Mapping, tensor: str) -> HostPrepackAuthorization:
    """Join host evidence to a particular source argument and compiler-selected ABI.

    A changed capture, normalization, tensor binding, encoding or actual input
    invalidates this grant. Callback selection and evidence creation belong to
    the trusted host policy; callers cannot submit serialized grants in a CB.
    """
    from xdsl.dialects.builtin import TensorType

    if type(constant) is not CapturedConstant or type(bridge) is not ArgumentIdentityBridge:
        raise ValueError("prepack authorization requires host-created typed capture and identity evidence")
    if (hashlib.sha256(raw_source_text.encode()).hexdigest() != bridge.source_sha256
            or hashlib.sha256(normalized_source_text.encode()).hexdigest() != bridge.normalized_sha256):
        raise ValueError("source bytes differ from the verified argument identity bridge")
    if hashlib.sha256(constant.logical_payload).hexdigest() != constant.payload_sha256:
        raise ValueError("capture evidence no longer binds its immutable byte snapshot")
    raw = parse_mlir_text(raw_source_text)
    weights_file = getattr(raw.attributes.get("prov.weights_file"), "data", None)
    if not isinstance(weights_file, str) or Path(weights_file).resolve() != Path(constant.safetensors_path).resolve():
        raise ValueError("raw capture source does not bind this verified weights file")
    normalized = parse_mlir_text(normalized_source_text)
    entries = [op for op in normalized.body.block.ops
               if op.name == "func.func" and op.sym_name.data == bridge.entry]
    if len(entries) != 1 or len(entries[0].body.blocks) != 1:
        raise ValueError("normalized source entry is not the verified identity boundary")
    arguments = tuple(entries[0].body.block.args)
    index = constant.entry_argument_index
    if (tuple(str(arg.type) for arg in arguments) != bridge.argument_types
            or type(index) is not int or not 0 <= index < len(arguments)):
        raise ValueError("captured argument index/types disagree with normalization bridge")
    value_type = arguments[index].type
    if (not isinstance(value_type, TensorType) or tuple(value_type.get_shape()) != constant.source_shape
            or str(value_type.get_element_type()) != constant.source_dtype):
        raise ValueError("captured constant type differs from the actual normalized argument")
    params = command_buffer.get("params")
    plan = params.get("global_program_plan") if isinstance(params, Mapping) else None
    encodings = params.get("storage_encodings") if isinstance(params, Mapping) else None
    if (not isinstance(plan, Mapping) or plan.get("source_sha256") != bridge.normalized_sha256
            or not isinstance(encodings, Mapping) or tensor not in encodings):
        raise ValueError("compiler plan/encoding is not bound to the normalized source")
    bindings = plan.get("entry_bindings")
    if (not isinstance(bindings, list) or len(bindings) != len(arguments)
            or bindings[index] != tensor or bindings.count(tensor) != 1):
        raise ValueError("captured argument is not uniquely bound to the requested ABI tensor")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("prepacking requires explicit source-task read/write accounting")
    for task in tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("writes"), list):
            raise ValueError("prepacking requires explicit source-task writes")
        if tensor in task["writes"]:
            raise ValueError("compiler plan writes the purported immutable input")
    encoding = GroupedAxesStorage.from_dict(encodings[tensor])
    if encoding.logical_shape != constant.source_shape or encoding.dtype != constant.source_dtype:
        raise ValueError("prepacking encoding does not preserve the captured logical type")
    specs = command_buffer.get("tensors")
    spec = specs.get(tensor) if isinstance(specs, Mapping) else None
    if (not isinstance(spec, Mapping) or spec.get("shape") != list(encoding.physical_shape)
            or spec.get("dtype") != encoding.dtype):
        raise ValueError("prepacking encoding differs from the physical ABI tensor")
    authority = HostPrepackAuthorization(tensor, _digest(command_buffer), encoding, constant, bridge)
    authority.validate_binding(command_buffer, tensor, encoding)
    return authority
