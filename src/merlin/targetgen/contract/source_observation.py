"""Explicit original-source/emitted-dataflow observation transport, without roles.

An independently selected private reader interprets target/source facts. Core
retains the exact original ABI, actual complete LLVM dataflow and callback/file
attribution. Neither reader labels nor this transport qualify source equivalence,
instruction effects, ownership, runtime support or a compiler stage.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from merlin.common import invocation_record

from .build_service import file_digest
from .compile_only import CompileOnlySourceAbi
from .emitted_dataflow import observe_emitted_dataflow
from .source_control_flow import READER_SCHEMA, ControlFlowObservationPlan

_METHODS = ("verify", "record", "observe")
_UNKNOWN = ("source_equivalence", "instruction_effects", "ownership", "compiler_stages", "runtime", "physical_timing")


def _plain(path):
    if type(path) is not Path and not isinstance(path, Path):
        raise ValueError("source observation requires explicit canonical file paths")
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("source observation requires canonical unlinked files")
    return path


def _identity(method):
    function = getattr(method, "__func__", None)
    instance = getattr(method, "__self__", None)
    source = inspect.getsourcefile(method) if callable(method) else None
    if function is None or instance is None or source is None:
        raise ValueError("source observation reader needs fixed source-owned bound methods")
    path = _plain(Path(source))
    return function, instance, path, file_digest(path), function.__module__, function.__qualname__, function.__code__


def _json(value):
    """Closed plain data only, never executable descriptors or object reprs."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) in (list, tuple):
        return [_json(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json(item) for key, item in value.items()}
    raise ValueError("source observation descriptors and results require plain data")


def _encoded(value):
    return json.dumps(_json(value), sort_keys=True, allow_nan=False)


@dataclass(frozen=True, eq=False)
class ExplicitSourceObservation:
    """Caller-selected source reader; attribution only, never an issuer.

    The source and typed ABI must originate outside the evaluated compiler.
    This transport checks their exact bytes/signature but cannot establish that
    independent origin. The phase owner retains that separate obligation.
    """

    target: str
    original_source: Path
    original_abi: CompileOnlySourceAbi
    pointer_bits: int
    max_operations: int
    reader: object
    source_pins: tuple[tuple[str, str], ...]
    control_flow_plan: ControlFlowObservationPlan | None = None
    implementation_pins: tuple[tuple[str, str], ...] = field(init=False, repr=False)
    _callbacks: tuple = field(init=False, repr=False)
    _selection: str = field(init=False, repr=False)
    _control_flow_selection: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        from .linalg_iface import parse_linalg_mlir

        files = {Path(__file__)} | {
            Path(inspect.getsourcefile(value))
            for value in (CompileOnlySourceAbi, parse_linalg_mlir, observe_emitted_dataflow)
        }
        if self.control_flow_plan is not None:
            from .emitted_control_flow import observe_emitted_control_flow
            from .mlir_source_admission import admit_mlir_source

            if type(self.control_flow_plan) is not ControlFlowObservationPlan:
                raise ValueError("source observation requires an exact explicit CFG plan")
            plan = self.control_flow_plan.record()
            object.__setattr__(self, "_control_flow_selection", _encoded(plan))
            files.update(
                Path(inspect.getsourcefile(value))
                for value in (
                    ControlFlowObservationPlan,
                    observe_emitted_control_flow,
                    admit_mlir_source,
                )
            )
        object.__setattr__(self, "implementation_pins", tuple((str(path), file_digest(path)) for path in sorted(files)))
        self._verify_pins()
        identities = tuple(_identity(getattr(self.reader, name, None)) for name in self._methods())
        if any((str(row[2]), row[3]) not in self.source_pins for row in identities):
            raise ValueError("source observation reader callback lacks explicit source membership")
        self.reader.verify()
        descriptor = self.reader.record()
        if type(descriptor) is not dict or not descriptor:
            raise ValueError("source observation reader needs an immutable explicit selection")
        if self.control_flow_plan is not None and descriptor.get("schema") != READER_SCHEMA:
            raise ValueError("CFG observation needs the explicit versioned control-flow reader contract")
        object.__setattr__(self, "_callbacks", identities)
        object.__setattr__(self, "_selection", _encoded(descriptor))
        self.verify()

    def _methods(self):
        return _METHODS if self.control_flow_plan is None else ("verify", "record", "observe_control_flow")

    def _verify_pins(self):
        if self.control_flow_plan is not None:
            if (
                type(self.control_flow_plan) is not ControlFlowObservationPlan
                or _encoded(self.control_flow_plan.record()) != self._control_flow_selection
                or self.control_flow_plan.pointer_bits != self.pointer_bits
                or self.control_flow_plan.max_operations != self.max_operations
                or any(
                    (str(path), digest) not in self.source_pins
                    for path, digest in (
                        (self.control_flow_plan.lowered_source, self.control_flow_plan.source_sha256),
                        (self.control_flow_plan.layout_source, self.control_flow_plan.layout_sha256),
                    )
                )
            ):
                raise ValueError("CFG original source/layout/width/bounds selection changed or lacks membership")
        elif self._control_flow_selection is not None:
            raise ValueError("source observation removed its selected CFG plan")
        if (
            type(self.target) is not str
            or not self.target
            or type(self.original_abi) is not CompileOnlySourceAbi
            or type(self.pointer_bits) is not int
            or not 1 <= self.pointer_bits <= 256
            or type(self.max_operations) is not int
            or not 1 <= self.max_operations <= 100000
            or type(self.source_pins) is not tuple
            or not self.source_pins
            or len(self.source_pins) != len(dict(self.source_pins))
        ):
            raise ValueError("source observation requires explicit original ABI/width/bound/source selections")
        self.original_abi.record()
        source = _plain(self.original_source)
        if (str(source), file_digest(source)) not in self.source_pins:
            raise ValueError("source observation omits its independently selected original source")
        for name, digest in (*self.source_pins, *self.implementation_pins):
            path = _plain(Path(name))
            if not path.is_file() or file_digest(path) != digest:
                raise ValueError("source observation source/tool membership changed")
        from .linalg_iface import parse_linalg_mlir

        signature = parse_linalg_mlir(source.read_text())
        for slots, key in ((self.original_abi.inputs, "args"), (self.original_abi.outputs, "results")):
            if len(slots) != len(signature[key]) or any(
                tuple(row["shape"]) != slot.shape or row["dtype"] != slot.dtype
                for slot, row in zip(slots, signature[key], strict=True)
            ):
                raise ValueError("source observation ABI changes original complete ordered tensor types")

    def verify(self):
        self._verify_pins()
        if tuple(_identity(getattr(self.reader, name, None)) for name in self._methods()) != self._callbacks:
            raise ValueError("source observation reader callback identity changed")
        self.reader.verify()
        if _encoded(self.reader.record()) != self._selection:
            raise ValueError("source observation reader immutable selection changed")
        return self.record()

    def record(self):
        result = {
            "target": self.target,
            "original_source": {"path": str(self.original_source), "sha256": file_digest(self.original_source)},
            "original_abi": self.original_abi.record(),
            "pointer_bits": self.pointer_bits,
            "max_operations": self.max_operations,
            "source_pins": [{"path": path, "sha256": digest} for path, digest in self.source_pins],
            "implementation_pins": [{"path": path, "sha256": digest} for path, digest in self.implementation_pins],
            "reader_callbacks": [(str(row[2]), row[3], row[4], row[5]) for row in self._callbacks],
            "reader_selection": json.loads(self._selection),
            "unknown": list(_UNKNOWN),
            "scope": "explicit source/dataflow observation attribution only; no semantic or runtime authority",
        }
        if self.control_flow_plan is not None:
            result["control_flow_plan"] = self.control_flow_plan.record()
            result["reader_contract"] = READER_SCHEMA
        return result

    @property
    def sha256(self):
        return hashlib.sha256(_encoded(self.verify()).encode()).hexdigest()

    def observe(self, *, source, lowered_mlir, command_buffer, command_buffer_path, entry_symbol, evidence_root):
        before = self.verify()
        paths = tuple(_plain(path) for path in (source, lowered_mlir, command_buffer_path, evidence_root))
        source, lowered_mlir, command_buffer_path, evidence_root = paths
        plan = self.control_flow_plan
        if plan is not None:
            text = plan.emitted_text(
                lowered_mlir,
                entry_symbol=entry_symbol,
                pointer_bits=self.pointer_bits,
                max_operations=self.max_operations,
            )
        observed_files = {path: file_digest(path) for path in (source, lowered_mlir, command_buffer_path)}
        if source.read_bytes() != self.original_source.read_bytes():
            raise ValueError("source observation changed independently selected original program bytes")
        if json.loads(command_buffer_path.read_bytes()) != command_buffer:
            raise ValueError("source observation did not consume actual emitted command buffer bytes")
        bindings = self.original_abi.bind(command_buffer)
        dataflow = (
            observe_emitted_dataflow(
                lowered_mlir.read_bytes().decode("utf-8"),
                entry_symbol=entry_symbol,
                pointer_bits=self.pointer_bits,
                max_operations=self.max_operations,
            )
            if plan is None
            else self._observe_control_flow(text, entry_symbol)
        )
        if len(dataflow.arguments) != bindings["pointer_arity"]:
            raise ValueError("source observation omits original input/output pointer slots")
        output = evidence_root / "source_observation.json"
        if output.exists() or output.is_symlink():
            raise ValueError("source observation requires a new private product")
        method = self.reader.observe if plan is None else self.reader.observe_control_flow
        selected_inputs = () if plan is None else (plan.lowered_source, plan.layout_source)
        with invocation_record.observe_call(
            evidence_root,
            stage="explicit_original_source_dataflow_observation"
            if plan is None
            else "explicit_original_source_cfg_observation",
            function=method,
            arguments={"selection": before, "entry_symbol": entry_symbol, "original_abi": self.original_abi.record()},
            inputs=tuple(
                dict.fromkeys((self.original_source, source, lowered_mlir, command_buffer_path, *selected_inputs))
            ),
            dependencies=(
                *(Path(p) for p, _ in self.implementation_pins),
                *(Path(p) for p, _ in self.source_pins),
            ),
            outputs=(output,),
        ) as observation:
            returned = method(
                source=source,
                lowered_mlir=lowered_mlir,
                **({"dataflow": dataflow} if plan is None else {"control_flow": dataflow}),
                original_abi=self.original_abi,
                command_buffer=command_buffer,
                entry_symbol=entry_symbol,
            )
            if type(returned) is not dict:
                raise ValueError("source observation reader returned no plain observation mapping")
            result = {
                "selection": before,
                "actual_dataflow_sha256": dataflow.source_sha256,
                "observations": _json(returned),
                "unknown": list(_UNKNOWN),
                "scope": "actual source/emitted observation only; not source equivalence or stage/runtime proof",
            }
            if plan is not None:
                result.pop("actual_dataflow_sha256")
                result.update(
                    schema="merlin.explicit_source_cfg_observation.v1",
                    actual_control_flow_sha256=dataflow.source_sha256,
                    control_flow=asdict(dataflow),
                    unknown=list(dict.fromkeys((*_UNKNOWN, *dataflow.unknown, *plan.record()["unknown"]))),
                    scope="actual complete static CFG/source observation only; no source/layout/stage/runtime proof",
                )
            encoded = _encoded(result)
            with output.open("x", encoding="utf-8") as file:
                file.write(encoded + "\n")
            observation.returned(stdout=encoded + "\n")
        if self.verify() != before:
            raise ValueError("source observation selection changed during invocation")
        if any(file_digest(path) != digest for path, digest in observed_files.items()):
            raise ValueError("source observation actual original/emitted inputs changed during evaluation")
        invocation_record.verify(observation.path)
        return result

    def _observe_control_flow(self, text, entry_symbol):
        from .emitted_control_flow import observe_emitted_control_flow

        plan = self.control_flow_plan
        limits = plan.record()["limits"]
        return observe_emitted_control_flow(
            text,
            entry_symbol=entry_symbol,
            pointer_bits=plan.pointer_bits,
            **{name: value for name, value in limits.items() if name != "max_layout_bytes"},
        )
