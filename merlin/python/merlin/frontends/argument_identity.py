"""Host-owned normalization replay that preserves captured entry argument identities.

This proves an input-binding bridge, not numerical equivalence of normalization.
Callbacks and their source pins must be chosen by the trusted host policy, never
loaded from candidate code or a candidate-supplied recipe. Serialized receipts are
audit output, not bearer permissions accepted from an untrusted caller.
"""
from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text as module_text


@dataclass(frozen=True)
class ArgumentIdentityStage:
    apply: Callable
    reparse_before: bool = False


@dataclass(frozen=True)
class ArgumentIdentityBridge:
    source_sha256: str
    normalized_sha256: str
    entry: str
    argument_types: tuple[str, ...]
    stages: tuple[tuple[str, bool], ...]
    source_pins: tuple[tuple[str, str], ...]

    def to_evidence(self) -> dict:
        return {"schema": "entry_argument_identity_bridge_v1",
                "source_sha256": self.source_sha256,
                "normalized_sha256": self.normalized_sha256, "entry": self.entry,
                "argument_types": list(self.argument_types),
                "argument_index_map": list(range(len(self.argument_types))),
                "stages": [{"callable": name, "reparse_before": parse} for name, parse in self.stages],
                "source_pins": dict(self.source_pins),
                "scope": "entry argument identity/order/type only; not normalization numerical equivalence"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _arguments(module, entry: str):
    functions = [op for op in module.body.block.ops
                 if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("argument identity requires one single-block source entry")
    return tuple(functions[0].body.block.args)


def replay_argument_identity(*, raw_text: str, source_sha256: str,
                            normalized_sha256: str, entry: str,
                            stages: Sequence[ArgumentIdentityStage],
                            source_pins: Mapping[Path, str]) -> ArgumentIdentityBridge:
    """Replay host-selected passes and reject replacement, permutation or retyping.

    Explicit print/parse boundaries preserve ordered typed arguments by syntax;
    every transform between those boundaries must preserve the actual argument
    objects. Both endpoint byte identities are checked, not inferred from shape.
    No model arithmetic is evaluated and no candidate is imported by this helper.
    """
    if _sha(raw_text) != source_sha256:
        raise ValueError("raw source differs from the host-pinned capture")
    pins = {Path(path).resolve(): digest for path, digest in source_pins.items()}
    own_source = Path(__file__).resolve()
    pins.setdefault(own_source, hashlib.sha256(own_source.read_bytes()).hexdigest())

    def verify_pins():
        for path, expected in pins.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("normalization implementation differs from host source pins")

    verify_pins()
    names = []
    for stage in stages:
        if not isinstance(stage, ArgumentIdentityStage) or type(stage.reparse_before) is not bool:
            raise ValueError("normalization stages must be explicit host-selected operations")
        implementation = inspect.getsourcefile(stage.apply)
        if implementation is None or Path(implementation).resolve() not in pins:
            raise ValueError("normalization callback has no pinned host implementation")
        names.append((f"{stage.apply.__module__}.{stage.apply.__qualname__}", stage.reparse_before))
    module = parse_mlir_text(raw_text)
    original = _arguments(module, entry)
    types = tuple(str(arg.type) for arg in original)
    for stage in stages:
        if stage.reparse_before:
            serialized = module_text(module)
            module = parse_mlir_text(serialized)
            reparsed = _arguments(module, entry)
            if tuple(str(arg.type) for arg in reparsed) != types:
                raise ValueError("serialization changed the ordered entry argument types")
        before = _arguments(module, entry)
        stage.apply(module)
        after = _arguments(module, entry)
        if (len(before) != len(after) or any(left is not right for left, right in zip(before, after))
                or tuple(str(arg.type) for arg in after) != types):
            raise ValueError("normalization changed entry argument identity, order or type")
    normalized = module_text(module)
    if _sha(normalized) != normalized_sha256:
        raise ValueError("normalization replay does not reproduce the exact compiled source")
    verify_pins()
    return ArgumentIdentityBridge(source_sha256, normalized_sha256, entry, types,
                                  tuple(names), tuple(sorted((str(path), digest) for path, digest in pins.items())))
