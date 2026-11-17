"""Lower returned tensor descriptors without selecting or linking an executor.

The caller supplies its selected LLVM layout/index contract. This module
retains the ordinary ownership/lifetime gates and checked allocation products;
it supplies no target observation, allocator context or execution verdict.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .owned_result_abi import ScalarSlot, TensorSlot, bridge, pipeline, prepare
from .target_data_layout import default_index_bits, parse


def _pin(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return str(path), digest


@dataclass(frozen=True)
class OwnedResultLowering:
    source: Path
    prepared: Path
    llvm: Path
    bridge: Path
    inputs: tuple[TensorSlot | ScalarSlot, ...]
    outputs: tuple[TensorSlot, ...]
    entry: str
    data_layout: str
    index_bits: int
    pins: tuple[tuple[str, str], ...]

    def verify(self):
        if any(_pin(path) != (path, digest) for path, digest in self.pins):
            raise ValueError("owned result lowering source/product/implementation changed")
        prepared, inputs, outputs = prepare(self.source.read_text(), entry=self.entry)
        if (
            (inputs, outputs) != (self.inputs, self.outputs)
            or prepared != self.prepared.read_text()
            or bridge(inputs, outputs, entry=self.entry) != self.bridge.read_text()
            or parse(self.llvm.read_text()) != self.data_layout
            or self.index_bits != 64
        ):
            raise ValueError("owned result lowering changed its original types or selected ABI")


def lower_owned_results(
    source, *, workdir, data_layout, index_bits, entry="forward", features=frozenset(),
    timeout=60, max_bytes=64 << 20, allow_existing_workdir=False,
):
    """Emit checked LLVM and C bridge products; never link or read invocations.

    Layout/index values are explicit caller premises. A target must observe
    and bind its actual compiler contract before calling this API. The selected
    returned-descriptor bridge currently admits only 64-bit indices/pointers;
    the eventual C compiler independently checks that layout with assertions.
    """
    from .owned_result_allocations import OwnedAllocationPolicy
    from .pipeline import lower_to_llvm_ir

    if type(data_layout) is not str or not data_layout or any(c in data_layout for c in "\n\r\x00"):
        raise ValueError("owned result lowering requires an explicit selected data layout")
    if type(index_bits) is not int or index_bits != 64:
        raise ValueError("owned result descriptor bridge requires selected 64-bit indices")
    if any(field.partition(":")[0] in ("p", "p0") for field in data_layout.split("-")):
        if default_index_bits(data_layout) != index_bits:
            raise ValueError("owned result pointer/index layout disagrees with its bridge")
    policy = OwnedAllocationPolicy(max_bytes)
    policy.verify()
    prepared, inputs, outputs = prepare(source, entry=entry)
    if type(allow_existing_workdir) is not bool:
        raise ValueError("owned result build-directory selection requires an explicit bool")
    work = Path(workdir).resolve()
    if allow_existing_workdir and work.exists():
        if not work.is_dir() or any(work.iterdir()):
            raise ValueError("owned result compiler directory must be empty")
    else:
        work.mkdir(parents=True, exist_ok=False)
    original, prepared_path = work / "source.mlir", work / "prepared.mlir"
    bridge_path, llvm_path = work / "owned_result_bridge.c", work / "model.ll"
    original.write_text(source)
    prepared_path.write_text(prepared)
    bridge_path.write_text(bridge(inputs, outputs, entry=entry))
    lowered = lower_to_llvm_ir(
        prepared, workdir=work / "lowering", timeout=timeout,
        pipeline=pipeline(features=features, index_bits=index_bits), features=features,
        data_layout=data_layout, retain_llvm_dialect=True, owned_result_allocations=policy,
    )
    llvm_path.write_text(lowered)
    dependencies = (Path(__file__), Path(__file__).with_name("owned_result_abi.py"),
                    Path(__file__).with_name("owned_result_c.py"),
                    Path(__file__).with_name("dealloc_lifetime.py"), Path(__file__).with_name("dealloc_promotions.py"),
                    Path(__file__).with_name("owned_result_allocations.py"))
    result = OwnedResultLowering(
        original, prepared_path, llvm_path, bridge_path, inputs, outputs, entry,
        data_layout, index_bits,
        tuple(_pin(path) for path in (original, prepared_path, llvm_path, bridge_path, *dependencies)),
    )
    result.verify()
    return result
