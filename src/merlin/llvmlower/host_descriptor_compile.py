"""Explicit original ranked-descriptor selection for ordinary host compilation.

This joins an unchanged typed source and complete caller storage declarations
to the actual translation, object and linked image. It establishes transport
custody only; implementation indexing, numerical and physical effects remain
unproved. No machine-width, storage or environment default is supplied.
"""

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

from xdsl.dialects import builtin, func

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads
from merlin.xdsl_dialects._common import text

from .descriptor_contract import DescriptorLimits, OriginalDescriptorSource
from .descriptor_layout import observe_descriptor_layout
from .descriptor_wrapper import observe_descriptor_wrapper, original_types, parse, read, require


@dataclass(frozen=True)
class HostDescriptorSelection:
    """Caller-owned original choices, not source or runtime admission authority."""

    source: OriginalDescriptorSource
    limits: DescriptorLimits
    environment: tuple[tuple[str, str], ...]
    query_timeout_s: float

    def verify(self):
        require(
            type(self.source) is OriginalDescriptorSource and type(self.limits) is DescriptorLimits,
            "host descriptor selection needs complete original source/storage and limits",
        )
        require(
            type(self.environment) is tuple
            and bool(self.environment)
            and all(
                type(row) is tuple and len(row) == 2 and all(type(part) is str for part in row)
                for row in self.environment
            )
            and len(dict(self.environment)) == len(self.environment),
            "host descriptor selection needs an exact explicit environment mapping",
        )
        require(
            type(self.query_timeout_s) in {int, float}
            and math.isfinite(self.query_timeout_s)
            and 0 < self.query_timeout_s <= 600,
            "host descriptor layout query needs an explicit bounded timeout",
        )
        require(self.source.slot_aliasing == "disjoint", "host descriptor execution requires explicit disjoint slots")
        original_types(self.source, self.limits)
        return {
            "source": self.source.record(self.limits),
            "limits": self.limits.record(),
            "environment": I.environment_identity(dict(self.environment)),
            "query_timeout_s": self.query_timeout_s,
        }

    def require_module(self, module):
        self.verify()
        original = parse(read(self.source.path, self.limits.source_bytes), self.limits, emitted=False)
        # Compare complete generic serialization, including all regions,
        # attributes, properties and ordered returns; names are not proofs.
        candidate = parse(text(module, generic=True).encode(), self.limits, emitted=False)
        require(
            text(original, generic=True) == text(candidate, generic=True),
            "host kernel differs from its original source",
        )


def _prepared(selection, product):
    source = selection.source
    original = parse(read(source.path, selection.limits.source_bytes), selection.limits, emitted=False)
    # Ordinary preprocessing adds only this explicit public calling marker in
    # the supported route. Other preprocessing changes require a separate proof.
    for op in original.body.block.ops:
        if type(op) is func.FuncOp and op.sym_name.data == source.entry_symbol:
            marker = op.attributes.get("llvm.emit_c_interface")
            require(marker is None or type(marker) is builtin.UnitAttr, "original C-interface marker is unsupported")
            op.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    prepared_path = Path(product["source"]["path"])
    raw = read(prepared_path, selection.limits.source_bytes)
    require(hashlib.sha256(raw).hexdigest() == product["source"]["sha256"], "prepared source bytes changed")
    prepared = parse(raw, selection.limits, emitted=False)
    require(
        text(original, generic=True) == text(prepared, generic=True),
        "host source preprocessing changed the original body/ordered ABI; correspondence unavailable",
    )
    return replace(source, path=prepared_path, sha256=product["source"]["sha256"])


def _pin(path, limits):
    path = Path(path).absolute()
    raw = read(path, max(limits.source_bytes, limits.receipt_bytes, limits.object_bytes))
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def join_host_image(*, selection, build_root, product, image):
    """Require actual original LLVM→object and runtime-object→image records."""
    selection.verify()
    environment = dict(selection.environment)
    records = []
    for path in sorted(Path(build_root).rglob("invocation.json")):
        # A query/call record cannot substitute for an original build boundary.
        row = loads(read(path, selection.limits.receipt_bytes))
        if row.get("stage") in {"object", "runtime_object", "link"}:
            records.append((path, I.require_environment(path, environment=environment)))
    objects = [(path, row) for path, row in records if row["stage"] == "object"]
    runtime = [(path, row) for path, row in records if row["stage"] == "runtime_object"]
    require(
        len(objects) == len(runtime) == 1 and len(records) == 3,
        "host descriptor image lacks its exact build boundaries",
    )
    object_path, object_row = objects[0]
    raw_translation = product["translated_llvm_ir"]
    require(
        len(object_row["inputs"]) == len(object_row["outputs"]) == 1
        and object_row["inputs"][0]["sha256"] == raw_translation["sha256"]
        and _pin(raw_translation["path"], selection.limits)["sha256"] == raw_translation["sha256"]
        and read(object_row["inputs"][0]["path"], selection.limits.source_bytes)
        == read(raw_translation["path"], selection.limits.source_bytes),
        "host descriptor object input differs from the complete retained translation",
    )
    runtime_path, runtime_row = runtime[0]
    require(
        len(runtime_row["inputs"]) == len(runtime_row["outputs"]) == 1,
        "host descriptor image lacks its complete runtime-source/object binding",
    )
    expected = sorted((object_row["outputs"][0], runtime_row["outputs"][0]), key=lambda row: row["path"])
    linked = [(path, row) for path, row in records if row["stage"] == "link"]
    require(
        len(linked) == 1
        and linked[0][1]["inputs"] == expected
        and linked[0][1]["outputs"] == [_pin(image, selection.limits)],
        "host descriptor layout object is disconnected from the loaded image",
    )
    require(
        all(row["kind"] == "subprocess" for _, row in records),
        "host descriptor build requires actual subprocess products",
    )
    return object_path, (object_path, runtime_path, linked[0][0])


def observe_host_descriptor_transport(*, selection, result):
    """Produce a live transport for the ordinary caller's own compiled image."""
    from .host_descriptor_call import HostDescriptorTransport, _retain_transport
    from .llvm_dialect_product import verify_llvm_dialect_product

    require(type(selection) is HostDescriptorSelection, "host descriptor selection is unsupported")
    selection.verify()
    root, image = Path(result.workdir).absolute(), Path(result.host_so).absolute()
    require(image.parent == root, "host descriptor image is outside its ordinary compilation owner")
    receipt = Path(result.stats["llvm_dialect_product"]["path"])
    read(receipt, selection.limits.receipt_bytes)
    product = verify_llvm_dialect_product(receipt)
    prepared = _prepared(selection, product)
    object_record, records = join_host_image(selection=selection, build_root=root, product=product, image=image)
    wrapper = observe_descriptor_wrapper(source=prepared, limits=selection.limits, llvm_product_receipt=receipt)
    observation = observe_descriptor_layout(
        wrapper=wrapper,
        object_record=object_record,
        environment=dict(selection.environment),
        output_root=root / "host-descriptor-layout",
        timeout_s=selection.query_timeout_s,
    )
    paths = (selection.source.path, prepared.path, receipt, image, Path(__file__))
    pins = tuple((row["path"], row["sha256"]) for row in (_pin(path, selection.limits) for path in paths))
    selected_records = tuple((row["path"], row["sha256"]) for row in (_pin(path, selection.limits) for path in records))
    transport = HostDescriptorTransport(
        selection, observation, root, image, pins, selected_records, canonical_json(selection.verify())
    )
    _retain_transport(transport)
    transport.verify()
    return transport
