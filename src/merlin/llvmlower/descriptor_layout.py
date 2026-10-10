"""Same object-driver descriptor offsets and conditional format-only packing.

Actual compiler-folded GEP constants establish this selected driver's layout.
No separately versioned LLVM library interprets its omitted layout defaults.
This is neither a body theorem nor physical allocation/runtime authority.
"""

import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json

from .compiled_layout_query import query_object_layout, select_object
from .descriptor_object import SECTION, layout_words
from .descriptor_wrapper import DescriptorWrapperObservation, read, require
from .layout_observation import _deadline, _remaining, _run
from .llvm_dialect_product import verify_llvm_dialect_product
from .target_data_layout import parse as parse_layout

_ISSUED = weakref.WeakKeyDictionary()


def _expression(typ, indices):
    # LLVM's literal struct index is i32; sequential pointer/array indices
    # below are query constants, not a descriptor or target index-width fact.
    typed = ", ".join(f"{'i32' if ordinal == 1 else 'i64'} {value}" for ordinal, value in enumerate(indices))
    return "ptrtoint (ptr getelementptr (" + typ + ", ptr null, " + typed + ") to i64)"


def _query(rows, layout, triple):
    require(
        all(type(value) is str and value and not any(c in value for c in '\n\r"\\') for value in (layout, triple)),
        "descriptor query needs complete actual compiler layout/triple strings",
    )
    values = [_expression("ptr", (1,))]
    for row in rows:
        typ = row["aggregate_type"]
        values.extend(
            (_expression(typ, (1,)), _expression("{ i8, " + typ + " }", (0, 1)), _expression(row["element_type"], (1,)))
        )
        for field in row["fields"]:
            values.extend(
                (
                    _expression(typ, (0, *field["path"])),
                    _expression(field["type"], (1,)),
                    _expression("{ i8, " + field["type"] + " }", (0, 1)),
                )
            )
    body = ",\n  ".join("i64 " + value for value in values)
    source = (
        f'target datalayout = "{layout}"\ntarget triple = "{triple}"\n'
        f'@merlin_descriptor_layout = constant [{len(values)} x i64] [\n  {body}\n], section "{SECTION}"\n'
    )
    return source, len(values)


def _describe(rows, words):
    pointer_bytes, position, result = words[0], 1, []
    require(pointer_bytes > 0, "compiler returned an empty pointer representation")
    for original in rows:
        size, alignment, element_bytes = words[position : position + 3]
        position += 3
        require(
            size > 0 and alignment > 0 and not alignment & (alignment - 1) and element_bytes > 0,
            "compiler returned unsupported descriptor size/alignment/element storage",
        )
        row = {**original, "descriptor_bytes": size, "descriptor_alignment": alignment, "element_bytes": element_bytes}
        selected, intervals = [], []
        for field in original["fields"]:
            offset, width, field_alignment = words[position : position + 3]
            position += 3
            require(
                width > 0
                and field_alignment > 0
                and not field_alignment & (field_alignment - 1)
                and offset % field_alignment == 0
                and offset + width <= size,
                "compiler descriptor field is outside/alignment-incompatible with its aggregate",
            )
            require(
                (field["type"] != "ptr" or width == pointer_bytes)
                and (field["type"] == "ptr" or width * 8 == int(field["type"][1:])),
                "descriptor field has unsupported padded/non-byte scalar representation",
            )
            require(
                all(offset + width <= lo or offset >= hi for lo, hi in intervals), "compiler descriptor fields overlap"
            )
            intervals.append((offset, offset + width))
            selected.append({**field, "offset_bytes": offset, "storage_bytes": width, "alignment": field_alignment})
        row["fields"] = selected
        result.append(row)
    require(position == len(words), "compiler layout omits/changes its full constant roster")
    return {"pointer_bytes": pointer_bytes, "slots": result}


def _pin(path, maximum):
    path = Path(path).absolute()
    return str(path), hashlib.sha256(read(path, maximum)).hexdigest()


@dataclass(frozen=True, eq=False)
class DescriptorLayoutObservation:
    wrapper: DescriptorWrapperObservation
    root: Path
    pins: tuple[tuple[str, str], ...]
    records: tuple[tuple[str, str], ...]
    observation: bytes

    def record(self):
        return {
            "schema": "merlin.descriptor_layout_observation.v1",
            "root": str(self.root),
            "observation": json.loads(self.observation),
            "pins": self.pins,
            "records": self.records,
            "scope": "same selected object-driver layout and wrapper transport; no body/storage/runtime authority",
        }

    def verify(self):
        require(
            _ISSUED.get(self) == canonical_json(self.record()),
            "descriptor layout needs its live actual compiler observation",
        )
        self.wrapper.verify()
        for path, digest in (*self.pins, *self.records):
            require(
                _pin(
                    path,
                    max(
                        self.wrapper.limits.source_bytes,
                        self.wrapper.limits.receipt_bytes,
                        self.wrapper.limits.object_bytes,
                    ),
                )
                == (path, digest),
                "descriptor layout source/product/tool record changed",
            )
        actual = tuple(
            _pin(path, self.wrapper.limits.receipt_bytes) for path in sorted(self.root.rglob("invocation.json"))
        )
        require(actual == self.records, "descriptor layout lost an actual compiler invocation")
        for path, _ in self.records:
            I.verify(Path(path))
        return self.record()


def observe_descriptor_layout(*, wrapper, object_record, environment, output_root, timeout_s=120):
    """Query the same actual object executable/options, not another LLVM API."""
    require(type(wrapper) is DescriptorWrapperObservation, "descriptor layout needs an exact live wrapper observation")
    wrapper.verify()
    require(type(environment) is dict and environment, "descriptor layout requires the original explicit environment")
    environment = dict(environment)
    record = Path(object_record).absolute()
    document, original, obj, driver, options = select_object(record, environment=environment)
    require(driver == "llvm", "this descriptor layout query supports the selected ordinary LLVM object driver only")
    product = verify_llvm_dialect_product(wrapper.receipt)
    require(
        hashlib.sha256(read(original, wrapper.limits.source_bytes)).hexdigest()
        == product["translated_llvm_ir"]["sha256"],
        "descriptor wrapper translation differs from the actual object compiler input",
    )
    root = Path(output_root).absolute()
    require(root.resolve() == root and not root.exists(), "descriptor layout needs a fresh direct output owner")
    deadline = _deadline(timeout_s)
    root.mkdir(parents=True, mode=0o700)
    selected, compiler, dependencies = query_object_layout(
        document=document,
        original=original,
        obj=obj,
        record=record,
        driver=driver,
        options=options,
        root=root,
        run=_run,
        deadline=deadline,
        environment=environment,
        max_observation_bytes=wrapper.limits.source_bytes,
    )
    text = read(selected, wrapper.limits.source_bytes).decode()
    triples = [line.strip() for line in text.splitlines() if line.strip().startswith('target triple = "')]
    require(len(triples) == 1 and triples[0].endswith('"'), "actual object compiler omitted its unique triple")
    layout, triple = parse_layout(text), triples[0][len('target triple = "') : -1]
    rows = wrapper.record()["observation"]["slots"]
    source, count = _query(rows, layout, triple)
    query, output = root / "descriptor-query.ll", root / "descriptor-query.o"
    require(len(source.encode()) <= wrapper.limits.source_bytes, "descriptor query exceeds its selected source bound")
    query.write_text(source)
    _run(
        [str(compiler), *options, str(query), "-o", str(output)],
        root=root,
        stage="descriptor_selected_object_layout",
        deadline=deadline,
        inputs=(query,),
        outputs=(output,),
        dependencies=(
            record,
            obj,
            original,
            selected,
            Path(__file__),
            Path(__file__).with_name("descriptor_object.py"),
            *dependencies,
        ),
        env=environment,
        cwd=document["cwd"],
    )
    words, byte_order = layout_words(read(output, wrapper.limits.object_bytes), count=count)
    require(
        layout.split("-", 1)[0] == ("e" if byte_order == "little" else "E"),
        "selected compiler layout and actual constant object disagree on byte order",
    )
    values = {
        **_describe(rows, words),
        "byte_order": byte_order,
        "data_layout": layout,
        "target_triple": triple,
        "word_count": count,
        "object_record": str(record),
        "original_wrapper": wrapper.record(),
    }
    I.require_environment(record, environment=environment)
    pins = tuple(
        _pin(path, max(wrapper.limits.source_bytes, wrapper.limits.receipt_bytes, wrapper.limits.object_bytes))
        for path in (
            record,
            original,
            obj,
            selected,
            root / "selected.mir",
            query,
            output,
            Path(__file__),
            Path(__file__).with_name("descriptor_object.py"),
        )
    )
    records = tuple(_pin(path, wrapper.limits.receipt_bytes) for path in sorted(root.rglob("invocation.json")))
    observation = DescriptorLayoutObservation(wrapper, root, pins, records, canonical_json(values))
    _ISSUED[observation] = canonical_json(observation.record())
    observation.verify()
    _remaining(deadline)
    return observation


def pack_descriptor(*, observation, ordinal, allocated_address, aligned_address):
    """Produce only descriptor bytes under explicit caller storage preconditions.

    The caller still owns real accessible allocation, lifetimes, cross-slot
    aliasing, selected execution ABI and implementation indexing/semantics.
    No pointed-to data is inspected or changed; signed zero/wide integers are
    neither converted nor reconstructed by this format-only bridge.
    """
    require(type(observation) is DescriptorLayoutObservation, "packing needs an actual selected descriptor layout")
    observation.verify()
    data = observation.record()["observation"]
    require(type(ordinal) is int and 0 <= ordinal < len(data["slots"]), "descriptor packing omits an original slot")
    row, declaration = data["slots"][ordinal], observation.wrapper.source.storage[ordinal]
    require(
        row["descriptor_bytes"] <= observation.wrapper.limits.object_bytes,
        "descriptor packing exceeds its explicit byte budget",
    )
    pointer_limit = 1 << (8 * data["pointer_bytes"])
    require(
        all(type(address) is int and 0 < address < pointer_limit for address in (allocated_address, aligned_address)),
        "descriptor caller addresses exceed the actually selected pointer representation",
    )
    require(
        aligned_address % declaration.alignment == 0
        and aligned_address + declaration.capacity_elements * row["element_bytes"] <= pointer_limit,
        "declared aligned object violates its alignment/complete byte extent",
    )
    values = {"allocated": allocated_address, "aligned": aligned_address, "offset": declaration.offset_elements}
    values.update({f"size_{axis}": size for axis, size in enumerate(row["shape"])})
    values.update({f"stride_{axis}": stride for axis, stride in enumerate(declaration.element_strides)})
    payload = bytearray(row["descriptor_bytes"])
    for field in row["fields"]:
        value, width = values[field["role"]], field["storage_bytes"]
        require(
            field["type"] == "ptr" or value < 1 << (row["index_bits"] - 1),
            "descriptor shape/stride/offset exceeds its actual signed index field",
        )
        start = field["offset_bytes"]
        payload[start : start + width] = value.to_bytes(width, data["byte_order"])
    return bytes(payload)


def require_descriptor_bytes(*, payload, observation, ordinal, allocated_address, aligned_address):
    expected = pack_descriptor(
        observation=observation, ordinal=ordinal, allocated_address=allocated_address, aligned_address=aligned_address
    )
    require(
        type(payload) is bytes and payload == expected,
        "descriptor bytes do not cover the exact original storage fields",
    )
