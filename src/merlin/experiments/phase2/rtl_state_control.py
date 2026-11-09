"""Render a private opaque-state native control, never a hardware simulator.

The selected engine produces the state layout and compiled evaluation function.
This owner only writes declared input bytes, calls that function and reads every
declared output. It supplies no clock, reset, memory, program loading or timing
semantics. Its source and exact produced layout must join the ordinary RTL probe
invocations; rendering or executing alone grants no target authority.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import StageGateError


def _identifier(value):
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or not (value[0].isalpha() or value[0] == "_")
        or any(not (char.isalnum() or char == "_") for char in value)
    ):
        raise StageGateError("native control requires a plain C identifier")
    return value


def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise StageGateError("native control integer is outside its bounded declaration")
    return value


def _document(payload):
    if type(payload) is not bytes or len(payload) > 2**20:
        raise StageGateError("native control document is unavailable or too large")
    try:

        def closed_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate native declaration field")
                result[key] = value
            return result

        return json.loads(payload, object_pairs_hook=closed_object)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise StageGateError("native control document is not bounded JSON data") from error


@dataclass(frozen=True)
class StatePort:
    name: str
    offset: int
    num_bits: int
    role: str

    @property
    def num_bytes(self):
        return (self.num_bits + 7) // 8


def _layout(payload, model):
    """Read the selected native producer's scalar I/O layout, without defaults."""
    document = _document(payload)
    if type(document) is not list or not 1 <= len(document) <= 128:
        raise StageGateError("native state layout requires an explicit model roster")
    names = []
    selected = None
    for item in document:
        if type(item) is not dict or set(item) != {"name", "numStateBytes", "states"}:
            raise StageGateError("native state layout has an unsupported model declaration")
        name = _identifier(item["name"])
        names.append(name)
        if name == model:
            selected = item
    if len(set(names)) != len(names) or selected is None:
        raise StageGateError("native state model selection is missing or ambiguous")
    size = _integer(selected["numStateBytes"], 1, 2**20)
    states = selected["states"]
    if type(states) is not list or not 1 <= len(states) <= 4096:
        raise StageGateError("native state member roster is unavailable or too large")
    ports = []
    occupied = set()
    for state in states:
        if type(state) is not dict or type(state.get("type")) is not str:
            raise StageGateError("native state member declaration is unavailable")
        if state["type"] not in ("input", "output", "register", "wire", "memory"):
            raise StageGateError("native state member role is unsupported")
        if state["type"] not in ("input", "output"):
            # Internal state belongs exclusively to the compiled native engine.
            continue
        if set(state) != {"name", "offset", "numBits", "type"}:
            raise StageGateError("native scalar I/O has an unsupported layout")
        port = StatePort(
            _identifier(state["name"]),
            _integer(state["offset"], 0, size - 1),
            _integer(state["numBits"], 1, 64),
            state["type"],
        )
        extent = set(range(port.offset, port.offset + port.num_bytes))
        if port.offset + port.num_bytes > size or extent & occupied:
            raise StageGateError("native scalar I/O layout exceeds or overlaps its state storage")
        occupied.update(extent)
        ports.append(port)
    if not ports or len({port.name for port in ports}) != len(ports):
        raise StageGateError("native scalar I/O membership is empty or duplicated")
    return size, tuple(ports)


def _stimuli(payload, ports):
    document = _document(payload)
    fields = {"byte_order", "inputs", "outputs", "samples"}
    if type(document) is not dict or set(document) != fields or document["byte_order"] not in ("little", "big"):
        raise StageGateError("native control requires its complete original stimulus declaration")
    rosters = {}
    for role in ("input", "output"):
        names = document[role + "s"]
        if (
            type(names) is not list
            or not names
            or len(names) > 128
            or any(type(name) is not str for name in names)
            or len(set(names)) != len(names)
            or set(names) != {port.name for port in ports if port.role == role}
        ):
            raise StageGateError("native control must cover every actual input and output port")
        if role == "output" and "sample" in names:
            raise StageGateError("native output name overlaps the observation index")
        by_name = {port.name: port for port in ports if port.role == role}
        rosters[role] = tuple(by_name[name] for name in names)
    samples = document["samples"]
    if type(samples) is not list or not 1 <= len(samples) <= 4096:
        raise StageGateError("native control samples are unavailable or too large")
    total = 0
    for index, sample in enumerate(samples):
        if (
            type(sample) is not dict
            or set(sample) != {"id", "values", "evaluations", "observe"}
            or type(sample["id"]) is not int
            or sample["id"] != index
            or type(sample["observe"]) is not bool
            or type(sample["values"]) is not list
            or len(sample["values"]) != len(rosters["input"])
        ):
            raise StageGateError("native control original sample membership is incomplete")
        total += _integer(sample["evaluations"], 1, 32)
        for value, port in zip(sample["values"], rosters["input"], strict=True):
            _integer(value, 0, (1 << port.num_bits) - 1)
    if total > 65536 or not any(sample["observe"] for sample in samples):
        raise StageGateError("native control evaluation budget or original outputs are unavailable")
    return document["byte_order"], rosters["input"], rosters["output"], samples


def render_opaque_state_control(*, layout_bytes, stimuli_bytes, model, entrypoint):
    """Render transport only. The actual producer/execution joins remain required.

    Native allocation follows C's allocator alignment; no hardware alignment is
    inferred from offsets or widths. Arbitrary byte layouts/functions require
    their own source and ABI qualification before any target execution use.
    """
    _identifier(model)
    _identifier(entrypoint)
    size, ports = _layout(layout_bytes, model)
    byte_order, inputs, outputs, samples = _stimuli(stimuli_bytes, ports)
    assignments = []
    for sample in samples:
        statements = [
            f"put(state, {port.offset}, {port.num_bytes}, UINT64_C({value}));"
            for port, value in zip(inputs, sample["values"], strict=True)
        ]
        statements.append(f"for(unsigned n=0;n<{sample['evaluations']};++n) {entrypoint}(state);")
        if sample["observe"]:
            statements.append(f'printf("{sample["id"]}");')
            statements.extend(
                f'printf(",%" PRIu64, take(state, {port.offset}, {port.num_bytes})'
                f" & UINT64_C({(1 << port.num_bits) - 1}));"
                for port in outputs
            )
            statements.append("putchar(10);")
        assignments.extend(statements)
    shift = "j" if byte_order == "little" else "bytes-1-j"
    body = "\n  ".join(assignments)
    roster = ",".join(("sample", *(port.name for port in outputs)))
    return f'''#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void {entrypoint}(void *state);
static void put(uint8_t *state, unsigned offset, unsigned bytes, uint64_t value) {{
  for(unsigned j=0;j<bytes;++j) state[offset+j]=(uint8_t)(value >> (8*({shift})));
}}
static uint64_t take(const uint8_t *state, unsigned offset, unsigned bytes) {{
  uint64_t value=0;
  for(unsigned j=0;j<bytes;++j) value|=((uint64_t)state[offset+j]) << (8*({shift}));
  return value;
}}
int main(void) {{
  uint8_t *state=calloc({size}, 1);
  if(!state) return 1;
  puts("{roster}");
  {body}
  free(state);
  return ferror(stdout) ? 2 : 0;
}}
'''.encode("ascii")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise StageGateError("native control requires a fresh generated source destination")
    source = render_opaque_state_control(
        layout_bytes=args.layout.read_bytes(),
        stimuli_bytes=args.stimuli.read_bytes(),
        model=args.model,
        entrypoint=args.entrypoint,
    )
    args.output.write_bytes(source)


if __name__ == "__main__":
    main()
