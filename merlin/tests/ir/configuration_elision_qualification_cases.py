"""Independent, target-neutral configuration-elision fixtures.

Synthetic ISA declarations exercise the real selected-contract and IR adapter.
Their parser-required ``reviewed`` status is test data, never human approval.
No source/model-to-hardware agreement is asserted by these fixtures.
"""

import hashlib
import json
from dataclasses import dataclass

TARGET = "configuration_fixture"
NAMESPACE = "configuration_fixture_isa"
QUALIFICATION = "synthetic-unapproved-effect-premise-v1"
INSTRUCTIONS = ("set", "pair", "work", "reset", "pulse", "unordered", "unknown")


def declarations(bits=32):
    """Pins are derived from a complete synthetic facts document, not a target."""
    facts = {
        "facts": {
            "target": TARGET,
            "word_bits": 16 + 2 * bits,
            "capacity": 4,
            "widths": [8, 8, bits, bits],
            "offsets": [0, 8, 16, 16 + bits],
            "codes": list(range(len(INSTRUCTIONS))),
        }
    }
    facts_bytes = (json.dumps(facts, sort_keys=True, indent=2) + "\n").encode()

    def fact(*parts):
        return {"fact": ["facts", *parts]}

    def entry(index, value=None):
        item = {"resource": "configuration", "index": index}
        if value is not None:
            item["value"] = value
        return item

    def constant(value):
        return {"constant": value}

    def argument(name):
        return {"argument": name}

    rows = {
        "set": dict(
            reads=[], writes=[entry(argument("slot"), argument("a"))], clobbers=[], ordered=True, pure_write=True
        ),
        "pair": dict(
            reads=[],
            writes=[entry(constant(0), argument("a")), entry(constant(1), argument("b"))],
            clobbers=[],
            ordered=True,
            pure_write=True,
        ),
        "work": dict(
            reads=[entry(constant(0), constant(3)), entry(constant(1), constant(5))],
            writes=[],
            clobbers=[],
            ordered=True,
            pure_write=False,
        ),
        "reset": dict(reads=[], writes=[], clobbers=[entry(argument("slot"))], ordered=True, pure_write=False),
        "pulse": dict(
            reads=[], writes=[entry(argument("slot"), argument("a"))], clobbers=[], ordered=True, pure_write=False
        ),
        "unordered": dict(
            reads=[], writes=[entry(argument("slot"), argument("a"))], clobbers=[], ordered=False, pure_write=False
        ),
    }
    instructions = []
    for code, name in enumerate(INSTRUCTIONS):
        operands = [
            dict(name=key, type=dict(kind="integer", bits=fact("widths", index), signed=False), unit="integer")
            for index, key in enumerate(("opcode", "slot", "a", "b"))
            if index
        ]
        encoding = [
            dict(
                offset=fact("offsets", index),
                width=fact("widths", index),
                **({"fixed": fact("codes", code)} if index == 0 else {"argument": key}),
            )
            for index, key in enumerate(("opcode", "slot", "a", "b"))
        ]
        instruction = dict(name=name, operands=operands, results=[], immediates=[], encoding=encoding)
        if name != "unknown":
            instruction["effects"] = dict(
                reads=["configuration", "input"] if name == "work" else [],
                writes=["output"] if name == "work" else ["configuration"],
            )
            if name == "pulse":
                instruction["effects"]["writes"].append("events")
        instructions.append(instruction)
    software = dict(
        schema="merlin.software_spec.v1",
        target=TARGET,
        status="reviewed",
        isa=dict(
            schema="merlin.isa_spec.v1",
            namespace=NAMESPACE,
            facts_sha256=hashlib.sha256(facts_bytes).hexdigest(),
            word_bits=fact("word_bits"),
            instructions=instructions,
            resources=[
                dict(name="configuration", kind="state", unit="fields", capacity=fact("capacity")),
                *[dict(name=name, kind="memory", unit="bytes") for name in ("input", "output", "events")],
            ],
        ),
        scheduling=dict(
            schema="merlin.schedule_effects.v1",
            numerical_policy="no-numerical-claim",
            instruction_byteorder="little",
            configuration=rows,
        ),
    )
    return facts_bytes, (json.dumps(software, sort_keys=True, indent=2) + "\n").encode()


def call(name, slot=0, a=0, b=0, *, bits=32):
    return f'"{NAMESPACE}.{name}"(%slot{slot}, %value{a}, %value{b}) : (i8, i{bits}, i{bits}) -> ()'


def function(body, *, bits=32, name="run", returns="void"):
    high = (1 << bits) - 1
    constants = [f"%slot{i} = arith.constant {i} : i8" for i in range(4)]
    constants += [f"%value{i} = arith.constant {i} : i{bits}" for i in (0, 3, 5, 7, high)]
    constants.append(f"%equal3 = arith.constant 3 : i{bits}")
    ending = "func.return" if returns == "void" else f"func.return %value3 : i{bits}"
    signature = f"func.func @{name}(%input: i{bits})" + ("" if returns == "void" else f" -> i{bits}")
    return signature + " {\n" + "\n".join("  " + line for line in [*constants, *body, ending]) + "\n}"


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    removed: int
    bits: int = 32
    purpose: str = ""


def cases():
    def make(name, body, removed, purpose, bits=32):
        return Case(name, "module {\n" + function(body, bits=bits) + "\n}\n", removed, bits, purpose)

    pair, work = call("pair", a=3, b=5), call("work")
    set0, set1 = call("set", a=3), call("set", slot=1, a=5)
    pair7 = call("pair", a=3, b=7)
    rows = [
        make(
            "single_fields",
            [set0, set1, work, set0, set1, work],
            2,
            "both repeated writes deleted; observer order retained",
        ),
        make("multifield", [pair, work, pair, work, pair], 2, "whole multi-field overwrite redundant"),
        make(
            "partially_changed_value",
            [pair, pair7, pair7, pair, work, pair],
            2,
            "one changed field retains entire setter",
        ),
        make(
            "partial_clobber",
            [pair, call("reset"), set1, set0, set0, work],
            2,
            "clobbered field forgotten; other field remains known",
        ),
        make("unknown_effect", [pair, call("unknown"), pair, pair, work], 1, "unknown retains and clears knowledge"),
        make(
            "unordered_effect", [pair, call("unordered", a=7), pair, pair, work], 1, "unknown ordering clears knowledge"
        ),
        make(
            "pulse_retained",
            [pair, call("pulse", a=3), call("pulse", a=3), set0, work],
            1,
            "both side-effecting pulses remain",
        ),
        make(
            "ssa_results",
            [
                pair,
                '%first = "fixture.token"() : () -> !fixture.token',
                '%second = "fixture.token"() : () -> !fixture.token',
                '"fixture.consume"(%first, %second) : (!fixture.token, !fixture.token) -> ()',
                pair,
                pair,
                work,
            ],
            1,
            "opaque result and token operations retained",
        ),
    ]
    for bits in (32, 64):
        pair_b, work_b = call("pair", a=3, b=5, bits=bits), call("work", bits=bits)
        high = call("set", slot=2, a=(1 << bits) - 1, bits=bits)
        rows.append(
            make(
                f"unsigned_i{bits}_bits",
                [pair_b, high, high, work_b],
                1,
                "unsigned all-one integer encoding decoded from actual constant",
                bits,
            )
        )
    return tuple(rows)


def boundary_cases():
    pair = call("pair", a=3, b=5)
    return {
        "independent_invocations": "module {\n"
        + function([pair, pair], name="first")
        + "\n"
        + function([pair, pair], name="second")
        + "\n}\n",
        "unchanged_helper": "module {\n"
        + function([pair, pair])
        + "\n"
        + function([], name="untouched", returns="scalar")
        + "\n}\n",
        "dynamic_operand": "module {\n"
        + function([pair, call("set", a=3).replace("%value3", "%input"), pair, pair])
        + "\n}\n",
        "metadata_barrier": "module {\n"
        + function([pair, pair.replace(" : (", " {extra = unit} : ("), pair, pair])
        + "\n}\n",
    }
