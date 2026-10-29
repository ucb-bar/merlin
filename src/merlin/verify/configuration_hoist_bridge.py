"""Independent full-Kernel projection for the bounded partial-hoist candidate.

Phase1 only: no compiled query, live completion, model approval or production
authority. The selected source-owned parser remains a reviewed Python boundary.
"""

from __future__ import annotations

import json

REVISION = "e0484f4886ae0780d0325cdfa998842b61bb647c"


class Unsupported(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def node(value, kind, keys):
    if (
        type(value) is not dict
        or set(value) != {"class", "fields"}
        or value["class"] != "merlin.sched.ir.kernel." + kind
        or type(value["fields"]) is not dict
        or set(value["fields"]) != set(keys)
    ):
        raise Unsupported("unsupported complete new-producer " + kind + " schema")
    return value["fields"]


def sequence(value):
    if type(value) is not dict or set(value) != {"tuple"} or type(value["tuple"]) is not list:
        raise Unsupported("complete typed Kernel tuple required")
    return value["tuple"]


def constant(value):
    if (
        type(value) is not dict
        or set(value) != {"class", "fields"}
        or value["class"] != "merlin.sched.ir.expr.Const"
        or set(value["fields"]) != {"value"}
        or type(value["fields"]["value"]) is not int
    ):
        raise Unsupported("constant integer actual configuration operands required; IV-dependent setters excluded")
    return value["fields"]["value"]


CALL_KEYS = ("instr", "args", "unit", "produces", "awaits", "stages", "sets", "assumes")
LOOP_KEYS = ("var", "extent", "body", "iter_args", "yields", "results")


def call_shape(value):
    fields = node(value, "Call", CALL_KEYS)
    if (
        fields["unit"] is not None
        or fields["produces"] is not None
        or sequence(fields["awaits"])
        or sequence(fields["stages"])
    ):
        raise Unsupported("phase1 has no asynchronous/unit/completion/staging semantics")
    pairs = [sequence(pair) for pair in sequence(fields["args"])]
    if any(len(pair) != 2 or type(pair[0]) is not str for pair in pairs):
        raise Unsupported("actual Call argument bindings require complete ordered name/value pairs")
    if len({pair[0] for pair in pairs}) != len(pairs):
        raise Unsupported("duplicate actual argument bindings")
    return fields, pairs


def decode_effect(value, isa, declarations):
    fields, pairs = call_shape(value)
    declaration = declarations.get(fields["instr"])
    if declaration is None or sequence(fields["sets"]) or sequence(fields["assumes"]):
        return None
    instruction = next((row for row in isa["instructions"] if row["name"] == fields["instr"]), None)
    if instruction is None or instruction["results"]:
        return None
    arguments = {row["name"]: row for row in [*instruction["operands"], *instruction["immediates"]]}
    encoded = {row["argument"] for row in instruction["fields"] if row["argument"] is not None}
    values = {key: constant(item) for key, item in pairs}
    if set(values) != encoded or any(arguments[key]["type"]["kind"] != "integer" for key in values):
        raise Unsupported("actual bindings must exactly cover selected integer encoding arguments")
    for field in instruction["fields"]:
        key = field["argument"]
        if key is None:
            continue
        argument = arguments[key]
        width = min(field["width"], argument["type"]["bits"])
        signed = argument["type"]["signed"]
        lower = -(1 << (width - 1)) if signed else 0
        upper = (1 << (width - int(signed))) - 1
        if not lower <= values[key] <= upper or values[key] % argument["alignment"]:
            raise Unsupported("actual argument outside selected encoded range/alignment")
    resources = {row["name"]: row for row in isa["resources"]}

    def binding(item):
        if set(item) == {"constant"} and type(item["constant"]) is int:
            return item["constant"]
        if set(item) == {"argument"} and item["argument"] in values:
            return values[item["argument"]]
        raise Unsupported("unsupported selected field binding")

    result = {
        "reads": [],
        "writes": [],
        "clobbers": [],
        "ordered": declaration["ordered"],
        "pureWrite": declaration["pure_write"],
    }
    for kind in ("reads", "writes", "clobbers"):
        seen = set()
        for item in declaration[kind]:
            resource = resources[item["resource"]]
            index = binding(item["index"])
            if (
                resource["kind"] != "state"
                or resource["unit"] != "fields"
                or type(resource["capacity"]) is not int
                or not 0 <= index < resource["capacity"]
            ):
                raise Unsupported("exact nonaliasing finite state field identity required")
            key = item["resource"] + ":" + str(index)
            if key in seen:
                raise Unsupported("decoded field aliases another field in one effect list")
            seen.add(key)
            result[kind].append(key if kind == "clobbers" else [key, binding(item["value"])])
    if result["pureWrite"] and (not result["ordered"] or result["reads"] or result["clobbers"] or not result["writes"]):
        raise Unsupported("pure setup must be ordered exact nonempty writes, without reads/clobbers")
    return result


def program(value, isa, effects, *, allow_multiple=False, extent_limit=1_000_000):
    if type(allow_multiple) is not bool:
        raise Unsupported("explicit multiple-loop parsing profile required")
    if type(extent_limit) is not int or extent_limit not in (1_000_000, (1 << 63) - 1):
        raise Unsupported("explicit legacy or signed-index64 extent profile required")
    kernel = node(value, "Kernel", ("name", "args", "body", "attrs"))
    # All actual tensor descriptors and arbitrary retained attrs stay in frame.
    frame = canonical({key: item for key, item in kernel.items() if key != "body"})
    atoms, commands, loops = [], [], []

    def atom(value):
        call_shape(value)
        effect = decode_effect(value, isa, effects)
        index = len(atoms)
        atoms.append(
            {
                "id": index,
                "token": canonical(value),
                "effect": effect,
                "eraseAllowed": effect is not None and effect["pureWrite"],
            }
        )
        return index

    for statement in sequence(kernel["body"]):
        if statement.get("class") == "merlin.sched.ir.kernel.Call":
            commands.append({"call": atom(statement)})
            continue
        loop = node(statement, "Loop", LOOP_KEYS)
        completion = {key: sequence(loop[key]) for key in ("iter_args", "yields", "results")}
        if any(completion.values()):
            raise Unsupported("phase1 refuses ALL actual Loop completion-token fields")
        if type(loop["extent"]) is not int or loop["extent"] <= 0 or loop["extent"] > extent_limit:
            raise Unsupported("positive bounded static tripcount required; zero-trip hoist excluded")
        if type(loop["var"]) is not str or not loop["var"]:
            raise Unsupported("actual loop induction-variable identity required")
        ids = [atom(item) for item in sequence(loop["body"])]
        metadata = canonical({key: item for key, item in loop.items() if key != "body"})
        command = {
            "loop": {
                "var": loop["var"],
                "extent": loop["extent"],
                "body": ids,
                "metadata": metadata,
                "iterArgs": completion["iter_args"],
                "yields": completion["yields"],
                "results": completion["results"],
            }
        }
        commands.append(command)
        loops.append((len(commands) - 1, command["loop"]))
    if (not allow_multiple and len(loops) != 1) or len(atoms) > 4096:
        raise Unsupported("phase1 requires exactly one complete top-level loop and bounded calls")
    return {"frame": frame, "atoms": atoms, "commands": commands}, loops if allow_multiple else loops[0]


def project(record):
    if record.get("schema") != "merlin.configuration-hoist.actual-kernel.v1" or record.get("revision") != REVISION:
        raise Unsupported("explicit new e048 producer/parser selection required; no4499fallback")
    return project_current(record["before"], record["after"], isa=record["isa"], effects=record["effects"])


def project_current(source, final, *, isa, effects, program_decoder=program):
    """Project current complete trees; historical producer selection stays above."""
    before, (position, loop) = program_decoder(source, isa, effects)
    after, _ = program_decoder(final, isa, effects)
    if len(loop["body"]) < 2:
        raise Unsupported("one setup plus nonempty retained body required")
    setup, *body = loop["body"]
    setter = before["atoms"][setup]["effect"]
    if setter is None or not setter["pureWrite"] or not setter["ordered"]:
        raise Unsupported("first loop call must be a known ordered total exact pure setter")
    leading_body = before["atoms"][body[0]]["effect"]
    if leading_body is not None and leading_body["pureWrite"]:
        raise Unsupported("phase1 supports exactly one call in the maximal pure setter prefix")
    fields = {key for key, _ in setter["writes"]}
    for index in body:
        effect = before["atoms"][index]["effect"]
        if effect is None or not effect["ordered"]:
            raise Unsupported("retained body requires explicit ordered effects/frame")
        if fields.intersection([*(key for key, _ in effect["writes"]), *effect["clobbers"]]):
            raise Unsupported("phase1 refuses ALL body write/clobber overlap, including identical writes")
    names = sorted(
        {
            key
            for p in (before, after)
            for a in p["atoms"]
            if a["effect"] is not None
            for key in [
                *(x[0] for x in a["effect"]["reads"]),
                *(x[0] for x in a["effect"]["writes"]),
                *a["effect"]["clobbers"],
            ]
        }
    )
    if len(names) > 256:
        raise Unsupported("actual field budget exceeded")
    for p in (before, after):
        for a in p["atoms"]:
            if a["effect"] is not None:
                for kind in ("reads", "writes"):
                    a["effect"][kind] = [[names.index(key), value] for key, value in a["effect"][kind]]
                a["effect"]["clobbers"] = [names.index(key) for key in a["effect"]["clobbers"]]
    move = {
        "earlier": [row["call"] for row in before["commands"][:position]],
        "later": [row["call"] for row in before["commands"][position + 1 :]],
        "setup": setup,
        "body": body,
        "var": loop["var"],
        "extent": loop["extent"],
        "metadata": loop["metadata"],
    }
    return {
        "schema": "merlin.configuration-hoist.projection.v1",
        "before": before,
        "after": after,
        "move": move,
        "fields": names,
        "authority": False,
        "kernel_outcome": "UNOBSERVED",
        "Python_projection_trusted": True,
        "actual_ISA_effect_model_approved": False,
    }
