"""Recursive actual Kernel projection; no proof authority or producer replay.

Decode both actual endpoints. Derive only untrusted single-focus stages from
their observed prefix positions; the future closed recursive checker must bind
every full stage/context and exact terminal join. No native/SDK execution.
"""

from __future__ import annotations

import copy

from merlin.sched.ir import Kernel, parse_kernel
from merlin.sched.ir.kernel import check_structure
from merlin.verify.compiler_configuration_hoist_indexed import _substitute
from merlin.verify.compiler_configuration_kernel import tree
from merlin.verify.configuration_elision_gate import Unavailable
from merlin.verify.configuration_hoist_bridge import Unsupported, canonical, decode_effect

MAX_NODES = 4096
MAX_DEPTH = 64
MAX_STAGE_WORK = 65536
MAX_CAPTURE_BYTES = 32 * 1024 * 1024


def done():
    return {"tag": "done"}


def call(ident, rest):
    return {"tag": "call", "id": ident, "rest": rest}


def prepend(ids, rest):
    for ident in reversed(ids):
        rest = call(ident, rest)
    return rest


def decode(raw, *, isa, effects, analyses, extent_limit=1_000_000):
    # Same bounded static extent admission as the unbound current profiles;
    # this prototype does not inherit C64 authority from PassValidation64.
    if type(raw) is not str or len(raw.encode()) > MAX_CAPTURE_BYTES:
        raise Unsupported("bounded actual recursive Kernel text required")
    kernel = parse_kernel(raw)
    if type(kernel) is not Kernel or check_structure(kernel) or kernel.text() != raw:
        raise Unsupported("exact positive complete typed canonical recursive Kernel required")
    atoms, nodes = [], 0

    def body(statements, domains, depth):
        nonlocal nodes
        if depth > MAX_DEPTH:
            raise Unsupported("recursive Kernel source depth budget exceeded")
        commands = []
        for statement in statements:
            nodes += 1
            if nodes > MAX_NODES:
                raise Unsupported("recursive Kernel source node budget exceeded")
            actual = tree(statement)
            if actual["class"] == "merlin.sched.ir.kernel.Call":
                substitute = _substitute(actual, isa, effects, domains, analyses)
                effect = decode_effect(substitute, isa, effects)
                ident = len(atoms)
                atoms.append(
                    {
                        "id": ident,
                        "token": canonical(actual),
                        "effect": effect,
                        "eraseAllowed": effect is not None and effect["pureWrite"],
                    }
                )
                commands.append({"tag": "call", "id": ident})
            else:
                loop = actual["fields"]
                if statement.iter_args or statement.yields or statement.results:
                    raise Unsupported("recursive profile refuses carried completion fields")
                if type(statement.extent) is not int or not 0 < statement.extent <= extent_limit:
                    raise Unsupported("positive original selected recursive tripcount profile required")
                if statement.var in domains:
                    raise Unsupported("recursive profile refuses induction-variable shadowing")
                commands.append(
                    {
                        "tag": "loop",
                        "var": statement.var,
                        "extent": statement.extent,
                        "metadata": canonical({key: value for key, value in loop.items() if key != "body"}),
                        "body": body(statement.body, {**domains, statement.var: (0, statement.extent - 1)}, depth + 1),
                    }
                )
        code = done()
        for command in reversed(commands):
            code = {**command, "rest": code}
        return code

    fields = tree(kernel)["fields"]
    fields.pop("body")
    return {"frame": canonical(fields), "atoms": atoms, "code": body(kernel.body, {}, 0)}


def at(code, path):
    for key in path:
        code = code[key]
    return code


def replace_at(code, path, replacement):
    if not path:
        return replacement
    target = code
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    return code


def context(code, path):
    if not path:
        return {"tag": "hole"}
    first, *tail = path
    if code["tag"] == "call" and first == "rest":
        return {"tag": "call", "id": code["id"], "rest": context(code["rest"], tail)}
    if code["tag"] == "loop":
        common = {key: code[key] for key in ("var", "extent", "metadata")}
        if first == "body":
            return {
                "tag": "loopBody",
                **common,
                "body": context(code["body"], tail),
                "rest": copy.deepcopy(code["rest"]),
            }
        if first == "rest":
            return {
                "tag": "loopRest",
                **common,
                "body": copy.deepcopy(code["body"]),
                "rest": context(code["rest"], tail),
            }
    raise Unsupported("invalid exact recursive focus path")


def plug(outer, inner):
    tag = outer["tag"]
    if tag == "hole":
        return inner
    if tag == "call":
        return call(outer["id"], plug(outer["rest"], inner))
    common = {key: outer[key] for key in ("var", "extent", "metadata")}
    if tag == "loopBody":
        return {"tag": "loop", **common, "body": plug(outer["body"], inner), "rest": outer["rest"]}
    if tag == "loopRest":
        return {"tag": "loop", **common, "body": outer["body"], "rest": plug(outer["rest"], inner)}
    raise Unsupported("unknown recursive context constructor")


def source(move):
    common = {key: move[key] for key in ("var", "extent", "metadata", "rest")}
    return {"tag": "loop", **common, "body": prepend(move["setups"], move["body"])}


def target(move):
    return prepend(
        move["setups"], {"tag": "loop", **{key: move[key] for key in ("var", "extent", "metadata", "rest", "body")}}
    )


def stages(before, after):
    current, result = copy.deepcopy(before), []
    # Copies are bounded using static source size, never dynamic tripcounts.
    units = len(canonical(before).encode()) + len(canonical(after).encode())

    def align(path, wanted):
        nonlocal current
        actual = at(current, path)
        if actual == wanted:
            return
        if actual["tag"] == "call" and wanted["tag"] == "call" and actual["id"] == wanted["id"]:
            align((*path, "rest"), wanted["rest"])
            return
        if actual["tag"] != "loop":
            raise Unsupported("actual recursive endpoints differ outside a loop-prefix scope")
        cursor, moved = wanted, []
        while cursor["tag"] == "call":
            moved.append(cursor["id"])
            cursor = cursor["rest"]
        if cursor["tag"] != "loop" or any(actual[key] != cursor[key] for key in ("var", "extent", "metadata")):
            raise Unsupported("actual recursive loop binder/domain/metadata changed")
        if moved:
            remaining = actual["body"]
            for ident in moved:
                if remaining["tag"] != "call" or remaining["id"] != ident:
                    raise Unsupported("observed recursive motion is not the exact original leading prefix")
                remaining = remaining["rest"]
            if (len(result) + 1) * units > MAX_CAPTURE_BYTES or len(result) >= MAX_STAGE_WORK:
                raise Unsupported("recursive context stage expansion budget exceeded")
            move = {key: copy.deepcopy(actual[key]) for key in ("var", "extent", "metadata", "rest")}
            move.update(setups=moved, body=copy.deepcopy(remaining))
            focus = context(current, path)
            assert current == plug(focus, source(move))
            current = replace_at(current, path, target(move))
            result.append({"after": copy.deepcopy(current), "context": focus, "move": move})
            align(path, wanted)
            return
        align((*path, "body"), cursor["body"])
        align((*path, "rest"), cursor["rest"])

    align((), after)
    return result, current


def _project(original, final, *, isa, effects, extent_limit):
    analyses = []
    before, after = [
        decode(raw, isa=isa, effects=effects, analyses=analyses, extent_limit=extent_limit) for raw in (original, final)
    ]
    fields = sorted(
        {
            key
            for program in (before, after)
            for atom in program["atoms"]
            if atom["effect"] is not None
            for key in [
                *(a for a, _ in atom["effect"]["reads"]),
                *(a for a, _ in atom["effect"]["writes"]),
                *atom["effect"]["clobbers"],
            ]
        }
    )
    if len(fields) > 256:
        raise Unsupported("recursive field registry exceeds existing bound")
    for program in (before, after):
        for atom in program["atoms"]:
            effect = atom["effect"]
            if effect is not None:
                for kind in ("reads", "writes"):
                    effect[kind] = [[fields.index(key), value] for key, value in effect[kind]]
                effect["clobbers"] = [fields.index(key) for key in effect["clobbers"]]
    proposed, terminal = stages(before["code"], after["code"])
    return {
        "schema": "merlin.configuration-hoist.recursive-projection.v1",
        "before": before,
        "after": after,
        "fields": fields,
        "stages": proposed,
        "actual_final_join": terminal == after["code"],
        "actual_frame_join": before["frame"] == after["frame"],
        "actual_atoms_join": before["atoms"] == after["atoms"],
        "indexed_data_analysis": analyses,
        "projection_is_trusted_host": True,
        "all_environment_model_premise_retained": True,
        "proof_authority": False,
        "kernel_outcome": "UNOBSERVED",
        "native_fidelity_proved": False,
        "model_approved": False,
    }


def contains_nested_kernel(value):
    """Actual typed source only; no caller-selected profile flag."""
    pending = [(row, 0) for row in value["fields"]["body"]["tuple"]]
    count = 0
    while pending:
        row, depth = pending.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise Unavailable("recursive source inspection exceeds static budget")
        if row.get("class") == "merlin.sched.ir.kernel.Loop":
            if depth:
                return True
            pending.extend((child, depth + 1) for child in row["fields"]["body"]["tuple"])
    return False


def _dimensions(code, depth=0):
    if code["tag"] == "done":
        return 0, depth
    if code["tag"] == "call":
        return _dimensions(code["rest"], depth)
    body_count, body_depth = _dimensions(code["body"], depth + 1)
    rest_count, rest_depth = _dimensions(code["rest"], depth)
    return 1 + body_count + rest_count, max(body_depth, rest_depth)


def project_current(source, final, *, isa, effects, c_consumer=None):
    """Full recursive actual endpoints under original selection/consumer custody.

    The proof is all-environment and count-generic. Static expression/width
    admission remains trusted host analysis; no native fidelity is inferred.
    """
    extent_limit = 1_000_000
    if c_consumer is not None:
        from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer

        if type(c_consumer) is not ConfigurationCConsumer:
            raise Unsupported("original typed configuration C consumer required")
        c_consumer.revalidate()
        for raw in (source, final):
            c_consumer.admit(parse_kernel(raw))
        extent_limit = (1 << (c_consumer.index_bits - 1)) - 1
    try:
        source_tree = tree(parse_kernel(source))
        if not contains_nested_kernel(source_tree):
            raise Unsupported("actual source requires at least one nested counted loop")
        result = _project(source, final, isa=isa, effects=effects, extent_limit=extent_limit)
        count, depth = _dimensions(result["before"]["code"])
        result.update(actual_loop_count=count, max_loop_depth=depth, focused_loop_count=len(result["stages"]))
        return result
    except RecursionError as error:
        raise Unavailable("recursive host projection exhausted its bounded Python stack") from error


def proof_sources():
    from merlin.common.paths import module_source_path

    from .compiler_configuration_hoist_prefix import proof_sources as prefix_sources
    from .configuration_elision_bridge import file_ref

    name = "Merlin.GraphStateConfigurationRecursiveHoist"
    path = (
        module_source_path("merlin.verify.compiler_configuration_hoist_recursive").parent
        / "lean"
        / (name.replace(".", "/") + ".lean")
    )
    if path.is_symlink() or [line for line in path.read_text().splitlines() if line.startswith("import ")] != [
        "import Merlin.GraphStateConfigurationPrefixHoist"
    ]:
        raise ValueError("exact canonical recursive hoist proof closure required")
    return [*prefix_sources(), {"module": name, **file_ref(path)}]
