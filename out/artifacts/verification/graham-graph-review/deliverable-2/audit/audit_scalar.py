"""Independent IR -> Lean correspondence audit for the scalar CSE/DCE -> LICM chains.

Graham independent review (agent), 2026-10-26. Read-only. Imports only the
standard library and the sibling independent reader `mlir_subset`; it imports
neither xDSL nor any Merlin projection/serializer. For each case it

1. reads before/middle/after MLIR text with the independent reader and builds
   its own de Bruijn term for every function (own primitive whitelist, own
   descriptor keys = name + merged metadata + operand/result types);
2. parses the generated Lean query (registry, inputs/outputs, every Program);
3. requires structural equality of the two terms up to ONE descriptor renaming
   shared by all stages, checks that renaming is a bijection, and checks the
   Lean registry signature of each descriptor against the reader's types;
4. converts the implementation's projection.json records into terms with its
   own descriptor numbers and requires byte-for-byte term equality with Lean
   (serializer check) and agreement of its registry descriptors with the
   reader's keys;
5. evaluates the MLIR directly and the Lean terms under one concrete model
   (wrapping iN arithmetic, signed scf.for, logged opaque effects) on a grid of
   inputs and compares returned values plus the full effect log.

Step 5 is a differential test of this instance under one chosen model, not a
proof of projection fidelity.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlir_subset as M  # noqa: E402

REPO = Path("/home/grammy/blast/merlin-pass-verification")
D2 = REPO / "out/artifacts/verification/graham-graph-review/deliverable-2"
WORKED = REPO / "out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261025/worked-example/original"
VARIANTS = D2 / "probes/scalar-variants"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Independent projection
# ---------------------------------------------------------------------------
def int_width(ty):
    if ty.startswith("i") and ty[1:].isdigit():
        width = int(ty[1:])
        if 1 <= width <= 256:
            return width
    return None


def type_code(ty):
    width = int_width(ty)
    if width is not None:
        return ("bits", width)
    if ty == "index":
        return ("index",)
    return ("opaque", ty)


def literal(text, width):
    if text in ("true", "false"):
        return 1 if text == "true" else 0
    value = int(text)
    return value


def primitive(op):
    """Reviewer's own whitelist: total, world-free, unflagged signless integer ops."""
    if op.regions or op.successors or op.attrs or len(op.results) != 1:
        return None
    widths = [int_width(t) for t in (*op.operand_types, *op.result_types)]
    if any(w is None for w in widths):
        return None
    props = dict(M.canonical_props(op))
    if op.name in ("arith.addi", "arith.subi", "arith.muli"):
        if props != {"overflowFlags": "#arith.overflow<none>"} or len(set(widths)) != 1:
            return None
        return ({"arith.addi": "add", "arith.subi": "sub", "arith.muli": "mul"}[op.name], widths[-1])
    if op.name == "arith.constant":
        value, _, ty = props.get("value", "").rpartition(":")
        if set(props) != {"value"} or ty != op.result_types[0]:
            return None
        try:
            return ("constant", widths[-1], literal(value, widths[-1]))
        except ValueError:
            return None
    kinds = {"arith.extsi": "sext", "arith.extui": "zext", "arith.trunci": "trunc"}
    if op.name in kinds and not props:
        return (kinds[op.name], widths[0], widths[-1])
    return None


def descriptor_key(op):
    return (op.name, M.canonical_props(op), tuple(op.operand_types), tuple(op.result_types))


class Refused(ValueError):
    pass


def block_term(block, captured, terminator, keys):
    env = [name for name, _ in block.args] + list(captured)
    ops = list(block.ops)
    if not ops or ops[-1].name != terminator:
        raise Refused("missing terminator")

    def idx(values):
        if any(v not in env for v in values):
            raise Refused("operand not lexically available")
        return tuple(env.index(v) for v in values)

    statements = []
    for op in ops:
        if op.successors:
            raise Refused("CFG successor")
        if op.name in ("func.return", "scf.yield"):
            if op is not ops[-1] or op.name != terminator or op.props or op.attrs:
                raise Refused("terminator misplaced/annotated")
            statements.append(("ret", idx(op.operands)))
            continue
        prim = primitive(op)
        if prim is not None:
            statements.append(("pure", prim, idx(op.operands)))
        elif op.name == "scf.for":
            if M.canonical_props(op) or len(op.regions) != 1 or len(op.regions[0]) != 1:
                raise Refused("loop metadata/geometry")
            body = op.regions[0][0]
            carries = tuple(type_code(t) for t in op.result_types)
            if tuple(type_code(t) for _, t in body.args) != (("index",), *carries):
                raise Refused("IV/carry slots")
            statements.append(("loop", idx(op.operands[:3]), idx(op.operands[3:]), carries,
                               block_term(body, env, "scf.yield", keys)))
        else:
            if op.regions:
                raise Refused("unsupported region operation")
            key = descriptor_key(op)
            keys.add(key)
            statements.append(("opaque", key, idx(op.operands), tuple(type_code(t) for t in op.result_types)))
        env = list(op.results) + env
    # fold into nested continuation form
    term = statements[-1]
    for st in reversed(statements[:-1]):
        if st[0] == "pure":
            term = ("pure", st[1], st[2], term)
        elif st[0] == "opaque":
            term = ("opaque", st[1], st[2], st[3], term)
        else:
            lb, ub, step = st[1]
            term = ("loop", lb, ub, step, st[2], st[3], st[4], term)
    return term


def project_module(module, keys):
    functions = []
    for op in module.ops:
        if op.name != "func.func":
            raise Refused("module-level operation other than func.func")
        props = dict(M.canonical_props(op))
        ftype = props["function_type"]
        inputs, outputs = M.Reader(ftype).function_type()
        row = {"name": props["sym_name"], "header": M.canonical_props(op),
               "inputs": tuple(type_code(t) for t in inputs), "outputs": tuple(type_code(t) for t in outputs)}
        if not op.regions or not op.regions[0]:
            row["term"] = None
        else:
            if len(op.regions[0]) != 1:
                raise Refused("multi-block function")
            block = op.regions[0][0]
            if [t for _, t in block.args] != inputs:
                raise Refused("entry block args differ from signature")
            row["term"] = block_term(block, [], "func.return", keys)
        functions.append(row)
    return functions


# ---------------------------------------------------------------------------
# Lean query reader
# ---------------------------------------------------------------------------
class LeanReader:
    def __init__(self, text):
        self.s, self.i = text, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \n\t":
            self.i += 1

    def eat(self, tok):
        self.ws()
        if self.s.startswith(tok, self.i):
            self.i += len(tok)
            return True
        return False

    def expect(self, tok):
        if not self.eat(tok):
            raise ValueError(f"lean: expected {tok!r} at {self.s[self.i:self.i + 30]!r}")

    def nat(self):
        self.ws()
        j = self.i
        while self.s[j].isdigit():
            j += 1
        value, self.i = int(self.s[self.i:j]), j
        return value

    def integer(self):
        self.expect("(")
        negative = self.eat("-")
        value = self.nat()
        self.expect(")")
        return -value if negative else value

    def nat_list(self):
        self.expect("[")
        out = []
        while not self.eat("]"):
            out.append(self.nat())
            self.eat(",")
        return tuple(out)

    def ty(self):
        if self.eat(".index"):
            return ("index",)
        self.expect("(")
        if self.eat(".bits"):
            out = ("bits", self.nat())
        else:
            self.expect(".opaque")
            out = ("opaque-id", self.nat())
        self.expect(")")
        return out

    def ty_list(self):
        self.expect("[")
        out = []
        while not self.eat("]"):
            out.append(self.ty())
            self.eat(",")
        return tuple(out)

    def prim(self):
        self.expect("(")
        self.expect(".")
        self.ws()
        j = self.i
        while self.s[j].isalpha():
            j += 1
        kind, self.i = self.s[self.i:j], j
        if kind == "constant":
            out = ("constant", self.nat(), self.integer())
        elif kind in ("add", "sub", "mul"):
            out = (kind, self.nat())
        else:
            out = (kind, self.nat(), self.nat())
        self.expect(")")
        return out

    def program(self):
        self.expect("(")
        if self.eat(".ret"):
            out = ("ret", self.nat_list())
        elif self.eat(".pure"):
            out = ("pure", self.prim(), self.nat_list(), self.program())
        elif self.eat(".opaque"):
            out = ("opaque", self.nat(), self.nat_list(), self.ty_list(), self.program())
        else:
            self.expect(".loop")
            lb, ub, st = self.nat(), self.nat(), self.nat()
            out = ("loop", lb, ub, st, self.nat_list(), self.ty_list(), self.program(), self.program())
        self.expect(")")
        return out

    def registry(self):
        self.expect("[")
        out = []
        while not self.eat("]"):
            self.expect("{inputs :=")
            inputs = self.ty_list()
            self.expect(",outputs :=")
            outputs = self.ty_list()
            self.expect("}")
            out.append((inputs, outputs))
            self.eat(",")
        return out


def read_lean(text):
    out = {"namespaces": {}}
    namespace = None
    for line in text.splitlines():
        if line.startswith("def registry : List Signature := "):
            out["registry"] = LeanReader(line.split(":= ", 1)[1]).registry()
        elif line.startswith("namespace F"):
            namespace = line.split()[1]
            out["namespaces"].setdefault(namespace, {})
        elif line.startswith("def ") and namespace and " : Program := " in line:
            name = line[4:].split(" ")[0]
            out["namespaces"][namespace][name] = LeanReader(line.split(":= ", 1)[1]).program()
        elif line.startswith("def inputs : List Ty := ") or line.startswith("def outputs : List Ty := "):
            name = line[4:].split(" ")[0]
            out["namespaces"][namespace][name] = LeanReader(line.split(":= ", 1)[1]).ty_list()
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def lean_type(ty, typemap):
    if ty[0] == "opaque-id":
        return ty
    return ty


def match_types(mine, theirs, typemap, problems, where):
    if len(mine) != len(theirs):
        problems.append(f"{where}: type arity {len(mine)} vs {len(theirs)}")
        return
    for a, b in zip(mine, theirs):
        if a[0] == "opaque":
            if b[0] != "opaque-id":
                problems.append(f"{where}: opaque type {a} vs {b}")
            elif typemap.setdefault(a[1], b[1]) != b[1]:
                problems.append(f"{where}: opaque type {a[1]} mapped to two ids")
        elif a != b:
            problems.append(f"{where}: type {a} vs {b}")


def match_terms(mine, lean, keymap, typemap, problems, where, counts):
    if mine[0] != lean[0]:
        problems.append(f"{where}: constructor {mine[0]} vs {lean[0]}")
        return
    kind = mine[0]
    counts[kind] = counts.get(kind, 0) + 1
    if kind == "ret":
        if mine[1] != lean[1]:
            problems.append(f"{where}: return slots {mine[1]} vs {lean[1]}")
        return
    if kind == "pure":
        prim_m, prim_l = mine[1], lean[1]
        if prim_m[0] == "constant" and prim_l[0] == "constant":
            width = prim_m[1]
            if prim_m[1] != prim_l[1] or (prim_m[2] - prim_l[2]) % (1 << width):
                problems.append(f"{where}: constant {prim_m} vs {prim_l}")
        elif prim_m != prim_l:
            problems.append(f"{where}: primitive {prim_m} vs {prim_l}")
        if mine[2] != lean[2]:
            problems.append(f"{where}: operands {mine[2]} vs {lean[2]}")
        return match_terms(mine[3], lean[3], keymap, typemap, problems, where + ">pure", counts)
    if kind == "opaque":
        if keymap.setdefault(mine[1], lean[1]) != lean[1]:
            problems.append(f"{where}: descriptor {mine[1][0]} mapped to ids {keymap[mine[1]]} and {lean[1]}")
        if mine[2] != lean[2]:
            problems.append(f"{where}: opaque operands {mine[2]} vs {lean[2]}")
        match_types(mine[3], lean[3], typemap, problems, where)
        return match_terms(mine[4], lean[4], keymap, typemap, problems, where + ">opaque", counts)
    # loop
    if mine[1:4] != lean[1:4]:
        problems.append(f"{where}: loop bounds {mine[1:4]} vs {lean[1:4]}")
    if mine[4] != lean[4]:
        problems.append(f"{where}: loop initial {mine[4]} vs {lean[4]}")
    match_types(mine[5], lean[5], typemap, problems, where)
    match_terms(mine[6], lean[6], keymap, typemap, problems, where + ">body", counts)
    match_terms(mine[7], lean[7], keymap, typemap, problems, where + ">next", counts)


# ---------------------------------------------------------------------------
# projection.json records -> terms (serializer check)
# ---------------------------------------------------------------------------
def record_type(value):
    if "bits" in value:
        return ("bits", value["bits"])
    if "index" in value:
        return ("index",)
    return ("opaque-id", value["opaque"])


def record_prim(value):
    kind = value["kind"]
    if kind == "constant":
        return ("constant", value["width"], value["value"])
    if kind in ("add", "sub", "mul"):
        return (kind, value["width"])
    return (kind, value["source"], value["target"])


def records_term(records):
    term = ("ret", tuple(records[-1]["ret"]))
    for record in reversed(records[:-1]):
        if "pure" in record:
            term = ("pure", record_prim(record["pure"]), tuple(record["arguments"]), term)
        elif "effect" in record:
            term = ("opaque", record["effect"], tuple(record["arguments"]),
                    tuple(record_type(t) for t in record["outputs"]), term)
        else:
            loop = record["loop"]
            lb, ub, st = loop["bounds"]
            term = ("loop", lb, ub, st, tuple(loop["initial"]), tuple(record_type(t) for t in loop["carries"]),
                    records_term(loop["body"]), term)
    return term


def flatten(value):
    """Leaf strings of an xDSL `exact` descriptor tree."""
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


# ---------------------------------------------------------------------------
# Concrete differential model
# ---------------------------------------------------------------------------
def wrap(value, width):
    return value % (1 << width)


def opaque_result(key, args, world_len, out_types):
    digest = hashlib.sha256(repr((key[0], key[1], tuple(args), world_len)).encode()).digest()
    out = []
    for k, ty in enumerate(out_types):
        raw = int.from_bytes(digest[k * 4:k * 4 + 4], "little")
        width = int_width(ty) if isinstance(ty, str) else (ty[1] if ty[0] == "bits" else None)
        out.append(wrap(raw, width) if width else raw % 97)
    return out


def index_constant(key):
    if key[0] == "arith.constant" and key[3] == ("index",):
        return int(dict(key[1])["value"].rpartition(":")[0])
    return None


def apply_prim(prim, args):
    kind = prim[0]
    if kind == "constant":
        return wrap(prim[2], prim[1])
    if kind in ("add", "sub", "mul"):
        a, b = args
        return wrap({"add": a + b, "sub": a - b, "mul": a * b}[kind], prim[1])
    if kind == "trunc":
        return wrap(args[0], prim[2])
    if kind == "zext":
        return args[0]
    if kind == "sext":
        value = args[0]
        if value >> (prim[1] - 1):
            value -= 1 << prim[1]
        return wrap(value, prim[2])
    raise ValueError(kind)


def eval_term(term, env, world, idmap, fuel=[0]):
    """Lean evalProgram transcribed, under the concrete model (Except = Python tuple)."""
    kind = term[0]
    if kind == "ret":
        return [env[i] for i in term[1]], world
    if kind == "pure":
        return eval_term(term[3], [apply_prim(term[1], [env[i] for i in term[2]])] + env, world, idmap)
    if kind == "opaque":
        key = idmap[term[1]]
        args = [env[i] for i in term[2]]
        constant = index_constant(key)
        if constant is not None:
            results = [constant]
        else:
            results = opaque_result(key, args, len(world), key[3])
            world = world + ((key[0], tuple(args)),)
        return eval_term(term[4], results + env, world, idmap)
    lb, ub, st = (env[i] for i in term[1:4])
    values = [env[i] for i in term[4]]
    if st <= 0:
        raise ValueError("nonpositive step outside concrete model")
    for iv in range(lb, ub, st):
        values, world = eval_term(term[6], [iv] + values + env, world, idmap)
    return eval_term(term[7], values + env, world, idmap)


def eval_mlir(function, args):
    """Direct interpretation of the independently read MLIR (no projection)."""
    block = function.regions[0][0]
    values = {name: v for (name, _), v in zip(block.args, args)}
    world = ()

    def run(block, world):
        for op in block.ops:
            ins = [values[o] for o in op.operands]
            if op.name in ("func.return", "scf.yield"):
                return ins, world
            prim = primitive(op)
            if prim is not None:
                outs = [apply_prim(prim, ins)]
            elif op.name == "scf.for":
                lb, ub, st = ins[:3]
                carried = ins[3:]
                body = op.regions[0][0]
                iv = lb
                while iv < ub:
                    values[body.args[0][0]] = iv
                    for (name, _), v in zip(body.args[1:], carried):
                        values[name] = v
                    carried, world = run(body, world)
                    iv += st
                outs = carried
            else:
                key = descriptor_key(op)
                constant = index_constant(key)
                if constant is not None:
                    outs = [constant]
                else:
                    outs = opaque_result(key, ins, len(world), key[3])
                    world = world + ((key[0], tuple(ins)),)
            for name, v in zip(op.results, outs):
                values[name] = v
        raise ValueError("no terminator")

    return run(block, world)


# ---------------------------------------------------------------------------
def audit_case(name, paths):
    result = {"case": name, "inputs": {k: {"path": str(v), "sha256": sha(v)} for k, v in paths.items()}}
    modules = {stage: M.read(Path(paths[stage]).read_text()) for stage in ("before", "middle", "after")}
    result["reader"] = {stage: {"generic_syntax": m.generic, "file_metadata": m.file_metadata, "locations": m.locs,
                                "module_attributes": m.attrs, "operations": sum(1 for _ in M.walk(m.ops))}
                        for stage, m in modules.items()}
    keys = set()
    mine = {stage: project_module(m, keys) for stage, m in modules.items()}
    problems = []
    # interfaces/untouched functions (host-only obligations in the implementation)
    names = [[f["name"] for f in mine[s]] for s in ("before", "middle", "after")]
    if not names[0] == names[1] == names[2]:
        problems.append(f"function roster differs {names}")
    for f0, f1, f2 in zip(mine["before"], mine["middle"], mine["after"]):
        if not f0["header"] == f1["header"] == f2["header"]:
            problems.append(f"function header differs for {f0['name']}")
    lean = read_lean(Path(paths["lean"]).read_text())
    keymap, typemap, counts = {}, {}, {}
    defined = [k for k, f in enumerate(mine["before"]) if f["term"] is not None]
    pairs = {"before": "before", "after": "after", "componentBefore": "before", "middleLeft": "middle",
             "middleRight": "middle", "componentAfter": "after"}
    checked_terms = []
    for index in defined:
        namespace = lean["namespaces"].get(f"F{index}")
        if namespace is None:
            problems.append(f"no Lean namespace for defined function {index}")
            continue
        match_types(mine["before"][index]["inputs"], namespace["inputs"], typemap, problems, f"F{index}.inputs")
        match_types(mine["before"][index]["outputs"], namespace["outputs"], typemap, problems, f"F{index}.outputs")
        for lean_name, stage in pairs.items():
            if lean_name in namespace:
                match_terms(mine[stage][index]["term"], namespace[lean_name], keymap, typemap, problems,
                            f"F{index}.{lean_name}", counts)
                checked_terms.append(f"F{index}.{lean_name}")
    extra = set(lean["namespaces"]) - {f"F{k}" for k in defined}
    if extra:
        problems.append(f"Lean namespaces without a defined function: {sorted(extra)}")
    # bijection and registry signatures
    inverse = {}
    for key, ident in keymap.items():
        if ident in inverse:
            problems.append(f"descriptor id {ident} denotes two distinct reader keys: {inverse[ident][0]} / {key[0]}")
        inverse[ident] = key
    unused = [k for k in keys if k not in keymap]
    registry = lean["registry"]
    for ident, key in inverse.items():
        expected_in = tuple(type_code(t) for t in key[2])
        expected_out = tuple(type_code(t) for t in key[3])
        sig_in, sig_out = registry[ident]
        sub = []
        match_types(expected_in, sig_in, typemap, sub, f"registry[{ident}].inputs")
        match_types(expected_out, sig_out, typemap, sub, f"registry[{ident}].outputs")
        problems += sub
    result["descriptor_map"] = [{"lean_id": ident, "operation": key[0], "metadata": dict(key[1]),
                                 "operand_types": key[2], "result_types": key[3],
                                 "lean_signature": [list(registry[ident][0]), list(registry[ident][1])]}
                                for ident, key in sorted(inverse.items())]
    result["registry_entries_unused_by_any_term"] = [i for i in range(len(registry)) if i not in inverse]
    result["reader_keys_without_term"] = [k[0] for k in unused]
    result["lean_terms_matched"] = checked_terms
    result["constructor_counts"] = counts
    # serializer check against projection.json
    projection = json.loads(Path(paths["projection"]).read_text())
    serializer = []
    for index in defined:
        record = projection["functions"][index]
        namespace = lean["namespaces"][f"F{index}"]
        if records_term(record["before"]) != namespace["before"]:
            serializer.append(f"F{index}.before records != Lean term")
        if records_term(record["after"]) != namespace["after"]:
            serializer.append(f"F{index}.after records != Lean term")
        stages = projection.get("chain_stages")
        if stages:
            if records_term(stages["first"]["functions"][index]["after"]) != namespace.get("middleLeft"):
                serializer.append(f"F{index}.middleLeft records != Lean term")
            if records_term(stages["second"]["functions"][index]["before"]) != namespace.get("middleRight"):
                serializer.append(f"F{index}.middleRight records != Lean term")
    for ident, key in inverse.items():
        descriptor = projection["registry"][ident]["descriptor"]
        leaves = list(flatten(descriptor))
        if key[0] not in leaves and descriptor.get("name") != key[0]:
            serializer.append(f"registry[{ident}] descriptor does not name {key[0]}")
        constant = index_constant(key)
        if constant is not None and projection["registry"][ident]["index_constant"] != constant:
            serializer.append(f"registry[{ident}] index_constant != reader value {constant}")
    if len(projection["registry"]) != len(registry):
        serializer.append("projection registry length != Lean registry length")
    result["serializer_problems"] = serializer
    result["problems"] = problems
    # concrete differential evaluation
    idmap = dict(inverse)
    grid = [0, 1, 2, 3, 7, 100, 127, 128, 255]
    diffs, runs = [], 0
    for index in defined:
        fn = {s: [op for op in modules[s].ops][index] for s in modules}
        arity = len(mine["before"][index]["inputs"])
        widths = [t[1] if t[0] == "bits" else None for t in mine["before"][index]["inputs"]]
        for args in itertools.product(grid, repeat=arity):
            args = [wrap(a, w) if w else a for a, w in zip(args, widths)]
            outcomes = {}
            for stage in ("before", "middle", "after"):
                outcomes["mlir:" + stage] = eval_mlir(fn[stage], args)
            namespace = lean["namespaces"][f"F{index}"]
            for lean_name in ("before", "after", "middleLeft", "middleRight"):
                if lean_name in namespace:
                    outcomes["lean:" + lean_name] = eval_term(namespace[lean_name], list(args), (), idmap)
            runs += 1
            reference = outcomes["mlir:before"]
            for label, outcome in outcomes.items():
                if outcome != reference:
                    diffs.append({"function": index, "args": args, "label": label, "outcome": repr(outcome),
                                  "reference": repr(reference)})
            if args == [wrap(g, w) if w else g for g, w in zip([2, 3, 1, 1][:arity], widths)]:
                result.setdefault("sample", {"args": args, "returned": reference[0], "effect_log": reference[1]})
    result["differential"] = {"input_vectors": runs, "mismatches": diffs[:20], "mismatch_count": len(diffs),
                              "model": "wrapping iN add/sub/mul; signed index scf.for; index constants by value; "
                                       "other opaque ops logged with deterministic hashed results"}
    result["verdict"] = "match" if not problems and not serializer and not diffs else "MISMATCH"
    return result


def main():
    cases = {
        "worked-example(sealed copy)": {
            "before": WORKED / "checked/before.mlir", "middle": WORKED / "checked/middle.mlir",
            "after": WORKED / "checked/after.mlir", "lean": WORKED / "checked/kernel/GraphStateScalarInstance.lean",
            "projection": WORKED / "checked/projection/projection.json",
        },
    }
    for name in ("chain", "zero-chain", "nested-chain"):
        root = VARIANTS / name
        if root.exists():
            cases[f"{name}(local regeneration)"] = {
                "before": root / "source.mlir", "middle": root / "actual/middle.mlir", "after": root / "actual/after.mlir",
                "lean": root / "GraphStateScalarInstance.lean", "projection": root / "checked-projection/projection.json",
            }
    results = []
    for name, paths in cases.items():
        try:
            results.append(audit_case(name, paths))
        except (M.Unsupported, Refused) as error:
            results.append({"case": name, "verdict": "reader refused", "reason": str(error)})
    report = {"reviewer": "Graham independent review (agent)", "date": "2026-10-26",
              "script": {"path": __file__, "sha256": sha(__file__)},
              "reader": {"path": M.__file__, "sha256": sha(M.__file__)},
              "imports_implementation": False, "cases": results}
    out = D2 / "audit/audit_scalar.out.json"
    out.write_text(json.dumps(report, indent=2, default=list) + "\n")
    for row in results:
        print(row["case"], row["verdict"], row.get("constructor_counts"), len(row.get("problems", [])),
              len(row.get("serializer_problems", [])), row.get("differential", {}).get("input_vectors"))


if __name__ == "__main__":
    main()
