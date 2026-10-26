"""Independent reader for the MLIR text used by the reviewed graph packets.

Graham independent review (agent), 2026-10-26. Read-only. This module imports
neither xDSL nor any Merlin module; it is a separate character-level reader
for the generic form plus the custom forms that occur in the reviewed buffers.
Unknown custom syntax raises `Unsupported` rather than being skipped. File
metadata (`{-# ... #-}`) and `loc(...)` suffixes are retained as raw text so
the audit can report what a consumer could silently drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class Unsupported(ValueError):
    pass


@dataclass
class Block:
    args: list  # [(name, type)]
    ops: list


@dataclass
class Op:
    name: str
    results: list  # names
    result_types: list
    operands: list  # names
    operand_types: list
    props: dict  # key -> canonical text (inherent + discardable merged by reader)
    attrs: dict  # discardable only, when the syntax distinguishes them
    regions: list  # [[Block]]
    successors: list = field(default_factory=list)
    loc: str | None = None
    raw_props: dict = field(default_factory=dict)


@dataclass
class Module:
    ops: list
    attrs: dict
    file_metadata: str | None
    generic: bool


IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-")


class Reader:
    def __init__(self, text):
        self.s = text
        self.i = 0
        self.locs = 0

    # ---- lexical helpers -------------------------------------------------
    def ws(self):
        s = self.s
        while self.i < len(s):
            c = s[self.i]
            if c in " \t\r\n":
                self.i += 1
            elif s.startswith("//", self.i):
                while self.i < len(s) and s[self.i] != "\n":
                    self.i += 1
            else:
                break

    def peek(self, token):
        self.ws()
        return self.s.startswith(token, self.i)

    def eat(self, token):
        if self.peek(token):
            self.i += len(token)
            return True
        return False

    def expect(self, token):
        if not self.eat(token):
            raise Unsupported(f"expected {token!r} at {self.where()}")

    def where(self):
        line = self.s.count("\n", 0, self.i) + 1
        return f"line {line}: {self.s[self.i:self.i + 40]!r}"

    def word(self):
        self.ws()
        start = self.i
        while self.i < len(self.s) and self.s[self.i] in IDENT_CHARS:
            if self.s.startswith("->", self.i):
                break
            self.i += 1
        if start == self.i:
            raise Unsupported(f"expected identifier at {self.where()}")
        return self.s[start:self.i]

    def sigil(self, prefix):
        self.ws()
        if not self.s.startswith(prefix, self.i):
            raise Unsupported(f"expected {prefix} name at {self.where()}")
        self.i += 1
        if self.s.startswith('"', self.i):
            return prefix + self.string()
        start = self.i
        while self.i < len(self.s) and (self.s[self.i].isalnum() or self.s[self.i] in "_.$-"):
            self.i += 1
        return prefix + self.s[start:self.i]

    def value_use(self):
        name = self.sigil("%")
        if self.s.startswith("#", self.i):
            self.i += 1
            start = self.i
            while self.i < len(self.s) and self.s[self.i].isdigit():
                self.i += 1
            name += "#" + self.s[start:self.i]
        return name

    def string(self):
        self.ws()
        if self.s[self.i] != '"':
            raise Unsupported(f"expected string at {self.where()}")
        j = self.i + 1
        out = []
        while self.s[j] != '"':
            if self.s[j] == "\\":
                out.append(self.s[j:j + 2])
                j += 2
            else:
                out.append(self.s[j])
                j += 1
        self.i = j + 1
        return "".join(out)

    def balanced(self, stops):
        """Raw text up to a top-level stop character; whitespace removed outside strings."""
        self.ws()
        depth = 0
        out = []
        s = self.s
        pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
        while self.i < len(s):
            c = s[self.i]
            if c == '"':
                out.append('"' + self.string() + '"')
                continue
            if s.startswith("->", self.i):
                out.append("->")
                self.i += 2
                continue
            if depth == 0 and c in stops:
                break
            if c in pairs:
                depth += 1
            elif c in ")]}>":
                depth -= 1
                if depth < 0:
                    break
            if c not in " \t\r\n":
                out.append(c)
            self.i += 1
        return "".join(out)

    # ---- types -------------------------------------------------------------
    def type_(self):
        self.ws()
        if self.peek("("):
            self.expect("(")
            inputs = self.type_list(")")
            self.expect(")")
            self.expect("->")
            if self.peek("("):
                self.expect("(")
                outputs = self.type_list(")")
                self.expect(")")
            else:
                outputs = [self.type_()]
            return "(" + ",".join(inputs) + ")->(" + ",".join(outputs) + ")"
        head = self.word() if not self.peek("!") else "!" + self.sigil("!")[1:]
        if self.s.startswith("<", self.i):
            self.i += 1
            body = self.balanced(">")
            self.expect(">")
            return head + "<" + body + ">"
        return head

    def type_list(self, close):
        types = []
        if self.peek(close):
            return types
        types.append(self.type_())
        while self.eat(","):
            types.append(self.type_())
        return types

    def function_type(self):
        self.expect("(")
        inputs = self.type_list(")")
        self.expect(")")
        self.expect("->")
        if self.peek("("):
            self.expect("(")
            outputs = self.type_list(")")
            self.expect(")")
        else:
            outputs = [self.type_()]
        return inputs, outputs

    # ---- attributes --------------------------------------------------------
    def dictionary(self, close="}"):
        entries = {}
        if self.peek(close) or self.peek("}"):
            return entries
        while True:
            self.ws()
            key = self.string() if self.peek('"') else self.word()
            if self.eat("="):
                entries[key] = self.balanced(",}" if close == "}" else ",>")
            else:
                entries[key] = "unit"
            if not self.eat(","):
                break
        return entries

    def trailing_loc(self):
        if self.peek("loc("):
            self.expect("loc(")
            text = self.balanced(")")
            self.expect(")")
            self.locs += 1
            return text
        return None

    # ---- structure ---------------------------------------------------------
    def module(self):
        self.ws()
        generic = self.peek('"builtin.module"')
        if generic:
            op = self.op()
            if op.name != "builtin.module":
                raise Unsupported("top-level operation is not builtin.module")
            body = op.regions[0][0].ops
            attrs = op.props
        else:
            self.expect("builtin.module")
            attrs = {}
            if self.eat("attributes"):
                self.expect("{")
                attrs = self.dictionary()
                self.expect("}")
            regions = self.region(entry_args=[])
            body = regions[0].ops
        metadata = None
        if self.peek("{-#"):
            start = self.i
            end = self.s.index("#-}", start) + 3
            metadata = self.s[start:end]
            self.i = end
        self.ws()
        if self.i != len(self.s):
            raise Unsupported(f"trailing text after module at {self.where()}")
        return Module(body, attrs, metadata, generic)

    def region(self, entry_args):
        self.expect("{")
        blocks = []
        first = True
        while not self.peek("}"):
            args = list(entry_args) if first else []
            if self.peek("^"):
                self.sigil("^")
                args = []
                if self.eat("("):
                    while not self.peek(")"):
                        name = self.sigil("%")
                        self.expect(":")
                        args.append((name, self.type_()))
                        self.trailing_loc()
                        self.eat(",")
                    self.expect(")")
                self.expect(":")
            ops = []
            while not self.peek("}") and not self.peek("^"):
                ops.append(self.op())
            blocks.append(Block(args, ops))
            first = False
        self.expect("}")
        if not blocks and entry_args:
            blocks.append(Block(list(entry_args), []))
        return blocks

    def op(self):
        results = []
        self.ws()
        if self.peek("%"):
            while True:
                name = self.sigil("%")
                count = 1
                if self.eat(":"):
                    self.ws()
                    start = self.i
                    while self.s[self.i].isdigit():
                        self.i += 1
                    count = int(self.s[start:self.i])
                results += [name] if count == 1 else [f"{name}#{k}" for k in range(count)]
                if not self.eat(","):
                    break
            self.expect("=")
        if self.peek('"'):
            op = self.generic(results)
        else:
            op = self.custom(results)
        op.loc = self.trailing_loc()
        return op

    def generic(self, results):
        name = self.string()
        self.expect("(")
        operands = []
        while not self.peek(")"):
            operands.append(self.value_use())
            self.eat(",")
        self.expect(")")
        successors = []
        if self.eat("["):
            while not self.peek("]"):
                successors.append(self.sigil("^"))
                self.eat(",")
            self.expect("]")
        props = {}
        if self.eat("<{"):
            props = self.dictionary(">")
            self.expect("}>")
        regions = []
        if self.peek("(") and self.s[self.i + 1:].lstrip().startswith("{"):
            self.expect("(")
            while not self.peek(")"):
                regions.append(self.region(entry_args=[]))
                self.eat(",")
            self.expect(")")
        attrs = {}
        if self.peek("{"):
            self.expect("{")
            attrs = self.dictionary()
            self.expect("}")
        self.expect(":")
        inputs, outputs = self.function_type()
        if len(inputs) != len(operands) or len(outputs) != len(results):
            raise Unsupported(f"generic arity mismatch for {name}")
        return Op(name, results, outputs, operands, inputs, props, attrs, regions, successors, raw_props=dict(props))

    # custom forms that occur in the reviewed buffers -----------------------
    def custom(self, results):
        name = self.word()
        handler = CUSTOM.get(name)
        if handler is None:
            prefix = name.split(".")[0] + "."
            handler = CUSTOM.get(prefix + "*")
        if handler is None:
            raise Unsupported(f"no independent custom reader for {name} at {self.where()}")
        return handler(self, name, results)


def _operands(r, close=None):
    values = [r.value_use()]
    while r.eat(","):
        values.append(r.value_use())
    return values


def _func(r, name, results):
    visibility = None
    for word in ("private", "public", "nested"):
        if r.peek(word + " ") or r.peek(word + "@"):
            r.expect(word)
            visibility = word
    sym = r.sigil("@")
    r.expect("(")
    args, types = [], []
    while not r.peek(")"):
        if r.peek("%"):
            arg = r.sigil("%")
            r.expect(":")
            ty = r.type_()
            args.append((arg, ty))
            types.append(ty)
        else:
            types.append(r.type_())
        if r.peek("{"):
            r.expect("{")
            r.dictionary()
            r.expect("}")
            raise Unsupported("argument attributes are outside the independent reader")
        r.eat(",")
    r.expect(")")
    outputs = []
    if r.eat("->"):
        if r.peek("("):
            r.expect("(")
            outputs = r.type_list(")")
            r.expect(")")
        else:
            outputs = [r.type_()]
    attrs = {}
    if r.eat("attributes"):
        r.expect("{")
        attrs = r.dictionary()
        r.expect("}")
    props = {"sym_name": '"' + sym[1:] + '"', "function_type": "(" + ",".join(types) + ")->(" + ",".join(outputs) + ")"}
    if visibility:
        props["sym_visibility"] = '"' + visibility + '"'
    regions = []
    if r.peek("{"):
        if len(args) != len(types):
            raise Unsupported("function body without named arguments")
        regions = [r.region(entry_args=args)]
    return Op("func.func", results, [], [], [], {**props, **attrs}, attrs, regions)


def _return(r, name, results):
    operands, types = [], []
    if r.peek("%"):
        operands = _operands(r)
        r.expect(":")
        types = r.type_list("\n")
    return Op(name, results, [], operands, types, {}, {}, [])


def _call(r, name, results):
    callee = r.sigil("@")
    r.expect("(")
    operands = [] if r.peek(")") else _operands(r)
    r.expect(")")
    r.expect(":")
    inputs, outputs = r.function_type()
    return Op(name, results, outputs, operands, inputs, {"callee": callee}, {}, [])


def _constant(r, name, results):
    r.ws()
    value = r.balanced(":\n")
    ty = None
    if r.eat(":"):
        ty = r.type_()
    else:
        raise Unsupported("constant without explicit type")
    return Op(name, results, [ty], [], [], {"value": value + ":" + ty}, {}, [])


def _arith(r, name, results):
    if name in ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast", "arith.index_castui"):
        operand = r.value_use()
        r.expect(":")
        source = r.type_()
        r.expect("to")
        target = r.type_()
        return Op(name, results, [target], [operand], [source], {}, {}, [])
    if name == "arith.constant":
        return _constant(r, name, results)
    operands = _operands(r)
    props = {}
    if name in ("arith.addi", "arith.subi", "arith.muli"):
        props["overflowFlags"] = "#arith.overflow<none>"
    if r.peek("overflow<"):
        r.expect("overflow<")
        flags = r.balanced(">")
        r.expect(">")
        props["overflowFlags"] = "#arith.overflow<" + flags + ">"
    attrs = {}
    if r.peek("{"):
        r.expect("{")
        attrs = r.dictionary()
        r.expect("}")
    r.expect(":")
    ty = r.type_()
    if len(results) != 1:
        raise Unsupported("multi-result custom arith operation")
    return Op(name, results, [ty], operands, [ty] * len(operands), {**props, **attrs}, attrs, [])


def _for(r, name, results):
    if r.peek("unsigned"):
        raise Unsupported("scf.for unsigned comparison is outside the independent reader")
    iv = r.sigil("%")
    r.expect("=")
    lower = r.value_use()
    r.expect("to")
    upper = r.value_use()
    r.expect("step")
    step = r.value_use()
    carried, initial, carry_types = [], [], []
    if r.eat("iter_args"):
        r.expect("(")
        while not r.peek(")"):
            carried.append(r.sigil("%"))
            r.expect("=")
            initial.append(r.value_use())
            r.eat(",")
        r.expect(")")
        r.expect("->")
        if r.peek("("):
            r.expect("(")
            carry_types = r.type_list(")")
            r.expect(")")
        else:
            carry_types = [r.type_()]
    iv_type = "index"
    if r.eat(":"):
        iv_type = r.type_()
    entry = [(iv, iv_type), *zip(carried, carry_types)]
    region = r.region(entry_args=entry)
    if len(region) == 1 and (not region[0].ops or region[0].ops[-1].name != "scf.yield"):
        if carry_types:
            raise Unsupported("implicit yield with carried values")
        region[0].ops.append(Op("scf.yield", [], [], [], [], {}, {}, [], loc="implicit-terminator"))
    return Op(
        name, results, list(carry_types), [lower, upper, step, *initial], [iv_type] * 3 + list(carry_types), {}, {}, [region]
    )


def _yield(r, name, results):
    return _return(r, name, results)


def _memref(r, name, results):
    attrs = {}
    if name == "memref.alloc":
        r.expect("(")
        dynamic = [] if r.peek(")") else _operands(r)
        r.expect(")")
        if dynamic:
            raise Unsupported("dynamic allocation sizes outside the independent reader")
        if r.peek("{"):
            r.expect("{")
            attrs = r.dictionary()
            r.expect("}")
        r.expect(":")
        ty = r.type_()
        return Op(name, results, [ty], [], [], dict(attrs), attrs, [])
    if name == "memref.dealloc":
        operand = r.value_use()
        r.expect(":")
        return Op(name, results, [], [operand], [r.type_()], {}, {}, [])
    if name == "memref.dim":
        source = r.value_use()
        r.expect(",")
        index = r.value_use()
        r.expect(":")
        return Op(name, results, ["index"], [source, index], [r.type_(), "index"], {}, {}, [])
    if name in ("memref.load", "memref.store"):
        stored = []
        if name == "memref.store":
            stored = [r.value_use()]
            r.expect(",")
        memref = r.value_use()
        r.expect("[")
        indices = [] if r.peek("]") else _operands(r)
        r.expect("]")
        if r.peek("{"):
            r.expect("{")
            attrs = r.dictionary()
            r.expect("}")
        r.expect(":")
        ty = r.type_()
        element = ty[ty.index("<") + 1:-1].split("x")[-1] if ty.startswith("memref<") else None
        if element is None or "," in element:
            raise Unsupported("memref layout/space outside the independent reader")
        operand_types = ([element] if stored else []) + [ty] + ["index"] * len(indices)
        return Op(
            name, results, [] if stored else [element], [*stored, memref, *indices], operand_types, dict(attrs), attrs, []
        )
    raise Unsupported(f"no independent custom reader for {name}")


CUSTOM = {
    "func.func": _func,
    "func.return": _return,
    "func.call": _call,
    "arith.*": _arith,
    "scf.for": _for,
    "scf.yield": _yield,
    "memref.*": _memref,
}

# Inherent attributes that the generic syntax spells as properties but which are
# structural (segment sizes) rather than distinct semantics.
STRUCTURAL_PROPS = {"operandSegmentSizes", "resultSegmentSizes"}


def canonical_props(op):
    """Merged semantic metadata key: inherent + discardable, structural sizes dropped."""
    merged = {**op.props, **op.attrs}
    for key in STRUCTURAL_PROPS:
        merged.pop(key, None)
    if "function_type" in merged:
        merged["function_type"] = Reader(merged["function_type"]).type_()
    return tuple(sorted(merged.items()))


def read(text):
    reader = Reader(text)
    module = reader.module()
    module.locs = reader.locs
    return module


def walk(ops):
    for op in ops:
        yield op
        for region in op.regions:
            for block in region:
                yield from walk(block.ops)
