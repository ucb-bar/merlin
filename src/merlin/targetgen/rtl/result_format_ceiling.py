"""Detached, negative-only native result-format evidence from selected sources.

An exhaustive ceiling can refuse a result outside the native data formats. A
member does not establish a conversion, an operation, or numerical correctness.
Selections are original-owner inputs; this module grants no original authority.
Unexplained output drivers leave the ceiling unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import quant_formats as Q
from merlin.targetgen.compute_units import compute_units, semantic_capability_map
from merlin.targetgen.rtl import firrtl_struct as F
from merlin.targetgen.rtl.result_format_validity import (
    UNDEFINED,
    BooleanPaths,
    conjunction,
    negate,
    parse_module,
    possible,
)
from merlin.targetgen.rtl.source_predicates import _active_source

SCHEMA = "merlin.selected_native_result_ceiling.v1"
_ISSUED = weakref.WeakKeyDictionary()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _capabilities(contract):
    return _canonical({name: asdict(value) for name, value in semantic_capability_map(compute_units(contract)).items()})


def _read(pin, *, maximum=8 << 20):
    if type(pin) is not dict or set(pin) != {"path", "sha256"}:
        raise ValueError("result-format evidence needs exact source pins")
    path = Path(pin["path"])
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("result-format source is not an ordinary canonical file")
    if path.stat().st_size > maximum:
        raise ValueError("result-format source exceeds its bounded input")
    data = path.read_bytes()
    if _digest(data) != pin["sha256"]:
        raise ValueError("result-format source bytes changed")
    return data


def _balanced(text, start, opening="(", closing=")"):
    if text[start : start + 1] != opening:
        raise ValueError("selected source declaration is not balanced")
    depth = 0
    for position in range(start, len(text)):
        depth += int(text[position] == opening) - int(text[position] == closing)
        if depth == 0:
            return text[start + 1 : position], position + 1
    raise ValueError("selected source declaration is incomplete")


def _split_arguments(text):
    result, begin, stack = [], 0, []
    partners = {"(": ")", "[": "]", "{": "}"}
    for index, char in enumerate(text):
        if char in partners:
            stack.append(partners[char])
        elif char in ")]}":
            if not stack or stack.pop() != char:
                raise ValueError("selected source arguments disagree")
        elif char == "," and not stack:
            if text[begin:index].strip():
                result.append(text[begin:index].strip())
            begin = index + 1
    if stack:
        raise ValueError("selected source arguments are incomplete")
    if text[begin:].strip():
        result.append(text[begin:].strip())
    return result


def _declaration(text, prefix):
    active = _active_source(text)
    matches = [index for index, line in enumerate(active.splitlines()) if line.strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("selected source declaration is not unique")
    offset = sum(len(line) for line in active.splitlines(keepends=True)[: matches[0]])
    return active[offset:]


def _signed_constructor(expression):
    """Read an explicit numeric constructor, never name a format from a wire width."""
    compact = "".join(expression.split())
    if not compact.startswith("SInt("):
        raise ValueError("selected native data prototype is not a signed integer constructor")
    arguments, end = _balanced(compact, 4)
    if end != len(compact) or not arguments.endswith(".W") or not arguments[:-2].isdigit():
        raise ValueError("selected signed prototype has no literal declared width")
    bits = int(arguments[:-2])
    name = "int" + str(bits)
    if not Q.has(name):
        raise ValueError("selected signed numeric prototype has no registered format")
    return name, bits


def _configuration(selection, sources):
    """Follow the selected constructor/default/value chain in original Scala."""
    row = selection["configuration"]
    top = _declaration(sources[row["top_source"]], "class " + row["top_symbol"] + " ")
    extends = top.partition("extends Config")[2].lstrip()
    call, _ = _balanced(extends, 0)
    constructors = [part.strip().split()[1] for part in call.split("++") if part.strip().startswith("new ")]
    if constructors.count(row["provider_constructor"]) != 1:
        raise ValueError("selected configuration does not select the declared provider")
    provider = _declaration(sources[row["provider_source"]], "class " + row["provider_symbol"])
    parameter_start = provider.find("(")
    parameters, _ = _balanced(provider, parameter_start)
    defaults = [
        argument for argument in _split_arguments(parameters) if argument.partition(":")[0].strip() == row["parameter"]
    ]
    if len(defaults) != 1 or defaults[0].rpartition("=")[2].strip() != row["value_symbol"]:
        raise ValueError("selected provider default differs from the selected value")
    value = _declaration(sources[row["value_source"]], "val " + row["value_symbol"].rsplit(".", 1)[-1] + " =")
    expression = value.partition("=")[2].lstrip()
    bracket = expression.find("[")
    type_arguments, type_end = _balanced(expression, bracket, "[", "]")
    if _split_arguments(type_arguments)[0] != "SInt":
        raise ValueError("selected provider data type is not signed integer arithmetic")
    arguments, _ = _balanced(expression, expression.find("(", type_end))
    fields = {}
    for argument in _split_arguments(arguments):
        key, separator, data = argument.partition("=")
        if not separator or key.strip() in fields:
            raise ValueError("selected configuration contains an unexplained positional/duplicate field")
        fields[key.strip()] = data.strip()
    formats = {role: _signed_constructor(fields[field]) for role, field in row["scalar_roles"].items()}
    # A floating scale register is not a floating data result. Every selected
    # scale callback must terminate in the source's signed integer conversion,
    # rather than reinterpreting a raw floating product as the data prototype.
    integer_class = row["integer_conversion_class"]
    for field in row["scale_fields"]:
        value = fields[field]
        if value == "None":
            continue
        arrow = value.find("=>")
        if arrow < 0:
            raise ValueError("selected scale callback is unexplained")
        body, _end = _balanced(value, value.find("{", arrow), "{", "}")
        compact = "".join(body.split())
        marker = "Module(new" + integer_class + "("
        declarations = []
        for line in body.splitlines():
            if marker in "".join(line.split()):
                lhs, _sep, _rhs = line.strip().partition("=")
                if lhs.startswith("val "):
                    declarations.append(lhs[4:].strip())
        if len(declarations) != 1:
            raise ValueError("selected scale callback lacks one explicit integer conversion")
        converter = declarations[0]
        if converter + ".io.signedOut:=true.B" not in compact:
            raise ValueError("selected scale callback does not request signed integer output")
        tail = body.rstrip().splitlines()[-1].strip()
        if not tail.startswith("Mux(") or not tail.endswith(converter + ".io.out.asTypeOf(t))"):
            raise ValueError("selected scale callback has an unexplained numerical return path")
    return formats


def _provenance(selection, sources, facts):
    origin = selection["origin"]
    note = sources[origin["record_source"]]
    if "CONFIG=" + selection["configuration"]["top_symbol"] not in note:
        raise ValueError("original elaboration record names another selected configuration")
    covered = {origin["record_source"], origin["checksums_source"]}
    for member in origin["repositories"]:
        root, revision = Path(member["root"]), member["revision"]
        spelling = member.get("record_spelling", str(root))
        if spelling not in note or revision not in note:
            raise ValueError("original elaboration record omits a selected source revision")
        if "parent" in member:
            parent = next(row for row in origin["repositories"] if row["name"] == member["parent"])
            link = subprocess.check_output(
                ["git", "-C", parent["root"], "ls-tree", parent["revision"], member["record_spelling"]], text=True
            ).split()
            if len(link) != 4 or link[:3] != ["160000", "commit", revision]:
                raise ValueError("selected generator revision differs from the original source gitlink")
        for name in member["sources"]:
            covered.add(name)
            source = selection["sources"][name]
            relative = Path(source["path"]).relative_to(root)
            raw = subprocess.check_output(["git", "-C", str(root), "show", revision + ":" + str(relative)])
            if _digest(raw) != source["sha256"]:
                raise ValueError("selected hardware source differs from its original revision")
    if covered != set(sources):
        raise ValueError("selected hardware sources lack complete original revision binding")
    firrtl = selection["firrtl"]
    declared = facts["inputs"]
    if firrtl != {"path": declared["fir_path"], "sha256": declared["fir_sha256"]}:
        raise ValueError("result-format projection does not use the unchanged original elaboration")
    sums = sources[origin["checksums_source"]]
    expected = firrtl["sha256"] + "  " + Path(firrtl["path"]).name
    if expected not in sums.splitlines():
        raise ValueError("original elaboration checksum record omits the selected source")


def _reference_type(module, reference):
    declaration = module.decls.get(reference.root)
    if declaration is None:
        return None
    return F.field_type(F.parse_type(declaration[1]), reference.segments)


def _input_port(module, reference):
    declaration = module.decls.get(reference.root)
    if declaration is None or declaration[0] not in {"input", "output"}:
        return False
    direction, typ = declaration[0] == "input", F.parse_type(declaration[1])
    for segment in reference.segments:
        if isinstance(segment, int):
            if typ[0] != "vec" or not 0 <= segment < typ[2]:
                return False
            typ = typ[1]
        else:
            if typ[0] != "bundle":
                return False
            fields = [field for field in typ[1] if field[0] == segment]
            if len(fields) != 1:
                return False
            _name, flipped, typ = fields[0]
            direction ^= flipped
    return direction


def _project(reference, prefix, value):
    if reference.root != prefix.root or reference.segments[: len(prefix.segments)] != prefix.segments:
        return None
    remaining = reference.segments[len(prefix.segments) :]
    if remaining:
        if not isinstance(value, F.Ref):
            raise ValueError("output aggregate projection is not an exact alias")
        return F.Ref(value.root, (*value.segments, *remaining), value.dynamic)
    return value


class _Paths:
    """Exact field projections; scalar inputs require explicit source semantics.

    Instance/state/memory leaves are retained, never treated as empty cones. A
    selected original operand or numeric-operation terminal can close one; its
    role is supplied by the checked configuration, not by its carrier width.
    """

    def __init__(self, modules, terminals, formats, memory_ports):
        self.modules, self.terminals, self.formats = modules, terminals, formats
        self.visited, self.unknown, self.used, self.lines = set(), [], set(), set()
        self.memo, self.active = {}, set()
        self.memory_ports = memory_ports
        self.boolean = BooleanPaths(modules, _project)

    def walk(self, module_name, expression, *, trail=(), condition=True):
        if not possible(condition):
            return set()
        if len(self.visited) >= 20000 or len(trail) > 64:
            self.unknown.append("output projection exceeds its bounded graph")
            return set()
        if isinstance(expression, F.Lit):
            if expression.value == 0:
                return set()
            roles = {
                role
                for role, (_name, bits) in self.formats.items()
                if expression.kind == "SInt" and expression.width is not None and expression.width <= bits
            }
            return roles or {"literal"}
        if isinstance(expression, F.Prim):
            if expression.op == "asSInt" and len(expression.args) == 1 and isinstance(expression.args[0], F.Lit):
                literal = expression.args[0]
                return self.walk(
                    module_name, F.Lit("SInt", literal.width, literal.value), trail=trail, condition=condition
                )
            if expression.op not in {"mux", "asUInt", "asSInt", "cat", "bits", "pad", "tail", "head", "shl", "shr"}:
                self.unknown.append("unexplained output data primitive: " + expression.op)
                return set()
            if expression.op == "mux":
                selector = self.boolean.read(module_name, expression.args[0], trail)
                return self.walk(
                    module_name, expression.args[1], trail=trail, condition=conjunction(condition, selector)
                ) | self.walk(
                    module_name, expression.args[2], trail=trail, condition=conjunction(condition, negate(selector))
                )
            arguments = expression.args
            values = set()
            for argument in arguments:
                if not isinstance(argument, F.IntArg):
                    values.update(self.walk(module_name, argument, trail=trail, condition=condition))
            return values
        if not isinstance(expression, F.Ref) or expression.dynamic or None in expression.segments:
            self.unknown.append("unexplained dynamic/native output expression")
            return set()
        key = (module_name, expression.root, expression.segments, trail, condition)
        if key in self.memo:
            return self.memo[key]
        if key in self.active:
            self.unknown.append("unclosed output state/interface cycle: " + module_name + ":" + expression.text())
            return set()
        self.active.add(key)
        self.visited.add(key)
        try:
            result = self._reference(module_name, expression, trail, condition)
            self.memo[key] = result
            return result
        finally:
            self.active.remove(key)

    def _reference(self, module_name, expression, trail, condition):
        module = self.modules[module_name]
        if expression.root == UNDEFINED:
            self.unknown.append("undefined output is reachable under consumer validity")
            return set()
        for index, terminal in enumerate(self.terminals):
            prefix = F.parse_expr(terminal["reference"])
            if terminal["module"] != module_name or _project(expression, prefix, prefix) is None:
                continue
            self.used.add(index)
            return {terminal["role"]}
        if expression.root in module.nodes:
            value, line, _locator = module.nodes[expression.root]
            self.lines.add((module_name, line))
            if expression.segments:
                self.unknown.append("unexplained node subfield projection")
                return set()
            return self.walk(module_name, value, trail=trail, condition=condition)
        drivers = []
        for connection in module.connects_to(expression.root):
            value = _project(expression, connection.lhs, connection.rhs)
            if value is not None:
                self.lines.add((module_name, connection.line))
                drivers.append((connection, value))
        if drivers:
            values = set()
            remaining = condition
            for connection, value in reversed(drivers):
                guard = self.boolean.guards(module_name, connection.guards, trail)
                active = conjunction(remaining, guard)
                if isinstance(value, F.Ref) and value.root == UNDEFINED and possible(active):
                    self.unknown.append(module_name + ":" + expression.text() + " has a reachable invalid driver")
                values.update(self.walk(module_name, value, trail=trail, condition=active))
                remaining = conjunction(remaining, negate(guard))
            declaration = module.decls.get(expression.root)
            if possible(remaining) and (declaration is None or declaration[0] not in {"reg", "regreset", "mport"}):
                self.unknown.append("output driver lacks a definition under consumer validity")
            return values
        declaration = module.decls.get(expression.root)
        memory = self.memory_ports.get(module_name, {}).get(expression.root)
        if declaration is not None and declaration[0] == "mport" and memory is not None:
            writes = [
                name
                for name, (kind, store) in self.memory_ports[module_name].items()
                if store == memory[1] and kind in {"write", "infer", "rdwr"} and module.connects_to(name)
            ]
            if writes:
                values = set()
                for name in writes:
                    # The stored value's format must hold for every admitted
                    # write, not just when its later read is valid.
                    values.update(self.walk(module_name, F.Ref(name, expression.segments), trail=trail))
                return values
        if declaration is not None and declaration[0] == "inst" and expression.segments:
            child = declaration[1]
            if child in self.modules:
                return self.walk(
                    child,
                    F.Ref(str(expression.segments[0]), expression.segments[1:]),
                    trail=(*trail, (module_name, expression.root)),
                    condition=condition,
                )
        if declaration is not None and declaration[0] in {"input", "output"} and trail:
            parent, instance = trail[-1]
            return self.walk(
                parent, F.Ref(instance, (expression.root, *expression.segments)), trail=trail[:-1], condition=condition
            )
        self.unknown.append(module_name + ":" + expression.text())
        return set()


def _projection(selection, formats, sources):
    declaration = selection["projection"]
    names = set(declaration["modules"])
    bodies = F.read_module_bodies(selection["firrtl"]["path"], names)
    modules = {name: parse_module(name, lines) for name, lines in bodies.items()}
    if set(modules) != names:
        raise ValueError("selected output projection omits actual source modules")
    terminals = declaration["terminals"]
    for terminal in terminals:
        if terminal["role"] not in formats or terminal["kind"] not in {
            "original_numeric_operand",
            "original_serialized_operand",
            "signed_integer_conversion",
        }:
            raise ValueError("output terminal lacks selected native numeric semantics")
        module = modules[terminal["module"]]
        reference = F.parse_expr(terminal["reference"])
        if terminal["kind"] == "original_numeric_operand":
            if not _input_port(module, reference):
                raise ValueError("native operand witness is not an actual input port")
            typ = _reference_type(module, reference)
            while typ is not None and typ[0] == "vec":
                typ = typ[1]
            if typ != ("sint", formats[terminal["role"]][1]):
                raise ValueError("native operand differs from its explicit signed configuration prototype")
        elif terminal["kind"] == "original_serialized_operand":
            # The format comes from the original source's logical Vec prototype,
            # not from this packed UInt carrier. A storage width alone cannot
            # choose this role. Keep this explicit serialization premise visible.
            binding = _declaration(sources[terminal["prototype_source"]], "val " + terminal["prototype_symbol"] + " =")
            expression = binding.partition("=")[2].lstrip()
            while expression.startswith("Vec("):
                arguments, _end = _balanced(expression, 3)
                values = _split_arguments(arguments)
                if len(values) != 2:
                    raise ValueError("native serialized operand has an unexplained logical prototype")
                expression = values[1]
            field = selection["configuration"]["scalar_roles"][terminal["role"]]
            if expression.strip() != field:
                raise ValueError("native serialized operand is not bound to its original scalar prototype")
            typ = _reference_type(module, reference)
            if typ is None or typ[0] != "uint" or not typ[1] or typ[1] % formats[terminal["role"]][1]:
                raise ValueError("native row serialization does not preserve its declared scalar slots")
        else:
            sign = F.parse_expr(terminal["signed_selector"])
            drivers = [c for c in module.connects if c.lhs == sign]
            if len(drivers) != 1 or drivers[0].guards or drivers[0].rhs != F.Lit("UInt", 1, 1):
                raise ValueError("numeric conversion does not select signed integer output")
            if (
                not reference.segments
                or reference.root not in module.decls
                or module.decls[reference.root][0] != "inst"
            ):
                raise ValueError("integer conversion terminal is not an actual selected instance")
            if not module.decls[reference.root][1].startswith(
                selection["configuration"]["integer_conversion_class"] + "_"
            ):
                raise ValueError("output producer is not the configured integer conversion")
    memory_ports = {}
    for name, lines in bodies.items():
        ports = {}
        for _number, raw in lines:
            statement = F.split_locator(raw.strip())[0]
            head = statement.split(None, 2)
            if len(head) == 3 and head[0] in {"read", "write", "infer", "rdwr"} and head[1] == "mport":
                port, separator, rhs = head[2].partition("=")
                if not separator or "[" not in rhs:
                    raise ValueError("native output memory port is unexplained")
                ports[port.strip()] = (head[0], rhs.split("[", 1)[0].strip())
        memory_ports[name] = ports
    paths = _Paths(modules, terminals, formats, memory_ports)
    values = set()
    for endpoint in declaration["endpoints"]:
        if "valid_when" not in endpoint:
            raise ValueError("selected outbound projection needs actual consumer validity")
        valid = paths.boolean.read(endpoint["module"], F.parse_expr(endpoint["valid_when"]))
        values.update(paths.walk(endpoint["module"], F.parse_expr(endpoint["reference"]), condition=valid))
    if paths.unknown or not values or "literal" in values:
        reason = "; ".join(paths.unknown[:8]) or "unexplained constant/empty value projection"
        raise ValueError("native output-domain proof remains UNKNOWN: " + reason)
    return sorted({formats[role][0] for role in values}), sorted(paths.lines), sorted(paths.used)


@dataclass(frozen=True, eq=False)
class NativeResultFormatCeiling:
    target: str
    contract_sha256: str
    facts_sha256: str
    selection_sha256: str
    families: tuple[str, ...]
    formats: tuple[str, ...]
    capability_fingerprint: str
    _selection: str

    def _owner(self):
        owner = _ISSUED.get(self)
        fields = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if (
            type(self) is not NativeResultFormatCeiling
            or owner is None
            or owner[:3] != (os.getpid(), threading.current_thread(), fields)
        ):
            raise ValueError("result-format ceiling is not the unchanged live original source proof")

    def record(self):
        self._owner()
        return {
            "target": self.target,
            "contract_sha256": self.contract_sha256,
            "facts_sha256": self.facts_sha256,
            "selection_sha256": self.selection_sha256,
            "capability_map_sha256": _digest(self.capability_fingerprint.encode()),
            "families": list(self.families),
            "native_formats": list(self.formats),
            "scope": "source-native output ceiling only; membership supplies no conversion support",
        }

    def revalidate_sources(self):
        self._owner()
        selection = json.loads(self._selection)
        sources = {name: _read(pin).decode("utf-8") for name, pin in selection["sources"].items()}
        _provenance(selection, sources, {"inputs": _ISSUED[self][3]})
        _configuration(selection, sources)
        _read(selection["firrtl"], maximum=256 << 20)

    def refusal(self, family, out_dtype, cap_map):
        self._owner()
        actual = _canonical({name: asdict(value) for name, value in cap_map.items()})
        if actual != self.capability_fingerprint:
            raise ValueError("negative result-format proof belongs to another original capability map")
        if family not in self.families or out_dtype is None or not Q.has(out_dtype):
            return None
        if Q.get(out_dtype).name in self.formats:
            return None  # membership never supplies positive conversion evidence
        return {
            "reason": "result format lies outside the original source-proven native output ceiling",
            "refusal": "result_dtype",
            "native_formats": list(self.formats),
            "selection_sha256": self.selection_sha256,
            "contract_sha256": self.contract_sha256,
            "facts_sha256": self.facts_sha256,
            "scope": "native numerical forms only; negative evidence, not conversion support or qualification",
        }


def verify_original_selection(
    *, original_contract_bytes, original_facts_bytes, selection_bytes, expected_selection_sha256
):
    """Re-derive one explicitly selected ceiling over unchanged original inputs."""
    import yaml

    if _digest(selection_bytes) != expected_selection_sha256:
        raise ValueError("result-format selection differs from its original byte pin")
    selected = json.loads(selection_bytes)
    if selected.get("schema") != SCHEMA:
        raise ValueError("result-format selection schema is unavailable")
    facts, contract = json.loads(original_facts_bytes), yaml.safe_load(original_contract_bytes)
    if selected["original_facts_sha256"] != _digest(original_facts_bytes) or selected[
        "original_contract_sha256"
    ] != _digest(original_contract_bytes):
        raise ValueError("result-format selection differs from the original facts/contract")
    if selected["target"] != contract["name"] or facts["inputs"]["target"] != selected["target"]:
        raise ValueError("result-format proof names another original target")
    sources = {name: _read(pin).decode("utf-8") for name, pin in selected["sources"].items()}
    _provenance(selected, sources, facts)
    formats = _configuration(selected, sources)
    _read(selected["firrtl"], maximum=256 << 20)
    derived, _lines, _terminals = _projection(selected, formats, sources)
    for pin in selected["sources"].values():
        _read(pin)
    _read(selected["firrtl"], maximum=256 << 20)
    proof = NativeResultFormatCeiling(
        selected["target"],
        _digest(original_contract_bytes),
        _digest(original_facts_bytes),
        expected_selection_sha256,
        tuple(selected["families"]),
        tuple(derived),
        _capabilities(contract),
        _canonical(selected),
    )
    fields = tuple(getattr(proof, field) for field in proof.__dataclass_fields__)
    _ISSUED[proof] = (os.getpid(), threading.current_thread(), fields, dict(facts["inputs"]))
    return proof
