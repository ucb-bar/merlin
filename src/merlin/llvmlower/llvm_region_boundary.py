"""Bounded structural boundaries for proposals on literal LLVM source regions.

Only direct branches and void returns are understood. Opaque instructions are
retained, never assigned an effect or arithmetic meaning. LLVM verification,
operation semantics, pointer bounds and external ownership remain obligations
of the original consumer. Exact outside bytes do not discharge those proofs.
"""

import bisect
import json

from merlin.compile.scheduling.source_tokens import llvm_tokens


class LLVMRegionBoundary:
    """Index one explicitly selected void function without rewriting its source."""

    def __init__(
        self,
        text: str,
        *,
        function_name: str,
        argument_types: tuple[str, ...],
        max_scan_bytes: int = 1 << 20,
        max_scan_lines: int = 32768,
        max_cfg_blocks: int = 4096,
    ):
        for value, ceiling in ((max_scan_bytes, 1 << 20), (max_scan_lines, 32768), (max_cfg_blocks, 4096)):
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError("explicit bounded LLVM scan limits required")
        self.scan_limits = dict(
            max_scan_bytes=max_scan_bytes, max_scan_lines=max_scan_lines, max_cfg_blocks=max_cfg_blocks
        )
        if (
            type(text) is not str
            or type(function_name) is not str
            or not function_name
            or type(argument_types) is not tuple
        ):
            raise ValueError("explicit LLVM function and argument types required")
        if any(
            type(t) is not str or t != "ptr" and (not t.startswith("i") or not t[1:].isdecimal() or int(t[1:]) <= 0)
            for t in argument_types
        ):
            raise ValueError("only explicit scalar integer/pointer arguments supported")
        if len(text.encode()) > max_scan_bytes:
            raise ValueError("LLVM scan byte bound exceeded")
        self.text, self.lines = text, text.splitlines(True)
        if len(self.lines) > max_scan_lines:
            raise ValueError("LLVM scan line bound exceeded")
        offsets = [0]
        for line in self.lines:
            offsets.append(offsets[-1] + len(line))
        self.tokens = [[] for _ in self.lines]
        for token in llvm_tokens(text, max_bytes=max_scan_bytes):
            index = bisect.bisect_right(offsets, token.start) - 1
            if token.end > offsets[index + 1]:
                raise ValueError("multiline LLVM token unsupported")
            self.tokens[index].append(token)
        self.words = [tuple(t.text for t in row) for row in self.tokens]
        if any("blockaddress" in row for row in self.words):
            raise ValueError("blockaddress control unsupported")
        headers = [i for i, row in enumerate(self.words) if row[:3] == ("define", "void", "@" + function_name)]
        if len(headers) != 1:
            raise ValueError("selected LLVM void function absent or ambiguous")
        self.first = headers[0]
        header = self.words[self.first]
        if header[3:4] != ("(",) or header[-2:] != (")", "{") or any(word.startswith(('%"', '@"')) for word in header):
            raise ValueError("unsupported LLVM function header")
        arguments, names, cursor = [], [], 4
        while cursor < len(header) - 2:
            typ, name = header[cursor : cursor + 2]
            if not name.startswith("%") or name in names:
                raise ValueError("duplicate or unsupported LLVM argument")
            arguments.append(typ)
            names.append(name)
            cursor += 2
            if cursor < len(header) - 2:
                if header[cursor] != ",":
                    raise ValueError("unsupported LLVM argument attributes")
                cursor += 1
        if tuple(arguments) != argument_types or cursor != len(header) - 2:
            raise ValueError("selected LLVM argument types differ")
        self.arguments = tuple(names)
        self.argument_types = argument_types
        self.last = next((i for i in range(self.first + 1, len(self.lines)) if self.words[i] == ("}",)), None)
        if self.last is None:
            raise ValueError("unterminated LLVM function")
        self.blocks, self.definition_lines, self.locations, self.line_blocks = {}, {}, {}, {}
        numeric = [int(name[1:]) for name in names if name[1:].isdecimal()]
        entry = str(max(numeric, default=-1) + 1)
        self.entry = entry
        self.blocks[entry] = []
        block, began = entry, False
        labels = set()
        for i in range(self.first + 1, self.last):
            row = self.words[i]
            if any(word.startswith(('%"', '@"')) for word in row) or row and row[0].startswith('"'):
                # Quoted and unquoted spellings can denote the same LLVM name.
                # Refuse that namespace rather than invent a textual distinction.
                raise ValueError("quoted LLVM local/symbol namespace unsupported")
            if len(row) == 2 and row[1] == ":":
                label = row[0]
                if label in labels or "%" + label in names or "%" + label in self.definition_lines:
                    raise ValueError("duplicate LLVM block/SSA name")
                if not began:
                    self.blocks.pop(entry)
                    self.entry = label
                elif label in self.blocks:
                    raise ValueError("duplicate LLVM block label")
                labels.add(label)
                block = label
                self.blocks[block] = []
                began = True
            elif row:
                began = True
            self.blocks[block].append(i)
            self.line_blocks[i] = block
            if len(row) >= 2 and row[0].startswith("%") and row[1] == "=":
                name = row[0]
                if name in names or name in self.definition_lines or name[1:] in self.blocks:
                    raise ValueError("duplicate LLVM SSA definition")
                self.definition_lines[name], self.locations[name] = i, block
        if len(self.blocks) > max_cfg_blocks:
            raise ValueError("LLVM CFG scan block bound exceeded")
        self.defs = {
            name: self.text[self.tokens[i][2].start : self.tokens[i][-1].end]
            for name, i in self.definition_lines.items()
        }
        self.edges, self.predecessors, self.cfg = {}, {}, {self.first}
        for block, indices in self.blocks.items():
            code = [i for i in indices if self.words[i] and self.words[i][-1] != ":"]
            if not code:
                raise ValueError("empty LLVM CFG block")
            for i in code[:-1]:
                instruction = self.words[i][2:] if self.words[i][1:2] == ("=",) else self.words[i]
                if instruction[0] in {
                    "br",
                    "ret",
                    "switch",
                    "indirectbr",
                    "invoke",
                    "callbr",
                    "resume",
                    "unreachable",
                    "catchswitch",
                    "catchret",
                    "cleanupret",
                    "landingpad",
                    "catchpad",
                    "cleanuppad",
                }:
                    raise ValueError("early or unsupported LLVM terminator")
            row = self.words[code[-1]]
            if len(row) == 3 and row[:2] == ("br", "label") and row[2].startswith("%"):
                edges = ((row[2][1:], None, True),)
            elif (
                len(row) == 9
                and row[:2] == ("br", "i1")
                and row[2].startswith("%")
                and row[3:5] == (",", "label")
                and row[6:8] == (",", "label")
                and row[5].startswith("%")
                and row[8].startswith("%")
            ):
                edges = ((row[5][1:], row[2], True), (row[8][1:], row[2], False))
            elif row == ("ret", "void"):
                edges = ()
            else:
                raise ValueError("unknown LLVM CFG terminator/metadata")
            self.edges[block] = edges
            self.cfg.add(code[-1])
            self.cfg.update(i for i in indices if self.words[i][-1:] == (":",))
            for target, condition, truth in edges:
                if target not in self.blocks:
                    raise ValueError("unknown LLVM CFG destination")
                if condition is not None and condition not in self.definition_lines and condition not in names:
                    raise ValueError("unknown LLVM branch condition")
                self.predecessors.setdefault(target, []).append((block, condition, truth))
        if any(not self.reachable(self.entry, b) for b in self.blocks):
            raise ValueError("unreachable LLVM CFG block")
        self.context = {self.first}
        self.cfg_encoded = json.dumps(self.edges, sort_keys=True, separators=(",", ":")).encode()
        self.cfg_components = len(self.blocks) + sum(len(x) for x in self.edges.values())

    def code(self, index):
        """Actual code span, excluding comments and surrounding whitespace."""
        row = self.tokens[index]
        return self.text[row[0].start : row[-1].end] if row else ""

    def reachable(self, start, target, avoid=None):
        todo, seen = [start], set()
        while todo:
            block = todo.pop()
            if block == avoid or block in seen:
                continue
            if block == target:
                return True
            seen.add(block)
            todo.extend(x[0] for x in self.edges[block])
        return False

    def dominates(self, block, target):
        return self.reachable(self.entry, target) and not self.reachable(self.entry, target, block)

    def closed_region(self, header, exit):
        """Require one entry and exactly the supplied outgoing continuation."""
        if header not in self.blocks or exit not in self.blocks or header == exit:
            raise ValueError("actual distinct LLVM region boundaries required")
        todo, nodes = [header], set()
        while todo:
            block = todo.pop()
            if block == exit or block in nodes:
                continue
            nodes.add(block)
            if not self.edges[block]:
                raise ValueError("LLVM region terminates before continuation")
            todo.extend(x[0] for x in self.edges[block])
        if any(src not in nodes for b in nodes - {header} for src, _, _ in self.predecessors.get(b, ())):
            raise ValueError("LLVM region has side entry")
        if {x[0] for b in nodes for x in self.edges[b] if x[0] not in nodes} != {exit}:
            raise ValueError("LLVM region has another continuation")
        if any(not self.reachable(b, exit) for b in nodes):
            raise ValueError("LLVM region cannot reach continuation")
        return nodes

    def require_diamond_loop(self, header, copy, nodes):
        """Check a mandatory guarded body and its single backedge topology."""
        top = self.edges[header]
        if len(top) != 2 or not top[0][2] or top[1][2]:
            raise ValueError("LLVM loop header is not a true-body conditional")
        guard = top[0][0]
        if nodes != self.closed_region(header, top[1][0]):
            raise ValueError("LLVM loop region membership changed")
        branch = self.edges[guard]
        if len(branch) != 2 or branch[0][0] != copy:
            raise ValueError("LLVM guarded body is not mandatory")
        pad = branch[1][0]

        def jump(block):
            edges = self.edges[block]
            if len(edges) != 1 or edges[0][1] is not None:
                raise ValueError("LLVM body/join/latch has conditional bypass")
            return edges[0][0]

        join = jump(copy)
        latch = jump(join)
        if (
            jump(pad) != join
            or jump(latch) != header
            or nodes != {header, guard, copy, pad, join, latch}
            or len(nodes) != 6
            or not self.dominates(guard, latch)
        ):
            raise ValueError("LLVM loop differs from mandatory six-block topology")

    def require_precedes(self, definition_line, use_lines):
        for use in use_lines:
            source, target = self.line_blocks[definition_line], self.line_blocks[use]
            if (source == target and definition_line >= use) or (
                source != target and not self.dominates(source, target)
            ):
                raise ValueError("LLVM definition does not precede/dominate use")

    def references(self, index):
        row = self.words[index]
        start = 2 if row[:1] and row[0].startswith("%") and row[1:2] == ("=",) else 0
        return {x for x in row[start:] if x.startswith("%")}

    def require_no_liveouts(self, nodes):
        local = {i for b in nodes for i in self.blocks[b]}
        definitions = {n for n, i in self.definition_lines.items() if i in local}
        if any(self.references(i) & definitions for i in range(self.first + 1, self.last) if i not in local):
            raise ValueError("LLVM region has outside-used SSA")

    def dependency_union(self, nodes, *, include_lines=()):
        """Retain actual SSA definitions, globals and phi-connected predicates.

        All CFG edges are separately retained. This does not select or prove
        path predicates: the semantic reader must supply any additional facts
        through include_lines and prove their connection to the actual uses.
        """
        indices = {self.first, *include_lines, *(i for b in nodes for i in self.blocks[b])}
        todo = [x for i in indices for x in self.references(i)]
        seen = set()
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in self.arguments or name[1:] in self.blocks:
                continue
            if name not in self.definition_lines:
                raise ValueError("unknown LLVM backward SSA dependency")
            i = self.definition_lines[name]
            indices.add(i)
            todo.extend(self.references(i))
            if self.words[i][2:3] == ("phi",):
                block = self.locations[name]
                for source in (block, *(x[0] for x in self.predecessors.get(block, ()))):
                    todo.extend(condition for _, condition, _ in self.edges[source] if condition)
        globals_ = [x for i in indices for x in self.words[i] if x.startswith("@")]
        seen_globals = set()
        while globals_:
            name = globals_.pop()
            if name in seen_globals:
                continue
            seen_globals.add(name)
            if name == "@" + self.words[self.first][2][1:]:
                continue
            found = [
                i
                for i, row in enumerate(self.words)
                if row[:2] == (name, "=") or row[:1] == ("declare",) and name in row
            ]
            if len(found) != 1:
                raise ValueError("LLVM global dependency absent or ambiguous")
            indices.add(found[0])
            globals_.extend(x for x in self.words[found[0]] if x.startswith("@"))
        forbidden = {"poison", "undef", "freeze", "inbounds", "nuw", "nsw", "exact", "disjoint", "!"}
        if any(forbidden.intersection(self.words[i]) for i in indices):
            raise ValueError("LLVM dependency has unsupported poison/metadata assumptions")
        return indices

    def proof_budget(self, indices, *, changed_lines=(), max_bytes=65536, max_components=4096):
        """Charge one complete CFG and the aggregate actual proof union."""
        if (
            type(max_bytes) is not int
            or not 0 < max_bytes <= 65536
            or type(max_components) is not int
            or not 0 < max_components <= 4096
        ):
            raise ValueError("explicit bounded LLVM proof limits required")
        if type(indices) not in (set, frozenset) or any(
            type(i) is not int or not 0 <= i < len(self.lines) for i in indices
        ):
            raise ValueError("bounded actual LLVM proof line set required")
        size, count = len(self.cfg_encoded), self.cfg_components

        def charge(value):
            nonlocal size, count
            if value:
                size += len((value + "\n").encode())
                count += 1
            if size > max_bytes or count > max_components:
                raise ValueError("aggregate LLVM region proof union bound exceeded")

        charge("")
        for i in indices:
            if self.words[i] and self.words[i][-1] != ":" and self.words[i][0] not in {"br", "ret"}:
                charge(self.code(i))
        raw_bytes, fragments = 0, 0
        for line in changed_lines:
            raw_bytes += len(line.encode())
            fragments += 1
            if raw_bytes > self.scan_limits["max_scan_bytes"] or fragments > self.scan_limits["max_scan_lines"]:
                raise ValueError("LLVM changed-line scan bound exceeded")
            tokens = llvm_tokens(line, max_bytes=65536)
            if tokens:
                charge(line[tokens[0].start : tokens[-1].end])
        return dict(
            bytes=size,
            components=count,
            complete_cfg_bytes=len(self.cfg_encoded),
            complete_cfg_blocks=len(self.blocks),
            complete_cfg_edges=sum(len(x) for x in self.edges.values()),
        )

    def verify_outside(self, candidate, allowed_changes):
        """Require exact replacement lines and byte-identical unselected source."""
        if type(candidate) is not str or len(candidate.encode()) > self.scan_limits["max_scan_bytes"]:
            raise ValueError("LLVM candidate scan byte bound exceeded")
        lines = candidate.splitlines(True)
        if len(lines) != len(self.lines) or any(
            type(i) is not int or i not in self.line_blocks for i in allowed_changes
        ):
            raise ValueError("LLVM replacement line membership changed")
        for i, (before, after) in enumerate(zip(self.lines, lines, strict=True)):
            if after != allowed_changes.get(i, before):
                raise ValueError("LLVM source outside selected replacements changed")
        other = LLVMRegionBoundary(
            candidate,
            function_name=self.words[self.first][2][1:],
            argument_types=self.argument_types,
            **self.scan_limits,
        )
        if (
            other.edges != self.edges
            or other.arguments != self.arguments
            or other.definition_lines != self.definition_lines
        ):
            raise ValueError("LLVM replacement changed control or SSA frontier")
