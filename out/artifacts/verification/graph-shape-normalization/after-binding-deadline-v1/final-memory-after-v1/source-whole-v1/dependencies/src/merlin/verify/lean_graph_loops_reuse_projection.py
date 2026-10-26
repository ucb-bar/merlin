"""Independent complete before/after buffer-reuse projection, trusted Python.

Uses full actual SSA trees, static allocation facts and effect-free-prefix facts.
No StoragePlan, witness, producer replay or frozen-evaluator bridge is trusted.
"""

from __future__ import annotations

from .lean_graph_loops_segment_projection import _ids, _key, _metadata, _one_block, _types


class Decoder:
    def __init__(self, before, after, *, index_bits, max_nodes):
        if index_bits not in (32, 64) or type(index_bits) is not int:
            raise ValueError("explicit signed32/64 index context required")
        self.before, self.after, self.bits = before, after, index_bits
        self.source_ops, self.target_ops, self.source_defs, self.target_defs = {}, {}, {}, {}
        self.nodes = 0
        self.collect(before["tree"], (), self.source_ops, self.source_defs)
        self.collect(after["tree"], (), self.target_ops, self.target_defs)
        if type(max_nodes) is not int or not 0 < self.nodes <= max_nodes:
            raise ValueError("complete structural node budget exceeded; no prefix accepted")
        self.source_allocs = {int(key): value for key, value in before["allocations"].items()}
        self.target_allocs = {int(key): value for key, value in after["allocations"].items()}
        self.env, self.slots, self.slot_of_target, self.live_targets = {}, [], {}, set()
        self.epochs, self.instructions, self.initializers = [], [], {}
        owners = {tuple(value["path"][:-1]) for value in self.source_allocs.values()}
        if len(owners) != 1:
            raise ValueError("one complete direct allocation-owner block is required")
        self.owner = next(iter(owners))
        self.epoch_of = {}
        for identity, facts in sorted(self.source_allocs.items(), key=lambda item: item[1]["path"]):
            self.analyze(identity, facts)
        if not self.epochs:
            raise ValueError("no complete eligible logical allocation epoch")

    def collect(self, op, path, operations, definitions):
        _metadata(op)
        self.nodes += 1
        operations[path] = op
        for identity, kind in op["results"]:
            if identity in definitions:
                raise ValueError("duplicate actual SSA definition")
            definitions[identity] = (op, kind)
        for ri, region in enumerate(op["regions"]):
            for bi, block in enumerate(region):
                if set(block) != {"arguments", "operations"}:
                    raise ValueError("unknown block fields")
                for identity, kind in block["arguments"]:
                    if identity in definitions:
                        raise ValueError("duplicate actual SSA argument")
                    definitions[identity] = (None, kind)
                for oi, child in enumerate(block["operations"]):
                    self.collect(child, (*path, ri, bi, oi), operations, definitions)

    @staticmethod
    def supported_allocation(op, facts):
        if (
            op["name"] != "memref.alloc"
            or op["attributes"]
            or op["operands"]
            or op["regions"]
            or len(op["results"]) != 1
            or set(dict(op["properties"])) - {"alignment", "operandSegmentSizes"}
            or not facts["contiguous"]
            or not facts["default_memory_space"]
            or facts["dynamic_sizes"]
            or facts["symbol_operands"]
            or not facts["shape"]
            or any(type(x) is not int or x <= 0 for x in facts["shape"])
            or facts["element_bits"] not in (8, 16, 32, 64)
            or type(facts["alignment"]) is not int
            or facts["alignment"] <= 0
            or facts["alignment"] & (facts["alignment"] - 1)
        ):
            raise ValueError("unsupported allocation shape/type/layout/alignment/space or metadata")

    def constant(self, value):
        op, kind = self.source_defs.get(value, (None, None))
        if (
            op is None
            or op["name"] != "arith.constant"
            or kind != "index"
            or set(dict(op["properties"])) != {"value"}
            or op["attributes"]
            or op["operands"]
            or op["regions"]
            or type(op["index_constant"]) is not int
            or op["index_constant"] < 0
        ):
            raise ValueError("this mode requires static nonnegative access/control indices")
        return op["index_constant"]

    @staticmethod
    def standard_yield(op):
        if op["name"] != "scf.yield" or any(
            op[key] for key in ("attributes", "properties", "operands", "results", "regions")
        ):
            raise ValueError("initializer requires exact standard empty yields")

    def initializer(self, op, path, buffer, shape):
        ivs, controls = [], []
        current, current_path = op, path
        pure = {tuple(value) for value in self.before["pure_paths"]}
        for extent in shape:
            if (
                current["name"] != "scf.for"
                or current["attributes"]
                or current["properties"]
                or current["results"]
                or len(current["operands"]) != 3
            ):
                raise ValueError("initializer is not complete rectangular pure synchronous control")
            lower, upper, step = [self.constant(value) for value in current["operands"]]
            controls.append([lower, upper, step])
            if (lower, upper, step) != (0, extent, 1):
                raise ValueError("initializer misses/wrongly indexes a coordinate")
            body = _one_block(current)
            if _types(body["arguments"]) != ["index"] or len(body["operations"]) < 2:
                raise ValueError("complete initializer body/IV/yield required")
            ivs.append(body["arguments"][0][0])
            self.standard_yield(body["operations"][-1])
            for index, prefix in enumerate(body["operations"][:-2]):
                prefix_path = (*current_path, 0, 0, index)
                if (
                    prefix_path not in pure
                    or prefix["regions"]
                    or any(value in self.source_allocs for value in prefix["operands"])
                ):
                    raise ValueError("initializer prefix is not admitted pure buffer-independent scalar computation")
            next_index = len(body["operations"]) - 2
            current, current_path = body["operations"][next_index], (*current_path, 0, 0, next_index)
        if (
            current["name"] != "memref.store"
            or current["attributes"]
            or current["properties"]
            or current["regions"]
            or current["results"]
            or current["operands"][1:] != [buffer, *ivs]
        ):
            raise ValueError("complete initializer must store once to exactly all loop IVs")
        return dict(
            path=list(path),
            controls=controls,
            store_path=list(current_path),
            indices=current["operands"][2:],
            loop_ivs=ivs,
            value=current["operands"][0],
            body=_key(op),
        )

    def analyze(self, identity, facts):
        op = self.source_defs[identity][0]
        self.supported_allocation(op, facts)
        users = [(path, user) for path, user in self.source_ops.items() if identity in user["operands"]]
        frees, accesses = [], []
        for path, user in users:
            if path[: len(self.owner)] != self.owner:
                raise ValueError("allocation use escapes owner scope")
            direct = len(path) == len(self.owner) + 1
            if user["name"] == "memref.dealloc" and direct and user["operands"] == [identity]:
                if any(user[key] for key in ("attributes", "properties", "results", "regions")):
                    raise ValueError("explicit free has unknown semantics/metadata")
                frees.append(path)
            elif user["name"] in ("memref.load", "memref.store", "memref.dim"):
                position = 1 if user["name"] == "memref.store" else 0
                if (
                    user["operands"][position] != identity
                    or user["attributes"]
                    or user["properties"]
                    or user["regions"]
                ):
                    raise ValueError("private access has unknown metadata or alias role")
                accesses.append((path, user))
            else:
                raise ValueError("private buffer alias/view/escape/unknown use refuses")
        if len(frees) != 1 or not accesses:
            raise ValueError("one direct explicit free and full initializer/accesses required")
        start, end = facts["path"][-1], frees[0][-1]
        positions = [path[len(self.owner)] for path, _op in accesses]
        first = min(positions)
        if not start < first <= max(positions) < end:
            raise ValueError("use outside explicit allocation lifetime")
        init_path = (*self.owner, first)
        init = self.initializer(self.source_ops[init_path], init_path, identity, facts["shape"])
        initial_uses = [(path, op) for path, op in accesses if path[len(self.owner)] == first]
        if len(initial_uses) != 1 or initial_uses[0][0] != tuple(init["store_path"]):
            raise ValueError("initial operation also reads or aliases this buffer")
        for path, user in accesses:
            if path[len(self.owner)] == first:
                continue
            if len(path) != len(self.owner) + 1:
                raise ValueError("noninitializer nested private access requires a separate semantics route")
            if user["name"] == "memref.dim":
                axis = self.constant(user["operands"][1])
                if axis >= len(facts["shape"]):
                    raise ValueError("dimension axis out of range")
            else:
                indices = user["operands"][2:] if user["name"] == "memref.store" else user["operands"][1:]
                coordinates = [self.constant(value) for value in indices]
                if len(coordinates) != len(facts["shape"]) or any(
                    index >= size for index, size in zip(coordinates, facts["shape"], strict=True)
                ):
                    raise ValueError("private direct access coordinate out of bounds")
        epoch = len(self.epochs)
        self.epoch_of[identity] = epoch
        self.initializers[init_path] = epoch
        self.epochs.append(
            dict(
                source_value=identity,
                slot=None,
                shape=facts["shape"],
                elementBits=facts["element_bits"],
                alignment=facts["alignment"],
                allocation=start,
                initialization=first,
                release=end,
                initializer=init,
                uses=[dict(path=list(path), operation=_key(user)) for path, user in users],
            )
        )

    def bind(self, first, second, env):
        if _types(first) != _types(second):
            raise ValueError("actual SSA argument/result arity/type changed")
        for source, target in zip(_ids(first), _ids(second), strict=True):
            if source in env:
                raise ValueError("source SSA definition rebound")
            env[source] = target

    def operands(self, first, second, env):
        if len(first) != len(second):
            raise ValueError("actual operand count changed")
        for source, target in zip(first, second, strict=True):
            if source in self.source_allocs and source not in env:
                if target not in self.slot_of_target or target not in self.live_targets:
                    raise ValueError("removed logical allocation does not map to a live actual prior physical buffer")
                a, b = self.source_allocs[source], self.target_allocs[target]
                if any(
                    a[key] != b[key]
                    for key in ("type", "shape", "element_bits", "alignment", "contiguous", "default_memory_space")
                ):
                    raise ValueError("reused buffer type/layout/alignment/shape changed")
                env[source] = target
                self.epochs[self.epoch_of[source]]["slot"] = self.slot_of_target[target]
            if env.get(source) != target:
                raise ValueError("actual retained operand/capture or same-typed private buffer changed")

    def operation(self, left, right, env, path=()):
        if _metadata(left) != _metadata(right):
            raise ValueError("actual retained operation/metadata changed")
        self.operands(left["operands"], right["operands"], env)
        self.bind(left["results"], right["results"], env)
        if len(left["regions"]) != len(right["regions"]):
            raise ValueError("actual retained region changed")
        a_regions, b_regions = [], []
        for ri, (a, b) in enumerate(zip(left["regions"], right["regions"], strict=True)):
            if len(a) != len(b):
                raise ValueError("actual retained block boundary changed")
            a_blocks, b_blocks = [], []
            for bi, (one, two) in enumerate(zip(a, b, strict=True)):
                x, y = self.block(one, two, dict(env), (*path, ri, bi))
                a_blocks.append(x)
                b_blocks.append(y)
            a_regions.append(a_blocks)
            b_regions.append(b_blocks)
        a = dict(_metadata(left), operands=left["operands"], results=left["results"], regions=a_regions)
        b = dict(_metadata(right), operands=left["operands"], results=left["results"], regions=b_regions)
        return a, b

    def block(self, left, right, env, path):
        self.bind(left["arguments"], right["arguments"], env)
        first, second, cursor = [], [], 0
        for index, op in enumerate(left["operations"]):
            op_path = (*path, index)
            candidate = right["operations"][cursor] if cursor < len(right["operations"]) else None
            if path == self.owner and op["name"] == "memref.alloc":
                identity = op["results"][0][0]
                epoch = self.epoch_of[identity]
                retained = candidate is not None and candidate["name"] == "memref.alloc"
                if retained:
                    self.supported_allocation(candidate, self.target_allocs[candidate["results"][0][0]])
                    self.operation(op, candidate, env, op_path)
                    target = candidate["results"][0][0]
                    self.slot_of_target[target] = len(self.slots)
                    self.slots.append(target)
                    self.live_targets.add(target)
                    self.epochs[epoch]["slot"] = self.slot_of_target[target]
                    cursor += 1
                a = b = dict(private_allocate=epoch)
                self.instructions.append(dict(kind="allocate", object=epoch, retained=retained))
                self.epochs[epoch]["retainedAllocation"] = retained
            elif path == self.owner and op["name"] == "memref.dealloc":
                identity = op["operands"][0]
                if identity not in self.epoch_of or identity not in env:
                    raise ValueError("free of unknown/uninitialized logical object")
                epoch = self.epoch_of[identity]
                target = env[identity]
                retained = (
                    candidate is not None
                    and candidate["name"] == "memref.dealloc"
                    and candidate["operands"] == [target]
                )
                if retained:
                    self.operation(op, candidate, env, op_path)
                    if target not in self.live_targets:
                        raise ValueError("physical free outside actual allocation lifetime")
                    self.live_targets.remove(target)
                    cursor += 1
                a = b = dict(private_release=epoch)
                self.instructions.append(dict(kind="release", object=epoch, retained=retained))
                self.epochs[epoch]["retainedRelease"] = retained
            else:
                if candidate is None:
                    raise ValueError("actual retained operation missing")
                a, b = self.operation(op, candidate, env, op_path)
                cursor += 1
                if path == self.owner:
                    if op_path in self.initializers:
                        epoch = self.initializers[op_path]
                        self.instructions.append(dict(kind="initialize", object=epoch, key=_key(a)))
                        self.epochs[epoch]["initializer"]["target_body"] = _key(b)
                    else:
                        private = [self.epoch_of[value] for value in op["operands"] if value in self.epoch_of]
                        names = {"memref.load": "load", "memref.store": "store", "memref.dim": "dimension"}
                        if private:
                            if len(private) != 1 or op["name"] not in names:
                                raise ValueError("unmodeled private retained operation")
                            self.instructions.append(dict(kind=names[op["name"]], object=private[0], key=_key(a)))
                        else:
                            self.instructions.append(dict(kind="retained", key=_key(a)))
            first.append(a)
            second.append(b)
        if cursor != len(right["operations"]):
            raise ValueError("extra actual operation outside complete boundary")
        return {"arguments": left["arguments"], "operations": first}, {
            "arguments": left["arguments"],
            "operations": second,
        }


def project_exports(before, after, *, index_bits, max_nodes):
    decoder = Decoder(before, after, index_bits=index_bits, max_nodes=max_nodes)
    source, target = decoder.operation(before["tree"], after["tree"], decoder.env)
    if decoder.live_targets or any(epoch["slot"] is None for epoch in decoder.epochs):
        raise ValueError("missing final physical free or unbound logical epoch")
    if len(decoder.slots) >= len(decoder.epochs):
        raise ValueError("identity/declined storage is not a changed reuse qualification")
    return dict(
        bits=index_bits,
        sourceContext=_key(source),
        targetContext=_key(target),
        objects=[
            {key: epoch[key] for key in ("slot", "shape", "elementBits", "alignment")} for epoch in decoder.epochs
        ],
        instructions=decoder.instructions,
        epochs=decoder.epochs,
        structural_nodes=decoder.nodes,
        actual_slot_values=decoder.slots,
        allocator_mode="total_private_unobservable",
        initializer_semantics="partial_complete_rectangular_pure",
        raw_SCF_bridge_proved=False,
    )
