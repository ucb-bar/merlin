"""Native deallocation placement and iteration-sensitive ownership checks.

The proof is deliberately closed: positive static serial loops, fresh direct
allocations, false entry/true backedge ownership, synchronous intrinsic uses,
and one final same-block release. Calls and unknown effects remain barriers.
This checks allocation generations; it neither moves storage nor admits arena
reuse for allocation cycles. The native runner and public checker share it.
"""

RUNNER_PRELUDE = r"""
_OBL_CALL_ROSTER = ()
_OBL_CALL_OUTPUT = None


def _obl_callee(op):
    if op.name != 'func.call' or op.results:
        return None
    try:
        return ir.FlatSymbolRefAttr(op.attributes['callee']).value
    except (ValueError, KeyError):
        return None


def _obl_borrowed_calls(module):
    if not _OBL_CALL_ROSTER:
        return set(), set(), []
    functions, supported, violations = {}, set(), []
    for op in module.body.operations:
        if op.operation.name != 'func.func':
            continue
        symbol = ir.StringAttr(op.attributes['sym_name']).value
        if symbol in functions:
            return set(), set(), ['borrowed ABI: duplicate current function symbol ' + symbol]
        functions[symbol] = op.operation
    for record in _OBL_CALL_ROSTER:
        try:
            effects = record['effects']
            if (not effects['qualification_identity'] or any(effects.get(k) is not True for k in (
                    'no_retained_aliases', 'no_releases', 'complete_before_return'))):
                raise ValueError('original complete borrowed effects are absent')
            wrapper_name, borrowed = record['wrapper_symbol'], record['borrowed_symbol']
            types = tuple(ir.Type.parse(t, context=module.context) for t in record['argument_types'])
            if any(not _obl_numeric_type(ir.MemRefType(t).element_type) for t in types):
                raise ValueError('borrowed ABI requires ranked numeric memory arguments')
            result_index = record['result_argument']
            declaration = functions.get(borrowed)
            if declaration is None:
                raise ValueError('original borrowed declaration is missing: ' + borrowed)
            typ = ir.FunctionType(ir.TypeAttr(declaration.attributes['function_type']).value)
            if (tuple(typ.inputs) != types or typ.results or any(r.blocks for r in declaration.regions)
                    or 'llvm.emit_c_interface' not in declaration.attributes):
                raise ValueError('current borrowed declaration/ABI differs: ' + borrowed)
            wrapper = functions.get(wrapper_name)
            if wrapper is not None:
                typ = ir.FunctionType(ir.TypeAttr(wrapper.attributes['function_type']).value)
                if (tuple(typ.inputs) != (*types, types[result_index]) or typ.results
                        or len(wrapper.regions) != 1 or len(wrapper.regions[0].blocks) != 1):
                    raise ValueError('current fresh-writer wrapper ABI differs: ' + wrapper_name)
                block = wrapper.regions[0].blocks[0]
                body = tuple(op.operation for op in block.operations)
                expected = tuple(block.arguments[len(types) if i == result_index else i]
                                 for i in range(len(types)))
                if (len(body) != 2 or _obl_callee(body[0]) != borrowed
                        or tuple(body[0].operands) != expected
                        or body[1].name != 'func.return' or body[1].operands):
                    raise ValueError('current fresh-writer forwarding body differs: ' + wrapper_name)
                supported.add(wrapper_name)
            supported.add(borrowed)
        except (ValueError, KeyError, IndexError) as error:
            violations.append('borrowed ABI: ' + str(error))
    if violations:
        return set(), set(), violations
    # Bind grants to actual operations in checked root functions. A matching
    # flat spelling in a nested symbol table grants no permission.
    uses = set()
    scopes = ('scf.for', 'scf.if', 'linalg.generic', 'linalg.fill')
    for function in functions.values():
        for op in _obl_ops(function):
            name = _obl_callee(op)
            if name not in supported:
                continue
            parent = op.block.owner.operation
            while parent.name in scopes:
                parent = parent.block.owner.operation
            if parent != function:
                continue
            typ = ir.FunctionType(ir.TypeAttr(functions[name].attributes['function_type']).value)
            if tuple(v.type for v in op.operands) == tuple(typ.inputs):
                uses.add(op.operation)
    return uses, supported, []


def _obl_known_use(op, calls):
    return op.name in _OBL_VIEWS | _OBL_USES or op.operation in calls


def _obl_write_observation(module, violations):
    if _OBL_CALL_OUTPUT is None:
        return
    import hashlib, json
    text = module.operation.get_asm(print_generic_op_form=True)
    with open(_OBL_CALL_OUTPUT['checkpoint'], 'w') as stream:
        stream.write(text)
    _, accepted, problems = _obl_borrowed_calls(module)
    report = dict(schema='merlin.borrowed_buffer_lifetime.v1',
                  status='checked' if not violations and not problems else 'refused',
                  original=_OBL_CALL_OUTPUT['original'], violations=list(violations),
                  checkpoint_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  accepted_symbols=sorted(accepted),
                  scope='conditional original borrowed effects and current typed forwarding bodies')
    with open(_OBL_CALL_OUTPUT['report'], 'w') as stream:
        json.dump(report, stream, sort_keys=True)
    with open(_OBL_CALL_OUTPUT['report'], 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    print('OK borrowed_buffer_lifetime', digest, report['checkpoint_sha256'])


_OBL_VIEWS = frozenset((
    'memref.cast', 'memref.subview', 'memref.reinterpret_cast',
    'memref.collapse_shape', 'memref.expand_shape', 'memref.extract_strided_metadata',
))
_OBL_USES = frozenset(('memref.copy', 'memref.load', 'memref.store', 'memref.dim',
                       'linalg.generic', 'linalg.fill'))
_OBL_SCALARS = frozenset((
    'arith.constant', 'arith.addi', 'arith.subi', 'arith.muli', 'arith.divsi',
    'arith.divui', 'arith.remsi', 'arith.remui', 'arith.andi', 'arith.ori',
    'arith.xori', 'arith.shli', 'arith.shrsi', 'arith.shrui', 'arith.cmpi',
    'arith.cmpf', 'arith.select', 'arith.index_cast', 'arith.index_castui',
    'arith.extsi', 'arith.extui', 'arith.trunci', 'arith.sitofp', 'arith.uitofp',
    'arith.fptosi', 'arith.fptoui', 'arith.extf', 'arith.truncf', 'arith.bitcast',
    'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.negf',
    'arith.maximumf', 'arith.minimumf', 'arith.maxnumf', 'arith.minnumf',
    'arith.maxsi', 'arith.minsi', 'arith.maxui', 'arith.minui',
    'math.exp', 'math.exp2', 'math.log', 'math.log2', 'math.sqrt', 'math.rsqrt',
    'math.absf', 'math.ceil', 'math.floor', 'math.roundeven', 'math.fma',
    'linalg.index', 'linalg.yield',
))


def _obl_constant(value):
    if not isinstance(value, ir.OpResult) or value.owner.name != 'arith.constant':
        return None
    if str(value.type) not in ('index', 'i1'):
        return None
    return ir.IntegerAttr(value.owner.attributes['value']).value


def _obl_ops(operation):
    for region in operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield inner.operation
                yield from _obl_ops(inner.operation)


def _obl_numeric_type(typ):
    if str(typ) in ('index', 'f16', 'bf16', 'f32', 'f64'):
        return True
    try:
        ir.IntegerType(typ)
        return True
    except ValueError:
        return False


def _obl_memory_types(op):
    for value in (*op.operands, *op.results):
        if _memref_typed(value):
            try:
                element = ir.MemRefType(value.type).element_type
            except ValueError:
                raise ValueError('unranked or unsupported buffer representation') from None
            if not _obl_numeric_type(element):
                raise ValueError('buffer element may contain an escaping handle')
        elif not _obl_numeric_type(value.type):
            raise ValueError('unsupported nonnumeric memory operand or result')
    if op.name == 'memref.store' and _memref_typed(op.operands[0]):
        raise ValueError('buffer cannot be a stored payload')
    if op.name == 'linalg.fill' and any(_memref_typed(v) for v in op.operands[:-1]):
        raise ValueError('buffer cannot be a fill payload')
    if op.name == 'memref.load' and any(_memref_typed(v) for v in op.results):
        raise ValueError('buffer handle cannot be loaded')


def _obl_alias_uses(value):
    # Each view preserves its own provenance. A subview may be read, but it is
    # not an allocation owner that can be deallocated. Only casts/base extraction
    # of an owner preserve release authority.
    aliases, pending, uses = {value: True}, [value], {}
    while pending:
        current = pending.pop()
        for use in current.uses:
            op = use.owner.operation
            uses[op.operation] = op
            if op.name not in _OBL_VIEWS:
                continue
            if use.operand_number != 0:
                raise ValueError('buffer used as a non-source view operand')
            owner = aliases[current] and op.name in ('memref.cast', 'memref.extract_strided_metadata')
            for result in op.results:
                if not _memref_typed(result):
                    continue
                if result not in aliases:
                    aliases[result] = owner
                    pending.append(result)
    return aliases, tuple(uses.values())


def _obl_release_value(op, aliases):
    return (op.name == 'memref.dealloc' and len(op.operands) == 1
            and aliases.get(op.operands[0], False))


def _obl_scalar_loop(op):
    # This admits the serial container, never its captured effects. Descendants
    # remain in _obl_prove's complete effect walk and previous-owner uses still
    # have to precede their outer guarded release. Numeric iteration state
    # introduces no buffer generation, even for dynamic or zero trip counts.
    if (len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or len(op.operands) < 3 or len(op.attributes)):
        raise ValueError('unsupported nested scalar loop structure or attributes')
    body = op.regions[0].blocks[0]
    if any(not _obl_numeric_type(v.type)
           for v in (*op.operands, *op.results, *body.arguments)):
        raise ValueError('nested serial loop must carry only numeric scalars')
    operations = tuple(body.operations)
    if (not operations or operations[-1].operation.name != 'scf.yield'
            or tuple(v.type for v in operations[-1].operands)
            != tuple(v.type for v in op.results)):
        raise ValueError('unsupported nested scalar loop yield')
    op.verify()


def _obl_prove(loop, calls):
    loop = loop.operation
    body = loop.regions[0].blocks[0]
    operations = tuple(op.operation for op in body.operations)
    terminal = operations[-1]
    incoming, carried = tuple(loop.operands[3:]), tuple(body.arguments[1:])
    yielded = tuple(terminal.operands)
    changed = tuple(i for i, arg in enumerate(carried)
                    if _memref_typed(arg) and yielded[i] != arg)
    if not changed:
        return None
    _obl_memory_types(loop)
    if any(_memref_typed(v) for v in terminal.operands) and terminal.operation.name != 'scf.yield':
        raise ValueError('unsupported loop terminator')
    lo, hi, step = (_obl_constant(v) for v in loop.operands[:3])
    if lo is None or hi is None or step is None or step <= 0 or hi <= lo:
        raise ValueError('fresh carried ownership requires a statically positive serial trip count')
    fresh = {}
    for i in changed:
        value = yielded[i]
        if (not isinstance(value, ir.OpResult) or value.owner.name != 'memref.alloc'
                or value.owner.operation.block != body or len(value.owner.results) != 1):
            raise ValueError('each replaced carried buffer must yield its own direct body allocation')
        fresh[i] = value.owner.operation
    if len({op.operation for op in fresh.values()}) != len(fresh):
        raise ValueError('replaced carried buffers share a fresh owner')
    flags = {arg for i, arg in enumerate(carried)
             if str(arg.type) == 'i1' and _obl_constant(incoming[i]) == 0
             and _obl_constant(yielded[i]) == -1}
    # MLIR's integer attribute prints boolean true as -1 (signless i1).
    if not flags:
        raise ValueError('ownership must enter false and yield true')
    position = {op.operation: i for i, op in enumerate(operations)}
    all_inside = {op.operation: op for op in _obl_ops(loop)}
    guards = {}
    for op in operations:
        if op.operation.name != 'scf.if':
            continue
        if op.results or len(op.operands) != 1 or op.operands[0] not in flags:
            raise ValueError('only the original ownership flag may guard previous releases')
        then = tuple(op.regions[0].blocks[0].operations)
        if any(inner.operation.name != 'memref.dealloc' for inner in then[:-1]):
            raise ValueError('ownership branch must contain only previous-buffer releases')
        if then[-1].operation.name != 'scf.yield' or then[-1].operands:
            raise ValueError('unsupported ownership branch terminator')
        for region in op.regions[1:]:
            for block in region.blocks:
                if any(inner.operation.name != 'scf.yield' or inner.operands for inner in block.operations):
                    raise ValueError('ownership false branch must not release or use buffers')
        for inner in then[:-1]:
            guards[inner.operation] = op
    releases = set()
    for i in changed:
        aliases, uses = _obl_alias_uses(carried[i])
        frees = [op for op in uses if op.name == 'memref.dealloc']
        if len(frees) != 1 or not _obl_release_value(frees[0], aliases):
            raise ValueError('each previous buffer requires exactly one owner release')
        release = frees[0]
        guard = guards.get(release.operation)
        if guard is None:
            raise ValueError('previous buffer release is not guarded by its ownership flag')
        releases.add(release.operation)
        for use in uses:
            if use.operation not in all_inside:
                raise ValueError('previous generation escapes the loop')
            if use.operation == release.operation:
                continue
            if not _obl_known_use(use, calls):
                raise ValueError('unknown or escaping previous-buffer use: ' + use.name)
            ancestor = use
            while ancestor.block != body:
                ancestor = ancestor.block.owner.operation
            if position[ancestor.operation] >= position[guard.operation]:
                raise ValueError('previous buffer or view used after its guarded release')
        aliases, uses = _obl_alias_uses(yielded[i])
        for use in uses:
            if use.operation not in all_inside:
                raise ValueError('fresh generation escapes the loop')
            if use.operation == terminal.operation:
                if tuple(j for j, v in enumerate(yielded) if v in aliases) != (i,):
                    raise ValueError('fresh owner yielded in more than one position')
            elif not _obl_known_use(use, calls):
                raise ValueError('unknown, releasing or escaping fresh-buffer use: ' + use.name)
    if releases != set(guards):
        raise ValueError('ownership branch releases an unrelated or duplicate buffer')
    # Close every operation/effect, including operations with no tracked operand
    # and scalar regions that could capture a buffer without a parent operand.
    for op in all_inside.values():
        name = op.operation.name
        _obl_memory_types(op)
        if name in _OBL_SCALARS | _OBL_VIEWS | _OBL_USES:
            if name in _OBL_SCALARS and any(_memref_typed(v) for v in (*op.operands, *op.results)):
                raise ValueError('memref escapes through a scalar operation')
            continue
        if op.operation in calls:
            continue
        if name == 'scf.for':
            _obl_scalar_loop(op)
            continue
        if name in ('memref.alloc', 'scf.yield'):
            continue
        if name == 'scf.if' and op.block == body:
            continue
        if name == 'memref.dealloc':
            if op.operation in releases:
                continue
            value = op.operands[0]
            if (op.block == body and isinstance(value, ir.OpResult)
                    and value.owner.name == 'memref.alloc' and value.owner.operation.block == body
                    and value.owner.operation not in {a.operation for a in fresh.values()}):
                continue
        raise ValueError('unknown or unsupported loop effect: ' + name)
    # The final generation must have one local release after all synchronous
    # observations. Returning ownership or conditional release needs another proof.
    outer = tuple(loop.block.operations)
    outer_position = {op.operation: i for i, op in enumerate(outer)}
    for i in changed:
        aliases, uses = _obl_alias_uses(loop.results[i])
        frees = [op for op in uses if op.name == 'memref.dealloc']
        if (len(frees) != 1 or frees[0].block != loop.block
                or not _obl_release_value(frees[0], aliases)):
            raise ValueError('final fresh generation requires exactly one same-block owner release')
        for use in uses:
            if use.block != loop.block:
                raise ValueError('final generation escapes into unsupported control flow')
            if use.operation == frees[0].operation:
                continue
            _obl_memory_types(use)
            if not _obl_known_use(use, calls):
                raise ValueError('unknown or escaping final-buffer use: ' + use.name)
            for inner in _obl_ops(use):
                _obl_memory_types(inner)
                if inner.name not in _OBL_SCALARS or any(
                        _memref_typed(v) for v in (*inner.operands, *inner.results)):
                    raise ValueError('unknown or capturing final-buffer consumer effect: ' + inner.name)
            if outer_position[use.operation] >= outer_position[frees[0].operation]:
                raise ValueError('final buffer or view used after release')
    return fresh


def _owned_buffer_loops(order, calls):
    proved, violations = {}, []
    for loop in order:
        if loop.operation.name != 'scf.for':
            continue
        try:
            fact = _obl_prove(loop, calls)
            if fact:
                proved[loop.operation] = fact
        except ValueError as error:
            # Refuse unsupported replacements, rather than relying on the old
            # allocation-site union to accidentally produce a diagnostic.
            violations.append('loop-carried buffer ownership: ' + str(error))
    return proved, violations
"""


# --- The use-after-free check that guards the sinking stage ---------------------------------
#
# Sinking a dealloc is the one edit here whose failure mode is WRONG NUMBERS, not a broken build: a
# free moved before a use is a use-after-free, and with the static arena bound on top of it the
# freed bytes are immediately handed to another buffer. `optimize-allocation-liveness` decides where
# a free may go from upstream's `BufferViewFlowAnalysis`; this re-checks the RESULT, on the IR the
# stage actually produced, in terms that do not depend on that analysis being right.
#
# The rule: for every `memref.alloc` whose `memref.dealloc` the stage placed in the alloc's OWN block
# (an allocation still freed inside an `scf.if` was not sunk -- upstream placed it, and it is left
# alone), every use of the allocation AND of every value derived from it must come before that free.
#
# "Derived from" is deliberately OVER-approximated: any op result of memref type, and any memref
# block argument of a region, of an op that consumes an aliasing value joins the alias set. An
# over-approximation can only make the check REFUSE a placement it could have allowed; it cannot let
# a real use-after-free through by narrowing what counts as a use. Anything it cannot read -- a use
# whose ancestor chain does not reach the alloc's block -- is reported, never skipped.
# Closed positive serial loops have a separate generation proof: their previous
# and fresh owners are checked inside the loop, and their final result is a new
# outer-scope root. Other operations retain the conservative alias union.
DEALLOC_CHECK_PRELUDE = (
    RUNNER_PRELUDE
    + r"""
def _memref_typed(v):
    return str(v.type).startswith('memref')


def _op_key(o):
    # OpView and Operation both answer `.operation`, and the bindings hand back equal (hashable)
    # objects for the same op however it was reached -- so this is a stable dict key.
    return o.operation


def _dealloc_placement_violations(module):
    # (violations, n_allocs, n_sunk) for a post-deallocation memref module.
    order, block_of, index_of, owner_of_block = [], {}, {}, {}
    counter = [0]

    def visit(block, owner):
        bid = counter[0]
        counter[0] += 1
        owner_of_block[bid] = owner
        for i, op in enumerate(block.operations):
            k = _op_key(op)
            block_of[k], index_of[k] = bid, i
            order.append(op)
            for region in op.regions:
                for b in region.blocks:
                    visit(b, op)

    for op in module.body.operations:
        for region in op.regions:
            for b in region.blocks:
                visit(b, op)

    calls, _, call_violations = _obl_borrowed_calls(module)
    proved, loop_violations = _owned_buffer_loops(order, calls)
    loop_violations.extend(call_violations)
    fresh_sites = {alloc.operation for fact in proved.values() for alloc in fact.values()}
    alias, roots, root_values, uses = {}, {}, {}, {}
    for op in order:
        name = op.operation.name
        if name == 'memref.alloc' and len(op.results) == 1:
            if op.operation in fresh_sites:
                continue  # all iteration-local uses were independently checked
            rid = len(roots)
            roots[rid] = op
            root_values[rid] = op.results[0]
            alias[op.results[0]] = {rid}
            continue
        hit = set()
        for v in op.operands:
            hit |= alias.get(v, set())
        if not hit and op.operation not in proved:
            continue
        for rid in hit:
            uses.setdefault(rid, []).append(op)
        if op.operation in proved:
            fact = proved[op.operation]
            # Non-replaced slots are positional pass-through aliases. Replaced
            # slots have a distinct final-generation owner after positive trips.
            body = op.regions[0].blocks[0]
            for i, (incoming, result) in enumerate(zip(op.operands[3:], op.results)):
                if not _memref_typed(result):
                    continue
                if i in fact:
                    rid = len(roots)
                    roots[rid], root_values[rid] = op, result
                    alias[result] = {rid}
                else:
                    value_roots = alias.get(incoming, set())
                    alias[result] = set(value_roots)
                    alias[body.arguments[i + 1]] = set(value_roots)
            continue
        for r in op.results:
            if _memref_typed(r):
                alias.setdefault(r, set()).update(hit)
        for region in op.regions:
            for b in region.blocks:
                for a in b.arguments:
                    if _memref_typed(a):
                        alias.setdefault(a, set()).update(hit)

    def ancestor_in_block(op, bid):
        o = op
        while True:
            k = _op_key(o)
            if k not in block_of:
                return None
            if block_of[k] == bid:
                return o
            parent = owner_of_block[block_of[k]]
            if parent is None:
                return None
            o = parent

    violations, sunk = list(loop_violations), 0
    for rid, alloc in roots.items():
        abid = block_of[_op_key(alloc)]
        # Moving an allocation outside a loop must also replan its free.
        # A same-block last-use check misses a retained per-iteration free.
        for use in uses.get(rid, []):
            if use.operation.name != 'memref.dealloc':
                continue
            bid = block_of[_op_key(use)]
            while bid != abid:
                parent = owner_of_block[bid]
                if parent is None:
                    break
                if parent.operation.name in ('scf.for', 'scf.while', 'scf.parallel', 'scf.forall',
                                              'affine.for', 'affine.parallel'):
                    violations.append(str(root_values[rid].type) +
                                      ': freed inside repeating control flow outside its allocation lifetime')
                    break
                bid = block_of[_op_key(parent)]
        here = [u for u in uses.get(rid, [])
                if u.operation.name == 'memref.dealloc' and block_of[_op_key(u)] == abid]
        if not here:
            continue                      # freed in a conditional/another block: not sunk
        sunk += 1
        d = min(here, key=lambda o: index_of[_op_key(o)])
        di = index_of[_op_key(d)]
        ty = str(root_values[rid].type)
        for u in uses.get(rid, []):
            if _op_key(u) == _op_key(d):
                continue
            anc = ancestor_in_block(u, abid)
            if anc is None:
                violations.append(ty + ': use by ' + u.operation.name +
                                  ' is outside the block its free was placed in')
            elif index_of[_op_key(anc)] > di:
                violations.append(ty + ': freed at #' + str(di) + ' but used at #' +
                                  str(index_of[_op_key(anc)]) + ' by ' + u.operation.name)
    return violations, len(roots), sunk
"""
)
