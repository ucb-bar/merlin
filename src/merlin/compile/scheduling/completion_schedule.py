"""Conservative completion decisions over existing asynchronous calls and loops.

No duration is assigned to event-driven work. The target must supply concrete
completion waits and spell their token semantics during emission. Every launch
is waited immediately, so tokens never remain in flight across loop boundaries.
"""

from dataclasses import replace

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Block, Region

from merlin.sched.ir import Call, Kernel, Loop
from merlin.sched.ir.kernel import check_structure
from merlin.xdsl_dialects.schedule import AwaitCompletionOp, BindOp, HandleType, PlanOp


def _launches(kernel):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("well-formed completion source required")
    launches = {}

    def walk(body, prefix):
        for i, statement in enumerate(body):
            path = (*prefix, str(i))
            if type(statement) is Loop:
                walk(statement.body, path)
            elif type(statement) is not Call:
                raise ValueError("unknown completion source statement")
            elif statement.produces is not None:
                if type(statement.produces) is not str or not statement.produces:
                    raise ValueError("nonempty source completion token required")
                launches["call:" + "/".join(path)] = statement

    walk(kernel.body, ())
    return launches


def make_completion_plan(kernel):
    """Bind one mandatory immediate completion decision to each source launch."""
    operations = []
    for target in _launches(kernel):
        bind = BindOp(result_types=[HandleType()], properties={"target": StringAttr(target)})
        operations.extend((bind, AwaitCompletionOp(operands=[bind.handle])))
    plan = PlanOp(
        properties={"sym_name": StringAttr("complete_each_launch")},
        attributes={"kernel_sha256": StringAttr(kernel.digest())},
        regions=[Region(Block(operations))],
    )
    plan.verify()
    return plan


def apply_completion_plan(kernel, plan, *, wait):
    """Insert target wait calls in place; preserve counted loops and annotations.

    ``wait(launch)`` must return a concrete call carrying exactly the launch's
    completion token in ``awaits``. Target-specific verification still proves its
    physical completion semantics; a matching token alone is not that proof.
    """
    launches = _launches(kernel)
    if type(plan) is not PlanOp:
        raise ValueError("completion decisions require schedule.plan")
    plan.verify()
    if len(plan.body.blocks) != 1 or plan.attributes != {"kernel_sha256": StringAttr(kernel.digest())}:
        raise ValueError("completion plan differs from current kernel")
    handles, selected = {}, set()
    for op in plan.body.block.ops:
        if op.attributes:
            raise ValueError("unknown completion decision attributes")
        if type(op) is BindOp:
            target = op.target.data
            if target not in launches or target in handles.values():
                raise ValueError("unknown or repeated completion binding")
            handles[op.handle] = target
        elif type(op) is AwaitCompletionOp:
            target = handles.get(op.target)
            if target is None or target in selected:
                raise ValueError("unbound or repeated completion wait")
            selected.add(target)
        else:
            raise ValueError("unconsumed completion decision")
    if selected != set(launches):
        raise ValueError("every asynchronous launch requires a completion wait")

    def rewrite(body, prefix):
        result = []
        for i, statement in enumerate(body):
            path = (*prefix, str(i))
            if type(statement) is Loop:
                result.append(replace(statement, body=rewrite(statement.body, path)))
                continue
            result.append(statement)
            if "call:" + "/".join(path) in selected:
                completion = wait(statement)
                if (
                    type(completion) is not Call
                    or completion.produces is not None
                    or completion.awaits != (statement.produces,)
                ):
                    raise ValueError("target wait must await exactly the source completion token")
                result.append(completion)
        return tuple(result)

    result = replace(kernel, body=rewrite(kernel.body, ()))
    if check_structure(result):
        raise ValueError("target wait produced malformed completion code")
    return result
