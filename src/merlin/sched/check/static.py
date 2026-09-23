"""G0: static legality of a kernel against a target's instruction set, before anything runs.

Three layers, cheapest first:

1. target-free structure (``merlin.sched.ir.kernel.check_structure``);
2. every call names an instruction of the set, with exactly its operands, of the right kinds;
3. every DYNAMIC instance, in program order: flags are 0/1, the target's ``check`` holds on the concrete
   operand values (with one ``state`` dict threaded through the kernel for cross-instruction rules), and
   every byte extent the target's ``footprint`` reports lies inside its tensor argument, with no write
   into a read-only argument.

Layer 3 enumerates instances, so its cost is the kernel's instruction count; schedules issue loop
macro-instructions, so that count is small. ``max_instances`` bounds it and says so when hit.
"""

from __future__ import annotations

from merlin.sched.ir.expr import Expr
from merlin.sched.ir.kernel import NULL, Kernel, KernelError, Loop, Ptr, check_structure, concretize, instances
from merlin.sched.isa import InstructionSet

_KIND_OK = {
    "int": lambda v: isinstance(v, Expr),
    "flag": lambda v: isinstance(v, Expr),
    "ptr": lambda v: isinstance(v, Ptr) or v is NULL,
    "float": lambda v: isinstance(v, float),
}


def _static_calls(body):
    for s in body:
        if isinstance(s, Loop):
            yield from _static_calls(s.body)
        else:
            yield s


def check_kernel(
    kernel: Kernel, iset: InstructionSet, *, max_instances: int = 2_000_000, max_errors: int = 50
) -> list[str]:
    errors = check_structure(kernel)
    for c in _static_calls(kernel.body):
        d = iset.instrs.get(c.instr)
        if d is None:
            errors.append(f"unknown instruction {c.instr!r}")
            continue
        names = tuple(k for k, _ in c.args)
        if names != d.operand_names():
            errors.append(f"{c.instr}: operands {names} != {d.operand_names()}")
            continue
        for o, (_, v) in zip(d.operands, c.args):
            if not _KIND_OK[o.kind](v):
                errors.append(f"{c.instr}.{o.name}: expected a {o.kind} operand, got {v!r}")
    if errors:
        return errors
    state: dict = {}
    for n, (c, env) in enumerate(instances(kernel)):
        if n >= max_instances:
            errors.append(f"stopped after {max_instances} instances; the rest are unchecked")
            break
        d = iset.instr(c.instr)
        where = f"{c.instr}@{env}"
        try:
            values = concretize(c, env)
        except KernelError as exc:
            errors.append(f"{where}: {exc}")
            continue
        for o in d.operands:
            if o.kind == "flag" and values[o.name] not in (0, 1):
                errors.append(f"{where}: flag {o.name} = {values[o.name]}")
        if d.check is not None:
            errors.extend(f"{where}: {m}" for m in d.check(values, state))
        if d.footprint is not None:
            for operand, nbytes, access in d.footprint(values):
                p = values[operand]
                if p is None or nbytes == 0:
                    continue
                t = kernel.tensor(p.tensor)
                if p.offset < 0 or p.offset + nbytes > t.nbytes:
                    errors.append(
                        f"{where}: {operand} touches bytes [{p.offset}, {p.offset + nbytes}) "
                        f"of {t.name} ({t.nbytes} bytes)"
                    )
                if access == "write" and t.access == "read":
                    errors.append(f"{where}: {operand} writes read-only {t.name}")
        if len(errors) >= max_errors:
            errors.append("too many errors; stopped")
            return errors
    else:
        if iset.finish is not None:
            errors.extend(f"end of kernel: {m}" for m in iset.finish(state))
    return errors
