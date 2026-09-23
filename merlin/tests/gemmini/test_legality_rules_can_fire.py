"""Every legality rule this target's binding states, fired on a kernel that breaks it.

WHY THIS EXISTS. `sched/AGENT.md` directs that the hand-written LOOP_WS legality checkers in
`gemmini_sched.py` be replaced by legality interpreted from declared semantics. Measured before writing
this file: **delete all four checkers plus the end-of-kernel `finish` hook and `pytest merlin/tests`
stays green.** Roughly sixty lines encoding real hardware constraints -- read off `LoopMatmul.scala`,
each one a wrong answer the device would return without complaint -- were unfalsified by any test. A
migration against no safety net cannot be checked; it can only be believed.

So this is the net, and it is built to be *complete rather than plausible*. A hand-listed table of
"the rules I noticed" is exactly the artifact that rots: someone adds a nineteenth rule, no case covers
it, and the table still reads as coverage. Instead the denominator is **read out of the binding's own
source** -- `_error_sites()` parses `gemmini_sched.py` and collects every string a checker can put in
its error list -- and `test_the_table_covers_every_rule_the_binding_states` holds the table to it by set
equality. Adding a rule without a case is red. Deleting a rule while its case stands is red. The count
is not written down anywhere here, because a written-down count is a claim and this is a measurement.

WHAT A CASE ASSERTS. `site` is the rule's fingerprint in the source -- the longest literal run of its
message -- and doubles as the table's key. `expect` is what the kernel must actually provoke, and is
the same string except where the fingerprint is too weak to be evidence: the packing-overflow rule
re-raises another module's exception as `f"{funct}: {exc}"`, whose only literal is `": "`, so that case
asserts the exception's own text instead.

MORE CASES THAN RULES, ON PURPOSE. Several rules are disjunctions reported through one message -- an
empty tile in any of three dimensions, residual-add *or* either reused scratchpad id, a stride mismatch
on any of the three operands, a pad too wide in any of three dimensions. The fingerprint cannot tell
those apart, so covering such a rule with a single case would leave the other disjuncts untested while
the coverage assertion read as satisfied. Each disjunct gets its own case; they share a `site`.

THE CONTROL MATTERS AS MUCH AS THE CASES. `test_the_legal_kernel_is_clean` pins that the shared baseline
passes with zero errors. Without it every assertion below would also pass for a checker that rejected
everything, which is the failure mode a table of negative cases invites.
"""

from __future__ import annotations

import ast
from functools import cache

import pytest

from merlin.runtime.backends import base as _bk
from merlin.sched.check.static import check_kernel
from merlin.sched.ir import NULL, Kernel, Ptr, TensorArg, call
from merlin.targetgen import target_registry


def _support_root():
    selected = target_registry.explicit_targets().get("gemmini")
    if selected is None:
        pytest.skip("requires explicit Gemmini support on MERLIN_TARGET_PATH", allow_module_level=True)
    info = target_registry.resolve("gemmini")
    assert info.base.resolve() == selected.resolve(), "selected support resolution drifted"
    return info.base


pytestmark = pytest.mark.target("gemmini")

#: The functions in the binding that decide legality. `finish` is included because the rule it carries
#: -- a partial sum the kernel never stores -- can only be stated at the end of the kernel.
_CHECKERS = ("check_config_ex", "check_config_st", "check_config_ld", "check_loop_ws", "finish")

_BINDING = _support_root() / "backend/gemmini_sched.py"


# -- the denominator, read out of the binding -------------------------------------------------------


def _fingerprint(node: ast.expr) -> str | None:
    """The longest literal run of an error message, or None if the node is not one.

    An f-string's interpolations are whatever the operands happened to be, so the literal text around
    them is the only part that identifies the rule across runs.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        literals = [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        return max(literals, key=len) if literals else None
    return None


def _messages(fn: ast.FunctionDef) -> list[ast.expr]:
    """Every expression `fn` can place in its error list.

    Two spellings, both used by the binding: `errs.append(<msg>)`, and returning a list literal --
    including the `return [] if ok else [<msg>]` form, whose two branches are one line and so cannot be
    told apart by line number. Parsed structurally rather than by matching text, per the repo's rule.
    """
    out: list[ast.expr] = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "errs"
        ):
            out.append(node.args[0])
        elif isinstance(node, ast.Return) and node.value is not None:
            branches = [node.value.body, node.value.orelse] if isinstance(node.value, ast.IfExp) else [node.value]
            for branch in branches:
                if isinstance(branch, ast.List):
                    out.extend(branch.elts)
    return out


def _sites_in(source: str) -> dict[str, str]:
    """``{fingerprint: checker name}`` for every legality rule ``source`` states.

    Takes text rather than reading the file, so the mutation tests can plant and remove a rule in a
    parsed copy and watch this answer change -- without touching a file other sessions are running
    against.
    """
    tree = ast.parse(source)
    sites: dict[str, str] = {}
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name in _CHECKERS):
            continue
        for node in _messages(fn):
            mark = _fingerprint(node)
            if mark is None:
                continue
            assert mark not in sites, (
                f"two rules share the fingerprint {mark!r} ({sites[mark]} and {fn.name}), so the table "
                "below cannot tell them apart. Give one of them distinguishing literal text."
            )
            sites[mark] = fn.name
    return sites


@cache
def _error_sites() -> dict[str, str]:
    return _sites_in(_BINDING.read_text())


# -- the kernels ------------------------------------------------------------------------------------


@cache
def _iset():
    return _bk.get_backend("gemmini").sched_instruction_set()


@cache
def _facts() -> dict:
    f = _iset().facts
    c = f["constants"]
    return {
        "dim": f["dim"],
        "elem_bytes": f["elem_bytes"],
        "ws": c["WEIGHT_STATIONARY"],
        "no_act": c["NO_ACTIVATION"],
        "relu": c["RELU"],
    }


#: Wide enough that no case runs out of tensor before it runs out of legality -- a footprint error
#: would mask the rule under test.
_SPAN = 4096
_ARGS = (
    TensorArg("A", (_SPAN, _SPAN), "i8", "read"),
    TensorArg("B", (_SPAN, _SPAN), "i8", "read"),
    TensorArg("D", (_SPAN,), "i32", "read"),
    TensorArg("C", (_SPAN, _SPAN), "i8", "write"),
)


def _ex(**over):
    f = _facts()
    v = dict(dataflow=f["ws"], sys_act=0, sys_shift=0, A_stride=1, A_transpose=0, B_transpose=0)
    return call("config_ex", **(v | over))


def _st(**over):
    f = _facts()
    v = dict(stride=f["dim"], acc_act=f["no_act"], acc_scale=1.0)
    return call("config_st", **(v | over))


def _ld(id_, **over):
    f = _facts()
    v = dict(stride=f["dim"] * f["elem_bytes"], scale=1.0, shrunk=0, id=id_)
    return call("config_ld", **(v | over))


def _ws(**over):
    f = _facts()
    v = dict(
        I=1,
        J=1,
        K=1,
        pad_I=0,
        pad_J=0,
        pad_K=0,
        A=Ptr("A", 0),
        B=Ptr("B", 0),
        D=Ptr("D", 0),
        C=Ptr("C", 0),
        A_stride=f["dim"],
        B_stride=f["dim"],
        D_stride=0,
        C_stride=f["dim"],
        A_transpose=0,
        B_transpose=0,
        full_C=0,
        low_D=0,
        ex_accumulate=1,
        act=f["no_act"],
        a_spad_id=0,
        b_spad_id=0,
        is_resadd=0,
    )
    return call("loop_ws", **(v | over))


def _config(**over):
    """The four configurations a legal LOOP_WS needs, with one of them perturbed by keyword."""
    return (
        _ex(**over.get("ex", {})),
        _st(**over.get("st", {})),
        _ld(0, **over.get("ld0", {})),
        _ld(1),
        _ld(2, stride=0),
    )


def _kernel(*body) -> Kernel:
    return Kernel("k", _ARGS, tuple(body))


def _legal() -> Kernel:
    return _kernel(*_config(), _ws())


@cache
def _table() -> dict[str, tuple[Kernel, str, str]]:
    """``{case: (kernel, site fingerprint, substring the kernel must provoke)}``."""
    f = _facts()
    dim, big_act = f["dim"], 1 << 8

    def case(kernel, site, expect=None):
        return (kernel, site, expect if expect is not None else site)

    return {
        "a dataflow the binding does not model": case(
            _kernel(*_config(ex={"dataflow": f["ws"] + 1}), _ws()),
            "only the weight-stationary dataflow is modelled",
        ),
        "a store stride of zero": case(
            _kernel(*_config(st={"stride": 0}), _ws()),
            "store stride must be positive",
        ),
        "a load configuration for a fourth operand": case(
            _kernel(*_config(), _ld(3), _ws()),
            " is not one of the three LOOP_WS operands",
        ),
        "a negative load stride": case(
            _kernel(*_config(), _ld(0, stride=-1), _ws()),
            "negative load stride",
        ),
        "a tile with no rows": case(_kernel(*_config(), _ws(I=0)), "empty tile I="),
        "a tile with no columns": case(_kernel(*_config(), _ws(J=0)), "empty tile I="),
        "a tile with no reduction": case(_kernel(*_config(), _ws(K=0)), "empty tile I="),
        "a row pad as wide as the block": case(
            _kernel(*_config(), _ws(pad_I=dim)), " outside [0, ", f"pad_I={dim} outside [0, {dim})"
        ),
        "a column pad as wide as the block": case(
            _kernel(*_config(), _ws(pad_J=dim)), " outside [0, ", f"pad_J={dim} outside [0, {dim})"
        ),
        "a reduction pad as wide as the block": case(
            _kernel(*_config(), _ws(pad_K=dim)), " outside [0, ", f"pad_K={dim} outside [0, {dim})"
        ),
        "operand tiles larger than this loop's half of the scratchpad": case(
            _kernel(*_config(), _ws(I=1, J=1, K=300)),
            " scratchpad rows > ",
        ),
        "an output tile larger than this loop's half of the accumulator": case(
            _kernel(*_config(), _ws(I=8, J=8, K=1)),
            " accumulator rows > ",
        ),
        "both operands transposed at once": case(
            _kernel(*_config(ex={"A_transpose": 1, "B_transpose": 1}), _ws(A_transpose=1, B_transpose=1)),
            "A and B transposed together (the RTL asserts against it)",
        ),
        "residual-add mode": case(
            _kernel(*_config(), _ws(is_resadd=1)),
            "resadd mode and scratchpad-id operand reuse are not modelled by this checker",
        ),
        "a reused scratchpad id for the A operand": case(
            _kernel(*_config(), _ws(a_spad_id=1)),
            "resadd mode and scratchpad-id operand reuse are not modelled by this checker",
        ),
        "a reused scratchpad id for the B operand": case(
            _kernel(*_config(), _ws(b_spad_id=1)),
            "resadd mode and scratchpad-id operand reuse are not modelled by this checker",
        ),
        "an activation the binding does not model": case(
            _kernel(*_config(st={"acc_act": f["relu"] + 1}), _ws(act=f["relu"] + 1)),
            " is not modelled by this checker",
        ),
        "an operand that spills into its neighbouring descriptor field": case(
            _kernel(*_config(st={"acc_act": big_act}), _ws(act=big_act)),
            ": ",
            "descriptor field overflow",
        ),
        "an A load stride that is not the stride the loop reads at": case(
            _kernel(*_config(), _ws(A_stride=dim * 2)), " != the loop's "
        ),
        "a B load stride that is not the stride the loop reads at": case(
            _kernel(*_config(), _ws(B_stride=dim * 2)), " != the loop's "
        ),
        "a bias load stride that is not the stride the loop reads at": case(
            _kernel(*_config(), _ws(D_stride=1)), " != the loop's "
        ),
        "no store configuration at all": case(
            _kernel(_ex(), _ld(0), _ld(1), _ld(2, stride=0), _ws()),
            "config_st stride does not match the loop's C stride",
        ),
        "a store activation that differs from the loop's": case(
            _kernel(*_config(st={"acc_act": f["relu"]}), _ws(act=f["no_act"])),
            "config_st activation differs from the loop's",
        ),
        "no execute configuration at all": case(
            _kernel(_st(), _ld(0), _ld(1), _ld(2, stride=0), _ws()),
            "config_ex transposes differ from the loop's (or no config_ex)",
        ),
        "a partial sum abandoned for a different output tile": case(
            _kernel(*_config(), _ws(C=NULL), _ws(I=2)),
            " is abandoned for tile ",
        ),
        "a continuation that overwrites the partial sum instead of accumulating": case(
            _kernel(*_config(), _ws(C=NULL), _ws(ex_accumulate=0)),
            "a loop continuing a partial sum must accumulate and must not reload D",
        ),
        "a fresh tile accumulating onto whatever the accumulator held": case(
            _kernel(*_config(), _ws(D=NULL, ex_accumulate=1)),
            "a fresh tile with no D accumulates onto whatever the accumulator half held",
        ),
        "a partial sum the kernel never stores": case(
            _kernel(*_config(), _ws(C=NULL)),
            "kernel ends with the partial sum on tile ",
        ),
    }


# -- the assertions ---------------------------------------------------------------------------------


def test_the_legal_kernel_is_clean():
    """The control. Every case below shares this baseline and perturbs one thing about it, so if the
    baseline itself were rejected the whole table would pass for a checker that refuses everything."""
    assert check_kernel(_legal(), _iset()) == []


@pytest.mark.parametrize("case", sorted(_table()))
def test_each_illegal_kernel_is_refused(case):
    kernel, _site, expect = _table()[case]
    errors = check_kernel(kernel, _iset())
    assert any(expect in e for e in errors), (
        f"{case}: the binding accepted a kernel it states a rule against. Expected an error containing "
        f"{expect!r}; got {errors or 'no errors at all'}."
    )


def test_the_table_covers_every_rule_the_binding_states():
    """The denominator. Read out of the binding's source, not written down here.

    This is the assertion that makes the table a net rather than a sample: a rule added to
    `gemmini_sched.py` with no case here is an uncovered fingerprint, and a case left behind by a
    deleted rule is a stale one. Both are named.
    """
    declared = {site for _k, site, _e in _table().values()}
    stated = set(_error_sites())
    assert declared == stated, (
        f"uncovered rules (in the binding, no case here): {sorted(stated - declared)}; "
        f"stale cases (a case here, no such rule in the binding): {sorted(declared - stated)}"
    )


# -- the budget the two capacity rules police --------------------------------------------------------


def test_the_capacity_rules_budget_half_the_machine_not_all_of_it():
    """The migration trap, pinned as a number.

    `mach.derive` reports this target's memories at their FULL depth, and the two capacity rules above
    police HALF of it, because the macro-instruction's FSM runs two concurrent loops and each owns one
    half of each memory. Both numbers are right about different things. Anyone replacing these rules
    with the derived machine will reach for the memory's `rows` -- it is the obvious field, it is
    correct about the hardware, and it is twice the budget a single loop may spend.
    """
    from merlin.sched.mach.derive import derive

    rows = {m.name: m.rows for m in derive("gemmini").memories}
    facts = _iset().facts
    assert facts["spad_rows_per_loop"] * 2 == rows["scratchpad"], (
        f"the per-loop operand budget {facts['spad_rows_per_loop']} is no longer half the machine's "
        f"{rows['scratchpad']} rows; one of the two moved and the double-buffer split is the reason "
        "they differ at all"
    )
    assert facts["acc_rows_per_loop"] * 2 == rows["accumulator"]


@pytest.mark.parametrize(
    "tile,what",
    [
        (dict(I=1, J=1, K=400), "operand tiles"),
        (dict(I=8, J=6, K=1), "an output tile"),
    ],
)
def test_a_tile_that_fits_the_whole_memory_but_not_one_loop_s_half_is_refused(tile, what):
    """The behavioural half of the same guard, which survives a refactor of the constants.

    Each of these fits the memory the machine reports and overruns the half a single loop owns. A
    migration that budgets the full depth accepts them, and the overflow returns wrong data rather than
    an error -- so asserting the refusal is the only thing that can catch it.
    """
    errors = check_kernel(_kernel(*_config(), _ws(**tile)), _iset())
    assert any("per loop" in e for e in errors), (
        f"{what} sized to overrun one loop's half of the memory was accepted: {errors or 'no errors'}"
    )


# -- the mutations: this file must go red when a rule stops being enforced --------------------------


def _with_rule_silenced(expect: str):
    """The instruction set with every error message containing ``expect`` dropped.

    Behaviourally identical to deleting that rule's line from the binding, which is the thing this
    file exists to make impossible to do quietly -- done in process, against a copy, so the shared
    working tree is never in a broken state.
    """
    from dataclasses import replace

    def silence(fn):
        return None if fn is None else lambda *a: [m for m in fn(*a) if expect not in m]

    iset = _iset()
    return replace(
        iset,
        instrs={n: replace(d, check=silence(d.check)) for n, d in iset.instrs.items()},
        finish=silence(iset.finish),
    )


@pytest.mark.parametrize("case", sorted(_table()))
def test_deleting_a_rule_makes_its_case_fail(case):
    """The per-case mutation. Each assertion above is load-bearing exactly when this passes: silence
    the one rule and that case's kernel is accepted."""
    kernel, _site, expect = _table()[case]
    errors = check_kernel(kernel, _with_rule_silenced(expect))
    assert not any(expect in e for e in errors), (
        f"{case}: silencing the rule did not stop it being reported, so the assertion that it fires "
        "would pass whether or not the rule exists."
    )


def test_the_coverage_check_notices_a_planted_rule():
    """The denominator's mutation, first direction: a rule added to the binding with no case here."""
    tree = ast.parse(_BINDING.read_text())
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "check_config_ex":
            fn.body.insert(0, ast.parse('errs.append("a rule nobody wrote a case for")').body[0])
            break
    else:
        raise AssertionError("check_config_ex is gone from the binding")
    planted = _sites_in(ast.unparse(tree))
    assert "a rule nobody wrote a case for" in planted
    assert planted.keys() - {site for _k, site, _e in _table().values()} == {"a rule nobody wrote a case for"}


def test_the_coverage_check_notices_a_removed_rule():
    """Second direction: a rule deleted from the binding while its case still stands here."""
    victim = "A and B transposed together (the RTL asserts against it)"
    tree = ast.parse(_BINDING.read_text())
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "check_loop_ws":
            fn.body = [s for s in fn.body if victim not in ast.unparse(s)]
            break
    remaining = _sites_in(ast.unparse(tree))
    assert victim not in remaining, "the deletion did not take; this mutation proves nothing"
    assert {site for _k, site, _e in _table().values()} - remaining.keys() == {victim}


def test_every_checker_the_binding_installs_is_in_this_file():
    """`_error_sites` only looks inside the functions `_CHECKERS` names, so a whole checker added to the
    binding and not named here would be invisible to the coverage test above -- coverage of a
    denominator that quietly shrank."""
    installed = {name for name, d in _iset().instrs.items() if d.check is not None}
    assert {f"check_{name}" for name in installed} | {"finish"} == set(_CHECKERS)
    assert _iset().finish is not None, "the end-of-kernel hook is gone; its rule cannot fire"
