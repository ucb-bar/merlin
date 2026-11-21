"""AST syntax observations for the inert-capability census.

These detectors describe source shapes. They neither import compiler providers
nor change capability, numerical or source-access admission.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import _source_layout

ROOT = Path(__file__).resolve().parents[2]


RANK = {
    "runtime-inert": 0,
    "unchecked-subprocess-input": 1,
    "self-comparison": 2,
    "always-empty-field": 3,
    "registry-asymmetry": 4,
    "tautological-gate": 5,
    "nonrecursive-aggregate": 6,
    "unproduced-member": 7,
    "dead-env-knob": 8,
    "unreferenced-def": 9,
}


AGGREGATE_TOKENS = ("footprint", "size", "bytes", "total", "usage", "disk", "weight")


NONRECURSIVE_LISTERS = ("iterdir", "listdir", "scandir", "glob")


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _parent(node: ast.AST) -> ast.AST | None:
    return getattr(node, "merlin_parent", None)


@dataclass
class Finding:
    kind: str
    ident: str  # stable ratchet key (no line numbers -- lines move, defects don't)
    file: str
    line: int
    declared: str  # what is declared
    fires_when: str  # what would have to happen for it to fire
    blocked_by: str  # why that cannot happen
    evidence: str = ""  # how this was verified -- never omit

    @property
    def key(self) -> str:
        return f"{self.kind} {_source_layout.policy_path(self.ident)}"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "id": self.ident,
            "file": self.file,
            "line": self.line,
            "declared": self.declared,
            "fires_when": self.fires_when,
            "blocked_by": self.blocked_by,
            "evidence": self.evidence,
            "rank": RANK.get(self.kind, 99),
        }


@dataclass
class _Corpus:
    """Everything the detectors need to ask "does anything else in the tree do X?"."""

    scan: dict[Path, ast.Module] = field(default_factory=dict)
    witness: dict[Path, ast.Module] = field(default_factory=dict)
    #: Identifier-shaped tokens appearing in NON-Python data (capsule yaml/mlir, schemas, headers).
    #: A vocabulary member can be produced by DATA as legitimately as by code -- `SCALE_KINDS`
    #: members are minted by `src/merlin/resources/schemas/quant_formats.registry.yaml`, not by any `.py` -- so a
    #: census that reads only Python manufactures false "never produced" findings.
    data_tokens: set = field(default_factory=set)
    root: Path = ROOT

    def all_trees(self) -> dict[Path, ast.Module]:
        merged = dict(self.witness)
        merged.update(self.scan)
        return merged


def _lister_call(node: ast.AST) -> str | None:
    """The name of a NON-recursive directory listing call, or ``None``.

    ``rglob``, ``walk`` and ``glob("**/...")`` are the recursive forms and never match.
    """
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    attr = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
    if attr not in NONRECURSIVE_LISTERS:
        return None
    if attr == "glob":
        if not node.args or not isinstance(node.args[0], ast.Constant):
            return None  # computed pattern: undecidable, skip
        pattern = node.args[0].value
        if not isinstance(pattern, str) or "**" in pattern:
            return None  # recursive
    return attr


def _function_nodes(fn):
    """Walk one lexical function body; nested functions own their process results."""
    pending = [fn]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        )


def _delegates_process(fn, call):
    names = {
        node.targets[0].id
        for node in _function_nodes(fn)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.value is call
    }
    return any(
        isinstance(node, ast.Return)
        and any(
            isinstance(value, ast.Name) and value.id in names
            for value in (node.value.elts if isinstance(node.value, ast.Tuple) else [node.value])
        )
        for node in _function_nodes(fn)
    )


def _recursive_listing(fn):
    nodes = list(_function_nodes(fn))
    for node in nodes:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("rglob", "walk"):
                return True
            if (
                node.func.attr == "glob"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and "**" in node.args[0].value
            ):
                return True
    # An explicit directory worklist is recursive even though scandir itself is shallow.
    for loop in (n for n in nodes if isinstance(n, ast.While) and isinstance(n.test, ast.Name)):
        calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)]
        methods = {
            n.func.attr
            for n in calls
            if isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == loop.test.id
        }
        if {"pop", "append"} <= methods and any(
            isinstance(n.func, ast.Attribute) and n.func.attr == "scandir" for n in calls
        ):
            return True
    return False


def _aggregate_reaches_return(fn, aggregate):
    """An ambiguity len() is not a returned footprint or numeric accounting value."""
    nodes = list(_function_nodes(fn))
    names = set()
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AugAssign)) and aggregate in ast.walk(node.value):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return any(
        isinstance(node, ast.Return)
        and (aggregate in ast.walk(node) or any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(node)))
        for node in nodes
    )


def detect_nonrecursive_aggregate(corpus: _Corpus) -> list[Finding]:
    """A size/count TOTAL computed over a non-recursive listing.

    ``compare/et_campaign.bundle_footprint`` was exactly this: it summed ``st_size`` over the
    bundle root's own entries, so a capture whose 1.83 GB of weights sat one directory down
    totalled ZERO and the fit check it fed could not fail.
    """
    findings: list[Finding] = []
    for path, tree in corpus.scan.items():
        rel = _rel(path, corpus.root)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            lowered = node.name.lstrip("_").lower()
            if not any(tok in lowered for tok in AGGREGATE_TOKENS):
                continue
            local = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            has_recursive = _recursive_listing(node) or any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"
                and n.func.attr in local
                and _recursive_listing(local[n.func.attr])
                for n in _function_nodes(node)
            )
            if has_recursive:
                continue
            listers = [(_lister_call(n), getattr(n, "lineno", node.lineno)) for n in ast.walk(node)]
            listers = [(a, ln) for a, ln in listers if a]
            if not listers:
                continue
            aggregates = [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in ("sum", "len", "max", "min")
            ]
            aggregates = [call for call in aggregates if _aggregate_reaches_return(node, call)]
            if not aggregates:
                continue
            attr, ln = listers[0]
            findings.append(
                Finding(
                    kind="nonrecursive-aggregate",
                    ident=f"{rel}:{node.name}",
                    file=rel,
                    line=node.lineno,
                    declared=f"{node.name}() totals something over a directory",
                    fires_when="the total reflects the whole tree, so a comparison against it "
                    "(a budget, a fit check, a coverage count) can fail",
                    blocked_by=f"the listing at line {ln} is NON-recursive (`{attr}`), so every entry "
                    f"one directory down contributes ZERO -- the total is structurally "
                    f"low, not merely small",
                    evidence="AST: the function aggregates (sum/len/max/min) over a non-recursive "
                    "listing and contains no rglob/walk/`**` form. Confirm the layout the "
                    "callers actually pass: a flat directory makes this a false positive.",
                )
            )
    return findings


def detect_self_comparison(corpus: _Corpus) -> list[Finding]:
    """``x == x`` and friends: a conjunct that can only ever be True.

    Found in the wild here as ``chk(all(...) and (art_names == art_names), ...)`` -- a consistency
    check whose second half is a tautology, so it reports on half of what its message claims.
    """
    findings: list[Finding] = []
    for path, tree in corpus.scan.items():
        rel = _rel(path, corpus.root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            if not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.LtE, ast.GtE, ast.Is, ast.IsNot)):
                continue
            left, right = node.left, node.comparators[0]
            if isinstance(left, ast.Constant) or isinstance(right, ast.Constant):
                continue  # `x == x` is the shape; `1 == 1` is a literal test
            if ast.dump(left) != ast.dump(right):
                continue
            # `cos == cos` / `v != v` on a bare NAME, standing alone, is the NaN idiom -- correct
            # and common (five sites in this tree). What is NOT an idiom is a self-comparison used
            # as one CONJUNCT of a larger claim: there the tautology silently shortens what the
            # surrounding message says was checked. Requiring a BoolOp parent (or a non-Name
            # operand) keeps the real finding and drops the idiom without an allowlist.
            par = _parent(node)
            if isinstance(left, ast.Name) and not isinstance(par, ast.BoolOp):
                continue
            # ... and the idiom also appears INSIDE a boolean guard, next to the finiteness test it
            # pairs with (`if v == v and abs(v) != float("inf")`). A conjunct naming inf/nan in the
            # same guard is the tell; without it, `x == x` beside an unrelated claim is the defect.
            if isinstance(par, ast.BoolOp):
                sibling_text = " ".join(ast.unparse(v) for v in par.values if v is not node).lower()
                if "inf" in sibling_text or "nan" in sibling_text:
                    continue
            src = ast.unparse(node)
            findings.append(
                Finding(
                    kind="self-comparison",
                    ident=f"{rel}:{node.lineno}:{src}",
                    file=rel,
                    line=node.lineno,
                    declared=f"a comparison `{src}` inside a predicate",
                    fires_when="the two sides differ",
                    blocked_by="both sides are the SAME expression, so the comparison has a constant "
                    "value -- whatever the surrounding message claims to check, this "
                    "conjunct does not check it",
                    evidence="AST: the two operands have byte-identical dumps. Constant-vs-constant "
                    "comparisons are excluded (those are deliberate literal tests).",
                )
            )
    return findings


def detect_unchecked_subprocess_input(corpus: _Corpus) -> list[Finding]:
    """A gate whose WORK LIST comes from a subprocess whose exit status it never reads.

    Measured in this tree: four pre-commit gates build their staged-file list from
    ``git diff --cached`` with no ``check=`` and no ``returncode`` test. Under a broken index the
    command fails, stdout is empty, the target list is empty, and every one of them prints ``[ ok]``
    and exits 0. A sibling gate (``check_no_answer_keys``) already fails closed here and says why.
    "We could not look" must never be indistinguishable from "we looked and it was clean".
    """
    findings: list[Finding] = []
    for path, tree in corpus.scan.items():
        rel = _rel(path, corpus.root)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = list(_function_nodes(fn))
            checks_status = any(
                (isinstance(n, ast.Attribute) and n.attr in ("returncode", "check_returncode"))
                or (isinstance(n, ast.keyword) and n.arg == "check")
                for n in body
            )
            if checks_status:
                continue
            for n in body:
                if not (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("run", "check_output", "Popen")
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "subprocess"
                ):
                    continue
                if _delegates_process(fn, n):
                    continue  # The original process result, including status, belongs to its caller.
                # Only when the OUTPUT decides what gets examined: `.stdout` consumed in the
                # same function. A subprocess whose output is merely logged is not this defect.
                if not any(isinstance(m, ast.Attribute) and m.attr == "stdout" for m in body):
                    continue
                # The defect is a WORK LIST built from stdout -- an empty list is a clean verdict
                # over nothing. A helper that reads a single scalar out of stdout (a git sha, a
                # version) fails differently and louder, so require the split/splitlines shape.
                if not any(isinstance(m, ast.Attribute) and m.attr in ("splitlines", "split") for m in body):
                    continue
                findings.append(
                    Finding(
                        kind="unchecked-subprocess-input",
                        ident=f"{rel}:{fn.name}",
                        file=rel,
                        line=n.lineno,
                        declared=f"{fn.name}() derives what it examines from a subprocess's stdout",
                        fires_when="the subprocess fails and the caller notices",
                        blocked_by="neither `check=` nor `.returncode` appears in the function, so a "
                        "failed command yields empty stdout, an empty work list, and a "
                        "clean verdict over nothing",
                        evidence="AST: a subprocess.run/check_output/Popen call whose stdout is "
                        "consumed in the same function, with no returncode/check inspection "
                        "anywhere in it. Verify by running the caller with a broken "
                        "environment (e.g. GIT_DIR=/nonexistent) and checking it still "
                        "reports success.",
                    )
                )
                break
    return findings


def _external_or_enumerated_vocabulary(tree: ast.Module, name: str) -> bool:
    """Direct input-map validation or finite value enumeration is not an absent producer."""

    def mapping_read(node):
        return (
            isinstance(node, ast.Subscript)
            and not any(
                isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
                for n in ast.walk(node.value)
            )
            or isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and bool(node.args)
        )

    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        inputs = {
            n.targets[0].id
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and mapping_read(n.value)
        }
        for node in ast.walk(fn):
            if isinstance(node, ast.Compare):
                for op, right in zip(node.ops, node.comparators):
                    if (
                        isinstance(op, (ast.In, ast.NotIn))
                        and isinstance(right, ast.Name)
                        and right.id == name
                        and (mapping_read(node.left) or isinstance(node.left, ast.Name) and node.left.id in inputs)
                    ):
                        return True
            value = node.iter if isinstance(node, (ast.For, ast.comprehension)) else None
            if isinstance(value, ast.Name) and value.id == name:
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "zip"
                and any(isinstance(a, ast.Name) and a.id == name for a in node.args)
            ):
                return True
    return False
