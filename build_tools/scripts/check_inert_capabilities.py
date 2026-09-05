#!/usr/bin/env python3
"""Gate: find capabilities that are DECLARED but cannot fire.

An inert capability presents as **silence**. Nothing errors, nothing warns; a feature simply
does nothing, and every downstream report says "fine". Four were found independently in one
afternoon of this repo's history:

  * ``llvmlower/passes_quant_int.lower_conv_int8`` -- a whole int8 conv datapath whose predicate
    wants a compound-affine indexing map. The frontend expands every conv into ``im2col + matmul``
    before it runs, so it has ZERO candidates on every model in the fleet. It ran, returned 0, and
    looked healthy for months.
  * ``xdsl_dialects/interface.KNOWN_EPILOGUE`` -- ``CommitOp.verify_`` validated epilogue stage
    names against a five-member vocabulary while the ONLY construction site hard-coded
    ``ArrayAttr([])``. No stage could ever be produced, so four of the five members were furniture.
  * ``compare/et_campaign.bundle_footprint`` -- priced only the bundle ROOT, non-recursively, so a
    capture with 1.83 GB of weights one directory down totalled ZERO and reported ``fits: True``.
    A fit check whose failure path did not exist.
  * a capture golden that was constant 1.0 over 1 of 2 outputs, so its gate could not fail.

Each is a different SHAPE of the same defect, and each was found by hand. This gate looks for the
shapes mechanically, over ``merlin/python/merlin/**`` (plus ``build_tools/scripts/**`` where a gate
lives), and reports them AS DATA -- it fixes nothing and routes nothing.

Detectors (``--kinds`` selects a subset; ``--list-kinds`` prints them):

  ``unproduced-member``      a string vocabulary that is VALIDATED against (an ``in`` / ``not in``
                             guard) and one of whose members is never CONSTRUCTED anywhere in
                             production code. The validator admits a value nothing can emit.
  ``always-empty-field``     a dict-literal field whose every statically-decidable construction site
                             writes an EMPTY literal, while some consumer iterates it. Everything the
                             consumer does per element is unreachable. This is the ``epilogue`` shape.
  ``registry-asymmetry``     a name registered in one table but absent from the table that CONSUMES
                             it (and the converse). Reported in BOTH directions, because both are
                             silent: an unregistered name makes the consumer's lookup raise into a
                             swallowed ``KeyError``, and an unranked registration is never offered.
  ``tautological-gate``      a function in the check/verify/validate/gate family with no reachable
                             failure: every return is a truthy constant, and it neither raises nor
                             asserts. It cannot report a problem it finds.
  ``nonrecursive-aggregate`` a size/count/footprint aggregate computed over a NON-RECURSIVE directory
                             listing. This is the ``bundle_footprint`` shape: the number is not small,
                             it is structurally zero whenever the payload is one directory down.
  ``self-comparison``        a predicate conjunct comparing an expression with ITSELF. Constant by
                             construction, so the message it carries checks less than it claims.
  ``unchecked-subprocess-input``
                             a gate whose work list comes from a subprocess whose exit status it
                             never reads: the command fails, the list is empty, and the gate reports
                             a clean verdict over nothing.
  ``dead-env-knob``          an env knob documented / advertised / set somewhere in the repo that no
                             code ever reads. Setting it does nothing, silently.
  ``unreferenced-def``       a module-private ``def``/``class`` whose name appears nowhere else --
                             not called, not exported, not reachable by dynamic dispatch.
  ``runtime-inert``          MEASURED, not inferred: from a ``--report`` file, a capability that saw a
                             healthy candidate population and transformed ZERO of it. Static analysis
                             cannot decide reachability of an IR predicate; this can.

**The runtime mode is the important one.** A predicate over IR shape is not statically decidable
here -- whether ``lower_conv_int8`` can fire depends on what a frontend outside this repo emits. So
passes are expected to COUNT what they scanned (``lower_conv_int8`` already does:
``generics_scanned`` / ``compound_map_generics`` / ``windowed_map_generics`` / ``conv_prov_ops`` /
``lowered``), and this gate reads those counts::

    check_inert_capabilities.py --report out/artifacts/.../quant_report.json

A capability with ``seen > 0`` and ``transformed == 0`` is flagged INERT-MEASURED. With no report,
the gate says so and does NOT call anything reachable -- "we did not look" must never read as clean.

Ratchet: findings that predate the gate live in ``inert_capabilities_ratchet.txt``, one
``<kind> <id>`` per line. **That file MAY ONLY SHRINK.** A finding not in it fails the check. The
count is printed machine-readably (``--json``) so CI can assert it never rises.

Usage::

    check_inert_capabilities.py                       # scan, fail on non-ratcheted findings
    check_inert_capabilities.py --json                # machine-readable
    check_inert_capabilities.py --report R.json       # add the measured runtime axis
    check_inert_capabilities.py --write-ratchet       # seed/refresh the ratchet (review the diff!)
    check_inert_capabilities.py --scan-root DIR       # audit a fixture tree instead of the repo
    check_inert_capabilities.py --stop-hook           # Claude Code Stop-hook JSON
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RATCHET = Path(__file__).resolve().parent / "inert_capabilities_ratchet.txt"

#: Trees whose Python is *library / tooling* code -- the code that declares capabilities.
DEFAULT_SCAN_ROOTS = ("merlin/python/merlin", "build_tools/scripts")
#: Trees consulted only to answer "does anything PRODUCE this value / READ this knob?".
DEFAULT_WITNESS_ROOTS = ("merlin/python/merlin", "build_tools/scripts", "merlin/contract",
                         "merlin/tests", "merlin/schemas", "examples", "docs",
                         # The session hooks are real readers: `MERLIN_ALLOW_ARTIFACT_WRITE` is
                         # consumed only by `.claude/hooks/guard_artifact_writes.py`, and a witness
                         # set that skipped them reported the repo's own escape hatch as dead.
                         ".claude/hooks")
#: Build-generated bundles copied in at wheel time; never source.
EXCLUDE_FRAGMENTS = ("/_data/", "/__pycache__/", "/site-packages/", "/.venv/",
                     "/.claude/worktrees/", "/third_party/", "/out/", "/build/",
                     "/_qa_ws/", "/node_modules/", "/.git/")

KINDS = ("unproduced-member", "always-empty-field", "registry-asymmetry", "tautological-gate",
         "nonrecursive-aggregate", "self-comparison", "unchecked-subprocess-input",
         "dead-env-knob", "unreferenced-def", "runtime-inert")

#: Consequence rank per kind, HIGH first. A dead accuracy gate matters more than a dead debug flag;
#: the ranking is per-KIND because a per-finding severity would be a guess.
RANK = {"runtime-inert": 0, "unchecked-subprocess-input": 1, "self-comparison": 2,
        "always-empty-field": 3, "registry-asymmetry": 4, "tautological-gate": 5,
        "nonrecursive-aggregate": 6, "unproduced-member": 7,
        "dead-env-knob": 8, "unreferenced-def": 9}

# --------------------------------------------------------------------------------------------
# vocabulary heuristics (all structural; this file, like the rest of the tree, uses no regex)
# --------------------------------------------------------------------------------------------

#: A module-level ALL-CAPS binding whose name carries one of these tokens is a candidate
#: *vocabulary* -- a set of accepted values, not an arbitrary constant. Membership in an ``in`` /
#: ``not in`` guard is ALSO required, so a name that merely looks like a vocabulary is not enough.
VOCAB_TOKENS = ("KNOWN", "VALID", "ALLOWED", "SUPPORTED", "ACCEPTED", "LEGAL", "RECOGNISED",
                "RECOGNIZED", "KINDS", "MODES", "STAGES", "FAMILIES", "TOKENS", "CHOICES",
                "OPTIONS", "VOCAB")

#: Function-name prefixes that promise a verdict. A member of this family with no reachable failure
#: is a gate that cannot fail.
GATE_PREFIXES = ("check", "verify", "validate", "assert", "ensure", "gate", "audit", "fits")

#: Aggregate-name tokens: a function whose name says it TOTALS something.
AGGREGATE_TOKENS = ("footprint", "size", "bytes", "total", "usage", "disk", "weight")

#: Non-recursive directory listings. ``rglob`` / ``walk`` / ``glob("**/...")`` are the recursive
#: forms and are never flagged.
NONRECURSIVE_LISTERS = ("iterdir", "listdir", "scandir", "glob")

ENV_READERS = ("getenv", "environ")

#: Non-Python files that can PRODUCE a vocabulary member (a capsule declares its own epilogue, a
#: registry yaml its own scale kinds, an ISA header its own opcode names).
DATA_SUFFIXES = (".yaml", ".yml", ".json", ".mlir", ".td", ".h", ".hpp", ".c", ".cc", ".cpp",
                 ".S", ".s", ".scala", ".sv", ".v", ".txt", ".csv", ".toml", ".cfg", ".ini")
#: Data files larger than this are captures/goldens, not declarations; reading them would cost
#: minutes and add nothing (a 6.8 GB contract tree lives under one of the witness roots).
MAX_DATA_BYTES = 4 * 1024 * 1024
#: Debt files (this gate's own ratchet, and its siblings' allowlists) NAME the very defects they
#: record. Counting them as data producers is self-poisoning: seeding this gate's ratchet once made
#: 40 of its own findings disappear on the next run, because every member it had reported was now a
#: token in `inert_capabilities_ratchet.txt`. A ratchet is a record of a defect, never a producer.
DEBT_FILE_TOKENS = ("ratchet", "allowlist", "_allow", "_debt")

#: A function whose name carries one of these is a DOC/EXAMPLE builder, not a production path. Its
#: construction sites are excluded from the always-empty census in BOTH directions: counting them as
#: producers would let a docstring example mask a field the compiler can only ever build empty
#: (``interface.build_example`` constructs the very epilogue stages the lowering could not).
EXAMPLE_TOKENS = ("example", "demo", "sample", "fixture", "docstring", "selftest")


def _enclosing_is_example(node: ast.AST) -> bool:
    p: ast.AST | None = node
    while p is not None:
        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            low = p.name.lower()
            if any(tok in low for tok in EXAMPLE_TOKENS):
                return True
        p = _parent(p)
    return False


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _safe_rglob(root: Path, *suffixes: str) -> list[Path]:
    """Every file under ``root`` with one of ``suffixes``, pruning excluded trees AS WE DESCEND.

    Pruning matters twice over: ``rglob`` would otherwise walk generated trees (``out/``,
    ``build/``, worktrees) that are orders of magnitude larger than the source, and this repo is a
    SHARED checkout where sibling sessions own directories this process cannot stat. An
    ``OSError`` on someone else's workspace must not decide the verdict, so it is skipped.
    """
    found: list[Path] = []
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            entries = sorted(cur.iterdir())
        except OSError:
            continue                       # another session's tree, or a broken link: not ours
        for e in entries:
            marker = f"/{e.as_posix()}/"
            if any(frag in marker for frag in EXCLUDE_FRAGMENTS):
                continue
            try:
                if e.is_dir():
                    if not e.is_symlink():
                        stack.append(e)
                elif e.suffix in suffixes:
                    found.append(e)
            except OSError:
                continue
    return sorted(found)


def _py_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        if r.is_file() and r.suffix == ".py":
            out.append(r)
            continue
        for p in _safe_rglob(r, ".py"):
            out.append(p)
    return out


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (SyntaxError, ValueError):
        return None


def _with_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.merlin_parent = node  # type: ignore[attr-defined]


def _parent(node: ast.AST) -> ast.AST | None:
    return getattr(node, "merlin_parent", None)


# --------------------------------------------------------------------------------------------


@dataclass
class Finding:
    kind: str
    ident: str            # stable ratchet key (no line numbers -- lines move, defects don't)
    file: str
    line: int
    declared: str         # what is declared
    fires_when: str       # what would have to happen for it to fire
    blocked_by: str       # why that cannot happen
    evidence: str = ""    # how this was verified -- never omit

    @property
    def key(self) -> str:
        return f"{self.kind} {self.ident}"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "id": self.ident, "file": self.file, "line": self.line,
                "declared": self.declared, "fires_when": self.fires_when,
                "blocked_by": self.blocked_by, "evidence": self.evidence,
                "rank": RANK.get(self.kind, 99)}


@dataclass
class _Corpus:
    """Everything the detectors need to ask "does anything else in the tree do X?"."""
    scan: dict[Path, ast.Module] = field(default_factory=dict)
    witness: dict[Path, ast.Module] = field(default_factory=dict)
    #: Identifier-shaped tokens appearing in NON-Python data (capsule yaml/mlir, schemas, headers).
    #: A vocabulary member can be produced by DATA as legitimately as by code -- `SCALE_KINDS`
    #: members are minted by `merlin/schemas/quant_formats.registry.yaml`, not by any `.py` -- so a
    #: census that reads only Python manufactures false "never produced" findings.
    data_tokens: set = field(default_factory=set)
    root: Path = ROOT

    def all_trees(self) -> dict[Path, ast.Module]:
        merged = dict(self.witness)
        merged.update(self.scan)
        return merged


# --------------------------------------------------------------------------------------------
# D1  unproduced-member
# --------------------------------------------------------------------------------------------

def _string_members(node: ast.AST) -> list[str] | None:
    """The string members of a set/frozenset/tuple/list/dict-keys literal, or ``None``.

    Only *fully* constant collections qualify: one non-literal element and the collection's real
    membership is unknown, so claiming a member is unproduced would be a guess.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
            node.func.id in ("frozenset", "set", "tuple", "list") and len(node.args) == 1:
        return _string_members(node.args[0])
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        elts = node.elts
        if elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elts):
            return [e.value for e in elts]  # type: ignore[attr-defined]
        return None
    if isinstance(node, ast.Dict):
        keys = node.keys
        if keys and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys):
            return [k.value for k in keys]  # type: ignore[union-attr]
        return None
    return None


def _collect_vocabularies(corpus: _Corpus) -> dict[str, tuple[Path, int, list[str], ast.AST]]:
    """Module-level ALL-CAPS string collections that LOOK like a vocabulary."""
    out: dict[str, tuple[Path, int, list[str], ast.AST]] = {}
    for path, tree in corpus.scan.items():
        for node in tree.body:
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            for t in targets:
                if not isinstance(t, ast.Name) or not t.id.isupper() or value is None:
                    continue
                if not any(tok in t.id for tok in VOCAB_TOKENS):
                    continue
                members = _string_members(value)
                if members and len(members) >= 2:
                    out[t.id] = (path, node.lineno, members, value)
    return out


def _validated_names(corpus: _Corpus) -> set[str]:
    """Names used as the right-hand side of an ``in`` / ``not in`` membership test."""
    names: set[str] = set()
    for tree in corpus.all_trees().values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    if isinstance(comp, ast.Name):
                        names.add(comp.id)
                    elif isinstance(comp, ast.Attribute):
                        names.add(comp.attr)
    return names


def _string_occurrences(corpus: _Corpus, decl_nodes: set[int]) -> dict[str, dict[str, int]]:
    """For every string constant in the corpus: how many times it appears in a PRODUCING position.

    "Producing" = anywhere that is not (a) inside a declaring vocabulary literal and (b) not the
    left/right of a membership comparison. Counted separately for production trees and for
    tests, because a value only a test can build is still unproducible by the compiler.
    """
    counts: dict[str, dict[str, int]] = {}
    for path, tree in corpus.all_trees().items():
        rel = _rel(path, corpus.root)
        bucket = "test" if "/tests/" in f"/{rel}" else "prod"
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            # inside a declaring vocabulary literal?
            p: ast.AST | None = node
            inside_decl = False
            for _ in range(4):
                if p is None:
                    break
                if id(p) in decl_nodes:
                    inside_decl = True
                    break
                p = _parent(p)
            if inside_decl:
                continue
            par = _parent(node)
            if isinstance(par, ast.Compare):
                continue                       # a membership/equality test is consumption
            counts.setdefault(node.value, {"prod": 0, "test": 0})[bucket] += 1
    return counts


def detect_unproduced_member(corpus: _Corpus) -> list[Finding]:
    vocabs = _collect_vocabularies(corpus)
    if not vocabs:
        return []
    validated = _validated_names(corpus)
    decl_nodes = {id(n) for _, (_, _, _, n) in vocabs.items()}
    occ = _string_occurrences(corpus, decl_nodes)
    findings: list[Finding] = []
    for name, (path, lineno, members, _n) in sorted(vocabs.items()):
        if name not in validated:
            continue                            # declared but not used as a gate -- other detectors
        rel = _rel(path, corpus.root)
        for m in members:
            c = occ.get(m, {"prod": 0, "test": 0})
            if c["prod"] > 0:
                continue
            where = ("only a TEST constructs it" if c["test"] else
                     "nothing anywhere constructs it")
            in_data = m in corpus.data_tokens
            if in_data:
                continue
            findings.append(Finding(
                kind="unproduced-member",
                ident=f"{rel}:{name}:{m}",
                file=rel, line=lineno,
                declared=f"{name} accepts the value {m!r}",
                fires_when=f"some code path constructs the string {m!r} and feeds it to the "
                           f"guard that validates against {name}",
                blocked_by=f"no producing occurrence of {m!r} in the scanned corpus -- {where}",
                evidence=f"AST string-constant census over {len(corpus.all_trees())} Python "
                         f"files (prod={c['prod']} test={c['test']} producing occurrences; "
                         f"membership comparisons and the declaration itself excluded) PLUS a "
                         f"token census of {len(corpus.data_tokens)} distinct tokens in the "
                         f"non-Python data trees, where this member does not appear either. "
                         f"A member a capsule/registry/ISA-header could produce is NOT inert; "
                         f"that is what the data census rules out."))
    return findings


# --------------------------------------------------------------------------------------------
# D2  always-empty-field
# --------------------------------------------------------------------------------------------

def _is_empty_literal(node: ast.AST) -> bool | None:
    """``True`` empty, ``False`` non-empty, ``None`` not statically decidable."""
    if isinstance(node, (ast.List, ast.Set, ast.Tuple, ast.Dict)):
        n = len(node.keys) if isinstance(node, ast.Dict) else len(node.elts)
        return n == 0
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return len(node.value) == 0
    if isinstance(node, ast.Call):
        # A one-argument wrapper (`ArrayAttr([...])`, `frozenset([...])`, `tuple(...)`) is
        # transparent for emptiness; a zero-argument call (`list()`, `dict()`) is empty.
        if not node.args and not node.keywords and isinstance(node.func, (ast.Name, ast.Attribute)):
            return True
        if len(node.args) == 1 and not node.keywords:
            return _is_empty_literal(node.args[0])
    return None


def _iterated_keys(corpus: _Corpus) -> set[str]:
    """String keys whose VALUE some consumer iterates or measures -- ``for x in d["k"]``,
    ``for x in d.get("k", ...)``, ``len(o.k)``, ``for s in self.k``."""
    keys: set[str] = set()
    for tree in corpus.all_trees().values():
        for node in ast.walk(tree):
            it = None
            if isinstance(node, ast.For):
                it = node.iter
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                it = node.generators[0].iter if node.generators else None
            if it is None:
                continue
            if isinstance(it, ast.Subscript) and isinstance(it.slice, ast.Constant) \
                    and isinstance(it.slice.value, str):
                keys.add(it.slice.value)
            elif isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) \
                    and it.func.attr == "get" and it.args \
                    and isinstance(it.args[0], ast.Constant) and isinstance(it.args[0].value, str):
                keys.add(it.args[0].value)
            elif isinstance(it, ast.Attribute):
                keys.add(it.attr)
    return keys


def _mutated_keys(corpus: _Corpus) -> set[str]:
    """String keys whose value is filled in AFTER the dict is built.

    ``d = {"covered": [], "uncovered": []}`` followed by ``d["covered"].append(x)`` is an
    ACCUMULATOR, not an always-empty field: the empty literal is an initialiser. Without this the
    detector reports every accumulator in the tree and buries the one real finding.
    """
    mutators = {"append", "add", "extend", "update", "insert", "setdefault", "sort", "pop"}
    keys: set[str] = set()
    for tree in corpus.all_trees().values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str):
                par = _parent(node)
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    keys.add(node.slice.value)
                elif isinstance(par, ast.Attribute) and par.attr in mutators:
                    keys.add(node.slice.value)
                elif isinstance(par, ast.AugAssign):
                    keys.add(node.slice.value)
            elif isinstance(node, ast.Attribute) and node.attr in mutators:
                base = node.value
                if isinstance(base, ast.Subscript) and isinstance(base.slice, ast.Constant) \
                        and isinstance(base.slice.value, str):
                    keys.add(base.slice.value)
    return keys


def detect_always_empty_field(corpus: _Corpus) -> list[Finding]:
    """A dict-literal field every one of whose decidable construction sites writes EMPTY.

    This is the ``interface.commit`` ``epilogue`` shape exactly: the op declared a five-member
    epilogue vocabulary and the verifier checked it, while the only builder wrote ``ArrayAttr([])``.
    A field that is structurally always empty makes every per-element behaviour downstream --
    validation, lowering, runtime semantics -- unreachable.
    """
    empty_sites: dict[str, list[tuple[str, int]]] = {}
    nonempty: set[str] = set()
    undecidable: dict[str, int] = {}
    for path, tree in corpus.scan.items():
        rel = _rel(path, corpus.root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            # A dict literal that is an ARGUMENT to a call is a construction (`Op(properties={...})`,
            # `dict(...)`, `json.dump({...})`). A bare `x = {...}` is usually a local accumulator that
            # is filled in afterwards, and reading its initialiser as "always empty" is wrong -- that
            # shape produced 60+ false positives (every `buckets = {"dead": [], ...}` in this tree).
            if not isinstance(_parent(node), (ast.Call, ast.keyword)):
                continue
            if _enclosing_is_example(node):
                continue
            for k, v in zip(node.keys, node.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                verdict = _is_empty_literal(v)
                if verdict is True:
                    empty_sites.setdefault(k.value, []).append((rel, k.lineno))
                elif verdict is False:
                    nonempty.add(k.value)
                else:
                    undecidable[k.value] = undecidable.get(k.value, 0) + 1
    consumed = _iterated_keys(corpus)
    mutated = _mutated_keys(corpus)
    findings: list[Finding] = []
    for key, sites in sorted(empty_sites.items()):
        if key in nonempty or key in mutated or key not in consumed:
            continue
        rel, line = sites[0]
        undec = undecidable.get(key, 0)
        findings.append(Finding(
            kind="always-empty-field",
            ident=f"{rel}:{key}",
            file=rel, line=line,
            declared=f"field {key!r} is consumed element-wise somewhere (iterated / measured)",
            fires_when=f"some construction site writes a NON-EMPTY {key!r}",
            blocked_by=f"every statically-decidable construction site writes an empty literal "
                       f"({len(sites)} site(s): "
                       + ", ".join(f"{r}:{ln}" for r, ln in sites[:4]) + ")",
            evidence=f"AST dict-literal census over construction sites (a dict literal passed to "
                     f"a call): {len(sites)} empty, 0 non-empty, {undec} not statically decidable; "
                     f"the key is never written or mutated by subscript afterwards, and it IS "
                     f"consumed element-wise. A non-zero 'undecidable' count means a computed "
                     f"value could be non-empty -- verify by hand before acting."))
    return findings


# --------------------------------------------------------------------------------------------
# D3  registry-asymmetry
# --------------------------------------------------------------------------------------------

#: (label, producer, consumer) triples. A producer supplies the names a thing CAN be; a consumer is
#: the list that decides which of them are ever OFFERED. Both directions of the difference are
#: silent failures, so both are reported. Each side is ``(module, attribute, extractor)`` where the
#: extractor turns the attribute's value into a set of names.
REGISTRY_PAIRS = [
    ("impr-features/ranked-levers",
     ("merlin.llvmlower.impr_features", "known", "call"),
     ("merlin.mining.wholemodel_proposer", "RANKED_LEVERS", "first-of-pairs")),
]


def _named_elsewhere(corpus: _Corpus, producer_file: str) -> tuple[set[str], set[str]]:
    """(literal names, composed prefixes) that appear OUTSIDE the module that registers them.

    A registry name is reachable in two ways and a census that sees only one is wrong in bulk. It
    can be NAMED (a literal in a ranking list, a config, a capsule yaml) or COMPOSED (a ladder
    building ``f"perop_register_block_mr{n}"``, which this repo does for ~54 of its 86 features).
    Flagging composed names as unreachable would bury the handful of genuinely orphaned entries
    under a list of levers the search reaches every run.
    """
    literals: set[str] = set()
    prefixes: set[str] = set()
    for path, tree in corpus.all_trees().items():
        is_producer = _rel(path, corpus.root).endswith(producer_file)
        for node in ast.walk(tree):
            # Composed prefixes count even inside the producer: the ladder that MINTS
            # `perop_register_block_mr4` lives beside the registration, and the consumer selects it
            # by walking that same ladder. A literal, by contrast, only proves reachability when
            # some OTHER module writes it -- the registration site names every feature it registers.
            if is_producer and not isinstance(node, ast.JoinedStr):
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.add(node.value)
            elif isinstance(node, ast.JoinedStr) and node.values:
                head = node.values[0]
                if isinstance(head, ast.Constant) and isinstance(head.value, str) \
                        and len(head.value) >= 6:
                    prefixes.add(head.value)
    literals |= corpus.data_tokens
    return literals, prefixes


def _extract(value, how: str) -> set[str]:
    if how == "call":
        value = value()
    if how == "first-of-pairs":
        return {v[0] if isinstance(v, (tuple, list)) else v for v in value}
    if isinstance(value, dict):
        return set(value)
    return set(value)


def detect_registry_asymmetry(corpus: _Corpus, *, enabled: bool) -> tuple[list[Finding], list[str]]:
    """Both directions of a producer/consumer name-table difference, resolved by IMPORT.

    Importing rather than parsing is deliberate: registration here happens as an import side effect
    (``ensure_registered()`` calls at module scope), so a purely static reading would systematically
    under-count the registry and manufacture false 'unregistered' findings.
    """
    notes: list[str] = []
    if not enabled:
        return [], ["registry-asymmetry: SKIPPED (no --imports); asymmetry NOT decided"]
    findings: list[Finding] = []
    for label, (pmod, pattr, phow), (cmod, cattr, chow) in REGISTRY_PAIRS:
        try:
            import importlib
            # BOTH modules are imported BEFORE either table is read. Registration in this repo
            # happens as an import side effect, so reading the producer first would report every
            # feature the consumer's own imports register (`prepack_weight_layout` and five others)
            # as "ranked but unregistered" -- an artifact of THIS script's import order, not a
            # defect. That the order matters at all is itself worth knowing: a process that imports
            # only one of them really does see a short registry.
            pm = importlib.import_module(pmod)
            cm = importlib.import_module(cmod)
            produced = _extract(getattr(pm, pattr), phow)
            consumed = _extract(getattr(cm, cattr), chow)
        except Exception as exc:                                  # noqa: BLE001 - report, never die
            notes.append(f"registry-asymmetry {label}: UNDECIDED ({type(exc).__name__}: {exc})")
            continue
        for name in sorted(consumed - produced):
            findings.append(Finding(
                kind="registry-asymmetry",
                ident=f"{label}:consumed-not-registered:{name}",
                file=cmod.replace(".", "/") + ".py", line=0,
                declared=f"{cattr} offers the lever {name!r}",
                fires_when=f"{name!r} is registered in {pmod} so the composition check can resolve it",
                blocked_by=f"{name!r} is NOT registered; the resolver raises KeyError, which the "
                           f"composition check swallows and returns False -- the lever is never "
                           f"declined, it is INVISIBLE",
                evidence=f"imported {pmod}.{pattr} and {cmod}.{cattr} and differenced the "
                         f"name sets ({len(produced)} registered, {len(consumed)} ranked)"))
        producer_file = pmod.rsplit(".", 1)[-1] + ".py"
        literals, prefixes = _named_elsewhere(corpus, producer_file)
        # A registry also publishes SUB-TABLES of its own names (`MRPAD_INT8_TILES`,
        # `PEROP_MR_LADDER`), and a consumer walks the table rather than writing the names. Reading
        # only literals would call ~15 ladder-reachable tiles orphaned. Public module-level string
        # containers on the producer therefore count as consumer-visible.
        tabled: set[str] = set()
        for attr, val in sorted(vars(pm).items()):
            if attr.startswith("_") or attr == pattr:
                continue
            if isinstance(val, (list, tuple, set, frozenset)):
                tabled |= {v for v in val if isinstance(v, str)}
            elif isinstance(val, dict):
                tabled |= {k for k in val if isinstance(k, str)}
        for name in sorted(produced - consumed):
            if name in literals or name in tabled \
                    or any(name.startswith(pre) for pre in prefixes):
                continue           # named, tabled or composed by a consumer: reachable, not inert
            findings.append(Finding(
                kind="registry-asymmetry",
                ident=f"{label}:registered-not-consumed:{name}",
                file=pmod.replace(".", "/") + ".py", line=0,
                declared=f"{pmod} registers the feature {name!r}",
                fires_when=f"{name!r} is listed in {cmod}.{cattr} (or another proposal list) so the "
                           f"search can offer it",
                blocked_by=f"{name!r} appears in no consumer list, so nothing ever proposes it; "
                           f"registration alone does not make a lever reachable",
                evidence=f"imported both tables and differenced them ({len(produced)} registered, "
                         f"{len(consumed)} ranked), then removed every name that ANY other module "
                         f"names as a literal or composes from a prefix (a refinement ladder, a "
                         f"microkernel resolver, a champion config). What is left is named by "
                         f"nothing. Some of these are DELIBERATE -- read the comment at the "
                         f"registration site before treating one as a defect."))
    return findings, notes


# --------------------------------------------------------------------------------------------
# D4  tautological-gate
# --------------------------------------------------------------------------------------------

def _returns_only_truthy(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool | None:
    """``True`` when every ``return`` yields a truthy CONSTANT, ``None`` when undecidable."""
    seen = False
    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not fn:
            continue                                   # a nested def has its own verdict
        if isinstance(node, ast.Return):
            if node.value is None:
                continue                               # bare `return` is a truthy-free exit
            if not (isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, bool)):
                return None            # a non-bool return is a VALUE, not a verdict -- not a gate
            if not node.value.value:
                return False                           # a falsy return IS a reachable failure
            seen = True
    return True if seen else None


def _is_gate_name(name: str) -> bool:
    lowered = name.lstrip("_").lower()
    return any(lowered.startswith(p) for p in GATE_PREFIXES)


def detect_tautological_gate(corpus: _Corpus) -> list[Finding]:
    findings: list[Finding] = []
    for path, tree in corpus.scan.items():
        rel = _rel(path, corpus.root)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_gate_name(node.name):
                continue
            if _returns_only_truthy(node) is not True:
                continue
            raises = [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
            asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
            if raises or asserts:
                continue                                # it CAN signal; not tautological
            findings.append(Finding(
                kind="tautological-gate",
                ident=f"{rel}:{node.name}",
                file=rel, line=node.lineno,
                declared=f"{node.name}() reads as a verdict (check/verify/validate family)",
                fires_when="the thing under test is bad and the function says so",
                blocked_by="every return path yields a truthy constant and the body neither "
                           "raises nor asserts -- there is no way for it to report a problem",
                evidence="AST: enumerated every Return in the function body (nested defs "
                         "excluded); all are truthy constants; zero Raise, zero Assert"))
    return findings


# --------------------------------------------------------------------------------------------
# D5  nonrecursive-aggregate
# --------------------------------------------------------------------------------------------

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
            return None                                  # computed pattern: undecidable, skip
        pattern = node.args[0].value
        if not isinstance(pattern, str) or "**" in pattern:
            return None                                  # recursive
    return attr


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
            has_recursive = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and (n.func.attr in ("rglob", "walk")
                     or (n.func.attr == "glob" and n.args
                         and isinstance(n.args[0], ast.Constant)
                         and isinstance(n.args[0].value, str) and "**" in n.args[0].value))
                for n in ast.walk(node))
            if has_recursive:
                continue
            listers = [(_lister_call(n), getattr(n, "lineno", node.lineno))
                       for n in ast.walk(node)]
            listers = [(a, ln) for a, ln in listers if a]
            if not listers:
                continue
            aggregates = [n for n in ast.walk(node)
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                          and n.func.id in ("sum", "len", "max", "min")]
            if not aggregates:
                continue
            attr, ln = listers[0]
            findings.append(Finding(
                kind="nonrecursive-aggregate",
                ident=f"{rel}:{node.name}",
                file=rel, line=node.lineno,
                declared=f"{node.name}() totals something over a directory",
                fires_when="the total reflects the whole tree, so a comparison against it "
                           "(a budget, a fit check, a coverage count) can fail",
                blocked_by=f"the listing at line {ln} is NON-recursive (`{attr}`), so every entry "
                           f"one directory down contributes ZERO -- the total is structurally "
                           f"low, not merely small",
                evidence="AST: the function aggregates (sum/len/max/min) over a non-recursive "
                         "listing and contains no rglob/walk/`**` form. Confirm the layout the "
                         "callers actually pass: a flat directory makes this a false positive."))
    return findings


# --------------------------------------------------------------------------------------------
# D5b  self-comparison
# --------------------------------------------------------------------------------------------

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
            if not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.LtE, ast.GtE,
                                            ast.Is, ast.IsNot)):
                continue
            left, right = node.left, node.comparators[0]
            if isinstance(left, ast.Constant) or isinstance(right, ast.Constant):
                continue                       # `x == x` is the shape; `1 == 1` is a literal test
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
                sibling_text = " ".join(ast.unparse(v) for v in par.values
                                        if v is not node).lower()
                if "inf" in sibling_text or "nan" in sibling_text:
                    continue
            src = ast.unparse(node)
            findings.append(Finding(
                kind="self-comparison",
                ident=f"{rel}:{node.lineno}:{src}",
                file=rel, line=node.lineno,
                declared=f"a comparison `{src}` inside a predicate",
                fires_when="the two sides differ",
                blocked_by="both sides are the SAME expression, so the comparison has a constant "
                           "value -- whatever the surrounding message claims to check, this "
                           "conjunct does not check it",
                evidence="AST: the two operands have byte-identical dumps. Constant-vs-constant "
                         "comparisons are excluded (those are deliberate literal tests)."))
    return findings


# --------------------------------------------------------------------------------------------
# D5c  unchecked-subprocess-input
# --------------------------------------------------------------------------------------------

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
            body = list(ast.walk(fn))
            checks_status = any(
                (isinstance(n, ast.Attribute) and n.attr in ("returncode", "check_returncode"))
                or (isinstance(n, ast.keyword) and n.arg == "check")
                for n in body)
            if checks_status:
                continue
            for n in body:
                if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("run", "check_output", "Popen")
                        and isinstance(n.func.value, ast.Name)
                        and n.func.value.id == "subprocess"):
                    continue
                # Only when the OUTPUT decides what gets examined: `.stdout` consumed in the
                # same function. A subprocess whose output is merely logged is not this defect.
                if not any(isinstance(m, ast.Attribute) and m.attr == "stdout" for m in body):
                    continue
                # The defect is a WORK LIST built from stdout -- an empty list is a clean verdict
                # over nothing. A helper that reads a single scalar out of stdout (a git sha, a
                # version) fails differently and louder, so require the split/splitlines shape.
                if not any(isinstance(m, ast.Attribute) and m.attr in ("splitlines", "split")
                           for m in body):
                    continue
                findings.append(Finding(
                    kind="unchecked-subprocess-input",
                    ident=f"{rel}:{fn.name}",
                    file=rel, line=n.lineno,
                    declared=f"{fn.name}() derives what it examines from a subprocess's stdout",
                    fires_when="the subprocess fails and the caller notices",
                    blocked_by="neither `check=` nor `.returncode` appears in the function, so a "
                               "failed command yields empty stdout, an empty work list, and a "
                               "clean verdict over nothing",
                    evidence="AST: a subprocess.run/check_output/Popen call whose stdout is "
                             "consumed in the same function, with no returncode/check inspection "
                             "anywhere in it. Verify by running the caller with a broken "
                             "environment (e.g. GIT_DIR=/nonexistent) and checking it still "
                             "reports success."))
                break
    return findings


# --------------------------------------------------------------------------------------------
# D6  dead-env-knob
# --------------------------------------------------------------------------------------------

def _env_read_prefixes(corpus: _Corpus) -> set[str]:
    """Constant PREFIXES of env names built at run time -- ``os.environ.get(f"MERLIN_EXT_{name}")``.

    Without this the check is wrong by construction: ``paths.ext_path`` composes every
    ``MERLIN_EXT_*`` key from a short name, so a name-literal census sees none of them read and
    every documented one would be reported dead. A knob whose name starts with a composed prefix is
    therefore treated as READ (conservative -- it costs recall, never correctness).
    """
    prefixes: set[tuple[str, str]] = set()

    def shape_of(n: ast.AST) -> tuple[str, str] | None:
        """(constant head, constant tail) of a composed name, e.g. f"MERLIN_{x}_REPO"."""
        if isinstance(n, ast.JoinedStr) and n.values:
            head = n.values[0]
            tail = n.values[-1]
            h = head.value if isinstance(head, ast.Constant) \
                and isinstance(head.value, str) else ""
            t = tail.value if len(n.values) > 1 and isinstance(tail, ast.Constant) \
                and isinstance(tail.value, str) else ""
            if h:
                return (h, t)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            if isinstance(n.left, ast.Constant) and isinstance(n.left.value, str):
                return (n.left.value, "")
            return shape_of(n.left)
        return None

    for tree in corpus.all_trees().values():
        for node in ast.walk(tree):
            arg = None
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                    and node.value.attr == "environ":
                arg = node.slice
            elif isinstance(node, ast.Call):
                fn = node.func
                attr = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else "")
                # `os.environ.get(...)` is a `.get` on an `environ` attribute -- the single most
                # common env read in this tree, and reading only `getenv`/`environ[...]` missed it.
                is_environ_get = (attr == "get" and isinstance(fn, ast.Attribute)
                                  and isinstance(fn.value, ast.Attribute)
                                  and fn.value.attr == "environ")
                if attr in ENV_READERS or attr == "env" or is_environ_get:
                    arg = node.args[0] if node.args else None
            if arg is None:
                continue
            shape = shape_of(arg)
            if shape and shape[0].endswith("_") and len(shape[0]) > 3:
                prefixes.add(shape)
            # `k.startswith("MERLIN_EXT_")` over os.environ is the other composed-read spelling.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "startswith" and node.args \
                    and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) \
                    and node.args[0].value.endswith("_") and node.args[0].value.isupper():
                prefixes.add((node.args[0].value, ""))
    return prefixes


def _env_names_read(corpus: _Corpus) -> set[str]:
    """Env var names READ, following ``os.environ[...]``, ``os.environ.get(...)``,
    ``os.getenv(...)`` and this repo's ``merlin.common.paths.env(...)``."""
    names: set[str] = set()

    def const_str(n: ast.AST | None) -> str | None:
        return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None

    for tree in corpus.all_trees().values():
        for node in ast.walk(tree):
            # ANY Python string literal of the name counts. The narrow rule ("an argument to
            # os.environ.get") is wrong in this tree in two ways at once: `passes.py` binds the name
            # to `VERIFY_LOG_ENV` and reads the CONSTANT, and `toolchain.py` reads it through a
            # private `_env(...)` wrapper. Both looked dead and neither is. Naming the knob in code
            # is the evidence; which call consumes it is not this gate's business.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # TOKENS of the literal, not the literal itself: these names also travel inside
                # composite strings (`f"-Wl,--defsym,MERLIN_STACK_BYTES={...}"` is a LINKER symbol,
                # not an env var, and reporting it as a dead knob is simply wrong). Any code that
                # spells the name is evidence something uses it.
                names |= {t for t in _tokens(node.value)
                          if t.isupper() and "_" in t and not t.endswith("_")}
            if isinstance(node, ast.Subscript):
                base = node.value
                if isinstance(base, ast.Attribute) and base.attr == "environ":
                    s = const_str(node.slice)
                    if s:
                        names.add(s)
            elif isinstance(node, ast.Call):
                fn = node.func
                attr = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else "")
                if attr in ENV_READERS or attr in ("env", "get", "pop", "setdefault"):
                    s = const_str(node.args[0]) if node.args else None
                    if s and s.isupper() and "_" in s:
                        names.add(s)
    return names


def _tokens(text: str, extra: str = "") -> set[str]:
    """Identifier-shaped tokens in arbitrary text. Structural split, no pattern matching.

    ``extra`` widens what counts as part of a token (``".-/:"`` keeps a namespaced name whole).
    """
    out: set[str] = set()
    cur: list[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_" or ch in extra:
            cur.append(ch)
        else:
            if cur:
                out.add("".join(cur))
            cur = []
    if cur:
        out.add("".join(cur))
    return out


def detect_dead_env_knob(corpus: _Corpus, prefix: str, mention_roots: list[Path]) -> list[Finding]:
    """A knob the repo advertises (docs, hooks, settings, shell) that no code READS.

    The direction matters: a knob read but never set is normal (the user sets it); a knob
    *documented* and never read is a promise the code cannot keep, and setting it is silent.
    """
    read = {n for n in _env_names_read(corpus) if n.startswith(prefix)}
    composed = {(h, t) for h, t in _env_read_prefixes(corpus) if h.startswith(prefix)}

    def is_read(name: str) -> bool:
        # A knob is read if its literal name appears in code, OR if it matches the SHAPE of a
        # composed read: `os.environ.get(f"MERLIN_{source.upper()}_REPO")` genuinely reads
        # MERLIN_TRITON_REPO / MERLIN_EXO_REPO / MERLIN_AUTOCOMP_REPO, none of which is ever
        # written as a literal. Head AND tail must both match, so the bare `MERLIN_` head of that
        # f-string does not excuse every knob in the namespace.
        return name in read or any(
            name.startswith(h) and name.endswith(t) and len(name) > len(h) + len(t)
            for h, t in composed)

    mentions: dict[str, tuple[str, int]] = {}
    for root in mention_roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else _safe_rglob(
            root, ".md", ".json", ".yaml", ".yml", ".sh", ".toml", ".txt")
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            toks = {t for t in _tokens(text)
                    if t.startswith(prefix) and len(t) > len(prefix)
                    and not t.endswith("_")          # a namespace prefix, not a knob
                    and not is_read(t)}
            if not toks:
                continue
            rel = _rel(p, corpus.root)
            for ln, line in enumerate(text.splitlines(), 1):
                for t in sorted(toks):
                    if t in line and t not in mentions:
                        mentions[t] = (rel, ln)
    return [Finding(
        kind="dead-env-knob",
        ident=name,
        file=where, line=line,
        declared=f"the repo advertises the env knob {name}",
        fires_when="some code path reads it and changes behaviour",
        blocked_by="no os.environ / os.getenv / paths.env read of this name anywhere in the "
                   "scanned trees -- exporting it does nothing",
        evidence=f"AST census of env reads ({len(read)} literal {prefix}* names, plus "
                 f"{len(composed)} composed name shape(s) {sorted(composed)}) vs a token scan of "
                 f"docs/settings/shell; first mention at {where}:{line}. A knob read through a "
                 f"name this census cannot see (a dict of names, a name from a data file) would "
                 f"make this a false positive -- check the read side before deleting a knob.")
        for name, (where, line) in sorted(mentions.items())]


# --------------------------------------------------------------------------------------------
# D7  unreferenced-def
# --------------------------------------------------------------------------------------------

def detect_unreferenced_def(corpus: _Corpus) -> list[Finding]:
    """A module-PRIVATE def/class nothing names. Restricted to ``_``-prefixed names on purpose:
    a public symbol may be someone's API, and calling it dead would be a guess."""
    referenced: set[str] = set()
    defined: dict[str, list[tuple[str, int]]] = {}
    for path, tree in corpus.all_trees().items():
        rel = _rel(path, corpus.root)
        in_scan = path in corpus.scan
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                referenced |= _tokens(node.value)          # dynamic dispatch by name
            elif isinstance(node, ast.alias):
                referenced.add(node.name.split(".")[-1])
                if node.asname:
                    referenced.add(node.asname)
        if not in_scan:
            continue
        for node in tree.body:                              # module level only
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and node.name.startswith("_") and not node.name.startswith("__") \
                    and not node.decorator_list:
                defined.setdefault(node.name, []).append((rel, node.lineno))
    findings: list[Finding] = []
    for name, sites in sorted(defined.items()):
        # A name is "referenced" if it is used anywhere OTHER than its own def; the def itself
        # contributes no Name node, so a single definition with zero references is unreferenced.
        if name in referenced or len(sites) > 1:
            continue
        rel, line = sites[0]
        findings.append(Finding(
            kind="unreferenced-def",
            ident=f"{rel}:{name}",
            file=rel, line=line,
            declared=f"module-private {name}()",
            fires_when="something calls it, imports it, or reaches it by name",
            blocked_by="the name appears nowhere else in the corpus -- not as a call, an "
                       "attribute, an import, or a string (which would mean dynamic dispatch)",
            evidence="AST name census over Name / Attribute / alias / string-constant tokens "
                     "across scan + witness trees. Decorated defs are excluded (a decorator can "
                     "register a callee invisibly)."))
    return findings


# --------------------------------------------------------------------------------------------
# D8  runtime-inert   (MEASURED)
# --------------------------------------------------------------------------------------------

#: Counter-name suffixes that mean "candidates the pass LOOKED at" and "candidates it CHANGED".
SEEN_TOKENS = ("scanned", "seen", "candidates", "considered", "visited", "examined", "population")
DONE_TOKENS = ("lowered", "transformed", "applied", "rewritten", "fused", "changed", "hit",
               "matched", "emitted", "promoted", "fired")


def _flat_counters(obj, prefix: str = "") -> dict[str, dict[str, int]]:
    """Flatten a report into ``{capability: {counter: int}}``.

    Two shapes are accepted, both of which already exist in this tree:
      * ``{"lower_conv_int8": {"generics_scanned": 190, "lowered": 0}}`` -- a nested report;
      * ``{"generics_scanned": 190, "lowered": 0}`` -- one pass's own ``report_out``, keyed by the
        report file's stem.
    """
    out: dict[str, dict[str, int]] = {}
    if isinstance(obj, dict):
        ints = {k: v for k, v in obj.items() if isinstance(v, bool) is False and isinstance(v, int)}
        nested = {k: v for k, v in obj.items() if isinstance(v, dict)}
        if ints:
            out[prefix or "<report>"] = ints
        for k, v in nested.items():
            out.update(_flat_counters(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            out.update(_flat_counters(item, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    return out


def detect_runtime_inert(report_paths: list[Path]) -> tuple[list[Finding], list[str]]:
    """A capability that SAW a healthy candidate population and transformed ZERO of it.

    Static analysis cannot decide whether an IR predicate is reachable -- that depends on what a
    frontend outside this repo emits. Measurement can. With no report this returns a NOTE, never a
    clean verdict: "we did not look" must not read as "nothing is inert".
    """
    if not report_paths:
        return [], ["runtime-inert: NO --report given; the measured axis was NOT evaluated. "
                    "A clean static run is NOT evidence that no pass is inert."]
    findings: list[Finding] = []
    notes: list[str] = []
    for rp in report_paths:
        if not rp.is_file():
            notes.append(f"runtime-inert: {rp} not found; that axis is UNMEASURED for this report")
            continue
        try:
            payload = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"runtime-inert: {rp} unreadable ({exc}); axis UNMEASURED")
            continue
        stem = rp.stem
        for cap, counters in sorted(_flat_counters(payload, "" if isinstance(payload, dict)
                                                   and any(isinstance(v, dict)
                                                           for v in payload.values()) else stem).items()):
            seen = max((v for k, v in counters.items()
                        if any(t in k.lower() for t in SEEN_TOKENS)), default=0)
            done = max((v for k, v in counters.items()
                        if any(t in k.lower() for t in DONE_TOKENS)), default=None)
            if done is None or seen <= 0 or done > 0:
                continue
            findings.append(Finding(
                kind="runtime-inert",
                ident=f"{cap}",
                file=_rel(rp, ROOT), line=0,
                declared=f"{cap} reports a candidate population of {seen}",
                fires_when="its predicate matches at least one of those candidates",
                blocked_by=f"it transformed ZERO of {seen} candidates on this run "
                           f"({', '.join(f'{k}={v}' for k, v in sorted(counters.items()))})",
                evidence=f"MEASURED from {_rel(rp, ROOT)} -- not inferred. A single run with a "
                         f"population but no transform is suggestive; the same result across the "
                         f"fleet is what makes it inert."))
    return findings, notes


# --------------------------------------------------------------------------------------------
# ratchet + CLI
# --------------------------------------------------------------------------------------------

def load_ratchet(path: Path) -> set[str]:
    keys: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                keys.add(entry)
    return keys


def run(scan_roots: list[Path], witness_roots: list[Path], *, root: Path,
        kinds: list[str], report_paths: list[Path], imports: bool,
        env_prefix: str, mention_roots: list[Path]) -> tuple[list[Finding], list[str]]:
    corpus = _Corpus(root=root)
    for p in _py_files(scan_roots):
        tree = _parse(p)
        if tree is not None:
            _with_parents(tree)
            corpus.scan[p] = tree
    for p in _py_files(witness_roots):
        if p in corpus.scan:
            continue
        tree = _parse(p)
        if tree is not None:
            _with_parents(tree)
            corpus.witness[p] = tree

    if "unproduced-member" in kinds:
        for wr in witness_roots:
            for dp in _safe_rglob(wr, *DATA_SUFFIXES):
                if any(tok in dp.name.lower() for tok in DEBT_FILE_TOKENS):
                    continue
                try:
                    if dp.stat().st_size > MAX_DATA_BYTES:
                        continue
                    text = dp.read_text(encoding="utf-8", errors="replace")
                    corpus.data_tokens |= _tokens(text)
                    # A dotted member (`quant_ext.dequantize_per_channel`) is ONE name; splitting
                    # it on `.` loses the very thing being looked up, and the corpus carries 2677
                    # of that op. Tokenize a second time keeping the namespace separators.
                    corpus.data_tokens |= _tokens(text, extra=".-/:")
                except OSError:
                    continue

    findings: list[Finding] = []
    notes: list[str] = []
    if "unproduced-member" in kinds:
        findings += detect_unproduced_member(corpus)
    if "always-empty-field" in kinds:
        findings += detect_always_empty_field(corpus)
    if "registry-asymmetry" in kinds:
        f, n = detect_registry_asymmetry(corpus, enabled=imports)
        findings += f
        notes += n
    if "tautological-gate" in kinds:
        findings += detect_tautological_gate(corpus)
    if "nonrecursive-aggregate" in kinds:
        findings += detect_nonrecursive_aggregate(corpus)
    if "self-comparison" in kinds:
        findings += detect_self_comparison(corpus)
    if "unchecked-subprocess-input" in kinds:
        findings += detect_unchecked_subprocess_input(corpus)
    if "dead-env-knob" in kinds:
        findings += detect_dead_env_knob(corpus, env_prefix, mention_roots)
    if "unreferenced-def" in kinds:
        findings += detect_unreferenced_def(corpus)
    if "runtime-inert" in kinds:
        f, n = detect_runtime_inert(report_paths)
        findings += f
        notes += n
    findings.sort(key=lambda f: (RANK.get(f.kind, 99), f.ident))
    return findings, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-root", action="append", default=None,
                    help="tree(s) whose declarations are audited (default: the library + tooling)")
    ap.add_argument("--witness-root", action="append", default=None,
                    help="tree(s) consulted only to answer 'does anything produce/read this?'")
    ap.add_argument("--mention-root", action="append", default=None,
                    help="tree(s) scanned for env-knob mentions (docs, settings, shell)")
    ap.add_argument("--repo-root", default=None, help="path roots are relative to (default: repo)")
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma-separated detectors to run")
    ap.add_argument("--list-kinds", action="store_true", help="print the detector names and exit")
    ap.add_argument("--report", action="append", default=[],
                    help="a pass/capability counter report (JSON) for the MEASURED axis")
    ap.add_argument("--env-prefix", default="MERLIN_", help="env-knob namespace to audit")
    ap.add_argument("--imports", dest="imports", action="store_true", default=True,
                    help="allow importing modules to resolve registries (default: on)")
    ap.add_argument("--no-imports", dest="imports", action="store_false")
    ap.add_argument("--ratchet", default=str(RATCHET), help="debt file; MAY ONLY SHRINK")
    ap.add_argument("--no-ratchet", action="store_true", help="report every finding as new")
    ap.add_argument("--write-ratchet", action="store_true",
                    help="rewrite the ratchet from this run (seeding only -- review the diff)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--stop-hook", action="store_true", help="Claude Code Stop-hook JSON")
    args = ap.parse_args(argv)

    if args.list_kinds:
        for k in sorted(KINDS, key=lambda k: RANK[k]):
            print(f"{RANK[k]}  {k}")
        return 0

    root = Path(args.repo_root).resolve() if args.repo_root else ROOT
    scan = [Path(p) if Path(p).is_absolute() else root / p
            for p in (args.scan_root or DEFAULT_SCAN_ROOTS)]
    witness = [Path(p) if Path(p).is_absolute() else root / p
               for p in (args.witness_root or DEFAULT_WITNESS_ROOTS)]
    mention = [Path(p) if Path(p).is_absolute() else root / p
               for p in (args.mention_root or ("docs", ".claude", "build_tools", "README.md",
                                               "CLAUDE.md", "AGENT.md", "pyproject.toml"))]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        print(f"[FAIL] unknown detector(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    findings, notes = run([p for p in scan if p.exists()],
                          [p for p in witness if p.exists()],
                          root=root, kinds=kinds,
                          report_paths=[Path(p) for p in args.report],
                          imports=args.imports, env_prefix=args.env_prefix,
                          mention_roots=mention)

    ratchet_path = Path(args.ratchet)
    ratcheted = set() if args.no_ratchet else load_ratchet(ratchet_path)
    new = [f for f in findings if f.key not in ratcheted]
    carried = [f for f in findings if f.key in ratcheted]
    stale = sorted(ratcheted - {f.key for f in findings})

    if args.write_ratchet:
        header = ("# Inert capabilities that MAY ONLY SHRINK.\n"
                  "#\n"
                  "# Each line is `<kind> <id>` -- a capability that is DECLARED and cannot fire.\n"
                  "# NEVER ADD A LINE. A new entry means a new feature that does nothing and says\n"
                  "# nothing; fix it, or delete the declaration. Regenerate with:\n"
                  "#   build_tools/scripts/check_inert_capabilities.py --write-ratchet\n"
                  "#\n"
                  f"# count: {len(findings)}\n")
        ratchet_path.write_text(header + "".join(f"{f.key}\n" for f in findings), encoding="utf-8")
        print(f"[write] {ratchet_path}: {len(findings)} entr(y|ies)")
        return 0

    payload = {"ratchet_count": len(ratcheted), "finding_count": len(findings),
               "new_count": len(new), "carried_count": len(carried),
               "stale_ratchet_entries": stale,
               "by_kind": {k: sum(1 for f in findings if f.kind == k) for k in kinds},
               "findings": [f.as_dict() for f in findings],
               "new": [f.key for f in new], "notes": notes}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if new else 0

    if args.stop_hook:
        if new:
            body = "\n- ".join(f"{f.key} ({f.file}:{f.line}) -- {f.blocked_by}" for f in new)
            print(json.dumps({"decision": "block",
                              "reason": f"New inert capability/ies (declared, cannot fire):\n- {body}"}))
        else:
            print(json.dumps({}))
        return 0

    for note in notes:
        print(f"[note] {note}")
    if stale:
        print(f"[  ok] {len(stale)} ratchet entr(y|ies) no longer reproduce -- delete them:")
        for s in stale:
            print(f"         {s}")
    if new:
        print(f"[FAIL] inert-capabilities: {len(new)} NEW finding(s) "
              f"({len(carried)} carried, ratchet={len(ratcheted)}):")
        for f in new:
            print(f"  - [{f.kind}] {f.file}:{f.line}  {f.ident}")
            print(f"      declared   : {f.declared}")
            print(f"      fires when : {f.fires_when}")
            print(f"      blocked by : {f.blocked_by}")
            print(f"      verified   : {f.evidence}")
        return 1
    print(f"[  ok] inert-capabilities: {len(findings)} finding(s), all ratcheted "
          f"(ratchet={len(ratcheted)}; this number MAY ONLY FALL).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
