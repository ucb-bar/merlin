"""Loop-tier -> cert-tier promotion, shared by BOTH grading brokers.

This lives in its own module because hooking only one broker is exactly the bug it was written to avoid.
Promotion was first wired into the async oracle alone, and a live run then showed the agent using the
SYNCHRONOUS self-check seven times to the async path's two -- so eight verdicts completed and promotion
fired zero times. The rule is: wherever a verdict is produced, promotion is considered. One module, two
call sites, no third copy.

The POLICY (which capsule earns the cert tier, and when a verdict may be reused) is not here either --
it is `merlin.targetgen.oracle_schedule`, which has the tests. This module is the plumbing that records
what a verdict said and enqueues what the policy asks for.

What this file DOES own is the content addressing: the whole-submission digest and its decomposition into
per-component digests (`submission_digests`). The decomposition exists because the whole-submission digest
invalidates every certificate on every edit -- fine while a round is proving functional completeness once,
ruinous for an optimization phase that edits the compiler continuously and would re-buy the minutes-per-
capsule cert tier for the whole corpus each time. The component vocabulary is DERIVED from the
submission's own manifest, never listed here; see `component_vocabulary`.

The decomposition has TWO sources, in this order -- declare, else derive:

  1. the submission's own `components:` block (`component_paths`), validated against that vocabulary;
  2. failing that, a decomposition the harness DERIVES from the submission's declared surface
     (`derived_components`) -- each command's entrypoint script, its transitive intra-submission imports,
     the paths it names as literals, and everything the manifest itself points at.

Only when BOTH refuse does the whole-submission digest come back. That fallback used to be the FIRST
answer, and it was: measured on a live arm-4 round, 19 capsules cleared the loop tier and 3 held a cert,
with the log dominated by `invalidated by <whole-submission> (changed)` -- while 17% of that round's
submission-mutating operations touched only the agent's notes, its report, and a scratch assembly file no
declared command can open. Nothing survives an edit it should have survived, and the cert tier is minutes
per capsule.

ON TOP of that source-level decomposition sits the identity that is actually PER CAPSULE: the EXECUTION
digest (`execution_digest`) -- the exact executable bytes plus the target and hardware revisions the cert
tier ran them on. The decomposition and the execution digest answer two different questions and the
scheme needs both, narrowest first:

  * `execution_digest` -- "is this capsule's PROGRAM the one that was certified?" It is per capsule, so a
    compiler edit that changes capsule 1's emitted code while capsule 2 still emits byte-identical code
    keeps capsule 2's certificate. It deliberately excludes Merlin's own source commit: an edit that
    emits identical code has not changed the program RTL certified.
  * the component decomposition -- "could any declared command even READ the bytes that moved?" Same
    answer for every capsule (see WHY NO CAPSULE DECLARES ITS OWN DEPENDENCY SET below), so it can only
    spare the inert bucket -- docs, notes, a scratch kernel.S -- measured at 17% of a live round's
    submission-mutating operations.
  * the whole-submission digest -- the fail-closed floor, which distinguishes nothing.

The source decomposition alone cannot make an optimization phase affordable, and this file's own history
is the evidence: the phase edits the compiler continuously, every such edit lands in some component, and
every component is charged to every capsule. Only the execution digest makes "capsule 1 moved, capsule 2
did not" a truthful statement. Where a narrower identity is unavailable the fallback is taken but it is
NEVER SILENT -- `_no_narrower_cause` names the missing input, on disk in `fallback_reason` and in the log,
because a correct conservative answer and a broken decomposition otherwise read identically.

The reconciliation rule the whole scheme rests on is unchanged and must stay that way: a certificate
belongs to the BYTES that earned it. `record_cert` resolves a pending entry against the identity recorded
when the job was enqueued and never re-hashes, so a result that cannot be attributed is not recorded
rather than credited to bytes edited since. Narrowing only ever changes WHICH bytes a capsule rides on;
it never lets a verdict be re-read against different ones.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from merlin.targetgen.rtl_engine_policy import ELABORATED_RTL as _ELABORATED_RTL

_NEUTRAL_SIM = "contract"   # "grade on whatever tier this target's contract resolves to"


def cert_sim(cert_tier: str) -> str | None:
    """The ``--sim`` token the BROKER will accept for ``cert_tier``, or None when none does.

    ``promote()`` used to write :data:`_NEUTRAL_SIM` unconditionally. That is correct only for a target
    whose ladder comes from its own contract, where ``--sim`` does not apply and the sentinel is the ONLY
    accepted token. A target that declares a bespoke sim ladder accepts that ladder's names and REJECTS
    the sentinel -- so every promotion request such a target wrote was refused, while the capsule had
    already been marked ``pending`` a few lines earlier and therefore stayed pending forever.

    Measured on the live gemmini round merlincirct_arm4_func_20260901_v4: 6 promotion requests, every one
    answered "rejected: --sim 'contract' is not accepted for this target. Use 'spike' or 'verilator' or
    'vcs'", and 2 capsules stranded at L3 pending. Promotion had never fired on a bespoke-sim target.

    Derived from the same allowlist the broker validates against, so the two cannot drift apart again.
    Returns None rather than a guess when nothing serves the tier: the caller must then NOT enqueue, and
    must not mark the capsule pending for a job that can never run.
    """
    try:
        from simjob_broker import _allowed_sims
        allowed = tuple(_allowed_sims())
    except Exception:  # noqa: BLE001 -- broker not importable: keep the historical sentinel
        return _NEUTRAL_SIM
    if allowed == (_NEUTRAL_SIM,):
        return _NEUTRAL_SIM
    try:
        import _common as _C
        from merlin.targetgen.runner_config import runner_config_from_manifest
        from merlin.targetgen.target_experiment import (load_capability_manifest,
                                                        load_target_experiment)
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        cfg = runner_config_from_manifest(load_capability_manifest(te.target))
        required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
        if required and cert_tier in set(cfg.rtl_tiers or ()):
            # A tier is a FIDELITY, not a historical binary binding. Under an experiment-wide engine pin
            # the required engine serves every elaborated-RTL tier it implements; consulting `tier_sim`
            # first would recover the manifest's old Verilator label, find it excluded by the broker, and
            # silently disable promotion for the whole of a GSIM-only run.
            return required if required in allowed else None
        sim = (cfg.tier_sim or {}).get(cert_tier)
        # THE CONTRACT NAMES A FIDELITY, NOT A BINARY. `tier_sim` used to read `{L3: verilator}` and the
        # broker's allowlist happened to contain that word, so this returned an engine by accident. Once
        # the contract said what it means -- `{L3: elaborated_rtl}` -- the sentinel matched no `--sim`
        # token, this returned None, and promotion silently switched off for every unpinned run: the
        # caller logs "no --sim serves L3" once and then never enqueues, which is indistinguishable from
        # a round with nothing to promote. Resolve the sentinel through the SAME availability policy the
        # contract comment names, so the fidelity is declared once and the engine chosen once.
        if sim == _ELABORATED_RTL:
            from merlin.targetgen.capsule_runner import chipyard_l3_selection
            sim = str((chipyard_l3_selection(te.target) or {}).get("engine") or "").strip() or None
    except Exception:  # noqa: BLE001 -- unresolvable map: no promotion, and the caller says so
        return None
    return sim if sim in allowed else None


def resolve_tiers(ws):
    """`(loop_tier, cert_tier, cover)` for this target, DERIVED from its own adapter map.

    loop = the fastest tier the corpus declares; cert = the deepest reachable above it. A target that
    exposes one tier gets `(None, None, None)` -- promotion disabled rather than a second tier invented.
    """
    try:
        from merlin.targetgen import capsule_runner as _CR
        from merlin.targetgen.contract.materialize import declared_oracle_tiers
        from merlin.targetgen.target_experiment import load_target_experiment
        import _common as _C
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        decl = declared_oracle_tiers(*te.graded_roots())
        loop = _CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=decl)
        full = _CR.oracle_adapters(te.target, te.sim_via)
        deeper = sorted(set(full) - set(loop))
        if not (loop and deeper):
            return None, None, None
        return sorted(loop)[0], deeper[-1], _cert_cover(ws)
    except Exception:  # noqa: BLE001 -- unresolvable ladder: no promotion, and the caller says so
        return None, None, None


def _submission_digest(ws) -> str:
    """A content address for the submission the verdict was earned against.

    Verdicts are keyed by BYTES, not by round: unchanged bytes never need re-grading, and changed bytes
    invalidate exactly the capsules they touch. Without this the loop re-certifies thirty-odd capsules to
    learn about the one that moved.
    """
    return submission_digests(ws)[0]


def execution_digest(capsule_result: str | Path) -> str | None:
    """Content identity for exactly what one capsule's hardware tier executes.

    The identity covers the ELF bytes and the target/hardware revisions recorded beside the result. It
    deliberately excludes Merlin's source commit: a source edit that emits byte-identical code has not
    changed the program RTL certifies. Missing ELF, target, or a concrete hardware revision returns
    ``None`` so scheduling falls back to the conservative submission/component digest.

    This is the identity BOTH verdict readers (``qa_check._execution_digest_from_result`` and
    ``agent_selfcheck``) put on every row. It must live here, next to the other content addressing:
    when it went missing the readers still called it, their ``except`` swallowed the ImportError, and
    every row reported ``execution_digest: null`` -- a digest that could not be computed presenting
    exactly like a submission with no artifact identity.
    """
    import hashlib
    import json
    import yaml

    cr = Path(capsule_result)
    try:
        result = json.loads(cr.read_text(encoding="utf-8"))
        manifest = yaml.safe_load((cr.parent / "run_manifest.yaml").read_text(encoding="utf-8"))
        elf = cr.parent / "generated" / "package_kernel.elf"
        target = manifest.get("target") if isinstance(manifest, dict) else None
        shas = result.get("toolchain_shas") if isinstance(result, dict) else None
        if not isinstance(target, str) or not target.strip() or not isinstance(shas, dict):
            return None
        hardware = {}
        for key, value in shas.items():
            if str(key).lower() == "merlin":
                continue
            # Hardware pins are full git/content hashes. UNKNOWN, abbreviated, or otherwise malformed
            # provenance cannot safely identify the design that executed the program.
            if (not isinstance(key, str) or not key or not isinstance(value, str)
                    or len(value) not in (40, 64)
                    or not all(c in "0123456789abcdef" for c in value)):
                return None
            hardware[key] = value
        if not hardware or not elf.is_file():
            return None
        payload = {
            "version": 1,
            "target": target,
            "hardware": hardware,
            "executable_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    except Exception:  # noqa: BLE001 -- missing/unreadable provenance is the conservative fallback
        return None


def _manifest(ws) -> dict | None:
    """The submission's own package manifest, or ``None`` if it cannot be read.

    ``None`` is the undeterminable state and is handled as such by every caller: no vocabulary, no
    decomposition, whole-submission digest. It is NOT "the submission declares no components".
    """
    import yaml
    try:
        doc = yaml.safe_load(Path(ws, "submission", "manifest.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- absent/unparseable manifest: undeterminable, never assumed empty
        return None
    return doc if isinstance(doc, dict) else None


def component_vocabulary(ws) -> set | None:
    """The legal component names for this submission, DERIVED from its manifest's ``commands`` keys.

    Why the command keys and not a list written down here. The experiment ABI already requires every
    package to declare exactly these entrypoints (``merlin/contract/schemas/manifest.schema.json``:
    ``commands.required = [parse, lower_interface_to_target, emit_command_buffer,
    lower_target_to_llvm]``, open for a target to add more), and they are the SAME axis the grader drives
    the submission through -- ``oot_runner._resolve_argv`` looks each capsule's compile step up by these
    keys. So "which command's files moved" is exactly "which step of grading this capsule could now answer
    differently", with no per-target code and no vocabulary invented here.

    The two alternatives were rejected on the facts: the Merlin pass registry
    (``xdsl_dialects.lowering.passes.CATALOG``) names Merlin's OWN passes, whose ``entry`` fields are
    ``merlin.*`` dotted paths -- an out-of-tree submission does not contain those modules, so its files
    cannot be attributed to them; and a bundle's ``allowed`` paths are read-only REPO grants
    (``merlin/contract/``, ``third_party/llvm-install/``), not submission files, so they cannot decompose
    a submission at all.

    Both spellings of the renamed 4th entrypoint resolve, reusing ``oot_runner``'s own alias map rather
    than restating it -- two copies of an alias table drift, and the one that drifts is the untested one.
    ``None`` when the manifest is unreadable (undeterminable), never an empty set.
    """
    m = _manifest(ws)
    if m is None or not isinstance(m.get("commands"), dict):
        return None
    names = {str(k) for k in m["commands"]}
    try:
        from merlin.targetgen.oot_runner import _ENTRYPOINT_ALIASES as _AL
    except Exception:  # noqa: BLE001 -- merlin not importable from the harness: names as declared
        _AL = {}
    return names | {_AL[n] for n in names if n in _AL}


def component_paths(ws) -> tuple[dict, list] | None:
    """``({component: (path-prefix, ...)}, [rejected, ...])`` from the submission's own manifest.

    The manifest's top-level ``components:`` block maps a component name to the submission-relative paths
    that implement it. It needs no schema change: ``manifest.schema.json`` is ``additionalProperties:
    true`` at the top level (its per-command objects are NOT, which is why the attribution lives beside
    ``commands`` rather than inside each one).

    A key outside :func:`component_vocabulary` is REJECTED and returned in the second slot rather than
    honoured, so a typo cannot quietly mint a component that owns no files -- a component with no files
    never changes, and since every capsule depends on every component, every certificate would then live
    forever across edits to the files the typo failed to claim.

    ``None`` when there is no vocabulary or no ``components`` block: undeterminable/undeclared, and the
    caller falls back to the whole-submission digest.
    """
    m = _manifest(ws)
    vocab = component_vocabulary(ws)
    if m is None or vocab is None or not isinstance(m.get("components"), dict):
        return None
    out, rejected = {}, []
    for name, paths in m["components"].items():
        name = str(name)
        if name not in vocab:
            rejected.append(name)
            continue
        if isinstance(paths, str):
            paths = [paths]
        out[name] = tuple(_rel_prefix(p) for p in (paths or []) if str(p).strip())
    # Returned even when `out` is empty: an all-rejected block must still be REPORTED. Swallowing it
    # would make "every component name was a typo" look exactly like "no components declared".
    return out, rejected


def _rel_prefix(p) -> str:
    """One declared path, normalized to submission-relative posix. Structural, no pattern matching."""
    s = str(p).strip().replace("\\", "/").strip("/")
    while s.startswith("./"):
        s = s[2:]
    if s == "submission":
        return ""
    if s.startswith("submission/"):
        s = s[len("submission/"):]
    return s


def _owner(rel: str, prefixes: dict) -> str | None:
    """Which component owns submission-relative ``rel``; the LONGEST declared prefix wins so a nested
    grant (``mlir_oot/lowering/tile.py``) is not swallowed by its parent (``mlir_oot/``).

    Two DIFFERENT components claiming it at the same depth is a tie, and a tie returns ``None`` -- the
    file falls to ``UNATTRIBUTED``, which every capsule depends on. Breaking the tie by dict order would
    make ownership depend on the order keys happen to appear in the manifest, so the same file could be
    attributed differently after a cosmetic reordering and a certificate would survive an edit it should
    not have. Fail closed: an ambiguously-owned file belongs to everyone.
    """
    best, best_len, tied = None, -1, False
    for name, pres in prefixes.items():
        for pre in pres:
            if pre and not (rel == pre or rel.startswith(pre + "/")):
                continue
            if len(pre) > best_len:
                best, best_len, tied = name, len(pre), False
            elif len(pre) == best_len and name != best:
                tied = True
    return None if tied else best


# ---------------------------------------------------------------------------------------------------
# The DERIVED decomposition: what a command's process can read, worked out from the submission's own
# declared surface. Reached only when the submission declares no usable `components:` block.
# ---------------------------------------------------------------------------------------------------
#
# "Declare, else derive". A submission that says which files implement which command gets exactly that
# (`component_paths`). One that says nothing used to get the whole-submission digest, which is the bug
# this file exists to remove: measured on a live arm-4 round, every certificate died on every edit, and
# the edits were dominated by files no command can even open (the agent's own notes, its report, a
# scratch `.S` it hand-assembled through a brokered tool). 17% of that run's submission-mutating
# operations touched ONLY such files.
#
# What "can read" means is fixed by the experiment ABI, not guessed: `oot_runner.run_entrypoint` invokes
# a declared command as a SUBPROCESS with `cwd=<package root>` and an argv in which the only non-literal
# tokens are the absolute `{input_mlir}` / `{output_json}` the runner substitutes. So the submission bytes
# that can change a verdict are the bytes that process reaches:
#
#   * the files the command's own argv names (its entrypoint script), and
#   * everything those transitively import from inside the submission, and
#   * everything they name by a resolvable path literal (a data file read at a fixed path), and
#   * everything the manifest itself names (build inputs, a second tool) -- charged to EVERY command,
#     since the manifest is what declares the commands in the first place.
#
# Three buckets come out, and the difference between the last two is the whole safety argument:
#
#   attributed   -- reached as above; feeds its component's digest.
#   UNATTRIBUTED -- NOT reached, but the same KIND of thing as reached code (its suffix is one the
#                   closure already has, one a dynamic loader in the closure names, or one the runner's
#                   integrity scan reads). A dead module and a plugin loaded by a path we could not
#                   resolve look identical from here, so this bucket is a dependency of every capsule.
#   inert        -- NOT reached and not that kind of thing. Nothing depends on it.
#
# The derivation REFUSES rather than guesses. A submission in a language whose imports this cannot trace,
# a command whose argv names no submission file, a closure file that will not parse -- any of those and
# `derived_components` returns None, and the caller falls back to the whole-submission digest exactly as
# before. The one judgement it does make is stated plainly: a read whose path expression contains no
# string literal is taken to be a read of a path handed in from outside the process, because under the
# ABI above that is what the runner passes it. A submission that computes an intra-package path out of
# data with no literal anywhere in the expression would defeat that, and would keep a certificate it
# should have lost; declaring `components:` is the answer for a package built that way.
#
# "Inert" means "cannot change a capsule's verdict" -- NOT "unimportant". The agent's REPORT.md is a
# required deliverable and is inert by this definition, because no graded command opens it.
INERT = "<inert>"

_DERIVED_CACHE: dict = {}


def _manifest_strings(node):
    """Every string anywhere in the manifest, however nested. Structural walk, no pattern matching."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _manifest_strings(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _manifest_strings(v)


def _under(sub: Path, token: str):
    """``token`` resolved to an existing path under the submission root, or None.

    Tokens arrive from the manifest, so they may be doubly rooted (``submission/mlir_oot/opt``) exactly as
    ``oot_runner._resolve_argv`` already documents; that prefix is stripped when, and only when, the
    remainder resolves. Anything absolute, empty, escaping the root, or containing whitespace is not a
    package-relative path and is skipped.
    """
    s = str(token).strip().replace("\\", "/")
    if not s or s.startswith("/") or any(c.isspace() for c in s):
        return None
    for pfx in ("./submission/", "submission/", "./"):
        if s.startswith(pfx) and not (sub / s).exists():
            s = s[len(pfx):]
    s = s.strip("/")
    if not s or s in (".", ".."):
        return None
    try:
        p = (sub / s).resolve()
        if not p.exists() or p == sub.resolve() or sub.resolve() not in p.parents:
            return None
    except OSError:                      # a "path" long enough to be prose, not a path
        return None
    return p


def _files_at(p: Path, token: str = "/"):
    """The file(s) a resolved token stands for. A DIRECTORY counts only when the token was written as a
    path (it contains a separator): a bare single-segment string that happens to match a directory name is
    almost always a module or an identifier, not a data directory, and treating `"xdsl"` as a grant over
    the whole vendored package pulls that package's READMEs into the executable surface -- which then
    makes every other `.md` in the submission look like the same kind of thing as reached code."""
    if p.is_dir():
        return [x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts] \
            if "/" in str(token) else []
    return [p] if "__pycache__" not in p.parts else []


def _command_roots(sub: Path, m: dict):
    """``{command: [entrypoint file, ...]}`` -- the submission files each declared command's argv names.

    ``{tool}`` is substituted from ``entrypoints.tool`` the way the runner substitutes it, so a package
    that references its tool through the placeholder and one that spells the path out both resolve. A
    command whose argv names NO submission file is the refusal case: we do not know what it runs.
    """
    cmds = m.get("commands")
    if not isinstance(cmds, dict) or not cmds:
        return None
    tool = ((m.get("entrypoints") or {}) if isinstance(m.get("entrypoints"), dict) else {}).get("tool")
    out = {}
    for name, spec in cmds.items():
        argv = (spec or {}).get("argv") if isinstance(spec, dict) else None
        if not isinstance(argv, (list, tuple)):
            return None
        roots = []
        for tok in argv:
            tok = str(tok)
            if "{tool}" in tok and tool:
                tok = tok.replace("{tool}", str(tool))
            if "{" in tok:                     # an unsubstituted I/O placeholder: not a submission file
                continue
            p = _under(sub, tok)
            if p is not None and p.is_file():
                roots.append(p)
        if not roots:
            return None
        out[str(name)] = roots
    return out


# Constructs that can pull a module in from outside the static import graph. Their presence does not
# abandon the derivation -- it WIDENS what counts as "the same kind of thing as reached code", so an
# unreached file a loader could still pick up lands in UNATTRIBUTED rather than in inert.
_DYNAMIC_IMPORT = frozenset({"__import__", "import_module", "load_module", "exec_module",
                             "spec_from_file_location", "module_from_spec", "iter_modules"})


def _py_module_file(sub: Path, dotted: str):
    p = sub.joinpath(*dotted.split("."))
    for cand in (p.with_suffix(".py"), p / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _py_imports(tree, rel_parts) -> set:
    """Dotted module names a parsed file imports, with relative imports resolved against its package."""
    pkg = list(rel_parts[:-1])
    out = set()
    for n in _ast().walk(tree):
        if isinstance(n, _ast().Import):
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, _ast().ImportFrom):
            base = n.module or ""
            if n.level:
                up = pkg[:len(pkg) - (n.level - 1)] if n.level > 1 else pkg
                base = ".".join(up + ([base] if base else []))
            if base:
                out.add(base)
            for a in n.names:
                out.add(f"{base}.{a.name}" if base else a.name)
    return out


def _ast():
    import ast
    return ast


def _scan(f: Path, memo=None):
    """``(tree, string_literals, has_dynamic_import)`` for one Python source file, or None if it will not
    parse. Unparseable means unknown content, and unknown content is the refusal case."""
    if memo is not None and f in memo:
        return memo[f]
    ast = _ast()
    try:
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 -- not Python, or broken Python: we cannot say what it reads
        if memo is not None:
            memo[f] = None
        return None
    lits, dyn = set(), False
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            lits.add(n.value)
        elif isinstance(n, ast.Call):
            fn = n.func
            nm = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if nm in _DYNAMIC_IMPORT:
                dyn = True
    out = (tree, lits, dyn)
    if memo is not None:
        memo[f] = out
    return out


def _closure(sub: Path, roots, memo=None):
    """``(files, literals, dynamic_suffixes)`` reachable from ``roots``, or None on any unparseable file.

    Follows intra-submission imports only: a name that resolves to no file under the submission root is a
    stdlib/third-party import the package does not carry, and cannot be edited by the agent.
    """
    seen, lits, dyn_sufs = set(), set(), set()
    stack = list(roots)
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        got = _scan(f, memo)
        if got is None:
            return None
        tree, flits, dyn = got
        lits |= flits
        if dyn:
            # Bounded by what the loader ITSELF names: a suffix-shaped literal in the file that does the
            # dynamic import. Widening on every literal in the closure drags in noise ('.text', '.word')
            # from unrelated code and would make the whole tree "the same kind of thing".
            dyn_sufs |= {s for s in flits
                         if s.startswith(".") and 2 <= len(s) <= 6 and s[1:].isalnum()}
        for mod in _py_imports(tree, f.relative_to(sub).parts):
            parts = mod.split(".")
            for i in range(1, len(parts) + 1):
                g = _py_module_file(sub, ".".join(parts[:i]))
                if g is not None and g not in seen:
                    stack.append(g)
    return seen, lits, dyn_sufs


def _src_suffixes() -> frozenset:
    """Suffixes the runner's own integrity scan READS. A violation in one of these fails the run even
    when nothing imports the file, so such a file is never inert. Taken from the runner, not restated."""
    try:
        from merlin.targetgen.oot_runner import _SRC_SUFFIXES
        return frozenset(_SRC_SUFFIXES)
    except Exception:  # noqa: BLE001 -- runner not importable here: the closure's own suffixes still apply
        return frozenset()


def derived_components(ws):
    """``({component: frozenset(rel)}, frozenset(unattributed), frozenset(inert))`` or None to refuse.

    Paths are submission-relative. See the block comment above for what each bucket means and for the one
    judgement this makes. ``None`` is "no derivation" and the caller must then use the whole-submission
    digest -- never a partial answer.
    """
    m = _manifest(ws)
    sub = Path(ws, "submission")
    if m is None or not sub.is_dir():
        return None
    if str(m.get("language") or "").strip().lower() != "python":
        # Only Python's import graph is traceable from here. Refusing is the honest answer for a package
        # whose surface is a compiled binary: we cannot see what it opens.
        return None
    roots = _command_roots(sub, m)
    if roots is None:
        return None
    key = (str(sub), tuple(sorted((f.relative_to(sub).as_posix(), f.stat().st_mtime_ns, f.stat().st_size)
                                  for f in sub.rglob("*") if f.is_file() and "__pycache__" not in f.parts)))
    hit = _DERIVED_CACHE.get(key)
    if hit is not None:
        return hit
    every = [f for f in sub.rglob("*") if f.is_file() and "__pycache__" not in f.parts]

    # Files the MANIFEST names OUTSIDE the per-command sections are charged to every component -- a build
    # input, a second tool, anything the package points at globally: a change to one of those, or to the
    # manifest itself, can change what every command does. `commands` and `entrypoints` are deliberately
    # excluded, because those ARE the per-command attribution (`_command_roots` resolves them); folding
    # them in here would charge each command's own script to all the others and collapse the whole
    # decomposition back to one bucket.
    _global = {k: v for k, v in m.items() if k not in ("commands", "entrypoints")}
    shared = {sub / "manifest.yaml"} if (sub / "manifest.yaml").is_file() else set()
    for s in _manifest_strings(_global):
        p = _under(sub, s)
        if p is not None:
            shared.update(_files_at(p, s))

    comps, kind_sufs, memo = {}, set(_src_suffixes()), {}
    for name, rs in roots.items():
        got = _closure(sub, rs, memo)
        if got is None:
            return None                      # a file in the surface we cannot read: refuse, do not guess
        files, lits, dyn_sufs = got
        for s in lits:                       # a data file opened at a fixed, resolvable path
            p = _under(sub, s)
            if p is not None:
                files.update(_files_at(p, s))
        files |= shared
        comps[name] = frozenset(f.relative_to(sub).as_posix() for f in files)
        kind_sufs |= {f.suffix for f in files} | dyn_sufs

    claimed = set().union(*comps.values()) if comps else set()
    unattributed, inert = set(), set()
    for f in every:
        rel = f.relative_to(sub).as_posix()
        if rel in claimed:
            continue
        (unattributed if f.suffix in kind_sufs else inert).add(rel)
    out = (comps, frozenset(unattributed), frozenset(inert))
    # Bounded: a broker lives for the whole run and the agent edits continuously, so an unbounded map
    # keyed by submission state grows one entry per edit for hours. Only the current state is ever a hit.
    if len(_DERIVED_CACHE) > 4:
        _DERIVED_CACHE.clear()
    _DERIVED_CACHE[key] = out
    return out


def decomposition(ws) -> dict:
    """How this submission's bytes decompose, and WHERE that decomposition came from.

    ``{"owners": {rel: {component, ...}}, "names": {component, ...}, "rejected": [...],
      "source": "declared"|"derived"|None, "inert": frozenset(rel)}``

    "Declare, else derive": an agent-declared ``components:`` block wins, and is used even when some of
    its names were rejected -- but a block whose names were ALL rejected leaves nothing to decompose with,
    so it falls through to the derivation rather than to the whole submission. Either way the rejected
    names are carried out for the caller to report; a typo that quietly costs the saving is the failure
    mode this exists to make visible.
    """
    from merlin.targetgen.oracle_schedule import UNATTRIBUTED
    sub = Path(ws, "submission")
    rels = [f.relative_to(sub).as_posix() for f in sub.rglob("*")
            if f.is_file() and "__pycache__" not in f.parts] if sub.is_dir() else []
    decl = component_paths(ws)
    prefixes, rejected = decl if decl else ({}, [])
    if prefixes:
        owners = {rel: {_owner(rel, prefixes) or UNATTRIBUTED} for rel in rels}
        return {"owners": owners, "names": set(prefixes) | {UNATTRIBUTED}, "rejected": rejected,
                "source": "declared", "inert": frozenset()}
    der = derived_components(ws)
    if der is None:
        return {"owners": {}, "names": set(), "rejected": rejected, "source": None,
                "inert": frozenset()}
    comps, unattributed, inert = der
    owners = {}
    for name, files in comps.items():
        for rel in files:
            owners.setdefault(rel, set()).add(name)
    for rel in unattributed:
        owners.setdefault(rel, set()).add(UNATTRIBUTED)
    names = set(comps) | ({UNATTRIBUTED} if unattributed else set())
    return {"owners": owners, "names": names, "rejected": rejected, "source": "derived", "inert": inert}


def submission_digests(ws) -> tuple:
    """``(whole_digest, {component: digest}, rejected_names)`` for the submission on disk.

    The whole digest is byte-for-byte what it always was -- one pass, sorted paths, path bytes then file
    bytes -- so turning components on never invalidates a certificate by itself. Each file additionally
    feeds the hasher of EVERY component that owns it (a shared helper genuinely belongs to each command
    that imports it, and a change to it must move all of their digests). A file the decomposition cannot
    place feeds ``UNATTRIBUTED``, which every capsule depends on; a file it places OUTSIDE every command's
    reach feeds nothing, which is where the saving comes from. The component map is ``{}`` when there is
    no decomposition at all, and the scheduler reads that as "compare the whole submission"
    (see ``CapsuleState._dep_components``).
    """
    import hashlib
    d = decomposition(ws)
    owners, names = d["owners"], d["names"]
    h = hashlib.sha256()
    parts = {}
    sub = Path(ws, "submission")
    for f in sorted(sub.rglob("*")):
        if not (f.is_file() and "__pycache__" not in f.parts):
            continue
        key, body = f.relative_to(ws).as_posix().encode(), f.read_bytes()
        h.update(key)
        h.update(body)
        for owner in owners.get(f.relative_to(sub).as_posix(), ()):
            ph = parts.get(owner)
            if ph is None:
                ph = parts[owner] = hashlib.sha256()
            ph.update(key)
            ph.update(body)
    # A declared component with no files on disk still gets an entry -- an EMPTY digest, not a missing
    # one. Missing would read as undeterminable and re-run forever; empty is a real, comparable state
    # that changes the moment the component gains a file.
    comps = {name: (parts[name].hexdigest()[:16] if name in parts
                    else hashlib.sha256().hexdigest()[:16]) for name in names}
    return h.hexdigest()[:16], comps, d["rejected"]


# ---------------------------------------------------------------------------------------------------
# WHY NO CAPSULE DECLARES ITS OWN DEPENDENCY SET
# ---------------------------------------------------------------------------------------------------
# `capsule.yaml` used to accept a `depends_on: [<component>, ...]` block, read here and handed to
# `CapsuleState.depends_on` so a capsule could keep its certificate across an edit to a component it did
# not name. Nothing ever declared one -- 0 of the 509 capsules in the shipped corpus -- and the field is
# gone from `capsule.schema.json` because a truthful narrow declaration cannot exist:
#
#   * THE GRADER RUNS EVERY COMMAND FOR EVERY CAPSULE. `capsule_common.run_entrypoints` -- the shared ABI
#     front half every target's runner goes through -- invokes parse, lower_interface_to_target,
#     emit_command_buffer and the fourth entrypoint unconditionally, and a nonzero rc from any of them is
#     a CertFailure. `oot_runner`'s K-ladder does the same at K2-K6. There is no per-capsule command
#     subset anywhere, so an edit to ANY command's files can flip ANY capsule's verdict. A capsule naming
#     a proper subset would keep a certificate its current bytes did not earn -- the one direction
#     `oracle_schedule` calls "far worse than re-running".
#   * THE NARROWING WOULD BUY NOTHING ANYWAY. Every package in this repo maps all four commands onto the
#     same `{tool}` argv, differing only in flags, so `_command_roots` hands the four components the same
#     entrypoint and `_closure` gives them identical file sets. Four identical digests move together on
#     every edit; picking a subset of them changes no answer.
#   * AND IT COULD NOT BE AUTHORED. The vocabulary is the SUBMISSION's manifest `commands` keys
#     (`component_vocabulary`, enforced by `component_paths` rejecting anything else). A capsule.yaml is
#     written against the corpus long before any submission exists, so its author would be naming
#     components of a package that has not been built.
#
# The saving the decomposition actually delivers needs no declaration: it is the `inert` bucket -- bytes
# the derivation PROVED no command can read -- and `promote()` gives it to every capsule by passing the
# full component set. That is measured at 17% of a live round's submission-mutating operations (see the
# module docstring), and it is a fact about the submission, not a claim by a capsule.


def _cert_cover(ws) -> set | None:
    """Which capsules are worth certifying at all. The hardware cannot tell two capsules in the same
    (family, dtype) cell apart, so certifying both spends minutes to learn nothing. `None` on any failure
    -> certify anything eligible, because a cover that silently comes back empty is indistinguishable
    from everything already being done."""
    try:
        from merlin.targetgen.contract.materialize import cert_capsule_cover
        from merlin.targetgen.target_experiment import load_target_experiment
        import _common as _C
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        # Pass the tile edge so the cover certifies PARTIAL tiles as their own cell. A cover built on
        # family and dtype alone can pick, per cell, the capsule whose extents happen to divide evenly and
        # then certify no ragged extent anywhere -- and a partial tile is exactly what a functional model
        # is least able to stand in for (a taped-out unit here got `n % 64 != 0` wrong while every
        # functional check passed).
        _td = None
        try:
            from merlin.targetgen.corpus_spec import _tile_dim
            from merlin.targetgen.target_experiment import load_capability_manifest
            _td = int(_tile_dim(te.target, load_capability_manifest(te.target).contract)) or None
        except Exception:  # noqa: BLE001 -- no derivable tile edge: cover without the alignment axis
            _td = None
        # Exclusions travel WITH the roots. A capsule the descriptor withholds from the paid loop cannot
        # stand for its cell, because promotion only enqueues capsules in the cover — picking one that
        # never runs retires the cell for a certificate nobody will produce.
        return set(cert_capsule_cover(te.graded_roots(), tile_dim=_td,
                                      exclude=set(getattr(te, "graded_exclude", ()) or ()))["capsules"])
    except Exception:  # noqa: BLE001 -- no resolvable corpus: stay permissive, never silently empty
        return None


def _tier_state(ws) -> dict:
    import json as _j
    f = Path(ws, "qa", "tier_state.json")
    try:
        return _j.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_tier_state(ws, st) -> None:
    """Replace the state file ATOMICALLY.

    Two brokers write this file (the synchronous self-check broker and the async simjob broker). A plain
    ``write_text`` truncates first, so the other broker can read a half-written one; `_tier_state` answers
    an unparseable read with ``{}``, and the next save then PERSISTS that empty dict -- every capsule's
    recorded verdict gone, and re-bought at minutes each. A temp file in the same directory plus
    ``os.replace`` makes the swap atomic, so a concurrent reader sees either the old state or the new one
    and never an empty one.
    """
    import json as _j
    f = Path(ws, "qa", "tier_state.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(_j.dumps(st, indent=2))
    os.replace(tmp, f)


# ------------------------------------------------------------------------------------------------
# The certificate LEDGER: one recorded verdict per (capsule, tier, the bytes it belongs to).
#
# The single-slot record this replaces was the reason a paid-for certificate was thrown away. A cert job
# takes minutes; the agent keeps editing while it runs; the next loop verdict re-enqueued the capsule and
# OVERWROTE the pending record. When the in-flight job then finished, `record_cert` found a record for
# different bytes, could not attribute the result (correctly -- see its docstring), and dropped it. So the
# RTL time bought nothing and the capsule read `certified: None` forever. Measured on
# merlincirct_arm4_func_20260901_codex1 round 1: 17 cert jobs, all passing 1/1, zero certificates recorded.
#
# Keyed per identity, a verdict is RETAINED instead of discarded: a completed job resolves the record that
# was enqueued FOR IT, and an edit that changes a capsule's program leaves the old record in place rather
# than destroying it -- so if those exact bytes come back, so does their certificate. Nothing is ever
# consulted unless its identity matches the current bytes EXACTLY, which is why retaining cannot turn a
# stale certificate into a valid one.
# ------------------------------------------------------------------------------------------------
_LEDGER = "<certs>"   # reserved key inside a capsule's tier map; delimiters no tier label can produce

# How many records one (capsule, tier) retains. The state file is re-read and re-written on EVERY verdict,
# and a continuous round produces dozens per hour, so an unbounded ledger would grow the hot file without
# limit. The bound only ever costs a re-run (the conservative direction): an evicted record means the
# scheduler no longer knows those bytes passed, never that it believes they did. An OUTSTANDING record is
# never evicted -- that is an in-flight job whose result would otherwise become unattributable.
_LEDGER_KEEP = 16


def _valid_execution(value) -> bool:
    """Whether *value* is the artifact identity the result readers emit.

    Imported, never restated: two copies of this predicate is how the recorder and the scheduler come to
    disagree about what counts as an identity, and the disagreement always resolves as a silent fallback.
    """
    from merlin.targetgen.oracle_schedule import valid_execution_digest
    return valid_execution_digest(value)


def record_identity(entry) -> str:
    """Which bytes one recorded verdict belongs to, as a ledger key.

    The exact executable identity when the run that earned it produced one, else the whole-submission
    digest. The two key spaces cannot collide: an execution identity is 64 hex characters, and the
    fallback carries a prefix no digest can spell.
    """
    ed = (entry or {}).get("execution_digest")
    if _valid_execution(ed):
        return str(ed)
    return "submission:" + str((entry or {}).get("digest") or "")


def current_identity(execution_digest, digest) -> str:
    """The ledger key for the bytes on disk RIGHT NOW, in the same space as :func:`record_identity`."""
    if _valid_execution(execution_digest):
        return str(execution_digest)
    return "submission:" + str(digest)


def _slots_ro(st, name, tier) -> dict:
    """``{identity: entry}`` for one (capsule, tier). READ-ONLY.

    It must not create the capsule: a result for a capsule nobody promoted has to leave no trace at all
    (that is what makes an unattributable result recordable as "nothing happened" rather than as a guess).

    A state file written before the ledger existed carries only the single mirror entry. That is seeded
    into the returned map rather than reported as absent -- absent would read as "nothing was ever
    recorded here" and re-buy the cert tier for bytes that already hold a verdict.
    """
    led = (st.get(name) or {}).get(_LEDGER)
    slots = led.get(tier) if isinstance(led, dict) else None
    if isinstance(slots, dict) and slots:
        return slots
    mirror = (st.get(name) or {}).get(tier)
    return {record_identity(mirror): mirror} if isinstance(mirror, dict) else {}


def _slots(st, name, tier) -> dict:
    """:func:`_slots_ro`, materialized in ``st`` so it can be written to."""
    per = st.setdefault(name, {})
    led = per.get(_LEDGER)
    if not isinstance(led, dict):
        led = per[_LEDGER] = {}
    slots = led.get(tier)
    if not isinstance(slots, dict):
        slots = led[tier] = dict(_slots_ro(st, name, tier))
    return slots


def _record(st, name, tier, entry) -> None:
    """Store one verdict for (capsule, tier): in the ledger under the bytes it belongs to, AND as the
    ``st[capsule][tier]`` mirror every existing reader of this file already uses. The mirror is the most
    recently written record; the ledger is every record, bounded by :data:`_LEDGER_KEEP`."""
    key = record_identity(entry)
    slots = _slots(st, name, tier)
    slots.pop(key, None)                 # re-insert so the most recently touched record is the newest
    slots[key] = entry
    st.setdefault(name, {})[tier] = entry
    # Evict oldest-first (dict order is insertion order, and the state file round-trips it), skipping the
    # record just written and every outstanding one.
    for old in list(slots):
        if len(slots) <= _LEDGER_KEEP:
            break
        if old == key or (slots[old] or {}).get("status") == "pending":
            continue
        del slots[old]


def _recorded_tiers(per) -> list:
    """Every tier one capsule holds a verdict for, from the mirror and the ledger both. The reserved
    ledger key is never mistaken for a tier -- a phantom tier would enter the scheduler as a verdict."""
    out = {t for t, v in (per or {}).items() if t != _LEDGER and isinstance(v, dict)}
    led = (per or {}).get(_LEDGER)
    if isinstance(led, dict):
        out |= {t for t, v in led.items() if isinstance(v, dict)}
    return sorted(out)


def _verdict_for(st, name, tier, identity):
    """The record to judge (capsule, tier) by: the one earned by EXACTLY the current bytes when the
    ledger holds it, otherwise the mirror.

    Preferring the exact-identity record is what stops a certificate being discarded by an edit it does
    not depend on. It cannot loosen anything: the record that comes back is still put through
    ``CapsuleState.invalidated_by``, so a record for other bytes is invalidated rather than trusted, and
    a capsule with no matching record falls back to exactly today's comparison.
    """
    slots = _slots_ro(st, name, tier)
    hit = slots.get(identity)
    if isinstance(hit, dict):
        return hit
    mirror = (st.get(name) or {}).get(tier)
    return mirror if isinstance(mirror, dict) else None


def _no_narrower_cause(execution_digest, comps) -> str | None:
    """Why this capsule's staleness cannot be decided by anything narrower than bytes every capsule
    shares, or ``None`` when the narrowest identity was available.

    The conservative fallback stays the fallback, but it stops being SILENT. Every line of the log that
    diagnosed this defect read ``invalidated by <whole-submission> (changed)`` and named none of the
    inputs that were missing, so a correct conservative answer and a broken decomposition looked
    identical. Each clause below names one input that was absent -- so the reason is recorded, on disk
    and in the log, rather than inferred.

    NOTE the two clauses are different KINDS of loss and both are worth naming. Without an execution
    identity, every capsule rides on the same submission bytes and one capsule's edit requeues the whole
    corpus. Without a decomposition on top of that, even bytes no declared command can read count.
    """
    if execution_digest is not None:
        return None                      # the narrowest identity is available; no fallback happened
    why = ["no execution identity for this capsule in this verdict"]
    if not comps:
        why.append("the submission has no component decomposition, declared or derived")
    return "; ".join(why)


def record_cert(ws, verdict, cert_tier, log=None, identity=None) -> list[str]:
    """Write a COMPLETED promotion's result into the tier state, against the bytes that earned it.

    `promote()` marks a capsule `pending` when it enqueues the cert job, and the broker's reap skips
    `_promote` for a job that WAS a promotion -- correctly, so a cert verdict cannot re-enqueue itself.
    But nothing then wrote the outcome back, so a capsule stayed `pending` forever: the certificate was
    earned on real RTL and discarded, and the next loop verdict re-certified the same bytes.

    Measured on merlincirct_arm4_func_20260901_v4 and _p2: promotions fired and COMPLETED (21 and 3
    `simdone_promo*` respectively, one verified `barrier_tier=L3 barrier_status=pass`), while both tier
    states showed only `L3: pending` and never once `L3: pass`.

    The digest is NOT recomputed here. A cert belongs to the exact bytes that were pending when the job
    was enqueued; re-hashing now would attribute it to whatever the agent has edited since. So this only
    resolves an existing pending entry, and leaves anything else alone -- a result with no pending entry
    is a result we cannot attribute, and that is recorded by doing nothing rather than by guessing.

    That property is enforced against the per-identity LEDGER rather than against a single slot. The
    single slot was overwritten by the next re-enqueue, so a completed job routinely arrived to find a
    record for OTHER bytes: with an artifact identity that was refused (the cert was dropped), and
    WITHOUT one it was silently accepted and the certificate was re-attributed to bytes that never earned
    it. Looking the completed job's own identity up in the ledger resolves exactly the record it belongs
    to, and refuses when no such record exists.

    ``identity`` is the ledger key the ENQUEUER wrote onto the request, forwarded by the reap. It is what
    lets a result whose reader produced no artifact identity still be attributed exactly, instead of
    falling back to "the one outstanding record" (which cannot be used at all once two are outstanding).
    A result that DOES carry an artifact identity always decides for itself: the job may have launched
    after an edit, and then only the identity it actually ran may be credited.
    """
    rows = (verdict or {}).get("per_capsule") or []
    if not rows:
        return []
    st = _tier_state(ws)
    resolved = []
    for row in rows:
        name = str((row or {}).get("capsule") or "")
        if not name:
            continue
        slots = _slots_ro(st, name, cert_tier)
        completed = row.get("execution_digest")
        if _valid_execution(completed):
            # The identity the job ACTUALLY ran. It selects its own record; a job whose bytes hold no
            # record is unattributable, however many other records this capsule has.
            key = str(completed)
            entry = slots.get(key)
            if not isinstance(entry, dict) or entry.get("status") != "pending":
                if log is not None:
                    print(f"[promote] {name} {cert_tier} result not recorded: no outstanding record "
                          f"for the execution identity this job ran", file=log, flush=True)
                continue
        elif identity is not None:
            # The enqueuer stated which record this job was launched for. That is a fact about the
            # request, not a guess about the result, so it attributes exactly -- including when several
            # records are outstanding at once, which is the case the heuristic below cannot serve.
            key = str(identity)
            entry = slots.get(key)
            if not isinstance(entry, dict) or entry.get("status") != "pending":
                if log is not None:
                    print(f"[promote] {name} {cert_tier} result not recorded: the record this job was "
                          f"enqueued for is no longer outstanding", file=log, flush=True)
                continue
            if _valid_execution(entry.get("execution_digest")):
                # The record names an exact artifact and the result does not carry one -- so whether the
                # job ran THAT artifact cannot be established. The request only says what was asked for;
                # the agent may have edited between enqueue and launch. Fail closed: an unreadable
                # artifact identity is never evidence that the right artifact ran.
                if log is not None:
                    print(f"[promote] {name} {cert_tier} result not recorded: the record names an "
                          f"execution identity and the result carries none", file=log, flush=True)
                continue
        else:
            # No artifact identity on the result and none on the request: the only attributable case is a
            # single outstanding record that ALSO carries no artifact identity. Two outstanding records
            # and a result that cannot say which it belongs to is precisely the misattribution this
            # guards -- refuse.
            outstanding = {i: e for i, e in slots.items()
                           if isinstance(e, dict) and e.get("status") == "pending"}
            if len(outstanding) != 1:
                if log is not None and outstanding:
                    print(f"[promote] {name} {cert_tier} result not recorded: {len(outstanding)} "
                          f"outstanding records and the result carries no execution identity, so which "
                          f"bytes earned it cannot be determined", file=log, flush=True)
                continue
            key, entry = next(iter(outstanding.items()))
            if _valid_execution(entry.get("execution_digest")):
                if log is not None:
                    print(f"[promote] {name} {cert_tier} result not recorded: the outstanding record "
                          f"names an execution identity this result does not carry", file=log, flush=True)
                continue
        passed = bool(row.get("pass"))
        entry = dict(entry)
        entry["status"] = "pass" if passed else "fail"
        slots_w = _slots(st, name, cert_tier)
        slots_w.pop(key, None)           # re-insert: a just-resolved record is the freshest, not the oldest
        slots_w[key] = entry
        # The mirror moves only when it is the SAME record. Moving it to a different identity is exactly
        # the re-attribution this function exists to prevent.
        mirror = (st.get(name) or {}).get(cert_tier)
        if isinstance(mirror, dict) and record_identity(mirror) == key:
            st[name][cert_tier] = entry
        resolved.append(f"{name}={'pass' if passed else 'fail'}")
    if resolved:
        _save_tier_state(ws, st)
        if log is not None:
            print(f"[promote] {cert_tier} recorded: {resolved}", file=log, flush=True)
    return resolved


def promote(ws, ch, verdict, loop_tier, cert_tier, cover, log):
    """Record what the loop tier just learned, and enqueue cert jobs for what it unlocked.

    Returns the capsule names promoted. Enqueues by writing a `simreq_` the broker's own queue picks up --
    the same path an agent request takes, so it inherits the constrained-runner validation rather than
    routing around it.
    """
    import json as _j
    from merlin.targetgen.oracle_schedule import (WHOLE_SUBMISSION, CapsuleState, Verdict, schedule,
                                                  valid_execution_digest)

    digest, comps, rejected = submission_digests(ws)
    d = decomposition(ws)
    st = _tier_state(ws)
    if rejected:
        # Loud, not silent: a rejected name owns no bytes, so the component it was meant to be never
        # moves and every certificate keeps riding on the files it failed to claim.
        print(f"[promote] manifest components outside the declared command vocabulary, ignored: "
              f"{sorted(rejected)}", file=log, flush=True)
    print(f"[promote] components: {d['source'] or 'none (whole-submission digest)'}"
          + (f" -- {len(comps)} component(s), {len(d['inert'])} file(s) outside every command's reach"
             if d["source"] else ""), file=log, flush=True)

    # What EVERY capsule rides on: all of it. "Everything" is expressed as the set of components rather
    # than as the whole digest, and the two differ by exactly the bytes the decomposition PROVED no
    # command can read. With no decomposition the fallback is `None`, which is the whole digest, i.e. the
    # behaviour before the decomposition existed. No capsule may narrow this -- see WHY NO CAPSULE
    # DECLARES ITS OWN DEPENDENCY SET above for why a per-capsule claim cannot be truthful.
    #
    # The per-capsule saving comes from the EXECUTION identity instead, one layer down: the source-level
    # decomposition can only spare bytes no command reads (the same set for every capsule), while the
    # executable identity distinguishes capsule from capsule, because each capsule emits its own program.
    # That is the layer at which "capsule 1 moved, capsule 2 did not" is a truthful statement.
    deps = tuple(sorted(comps)) or None

    # Record what the loop tier just learned, keyed by the bytes that earned it -- the exact per-capsule
    # executable identity when the reader produced one, and BOTH the whole digest and the per-component
    # decomposition regardless, because a verdict that carries only the whole digest cannot be
    # re-examined per component later (it comes back UNDETERMINABLE, which re-runs).
    execution_by_name = {}
    for row in (verdict.get("per_capsule") or []):
        name = row.get("capsule")
        if not name:
            continue
        name = str(name)
        execution_digest = row.get("execution_digest")
        if not valid_execution_digest(execution_digest):
            execution_digest = None
        execution_by_name[name] = execution_digest
        entry = {"status": "pass" if row.get("pass") else "fail",
                 "digest": digest, "components": dict(comps)}
        if execution_digest is not None:
            entry["execution_digest"] = execution_digest
        # RECORD WHY, on disk, when this verdict can only be compared against bytes every capsule shares.
        # A conservative fallback that says nothing is why the defect read as correct behaviour for a
        # whole round.
        _why_broad = _no_narrower_cause(execution_digest, comps)
        if _why_broad:
            entry["fallback_reason"] = _why_broad
        _record(st, name, loop_tier, entry)

    # WHAT to run next is `oracle_schedule`'s decision, not this file's. The rules (a cert tier is gated
    # on the loop tier passing; the cert tier runs a representative cover; a verdict already earned by
    # these bytes is never re-run) were implemented here once and in the scheduler once, which is one
    # implementation too many -- two expressions of the same policy drift, and the one that drifts is
    # whichever has no tests. The scheduler has them; this is now only plumbing.
    # Each capsule is judged against the record earned by EXACTLY its current bytes when the ledger holds
    # one, so an edit elsewhere does not throw its certificate away; anything else keeps today's
    # comparison and is still put through `invalidated_by`.
    identity_by_name = {n: current_identity(execution_by_name.get(n), digest) for n in st}
    states = []
    for n, e in st.items():
        _id = identity_by_name[n]
        verdicts = {}
        for t in _recorded_tiers(e):
            v = _verdict_for(st, n, t, _id)
            if isinstance(v, dict):
                verdicts[t] = Verdict(v.get("status"), v.get("digest"),
                                      dict(v.get("components") or {}), v.get("execution_digest"))
        states.append(CapsuleState(name=n, digest=digest, verdicts=verdicts,
                                   components=dict(comps), depends_on=deps,
                                   execution_digest=execution_by_name.get(n)))
    want = [w for w in schedule(states, tier_order=[loop_tier, cert_tier], cert_tiers=(cert_tier,),
                                cert_cover=cover)
            if w.tier == cert_tier]

    # WHICH component requeued each capsule, so a reader of the log can see why a certificate was dropped
    # rather than only that the count went up. A run that requeues everything and one that requeues one
    # capsule are indistinguishable from the promotion count alone. A whole-submission cause additionally
    # NAMES the inputs that were missing -- the diagnosis of this defect stalled on 21 identical
    # `<whole-submission> (changed)` lines that named none of them.
    for s_ in states:
        _broad = _no_narrower_cause(execution_by_name.get(s_.name), comps)
        for tier in (loop_tier, cert_tier):
            # Only tiers that HAD a verdict: a tier nobody ever ran was not invalidated by anything, and
            # logging that would bury the real signal under one line per capsule per round.
            for why in (s_.invalidated_by(tier) if tier in s_.verdicts else ()):
                _extra = f"; no narrower cause: {_broad}" if (why.component == WHOLE_SUBMISSION
                                                              and _broad) else ""
                print(f"[promote] {s_.name} {tier} invalidated by {why}{_extra}", file=log, flush=True)

    promoted = []
    _sim = cert_sim(cert_tier)
    if want and _sim is None:
        # Say it once, loudly. Marking a capsule pending for a job that cannot be enqueued is how a
        # capsule strands at `pending` and never resolves.
        print(f"[promote] no --sim this broker accepts serves {cert_tier}; {len(want)} capsule(s) NOT "
              f"enqueued and NOT marked pending", file=log, flush=True)
    for w in want:
        if _sim is None:
            continue
        execution_digest = execution_by_name.get(w.capsule)
        token = execution_digest[:16] if execution_digest is not None else digest
        key = identity_by_name.get(w.capsule) or current_identity(execution_digest, digest)
        if key in _slots_ro(st, w.capsule, cert_tier):
            # These exact bytes already hold a cert-tier record -- outstanding, passed, or failed. Buying
            # a second copy of a verdict we already own is the 17-jobs-for-two-capsules waste itself.
            continue
        jid = f"promo{len(promoted)}_{token}_{w.capsule}"[:80]
        req = ch / f"simreq_{jid}.json"
        if req.exists():
            # The ledger says these bytes hold no record, yet a request for them is on the queue. That is
            # a state the ledger cannot explain (a hand-cleared state file, a request written by a broker
            # whose state write was lost). Say so instead of skipping silently: the silent version is how
            # a capsule stranded with a request nobody would ever answer.
            print(f"[promote] {w.capsule} {cert_tier}: {req.name} is already queued while the tier "
                  f"state holds no record for these bytes; NOT re-enqueued and NOT marked outstanding",
                  file=log, flush=True)
            continue
        req.write_text(_j.dumps(
            # `identity` travels WITH the request so the reap can hand a completed result back to the
            # exact record it was launched for. Without it, a result whose reader produced no artifact
            # identity is unattributable as soon as a second record is outstanding -- and the certificate
            # the RTL just paid for is dropped.
            {"sim": _sim, "capsules": w.capsule, "workers": 1, "tiers": cert_tier,
             "promoted": True, "identity": key, "submitted_at": time.time()}))
        # Mark pending only once the request is actually on the queue.
        pending = {"status": "pending", "digest": digest, "components": dict(comps)}
        if execution_digest is not None:
            pending["execution_digest"] = execution_digest
        _why_broad = _no_narrower_cause(execution_digest, comps)
        if _why_broad:
            pending["fallback_reason"] = _why_broad
        _record(st, w.capsule, cert_tier, pending)
        promoted.append(w.capsule)
    _save_tier_state(ws, st)
    if promoted:
        print(f"[promote] {loop_tier} pass -> {cert_tier}: {promoted}", file=log, flush=True)
    return promoted
