#!/usr/bin/env python3
"""Lint gate: a HARDWARE FACT may only be derived from CIRCT or the target's own RTL repo.

The rule
========
A fact about a hardware target -- a geometry, a capacity, a width, an opcode, a clock, an address --
may be derived ONLY from:

  * **CIRCT** output: FIRRTL text, the HW dialect (``*.hw.mlir``), an ``arcilator``/``circt-arc`` model
    or its state manifest, a firtool-emitted hierarchy / memmap / device tree / ``mems.conf``. Arc is
    allowed for EVERY target: it is CIRCT compiling that target's RTL, so it carries RTL provenance.
  * **the target's own RTL repository**: its Chisel/Verilog sources, the headers and memory maps it
    ships, its elaborated ``generated-src`` tree.

It may NOT be derived from a separate simulator or a hand-written model. Specifically forbidden as a
fact SOURCE (each may CROSS-CHECK a fact that is already derived; none may BE its source):

  * **cyclotron** (``MERLIN_MUON_CONFIG``, ``MERLIN_MUON_CYCLOTRON``) -- a Rust cycle model;
  * **spike** (``MERLIN_SPIKE``) -- a functional ISS;
  * **gsim** (``MERLIN_MUON_GSIM_EMU``) -- a separate FIRRTL simulator binary;
  * **npu_model** (``MERLIN_EXT_NPU_MODEL``) -- whose own hardware pin says "Not RTL, but a
    verdict-bearing input";
  * **declared_config** -- a hand-authored machine declaration reached through a target's
    ``config_path()`` accessor. Added because the muon geometry leak SURVIVED a target-agnostic
    rewrite by hiding behind that indirection: ``perf/simt_occupancy`` names no target and no env var,
    it asks the backend for its ``config_path()`` and parses the toml -- which for the SIMT reference
    target is the cyclotron config. A person wrote those numbers; CIRCT did not emit them.

Two detection axes
==================
A gate that looks only at string literals cannot see the two cases that actually matter here, both of
which are real and both of which are in the ratchet:

  * ``dse/hardware_space.cost_model_from_npu(hw)`` takes the npu_model ``HardwareConfig`` as a
    PARAMETER -- the only textual trace of its source is the function's own name;
  * ``perf/simt_occupancy`` reaches the cyclotron config through a target-supplied accessor name.

So references are detected as (1) string literals naming a source (env var, ``ext_path`` key,
filename) and (2) IDENTIFIERS -- a def name, an imported module, a called name -- declaring it.

Telling an arc model from a non-RTL model
=========================================
This is the trap the gate exists to survive, and it cannot be done by looking for the word "model".
Two atlas paths both read as "the model":

  * ``MERLIN_EXT_ATLAS_VSIM = .../runs/circt-arc/atlas/vsim_oracle``  -- an ARC model. ALLOWED.
  * ``MERLIN_EXT_NPU_MODEL  = .../third_party/atlas-npu/npu-model``   -- a hand model. FORBIDDEN.

A second, quieter trap: "lives in the RTL repo" is NOT sufficient. The cyclotron config sits at
``$CHIPYARD/generators/radiance/cyclotron/config.toml`` -- inside the RTL checkout, vendored beside the
Chisel -- and it is still the Rust model's own configuration, not the hardware. What makes a file a
fact source is that the RTL PRODUCED it (elaboration output) or that it IS the RTL (Chisel/Verilog,
the headers the design ships), never merely that it is stored nearby.

So the classification is by IDENTITY, never by spelling. :data:`FORBIDDEN_SOURCES` enumerates the
specific artifacts that are not RTL -- by env-var name and by the executable/module they invoke -- and
:data:`ARC_MARKERS` names the path shapes that make something a CIRCT product. A reference that is
BOTH (an env var on the forbidden list whose resolved value sits under ``circt-arc/``) resolves to
ALLOWED, because what the artifact IS wins over what it is called. Anything not on the forbidden list
is not flagged: this gate reports a known-bad source, it does not try to prove a source good.

Fact vs verdict -- where the line is drawn, and why
===================================================
A grader or an oracle legitimately runs spike, cyclotron or a model to decide whether a PROGRAM
produced the right answer. That is a verdict about a program, not a claim about the hardware, and
flagging it would make this gate noise -- ``capsule_runner``'s spike/verilator/cyclotron tiers,
``muon_oracles``' GSIM comparison and ``program_oracle``'s npu_model golden are all correct and all
must stay silent.

The line is drawn on the **destination**, decided structurally from the AST, not on the caller's name:

  a. the module is a fact producer if it is on a fact-producing PATH (``**/rtl/**``,
     ``*introspect*``, ``capability_manifests``, ``capability_discovery``, ``address_space``,
     ``liveness/facts``, ``perf/profile``, ``perf/simt_occupancy``, ``*_facts.py``, ``dse/hardware_space``)
     -- these modules exist to emit facts, so any forbidden read in them lands in one; OR
  b. the forbidden read reaches a FACT SINK: a value assigned into a subscript/key whose name is a
     fact key (``facts``, ``fields``, ``capabilities``, ``geometry``, ``evidence``, ``provenance``,
     ``arch``...), a call to a fact-writing API (``write_facts*``, ``dump_facts``, ``new_measurement``
     ...), or a ``return``/assignment inside a function whose name says fact (``*_facts``,
     ``*_geometry``, ``*_capabilit*``, ``cost_model_from_*``...).

A module that only runs the binary and compares outputs hits neither and is silent. The cost of
getting this wrong is asymmetric, so (b) is deliberately narrow: a missed violation is one line in a
ratchet away from being caught, a false one teaches people to ignore the gate.

Three states
============
Reporting-only by default (exit 0) so it can be run anywhere; ``--fail-on-new`` fails on anything not
in the ratchet, ``--fail-on-any`` fails on every finding including ratcheted debt. Findings are one of

  * ``violation``  -- a fact producer reads a forbidden source, and it is not ratcheted;
  * ``ratcheted``  -- known, reasoned debt (``fact_provenance_ratchet.txt``);
  * ``cross_check`` -- the read is annotated ``# fact-source-ok: <why>`` on its line as a cross-check.

The ratchet is scoped per (path, source) so one module's accepted debt cannot excuse another's, and
every entry carries a reason. It may only SHRINK.

Run::

    python build_tools/scripts/check_fact_provenance.py                # report (exit 0)
    python build_tools/scripts/check_fact_provenance.py --json         # machine-readable
    python build_tools/scripts/check_fact_provenance.py --fail-on-new  # gate on un-ratcheted findings
    python build_tools/scripts/check_fact_provenance.py --staged       # only git-staged files
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RATCHET_FILE = Path(__file__).resolve().parent / "fact_provenance_ratchet.txt"
SCAN_ROOTS = ("merlin/python/merlin", "merlin/targets", "build_tools/scripts")
#: Path fragments that mark COPIED sandbox workspaces / build output, not source.
EXCLUDE_FRAGMENTS = ("/_qa_ws/", "/_data/", "/bundle_inputs/", "/workspace/", "/__pycache__/")
INLINE_MARKER = "# fact-source-ok:"


# --------------------------------------------------------------------------- what is a bad source
#: ``id -> (kind, tokens)``. ``tokens`` are the literal strings whose appearance in a module means it
#: REACHES that source: the env var it resolves, the ``ext_path`` key, the executable it spawns, the
#: module it imports. Matched against string literals and dotted names in the AST, never against
#: prose: a docstring that merely NAMES spike is not a use of spike.
FORBIDDEN_SOURCES: dict[str, dict] = {
    "cyclotron": {
        "what": "a Rust cycle model — timing/geometry it declares are the model's, not the RTL's",
        "tokens": ("MERLIN_MUON_CONFIG", "MERLIN_MUON_CYCLOTRON", "config_muon.toml"),
        "names": ("cyclotron", "from_cyclotron"),
    },
    "spike": {
        "what": "a functional ISS — what it reports is what it was configured to simulate",
        "tokens": ("MERLIN_SPIKE",),
        "names": ("from_spike", "spike_facts", "run_on_spike"),
    },
    "gsim": {
        "what": "a separate FIRRTL simulator binary — a verdict engine, not a fact source",
        "tokens": ("MERLIN_MUON_GSIM_EMU",),
        "names": ("from_gsim",),
    },
    "npu_model": {  # target-ok: this gate hunts a named non-RTL artifact; the name IS the subject
        "what": "a hand-written model; its own hardware pin says 'Not RTL, but a verdict-bearing input'",
        # both the env var and the ext_path SHORT KEY, since `ext_path('npu_model')` reaches the same
        # artifact without ever spelling the variable.
        "tokens": ("MERLIN_EXT_NPU_MODEL", "npu_model"),  # target-ok: the artifact being hunted
        # `cost_model_from_npu` names neither the env var nor the package — the ONLY textual trace of
        # its source is its own name, and the whole HardwareConfig arrives as a parameter. A gate that
        # looked only at literals could not see it at all.
        "names": ("npu_model", "from_npu", "npu_config"),  # target-ok: the artifact being hunted
    },
    "declared_config": {
        "what": ("a hand-authored machine declaration reached through a target's `config_path()` "
                 "accessor — a person wrote the numbers in it; CIRCT did not emit them"),
        # The indirection that made the muon geometry leak survive a target-agnostic rewrite:
        # `simt_occupancy` never spells a target or an env var, it asks the backend for its
        # `config_path()` and parses the toml. The accessor NAME is the only textual trace, so that is
        # what is matched — as a literal (the `getattr` key) and as a called/defined name.
        "tokens": ("config_path",),
        "names": ("config_path",),
    },
}

#: Path shapes that make an artifact a CIRCT product, and therefore an ALLOWED source no matter what
#: the variable holding it is called. This is what separates ``MERLIN_EXT_ATLAS_VSIM`` (which resolves
#: to ``.../runs/circt-arc/atlas/vsim_oracle`` — an arc model built by CIRCT from the RTL, allowed)
#: from ``MERLIN_EXT_NPU_MODEL`` (``.../atlas-npu/npu-model`` — a hand model, forbidden). Identity, not
#: spelling: both end in something that reads as "the model".
ARC_MARKERS = ("circt-arc", "circt_arc", "arcilator", "hw.mlir", "firtool", "generated-src", ".fir")


def _resolves_to_arc(literal: str) -> bool:
    """True when a string literal names a CIRCT/arc artifact, so a forbidden token inside it is a
    false alarm (e.g. a path like ``runs/circt-arc/<t>/outputs`` that happens to contain a name)."""
    low = literal.lower()
    return any(m in low for m in ARC_MARKERS)


# ------------------------------------------------------------------- what counts as a fact producer
#: Path fragments of modules whose PURPOSE is to emit hardware facts. A forbidden read anywhere in one
#: of these lands in a fact by construction, so the sink analysis is not required.
FACT_PRODUCER_FRAGMENTS = (
    "/targetgen/rtl/",
    "/targetgen/capability_manifests.py",
    "/targetgen/capability_discovery.py",
    "/targetgen/address_space.py",
    "/liveness/facts.py",
    "/perf/profile.py",
    "/perf/simt_occupancy.py",
    "/dse/hardware_space.py",
    "introspect",           # any */*introspect*.py — a target's own RTL extractor
)

#: Names that, used as a dict key / attribute / subscript, mean "this value is being filed as a fact".
FACT_SINK_KEYS = frozenset({
    "facts", "fact", "fields", "capabilities", "capability", "geometry", "geom", "provenance",
    "evidence", "hardware", "hw_facts", "rtl_facts", "manifest", "arch", "peak", "datapath",
    "memories", "mesh", "simt", "clock_hz", "dtypes", "encoding",
})

#: Callables that WRITE a fact artifact.
FACT_SINK_CALLS = frozenset({
    "write_facts", "write_facts_guarded", "dump_facts", "build_facts", "load_facts",
    "simt_facts", "target_fact_bundle", "spatial_fact_bundle", "new_measurement", "record",
})

#: Function-name shapes whose RETURN VALUE is a fact.
FACT_FN_SUFFIXES = ("_facts", "_geometry", "_fact_bundle", "_capabilities", "_capability",
                    "_manifest", "_from_npu", "_provenance")


def _is_fact_producer_path(rel: str) -> bool:
    return any(frag in f"/{rel}" for frag in FACT_PRODUCER_FRAGMENTS)


# ------------------------------------------------------------------------------------- the AST scan
def _dotted(node: ast.AST) -> str:
    """``a.b.c`` for an Attribute/Name chain, else ``""`` (structural — no regex, per the sibling gate)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


class _Scanner(ast.NodeVisitor):
    """Collect every reference to a forbidden source, with the enclosing function and sink context.

    Deliberately ignores docstrings (a module that documents the rule is not breaking it) and honours
    ``ARC_MARKERS`` inside the literal itself, so a path under ``runs/circt-arc/`` never trips."""

    def __init__(self) -> None:
        self.docstrings: set[int] = set()
        self.fn_stack: list[str] = []
        self.sink_stack: list[str] = []
        self.hits: list[dict] = []

    # -- docstring bookkeeping ---------------------------------------------------------------
    def _mark_docstring(self, node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            self.docstrings.add(id(body[0].value))

    def visit_Module(self, node: ast.Module) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def _visit_fn(self, node) -> None:
        self._mark_docstring(node)
        # Pushed BEFORE the name check so a def-name hit is attributed to the function it names, not
        # to its enclosing scope: `cost_model_from_npu` must reach the sink analysis as its own name,
        # which is the only thing that identifies it as a fact function.
        self.fn_stack.append(node.name)
        self._check_name(node, node.name, "function name")
        self.generic_visit(node)
        self.fn_stack.pop()

    visit_FunctionDef = _visit_fn
    visit_AsyncFunctionDef = _visit_fn

    # -- imports and called names ------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self._check_name(node, a.name, "import")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for part in [node.module or ""] + [a.name for a in node.names]:
            self._check_name(node, part, "import")
        self.generic_visit(node)

    # -- sink context ------------------------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self.sink_stack.append(self._target_name(t))
        self.generic_visit(node)
        for _ in node.targets:
            self.sink_stack.pop()

    def visit_Return(self, node: ast.Return) -> None:
        self.sink_stack.append("<return>")
        self.generic_visit(node)
        self.sink_stack.pop()

    @staticmethod
    def _target_name(t: ast.AST) -> str:
        if isinstance(t, ast.Name):
            return t.id
        if isinstance(t, ast.Attribute):
            return t.attr
        if isinstance(t, ast.Subscript):
            k = t.slice
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                return k.value
            return _dotted(t.value)
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        leaf = name.rpartition(".")[2] or name
        self._check_name(node, leaf, "call")
        self.sink_stack.append(leaf)
        self.generic_visit(node)
        self.sink_stack.pop()

    # -- the actual detection ----------------------------------------------------------------
    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self.docstrings:
            self._check(node, node.value)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        # A fact key spelled as a dict literal key is a sink for its own value.
        for k, v in zip(node.keys, node.values):
            key = k.value if isinstance(k, ast.Constant) and isinstance(k.value, str) else ""
            self.sink_stack.append(str(key))
            self.visit(v)
            self.sink_stack.pop()
            if k is not None:
                self.visit(k)

    def _record(self, node: ast.AST, src: str, text: str, how: str) -> None:
        self.hits.append({
            "lineno": node.lineno, "source": src, "literal": text[:80], "how": how,
            "fn": self.fn_stack[-1] if self.fn_stack else "<module>",
            "sinks": [s for s in self.sink_stack if s],
        })

    def _check(self, node: ast.AST, text: str) -> None:
        """A STRING LITERAL naming a forbidden source (an env var, an ext_path key, a filename)."""
        if _resolves_to_arc(text):
            return          # a CIRCT/arc artifact: allowed whatever it is named
        for src, spec in FORBIDDEN_SOURCES.items():
            if any(tok == text or (tok in text and len(tok) > 6) for tok in spec["tokens"]):
                self._record(node, src, text, "literal")
                return

    def _check_name(self, node: ast.AST, name: str, how: str) -> None:
        """An IDENTIFIER (a def, an import, a called name) declaring the source it reads.

        The second detection axis, and the one that catches what literals cannot: a function whose
        HardwareConfig arrives as a parameter leaves no literal behind, and a module reached through a
        target-supplied accessor spells neither the target nor the env var. The identifier is then the
        only textual trace of the source, and matching it is the difference between a gate that sees
        the indirect cases and one that only sees the ones already easy to spot."""
        if not name or _resolves_to_arc(name):
            return
        low = name.lower()
        for src, spec in FORBIDDEN_SOURCES.items():
            if any(tok in low for tok in spec.get("names", ())):
                self._record(node, src, name, how)
                return


def _reaches_fact_sink(hit: dict) -> str | None:
    """Why this reference lands in a FACT rather than a verdict, or None when it does not."""
    for s in hit["sinks"]:
        if s in FACT_SINK_KEYS:
            return f"assigned into fact key {s!r}"
        if s in FACT_SINK_CALLS:
            return f"passed to fact-writing call {s}()"
    fn = hit["fn"]
    if any(fn.endswith(suf) for suf in FACT_FN_SUFFIXES):
        return f"returned from fact function {fn}()"
    return None


def scan_file(path: Path, rel: str) -> list[dict]:
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    sc = _Scanner()
    sc.visit(tree)
    if not sc.hits:
        return []
    lines = src.splitlines()
    producer = _is_fact_producer_path(rel)
    out: list[dict] = []
    for hit in sc.hits:
        line = lines[hit["lineno"] - 1] if 0 < hit["lineno"] <= len(lines) else ""
        why = ("the module's purpose is producing hardware facts (fact-producing path)"
               if producer else _reaches_fact_sink(hit))
        if why is None:
            continue        # a verdict path: runs the thing, does not file the answer as a fact
        rec = dict(hit)
        rec.update({"path": rel, "why_fact": why,
                    "kind": "cross_check" if INLINE_MARKER in line else "violation"})
        if rec["kind"] == "cross_check":
            rec["annotation"] = line.split(INLINE_MARKER, 1)[1].strip()
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------- the ratchet
def load_ratchet(path: Path | None = None) -> dict[tuple[str, str], str]:
    """``{(path, source): reason}``. Scoped per (path, source) ON PURPOSE: a blanket per-file entry
    would let a module that has accepted its cyclotron debt silently start reading spike too."""
    out: dict[tuple[str, str], str] = {}
    ratchet = path or RATCHET_FILE
    if not ratchet.is_file():
        return out
    for line in ratchet.read_text(encoding="utf-8").splitlines():
        body = line.strip()
        if not body or body.startswith("#"):
            continue
        entry, _, reason = body.partition("#")
        path, _, source = entry.strip().partition("::")
        if path and source:
            out[(path, source.strip())] = reason.strip()
    return out


# ------------------------------------------------------------------------------------------ driver
def _iter_targets(staged: bool) -> list[str]:
    if staged:
        # FAIL CLOSED on an unreadable index. This gate's entire work list comes from `git`, so a `git`
        # that cannot run (bad GIT_DIR, no repo, no binary) yielded an EMPTY list and the gate printed
        # OK -- a green that could not have gone red. `check=True` turns that into an exception the
        # caller reports; see check_no_answer_keys.py, which fixed the same shape first.
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        rels = [ln for ln in out.splitlines() if ln.strip()]
    else:
        rels = []
        for root in SCAN_ROOTS:
            for p in sorted((ROOT / root).rglob("*.py")):
                rels.append(p.relative_to(ROOT).as_posix())
    keep = []
    for rel in rels:
        if not rel.endswith(".py"):
            continue
        if any(frag in f"/{rel}" for frag in EXCLUDE_FRAGMENTS):
            continue
        if any(rel.startswith(r + "/") or rel == r for r in SCAN_ROOTS):
            keep.append(rel)
    return keep


def findings(staged: bool = False, ratchet_path: Path | None = None) -> list[dict]:
    ratchet = load_ratchet(ratchet_path)
    out: list[dict] = []
    for rel in _iter_targets(staged):
        for rec in scan_file(ROOT / rel, rel):
            key = (rec["path"], rec["source"])
            if rec["kind"] == "violation" and key in ratchet:
                rec["kind"] = "ratcheted"
                rec["ratchet_reason"] = ratchet[key]
            out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable findings")
    ap.add_argument("--staged", action="store_true", help="only git-staged files")
    ap.add_argument("--fail-on-new", action="store_true",
                    help="exit 1 on any un-ratcheted violation")
    ap.add_argument("--fail-on-any", action="store_true",
                    help="exit 1 on any finding, ratcheted debt included")
    ap.add_argument("--ratchet", default=None, metavar="PATH",
                    help=f"the accepted-debt list (default: {RATCHET_FILE.name}). It may only SHRINK; "
                         "each entry is `<path>::<source>  # reason`.")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    ratchet_path = Path(a.ratchet) if a.ratchet else None
    try:
        found = findings(a.staged, ratchet_path)
    except (OSError, subprocess.CalledProcessError) as exc:
        # FAIL CLOSED: the work list comes from `git`, and a `git` that cannot run used to yield an
        # empty list and a printed "[  ok]" -- a green that could not have gone red. "We could not
        # look" is not "there is nothing to find" (see check_no_answer_keys.py, same fix).
        print(f"[FAIL] fact-provenance: could not list the files to examine ({exc}); NOTHING was "
              f"examined, which is not the same as clean.", file=sys.stderr)
        return 1
    viol = [f for f in found if f["kind"] == "violation"]
    ratch = [f for f in found if f["kind"] == "ratcheted"]
    xchk = [f for f in found if f["kind"] == "cross_check"]

    if a.json:
        print(json.dumps({"violations": viol, "ratcheted": ratch, "cross_checks": xchk,
                          "n_ratchet_entries": len(load_ratchet(ratchet_path))}, indent=2))
    else:
        if viol:
            print(f"[FAIL] fact-provenance: {len(viol)} fact producer(s) reading a forbidden source:")
            for f in viol:
                print(f"  - {f['path']}:{f['lineno']}: [{f['source']}] {f['literal']!r} in "
                      f"{f['fn']}() — {f['why_fact']}. "
                      f"{FORBIDDEN_SOURCES[f['source']]['what']}")
        else:
            print("[  ok] fact-provenance: no un-ratcheted fact producer reads a forbidden source.")
        if ratch:
            print(f"[DEBT] {len(ratch)} ratcheted reference(s) (this list may only SHRINK):")
            for f in ratch:
                print(f"  - {f['path']}:{f['lineno']}: [{f['source']}] — {f['ratchet_reason']}")
        if xchk:
            print(f"[NOTE] {len(xchk)} reference(s) declared as cross-checks:")
            for f in xchk:
                print(f"  - {f['path']}:{f['lineno']}: [{f['source']}] — {f['annotation']}")

    if a.fail_on_any and found:
        return 1
    if a.fail_on_new and viol:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
