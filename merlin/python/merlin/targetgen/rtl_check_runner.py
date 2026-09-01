"""Run the RTL-derived FileCheck assertions against an agent's emitted artifacts (Pillar 1 runner).

Ties the deterministic RTL facts + the FileCheck compiler to a candidate capsule run:

  1. render the endpoint's emitted stream to a canonical text — a RoCC target's decoded trace (counts +
     ABI + per-instruction lines) or a self-hosted target's decoded kernel instruction stream,
  2. compile the FileCheck assertions for the capsule (:mod:`rtl_check_compiler`),
  3. invoke the **FileCheck LLVM binary** over that rendered decode of the target's ACTUAL emitted
     commands/instructions (never the agent's dialect MLIR — its op mnemonics are un-derivable per run),
  4. additionally run the Python :func:`rtl_checks.screen` for numeric bounds FileCheck can't express
     (scratchpad/accumulator capacity, multi-matmul tile lower bound),

and return a combined result whose ``verdict`` a caller may use to SKIP the expensive spike/verilator/VCS
oracle on a hard reject — turning a multi-minute failed RTL run into an instant FileCheck diagnostic.

Frozen runner/grader/contract are never touched; this runs *around* them.

CLI::

    python -m merlin.targetgen.rtl_check_runner <run_capsule_dir | runs_root> [--write] [--quantify]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from . import rtl_check_compiler as CC
from . import rtl_checks as RC
from .rtl.facts import load_facts
from .corpora import capsule_corpus_roots
from merlin.common.paths import ext_path, repo_root

_REPO = repo_root()
# RTL facts are the generated artifact (regenerated from the RTL on demand by load_facts); the
# resolver defaults to gemmini but honors $MERLIN_RTL_FACTS. General callers should pass target=.
_CAPSULE_ROOTS = capsule_corpus_roots()   # canonical + perf corpus, resolved by the corpus locator
_FILECHECK_CANDIDATES = [
    str(_REPO / "third_party/llvm-build/bin/FileCheck"),
    f"{ext_path("chipyard")}/.conda-env/riscv-tools/bin/FileCheck",
]
_COMPUTE_CLASSES = {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"}


def find_filecheck() -> str | None:
    for c in _FILECHECK_CANDIDATES:
        if Path(c).is_file():
            return c
    return shutil.which("FileCheck")


# Compiled checks are a pure function of (capsule name, RTL-facts sha) — memoize so a batch/beam over
# one capsule compiles the FileCheck assertions once, not once per candidate run.
_COMPILED_CACHE: dict[tuple[str, str], dict] = {}
_FACTS_SHA: dict[int, str] = {}


def _facts_sha(facts_rec: dict) -> str:
    """Cheap, stable fingerprint of an RTL-facts record (cached by object identity within a run)."""
    k = id(facts_rec)
    s = _FACTS_SHA.get(k)
    if s is None:
        s = hashlib.sha1(json.dumps(facts_rec, sort_keys=True, default=str).encode()).hexdigest()[:12]
        _FACTS_SHA[k] = s
    return s


def compiled_checks(facts_rec: dict, capsule: dict, target: str) -> dict:
    """Memoized :func:`rtl_check_compiler.compile_checks`, keyed by (capsule name, facts sha, target)."""
    key = (capsule.get("name") or "?", _facts_sha(facts_rec), target)
    c = _COMPILED_CACHE.get(key)
    if c is None:
        c = CC.compile_checks(facts_rec, capsule, target)
        _COMPILED_CACHE[key] = c
    return c


def _legal_funct(facts_rec: dict) -> set[int]:
    """The RTL-derived legal RoCC funct set, or empty when the facts carry no decode table. Fail-closed:
    an EMPTY set means "legality not derivable" (the caller renders ILLEGAL_FUNCT_COUNT as unknown and the
    compiler omits the ILLEGAL_FUNCT_COUNT assertion) — never a baked gemmini funct block substituted for a
    target whose decoder we could not read."""
    facts = facts_rec.get("facts", facts_rec)
    for i in (facts.get("interfaces") or []):
        if i.get("name") == "funct_decode_table":
            return set(i.get("legal_funct") or [])
    return set()


def render_trace(trace: dict, facts_rec: dict) -> str:
    """Canonical text the TRACE FileCheck lines are matched against."""
    instrs = trace.get("instructions", [])
    hist: dict[str, int] = {}
    for i in instrs:
        hist[i.get("class")] = hist.get(i.get("class"), 0) + 1
    n_mvin = hist.get("MVIN", 0)
    n_mvout = hist.get("MVOUT", 0)
    n_compute = sum(hist.get(c, 0) for c in _COMPUTE_CLASSES)
    legal = _legal_funct(facts_rec)
    # legality is only computable when the RTL facts carry the legal set; else render '-' (unknown) so a
    # missing decode table is NOT silently treated as "every funct illegal" (and the compiler omits the
    # matching ILLEGAL_FUNCT_COUNT assertion) — fail-closed, not a gemmini default.
    illegal_str = (str(sum(1 for i in instrs
                           if isinstance(i.get("funct"), int) and i["funct"] not in legal))
                   if legal else "-")
    abi = trace.get("abi") or {}
    custom = abi.get("custom_opcode", "-")
    funct3 = abi.get("funct3", "-")
    L = [f"# {CC.RENDER_SCHEMA}",
         f"ABI custom={custom} funct3={funct3}",
         f"MVIN_COUNT {n_mvin}",
         f"MVOUT_COUNT {n_mvout}",
         f"COMPUTE_COUNT {n_compute}",
         f"ILLEGAL_FUNCT_COUNT {illegal_str}",
         f"COMPUTE_PRESENT {'yes' if n_compute else 'no'}",
         f"MVIN_PRESENT {'yes' if n_mvin else 'no'}"]
    for i in instrs:
        f = i.get("funct")
        L.append(f"INSTR {i.get('index')} {i.get('class')} funct={f if f is not None else '-'}")
    return "\n".join(L) + "\n"


def _parse_words(kernel_text: str) -> list[int]:
    """Parse the ``.word``/``.insn`` instruction values out of an assembled kernel — STRUCTURED, no regex.
    Per line: drop ``#``/``//`` comments, tokenize on whitespace, and if the first token is a ``.word`` or
    ``.insn`` directive take its first integer operand (``0x…`` or decimal). Non-directive lines (labels,
    ``.text``/``.globl``, ``ret``) are skipped."""
    words: list[int] = []
    for raw in kernel_text.splitlines():
        line = raw.split("#", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue
        toks = line.replace(",", " ").split()
        if not toks or toks[0] not in (".word", ".insn"):
            continue
        for t in toks[1:]:
            try:
                words.append(int(t, 16) if t.lower().startswith("0x") else int(t))
                break
            except ValueError:
                continue
    return words


def _legal_opcodes(facts_rec: dict) -> tuple[set[int], int] | None:
    """(legal decode-value set, field width) DERIVED from the RTL/ISA decode facts, or None if the target
    ships none. The width is inferred from the largest legal value (the extractor's icmp-eq field), so the
    legality test compares the emitted instruction's low-``width`` bits — the field the hardware decoder
    actually matches. No target literals: the set + width both come from the discovered facts."""
    facts = facts_rec.get("facts", facts_rec)
    dt = next((i for i in (facts.get("interfaces") or []) if i.get("name") == "funct_decode_table"), None)
    vals = set((dt or {}).get("legal_funct") or [])
    if not vals:
        return None
    width = max(vals).bit_length()
    return vals, width


def render_kernel_decode(kernel_text: str, facts_rec: dict, taxonomy: dict | None = None) -> str:
    """Canonical text the KERNEL FileCheck lines are matched against — a decode of the emitted self-hosted
    kernel's `.word`/`.insn` instruction stream. Two layers, both fully DERIVED (no target literals):

    * LEGALITY — each word's low-``width`` decode field vs the RTL-discovered legal-opcode set
      (``ILLEGAL_OPCODE_COUNT`` = what the hardware decoder would reject).
    * CLASS DECODE — when a taxonomy is given, each word is classified into its SEMANTIC class using the
      per-op decode signatures (fixed_mask/fixed_value from the ISA def's own encoder). This exposes what a
      matmul kernel actually emitted (e.g. VADD instead of the MXU matmul), which legality alone misses —
      a ``CLASS_PRESENT <c>`` line per class actually emitted lets the checks assert the required classes.

    This is the static RTL/ISA-structural signal (no Verilog run, beyond spike/npu_model's functional
    output). Everything comes from ``facts_rec`` + the derived ``taxonomy``."""
    from . import isa_taxonomy as IT
    words = _parse_words(kernel_text)
    lo = _legal_opcodes(facts_rec)
    legal, width = (lo if lo else (set(), 0))
    mask = (1 << width) - 1 if width else 0
    n_illegal = 0
    lines = []
    present: list[str] = []
    counts: dict[str, int] = {}
    zeroops: dict[str, int] = {}                          # per-class count of all-zero-operand instructions
    for idx, w in enumerate(words):
        matches = IT.classify(w, taxonomy) if taxonomy else []
        classes = [c for c, _m in matches]
        # LEGALITY = "the decoder accepts this instruction". With the derived per-op decode signatures the
        # authoritative test is that the word matches SOME op's opcode/funct bits (classify non-empty) —
        # robust to operand values. Only when no taxonomy is available do we fall back to the coarse
        # low-width membership in the discovered legal-value set.
        if taxonomy:
            ok = bool(matches)
            field = w
        else:
            field = w & mask if mask else w
            ok = (field in legal) if legal else True
        if not ok:
            n_illegal += 1
        for c, fmask in matches:
            if c not in present:
                present.append(c)
            counts[c] = counts.get(c, 0) + 1
            if (w & (~fmask & 0xFFFFFFFF)) == 0:          # operand payload (bits outside the fixed opcode/funct)
                zeroops[c] = zeroops.get(c, 0) + 1
        cls_s = "|".join(classes) if classes else ("-" if taxonomy else "?")
        lines.append(f"INSTR {idx} word=0x{w:08x} opcode={field} legal={'yes' if ok else 'no'} class={cls_s}")
    # legality is determinable only with a taxonomy (per-op decode signatures) OR a discovered legal set;
    # with neither, render '-' (unknown) instead of 0 so a target we could not ground is NOT vacuously
    # passed — the compiler correspondingly omits the ILLEGAL_OPCODE_COUNT assertion (fail-closed).
    determinable = bool(taxonomy) or bool(legal)
    L = [f"# {CC.RENDER_SCHEMA}",
         f"EMPTY_KERNEL {'yes' if not words else 'no'}",
         f"INSTR_COUNT {len(words)}",
         f"LEGAL_OPCODE_SET_SIZE {len(legal)}",
         f"ILLEGAL_OPCODE_COUNT {n_illegal if determinable else '-'}"]
    L += [f"CLASS_PRESENT {c}" for c in present]
    L += [f"CLASS_COUNT {c} {counts[c]}" for c in present]              # for the mesh-tiling count check
    L += [f"CLASS_ZEROOPS {c} {zeroops.get(c, 0)}" for c in present]    # for the field-sanity (base≠0) check
    return "\n".join(L + lines) + "\n"


def run_filecheck(fc: str, check_text: str, input_text: str,
                  prefixes: str | list[str]) -> tuple[bool, str]:
    """Run FileCheck(check_text) over input_text with one or more --check-prefixes. (ok, diagnostics)."""
    prefs = prefixes if isinstance(prefixes, str) else ",".join(prefixes)
    with tempfile.NamedTemporaryFile("w", suffix=".checks", delete=False) as cf:
        cf.write(check_text)
        check_path = cf.name
    try:
        p = subprocess.run([fc, f"--check-prefixes={prefs}", "--allow-unused-prefixes", check_path],
                           input=input_text, capture_output=True, text=True)
        return (p.returncode == 0, (p.stderr or p.stdout).strip())
    finally:
        Path(check_path).unlink(missing_ok=True)


def _capsule_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for root in _CAPSULE_ROOTS:
        if root.is_dir():
            for cy in root.rglob("capsule.yaml"):
                idx.setdefault(cy.parent.name, cy)
    return idx


def _load_capsule(name: str, index: dict[str, Path]) -> dict | None:
    p = index.get(name)
    return yaml.safe_load(p.read_text()) if p else None


def _screen_object(run_capsule_dir: Path, facts_rec: dict, capsule: dict | None, fc: str | None,
                   target: str, *, write: bool) -> dict | None:
    """The EMITTED-OBJECT check family: screen the machine-code words the target's emit path recorded at
    ``generated/<rtl_object_screen.WORDS_ARTIFACT>``. Returns None when this run recorded none (the caller
    then reports the family as not applicable — the coverage census makes that visible), and never
    substitutes a clean result for a missing one."""
    from . import rtl_object_screen as OS
    from .isa_model import isa_model_for_target

    doc = OS.load_words(run_capsule_dir / "generated")
    if doc is None:
        return None
    words = [int(w) for w in doc.get("words") or []]
    try:
        model = isa_model_for_target(target)
    except Exception as e:  # noqa: BLE001 — no derived ISA model -> the family cannot run, honestly
        return {"capsule": (capsule or {}).get("name") or run_capsule_dir.name, "filecheck": {},
                "screen": {"verdict": "unknown", "checks": [],
                           "dropped": {"all": f"no derived ISA model for {target!r}: {e!r}"}},
                "verdict": "unknown"}
    rep = OS.screen(words, model, facts_rec, capsule, lint_enforced=bool(doc.get("lint_enforced")))
    res: dict[str, Any] = {"capsule": (capsule or {}).get("name") or run_capsule_dir.name,
                           "filecheck": {}, "screen": rep, "object_source": doc.get("source"),
                           "object_decode": OS.render(rep)}
    checks = OS.compile_object_checks(capsule or {}, rep)
    # Do NOT FileCheck a screen that grounded nothing: its render carries only '-' values, so the run would
    # come back ``ok`` beside a verdict of ``unknown`` and read as a pass. No assertion, no reassurance.
    if fc and checks and rep.get("grounded"):
        ok, diag = run_filecheck(fc, checks, res["object_decode"], ["OBJECT"])
        res["filecheck"]["object"] = {"ok": ok, "diag": diag}
    fc_fail = (res["filecheck"].get("object") or {}).get("ok") is False
    res["verdict"] = "reject" if (fc_fail or rep["verdict"] == "reject") else (
        "warn" if rep["verdict"] == "warn" else rep["verdict"])
    if write:
        (run_capsule_dir / "rtl_checks.json").write_text(json.dumps(res, indent=2))
    return res


def screen_run(run_capsule_dir: Path, facts_rec: dict, index: dict[str, Path],
               fc: str | None, write: bool = False, *, target: str) -> dict | None:
    """Run the full RTL-check suite (FileCheck trace/kernel + Python numeric screen) on one run dir.

    ``target`` selects the check family by DERIVED endpoint: a RoCC command-ISA target (endpoint
    ``inline_asm_insn``) gets the TRACE FileCheck over its decoded RoCC stream; a self-hosted-ISA target
    (``external_backend``) gets the KERNEL opcode-legality FileCheck over its emitted instruction stream.
    Both check the target's actual emitted commands; the Python numeric screen adds capacity bounds."""
    gen = run_capsule_dir / "generated"
    trace_p = gen / "instruction_trace.json"
    kernel_p = gen / "kernel.S"
    capsule = _load_capsule(_capsule_name_for(run_capsule_dir), index)
    compiled = compiled_checks(facts_rec, capsule or {}, target)

    # SELF-HOSTED-ISA (external_backend, e.g. atlas): no RoCC instruction_trace — the graded artifact is
    # the emitted kernel.S. Run the kernel opcode-LEGALITY FileCheck (every emitted opcode ∈ the RTL/ISA
    # legal set) over its rendered decode. This is the RTL-grounded, no-Verilog structural check for a
    # self-hosted target, fully derived from facts_rec. Verdict rides this check.
    if compiled.get("kernel") is not None:
        if not kernel_p.is_file():
            # The same endpoint, a DIFFERENT emitted artifact: a target whose codegen endpoint is an
            # LLVM-dialect MLIR module has no hand-authored kernel.S — its machine code only exists once a
            # real toolchain compiles the lowering. Screen the words that emit path recorded instead. This
            # branch is why the family census below reports a denominator: returning None here is what made
            # 18 consecutive rounds of the RTL-checks arm read as "nothing wrong" when nothing was looked at.
            obj = _screen_object(run_capsule_dir, facts_rec, capsule, fc, target, write=write)
            return obj
        res = {"capsule": (capsule or {}).get("name") or run_capsule_dir.name,
               "filecheck": {}, "screen": None}
        from . import isa_taxonomy as IT
        tax = IT.taxonomy_for_target(target)             # DERIVED at run time; {} if unavailable
        decode_txt = render_kernel_decode(kernel_p.read_text(), facts_rec, tax)
        if fc:
            # KERNEL = order-independent -DAG (legality, coverage, tiling, field-sanity); KERNELORDER =
            # the ordered first-occurrence class sequence. Disjoint vocabularies, one FileCheck pass.
            ok, diag = run_filecheck(fc, compiled["kernel"], decode_txt, ["KERNEL", "KORDER"])
            res["filecheck"]["kernel"] = {"ok": ok, "diag": diag}
            res["verdict"] = "reject" if ok is False else "ok"
        else:
            res["verdict"] = "ok"
        res["kernel_decode"] = decode_txt
        if write:
            (run_capsule_dir / "rtl_checks.json").write_text(json.dumps(res, indent=2))
        return res

    if not trace_p.is_file():
        return None
    trace = json.loads(trace_p.read_text())
    res: dict[str, Any] = {"capsule": (capsule or {}).get("name") or run_capsule_dir.name,
                           "filecheck": {}, "screen": None}

    if fc and compiled["trace"]:
        # The structural verdict rides the format-agnostic TRACE FileCheck over the DECODED RoCC stream —
        # the target's actual emitted commands. We do NOT FileCheck the agent's dialect MLIR: its op
        # mnemonics are invented per generated OOT dialect (no derivation source), and corroboration
        # against 383 real agent runs showed op-name patterns over lowered.target.mlir false-fail on
        # several legal MLIR surface forms while the decoded trace never did. The trace is canonical.
        trace_txt = render_trace(trace, facts_rec)
        ok, diag = run_filecheck(fc, compiled["trace"], trace_txt, "TRACE")
        res["filecheck"]["trace"] = {"ok": ok, "diag": diag}
    # Python numeric/lower-bound checks (capacity, multi-matmul tile bound) the RTL facts feed.
    rc_facts = CC._facts_to_rc(facts_rec)
    rep = RC.screen(trace, capsule, rc_facts, target=target)
    res["screen"] = rep.to_dict()

    # VERDICT rides the format-agnostic, RTL-grounded checks: the TRACE FileCheck (over the decoded RoCC
    # stream — the target's actual emitted commands) + the Python numeric screen. The decoded trace is
    # canonical and format-independent, so it never false-positives on a legal MLIR surface form.
    fc_fail = (res["filecheck"].get("trace") or {}).get("ok") is False
    res["verdict"] = "reject" if (fc_fail or rep.verdict == "reject") else (
        "warn" if rep.verdict == "warn" else "ok")
    if write:
        (run_capsule_dir / "rtl_checks.json").write_text(json.dumps(res, indent=2))
    return res


def _capsule_name_for(d: Path) -> str:
    r = d / "capsule_result.json"
    if r.is_file():
        try:
            return json.loads(r.read_text()).get("capsule") or d.name
        except Exception:
            pass
    return d.name


def _target_of_run(run_capsule_dir: Path) -> str:
    """DERIVE the target a capsule run belongs to from its own ``run_manifest.yaml`` (the runner stamps
    ``target`` there). No gemmini default: a run dir without a recorded target is a loud error, so a
    caller that omits ``target`` still screens against the run's ACTUAL target, never an assumed one."""
    mf = Path(run_capsule_dir) / "run_manifest.yaml"
    doc = yaml.safe_load(mf.read_text()) if mf.is_file() else None
    target = (doc or {}).get("target") if isinstance(doc, dict) else None
    if not target:
        raise ValueError(f"cannot derive target for {run_capsule_dir}: no 'target' in run_manifest.yaml; "
                         "pass target= explicitly")
    return str(target)


def prescreen(run_capsule_dir: Path, target: str | None = None) -> dict | None:
    """Opt-in cost gate: compile+run the RTL checks; caller may skip the oracle on verdict=='reject'.

    ``target`` selects the facts + check family. When omitted it is DERIVED from the run's own
    ``run_manifest.yaml`` (:func:`_target_of_run`) — never defaulted to gemmini — so a legacy caller that
    passes only the run dir still screens against that run's actual target."""
    target = target or _target_of_run(Path(run_capsule_dir))
    facts = load_facts(target)
    return screen_run(Path(run_capsule_dir), facts, _capsule_index(), find_filecheck(),
                      write=False, target=target)


def iter_run_dirs(root: Path):
    """Every capsule run dir under ``root`` that carries an artifact SOME check family consumes. Keyed on
    the union of the families' entry artifacts, not just the RoCC trace: keyed on the trace alone, the CLI
    silently found nothing for a target whose compiler emits a lowering — the same blind spot that made the
    advisory itself come out empty. ``screen_run`` still decides applicability per dir."""
    from . import rtl_object_screen as OS

    entry = ("instruction_trace.json", "kernel.S", OS.WORDS_ARTIFACT)
    if any((root / "generated" / a).is_file() for a in entry):
        yield root
        return
    seen: set[Path] = set()
    for a in entry:
        for t in sorted(root.rglob(f"generated/{a}")):
            d = t.parent.parent
            if d not in seen:
                seen.add(d)
                yield d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a capsule run dir or a runs/ tree")
    ap.add_argument("--target", required=True,
                    help="target whose RTL facts + check family to screen")
    ap.add_argument("--write", action="store_true", help="write rtl_checks.json beside capsule_result.json")
    ap.add_argument("--quantify", action="store_true",
                    help="summarize how many runs the pre-screen would reject (oracle skips)")
    a = ap.parse_args(argv)
    fc = find_filecheck()
    if not fc:
        print("WARNING: FileCheck binary not found; running Python screen only")
    facts = load_facts(a.target)
    index = _capsule_index()
    rejects = warns = oks = n = 0
    for d in sorted(iter_run_dirs(Path(a.root))):
        r = screen_run(d, facts, index, fc, write=a.write, target=a.target)
        if r is None:
            continue
        n += 1
        v = r["verdict"]
        rejects += v == "reject"; warns += v == "warn"; oks += v == "ok"
        if not a.quantify:
            fcs = " ".join(f"{k}={'ok' if vv['ok'] else 'FAIL'}" for k, vv in r["filecheck"].items())
            print(f"  {v:6s} {r['capsule']:34s} filecheck[{fcs}] screen={r['screen']['verdict']}")
    print(f"\n{n} runs: {rejects} reject, {warns} warn, {oks} ok "
          f"(reject => oracle run can be skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
