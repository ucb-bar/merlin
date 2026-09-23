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
import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from merlin.common.facts_view import interface as _facts_interface

from . import rtl_check_compiler as CC
from . import rtl_checks as RC
from .corpora import graded_capsule_roots, perf_capsule_roots
from .rtl.facts import load_facts


def find_filecheck(candidates: Sequence[str | Path] = ()) -> str | None:
    """Discover at invocation from explicit machine candidates, then PATH."""
    for c in candidates:
        if Path(c).is_file():
            return str(c)
    return shutil.which("FileCheck")


def compiled_checks(facts_rec: dict, capsule: dict, target: str, *, checks=None) -> dict:
    """Compile current declarations through the current selected support owner.

    Capsule names and facts object identities are not content identities. Compilation
    is cheap; do not reuse assertions after capsule, facts or provider changes.
    """
    return CC.compile_checks(facts_rec, capsule, target, checks=checks)


def render_trace(trace: dict, facts_rec: dict, *, target: str, checks=None) -> str:
    """Render through the same selected protocol that compiles TRACE assertions."""
    checks = RC.selected_checks(target) if checks is None else checks
    result = checks.render_trace(trace, facts_rec)
    if not isinstance(result, str) or not result.strip():
        raise RC.RtlChecksUnavailable("selected RTL check provider returned malformed trace rendering")
    return result


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
    dt = _facts_interface(facts, "funct_decode_table")
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
    legal, width = lo if lo else (set(), 0)
    mask = (1 << width) - 1 if width else 0
    n_illegal = 0
    lines = []
    present: list[str] = []
    counts: dict[str, int] = {}
    zeroops: dict[str, int] = {}  # per-class count of all-zero-operand instructions
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
            if (w & (~fmask & 0xFFFFFFFF)) == 0:  # operand payload (bits outside the fixed opcode/funct)
                zeroops[c] = zeroops.get(c, 0) + 1
        cls_s = "|".join(classes) if classes else ("-" if taxonomy else "?")
        lines.append(f"INSTR {idx} word=0x{w:08x} opcode={field} legal={'yes' if ok else 'no'} class={cls_s}")
    # legality is determinable only with a taxonomy (per-op decode signatures) OR a discovered legal set;
    # with neither, render '-' (unknown) instead of 0 so a target we could not ground is NOT vacuously
    # passed — the compiler correspondingly omits the ILLEGAL_OPCODE_COUNT assertion (fail-closed).
    determinable = bool(taxonomy) or bool(legal)
    L = [
        f"# {CC.RENDER_SCHEMA}",
        f"EMPTY_KERNEL {'yes' if not words else 'no'}",
        f"INSTR_COUNT {len(words)}",
        f"LEGAL_OPCODE_SET_SIZE {len(legal)}",
        f"ILLEGAL_OPCODE_COUNT {n_illegal if determinable else '-'}",
    ]
    L += [f"CLASS_PRESENT {c}" for c in present]
    L += [f"CLASS_COUNT {c} {counts[c]}" for c in present]  # for the mesh-tiling count check
    L += [f"CLASS_ZEROOPS {c} {zeroops.get(c, 0)}" for c in present]  # for the field-sanity (base≠0) check
    return "\n".join(L + lines) + "\n"


def run_filecheck(fc: str, check_text: str, input_text: str, prefixes: str | list[str]) -> tuple[bool, str]:
    """Run FileCheck(check_text) over input_text with one or more --check-prefixes. (ok, diagnostics)."""
    prefs = prefixes if isinstance(prefixes, str) else ",".join(prefixes)
    with tempfile.NamedTemporaryFile("w", suffix=".checks", delete=False) as cf:
        cf.write(check_text)
        check_path = cf.name
    try:
        p = subprocess.run(
            [fc, f"--check-prefixes={prefs}", "--allow-unused-prefixes", check_path],
            input=input_text,
            capture_output=True,
            text=True,
        )
        return (p.returncode == 0, (p.stderr or p.stdout).strip())
    finally:
        Path(check_path).unlink(missing_ok=True)


def capsule_index(roots: Sequence[Path]) -> dict[str, Path]:
    """Index only supplied corpus roots, preserving first-name-wins discovery."""
    idx: dict[str, Path] = {}
    for root in roots:
        if root.is_dir():
            for cy in root.rglob("capsule.yaml"):
                idx.setdefault(cy.parent.name, cy)
    return idx


def _load_capsule(name: str, index: dict[str, Path]) -> dict | None:
    p = index.get(name)
    return yaml.safe_load(p.read_text()) if p else None


def screen_run(
    run_capsule_dir: Path, facts_rec: dict, index: dict[str, Path], fc: str | None, write: bool = False, *, target: str
) -> dict | None:
    """Run the full RTL-check suite (FileCheck trace/kernel + Python numeric screen) on one run dir.

    ``target`` selects the check family by DERIVED endpoint: a RoCC command-ISA target (endpoint
    ``inline_asm_insn``) gets the TRACE FileCheck over its decoded RoCC stream; a self-hosted-ISA target
    (``external_backend``) gets the KERNEL opcode-legality FileCheck over its emitted instruction stream.
    Both check the target's actual emitted commands; the Python numeric screen adds capacity bounds."""
    gen = run_capsule_dir / "generated"
    trace_p = gen / "instruction_trace.json"
    kernel_p = gen / "kernel.S"
    capsule = _load_capsule(_capsule_name_for(run_capsule_dir), index)
    checks = RC.selected_checks(target) if CC._is_rocc_target(target, facts_rec) else None
    compiled = compiled_checks(facts_rec, capsule or {}, target, checks=checks)

    # SELF-HOSTED-ISA (external_backend, e.g. atlas): no RoCC instruction_trace — the graded artifact is
    # the emitted kernel.S. Run the kernel opcode-LEGALITY FileCheck (every emitted opcode ∈ the RTL/ISA
    # legal set) over its rendered decode. This is the RTL-grounded, no-Verilog structural check for a
    # self-hosted target, fully derived from facts_rec. Verdict rides this check.
    if compiled.get("kernel") is not None:
        if not kernel_p.is_file():
            return None
        res = {"capsule": (capsule or {}).get("name") or run_capsule_dir.name, "filecheck": {}, "screen": None}
        from . import isa_taxonomy as IT

        tax = IT.taxonomy_for_target(target)  # DERIVED at run time; {} if unavailable
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
    res: dict[str, Any] = {
        "capsule": (capsule or {}).get("name") or run_capsule_dir.name,
        "filecheck": {},
        "screen": None,
    }

    if fc and compiled["trace"]:
        # The structural verdict rides the format-agnostic TRACE FileCheck over the DECODED RoCC stream —
        # the target's actual emitted commands. We do NOT FileCheck the agent's dialect MLIR: its op
        # mnemonics are invented per generated OOT dialect (no derivation source), and corroboration
        # against 383 real agent runs showed op-name patterns over lowered.target.mlir false-fail on
        # several legal MLIR surface forms while the decoded trace never did. The trace is canonical.
        trace_txt = render_trace(trace, facts_rec, target=target, checks=checks)
        ok, diag = run_filecheck(fc, compiled["trace"], trace_txt, "TRACE")
        res["filecheck"]["trace"] = {"ok": ok, "diag": diag}
    # Python numeric/lower-bound checks (capacity, multi-matmul tile bound) the RTL facts feed.
    checks = RC.selected_checks(target) if checks is None else checks
    rc_facts = checks.project_facts(facts_rec)
    if not isinstance(rc_facts, dict):
        raise RC.RtlChecksUnavailable("selected RTL check provider returned malformed fact projection")
    # The package's OWN emitted command buffer: the declaration that binds each kernel argument to a
    # declared tensor, which the encoded-field-intent check needs. Absent -> that check reports skipped
    # with that reason (never a pass); a malformed one is treated the same way.
    cb_p = gen / "command_buffer.json"
    command_buffer = None
    if cb_p.is_file():
        try:
            cb_loaded = json.loads(cb_p.read_text())
            command_buffer = cb_loaded if isinstance(cb_loaded, dict) else None
        except (ValueError, OSError):
            command_buffer = None
    rep = RC.screen(trace, capsule, rc_facts, target=target, command_buffer=command_buffer, checks=checks)
    res["screen"] = rep.to_dict()

    # VERDICT rides the format-agnostic, RTL-grounded checks: the TRACE FileCheck (over the decoded RoCC
    # stream — the target's actual emitted commands) + the Python numeric screen. The decoded trace is
    # canonical and format-independent, so it never false-positives on a legal MLIR surface form.
    fc_fail = (res["filecheck"].get("trace") or {}).get("ok") is False
    res["verdict"] = "reject" if (fc_fail or rep.verdict == "reject") else ("warn" if rep.verdict == "warn" else "ok")
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
        raise ValueError(
            f"cannot derive target for {run_capsule_dir}: no 'target' in run_manifest.yaml; pass target= explicitly"
        )
    return str(target)


def prescreen(
    run_capsule_dir: Path,
    target: str | None = None,
    *,
    capsule_roots: Sequence[Path],
    filecheck_candidates: Sequence[str | Path] = (),
) -> dict | None:
    """Opt-in cost gate: compile+run the RTL checks; caller may skip the oracle on verdict=='reject'.

    ``target`` selects the facts + check family. When omitted it is DERIVED from the run's own
    ``run_manifest.yaml`` (:func:`_target_of_run`), never a default target.
    ``capsule_roots`` is always explicit: callers supply the actual corpus being screened,
    including frozen roots when screening an admitted experiment."""
    target = target or _target_of_run(Path(run_capsule_dir))
    facts = load_facts(target)
    return screen_run(
        Path(run_capsule_dir),
        facts,
        capsule_index(capsule_roots),
        find_filecheck(filecheck_candidates),
        write=False,
        target=target,
    )


def iter_run_dirs(root: Path):
    if (root / "generated" / "instruction_trace.json").is_file():
        yield root
        return
    for t in root.rglob("generated/instruction_trace.json"):
        yield t.parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a capsule run dir or a runs/ tree")
    ap.add_argument("--target", required=True, help="target whose RTL facts + check family to screen")
    ap.add_argument("--write", action="store_true", help="write rtl_checks.json beside capsule_result.json")
    ap.add_argument("--filecheck", type=Path, help="explicit FileCheck binary (otherwise search PATH)")
    ap.add_argument(
        "--quantify", action="store_true", help="summarize how many runs the pre-screen would reject (oracle skips)"
    )
    a = ap.parse_args(argv)
    if a.filecheck is not None and not a.filecheck.is_file():
        ap.error("--filecheck must name an existing file")
    fc = find_filecheck((a.filecheck,) if a.filecheck is not None else ())
    if not fc:
        print("WARNING: FileCheck binary not found; running Python screen only")
    facts = load_facts(a.target)
    roots = [*graded_capsule_roots(a.target), *perf_capsule_roots(a.target)]
    if not roots:
        ap.error("target has no descriptor-selected capsule roots; select a current target descriptor")
    index = capsule_index(roots)
    rejects = warns = oks = n = 0
    for d in sorted(iter_run_dirs(Path(a.root))):
        r = screen_run(d, facts, index, fc, write=a.write, target=a.target)
        if r is None:
            continue
        n += 1
        v = r["verdict"]
        rejects += v == "reject"
        warns += v == "warn"
        oks += v == "ok"
        if not a.quantify:
            fcs = " ".join(f"{k}={'ok' if vv['ok'] else 'FAIL'}" for k, vv in r["filecheck"].items())
            print(f"  {v:6s} {r['capsule']:34s} filecheck[{fcs}] screen={r['screen']['verdict']}")
    print(f"\n{n} runs: {rejects} reject, {warns} warn, {oks} ok (reject => oracle run can be skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
