"""Single-source IRDL bridge for the `merlin_iface` contract dialect.

The C++ OOT backend and the Merlin/xDSL infra must agree on ONE dialect spec. Rather than hand-write
(and hand-sync) a C++ ODS dialect, this emits the canonical **IRDL** description of `merlin_iface` —
which `mlir-opt --irdl-file=...` registers DYNAMICALLY. The same IRDL is the portable spec the
xDSL side already mirrors (the contract grammar in `merlin/contract/interface_grammar.md`).

WHAT THIS BUYS, AND WHAT IT DOES NOT (measured 2026-09-04, LLVM-23, 509 tracked capsules):
  * A dynamically-registered dialect has NO custom parser, so mlir-opt reads only the GENERIC form
    (`"merlin_iface.matmul"(%a, %b) : (...) -> ...`). Capsules are written in the pretty form, so
    `--irdl-file` used to parse 0/370 of them -- failing with rc=1 and an EMPTY stderr, which is why
    that sat unnoticed. `verify()` now re-spells each capsule via
    `merlin.targetgen.contract.interface_emit.to_generic_form` first; the corpus is checked, not the
    fixtures. The pretty form remains the contract surface: nothing on the agent side changes.
  * The emitted constraints are only those the IRDL interpreter EVALUATES. `irdl.c_pred` is not one
    of them -- mlir-opt drops it from its enclosing `irdl.all_of` with no diagnostic, so it can
    never fail. Each c_pred is therefore resolved (see `_CPRED_LOWERING`) into IRDL's own vocabulary
    where the vocabulary reaches, and otherwise OUT of the file and into its generated header, which
    names what is not checked. Structural checking of those stays the grader's job.

Pipeline (stock LLVM-23 tools, no custom C++):
  ODS (.td, the contract's reference dialect)  --tblgen-to-irdl-->  raw IRDL
                                               --normalize-------->  merlin_iface.irdl.mlir
  C++ OOT tool:  mlir-opt --irdl-file=merlin_iface.irdl.mlir  <capsule, generic form>

Normalization — four upstream `tblgen-to-irdl` quirks, each of which silently breaks registration
or silently produces a constraint that cannot fail:
  1. `irdl.base "!builtin.string"` (from `StrAttr`) uses the TYPE sigil for an ATTRIBUTE base, so it
     does not resolve; the sigil is corrected to `#`, which keeps string-ness enforced.
  2. Type symbols are spelled WITH their `!` sigil (`irdl.type @"!acc"`), which names the type
     `!acc` -- the parser then looks for `!merlin_iface.!acc` and rejects every valid module.
     The sigil belongs to the printer, not the symbol, so it is stripped.
  3. `let parameters` is dropped entirely, so `!merlin_iface.acc<bf16>` (what the corpus writes)
     fails with "expected 0 type arguments, but had 1". The parameters are read back from the ODS
     via llvm-tblgen's JSON record dump and re-attached -- whatever the ODS declares, nothing
     about this dialect assumed.
  4. Shape / element-type / attribute constraints come out as `irdl.c_pred` C++ predicate STRINGS,
     which the interpreter cannot evaluate and silently ignores. See `_CPRED_LOWERING`.

Usage: python -m merlin.targetgen.rtl.gen_iface_irdl [--out <irdl.mlir>] [--verify]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from merlin.common.paths import artifacts_dir, repo_root

_REPO = repo_root()
_LLVM = _REPO / "third_party" / "llvm-install"
_T2I = _LLVM / "bin" / "tblgen-to-irdl"
_MLIROPT = _LLVM / "bin" / "mlir-opt"
# merlin_iface is the SHARED contract dialect (not target-specific) — its pinned spec lives next to
# interface_grammar.md in merlin/contract/, NOT under a per-target dir.
_DEFAULT_OUT = _REPO / "merlin/contract/merlin_iface.irdl.mlir"
_CAPSULES = _REPO / "merlin/contract" / "capsules"


# tblgen-to-irdl renders every `StrAttr` constraint as
#     %2 = irdl.base "!builtin.string"
# with the TYPE sigil `!`. StringAttr is an ATTRIBUTE, and the IRDL runtime resolves the two through
# different registries (`AbstractType::lookup` vs `AbstractAttribute::lookup`), so the sigil is not
# cosmetic: mlir-opt fails with "no registered type with name !builtin.string" and the whole dialect
# refuses to load. CORRECTING the sigil, rather than relaxing the constraint to `irdl.any` as this
# did before, keeps string-ness enforced -- `irdl.any` accepted `name = 42 : i64`.
_STR_BASE = 'irdl.base "!builtin.string"'
_STR_BASE_FIXED = 'irdl.base "#builtin.string"'


def _fix_string_base_sigil(raw: str) -> str:
    return raw.replace(_STR_BASE, _STR_BASE_FIXED)


# ---------------------------------------------------------------------------------------------
# c_pred resolution
# ---------------------------------------------------------------------------------------------
# `irdl.c_pred` carries a C++ predicate SOURCE STRING, and it is the one IRDL constraint op that
# does not implement `VerifyConstraintInterface`. `irdl::createVerifier` therefore never gives it a
# constraint slot, and `getConstraintIndicesForArgs` then drops it from the enclosing `irdl.all_of`
# WITH NO DIAGNOSTIC (LLVM: mlir/lib/Dialect/IRDL/IRDLLoading.cpp, mlir/lib/Dialect/IRDL/IR/
# IRDLOps.cpp). `irdl.all_of(%c_pred)` consequently loads as an EMPTY conjunction -- a constraint
# that cannot fail. Measured before this existed: a `commit` returning `i32` instead of a tensor and
# an `epilogue = 42 : i64` against a string-array constraint BOTH verified clean.
#
# A constraint incapable of failing is worse than an absent one, because it reads as enforcement. So
# every c_pred is resolved here: into IRDL's own vocabulary where the vocabulary reaches, and
# otherwise out of the file and into a stated exclusion in its header.
#
# The key is the EXACT predicate text tblgen-to-irdl emits for one ODS constraint. The value is
# either the IRDL-native replacement, or None plus the reason IRDL cannot express it. An
# unrecognised predicate matches nothing, stays in the file, and is reported as UNKNOWN by
# `lower_c_preds` -- an upstream respelling becomes loud rather than quietly inert.
_CPRED_LOWERING: dict[str, tuple[str | None, str]] = {
    # ODS `AnyRankedTensor`. `builtin.tensor` IS RankedTensorType exactly: an unranked tensor has a
    # different base name (`builtin.unranked_tensor`), so rankedness survives the translation.
    "(::llvm::isa<::mlir::RankedTensorType>($_self))":
        ('irdl.base "!builtin.tensor"', "ranked-tensor-ness"),
    # ODS `ArrayAttr` -- the outer half of `StrArrayAttr` and `I64ArrayAttr`.
    "(::llvm::isa<::mlir::ArrayAttr>($_self))":
        ('irdl.base "#builtin.array"', "array-attribute-ness"),
    # The inner half of `StrArrayAttr`.
    "::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) "
    "{ return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); })":
        (None, "each element of the array must be a string -- IRDL has no element-wise constraint "
               "over a builtin ArrayAttr"),
    # The inner half of `I64ArrayAttr`.
    "::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) "
    "{ return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && "
    "((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))); })":
        (None, "each element of the array must be a signless i64 -- IRDL has no element-wise "
               "constraint over a builtin ArrayAttr"),
    # The element-type half of `RankedTensorOf<[AnyType]>`.
    "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }"
    "(::llvm::cast<::mlir::ShapedType>($_self).getElementType())":
        (None, "the tensor element type must not be a token -- `irdl.parametric` reaches the "
               "parameters of IRDL-declared types only, not a builtin tensor's, and IRDL has no "
               "negation"),
}

_CPRED = " = irdl.c_pred "
_NARY = (" = irdl.all_of(", " = irdl.any_of(")


def _quoted(body: str) -> str:
    """The string literal on an `irdl.c_pred` line, without its quotes.

    Split on the first and last `"` rather than matched: the payload is C++ source and contains
    anything but a double quote (tblgen would have escaped one)."""
    rest = body.partition('"')[2]
    return rest[: rest.rfind('"')]


def _arg_names(body: str) -> list[str]:
    """The `%a, %b` argument list of an `irdl.all_of(...)` / `irdl.any_of(...)` line."""
    inner = body.partition("(")[2].rpartition(")")[0]
    return [a.strip() for a in inner.split(",") if a.strip()]


def _referenced(body: str) -> set[str]:
    """The `%ssa` names a line mentions. Structural split on `%`, no pattern matching."""
    names = set()
    for chunk in body.split("%")[1:]:
        head = chunk.split()[0] if chunk.split() else ""
        tok = head.strip("(),:<>")
        if tok:
            names.add("%" + tok)
    return names


def _rewrite_op_body(lines: list[str], op: str) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Resolve the c_preds of ONE `irdl.operation` body. See :func:`lower_c_preds`."""
    drop: dict[str, str] = {}
    notes: list[tuple[str, str]] = []
    unknown: list[str] = []

    # Pass 1 -- decide each c_pred's fate.
    for line in lines:
        body = line.strip()
        if body.startswith("%") and _CPRED in body:
            name, pred = body.partition(" = ")[0], _quoted(body)
            if pred not in _CPRED_LOWERING:
                unknown.append(f"{op}: {pred[:70]}")
            elif _CPRED_LOWERING[pred][0] is None:
                drop[name] = _CPRED_LOWERING[pred][1]

    # Pass 2 -- a dropped constraint named directly by an `irdl.operands`/`results`/`attributes`
    # SLOT cannot simply vanish; the slot would dangle and the file would not parse. Those stay,
    # spelled `irdl.any`, which is what an unenforceable constraint in fact is. Only slot lines are
    # scanned: an `%N = ...` definition mentions its own name, so scanning definitions would keep
    # every dropped constraint alive as a vacuous `irdl.any` -- the exact shape being removed.
    keep_as_any = {
        name for line in lines if line.strip().startswith("irdl.")
        for name in drop if name in _referenced(line.strip())
    }

    out: list[str] = []
    for line in lines:
        body = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if body.startswith("%") and _CPRED in body:
            name = body.partition(" = ")[0]
            entry = _CPRED_LOWERING.get(_quoted(body))
            if entry is None:                       # unrecognised: leave it, report it
                out.append(line)
            elif entry[0] is not None:              # expressible: emit the IRDL-native form
                out.append(f"{indent}{name} = {entry[0]}\n")
            else:
                notes.append((op, entry[1]))
                out.append(f"{indent}// NOT CHECKED -- {entry[1]}\n")
                if name in keep_as_any:
                    out.append(f"{indent}{name} = irdl.any\n")
            continue

        if body.startswith("%") and any(k in body for k in _NARY):
            name, _, rhs = body.partition(" = ")
            kind = rhs.partition("(")[0]
            kept = [a for a in _arg_names(body) if a not in drop]
            # An all_of of nothing is a conjunction of nothing: it accepts everything and says so
            # nowhere. `irdl.any` is the same acceptance, spelled honestly.
            rendered = f"{kind}({', '.join(kept)})" if kept else "irdl.any"
            out.append(f"{indent}{name} = {rendered}\n")
            continue

        out.append(line)
    return out, notes, unknown


def lower_c_preds(raw: str) -> tuple[str, list[tuple[str, str]], list[str]]:
    """Resolve every `irdl.c_pred` into IRDL's own vocabulary, or out of the file.

    Returns the rewritten IRDL, the constraints DROPPED because IRDL cannot express them (each an
    ``(op, reason)`` pair), and the predicates the table did not recognise -- those are left in place,
    so an unhandled upstream spelling shows up instead of loading as a constraint that never fires.
    """
    out: list[str] = []
    notes: list[tuple[str, str]] = []
    unknown: list[str] = []
    buf: list[str] | None = None
    op = ""
    for line in raw.splitlines(keepends=True):
        body = line.strip()
        if body.startswith("irdl.operation @"):
            op = body.split("@", 1)[1].split()[0].rstrip("{").strip()
            out.append(line)
            buf = []
            continue
        if buf is not None and body == "}":
            done, n, u = _rewrite_op_body(buf, op)
            out.extend(done)
            notes.extend(n)
            unknown.extend(u)
            out.append(line)
            buf = None
            continue
        (buf if buf is not None else out).append(line)
    return "".join(out), notes, unknown


def _header(notes: list[tuple[str, str]], unknown: list[str]) -> str:
    """The generated preamble that STATES what this file does not check.

    Without it the file's silence reads as completeness. Every constraint below the header is one
    the IRDL interpreter evaluates; the ODS constraints its vocabulary cannot reach are named here,
    grouped by reason with the ops they were dropped from, and remain the grader's job."""
    by_reason: dict[str, list[str]] = {}
    for op, reason in notes:
        by_reason.setdefault(reason, []).append(op)
    lines = [
        "// GENERATED by merlin.targetgen.rtl.gen_iface_irdl from the reference merlin_iface ODS.",
        "// Do not hand-edit; regenerate.",
        "//",
        "// Every constraint in this file is one the IRDL interpreter actually evaluates. ODS",
        "// constraints IRDL cannot express are NOT carried here as `irdl.c_pred`: mlir-opt drops a",
        "// c_pred from its enclosing `irdl.all_of` without a diagnostic, so such a constraint can",
        "// never fail, and one that cannot fail reads as enforcement while providing none. These",
        "// are what this file does NOT check; the grader does:",
    ]
    for reason, ops in by_reason.items():
        lines.append(f"//   * {reason}")
        lines.append(f"//     on: {', '.join(sorted(dict.fromkeys(ops)))}")
    if not by_reason:
        lines.append("//   (none)")
    if unknown:
        lines += ["//",
                  "// UNRECOGNISED predicates, left in place and therefore INERT -- teach",
                  "// _CPRED_LOWERING about them:"]
        lines += [f"//   ! {u}" for u in dict.fromkeys(unknown)]
    return "\n".join(lines) + "\n"


def _strip_type_sigil(raw: str) -> str:
    """Drop the `!` tblgen-to-irdl bakes into type SYMBOLS (`irdl.type @"!acc"`).

    The symbol is the bare mnemonic; the leading `!` is added by the printer when the type is
    spelled. Left in, the dialect declares a type literally named `!acc`, so a module writing the
    correct `!merlin_iface.acc` is rejected with "expected dynamic type" -- i.e. the IRDL rejects
    every valid module, including the ones it was generated from."""
    return raw.replace('@"!', '@"')


def _ods_type_params(td: Path, includes: list[Path]) -> dict[str, list[str]]:
    """Map each ODS TypeDef mnemonic to its declared parameter names.

    Read from llvm-tblgen's JSON record dump (a real parser over the same .td tblgen-to-irdl
    consumed) rather than assumed, so this reflects whatever the contract dialect declares.
    IRDL rejects uppercase parameter names, so ODS camelCase is folded down; the name is
    positional-only in IRDL, so the fold is cosmetic."""
    args = [str(_LLVM / "bin" / "llvm-tblgen"), "--dump-json", str(td)]
    for inc in includes:
        args += ["-I", str(inc)]
    rec = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    out: dict[str, list[str]] = {}
    for name in rec.get("!instanceof", {}).get("TypeDef", []):
        entry = rec[name]
        params = (entry.get("parameters") or {}).get("args", [])
        names = [pair[1].lower() for pair in params if isinstance(pair, list) and len(pair) == 2]
        out[entry["mnemonic"]] = names
    return out


def _restore_type_params(raw: str, params: dict[str, list[str]]) -> str:
    """Re-attach the `irdl.parameters` tblgen-to-irdl drops.

    Without them a parameterised type is declared arity-0, so the corpus's `!merlin_iface.acc<bf16>`
    fails with "expected 0 type arguments, but had 1". Each parameter is `irdl.any`: the ODS type
    is `mlir::Type`, and the IRDL interpreter cannot evaluate a narrower `c_pred` anyway."""
    lines = []
    for line in raw.splitlines(keepends=True):
        body = line.strip()
        if body.startswith("irdl.type @"):
            mnemonic = body.split('@"', 1)[1].split('"', 1)[0] if '@"' in body else body.split("@", 1)[1].strip()
            names = params.get(mnemonic) or []
            if names:
                indent = line[: len(line) - len(line.lstrip())]
                inner = indent + "  "
                decls = "".join(f"{inner}%param{i} = irdl.any\n" for i, _ in enumerate(names))
                sig = ", ".join(f"{n}: %param{i}" for i, n in enumerate(names))
                lines.append(f'{indent}irdl.type @"{mnemonic}" {{\n{decls}{inner}irdl.parameters({sig})\n{indent}}}\n')
                continue
        lines.append(line)
    return "".join(lines)


def _iface_spec_digest(iface_dir: Path) -> str:
    """Digest of the whole ``MerlinIface`` ODS include dir -- the bytes that define the dialect."""
    h = hashlib.sha256()
    for f in sorted(iface_dir.rglob("*.td")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _discover_ref_ods_inc() -> Path:
    """Locate the reference ``merlin_iface`` ODS include dir, FAILING CLOSED on disagreement.

    ``merlin_iface`` is the SHARED contract dialect, so every OOT package under the generated targets
    tree must ship the SAME spec -- that premise is what makes "any package serves as the reference"
    sound. It is therefore checked rather than assumed: packages are grouped by the digest of their
    ODS, and a divergence raises instead of resolving by sort order.

    This is not hypothetical. Taking the first sorted match silently selected a 5-op ``agent_spec_v0``
    over the 7-op ``agent_spec_v1`` the tracked IRDL was generated from, which would have dropped
    ``conv2d`` and ``movement`` from the contract with no diagnostic. Pass ``--ref-ods`` to choose
    deliberately."""
    base = artifacts_dir() / "targets"
    cands = sorted(base.rglob("MerlinIface/MerlinIfaceOps.td"))
    if not cands:
        raise FileNotFoundError(
            f"no MerlinIface/MerlinIfaceOps.td found under {base}; build an OOT package first "
            f"(merlin-targetgen) or pass --ref-ods <mlir_oot/include>.")
    by_digest: dict[str, list[Path]] = {}
    for td in cands:
        by_digest.setdefault(_iface_spec_digest(td.parent), []).append(td)
    if len(by_digest) > 1:
        detail = "\n".join(
            f"  {d[:12]}: {len(tds)} package(s), e.g. {tds[0].relative_to(base)}"
            for d, tds in sorted(by_digest.items()))
        raise RuntimeError(
            "the merlin_iface contract dialect DIVERGES across OOT packages, so no single one is "
            "the reference; regenerating from an arbitrary pick would silently change the contract."
            f"\n{detail}\nRe-run naming the intended spec: --ref-ods <mlir_oot/include>")
    return cands[0].parent.parent


def generate(out: Path, ref_ods_inc: Path | None = None) -> Path:
    """tblgen-to-irdl on the reference ODS + normalize -> canonical merlin_iface.irdl.mlir."""
    ref_ods_inc = ref_ods_inc or _discover_ref_ods_inc()
    _REF_ODS_INC = ref_ods_inc
    td = _REF_ODS_INC / "MerlinIface" / "MerlinIfaceOps.td"
    raw = subprocess.run(
        [str(_T2I), str(td), "-I", str(_REF_ODS_INC), "-I", str(_LLVM / "include"),
         "--gen-dialect-irdl-defs", "--dialect=merlin_iface"],
        capture_output=True, text=True, check=True).stdout
    norm, notes, unknown = lower_c_preds(_restore_type_params(
        _strip_type_sigil(_fix_string_base_sigil(raw)),
        _ods_type_params(td, [_REF_ODS_INC, _LLVM / "include"]),
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_header(notes, unknown) + norm)
    return out


def verify(irdl: Path) -> tuple[int, int, list[str], int]:
    """Parse every tracked ``merlin_iface`` capsule through the IRDL-registered dialect.

    Each capsule is re-spelled in GENERIC form first
    (:func:`merlin.targetgen.contract.interface_emit.to_generic_form`). A dialect registered from
    IRDL has no custom parser, so the corpus's pretty form is unreadable by this route -- measured
    0/370, failing with rc=1 and an EMPTY stderr, which is why the gap sat unnoticed. The spelling
    bridge is what makes this a real check of the contract against the real corpus rather than
    against hand-written fixtures.

    A residual failure is now a genuine finding with a diagnostic, and the two kinds are separated:
    an op the interface GRAMMAR defines but the reference ODS does not declare is a contract
    divergence, not a capsule defect.
    """
    from merlin.targetgen.contract.interface_emit import to_generic_form, InterfaceGrammarError

    # Scope the denominator: `capsule.interface.mlir` is also the filename for MODEL capsules, which
    # are linalg-on-tensors and contain no merlin_iface op at all. Counting those as failures blamed
    # the corpus for a dialect this contract does not describe (measured: 139 of 509).
    every = sorted(_CAPSULES.rglob("capsule.interface.mlir"))
    caps = [c for c in every if "merlin_iface." in c.read_text()]
    not_applicable = len(every) - len(caps)
    ok = 0
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        gen = Path(td) / "generic.mlir"
        for c in caps:
            try:
                gen.write_text(to_generic_form(c.read_text()))
            except InterfaceGrammarError as e:
                fails.append(f"{c.parent.name}: {e}"[:140])
                continue
            # No --verify-diagnostics: it would mask a parse failure. rc==0 AND ops printed back.
            r = subprocess.run([str(_MLIROPT), f"--irdl-file={irdl}", str(gen)],
                               capture_output=True, text=True)
            if r.returncode == 0 and "merlin_iface." in r.stdout:
                ok += 1
                continue
            msg = (r.stderr or "").strip().splitlines()
            why = msg[0].split(": ", 1)[-1][:100] if msg else "rc=1 with an EMPTY stderr"
            fails.append(f"{c.parent.name}: {why}")
    return ok, len(caps), fails, not_applicable


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    ap.add_argument("--ref-ods", default=None,
                    help="reference merlin_iface ODS include dir (mlir_oot/include); "
                         "auto-discovered from any OOT package under out/artifacts/targets if omitted")
    ap.add_argument("--verify", action="store_true", help="parse every capsule via the generated IRDL")
    a = ap.parse_args(argv)
    out = generate(Path(a.out), Path(a.ref_ods) if a.ref_ods else None)
    nops = out.read_text().count("irdl.operation")
    print(f"wrote {out} ({out.read_text().count(chr(10))} lines, {nops} ops)")
    if a.verify:
        ok, n, fails, n_a = verify(out)
        print(f"IRDL parse check: {ok}/{n} merlin_iface capsules parsed+verified via "
              f"mlir-opt --irdl-file ({n_a} non-merlin_iface capsules out of scope)")
        # An op the interface grammar defines but the reference ODS does not declare is a divergence
        # between the two halves of the contract, not a defect in the capsule that uses it.
        undeclared = sorted({f.split("'")[1] for f in fails if "unregistered operation" in f})
        if undeclared:
            print(f"  {sum(1 for f in fails if 'unregistered operation' in f)}/{n} use grammar op(s) "
                  f"the reference ODS does not declare: {', '.join(undeclared)}")
        for f in fails[:10]:
            print(f"  FAIL {f}")
        return 0 if ok == n and n > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
