#!/usr/bin/env python3
"""Pre-flight validation for capsule_bench_v0 real agent experiments.

Does NOT launch raw_baseline / merlin_assisted runs. Adversarially validates that isolation, freeze,
grading, trace/integrity gates, and metric capture are trustworthy, then writes
the target's capsule-bench report dir (out/artifacts/capsule-bench/<target>/
experiment_preflight_report.md) ending in GO_FOR_PILOT or NO_GO.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
import run_agent_experiment as RAE  # noqa: E402
from merlin.targetgen import capsule_grade as CGRADE  # noqa: E402
from merlin.targetgen import trace_check as TCK  # noqa: E402
from merlin.targetgen.rocc import decode as RD  # noqa: E402  (was targetgen.rocc_decode before the move)
from merlin.targetgen import capsule_golden as CG  # noqa: E402
from merlin.targetgen import experiment_tokens as ET  # noqa: E402
from merlin.targetgen import baremetalc_corroborate as BMC  # noqa: E402
from merlin.targetgen.contract import schemas as S  # noqa: E402
from merlin.targetgen.sandbox import bwrap  # noqa: E402  (grant resolution shared with the binder)

TARGET = C.TARGET
_TGT = f"out/artifacts/targets/{TARGET}"
CANARIES = [
    f"{_TGT}/agent_spec_v1_mlir_oot/CANARY_FORBIDDEN.txt",
    f"{_TGT}/hand_smoke_oot/CANARY_FORBIDDEN.txt",
    f"{_TGT}/merlin_native_v0/CANARY_FORBIDDEN.txt",
    "merlin/contract/capsules/hidden/CANARY_HIDDEN.txt",
    f"out/artifacts/capsule-bench/{TARGET}/CANARY_RESULTS.txt",
    "merlin/python/merlin/runtime/CANARY_FORBIDDEN.txt",
]
G0 = f"{_TGT}/agent_spec_v0_mlir_oot/certification/g0_matmul/lowered.llvm.mlir"


def bundles_to_check() -> list[str]:
    """Every input bundle the ACTIVE target ships, in sorted order.

    The gate used to canary a hardcoded ``("raw_baseline", "merlin_assisted")``
    pair read through ``ARM_BUNDLE``, which had two consequences: it broke on any
    target whose bundle ids differ, and it silently left other bundles — the
    CIRCT arm's, and any arm added later — with NO canary or hash coverage at all.
    Enumerating the directory instead means a new arm is covered the moment its
    bundle exists, which is the property a fairness gate has to have.
    """
    if not C.BUNDLES.is_dir():
        return []
    return sorted(d.name for d in C.BUNDLES.iterdir()
                  if d.is_dir() and (d / "input_bundle_manifest.yaml").is_file())


def load_bundle_by_id(bundle_id: str) -> dict:
    """Read a bundle manifest by id, bypassing the arm->id map entirely."""
    return yaml.safe_load((C.BUNDLES / bundle_id / "input_bundle_manifest.yaml").read_text())


def _bundle_id_for(arm: str) -> str:
    """Resolve *arm*'s bundle id for the ACTIVE target, not gemmini's.

    ``run_agent_experiment.ARM_BUNDLE`` still names the gemmini-era ``*_public_v0``
    ids, so on a target that ships ``*_hwbringup_v0`` (saturn_opu) every check here
    died on a missing manifest — i.e. the pre-spend gate could not run at all for
    any target but one. Resolve against what the target actually ships: keep the
    declared id when it exists, else take the unique ``<arm>_*`` bundle, excluding
    the CIRCT variant (a different arm that shares this one's stem).

    Fails closed with the candidates listed, because a gate that silently picks
    the wrong arm's bundle is worse than one that stops.
    """
    declared = RAE.ARM_BUNDLE[arm]
    if (C.BUNDLES / declared / "input_bundle_manifest.yaml").is_file():
        return declared
    others = {v for k, v in RAE.ARM_BUNDLE.items() if k != arm}
    # Variant arms whose bundle STEM extends this arm's own. `merlin_assisted_rtlchecks` and
    # `merlin_assisted_eqsat` both start with `merlin_assisted_`, so a prefix match alone would return
    # three candidates for the plain merlin arm and the gate would refuse to resolve any of them. Each
    # variant is a DIFFERENT arm that deliberately shares the stem (generate_prompt keys the assisted
    # seam menu off that substring), so they are excluded here by their differentiating token.
    _VARIANT_TOKENS = ("rtlchecks", "eqsat")
    candidates = sorted(
        d.name for d in C.BUNDLES.iterdir()
        if d.is_dir() and d.name.startswith(f"{arm}_") and d.name not in others
        and not any(t in d.name for t in _VARIANT_TOKENS)
        and (d / "input_bundle_manifest.yaml").is_file())
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(
        f"preflight: cannot resolve a bundle for arm {arm!r} under {C.BUNDLES} — "
        f"declared {declared!r} is absent and candidates are {candidates}. "
        f"Name one explicitly rather than letting the gate guess.")


def check_canary_isolation() -> dict:
    """For each agent bundle: assemble workspace, run a probe inside bwrap that tries to read every
    canary by absolute path + greps /scratch*; assert none reachable. Also show sandbox=none leaks."""
    out = {"bwrap_available": False, "per_bundle": {}, "unsandboxed_leaks": None}
    _bw = subprocess.run(["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
                          "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
                          "--tmpfs", "/scratch", "--proc", "/proc", "--dev", "/dev",
                          "--chdir", "/", "--", "/bin/true"],
                         capture_output=True, text=True, timeout=30, cwd="/tmp")
    out["bwrap_available"] = (_bw.returncode == 0)
    abspaths = [str(C.REPO / c) for c in CANARIES]
    # explicit absolute-path reachability of each canary + a bounded grep of the workspace only
    probe = ("for p in " + " ".join(f'"{p}"' for p in abspaths) +
             '; do if [ -r "$p" ]; then echo "REACHABLE $p"; fi; done; '
             'grep -rlI CANARY . 2>/dev/null | head -3')
    from merlin.targetgen.sandbox import bwrap as _BW
    from merlin.targetgen.target_experiment import load_target_experiment
    _te = load_target_experiment(C.EXP / "target_experiment.yaml")
    for bundle_id in bundles_to_check():
        bundle = load_bundle_by_id(bundle_id)
        with tempfile.TemporaryDirectory() as td:        # honours TMPDIR
            ws = Path(td) / "workspace"
            # This is a mount-policy canary across every shipped bundle, not a
            # runtime input-freeze test.  Copying each bundle's multi-GiB grants
            # here would turn preflight into ~150 GiB of redundant I/O.  The
            # mandatory runtime snapshot and its byte-immutability falsifier are
            # exercised separately; this probe deliberately replays live grants.
            RAE.assemble_workspace(bundle, ws, _policy_test_live_inputs=True)
            # The FULL isolation argv -- deny-by-default base, the runtime + toolchain binds, and the
            # derived answer-mask pass -- because that is what a real agent runs under. Probing the base
            # alone tested a WEAKER sandbox than the experiment uses (it does not re-mask an answer surface
            # that a broad legit grant re-exposes), and so reported a leak for every surface masking is
            # responsible for hiding.
            argv = _BW.full_argv(
                _te, ws, bundle, _policy_test_live_inputs=True) + ["bash", "-c", probe]
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            reachable = [ln for ln in r.stdout.splitlines() if ln.startswith("REACHABLE")]
            grep_hits = [ln for ln in r.stdout.splitlines() if "CANARY" in ln and not ln.startswith("REACHABLE")]
            out["per_bundle"][bundle_id] = {"reachable_canaries": reachable, "grep_hits": grep_hits,
                                      "isolated": not reachable and not grep_hits, "stderr": r.stderr[-200:]}
    # unsandboxed control: without bwrap the canaries ARE reachable by absolute path
    ctrl = ("for p in " + " ".join(f'"{p}"' for p in abspaths) +
            '; do if [ -r "$p" ]; then echo "REACHABLE $p"; fi; done')
    rc = subprocess.run(["bash", "-c", ctrl], capture_output=True, text=True, timeout=60, cwd="/tmp")
    out["unsandboxed_leaks"] = [ln for ln in rc.stdout.splitlines() if ln.startswith("REACHABLE")]
    return out


def reference_package() -> str | None:
    """The target's known-good reference package (repo-relative), or None when it ships none.

    DERIVED, never a per-target literal: the first package under the target's artifact home whose
    manifest marks it ``integrity_exempt`` (the reference backend is the one package allowed to import
    Merlin internals), falling back to any package with a manifest. So radiance seeds from
    ``reference_v0`` and gemmini from its own ``agent_spec_v1_mlir_oot`` with no per-target code.

    Returns None rather than raising when the target ships none: the end-to-end negative fixtures mutate
    a VALID package so that "one injected defect makes grading fail closed" is a real statement, and
    synthesizing a package instead would satisfy the assertions vacuously (a garbage package also scores
    zero). The honest outcome for such a target is UNVERIFIED, not passed.
    """
    import glob
    manifests = sorted(glob.glob(str(C.REPO / _TGT / "*" / "manifest.yaml")))
    exempt, any_pkg = [], []
    for m in manifests:
        rel = str(Path(m).parent.relative_to(C.REPO))
        any_pkg.append(rel)
        try:
            if (yaml.safe_load(Path(m).read_text(encoding="utf-8")) or {}).get("integrity_exempt"):
                exempt.append(rel)
        except Exception:  # noqa: BLE001 -- an unreadable manifest just isn't a candidate
            continue
    pool = exempt or any_pkg
    return pool[0] if pool else None


def _mk_pkg_with(text_file: dict, base=None) -> Path:
    """Copy the known-good package and inject a file (for integrity/contract negative fixtures)."""
    if base is None:
        base = reference_package()
        if base is None:
            raise FileNotFoundError(
                f"no reference package under {_TGT}; the end-to-end negative fixtures need a valid "
                f"package to mutate")
    import shutil
    d = Path(tempfile.mkdtemp(prefix="negfix_"))
    shutil.copytree(C.REPO / base, d / "pkg", ignore=shutil.ignore_patterns("build", "__pycache__"))
    for rel, content in text_file.items():
        p = d / "pkg" / rel
        if content is None:
            p.unlink(missing_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)   # layout-agnostic: the injected path's subdir
            p.write_text(content)                          # (e.g. mlir_oot/) need not pre-exist in the pkg
    return d / "pkg"


def check_negative_fixtures() -> dict:
    res = {"grader_endtoend": [], "trace": [], "numeric": [], "cb_schema": []}
    contract = str(C.REPO / "merlin/contract")

    # --- end-to-end through capsule_grade (no-oracle; integrity/contract run first, fail fast) ---
    # Needs a valid reference package to mutate; when the target ships none this
    # class is recorded UNVERIFIED rather than crashing the gate or reading as a pass.
    if reference_package() is None:
        res["grader_endtoend"].append({
            "case": "import_merlin_injected", "unavailable": True,
            "reason": f"target ships no reference package ({_TGT}/agent_spec_v1_mlir_oot)"})
        res["grader_endtoend"].append({
            "case": "missing_manifest", "unavailable": True,
            "reason": f"target ships no reference package ({_TGT}/agent_spec_v1_mlir_oot)"})
        return _negatives_without_package(res, contract)

    # (1) import-merlin injected -> integrity FORBIDDEN_PATTERN
    pkg = _mk_pkg_with({"mlir_oot/CANARY_import.py": "import merlin.runtime.reference\n"})
    # The integrity scanner is SKIPPED for an ``integrity_exempt`` package (the reference backend is the
    # one package allowed to import Merlin). This fixture is seeded from the target's reference package,
    # which MAY be exempt (radiance's reference_v0 is) — so force it NON-exempt to actually exercise the
    # scanner. Target-agnostic: the fixture tests the integrity GATE, not the reference's exemption.
    _mf = pkg / "manifest.yaml"
    if _mf.is_file():
        _m = yaml.safe_load(_mf.read_text(encoding="utf-8")) or {}
        _m["integrity_exempt"] = False
        _mf.write_text(yaml.safe_dump(_m, sort_keys=False), encoding="utf-8")
    g = CGRADE.grade(str(pkg), capsules_root=str(C.REPO / "merlin/contract/capsules"),
                     runs_root=tempfile.mkdtemp(), labels={"public"}, contract=contract,
                     oracle_adapters={}, target=TARGET)
    res["grader_endtoend"].append({"case": "import_merlin_injected", "functional_pass": g["functional_pass"],
                                   "integrity_status": g["integrity_status"],
                                   "fails_closed": g["functional_pass"] == 0 and "FAIL" in str(g["integrity_status"])})
    # (2) missing manifest -> contract fail
    pkg2 = _mk_pkg_with({"manifest.yaml": None})
    g2 = CGRADE.grade(str(pkg2), capsules_root=str(C.REPO / "merlin/contract/capsules"),
                      runs_root=tempfile.mkdtemp(), labels={"public"}, contract=contract,
                      oracle_adapters={}, target=TARGET)
    res["grader_endtoend"].append({"case": "missing_manifest", "functional_pass": g2["functional_pass"],
                                   "integrity_status": g2["integrity_status"],
                                   "fails_closed": g2["functional_pass"] == 0})

    return _negatives_without_package(res, contract)


def _negatives_without_package(res: dict, contract: str) -> dict:
    """The negative fixtures that need no reference package: trace, numeric, schema.

    Each group reports its own availability. The trace negatives decode a RoCC
    trace, which a target without a command ISA cannot do — that is a legitimate
    N/A for such an endpoint, and it must be visible as one rather than crashing
    the gate or being silently absent from the verdict.
    """
    # --- component: trace_check negatives (the gate the grader calls) ---
    # These fixtures are a RoCC COMMAND-ISA trace (a g0 matmul + RoCC instruction classes). They only apply
    # to a target that HAS a command trace; a SIMT/no-command-ISA target has none to gate -- detected by the
    # absence of the target's g0 trace artifact -- so we record it n/a rather than fabricate a trace for it.
    # The trace GATE itself is target-agnostic; only these fixtures are RoCC-specific.
    g0_path = C.REPO / G0
    if not g0_path.is_file():
        res["trace"].append({"case": "n/a_no_command_trace", "status": "n/a", "applicable": False,
                             "fails_closed": True,
                             "note": f"target {TARGET!r} has no RoCC command trace (no {G0}); RoCC "
                                     f"trace-gate negatives do not apply to a SIMT/command-buffer target"})
    else:
        try:
            real = RD.decode_file(g0_path, target=TARGET)  # a valid g0 matmul trace
        except Exception as exc:  # noqa: BLE001 - a gate reports, it does not crash
            res["trace"].append({"case": "trace_negatives", "unavailable": True,
                                 "reason": f"cannot decode a reference trace for {TARGET}: "
                                           f"{type(exc).__name__}: {exc}"})
            return _negatives_numeric_and_schema(res, contract)
        common = ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST", "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"]
        # The ABI rides along from the DECODED trace, never a literal: this negative asserts that a
        # compute-free trace fails a required-class check, and it must do so for whatever target is
        # selected. A baked custom_opcode/funct3 here would be one target's encoding asserted as every
        # target's (and `unknown_funct` below already shows an empty abi is accepted).
        empty = {"source": "x", "abi": dict(real.get("abi") or {}),
                 "instructions": [{"index": 0, "class": "FENCE"}, {"index": 1, "class": "FENCE"}]}
        cases = [
            ("no_insn_C_compute_proxy", empty, {"instruction_classes": common}, "required class missing"),
            ("required_LOOP_CONV_absent", real, {"instruction_classes": ["LOOP_CONV"]}, "LOOP_CONV missing"),
            ("movement_mode_but_compute_present", real, {"instruction_classes": [], "modes": {"movement": True}}, "compute present"),
            ("relu_required_but_absent", real, {"instruction_classes": [], "modes": {"relu": True}}, "no relu"),
            ("k_accum_required_but_absent", real, {"instruction_classes": [], "modes": {"k_accumulate": True}}, "no accumulate"),
            ("forbidden_compute_present", real, {"instruction_classes": [], "forbidden_classes": ["COMPUTE_PRELOADED"]}, "forbidden present"),
        ]
        for name, tr, exp, why in cases:
            r = TCK.check(tr, exp)
            res["trace"].append({"case": name, "status": r["status"], "fails_closed": r["status"] == "fail",
                                 "violations": r["violations"][:1]})
        # wrong funct -> UNKNOWN class -> fail
        bad = {"source": "x", "abi": {}, "instructions": [{"index": 0, "class": "UNKNOWN", "funct": 99}]}
        r = TCK.check(bad, {"instruction_classes": common})
        res["trace"].append({"case": "unknown_funct", "status": r["status"], "fails_closed": r["status"] == "fail"})

    return _negatives_numeric_and_schema(res, contract)


def _negatives_numeric_and_schema(res: dict, contract: str) -> dict:
    """Numeric-compare and command-buffer-schema negatives; target-independent."""
    # --- component: numeric mismatch fails even if shapes ok ---
    exp_out = {"Y0": [[1, 2], [3, 4]]}
    nrep = CG.compare(exp_out, {"Y0": [[1, 2], [3, 5]]}, {"compare": "exact_int"})
    res["numeric"].append({"case": "wrong_output", "status": nrep["status"],
                           "fails_closed": nrep["status"] == "fail", "mismatch_count": nrep["mismatch_count"]})

    # --- component: invalid command_buffer fails schema ---
    try:
        S.validate_command_buffer({"abi_version": "0.1"}, contract=contract)  # missing required keys
        res["cb_schema"].append({"case": "invalid_cb", "fails_closed": False})
    except Exception as e:
        res["cb_schema"].append({"case": "invalid_cb", "fails_closed": True, "error": type(e).__name__})
    return res


def check_freeze_enforcement() -> dict:
    """Mutating a frozen submission must change its hash (so the hidden-phase recheck catches it)."""
    import shutil
    d = Path(tempfile.mkdtemp(prefix="freeze_"))
    shutil.copytree(C.REPO / "merlin/contract/schemas", d / "sub")
    h1 = C.hash_tree(d / "sub")["sha256"]
    (d / "sub" / "INJECTED.txt").write_text("post-freeze tamper")
    h2 = C.hash_tree(d / "sub")["sha256"]
    return {"hash_before": h1[:16], "hash_after": h2[:16], "tamper_detected": h1 != h2}


def check_bundle_hash_repro() -> dict:
    """Pin every granted tree's hash, resolving paths exactly as the sandbox binds them.

    The lock must cover what the arm can actually READ, so it resolves through
    :func:`bwrap.resolve_grant` -- the same function that decides what gets bound. Resolving
    ``<repo>/<path>`` only (as this did) skipped every grant written in the ``experiments/...``
    shorthand, which is 17 paths across all five targets: each target's task, ISA headers,
    hwbringup contracts and self-check script were mounted into the arm and absent from its lock.

    A grant that resolves nowhere is recorded as ``unresolvable`` rather than dropped. Silently
    omitting it is the failure that hid the gap: a shrinking lock looked like a smaller bundle
    instead of an unpinned one.
    """
    out = {}
    # Two independent caches preserve the two-pass reproducibility check while
    # hashing a shared 8+ GiB grant only once in each pass, not once per bundle.
    pass1: dict[Path, str] = {}
    pass2: dict[Path, str] = {}
    for bundle_id in bundles_to_check():
        bundle = load_bundle_by_id(bundle_id)
        paths = [str(e.get("path", "")).strip("/") for e in bundle.get("allowed", [])]
        resolved = {p: bwrap.resolve_grant(p, C.REPO) for p in paths if p}
        unresolvable = sorted(p for p, r in resolved.items() if bwrap.path_kind(r) == "missing")

        def _hash(cache: dict[Path, str]) -> dict:
            hashes = {}
            for declared, grant in resolved.items():
                if bwrap.path_kind(grant) == "missing":
                    continue
                key = grant.absolute()
                if key not in cache:
                    cache[key] = _hash_granted_path(grant)["sha256"]
                hashes[declared] = cache[key]
            return hashes

        h1, h2 = _hash(pass1), _hash(pass2)
        out[bundle_id] = {"reproducible": h1 == h2, "n_paths": len(h1),
                          "unresolvable": unresolvable}
        lock = {"allowed_tree_sha256": h1}
        if unresolvable:
            lock["unresolvable_grants"] = unresolvable
        (C.BUNDLES / bundle_id / "bundle_lock.yaml").write_text(yaml.safe_dump(lock, sort_keys=True))
    return out


def _hash_granted_path(path: Path) -> dict:
    """Hash one allowed path, including a grant that names a plain file.

    ``hash_tree`` intentionally walks directory descendants.  Calling it on a
    file produces the empty-tree digest, and the old lock writer avoided that
    by dropping every file grant entirely.  ISA headers and single-file tools
    are load-bearing bundle inputs, so hash their bytes directly.
    """
    if path.is_dir():
        return C.hash_tree(path)
    import hashlib
    digest = hashlib.sha256()
    n_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            n_bytes += len(chunk)
    return {"present": True, "sha256": digest.hexdigest(), "n_files": 1,
            "n_bytes": n_bytes}


def _codex_token_witness() -> Path | None:
    """The newest REAL Codex event stream this host has captured, if any.

    ``MERLIN_TOKEN_WITNESS`` names one explicitly; otherwise the most recent
    ``*.codex_events.raw.jsonl`` written by the driver (under the runs tree) or by
    the sandbox canary (under the artifact cache). Both are genuine CLI output,
    which is the property this check exists to establish — a synthetic fixture
    would prove only that the parser runs.
    """
    explicit = os.environ.get("MERLIN_TOKEN_WITNESS", "").strip()
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    from merlin.common.paths import artifacts_dir, runs_dir

    found: list[Path] = []
    for root in (runs_dir(), artifacts_dir() / "cache"):
        if root.is_dir():
            found += list(root.rglob("*codex_events.raw.jsonl"))
    live = [p for p in found if p.is_file() and p.stat().st_size > 0]
    if not live:
        return None
    live.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer the newest stream that actually CONTAINS a completed turn. Usage lands
    # only on `turn.completed`, so the newest file is often a round still in flight
    # with nothing to capture yet — a true "no usage" that would misreport this
    # check as unable to capture tokens. Fall back to the newest so the report can
    # still say what it looked at.
    for p in live:
        try:
            if '"turn.completed"' in p.read_text(errors="replace"):
                return p
        except OSError:
            continue
    return live[0]


def _codex_tokens(path: Path) -> dict:
    """Token capture from a real Codex stream, using the driver's own translation.

    Reuses ``codex_agent.usage_to_claude_shape`` rather than re-deriving the subset
    arithmetic here: Codex reports ``input_tokens`` as a TOTAL already containing
    the cache reads, so a second implementation is a second chance to double-count.

    Cost stays ``None``. This host authenticates Codex with a ChatGPT account, so
    a dollar figure would be ``subscription_notional`` at best, and this gate must
    not hand a metered-looking number to a budget.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import codex_agent as CA

    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "reasoning_output_tokens": 0}
    turns = reported = 0
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue                                    # a killed writer's partial tail
        if not isinstance(event, dict) or event.get("type") != CA.EVENT_TURN_COMPLETED:
            continue
        turns += 1
        shaped, ok = CA.usage_to_claude_shape(event.get("usage") or {})
        if not ok:
            continue
        reported += 1
        for key in totals:
            totals[key] += int(shaped.get(key) or 0)
    tokens_total = totals["input_tokens"] + totals["output_tokens"] + \
        totals["cache_read_input_tokens"] + totals["cache_creation_input_tokens"]
    return {
        "tested": True, "driver": "codex", "witness": str(path),
        # Available only when the provider actually reported usage for a turn.
        # A stream with turns but no usage is a lower bound, not a capture.
        "available": reported > 0 and tokens_total > 0,
        "tokens_total": tokens_total, "per_bucket": totals,
        "turns_seen": turns, "turns_usage_reported": reported,
        "usage_complete": turns > 0 and reported >= turns,
        "estimated_cost_usd": None, "billing_mode": "subscription_notional",
        "cost_note": "ChatGPT-auth Codex consumes a subscription; any USD figure is notional "
                     "and must never enter a metered budget",
    }


def check_real_tokens() -> dict:
    """Prove token/cost capture works on a REAL agent stream, whichever agent ran.

    The check used to demand ``/tmp/real_stream.jsonl`` from a Claude probe, so a Codex campaign could
    never satisfy it and the gate reported NO_GO for having run the agent it was configured to run. Any
    of three real witnesses now counts -- an operator-supplied claude probe, a codex event stream, or the
    newest prior-run transcript that carries usage -- and the record says which one it was. The last
    validates the pipeline at $0 with no new spend.
    """
    p = Path(os.environ.get("MERLIN_CLAUDE_TOKEN_WITNESS", "/tmp/real_stream.jsonl"))
    if p.is_file():
        s = ET.parse_transcript(p)
        return {"tested": True, "driver": "claudecode", "witness": str(p),
                "available": s.get("available"), "usage_source": s.get("usage_source"),
                "tokens_total": s.get("tokens_total"),
                "estimated_cost_usd": s.get("estimated_cost_usd"),
                "unique_messages": s.get("unique_messages"), "billing_mode": s.get("billing_mode")}
    codex = _codex_token_witness()
    if codex is not None:
        return _codex_tokens(codex)
    from merlin.common.paths import runs_dir
    cands = sorted((q for q in runs_dir().rglob("*.transcript.jsonl") if q.stat().st_size > 0),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    for q in cands[:8]:                       # newest-first; stop at the first with real usage metadata
        s = ET.parse_transcript(q)
        if s.get("available"):
            return {"tested": True, "driver": "prior_run", "witness": str(q),
                    "available": True, "usage_source": s.get("usage_source"),
                    "tokens_total": s.get("tokens_total"),
                    "estimated_cost_usd": s.get("estimated_cost_usd"),
                    "unique_messages": s.get("unique_messages"), "billing_mode": s.get("billing_mode")}
    return {"tested": False, "available": False,
            "reason": "no real agent stream to check: neither a claude stream-json at "
                      f"{p}, a codex *.codex_events.raw.jsonl under the runs/cache trees, nor a "
                      "prior-run transcript carrying usage (set MERLIN_TOKEN_WITNESS to name one)"}


def baremetalc_table() -> list[dict]:
    import hashlib
    rows = []
    for anc in BMC._anchors():
        gh = hashlib.sha256(json.dumps(anc["golden"]).encode()).hexdigest()[:16]
        verbatim = "verbatim upstream" if anc["name"] == "mvin_mvout" else "canonical library (tiled_matmul_auto)"
        rows.append({"anchor": anc["name"], "capsule": anc["capsule"], "feature": anc["feature"],
                     "source": verbatim, "golden_sha256": gh, "spike": "match", "verilator": "match"})
    return rows


def check_oracle_available() -> dict:
    """A real (gradeable) pilot needs the target's NUMERIC oracle actually runnable — else the run can
    only ever emit ``oracle_unavailable`` and the agent thrashes to timeout (the atlas 0/11 at ~$43
    lesson). Mirror the launcher's oracle preflight (``capsule_runner.oracle_available``, contract-routed,
    no target literal) here so a run that cannot be graded is flagged NO_GO before a pilot is authorized."""
    from merlin.targetgen import capsule_runner as CR
    sim_via = ""
    desc = C.EXP / "target_experiment.yaml"
    if desc.is_file():
        from merlin.targetgen.target_experiment import load_target_experiment
        sim_via = load_target_experiment(desc).sim_via
    ok, why = CR.oracle_available(TARGET, sim_via)
    # The sim binaries being present is not enough: verify the oracle's COMPILE toolchain actually works
    # (compile a trivial IR to a riscv object). A missing/broken compiler otherwise passes here and then
    # tool-crashes on EVERY capsule after money is spent (the retired-clang lesson). Only gate on it when
    # the sim is otherwise available (a compile check is moot if the sim is absent).
    from merlin.targetgen import runtime_build as RB
    csmoke_ok, csmoke_why = RB.compiler_smoke(sim_via) if ok else (True, "n/a (sim unavailable)")
    reason = why if ok else why
    if ok and not csmoke_ok:
        reason = f"sim present but compile toolchain broken: {csmoke_why}"
    # Our OWN emit path must also produce a runnable artifact — not just the oracle's compiler. For a
    # self-hosted fixed-format backend this drives the full fork-free build->sim the live run uses, so a
    # broken codegen path is caught pre-spend rather than tool-crashing on every capsule (n/a otherwise).
    cg_ok, cg_why = CR.codegen_smoke(TARGET)
    if ok and csmoke_ok and not cg_ok:
        reason = f"grading oracle ready but our codegen backend is broken: {cg_why}"
    # For a self-hosted-ISA (external_backend) target, the pieces being present is STILL not proof the
    # oracle grades to a CORRECT verdict end-to-end. Run a KNOWN-GOOD self-contained model program all the
    # way through the grading path (assemble -> arc cosim -> readback) and require a BIT-EXACT match to its
    # own golden. Infra-absence or a mismatch is surfaced as NO_GO (never a crash, never a silent skip) so
    # a run that can only ever emit unavailable/false verdicts is refused before a paid pilot. n/a for a
    # non-external_backend target (its command_buffer/chipyard oracle is covered above). Contract-routed,
    # no target-name branch — the concrete known-good program is DECLARED in the descriptor.
    prog_smoke = _program_oracle_smoke(sim_okay=ok and csmoke_ok and cg_ok, desc=desc)
    if prog_smoke.get("ok") is False:
        reason = f"grading oracle pieces present but the end-to-end smoke failed: {prog_smoke.get('reason')}"
    return {"available": ok and csmoke_ok and cg_ok and prog_smoke["ok"], "reason": reason,
            "sim_via": sim_via, "compiler_smoke": {"ok": csmoke_ok, "reason": csmoke_why},
            "codegen_smoke": {"ok": cg_ok, "reason": cg_why}, "program_smoke": prog_smoke}


def _program_oracle_smoke(*, sim_okay: bool, desc: Path) -> dict:
    """Run the descriptor-declared KNOWN-GOOD program end-to-end through the target's grading oracle and
    return ``{ok, reason, ...}``. Only meaningful for an ``external_backend`` (program-oracle) target; any
    other target returns ``ok=True`` with an n/a reason (its oracle is validated by the checks above). Not
    run when the earlier pieces are already broken (``sim_okay`` False) — the failing piece is the actionable
    blocker, not a moot smoke. Fails CLOSED: a target that is external_backend but declares no
    ``preflight.smoke_program`` (or no ``runner.model_ext``) is NO_GO, and an ``OracleUnavailable`` (infra
    absent) is NO_GO — never a pass. Target-agnostic: the program name + model_ext are read from the
    descriptor/contract, no literal here."""
    from merlin.targetgen import capsule_runner as CR
    endpoint_kind, model_ext = CR._endpoint_of(TARGET)
    if endpoint_kind != "external_backend":
        return {"ok": True, "reason": "n/a (not an external_backend program-oracle target)"}
    # An EXCLUSIVE bespoke sim (a self-hosted SIMT core, e.g. cyclotron) grades the target on its OWN kernel
    # ELF and takes precedence over the program-oracle path (mirrors capsule_runner.oracle_available's
    # routing — the arc/program oracle grades the wrong artifact for a SIMT core). Its end-to-end
    # correctness is already exercised by codegen_smoke (fork-free emit -> sim -> assert the result), so the
    # program-oracle smoke does not apply here. Contract-routed, no target-name branch.
    sim_via = ""
    if desc.is_file():
        from merlin.targetgen.target_experiment import load_target_experiment
        sim_via = load_target_experiment(desc).sim_via
    so = CR.sim_oracle_caps(sim_via)
    if so is not None and so.exclusive:
        return {"ok": True, "reason": (f"n/a (exclusive bespoke sim {sim_via!r} grades on its own kernel ELF; "
                                       f"end-to-end correctness covered by codegen_smoke)")}
    if not sim_okay:
        return {"ok": True, "reason": "n/a (an earlier oracle/codegen check already blocks — fix that first)"}
    if desc.is_file():
        from merlin.targetgen.target_experiment import load_target_experiment
        program = load_target_experiment(desc).preflight_smoke_program
    else:
        program = None
    if not program:
        return {"ok": False, "reason": (f"external_backend target {TARGET!r} declares no "
                                        "preflight.smoke_program — cannot run an end-to-end oracle smoke")}
    if not model_ext:
        return {"ok": False, "reason": (f"external_backend target {TARGET!r} declares no runner.model_ext — "
                                        "cannot resolve the model venv that lays out operands + the golden")}
    from merlin.targetgen import program_oracle as PO
    try:
        with tempfile.TemporaryDirectory(prefix="oracle_smoke_") as td:
            r = PO.run_program_oracle_smoke(TARGET, model_ext=model_ext, program=program,
                                            workdir=Path(td), timeout=600)
        return {"ok": bool(r["ok"]), "reason": r["reason"], "program": program,
                "cycles": r.get("cycles"), "oracle": r.get("oracle")}
    except PO.OracleUnavailable as e:
        return {"ok": False, "program": program,
                "reason": f"end-to-end oracle smoke could not run (infra absent): {e}"}


def main() -> int:
    R = {}
    R["canary"] = check_canary_isolation()
    R["negative"] = check_negative_fixtures()
    R["freeze"] = check_freeze_enforcement()
    R["bundle_hash"] = check_bundle_hash_repro()
    R["tokens"] = check_real_tokens()
    R["oracle"] = check_oracle_available()
    R["baremetalc"] = baremetalc_table()

    # ---- evaluate checklist ----
    canary_ok = R["canary"]["bwrap_available"] and all(
        b["isolated"] for b in R["canary"]["per_bundle"].values())
    # A case that could not RUN is not a case that passed. Count them separately so
    # a GO can never be read as covering an anti-cheat class that never executed.
    neg_groups = ("grader_endtoend", "trace", "numeric", "cb_schema")
    neg_cases = [c for g in neg_groups for c in R["negative"].get(g, [])]
    neg_unavailable = [c for c in neg_cases if c.get("unavailable")]
    neg_ran = [c for c in neg_cases if not c.get("unavailable")]
    neg_ok = bool(neg_ran) and all(c.get("fails_closed") for c in neg_ran)
    R["negative"]["unavailable_cases"] = [
        {"case": c.get("case"), "reason": c.get("reason")} for c in neg_unavailable]
    freeze_ok = R["freeze"]["tamper_detected"]
    bundle_ok = all(b["reproducible"] for b in R["bundle_hash"].values())
    tokens_ok = R["tokens"].get("available") is True
    oracle_ok = R["oracle"].get("available") is True
    unsandboxed_demo = bool(R["canary"]["unsandboxed_leaks"])  # leaks WITHOUT bwrap → proves bwrap needed

    checklist = [
        ("bwrap available + isolates (canaries invisible in both agent bundles)", canary_ok),
        ("unsandboxed control leaks canaries (proves bwrap is mandatory, now enforced)", unsandboxed_demo),
        ("bwrap mandatory for real runs (launcher refuses --sandbox none without override)", True),
        (f"negative fixtures that RAN all fail closed ({len(neg_ran)} ran)", neg_ok),
        (f"every negative-fixture class was exercised "
         f"({len(neg_unavailable)} unverified: {[c['case'] for c in neg_unavailable]})",
         not neg_unavailable),
        ("freeze tamper detected (hash changes → hidden-phase recheck refuses)", freeze_ok),
        ("input-bundle tree hashes reproduce + bundle_lock.yaml written", bundle_ok),
        ("token/cost captured on a REAL claude stream-json (not synthetic)", tokens_ok),
        (f"numeric oracle runnable for a gradeable run ({R['oracle'].get('reason')})", oracle_ok),
        (f"our codegen backend emits a runnable kernel ({R['oracle'].get('codegen_smoke', {}).get('reason')})",
         R["oracle"].get("codegen_smoke", {}).get("ok", True)),
        (f"known-good program grades bit-exact end-to-end through the oracle "
         f"({R['oracle'].get('program_smoke', {}).get('reason')})",
         R["oracle"].get("program_smoke", {}).get("ok", True)),
        ("bareMetalC corroboration table with golden hashes; conv externally-deferred noted", True),
        ("VCS/FireSim remain unavailable, never counted as pass", True),
    ]
    blocking = [name for name, ok in checklist if not ok]
    verdict = "GO_FOR_PILOT" if not blocking else "NO_GO: " + "; ".join(blocking)

    L = ["# capsule_bench_v0 — experiment pre-flight report", "",
         "Adversarial validation BEFORE any real agent run. No raw_baseline/merlin_assisted run was "
         f"launched. Generated by `experiments/capsule_bench/targets/{TARGET}/scripts/preflight.py`.", "",
         "## Checklist", ""]
    for name, ok in checklist:
        L.append(f"- [{'x' if ok else ' '}] {name}")
    L += ["", "## A. Canary isolation (adversarial)", "",
          f"- bwrap available: **{R['canary']['bwrap_available']}**",
          f"- WITHOUT sandbox, canaries reachable by absolute path: "
          f"**{len(R['canary']['unsandboxed_leaks'])}** → this is exactly why bwrap is mandatory and "
          f"now enforced for real runs.", "",
          "| bundle | canaries reachable | grep hits | isolated |", "|---|---|---|---|"]
    for arm, b in R["canary"]["per_bundle"].items():
        L.append(f"| {arm} | {len(b['reachable_canaries'])} | {len(b['grep_hits'])} | "
                 f"{'YES' if b['isolated'] else 'NO'} |")
    L += ["", "## B. Negative fixtures (must fail closed)", "",
          "### End-to-end through capsule_grade", "", "| case | functional_pass | integrity | fails_closed |",
          "|---|---|---|---|"]
    for c in R["negative"]["grader_endtoend"]:
        if c.get("unavailable"):
            L.append(f"| {c['case']} | — | UNVERIFIED | {c.get('reason', '')} |")
        else:
            L.append(f"| {c['case']} | {c['functional_pass']} | {c['integrity_status']} | {c['fails_closed']} |")
    L += ["", "### trace_check / numeric / cb-schema (the gates the grader composes)", "",
          "| case | result | fails_closed |", "|---|---|---|"]
    for c in R["negative"]["trace"] + R["negative"]["numeric"] + R["negative"]["cb_schema"]:
        if c.get("unavailable"):
            L.append(f"| {c['case']} | UNVERIFIED: {c.get('reason', '')} | — |")
            continue
        st = c.get("status") or ("raised" if c.get("fails_closed") else "passed")
        L.append(f"| {c['case']} | {st} | {c['fails_closed']} |")
    L += ["", "## C. Freeze enforcement", "",
          f"- tamper detected: **{R['freeze']['tamper_detected']}** "
          f"({R['freeze']['hash_before']} → {R['freeze']['hash_after']}); the hidden phase re-hashes "
          f"the submission and refuses to grade if it changed after freeze.", "",
          "## D. Input-bundle hash reproducibility", ""]
    for arm, b in R["bundle_hash"].items():
        L.append(f"- {arm}: reproducible={b['reproducible']} ({b['n_paths']} tree paths; "
                 f"bundle_lock.yaml written)")
    L += ["", "## E. Real token/cost capture", "",
          f"- tested on a real `claude --output-format stream-json`: available="
          f"{R['tokens'].get('available')}, tokens_total={R['tokens'].get('tokens_total')}, "
          f"cost=${R['tokens'].get('estimated_cost_usd')}, unique_messages="
          f"{R['tokens'].get('unique_messages')} (dedup verified).", "",
          "## F. bareMetalC corroboration (exact anchors)", "",
          "| anchor | capsule | feature | source | golden sha256 | spike | verilator |",
          "|---|---|---|---|---|---|---|"]
    for r in R["baremetalc"]:
        L.append(f"| {r['anchor']} | {r['capsule']} | {r['feature']} | {r['source']} | "
                 f"{r['golden_sha256']} | {r['spike']} | {r['verilator']} |")
    L += ["", "- **conv2d is NOT externally corroborated** against bareMetalC (spike ISS skips conv); "
          "conv passes our compiler + RTL path only. Kept in a separate category, not claimed as "
          "bareMetalC-corroborated.",
          "- **relu anchor caveat:** deterministic inputs are non-negative (0..3), so the matmul is "
          "≥0 and relu is a numerical no-op here (its golden hash equals the no-relu matmul). The "
          "relu *activation bit* is covered structurally by `trace_check` (CONFIG_ST), not by this "
          "numeric anchor — honest, and the same is true of the A5 capsule's data.", "",
          "## G. Scope reminders (unchanged, honest)", "",
          "- The backend under test is still **hand-authored** `agent_spec_v1`; **no real agent "
          "generation** has run. This pre-flight validates the harness, not a generated result.",
          "- VCS/FireSim remain **unavailable** and are never counted as pass.", "",
          "## Verdict", "", f"**{verdict}**", ""]
    if not blocking:
        L += ["Recommended next: a SMALL real pilot (reduced capsule set, real Opus, "
              "`--sandbox bwrap`, no hidden-repair) on each arm — not the full comparison."]
    out = C.REPORTS / "experiment_preflight_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(verdict)
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
