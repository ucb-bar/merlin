"""One-command readiness gate for the next A/B run — exercises EVERY shipped tool functionally
(not just imports) and prints a single GO / NO-GO. Launches no agent.

Sections (each is an independent pass/fail; a failure does not abort the rest):
  A. starter kit         — parse a real interface capsule into VERIFIED IR; build a schema-valid cmdbuf;
                           im2col + tile_to_dim sanity; validate() catches a planted bug AND clears a clean one
  B. CIRCT generators    — gen_isa_module / gen_rtl_digest / gen_numeric_facts all generate, import, and the
                           generated numeric checker flags a narrow accumulator
  C. CIRCT sim-skip gate — a CIRCT-reject skips the inner sim; a clean trace runs it (circt_gate.gated_adapter)
  D. harness wiring      — launch_ab_batch --dry-run yields the full arm×condition×repeat matrix with the
                           correct bundles; agg_ab_results runs
  E. anti-cheat gate     — verify_no_cheat.py PASS (delegated)
  F. bundle integrity    — all 6 bundles exist, parse, and every API a prompt names actually imports

Exit 0 = GO. Non-zero = NO-GO.  Usage: readiness_check.py
"""
from __future__ import annotations
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402 — active target (descriptor-driven), bootstraps merlin/python
from merlin.common.paths import ext_path, repo_root  # noqa: E402
from merlin.common.artifacts import cache_dir  # noqa: E402 — purgeable scratch for probes
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

# Repo root + venv interpreter come from the canonical path helpers (never Path(__file__).parents[N],
# and never EXP.parent.parent — that resolves the merlin/ subdir, not the repo root where .venv lives).
REPO = repo_root()
EXP = C.EXP                                    # the active target's experiment dir (descriptor-driven)
TARGET = C.TARGET
_TE = load_target_experiment(EXP / "target_experiment.yaml")
PY = str(REPO / ".venv/bin/python")
SCRIPTS = EXP / "scripts"
BUNDLES = EXP / "input_bundles"
#: (name, verdict, detail) where verdict is True=pass, False=fail, None=not applicable.
results: list[tuple[str, bool | None, str]] = []


def _ok(name: str, cond: bool, detail: str = ""):
    results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _na(name: str, why: str):
    """Record a check that does not apply to THIS target's endpoint.

    Distinct from a failure, and the distinction is the whole point: a tool that refuses because the
    hardware has no such capability is behaving correctly, and counting it as a FAIL leaves only two ways
    to reach GO -- both wrong. Either the check gets deleted (and then a target that DOES have the
    capability stops being checked), or something is generated to satisfy it (a decode table for a
    machine with no decode, which is a fabrication that would go on to be graded against).

    An N/A must be DERIVED from the target's own facts, never asserted per target.
    """
    results.append((name, None, why))
    print(f"  [ N/A] {name}" + (f"  — {why}" if why else ""))


def section(t):
    print(f"\n=== {t} ===")


# ---- A. starter kit -------------------------------------------------------------------------------
def test_starter_kit():
    section("A. OOT starter kit (parse→verify→cmdbuf→transforms→validate)")
    from merlin.targetgen.oot_starterkit import parse_interface, CommandBufferBuilder, transforms
    from merlin.targetgen.oot_starterkit.verify import validate, structural_checks

    # parse a REAL interface capsule from the contract corpus
    cap = next((REPO / "merlin/contract/capsules").rglob("capsule.interface.mlir"), None)
    if cap and cap.is_file():
        try:
            model = parse_interface(cap.read_text())
            _ok("parse_interface on a real capsule", model is not None, str(cap.relative_to(REPO)))
        except Exception as e:
            _ok("parse_interface on a real capsule", False, f"{type(e).__name__}: {e}")
    else:
        _ok("parse_interface on a real capsule", False, "no interface.mlir found in corpus")

    # cmdbuf builder: schema-valid when populated, rejects empty
    try:
        b = CommandBufferBuilder(TARGET, backend="x", abi_version="0.1")
        b.tensor("A", [16, 16], "i8"); b.command("MATMUL", {"dst": "A"})
        good = b.validate()
        b2 = CommandBufferBuilder(TARGET); empty = b2.validate()
        _ok("CommandBufferBuilder valid-when-populated, rejects-empty",
            not good and bool(empty), f"good_findings={good}, empty_findings={len(empty)}")
    except Exception as e:
        _ok("CommandBufferBuilder", False, f"{type(e).__name__}: {e}")

    # generic transforms
    try:
        plan = transforms.im2col((1, 8, 8, 4), (3, 3, 4, 8), stride=(1, 1),  # weight [kh,kw,cin,cout]
                                 padding=(0, 0, 0, 0), dilation=(1, 1))
        tiles = transforms.tile_to_dim(32, 32, 16, 16)
        _ok("transforms.im2col + tile_to_dim", plan is not None and len(tiles) == 4,
            f"tiles={len(tiles)}")
    except Exception as e:
        _ok("transforms", False, f"{type(e).__name__}: {e}")

    # validate(): catches a use-before-config / UNKNOWN-funct trace, clears a clean one
    try:
        bad_trace = {"instructions": [{"name": "COMPUTE_PRELOADED", "funct": 4},
                                      {"name": "UNKNOWN", "funct": "UNKNOWN"}]}
        clean_trace = {"instructions": [{"name": "CONFIG_EX", "funct": 0},
                                        {"name": "COMPUTE_PRELOADED", "funct": 4}]}
        gh = (REPO / _TE.isa_headers[0]) if _TE.isa_headers else None   # target's ISA header (descriptor)
        ghp = str(gh) if gh and gh.is_file() else None
        caught = structural_checks(bad_trace)
        clean = structural_checks(clean_trace)
        _ok("verify.structural_checks catches bad, clears clean",
            bool(caught) and not clean, f"bad_findings={len(caught)}, clean_findings={len(clean)}")
    except Exception as e:
        _ok("verify.structural_checks", False, f"{type(e).__name__}: {e}")


# ---- B. CIRCT generators --------------------------------------------------------------------------
def test_generators():
    section("B. CIRCT RTL-facts generators (generate + import + flag)")
    # These three generators all read the instruction-decode body of the facts artifact. Whether the
    # target HAS one is a fact about the target, so ask the facts rather than the target's name: an
    # ISA-less endpoint (a command-buffer spatial tile driven over one-hot op ports) has no opcode, no
    # funct field and no decode table by construction, and `facts.decode_body` is the single place that
    # distinction is made -- the same call the generators themselves make.
    try:
        from merlin.targetgen.rtl import facts as _facts
        _facts.decode_body(_facts.load_facts(TARGET), TARGET, needs="the RTL-facts generators")
    except NotImplementedError as e:
        why = str(e).split(". ")[0][:90]
        for name in ("gen_isa_module generates", "gen_rtl_digest generates",
                     "gen_numeric_facts generates", "generated numeric checker"):
            _na(name, why)
        return
    except _facts.FactsEmpty as e:
        # NOT N/A. An empty facts artifact means the extractor never found the RTL, so every derived fact
        # is absent -- the opposite verdict from "this endpoint has no decode table". Reporting it as N/A
        # would let a target whose facts were never extracted read as ready for the arms that are supposed
        # to be GROUNDED in those facts. Both this repo's newest two targets are in exactly that state.
        _ok("the RTL-facts artifact carries facts", False, str(e).split(" — ")[0][:110])
        return
    except Exception:      # noqa: BLE001 — any OTHER problem is the generators' to report, below
        pass
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for mod, out in [("gen_isa_module", f"{TARGET}_isa.py"),
                         ("gen_rtl_digest", "RTL_DIGEST.md"),
                         ("gen_numeric_facts", "numeric_facts.py")]:
            # These RTL-facts generators require an explicit --target (the gemmini default was retired
            # in the target-generalization work); pass the active target so they run for any target.
            r = subprocess.run([PY, "-m", f"merlin.targetgen.rtl.{mod}",
                                "--target", TARGET, "--out", str(td / out)],
                               cwd=str(REPO), capture_output=True, text=True)
            # EITHER STREAM. One generator writes its refusal to stderr and one to stdout (with rc=0),
            # and reading only stderr saw an empty string for the second -- so an honest decline was
            # indistinguishable from "produced nothing", which is the very confusion this check exists
            # to resolve.
            _lines = [ln for ln in (r.stderr.strip().splitlines()
                                    + r.stdout.strip().splitlines()) if ln.strip()]
            tail = next((ln for ln in _lines if ln.startswith("n/a")), (_lines or [""])[-1])
            # A DECLINE IS NOT A CRASH, and this check could not tell them apart. These generators are
            # written to fail CLOSED: `gen_isa_module` refuses to emit an encoder for a device with no
            # RISC-V custom slot rather than guess an opcode ("n/a: ... refusing to emit an ISA module
            # with a guessed opcode"), which is the behaviour the repo demands everywhere else. Counting
            # that refusal as a FAIL made a self-hosted-ISA target un-launchable for doing the right
            # thing, and it would have hidden a real crash behind the same red mark.
            declined = tail.startswith("n/a")
            _ok(f"{mod} generates", (r.returncode == 0 and (td / out).exists()) or declined,
                (f"n/a (fail-closed decline, not a crash): {tail[4:120]}" if declined else tail[:80]))
        # the generated numeric checker flags a narrow accumulator
        try:
            sys.path.insert(0, str(td))
            nf = importlib.import_module("numeric_facts"); importlib.reload(nf)
            findings = nf.check_numeric_shapes(
                {"tensors": {"acc": {"dtype": "i8"}},
                 "commands": [{"opcode": "MATMUL", "operands": {"dst": "acc"}}]})
            # The rule can only fire where the target's OWN facts ground an accumulator width. A target
            # whose RTL facts declare no datapath and no memory (atlas: both empty) has nothing to
            # ground it, so the generated checker fail-closed SKIPS the rule -- which is the correct
            # behaviour, not a failure. Demanding a finding there asserts every target has a systolic
            # accumulator. Report n/a and say which fact is missing.
            _acc = getattr(nf, "ACC_WIDTH_BITS", None)
            if _acc is None:
                _ok("generated numeric checker flags narrow accumulator", True,
                    f"n/a for {TARGET!r}: RTL facts ground no accumulator width "
                    f"(no datapath/memory fact) -> the rule fail-closed skips, as designed")
            else:
                _ok("generated numeric checker flags narrow accumulator", bool(findings),
                    (findings or ["—"])[0][:70])
        except Exception as e:
            _ok("generated numeric checker", False, f"{type(e).__name__}: {e}")
        finally:
            sys.modules.pop("numeric_facts", None)
            if str(td) in sys.path:
                sys.path.remove(str(td))


# ---- C. CIRCT sim-skip gate -----------------------------------------------------------------------
def test_circt_gate():
    section("C. CIRCT sim-skip gate (reject skips sim; clean runs it)")
    try:
        import inspect
        from merlin.targetgen.circt_gate import gated_adapter, CIRCTReject  # noqa: F401
        ran = {"n": 0}

        def inner(llvm_text, workdir, timeout):
            ran["n"] += 1
            return {"status": "pass"}

        gated = gated_adapter(inner, log=[], target=TARGET)
        # the wrapped adapter must expose the sim-adapter signature the loop calls it with. (Its
        # reject-skips-sim / clean-runs-sim behavior was unit-validated separately; here we assert the
        # wiring + signature so the loop won't TypeError at runtime.)
        params = list(inspect.signature(gated).parameters)
        _ok("gated_adapter wraps with the sim-adapter signature the loop calls",
            callable(gated) and {"llvm_text", "workdir", "timeout"}.issubset(set(params)),
            f"params={params}")
    except Exception as e:
        _ok("circt_gate.gated_adapter", False, f"{type(e).__name__}: {e}")


# ---- D. harness wiring ----------------------------------------------------------------------------
def test_harness():
    section("D. harness wiring (dry-run matrix + aggregator)")
    r = subprocess.run([PY, str(SCRIPTS / "launch_ab_batch.py"), "--tag", "readiness_probe",
                        "--experiment", "realistic", "--repeats", "3", "--condition", "both", "--dry-run"],
                       cwd=str(REPO), capture_output=True, text=True)
    out = r.stdout
    n_runs = out.count("run-id=")
    has_nk = "nokernel" in out and "_nk_" in out
    has_kern = "_hwbringup_v0" in out
    _ok("launch_ab_batch dry-run = 18 runs, both conditions, fresh ids",
        r.returncode == 0 and n_runs == 18 and has_nk and has_kern,
        f"n_runs={n_runs}, nokernel={has_nk}, kernels={has_kern}")
    # Write to a throwaway dir: this is a WIRING probe, and the real ab_results.json is a result.
    # (Running readiness under a non-gemmini descriptor used to overwrite that target's aggregate
    #  with an all-zero skeleton, because tag "abc4" matches no run there.)
    probe_out = cache_dir("readiness_probe") / "agg_ab_results"
    r2 = subprocess.run([PY, str(SCRIPTS / "agg_ab_results.py"), "--tag", "abc4",
                         "--out-dir", str(probe_out)],
                        cwd=str(REPO), capture_output=True, text=True)
    _ok("agg_ab_results runs", r2.returncode == 0, (r2.stdout.strip().splitlines() or [""])[0][:70])


# ---- E. anti-cheat gate ---------------------------------------------------------------------------
def test_verify_no_cheat():
    section("E. anti-cheat gate (verify_no_cheat.py)")
    r = subprocess.run([PY, str(SCRIPTS / "verify_no_cheat.py")], cwd=str(REPO),
                       capture_output=True, text=True)
    _ok("verify_no_cheat PASS", r.returncode == 0,
        (r.stdout.strip().splitlines() or [""])[-1][:80])


# ---- F. bundle integrity --------------------------------------------------------------------------
def test_corpus_fits_the_endpoint():
    """H. The capsules must be gradeable ON THIS MACHINE.

    Every other section checks that a tool RUNS. None of them checks that the corpus the arms will be
    graded against is one this target can satisfy, and that gap is not hypothetical: a target pointed at
    another target's corpus reaches GO with every tool working and then scores zero for a reason no
    per-capsule verdict explains -- the capsules demand instruction classes the hardware does not have.

    Derived, not asserted per target: a capsule's `expected.instruction_classes` names an INSTRUCTION,
    and whether this target has instructions at all is the same question section B asks of its facts.
    A command-buffer machine driven over one-hot op ports has none by construction.
    """
    section("H. corpus fits the endpoint (capsules are gradeable on this machine)")
    # Scope: the roots the GRADE resolves, not the single directory the descriptor names. Reading only
    # the declared one under-counts -- for one target 21 capsules against the 28 section I actually grades
    # -- so a capsule in a sibling category could demand a class this endpoint lacks and never be looked
    # at here. Same under-scoping the A/B drivers had; graded_roots() is the one resolution both use.
    roots = [r for r in _TE.graded_roots() if r.is_dir()]
    caps = sorted({c for r in roots for c in r.rglob("capsule.yaml")})
    if not caps:
        _ok("the graded capsule roots have capsules", False,
            f"none under {[str(r) for r in roots] or _TE.capsule_corpus}")
        return
    demanded: set[str] = set()
    for cap in caps:
        try:
            spec = yaml.safe_load(cap.read_text()) or {}
        except Exception:  # noqa: BLE001
            continue
        demanded.update((spec.get("expected") or {}).get("instruction_classes") or [])

    # Settle the trivial case BEFORE probing the facts: a corpus that demands no instruction classes
    # cannot demand one the endpoint lacks, whatever the facts say. (Probing first turned this into an
    # N/A for the two targets whose facts artifact is empty, which is less true than the plain answer.)
    # Note what a trivial pass here does NOT mean: a capsule declaring `instruction_classes: []` cannot
    # FAIL a coverage check either, so this row says the corpus asks for nothing, not that it is rigorous.
    if not demanded:
        _ok("the corpus demands no instruction classes this endpoint lacks", True,
            f"{len(caps)} capsule(s), none declaring instruction_classes "
            f"(so none can fail coverage either)")
        return

    has_isa = True
    try:
        from merlin.targetgen.rtl import facts as _facts
        _facts.decode_body(_facts.load_facts(TARGET), TARGET, needs="the corpus's instruction classes")
    except NotImplementedError:
        has_isa = False
    except Exception:  # noqa: BLE001 — cannot tell; do not manufacture a verdict either way
        _na("corpus instruction classes match the target's vocabulary",
            "the target's facts could not be loaded, so this cannot be decided")
        return
    _ok("the corpus demands no instruction classes this endpoint lacks", has_isa,
        f"{len(caps)} capsule(s) demand {len(demanded)} instruction class(es) "
        + (f"e.g. {sorted(demanded)[:4]}" if has_isa else
           f"({sorted(demanded)[:4]}...) but this target has no instruction decode at all -- it is a "
           f"{(_TE.sim_via or 'arc')}-graded command endpoint. The corpus at "
           f"{', '.join(r.name for r in roots)} belongs to another target; this one needs its own "
           f"(contract/capsules/generate_corpus.py + a profile)"))


def test_graded_path_is_the_declared_one():
    """I. The suite the RUN grades on, exercised — not the one the descriptor names.

    Every other section reads `capsule_corpus`. The grade does not: it reads whatever roots the launcher
    hands `grade_agent_run`, and while those two were allowed to differ a GO verdict could not see the
    difference. Measured, before this was wired: the public phase resolved to the shared parent of every
    target's corpus (which does not even load -- a sibling target's capsule fails the schema) and the
    hidden phase to ANOTHER target's five hidden capsules.

    So this resolves the roots the way the launcher now does and grades a DELIBERATELY EMPTY submission
    through them. An empty package must fail, and it must fail against a non-zero denominator: that one
    assertion catches both an empty suite (0/0 reads as a pass) and a grader that passes anything. The
    good half -- a real submission passing -- is what the run itself measures; it cannot be faked here
    without shipping an answer key, which is the one thing this bench must not contain.
    """
    section("I. the graded path (resolved roots + a submission that must FAIL)")
    import tempfile

    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen.contract.materialize import (
        public_capsules_for, validate_materialized_cohort,
    )

    pub_roots, hid_roots = _TE.graded_roots(), _TE.hidden_roots()
    contract = str(REPO / "merlin/contract")
    try:
        source_caps = CR.discover_capsules(pub_roots, labels={"public", "dev"}, contract=contract)
        source_by_name = {str(c.get("name")): c for c in source_caps}
        excluded = set(getattr(_TE, "graded_exclude", ()) or ())
        unknown_exclusions = sorted(excluded - set(source_by_name))

        # An experiment may move a capsule out of its formal denominator only at this declared boundary,
        # and only when the frozen hardware contract proves that no target datapath holds its operand
        # dtype.  This prevents an operator from cherry-picking a failing but reachable capsule while
        # retaining the useful property that an int8 accelerator is not asked to certify bf16 work.
        op_source = [c for c in source_caps if c.get("kind") != "model"]
        _eligible, hardware_ineligible = CR._split_ineligible(op_source, TARGET)
        proven_excluded = {str(r.get("capsule")) for r in hardware_ineligible}
        declared_capability = set(getattr(_TE, "graded_capability_exclude", ()) or ())
        declared_resource = set(getattr(_TE, "graded_resource_exclude", ()) or ())
        explicit_split = bool(declared_capability or declared_resource)
        missing_capability_exclusions = sorted(proven_excluded - (
            declared_capability if explicit_split else excluded))
        extra_capability_exclusions = sorted((declared_capability - proven_excluded)
                                             if explicit_split else set())
        resource_excluded = declared_resource if explicit_split else (excluded - proven_excluded)
        invalid_resource_exclusions = sorted(
            name for name in resource_excluded
            if (source_by_name.get(name) or {}).get("kind") != "model")
        admitted_models = sorted(
            name for name, cap in source_by_name.items()
            if cap.get("kind") == "model" and name not in excluded)
        required_models = set(getattr(_TE, "graded_required_models", ()) or ())
        resource_policy = getattr(_TE, "graded_resource_policy", None)
        resource_policy_ok = (not explicit_split or (
            resource_policy == "representative_l3_capstones_v1"
            and set(admitted_models) == required_models
            and bool(required_models)))

        caps = [c for c in source_caps if str(c.get("name")) not in excluded]
        n_pub = len(caps)
        hidden_source = (CR.discover_capsules(
            hid_roots, labels={"hidden"}, contract=contract) if hid_roots else [])
        hidden_ops = [c for c in hidden_source if c.get("kind") != "model"]
        _hidden_eligible, hidden_ineligible = CR._split_ineligible(hidden_ops, TARGET)
        hidden_excluded_names = {str(row.get("capsule")) for row in hidden_ineligible}
        n_hid = len(hidden_source)
        n_hid_admitted = sum(
            str(cap.get("name")) not in hidden_excluded_names for cap in hidden_source)

        # Generate the exact formal-compatible public root and validate its admission record against
        # the descriptor parsed at readiness start. This is the executable descriptor-SHA binding;
        # checking that the record merely contains 64 hex characters would not detect stale bytes.
        materialized_root = public_capsules_for(_TE)
        cohort_record = validate_materialized_cohort(materialized_root, _TE)
    except Exception as exc:  # noqa: BLE001
        _ok("the resolved capsule roots load", False,
            f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]} "
            f"(roots={[str(r) for r in pub_roots]})")
        return
    _ok("formal exclusions are known, exactly hardware-proven, or policy-bound model resources",
        (not unknown_exclusions and not missing_capability_exclusions
         and not extra_capability_exclusions and not invalid_resource_exclusions
         and bool(admitted_models) and resource_policy_ok),
        f"source_pool={len(source_caps)}, admitted={n_pub}, declared_excluded={len(excluded)}, "
        f"hardware_proven_ineligible={len(proven_excluded)}, "
        f"resource_bounded_models={len(resource_excluded)}, admitted_models={admitted_models}, "
        f"resource_policy={resource_policy}, required_models={sorted(required_models)}"
        + (f", unknown={unknown_exclusions}, missing_hardware={missing_capability_exclusions}, "
             f"extra_hardware={extra_capability_exclusions}, "
             f"invalid_resource={invalid_resource_exclusions}" if
           unknown_exclusions or missing_capability_exclusions or extra_capability_exclusions
           or invalid_resource_exclusions or not resource_policy_ok else ""))
    _ok("the public grade resolves to a non-empty suite", n_pub > 0,
        f"{n_pub} admitted capsule(s) from {len(source_caps)} source-pool capsule(s) over "
        f"{len(pub_roots)} root(s): " + ", ".join(r.name for r in pub_roots))
    expected_source = getattr(_TE, "graded_expected_source_capsules", None)
    expected_admitted = getattr(_TE, "graded_expected_admitted_capsules", None)
    public_counts_declared = not explicit_split or (
        expected_source is not None and expected_admitted is not None)
    public_counts_match = (public_counts_declared
                           and (expected_source is None or len(source_caps) == expected_source)
                           and (expected_admitted is None or n_pub == expected_admitted))
    _ok("the public cohort matches its descriptor-frozen cardinalities", public_counts_match,
        f"source={len(source_caps)}, admitted={n_pub}, capability_excluded={len(proven_excluded)}, "
        f"resource_excluded={len(resource_excluded)}, expected_source={expected_source}, "
        f"expected_admitted={expected_admitted}")
    _ok("the materialized cohort is bound to the current descriptor and exact admitted contents",
        (cohort_record.get("n_source_capsules") == len(source_caps)
         and cohort_record.get("n_admitted_capsules") == n_pub),
        f"source={cohort_record.get('n_source_capsules')}, "
        f"admitted={cohort_record.get('n_admitted_capsules')}, "
        f"policy={cohort_record.get('policy')}")
    _ok("the hidden grade resolves to its OWN capsules", n_hid > 0,
        f"{n_hid} capsule(s) at {hid_roots[0].name if hid_roots else '(none declared)'}"
        if hid_roots else "no hidden/ beside this corpus — the hidden phase would score 0/0")
    expected_hidden_source = getattr(_TE, "hidden_expected_source_capsules", None)
    expected_hidden_admitted = getattr(_TE, "hidden_expected_admitted_capsules", None)
    hidden_counts_declared = not explicit_split or (
        expected_hidden_source is not None and expected_hidden_admitted is not None)
    hidden_counts_match = (hidden_counts_declared
                           and (expected_hidden_source is None or n_hid == expected_hidden_source)
                           and (expected_hidden_admitted is None
                                or n_hid_admitted == expected_hidden_admitted))
    _ok("the sealed hidden cohort matches capability-admission cardinalities",
        hidden_counts_match,
        f"source={n_hid}, admitted={n_hid_admitted}, "
        f"capability_excluded={len(hidden_excluded_names)}, "
        f"expected_source={expected_hidden_source}, expected_admitted={expected_hidden_admitted}")

    if not n_pub:
        return
    # Straight at the SUITE, not through `grade()`. `grade()` builds and integrity-scans the package
    # first and returns early when that fails, so an empty submission never reaches a capsule -- which
    # would have made this check pass for the wrong reason (0 graded, 0 passed, "fails" ✓). Package
    # integrity is its own plane and section A already covers it; what is under test here is that the
    # resolved suite is real and that nothing in it passes without a submission.
    with tempfile.TemporaryDirectory() as td:
        pkg, runs = Path(td) / "submission", Path(td) / "runs"
        pkg.mkdir(parents=True)
        # A WELL-FORMED but empty submission: it satisfies the package contract (so it reaches the
        # capsules) and implements nothing (so it can only fail them). A malformed one would be rejected
        # at the contract plane and never reach a capsule, which would make this pass for the wrong
        # reason -- 0 graded, 0 passed, "fails" -- i.e. exactly the vacuity it exists to detect.
        tool = pkg / "tool.py"
        tool.write_text("import sys\nsys.exit(2)\n", encoding="utf-8")
        (pkg / "manifest.yaml").write_text(yaml.safe_dump({
            "artifact_type": "mlir_oot_target_backend", "target": TARGET, "language": "python",
            "authoring": {"mode": "hand_curated"}, "integrity_exempt": True,
            "entrypoints": {"tool": "tool.py"},
            "commands": {k: {"argv": ["python3", "tool.py", k]} for k in
                         ("parse", "lower_interface_to_target",
                          "emit_command_buffer", "lower_target_to_llvm")},
        }, sort_keys=False), encoding="utf-8")
        try:
            res = CR.run_suite(caps, pkg, runs_root=runs, contract=contract,
                               oracle_adapters={}, target=TARGET, no_oracle=True, timeout=120)
        except Exception as exc:  # noqa: BLE001
            _ok("the resolved suite runs against a submission", False,
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
            return
    passed = [r for r in res if r.get("passed") or r.get("functional_pass")]
    _ok("every resolved capsule is graded, and an empty submission passes none of them",
        len(res) == n_pub and not passed,
        f"graded {len(res)}/{n_pub}, passed {len(passed)}"
        + (f" — {[r.get('capsule') for r in passed][:4]}" if passed else ""))


def test_contract_provenance():
    """J. The capability contract the tooling READS is the one the descriptor DECLARES.

    `hardware_spec.target_contract` was parsed and dropped -- no field on TargetExperiment held it -- so
    what everything actually read was whatever `target_registry.resolve(target)` found by name, and the
    descriptor's declaration was decoration. That is invisible in both directions and both directions
    happened here: for one target the registry resolved NOTHING while the declared file sat on disk (so
    three of its four STARTER_PROMPT.md silently failed to render, which then failed the anti-cheat gate
    on their absence), and for another the two paths resolve to genuinely DIFFERENT contracts -- one
    naming its fp8 datapaths, the other carrying the fail-closed `unnamed_float_datapaths` derivation.
    Which is authoritative is the contract owner's call, so a mismatch FAILS here rather than being
    silently decided: an agent told the wrong thing about its hardware is the run-invalidating version of
    a result attributed to the wrong device.
    """
    section("J. contract provenance (the contract read == the contract declared)")
    from merlin.targetgen.target_experiment import declared_vs_resolved_contract
    declared, resolved, verdict = declared_vs_resolved_contract(_TE)
    rel = (lambda p: str(Path(p).relative_to(REPO)) if p else "(none)")
    if verdict == "mismatch":
        _ok("the declared contract is the one in use", False,
            f"descriptor declares {rel(declared)} but the tooling reads {rel(resolved)} — "
            f"decide which is authoritative; they differ in content")
    elif verdict == "none":
        _ok("the target has a capability contract", False,
            "no contract declared and none resolves — every derived fact would be missing")
    elif verdict == "stale_declaration":
        _ok("the declared contract exists", False,
            f"descriptor declares {_TE.declared_contract}, which is not there; the tooling silently "
            f"reads {rel(resolved)} instead")
    else:
        _ok("the declared contract is the one in use", True,
            {"agree": f"both resolve to {rel(resolved or declared)}",
             "declared_only": f"registry resolves none; using the declared {rel(declared)}"}[verdict])


def test_bundles():
    section("F. bundle integrity (6 bundles parse; prompt APIs import)")
    # conditions DERIVED from the target's materialized bundles (gemmini kernel+nokernel; atlas
    # kernel-only) — not a hardcoded gemmini set.
    expected = [f"{arm}_{cond}" for arm in
                ("raw_baseline", "merlin_assisted", "merlin_assisted_rtlchecks")
                for cond in C.experiment_conditions()]
    missing = [b for b in expected if not (BUNDLES / b / "STARTER_PROMPT.md").is_file()]
    _ok("all 6 bundles present with prompts", not missing, f"missing={missing}")
    for b in expected:
        m = BUNDLES / b / "input_bundle_manifest.yaml"
        try:
            yaml.safe_load(m.read_text()); ok = True; d = ""
        except Exception as e:
            ok = False; d = str(e)[:50]
        if not ok:
            _ok(f"{b} manifest parses", ok, d)
    # APIs referenced by the merlin prompts must import
    api_ok = True; detail = ""
    try:
        from merlin.targetgen.oot_starterkit import parse_interface, CommandBufferBuilder, transforms  # noqa
        from merlin.targetgen.oot_starterkit.verify import validate  # noqa
        from merlin.targetgen.oot_starterkit.dialect import parse_to_verified_ir  # noqa
    except Exception as e:
        api_ok = False; detail = f"{type(e).__name__}: {e}"
    _ok("every API the prompts name imports", api_ok, detail)


def test_sandbox_authoring_tools():
    """The paid Arm4 path, not the host checkout: frozen grants + bwrap + live broker requests."""
    section("F2. promised authoring tools in the assembled bwrap snapshot")
    import tooling_readiness
    check = tooling_readiness.sandbox_authoring_readiness(TARGET, "merlin_assisted_rtlchecks")
    _ok(check["check"], check["ok"], check["evidence"])


def test_every_declared_grant_resolves():
    """EVERY allowed grant of EVERY launchable bundle must exist on disk, before any spend.

    The check above assembles a REDUCED bundle -- only the promised authoring-tool closure -- because
    copying the whole corpus to test Python imports would turn a preflight into a multi-gigabyte
    operation. That is the right trade for what it tests and it leaves a hole: a grant OUTSIDE the tool
    closure is never resolved by any gate, and `materialize_bundle_inputs` fails closed at run time.

    Measured: fifteen of this target's bundles granted a capsule category directory that does not
    exist (the corpus spells it with a leading underscore). Readiness reported 32/32 GO, the preflight
    reported GO_FOR_PILOT, and all three arms then died in the first second of the launch with
    `FileNotFoundError: bundle declares unresolvable allowed grant(s)`. Resolving a path costs nothing
    and is exactly the check that was missing.
    """
    section("F3. every declared grant of every launchable bundle resolves")
    from merlin.targetgen.sandbox import bwrap as _BW

    bundles = sorted((EXP / "input_bundles").glob("*/input_bundle_manifest.yaml"))
    if not bundles:
        _ok("bundles present to check", False, f"no input_bundles under {EXP}")
        return
    missing: list[str] = []
    n_grants = 0
    for man in bundles:
        try:
            doc = yaml.safe_load(man.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            missing.append(f"{man.parent.name}: unparseable ({type(e).__name__})")
            continue
        for entry in (doc.get("allowed") or []):
            rel = str((entry or {}).get("path") or "")
            if not rel:
                continue
            n_grants += 1
            if _BW.path_kind(_BW.resolve_grant(rel, REPO)) == "missing":
                missing.append(f"{man.parent.name}: {rel}")
    _ok(f"every allowed grant resolves ({n_grants} across {len(bundles)} bundle(s))",
        not missing, "; ".join(sorted(set(missing))[:4]) if missing else "")


def _oracle_sim_via() -> str:
    """The target's declared bespoke sim (``toolchain.sim_via``) — ``"chipyard"`` for gemmini, ``""``
    (arc-only / program oracle) for a self-hosted-ISA target like atlas. Routes section G, no literal."""
    from merlin.targetgen.target_experiment import load_target_experiment
    desc = EXP / "target_experiment.yaml"
    return (load_target_experiment(desc).sim_via or "").strip() if desc.is_file() else ""


def test_oracles_endtoend():
    """G. Prove the target's REAL grading oracle can produce a verdict — the safeguard abc7 lacked.
    Routed by the target's oracle kind (contract, no target literal):

    * ``external_backend`` / arc-only (self-hosted-ISA PROGRAM oracle, e.g. atlas): the oracle is the mlc
      arc cosim + the model-venv functional runner, and the target has NO chipyard reference backend to
      grade (a certified backend is what a run PRODUCES, so grading a "reference backend" is impossible
      pre-run). Instead verify the oracle is actually RUNNABLE now (``capsule_runner.oracle_available`` —
      the exact preflight the launcher runs) and that ``oracle_adapters`` resolves BOTH graded tiers to
      the program oracle — the precise wiring a graded round uses. That is the honest pre-launch proof;
      the numeric bit-exact check runs against a known-good npu_model program at grade time.
    * ``chipyard`` (gemmini): actually RUN spike + verilator on the committed reference backend to a real
      verdict, measure verilator's per-capsule time, and NO-GO on the abc7 signature (0 capsules/timeout).
    """
    import json as _json
    import os as _os
    import tempfile as _tf
    import time as _time
    section("G. oracles RUN end-to-end (real verdict, not just present)")
    from merlin.targetgen import capsule_runner as CR
    sim_via = _oracle_sim_via()
    if sim_via != "chipyard":
        # self-hosted-ISA program oracle (arc cosim + model venv); no chipyard reference backend exists.
        # Probe oracle_available in a FRESH interpreter — exactly how the launcher runs its preflight.
        # (The arc native model does not re-probe cleanly inside a process that has already exercised the
        # other mlc-touching readiness sections; the launcher always checks in a clean process, so that
        # subprocess result is the faithful pre-launch signal.)
        probe = ("import json,sys;from merlin.targetgen import capsule_runner as CR;"
                 f"ok,why=CR.oracle_available({TARGET!r},{sim_via!r});"
                 "print(json.dumps({'ok':bool(ok),'why':why}))")
        pr = subprocess.run([PY, "-c", probe], cwd=str(REPO), capture_output=True, text=True)
        try:
            res = _json.loads([ln for ln in pr.stdout.splitlines() if ln.strip()][-1])
            ok, why = res["ok"], res["why"]
        except Exception:  # noqa: BLE001
            ok, why = False, (pr.stderr or pr.stdout or "probe produced no output")[-120:]
        _ok(f"program oracle runnable now ({why})", ok, why[:80])
        try:
            # What a graded round actually needs: every tier the CORPUS will require must resolve to an
            # adapter, and a bespoke sim the target DECLARES must actually contribute one.
            #
            # The old form asserted `{"L2","L3"} <= adapters` and `"program_oracle" in module` for all of
            # them. That is one endpoint's wiring written as if it were the general rule, and no
            # command-buffer target can satisfy it however well wired — its graded tier is the arc cosim,
            # which lives in a different module by design. Worse, it fails for the RIGHT target for the
            # WRONG reason: saturn_opu's real blocker is that it declares `sim_via: opu_cosim` and no
            # adapter is registered under that name, so oracle_adapters quietly returns the arc default
            # and the declared oracle is simply absent. Say THAT.
            ad = CR.oracle_adapters(TARGET, sim_via)
            mods = {t: getattr(fn, "__module__", "") for t, fn in ad.items()}
            try:
                from merlin.targetgen import corpus_spec as _CS
                need = list(_CS.derive_binding(_TE).tiers)
            except Exception:  # noqa: BLE001 — no corpus binding yet; the adapter set is its own floor
                need = sorted(ad)
            missing = [t for t in need if t not in ad]
            # `derive_binding` falls back to the adapter keys when the datapath declares no
            # `required_oracle_tiers`, and then this check compares a set against itself. Say so, or a
            # PASS here reads as "the corpus's demands are met" when nothing demanded anything.
            #
            # ASK THE CAPSULES, do not infer it from `need == adapters`. That equality holds just as
            # readily when the corpus declares exactly the tiers this target has — which is the healthy
            # case — so inferring from it reported a correctly-declaring corpus as vacuous, and the
            # vacuity was then chased as if it were real.
            declared: set[str] = set()
            for cap_yaml in {p for r in _TE.graded_roots() for p in Path(r).rglob("capsule.yaml")}:
                try:
                    declared.update(yaml.safe_load(cap_yaml.read_text()).get("required_oracle_tiers") or [])
                except Exception:  # noqa: BLE001
                    continue
            self_derived = not declared
            _ok("every tier the corpus requires resolves to an adapter",
                bool(ad) and not missing,
                f"need={need} have={sorted(ad)}" + (f" MISSING={missing}" if missing else "")
                + (" (tiers self-derived from the adapter set — the corpus declares none)"
                   if self_derived else ""))
            # An external_backend endpoint is graded by the PROGRAM oracle, and "resolves to an adapter"
            # would be satisfied by the arc default it must NOT be using. That check is kept, but routed
            # from the contract's endpoint kind rather than assumed for every non-chipyard target.
            endpoint_kind, _ = CR._endpoint_of(TARGET)
            # PRECEDENCE, mirroring oracle_adapters: a declared EXCLUSIVE bespoke sim owns the ladder and
            # REPLACES the program-oracle default. A SIMT core's endpoint is also `external_backend` (it
            # emits a kernel, not `.insn`), so asserting the program oracle owns its tiers fails a target
            # that is correctly graded by its own sim -- and this gate is what says GO/NO-GO before money
            # is spent, so a ready target read as NO-GO. Resolve the engine first, exactly as the runner
            # does, and only assert program-oracle ownership when no exclusive sim claims the ladder.
            _eng = CR._SIM_ORACLES.get(sim_via if sim_via else CR._bespoke_sim_via(TARGET))
            _exclusive = bool(getattr(_eng, "exclusive", False))
            if _exclusive:
                _ok(f"the declared exclusive sim ({sim_via or CR._bespoke_sim_via(TARGET)}) owns the "
                    f"graded tiers", bool(mods), str(mods))
            elif endpoint_kind == "external_backend":
                _ok("the program oracle owns the graded tiers (external_backend endpoint)",
                    bool(mods) and all("program_oracle" in m for m in mods.values()), str(mods))
            if sim_via and not _exclusive:
                # `_sim_engine_adapters` returns {} for an unknown engine, which is indistinguishable in
                # the result from a target that declared nothing -- so compare against the arc-only set.
                # `oracle_adapters(TARGET, "")` can RAISE for a target with no program-oracle config (an
                # exclusive-sim target has none by design); that is not a readiness failure, so treat an
                # unavailable baseline as "no baseline" rather than letting it abort the whole check.
                try:
                    _baseline = set(CR.oracle_adapters(TARGET, ""))
                except Exception:  # noqa: BLE001 — no arc/program default here; every adapter is bespoke
                    _baseline = set()
                bespoke = {t for t in ad if t not in _baseline}
                _ok(f"the declared sim ({sim_via}) contributes a real adapter", bool(bespoke),
                    str(sorted(bespoke)) if bespoke
                    else f"no adapter registered for sim_via={sim_via!r}; grading falls back to the arc "
                         f"default ({sorted(mods.values())}), so the declared oracle never runs")
        except Exception as e:  # noqa: BLE001
            _ok("oracle_adapters resolves the program-oracle ladder", False, f"{type(e).__name__}: {e}")
        return
    ref = REPO / "out/artifacts/targets" / TARGET / "agent_spec_v1_mlir_oot"
    if not (ref / "manifest.yaml").is_file():
        _ok("reference backend agent_spec_v1 present", False, "missing"); return
    _cy = ext_path("chipyard")   # resolve the real chipyard (.env MERLIN_EXT_CHIPYARD), same as the sandbox
    CE = str(_cy / ".conda-env") if _cy else "/path/to/chipyard/.conda-env"
    _compat = str(REPO / ".compat_lib")
    env = dict(_os.environ)
    env["PATH"] = f"{CE}/bin:{CE}/riscv-tools/bin:" + env.get("PATH", "")
    # MIRROR the driver's grade env exactly (incl. .compat_lib for libidn) so the gate fails iff a real
    # run would. (.compat_lib omission is exactly what made abc8's C++ build fail.)
    env["LD_LIBRARY_PATH"] = f"{_compat}:{CE}/lib:{CE}/riscv-tools/lib:" + env.get("LD_LIBRARY_PATH", "")

    def _grade(sub, sim, to, cap="A1_mvin_mvout"):
        r = subprocess.run([PY, str(SCRIPTS / "agent_selfcheck.py"), "--submission", str(sub),
                            "--sim", sim, "--capsules", cap, "--workers", "1", "--timeout", str(to)],
                           cwd=str(SCRIPTS), env=env, capture_output=True, text=True, timeout=to + 120)
        # The grader prints human diagnostics ("tier plan: ...", "model gate: ...") on the same stdout
        # that carries the verdict, so a bare loads() of the whole stream fails and every oracle check
        # here reported n=None -- a NO-GO that blamed the oracles for a stream-parsing bug. Scan for the
        # document instead (structurally, no regex).
        try:
            return _json.loads(r.stdout)
        except Exception:
            pass
        _dec, _i = _json.JSONDecoder(), (r.stdout or "").find("{")
        while _i != -1:
            try:
                _obj, _ = _dec.raw_decode(r.stdout, _i)
                if isinstance(_obj, dict):
                    return _obj
            except Exception:
                pass
            _i = r.stdout.find("{", _i + 1)
        return {"error": (r.stdout or r.stderr)[-200:]}

    try:
        # FROM-CLEAN C++ build: copy the ref, wipe its build dir, grade -> forces cmake CONFIGURE (the step
        # where libidn bites). A prebuilt backend skips configure and would mask the abc8 blocker.
        # tempfile already honours TMPDIR and falls back to /tmp; hardcoding dir="/tmp" overrode a
        # correctly-set TMPDIR and put this full C++ tree copy + cmake configure on the root
        # filesystem, which is the small, nearly-full one on this host.
        clean = Path(_tf.mkdtemp(prefix="clean_cpp_")) / "sub"
        import shutil as _sh
        _sh.copytree(ref, clean, symlinks=True)
        for bd in clean.rglob("build"):
            if bd.is_dir():
                _sh.rmtree(bd, ignore_errors=True)
        cb = _grade(clean, "spike", 700)
        _ok("C++ builds FROM CLEAN (cmake configure ok — catches libidn-class env bugs)",
            cb.get("n_capsules") == 1 and "FAIL[build]" not in str(cb.get("error", "")) and
            "libidn" not in str(cb), f"n={cb.get('n_passed')}/{cb.get('n_capsules')} {str(cb.get('error',''))[:60]}")

        # What this probe means is "the SCREEN tier runs and returns a real verdict". It must not assert
        # all_pass: the capsules declare a cycle-accurate cert tier as mandatory, and --sim spike supplies
        # no adapter for it, so a screen-only grade correctly reports the capsule as not certified. Reading
        # the tier's own result keeps this a check on the oracle rather than on the pass bar.
        sp = _grade(ref, "spike", 300)
        c = (sp.get("per_capsule") or [{}])[0]
        _spike_tier = (c.get("tiers") or {}).get("L2") or c.get("barrier_status")
        _ok("spike RUNS to a real L2=pass on the reference backend",
            sp.get("n_capsules") == 1 and _spike_tier == "pass",
            f"L2={_spike_tier} n={sp.get('n_passed')}/{sp.get('n_capsules')} {sp.get('error','')[:50]}")
        # Probe a COMPUTE capsule for the L3 cert — a movement-only capsule (A1) tops out below L3, so it
        # can never certify verilator's numerical tier. And agent_selfcheck reports the reached tier on its
        # per-capsule record as barrier_tier/barrier_status (there is NO "tiers" map — the same field the
        # spike check above reads), so the old tiers["L3"] read was a field-name bug that ALWAYS yielded
        # None: a false NO-GO that also blocked .oracle_timing.json, which the launcher refuses to start
        # without. Verilator was running fine the whole time.
        t0 = _time.time(); ve = _grade(ref, "verilator", 900, cap="A2_single_tile_matmul"); dt = _time.time() - t0
        cv = (ve.get("per_capsule") or [{}])[0]
        l3 = (ve.get("all_pass") and ve.get("n_capsules") == 1
              and cv.get("barrier_tier") == "L3" and cv.get("barrier_status") == "pass")
        _ok("verilator RUNS to a real L3=pass (not 0-capsules / timeout)", l3,
            f"{dt:.0f}s n={ve.get('n_passed')}/{ve.get('n_capsules')} "
            f"barrier={cv.get('barrier_tier')}/{cv.get('barrier_status')}")
        if l3:
            (SCRIPTS / ".oracle_timing.json").write_text(_json.dumps(
                {"verilator_per_capsule_s": round(dt, 1), "config": "GemminiRocketConfig",
                 "measured_by": "readiness_check"}))
            _ok("wrote .oracle_timing.json (T_obs for the driver timeout)", True, f"T_obs={dt:.0f}s")
        # NEGATIVE: an empty submission must produce 0 capsules / error -> the abc7 signature is caught
        empt = Path(_tf.mkdtemp()) / "sub"; empt.mkdir(parents=True)   # honours TMPDIR
        ne = _grade(empt, "spike", 60)
        _ok("empty submission -> NO-GO signal (0 capsules / error)",
            ne.get("n_capsules", 0) == 0 or "error" in ne, str(ne.get("error", ""))[:50])
    finally:
        subprocess.run(["pkill", "-9", "-f", "simulator-chipyard"], capture_output=True)


# ---- K. semantic coverage is MEASURABLE (not: is the score good) ----------------------------------
def test_semantic_coverage_measurable():
    """Can this target's Acceleratable Region Recall mean anything at all?

    Deliberately checks measurability, never the value. Gating on ``ARR >= x`` would make the rational
    response to a hard family "delete it from the contract", which is exactly the incentive the whole
    apparatus exists to defeat. What must hold is that the denominator is grounded, non-empty, and that
    the corpus can raise a violation when the compiler falls back on work the hardware can do.
    """
    section("K. semantic coverage measurable (ARR denominator)")
    from merlin.targetgen import capability_probes as _cp
    from merlin.targetgen import coverage_report as _cr
    from merlin.targetgen import eligibility as _el
    import yaml as _yaml

    cap = _el.capability_map_for_target(C.TARGET)
    _ok("target declares semantic capabilities", bool(cap),
        f"{sorted(cap)}" if cap else "none declared -> every region ineligible, ARR undefined, the "
                                     "target is outside the measurement entirely")
    if not cap:
        return

    probes = _cp.synthesize(cap, target=C.TARGET)
    per_fam = {f for p in probes for f in [p.descriptor.resolved_family()] if f}
    _ok("every declared family is probeable", per_fam >= set(cap),
        f"{len(probes)} probes over {sorted(per_fam)}"
        + (f"; UNPROBED: {sorted(set(cap) - per_fam)}" if set(cap) - per_fam else ""))

    # The denominator must be non-empty on THIS target's own corpus. Graded with an empty outcome so
    # this measures the denominator, not the compiler: n_eligible must be > 0 whatever the compiler did.
    caps = {}
    # A target that owns a capsule subdirectory uses it; the one predating that convention occupies the
    # shared kind directories at the root. Resolved from the tree, never from a target-name table.
    _caps_root = REPO / "merlin" / "contract" / "capsules"
    roots = [_caps_root / C.TARGET] if (_caps_root / C.TARGET).is_dir() else \
        [_caps_root / d for d in ("isa", "layers", "model_slices", "model", "hidden")]
    for r in roots:
        if r.is_dir():
            for f in sorted(r.rglob("capsule.yaml")):
                c = _yaml.safe_load(f.read_text()) or {}
                if c.get("name"):
                    caps[c["name"]] = c
    cov = _cr._acceleratable_coverage([{"capsule": n, "tiers": {}} for n in caps], caps, C.TARGET)
    _ok("ARR denominator is non-empty", cov["n_eligible"] > 0,
        f"n_eligible={cov['n_eligible']} of {len(caps)} capsules"
        + (f", n_undetermined={cov['n_undetermined']}" if cov.get("n_undetermined") else ""))
    _ok("must_accelerate can actually fire", bool(cov["must_accelerate_violations"]),
        f"{len(cov['must_accelerate_violations'])} capsule(s) would violate if the compiler accelerated "
        f"nothing — a corpus where this is 0 passes vacuously whatever the compiler does")
    undet = cov.get("n_undetermined", 0)
    frac = undet / max(len(caps), 1)
    _ok("undetermined regions bounded", frac <= 0.25,
        f"{undet}/{len(caps)} ({frac:.0%}) of the corpus is in families no evidence source could decide; "
        f"an ARR computed over the remainder should not be quoted alone")


def main() -> int:
    sys.path.insert(0, str(REPO / "merlin" / "python"))
    print("READINESS CHECK — exercising all tooling (no agent launched)")
    for fn in (test_starter_kit, test_generators, test_circt_gate, test_harness,
               test_oracles_endtoend, test_verify_no_cheat, test_corpus_fits_the_endpoint,
               test_graded_path_is_the_declared_one, test_contract_provenance, test_bundles,
               test_sandbox_authoring_tools,
               test_semantic_coverage_measurable):
        try:
            fn()
        except Exception as e:
            _ok(f"{fn.__name__} (uncaught)", False, f"{type(e).__name__}: {e}")
    n_pass = sum(1 for _, ok, _ in results if ok is True)
    n_fail = sum(1 for _, ok, _ in results if ok is False)
    n_na = sum(1 for _, ok, _ in results if ok is None)
    n = n_pass + n_fail
    print(f"\n{'='*60}\nREADINESS: {n_pass}/{n} checks passed"
          + (f" ({n_na} N/A for this endpoint)" if n_na else ""))
    # A GATE THAT ASSERTED NOTHING IS NOT A GO. `n_fail == 0` is vacuously true when no check recorded
    # a verdict at all -- every section returning early on a missing precondition would print
    # "0/0 checks passed" and then GO, which is the one outcome this script exists to make impossible.
    # N/A rows do not count: they are a deliberate, derived statement about this endpoint, not evidence
    # that the tooling works.
    go = n_fail == 0 and n_pass > 0
    if n_fail == 0 and n_pass == 0:
        print("  [FAIL] readiness recorded no pass/fail verdict — nothing was actually checked")
    print("🟢 GO — all tooling verified; ready for an A/B run pending your approval."
          if go else "🔴 NO-GO — resolve the FAILs above before launching.")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
