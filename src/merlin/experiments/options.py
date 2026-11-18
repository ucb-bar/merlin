"""Command-line arguments for experiment, corpus and observation workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from merlin.experiments.definition import STAGE_ENGINES
from merlin.experiments.definition import catalog as catalog


def parse_arguments(
    *,
    raw,
):
    from merlin.experiments.cli import _dashboard_arguments

    parser = argparse.ArgumentParser(prog="merlin experiment", description=__doc__)
    parser.add_argument("--catalog", type=Path, help="catalog YAML; paths inside it are relative to that file")
    commands = parser.add_subparsers(dest="verb", required=True)
    commands.add_parser("list", help="list the versioned experiment catalog")
    commands.add_parser("export", help="export an existing phase champion and inspect preserved backend metadata")
    commands.add_parser("model-observe", help="observe every built-model output against an explicit reference")
    commands.add_parser("chain-observe", help="observe every recurrent output against an explicit reference collection")
    commands.add_parser(
        "instruction-admission", help="check or explicitly prepare original linked-instruction admission"
    )
    commands.add_parser("levels", help="show backend generation experiment levels and their stable bundle-arm ids")
    stored = commands.add_parser("runs", help="discover stored phase orchestrations (read-only)")
    stored.add_argument("--root", type=Path, help="run root; defaults to the configured out/runs")
    stored.add_argument("--target", help="filter by exact target identity")
    stored.add_argument("--experiment", help="filter by exact experiment identity")
    for verb in ("inspect", "preflight", "run"):
        child = commands.add_parser(verb)
        child.add_argument(
            "spec",
            help="definition path or catalog id"
            + ("; with --group, a measured job directory or a package directory" if verb == "inspect" else ""),
        )
        child.add_argument(
            "--workflow",
            choices=(*STAGE_ENGINES, "all"),
            help="test generation or combined correctness/performance backend generation; default: all declared stages",
        )
        child.add_argument("--phase", choices=("0", "1", "2", "all"), help=argparse.SUPPRESS)
        child.add_argument("--run-dir", type=Path, help="explicit output; otherwise use the configured run root")
        child.add_argument(
            "--corpus-seal", type=Path, help="reviewed test generation release seal for backend generation"
        )
        child.add_argument("--bundle-manifest", type=Path, help="reviewed replacement backend generation input bundle")
        child.add_argument(
            "--backend-joint-campaign", type=Path, help="reviewed combined correctness/performance campaign policy"
        )
        child.add_argument(
            "--backend-oracle-timing", type=Path, help="reviewed timing evidence for the selected backend run"
        )
        child.add_argument(
            "--backend-driver",
            dest="phase1_driver",
            help="backend generation agent driver; frozen with the selected run",
        )
        child.add_argument(
            "--backend-model", dest="phase1_model", help="backend generation model; frozen with the selected run"
        )
        child.add_argument(
            "--backend-effort",
            dest="phase1_effort",
            help="backend generation reasoning effort; frozen with the selected run",
        )
        child.add_argument(
            "--backend-provider",
            dest="phase1_provider",
            help="backend generation provider; frozen with the selected run",
        )
        child.add_argument(
            "--testgen-conformance-spec",
            dest="phase0_conformance_spec",
            type=Path,
            help="new reviewed test generation requirement (select with synth profile)",
        )
        child.add_argument(
            "--testgen-synth-profile",
            dest="phase0_synth_profile",
            type=Path,
            help="new synthesized test generation profile (select with requirement)",
        )
        child.add_argument(
            "--testgen-generation-scope", dest="phase0_generation_scope", choices=("full", "phase1-functional")
        )
        child.add_argument(
            "--testgen-hidden-profile",
            dest="phase0_hidden_profile",
            type=Path,
            help="operator-owned private test generation profile; never put it in examples",
        )
        child.add_argument(
            "--testgen-component-coverage",
            dest="phase0_component_coverage",
            type=Path,
            help="explicit reviewed private independent component coverage plan",
        )
        child.add_argument(
            "--testgen-rtl-facts",
            dest="phase0_rtl_facts",
            type=Path,
            help="select exact extracted facts for a new test generation run",
        )
        child.add_argument(
            "--testgen-rtl-source-relocations",
            dest="phase0_rtl_source_relocations",
            type=Path,
            help="explicit byte-identical recorded RTL source custody mapping; grants no new extraction or audit",
        )
        child.add_argument(
            "--testgen-capability-contract",
            dest="phase0_capability_contract",
            type=Path,
            help="select exact same-target capability contract for a new test generation run",
        )
        child.add_argument(
            "--testgen-evidence-mode",
            dest="phase0_evidence_mode",
            choices=("diagnostic", "verified"),
            help="diagnostic preserves unknowns; verified refuses unresolved required evidence",
        )
        child.add_argument(
            "--testgen-m2m-root",
            dest="phase0_m2m_root",
            type=Path,
            help="explicit Model2MLIR source root for diagnostic capture",
        )
        child.add_argument(
            "--testgen-capture-timeout-seconds",
            dest="phase0_capture_timeout_seconds",
            type=int,
            help="sandbox timeout (120..14400 s) frozen into the plan for every sealed generation-time "
            "capture; default keeps the historical fixed 120 s",
        )
        child.add_argument(
            "--testgen-m2m-python",
            dest="phase0_m2m_python",
            type=Path,
            help="explicit Model2MLIR venv Python for diagnostic capture",
        )
        child.add_argument(
            "--testgen-capture-runtime-reuse",
            dest="phase0_capture_runtime_reuse",
            type=Path,
            help="exact read-only runtime selection JSON; independently checked against the selected capture runtime",
        )
        child.add_argument(
            "--testgen-bwrap",
            dest="phase0_bwrap",
            type=Path,
            help="absolute bubblewrap binary frozen into the plan for every sealed generation-time capture; "
            "default resolves the system bwrap",
        )
        if verb == "inspect":
            from merlin.experiments.inspection import group_inspect

            group_inspect.configure_parser(child)
    status = commands.add_parser(
        "status", help="an orchestration's phases, or a whole-model measured run's status from its records"
    )
    status.add_argument("run_dir", type=Path)
    status.add_argument(
        "--stall-hours", type=float, help="a measured run with no candidate measured for this long is STALLED"
    )
    commands.add_parser(
        "measured",
        help="the whole-model measured mode's own commands: `merlin experiment measured --help`",
        add_help=False,
    )
    commands.add_parser(
        "cell",
        help="a cell run: `cell prepare <loop run> --cell ID`, `cell launch <run> --profile P`, `cell status`",
        add_help=False,
    )
    commands.add_parser(
        "census",
        help="exact censuses of one observed execution: `census locality|boundaries|boundary-domain`",
        add_help=False,
    )
    commands.add_parser(
        "study",
        help="a comparison study, read-only: `study board` (register vs run matrices), `study baseline`",
        add_help=False,
    )
    stop = commands.add_parser(
        "stop", help="ask a whole-model measured run to stop at its next session boundary (signals nothing)"
    )
    stop.add_argument("run_dir", type=Path, help="the measured run, or an orchestration run that points at one")
    stop.add_argument("--why", required=True, help="recorded with the request and as the run's stop reason")
    lineage_parser = commands.add_parser(
        "lineage", help="read frozen phase inputs and handoffs without executing engines"
    )
    lineage_parser.add_argument("run_dir", type=Path, nargs="?")
    lineage_parser.add_argument(
        "--target", help="print the target's index (phase-0 releases, frozen compilers, champions) instead"
    )
    index = commands.add_parser(
        "index", help="regenerate out/artifacts/targets/<target>/INDEX.yaml from existing records"
    )
    index.add_argument("target")
    index.add_argument("--check", action="store_true", help="exit 1 when the written index is stale; write nothing")
    from merlin.experiments.tracking.readers.records import DEFAULT_STALL_HOURS

    dashboard = commands.add_parser(
        "dashboard",
        help="write one self-contained HTML view of a run or a target, read from existing records only",
    )
    dashboard.add_argument("run_dir", type=Path, nargs="?", help="a run directory (orchestration, phase 1 or 2)")
    dashboard.add_argument("--target", help="every run of this target across phases, with champion lineage")
    dashboard.add_argument(
        "--out", type=Path, help="HTML file; defaults to out/artifacts/experiments/<target>/dashboard/<run>.html"
    )
    dashboard.add_argument("--open", action="store_true", help="also open the written page in a browser")
    _dashboard_arguments(dashboard)
    watch = commands.add_parser("watch", help="live terminal view of a run's records; refreshes until Ctrl-C")
    watch.add_argument("run_dir", type=Path, nargs="?")
    watch.add_argument("--task", type=Path, help="unified animated console following a managed campaign task")
    watch.add_argument(
        "--operator-private", action="store_true", help="include recorded capsule, agent and performance details"
    )
    watch.add_argument("--window", type=int, default=120, help="--task core timeline window in seconds")
    watch.add_argument("--width", type=int, help="--task display columns")
    watch.add_argument("--height", type=int, help="--task display rows")
    watch.add_argument("--interval", type=float, help="refresh seconds (run summary: 30; task console: 1)")
    watch.add_argument("--once", action="store_true", help="print once and exit")
    watch.add_argument("--no-color", action="store_true", help="plain text even on a terminal")
    watch.add_argument(
        "--legacy", action="store_true", help="--task: keep the single-screen view instead of the command console"
    )
    console = commands.add_parser(
        "console",
        help="experiment command console: every experiment on this machine, live or stored, in one view",
        description="Track, inspect and debug any experiment on this machine from its records (read-only).",
        epilog="Inside the console press ? for keys: b browse runs, L running experiment, Enter inspect, "
        "z zoom, q quit. Guide: docs/guides/experiment_console.md",
    )
    console.add_argument("--task", type=Path, action="append", help="limit discovery to this task folder (repeatable)")
    console.add_argument("--run-dir", type=Path, action="append", help="also show this run or record directory")
    console.add_argument("--experiment", help="open this experiment first (run id or run directory)")
    console.add_argument("--tasks-root", type=Path, help="directory holding the task folders (default: derived)")
    console.add_argument(
        "--operator-private", action="store_true", help="show capsule names, paths and golden references"
    )
    console.add_argument("--interval", type=float, help="fast refresh seconds for a running experiment (default 1)")
    console.add_argument("--once", action="store_true", help="print the selected experiment as text and exit")
    console.add_argument("--list", action="store_true", help="print the discovered experiments and exit")
    console.add_argument("--json", action="store_true", help="with --once: print the snapshot as JSON")
    console.add_argument(
        "--build", action="store_true", help="compile the Bubble Tea frontend (needs Go) before starting"
    )
    console.add_argument("--plain", action="store_true", help="never start the frontend; refresh plain text instead")
    console.add_argument("--width", type=int, help="text width for --once/--plain (default: terminal)")
    console.add_argument("--no-color", action="store_true", help="plain text even on a terminal")
    for view in (dashboard, watch):
        view.add_argument(
            "--store", type=Path, help="phase-2 measurement store when the run records none (store_roots.screen)"
        )
        view.add_argument(
            "--stall-hours",
            type=float,
            default=DEFAULT_STALL_HOURS,
            help="hours without a measured candidate (or grade) before a run is STALLED",
        )
    child = commands.add_parser("resume")
    child.add_argument("run_dir", type=Path)
    corpus = commands.add_parser(
        "corpus", help="derive capsule groups, inspect run coverage, or prepare and review a corpus release"
    )
    operations = corpus.add_subparsers(dest="operation", required=True)
    derive = operations.add_parser("derive", help="deterministic requirements and complete census; no agent execution")
    derive.add_argument("definition", help="explicit experiment definition or catalog id")
    derive.add_argument("--application-capture", action="append", required=True, metavar="LABEL=PATH")
    derive.add_argument(
        "--application-capture-selection",
        action="append",
        default=[],
        metavar="LABEL=PATH@SHA256",
        help="pre-execution selection for each selected capture; omitted legacy captures remain diagnostic",
    )
    derive.add_argument(
        "--application-quant-policy",
        action="append",
        default=[],
        metavar="LABEL=PATH@SHA256",
        help="independently selected policy bytes for each externally quantized capture",
    )
    derive.add_argument(
        "--performance-capture",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "independent declared performance_applications capture; "
            "explicit connected policies also derive public sources"
        ),
    )
    derive.add_argument(
        "--performance-capture-selection",
        action="append",
        default=[],
        metavar="LABEL=PATH@SHA256",
        help="pre-execution selection for each performance-scale capture (all or none)",
    )
    derive.add_argument(
        "--heldout-layer-shapes",
        type=Path,
        help="OPERATOR-PRIVATE merlin.heldout_layer_shapes.v1 file (owner-only, outside the repository): "
        "refuse any derived form member whose contraction equals a held-out network layer",
    )
    derive.add_argument(
        "--native-qualification",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="optional exact generated native-host qualification receipt; never grants RVV support",
    )
    derive.add_argument(
        "--rtl-facts", type=Path, required=True, help="exact extraction artifact; never re-extract implicitly"
    )
    derive.add_argument("--output", type=Path, required=True, help="new immutable artifact root")
    capture = operations.add_parser("capture", help="preselect and issue one fresh sealed CPU capture")
    capture_ops = capture.add_subparsers(dest="capture_operation", required=True)
    select_capture = capture_ops.add_parser("select", help="freeze source/runtime/tool bytes before capture")
    select_capture.add_argument("--m2m-root", type=Path, required=True)
    select_capture.add_argument("--workload-root", type=Path, required=True)
    select_capture.add_argument("--venv", type=Path, required=True)
    select_capture.add_argument("--dtype", choices=("fp32", "int8"), default="fp32")
    select_capture.add_argument("--recipe", type=Path)
    select_capture.add_argument("--run-dir", type=Path, required=True)
    select_capture.add_argument("--output", type=Path, required=True, help="fresh owner-only selection directory")
    select_capture.add_argument("--bwrap", type=Path)
    select_capture.add_argument(
        "--checkpoint",
        metavar="GUEST_MEMBER=PATH",
        help="explicit checkpoint file or tree (v2 full-model selection); the loader reads it at GUEST_MEMBER "
        "under its read-only input root",
    )
    select_capture.add_argument(
        "--extra-input",
        action="append",
        default=[],
        metavar="GUEST_MEMBER=PATH",
        help="additional selected input file or tree (v2 only; repeatable)",
    )
    select_capture.add_argument(
        "--loader-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="declared loader environment value (v2 only; repeatable); every literal loader read must be declared",
    )
    select_capture.add_argument(
        "--loader-env-unset",
        action="append",
        default=[],
        metavar="NAME",
        help="loader environment read selected as deliberately absent (v2 only; repeatable)",
    )
    select_capture.add_argument(
        "--cpu-threads",
        type=int,
        help="intra-op CPU threads for the capture worker; the reference bytes depend on the count, so it "
        "is selected into the plan, the command and the replay (default: one thread, recorded nowhere so "
        "old selections keep their bytes)",
    )
    select_capture.add_argument(
        "--stage-fp32",
        action="store_true",
        help="stage the exported program to FP32 before capture (and before an int8 recipe quantizes it), "
        "with the worker's exact-source precision audit; needed for models that compute in bf16/fp16",
    )
    select_capture.add_argument(
        "--execution-timeout-seconds",
        type=int,
        help="bounded sandbox execution time, used by issue and by the attestation replay: a checkpoint "
        "selection requires 120..43200; a checkpoint-free selection may select 120..14400 (default: the "
        "historical fixed 120 s, recorded nowhere so old selections keep their bytes)",
    )
    issue_capture = capture_ops.add_parser("issue", help="capture only from an exact preselected identity")
    issue_capture.add_argument("--selection", type=Path, required=True)
    issue_capture.add_argument("--expected-sha256", required=True)
    issue_capture.add_argument(
        "--attestation-output",
        type=Path,
        help="after issuing, replay the capture and write its sealed execution attestation here "
        "(v3 for a v2 checkpoint selection) as a fresh owner-only file",
    )
    issue_capture.add_argument(
        "--concurrent-replay",
        action="store_true",
        help="with --attestation-output: run the attestation replay beside issuance instead of after it, "
        "the issue on the lower half of this process's CPUs and the replay on the upper half; the "
        "replay is the same fresh sandboxed execution with the same byte comparison",
    )
    attest_capture = capture_ops.add_parser(
        "attest", help="replay an issued preselected capture and write its sealed execution attestation"
    )
    attest_capture.add_argument("--selection", type=Path, required=True)
    attest_capture.add_argument("--expected-sha256", required=True)
    attest_capture.add_argument("--output", type=Path, required=True, help="fresh attestation JSON path")
    demand = operations.add_parser(
        "performance-demand",
        help="derive performance cases and per-class holdouts from captured programs' layer inventories",
    )
    demand.add_argument(
        "--program",
        action="append",
        required=True,
        metavar="MODEL/PROGRAM=TRACE@SHA256",
        help="pinned frontend trace of one captured program (value-free projection only)",
    )
    demand.add_argument("--policy", type=Path, required=True, help="merlin.phase0.performance_demand_policy.v1")
    demand.add_argument("--rtl-facts", type=Path, required=True, help="exact extraction artifact (tile edge)")
    demand.add_argument("--template", type=Path, required=True, help="reviewed raw int8-to-int32 contract")
    demand.add_argument("--existing-suite", type=Path, help="staged scalability cases already public")
    demand.add_argument("--output", type=Path, required=True, help="fresh directory; holdouts are owner-only")
    epoch = operations.add_parser(
        "performance-epoch",
        help="apply case redundancy to a release with a demand plan; write the next suite selections",
    )
    epoch.add_argument("--previous", type=Path, required=True, help="previous prepared release root")
    epoch.add_argument("--plan", type=Path, required=True, help="performance-demand plan.json")
    epoch.add_argument("--sources", type=Path, required=True, help="performance-demand sources/ directory")
    epoch.add_argument("--evidence", type=Path, required=True, help="exported test generation evidence root")
    epoch.add_argument("--chain-plan", type=Path, help="chain-demand plan.json (with --chain-sources)")
    epoch.add_argument("--chain-sources", type=Path, help="chain-demand sources/ directory")
    epoch.add_argument("--target", required=True)
    epoch.add_argument("--output", type=Path, required=True, help="fresh directory")
    chains = operations.add_parser(
        "chain-demand",
        help="derive connected chain cases (fusion, shared inputs) from captured programs' graph structure",
    )
    chains.add_argument(
        "--program",
        action="append",
        required=True,
        metavar="MODEL/PROGRAM=TRACE@SHA256",
        help="pinned frontend trace of one captured program (value-free structure only)",
    )
    chains.add_argument("--policy", type=Path, required=True, help="merlin.phase0.chain_demand_policy.v1")
    chains.add_argument("--rtl-facts", type=Path, required=True, help="exact extraction artifact (tile edge)")
    chains.add_argument("--descriptor", type=Path, required=True, help="target descriptor (claim models, binding)")
    chains.add_argument("--recipe", type=Path, required=True, help="test generation recipe (datapath)")
    chains.add_argument("--performance-template", type=Path, required=True, help="shared performance template")
    chains.add_argument("--software-spec", type=Path, required=True, help="selected software spec")
    chains.add_argument("--output", type=Path, required=True, help="fresh directory")
    heldout = operations.add_parser(
        "heldout-layers",
        help="OPERATOR: write the private held-out layer-shape file from per-program inventories",
    )
    heldout.add_argument("--inventory", type=Path, action="append", required=True)
    heldout.add_argument("--output", type=Path, required=True, help="fresh owner-only file outside the repository")
    staging = operations.add_parser(
        "stage-workloads",
        help="write a fresh workload root (loader and optional profile) per declared roster label",
    )
    staging.add_argument("--roster", type=Path, required=True, help="merlin.phase0_workload_roster.v1 file")
    staging.add_argument("--output", type=Path, required=True, help="fresh capture-input directory")
    staging.add_argument(
        "--descriptor", type=Path, help="require the roster to stage exactly the descriptor's declared labels"
    )
    variants = operations.add_parser(
        "variants", help="derive bounded independent workload roots from selected RTL facts"
    )
    variants.add_argument("--facts", type=Path, required=True)
    variants.add_argument("--template", type=Path, required=True)
    variants.add_argument("--loader", type=Path, required=True)
    variants.add_argument("--output", type=Path, required=True, help="fresh generated capture-input directory")
    from merlin.development.targetgen.corpus import group_capsules

    groups = operations.add_parser(
        "groups",
        help="derive capsules from captured compute groups (does not seal or approve)",
        description=group_capsules.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_capsules.configure_parser(groups)
    compare = operations.add_parser("compare", help="compare public recipes from explicit definitions; prints JSON")
    compare.add_argument("definitions", nargs="+", help="definition paths or catalog ids")
    coverage = operations.add_parser("coverage", help="inspect public coverage of one completed test generation run")
    coverage.add_argument("run_dir", type=Path)
    coverage.add_argument("--spec", type=Path, required=True, help="explicit conformance requirement YAML")
    prepare = operations.add_parser("prepare")
    prepare.add_argument(
        "--with-level",
        action="append",
        choices=("EL5", "EL6"),
        default=[],
        help="also prepare an optional ISA-generation or interactive Lean level; never changes existing runs",
    )
    prepare.add_argument("--lean-toolchain-manifest", type=Path, help="explicit pinned Lean executable/import closure")
    from merlin.development.targetgen.authoring.tool_profile import CURRENT_COMMON_SUBSTRATE

    prepare.add_argument(
        "--common-substrate",
        choices=(CURRENT_COMMON_SUBSTRATE,),
        default=CURRENT_COMMON_SUBSTRATE,
        help="the current common substrate; retired substrates are refused",
    )
    prepare.add_argument("--python-source-root", type=Path, help="explicit reviewed current compiler SDK source root")
    prepare.add_argument("--public-scalability", type=Path, help="required separate Spike-only public suite selector")
    prepare.add_argument("--public-supplement", type=Path, help="pinned operator-requested public test selection JSON")
    prepare.add_argument("--host-index-bits", type=int, help="reviewed index width of the selected host ABI")
    prepare.add_argument("run_dir", type=Path)
    prepare.add_argument(
        "--output",
        type=Path,
        help="fresh release root; defaults to out/artifacts/protocols/<target>/phase0-<TS>-<sha7>",
    )
    corpus_mode = prepare.add_mutually_exclusive_group()
    prepare.set_defaults(generated_only=True)
    corpus_mode.add_argument(
        "--generated-only",
        action="store_true",
        help="use only selected Phase-0 public capsules (default); never read the historical corpus",
    )
    corpus_mode.add_argument(
        "--historical-overlay",
        dest="generated_only",
        action="store_false",
        help="explicitly retain a historical public corpus; held-out validation sources are refused",
    )
    prepare.add_argument(
        "--backend-policy-descriptor",
        dest="phase1_policy_descriptor",
        type=Path,
        help="explicit Phase-1 gate policy; all other descriptor fields must match the frozen Phase-0 source",
    )
    prepare.add_argument(
        "--private-baseline",
        type=Path,
        help="operator-owned hidden capsule category when it is absent from the public source checkout",
    )
    prepare.add_argument(
        "--retirements",
        type=Path,
        help="reviewed public baseline-generated members intentionally retired by this derivation",
    )
    operations.add_parser("inspect").add_argument("release", type=Path)
    seal = operations.add_parser("seal")
    seal.add_argument("release", type=Path)
    seal.add_argument("--expected-digest", required=True)
    seal.add_argument("--reviewed-by", required=True)
    seal.add_argument("--review-note", required=True)
    from merlin.experiments.corpus import evidence_custody

    evidence_custody.add_parser(operations)
    args = parser.parse_args(raw)

    return (
        args,
        derive,
        evidence_custody,
        group_capsules,
    )
