#!/usr/bin/env python3
# tools/compile.py
"""Backs `./merlin compile`: lowers `.mlir` / `.onnx` model inputs to target
artifacts (`.vmfb`, intermediate dumps) using flag bundles defined in
`models/<target>.yaml`.

Outputs land under `build/compiled_models/<model>/<target>/`.
"""

import argparse
import json
import pathlib
import sys

import yaml

import utils

# Helpers extracted to compile.iree_tools / compile.postprocess / compile.radiance
from compile.iree_tools import get_iree_tool, import_onnx
from compile.postprocess import zip_artifacts
from compile.radiance import compile_radiance_muon


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("input_path", help="Path to the model directory OR specific .mlir/.onnx file")
    parser.add_argument("--target", required=True, help="Target YAML config file name (e.g., spacemit_x60)")
    parser.add_argument(
        "--hw", help="Hardware sub-target from YAML (e.g., RVV, OPU). If omitted, uses default_hw from YAML."
    )
    parser.add_argument(
        "--quantized", action="store_true", help="Force quantized mode (auto-detected if .q. in filename)"
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Override output directory (default: build/compiled_models/<model>/<target>_<basename>/). "
            "If set, all generated files/artifacts are written under this directory."
        ),
    )

    # NEW: Build Directory Selector
    parser.add_argument(
        "--build-dir",
        default="host-vanilla-release",
        help=(
            "Which build directory to use for compiler tools "
            "(default: host-vanilla-release). If omitted and target YAML uses "
            "plugin_flags, compile.py auto-selects host-merlin-release."
        ),
    )
    parser.add_argument(
        "--compile-to",
        help=(
            "Stop compilation at the given phase (for example: global-optimization). "
            "When set, output is emitted as an intermediate MLIR file."
        ),
    )
    parser.add_argument(
        "--dump-compilation-phases-to",
        help=(
            "Directory for --dump-compilation-phases-to. "
            "If omitted and --dump-phases is set, defaults to <output_dir>/phases/."
        ),
    )
    parser.add_argument(
        "--iree-compile-arg",
        "--compilation-custom-arg",
        action="append",
        dest="iree_compile_arg",
        default=[],
        help=("Extra flag forwarded directly to iree-compile. " "Repeat for multiple flags."),
    )
    parser.add_argument(
        "--reuse-imported-mlir",
        action="store_true",
        help=(
            "Reuse an existing output MLIR instead of refreshing from explicit input files. "
            "By default, explicit input files are re-imported/re-copied."
        ),
    )

    # Tracy / Profiling
    parser.add_argument(
        "--tracy",
        action="store_true",
        help=(
            "Enable Tracy profiling flags: embed debug info, use system linking, "
            "and enable debug symbols in generated code. "
            "Equivalent to --iree-hal-executable-debug-level=3 "
            "--iree-llvmcpu-link-embedded=false --iree-llvmcpu-debug-symbols=true"
        ),
    )

    # Optional Artifacts
    parser.add_argument("--dump-artifacts", action="store_true", help="Dump executable sources, binaries, and configs")
    parser.add_argument("--dump-phases", action="store_true", help="Dump MLIR compilation phases")
    parser.add_argument("--dump-graph", action="store_true", help="Dump the flow dispatch graph (.dot)")
    parser.add_argument(
        "--qnn-partition",
        action="store_true",
        help=(
            "Run the QNN subgraph partitioner on the imported MLIR and "
            "emit a JSON dump of the per-island partition decision to "
            "<output_dir>/qnn_partition.json. Inspectable artifact for "
            "Phase 3b debugging; does not (yet) drive the final compile."
        ),
    )
    parser.add_argument(
        "--build-benchmarks", action="store_true", help="Recompile individual dispatch benchmarks and zip them"
    )
    parser.add_argument(
        "--qnn-preprocess-nhwc",
        action="store_true",
        help=(
            "Insert iree-preprocessing-convert-conv-to-channels-last "
            "before the input-conversion phase so NCHW convs become NHWC. "
            "Required for the nhwc_int8_conv recognizer to match YOLOv8's "
            "stem/trunk/head convs (NCHW-anchored convs only get the "
            "Transpose-wrapped recognizer which HTA + Adreno reject on "
            "QAIRT 2.45). Auto-on when --with-schedule references HTA or GPU."
        ),
    )

    # XPU-RT schedule directives (Part A: scheduler-driven dispatch control).
    parser.add_argument(
        "--with-schedule",
        help=(
            "Path to an XPU-RT schedule.json. Compiled with "
            "--iree-merlin-schedule-spec=<path> so DispatchCreation stamps "
            "stream.affinity (and split/grow/shard, when those land) per "
            "dispatch id."
        ),
    )
    parser.add_argument(
        "--with-feedback",
        help=(
            "Path to an XPU-RT feedback.json (the persisted form written by "
            "targetgen_mcp.ingest_xpurt_feedback). When set, compile.py "
            "logs the overlay summary, derives a model-level granularity "
            "disposition, and writes <output_dir>/feedback_applied.json so "
            "downstream tooling (target-specific compile scripts, "
            "tools/run_full_loop.py) can read it. Inert when omitted — "
            "compile behavior is unchanged. See docs/merlin_integration.md."
        ),
    )

    # Kernel embedding (Part B: KernelBlaster custom dispatches).
    parser.add_argument(
        "--kernels-dir",
        help=(
            "Directory containing a kernels manifest.json (e.g. "
            "models/compiled_models/<model>/<target>/kernels/). When set, "
            "compile.py precompiles each kernel to its target object, "
            "auto-generates a transform-dialect spec, and threads "
            "--iree-preprocessing-transform-spec-filename + "
            "--iree-hal-executable-object-search-path into iree-compile."
        ),
    )
    parser.add_argument(
        "--kernel-manifest",
        help="Explicit manifest path; overrides --kernels-dir/manifest.json.",
    )
    parser.add_argument(
        "--kernel-cache-dir",
        help=(
            "Where to write precompiled kernel objects + the generated "
            "transform spec. Defaults to <output_dir>/kernels_cache/."
        ),
    )
    parser.add_argument(
        "--no-kernel-embedding",
        action="store_true",
        help=(
            "Disable the kernel embedding pipeline even if a manifest is "
            "discoverable from --kernels-dir / YAML custom_kernels."
        ),
    )
    parser.add_argument(
        "--kernels-strict-coverage",
        action="store_true",
        help=(
            "After a kernel-embedded compile, fail with a non-zero exit if "
            "ANY linalg op in the input survived past the rewrite (i.e. "
            "fell through to IREE codegen). Use to verify that the manifest "
            "covers every op in the model. Implies --dump-phases."
        ),
    )


def main(args: argparse.Namespace) -> int:
    if args.build_benchmarks and args.compile_to:
        utils.eprint("❌ Error: --build-benchmarks is only supported for full VMFB compilation (no --compile-to).")
        return 1

    # Short-circuit for the radiance_muon target: kernel descriptor (yaml) ->
    # Jinja template -> kernel.cpp + data sidecar -> `merlin build --profile
    # radiance_muon`. No MLIR, no iree-compile.
    if args.target == "radiance_muon":
        return compile_radiance_muon(args)

    input_p = pathlib.Path(args.input_path).resolve()

    is_quantized = args.quantized
    quant_type = "int8"

    if input_p.is_file():
        model_dir = input_p.parent
        model_name = model_dir.name
        explicit_file = input_p

        parts = input_p.name.split(".")
        if "q" in parts:
            is_quantized = True
            q_idx = parts.index("q")
            if q_idx + 1 < len(parts):
                quant_type = parts[q_idx + 1].lower()

        basename = input_p.name.replace(".mlir", "").replace(".onnx", "")
    else:
        model_dir = input_p
        model_name = model_dir.name
        explicit_file = None
        suffix = f".q.{quant_type}" if is_quantized else ""
        basename = f"{model_name}{suffix}"

    config_path = utils.REPO_ROOT / "models" / f"{args.target}.yaml"
    if not config_path.exists():
        utils.eprint(f"❌ Error: Config file not found at {config_path}")
        return 1

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    plugin_flags = cfg.get("plugin_flags", [])
    effective_build_dir = args.build_dir
    if args.build_dir == "host-vanilla-release" and plugin_flags:
        # Plugin-backed targets are expected in merlin builds.
        effective_build_dir = "host-merlin-release"
        print(
            "  🔧 Target uses plugin flags; selecting build dir "
            f"'{effective_build_dir}' (override with --build-dir)."
        )

    hw_choice = args.hw
    if not hw_choice and "default_hw" in cfg:
        hw_choice = cfg["default_hw"]

    if "targets" in cfg:
        if not hw_choice:
            utils.eprint(f"❌ Error: {args.target}.yaml requires a --hw sub-target, but no default_hw is set.")
            return 1
        if hw_choice not in cfg["targets"]:
            utils.eprint(f"❌ Error: Unknown --hw '{hw_choice}'.")
            return 1

    hw_print = f" ({hw_choice})" if hw_choice else ""
    mode_msg = f"Quantized ({quant_type.upper()})" if is_quantized else "Float (FP32)"

    print("=" * 80)
    print(f"🚀 Processing Model: {model_name} | Target: {args.target}{hw_print} | Mode: {mode_msg}")
    print("=" * 80)

    hw_suffix = f"_{hw_choice}" if hw_choice else ""
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir).resolve()
    else:
        output_dir = utils.REPO_ROOT / "build" / "compiled_models" / model_name / f"{args.target}{hw_suffix}_{basename}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mlir_file = output_dir / f"{basename}.mlir"
    vmfb_file = output_dir / f"{basename}.vmfb"
    compile_output_file = vmfb_file
    if args.compile_to:
        compile_output_file = output_dir / f"{basename}_{args.compile_to.replace('-', '_')}.mlir"
    graph_out = output_dir / f"{basename}_dispatch_graph.dot"

    should_refresh_mlir = False
    if not mlir_file.exists():
        should_refresh_mlir = True
    elif explicit_file and not args.reuse_imported_mlir:
        should_refresh_mlir = True

    if should_refresh_mlir:
        if explicit_file:
            if explicit_file.suffix == ".onnx":
                import_onnx(explicit_file, mlir_file, effective_build_dir, args.dry_run)
            elif explicit_file.suffix == ".mlir":
                print(f"  📄 Using explicit MLIR file: {explicit_file}")
                if args.dry_run:
                    print(f"+ cp {explicit_file} {mlir_file}")
                else:
                    mlir_file.write_bytes(explicit_file.read_bytes())
            else:
                utils.eprint(f"❌ Error: Unsupported file type: {explicit_file}")
                return 1
        else:
            source_onnx = model_dir / f"{model_name}{'.q.' + quant_type if is_quantized else ''}.onnx"
            source_mlir = model_dir / f"{basename}.mlir"
            if source_onnx.exists():
                import_onnx(source_onnx, mlir_file, effective_build_dir, args.dry_run)
            elif source_mlir.exists():
                print(f"  📄 Found Source MLIR: {source_mlir}")
                if args.dry_run:
                    print(f"+ cp {source_mlir} {mlir_file}")
                else:
                    mlir_file.write_bytes(source_mlir.read_bytes())
            else:
                utils.eprint(f"❌ Error: Could not find ONNX or MLIR in {model_dir}")
                return 1
    else:
        print(f"  ♻️  Reusing imported MLIR: {mlir_file}")

    # Phase 3b — QNN subgraph partitioner side-channel dump. Reads the
    # imported MLIR, partitions it via `kernels/qnn/partition.py`,
    # and writes a JSON inventory of islands to <output_dir>/
    # qnn_partition.json. Doesn't drive the final compile (Phase 4
    # wires the routing decision; Phase 5 turns it into per-island
    # kernels) — purely an inspection artifact.
    if args.qnn_partition and not args.dry_run:
        partition_out = output_dir / "qnn_partition.json"
        try:
            sys.path.insert(0, str(utils.REPO_ROOT / "kernels" / "qnn"))
            from kernels.qnn.partition import parse_and_partition  # noqa: PLC0415

            print(f"  🧩 Running QNN partitioner on {mlir_file}")
            islands = parse_and_partition(mlir_file.read_text())
            import json as _json  # noqa: PLC0415

            payload = {
                "source_mlir": str(mlir_file),
                "island_count": len(islands),
                "islands": [
                    {
                        "name": isl.name,
                        "recognizer": isl.recognizer_name,
                        "target": isl.target,
                        "op_count": len(isl.op_names),
                        "op_names": list(isl.op_names),
                        "boundary_inputs": [
                            {
                                "ssa": bv.ssa_name,
                                "shape": list(bv.shape),
                                "dtype": bv.dtype,
                            }
                            for bv in isl.boundary_inputs
                        ],
                        "boundary_outputs": [
                            {
                                "ssa": bv.ssa_name,
                                "shape": list(bv.shape),
                                "dtype": bv.dtype,
                                "producing_op": bv.producing_op,
                            }
                            for bv in isl.boundary_outputs
                        ],
                    }
                    for isl in islands
                ],
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            partition_out.write_text(_json.dumps(payload, indent=2))
            from collections import Counter as _Counter  # noqa: PLC0415

            counts = _Counter(i.recognizer_name for i in islands)
            print(
                f"  🧩 Partition: {len(islands)} islands → " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            )
            print(f"  💾 Wrote {partition_out}")

            # Phase 4 — per-island routing decisions. Profile-driven
            # when `<output_dir>/qnn_per_island_profile.csv` exists
            # (drop it there via `tools/profile_per_island.py`);
            # heuristic otherwise.
            from qnn_route import (  # noqa: PLC0415
                load_profile_csv,
                route_islands,
                summarize,
            )

            profile = None
            profile_csv_path = output_dir / "qnn_per_island_profile.csv"
            if profile_csv_path.exists():
                profile = load_profile_csv(profile_csv_path)
                print(f"  📈 Loaded {len(profile)} profile points from {profile_csv_path}")
            decisions = route_islands(islands, profile=profile)
            target_counts = summarize(decisions)
            routing_out = output_dir / "qnn_routing.json"
            routing_payload = {
                "island_count": len(islands),
                "profile_source": (str(profile_csv_path) if profile is not None else None),
                "target_counts": target_counts,
                "decisions": [
                    {
                        "island_name": d.island_name,
                        "target": d.target,
                        "rule": d.rule,
                        "metric": d.metric,
                    }
                    for d in decisions
                ],
            }
            routing_out.write_text(_json.dumps(routing_payload, indent=2))
            print("  🎯 Routing: " + ", ".join(f"{k}={v}" for k, v in sorted(target_counts.items())))
            print(f"  💾 Wrote {routing_out}")
        except Exception as e:  # noqa: BLE001
            utils.eprint(f"  ⚠️  --qnn-partition failed: {e}")

    # Stack Flags (UPDATED KEYS TO MATCH YOUR YAML)
    static_flags = cfg.get("generic", []) + plugin_flags

    if "targets" in cfg and hw_choice:
        static_flags.extend(cfg["targets"][hw_choice])

    if is_quantized:
        print(f"  🧊 Applying {quant_type.upper()} quantized flag overrides...")
        quant_flags = cfg.get(f"quantized_{quant_type}", cfg.get("quantized", []))
        static_flags.extend(quant_flags)

    model_overrides = cfg.get("models", {}).get(model_name, [])
    if model_overrides:
        print(f"  🎯 Applying model-specific overrides for '{model_name}'...")
        static_flags.extend(model_overrides)

    # Tracy profiling flags.
    if args.tracy:
        print("  🔬 Applying Tracy profiling flags (debug-level=3, system linking, debug symbols)...")
        static_flags.extend(
            [
                "--iree-hal-executable-debug-level=3",
                "--iree-llvmcpu-link-embedded=false",
                "--iree-llvmcpu-debug-symbols=true",
            ]
        )

    dynamic_flags = []
    if args.dump_artifacts or args.build_benchmarks:
        dynamic_flags.extend(
            [
                f"--iree-hal-dump-executable-sources-to={output_dir}/sources/",
                f"--iree-hal-dump-executable-files-to={output_dir}/files/",
                f"--iree-hal-dump-executable-binaries-to={output_dir}/binaries/",
                f"--iree-hal-dump-executable-configurations-to={output_dir}/configs/",
                f"--iree-hal-dump-executable-benchmarks-to={output_dir}/benchmarks/",
            ]
        )

    dump_phases_dir = args.dump_compilation_phases_to
    if dump_phases_dir:
        dynamic_flags.append(f"--dump-compilation-phases-to={pathlib.Path(dump_phases_dir)}")
    elif args.dump_phases or args.kernels_strict_coverage:
        # Strict coverage requires phase 5 to audit the rewrite outcome.
        dynamic_flags.append(f"--dump-compilation-phases-to={output_dir}/phases/")

    if args.compile_to:
        dynamic_flags.append(f"--compile-to={args.compile_to}")

    if args.dump_graph:
        dynamic_flags.extend(
            ["--iree-flow-dump-dispatch-graph", f"--iree-flow-dump-dispatch-graph-output-file={graph_out}"]
        )

    # XPU-RT schedule spec passthrough (Part A).
    schedule_machines: list[str] = []
    if args.with_schedule:
        schedule_path = pathlib.Path(args.with_schedule).resolve()
        if not schedule_path.exists():
            utils.eprint(f"❌ schedule file not found: {schedule_path}")
            return 1
        print(f"  📋 Applying schedule spec: {schedule_path}")
        static_flags.append(f"--iree-merlin-schedule-spec={schedule_path}")
        # Read the machines list so the NHWC preprocess + multi-target
        # compile branches below can detect QNN routing.
        try:
            with schedule_path.open() as _sf:
                _sched = json.load(_sf)
                schedule_machines = list(_sched.get("machines", []))
        except (OSError, ValueError):
            schedule_machines = []

    # Multi-machine target-device emission (in-compiler-codegen path).
    #
    # When the schedule's machines list mixes 2+ distinct backends (e.g.
    # HTA+GPU, HTA+CPU, HTA+GPU+CPU), the YAML-injected single-device
    # flags are insufficient: every variant inherits the global
    # --iree-hal-qnn-backend, collapsing both HTA and GPU dispatches
    # onto whichever backend the global flag selected. We emit explicit
    # `--iree-hal-target-device='<sym>=#hal.device.target<...>'` per
    # machine, matching schedule-spec's auto-derived device symbols
    # (machines[0]→@device_a, machines[1]→@device_b, …; see
    # third_party/iree_bar/.../ScheduleSpec.cpp:deviceSymbolForIndex).
    #
    # The alias-shorthand `#hal.device.alias<"qnn", {qnn_backend="hta"}>`
    # does NOT propagate config to the executable target; only the full
    # `#hal.device.target<...>` form does (verified empirically — see
    # memory/reference_iree_multi_qnn_target_device_syntax.md).
    #
    # NOTE: this branch only fires when --with-schedule supplies a
    # machines list with 2+ entries spanning different backends. A
    # single-machine schedule keeps the legacy YAML flag path.
    qnn_machine_to_backend = {"HTA": "hta", "GPU": "gpu", "HTP": "htp"}
    cpu_machines = {"CPU", "CPU_P", "CPU_E"}

    def _is_multi_backend(machines: list[str]) -> bool:
        kinds = set()
        for m in machines:
            if m in qnn_machine_to_backend:
                kinds.add(("qnn", qnn_machine_to_backend[m]))
            elif m in cpu_machines:
                kinds.add(("cpu",))
            else:
                # Unknown machine kind — fall through to YAML defaults.
                return False
        return len(kinds) >= 2

    if schedule_machines and _is_multi_backend(schedule_machines):
        # Strip YAML-injected single-device QNN flags so they don't
        # conflict with our explicit per-machine flags.
        def _strip(prefix: str) -> None:
            static_flags[:] = [f for f in static_flags if not f.startswith(prefix)]

        _strip("--iree-hal-target-device=")
        _strip("--iree-hal-target-backends=")
        _strip("--iree-hal-qnn-backend=")
        _strip("--iree-llvmcpu-target-triple=")  # CPU triple set per-device below.

        # Stable letter suffix machines[i] → @device_<a..z..>; mirrors
        # ScheduleSpec.cpp:deviceSymbolForIndex base-26 encoding.
        def _device_sym(index: int) -> str:
            suffix = ""
            n = index
            while True:
                suffix = chr(ord("a") + (n % 26)) + suffix
                if n < 26:
                    break
                n = n // 26 - 1
            return f"device_{suffix}"

        emitted_flags: list[str] = []
        # Default QNN executable format remains "qnn-context-binary"; the
        # backend-suffixed `qnn-graph-{hta|gpu|htp}` is set per-variant at
        # serialize time. The {opaque_binary, qnn_backend} dict drives
        # both the per-device override in QNNTargetBackend::
        # getDefaultExecutableTargets and the format suffix in
        # serializeExecutable.
        for i, m in enumerate(schedule_machines):
            sym = _device_sym(i)
            if m in qnn_machine_to_backend:
                be = qnn_machine_to_backend[m]
                exe = (
                    '#hal.executable.target<"qnn", "qnn-context-binary", '
                    f'{{qnn_backend = "{be}", opaque_binary = true}}>'
                )
                dev = f'#hal.device.target<"qnn", [{exe}]>'
                emitted_flags.append(f"--iree-hal-target-device={sym}={dev}")
            elif m in cpu_machines:
                # local-task CPU device — the embedded ELF format covers
                # the QRB5165 aarch64 target. Keep alias form here since
                # the local backend has no per-device config knobs we
                # need to pin via deviceConfigAttr.
                emitted_flags.append(f'--iree-hal-target-device={sym}=#hal.device.alias<"local">')

        # Ensure the QNN compiler plugin is linked for this invocation
        # (it's PluginActivationPolicy::Explicit). Inert duplicate when
        # the YAML already requested it.
        plugin_flag = "--iree-plugin=hal_target_qnn"
        if any(m in qnn_machine_to_backend for m in schedule_machines):
            if plugin_flag not in static_flags:
                static_flags.append(plugin_flag)

        static_flags.extend(emitted_flags)
        print(
            f"  🔀 Multi-backend schedule: emitting {len(emitted_flags)} "
            f"per-machine target-device flags ({', '.join(schedule_machines)})"
        )

    # NHWC preprocess: rewrite NCHW convs → NHWC so the proven
    # nhwc_int8_conv recognizer matches. Required for HTA/Adreno (their
    # Conv2d only accepts NHWC + UFIXED_POINT_8 / fp16 on QAIRT 2.45;
    # the NCHW recognizer's Transpose adapter is rejected by both
    # backends). Auto-on when the schedule routes any dispatch to QNN.
    has_qnn_in_schedule = any(m in ("HTA", "GPU") for m in schedule_machines)
    if args.qnn_preprocess_nhwc or has_qnn_in_schedule:
        # Pass-pipeline syntax: nest the channels-last pass inside the
        # function-level scope it expects. The preprocess pass runs
        # before InputConversion / GlobalOptimization / DispatchCreation,
        # so the post-preprocess phase-3 IR contains the NHWC variant
        # the recognizer anchors on.
        pp = "builtin.module(util.func(iree-preprocessing-convert-conv-to-channels-last))"
        static_flags.append(f"--iree-preprocessing-pass-pipeline={pp}")
        print("  🔁 NHWC preprocess: NCHW convs will be rewritten before dispatch creation")

    # XPU-RT feedback overlay passthrough. Inert when --with-feedback
    # absent. When present, log the overlay summary and persist a
    # feedback_applied.json next to the artifacts so downstream
    # target-specific tooling (e.g. SpaceMit ukernel compile scripts) can
    # consult the disposition without re-parsing the source. We DO NOT
    # alter --iree-compile-arg silently from this overlay — IREE's
    # tile/ukernel selection lives in target specs and pass config, and
    # silent flag injection would break the additive-only invariant.
    if args.with_feedback:
        feedback_path = pathlib.Path(args.with_feedback).resolve()
        if not feedback_path.exists():
            utils.eprint(f"❌ feedback file not found: {feedback_path}")
            return 1
        from compile.feedback_overlay import load_feedback_overlay

        # Ingest can either point at <merlin_dir>/breakdowns/feedback.json
        # directly OR at any sibling location. Resolve both forms.
        if feedback_path.parent.name == "breakdowns":
            overlay = load_feedback_overlay(feedback_path.parent.parent)
        else:
            # Caller pointed at the file directly; load it via a temp dir
            # so the overlay reuses one parser.
            import json as _json

            payload = _json.loads(feedback_path.read_text())
            from feedback_overlay import DispatchDecision, FeedbackOverlay  # type: ignore

            decisions: dict = {}
            for d_id, entry in (payload.get("dispatches") or {}).items():
                hints = tuple(h for h in (entry.get("hints") or []) if isinstance(h, str))
                pin = next((h.split("=", 1)[1] for h in hints if h.startswith("pin_target=")), None)
                advisory = tuple(h for h in hints if not h.startswith("pin_target="))
                decisions[str(d_id)] = DispatchDecision(
                    dispatch_id=str(d_id),
                    hints=hints,
                    pin_target=pin,
                    advisory=advisory,
                    rationale=str(entry.get("rationale") or ""),
                )
            overlay = FeedbackOverlay(
                source_path=feedback_path,
                run_id=payload.get("run_id"),
                model_signals=dict(payload.get("model_signals") or {}),
                decisions_by_id=decisions,
            )

        if overlay.is_empty:
            print(f"  📡 Feedback overlay loaded ({feedback_path}) — " f"no per-dispatch hints; behavior unchanged")
        else:
            counts = overlay.summary()
            n_finer = counts.get("prefer_finer", 0)
            n_coarser = counts.get("prefer_coarser", 0)
            n_fuse = counts.get("consider_fuse_with_pred", 0)
            n_pin = counts.get("pin_target", 0)
            if n_finer > n_coarser * 1.2:
                disposition = "finer"
            elif n_coarser > n_finer * 1.2:
                disposition = "coarser"
            else:
                disposition = "neutral"
            print(
                f"  📡 Feedback overlay loaded ({feedback_path}): "
                f"run_id={overlay.run_id}, dispatches_with_hints="
                f"{len(overlay.decisions_by_id)}, hint_counts={counts}, "
                f"model_disposition={disposition}"
            )
            applied_marker = output_dir / "feedback_applied.json"
            import json as _json

            applied_marker.write_text(
                _json.dumps(
                    {
                        "schema_version": 1,
                        "source_feedback": str(feedback_path),
                        "run_id": overlay.run_id,
                        "target": args.target,
                        "hw": hw_choice,
                        "hint_counts": counts,
                        "model_disposition": disposition,
                        "model_signals": overlay.model_signals,
                        "n_dispatches_with_hints": len(overlay.decisions_by_id),
                        "n_pin_target": n_pin,
                        "n_consider_fuse_with_pred": n_fuse,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            print(f"  📡 feedback_applied.json -> {applied_marker}")

    # Kernel embedding pipeline (Part B). Resolution order: explicit
    # --kernel-manifest > --kernels-dir/manifest.json > YAML custom_kernels >
    # off. --no-kernel-embedding skips entirely.
    if not args.no_kernel_embedding:
        manifest_path = None
        if args.kernel_manifest:
            manifest_path = pathlib.Path(args.kernel_manifest).resolve()
        elif args.kernels_dir:
            manifest_path = pathlib.Path(args.kernels_dir).resolve() / "manifest.json"
        elif "custom_kernels" in cfg:
            yaml_manifest = cfg["custom_kernels"].get("manifest")
            if yaml_manifest:
                # YAML paths are interpreted relative to the model directory.
                base = model_dir if model_dir else mlir_file.parent
                manifest_path = (base / yaml_manifest).resolve()

        if manifest_path is not None:
            if not manifest_path.exists():
                utils.eprint(f"❌ kernel manifest not found: {manifest_path}")
                return 1
            from kernels.core import manifest as _kmanifest
            from kernels.core import precompile as _kprecompile
            from kernels.core import spec_gen as _kspec_gen

            print(f"  🧬 Loading kernel manifest: {manifest_path}")
            kmanifest_full = _kmanifest.load(manifest_path)
            # Honor the manifest's `select` list (if present) — pass only the
            # opt-in subset through precompile + spec_gen. Catalog stays
            # intact in source; just this compile sees the filtered view.
            active_kernels = kmanifest_full.selected_kernels()
            if len(active_kernels) != len(kmanifest_full.kernels):
                inactive_n = len(kmanifest_full.kernels) - len(active_kernels)
                print(
                    f"  🧬 select: {len(active_kernels)} of "
                    f"{len(kmanifest_full.kernels)} kernels enabled "
                    f"({inactive_n} kept in catalog but inert)"
                )
            kmanifest = _kmanifest.Manifest(
                path=kmanifest_full.path,
                schema_version=kmanifest_full.schema_version,
                kernels=active_kernels,
                select=kmanifest_full.select,
            )

            cache_dir = (
                pathlib.Path(args.kernel_cache_dir).resolve() if args.kernel_cache_dir else output_dir / "kernels_cache"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not active_kernels:
                print("  🧬 No kernels selected; skipping kernel embed.")
            else:
                # When the target yaml selects a specific QNN backend, only
                # build the matching per-target artifact for each kernel.
                # Without this, a manifest entry that lists both `qnn-gpu`
                # and `qnn-hta` would attempt to build the fp32 kernel
                # against libQnnHta (which only accepts uint8) — guaranteed
                # validation failure for any cross-dtype kernel.
                qnn_targets_filter: tuple[str, ...] | None = None
                for f in static_flags:
                    if f.startswith("--iree-hal-qnn-backend="):
                        qnn_be = f.split("=", 1)[1].strip()
                        qnn_targets_filter = (f"qnn-{qnn_be}",)
                        break

                print(f"  🧬 Precompiling {len(active_kernels)} kernel(s) -> {cache_dir}")
                if qnn_targets_filter is not None:
                    print(f"  🧬 QNN target filter: {qnn_targets_filter[0]}")
                try:
                    objects = _kprecompile.precompile(
                        kmanifest,
                        cache_dir,
                        targets_filter=qnn_targets_filter,
                    )
                except RuntimeError as e:
                    utils.eprint(f"❌ kernel precompile failed: {e}")
                    return 1

                spec_path = cache_dir / "transform_spec.mlir"
                print(f"  🧬 Generating transform spec: {spec_path}")
                try:
                    gen = _kspec_gen.emit(kmanifest, objects, spec_path, object_search_path=cache_dir)
                except (ValueError, RuntimeError) as e:
                    utils.eprint(f"❌ kernel spec generation failed: {e}")
                    return 1
                static_flags.append(f"--iree-preprocessing-transform-spec-filename={gen.spec_path}")
                static_flags.append(f"--iree-hal-executable-object-search-path={gen.object_search_path}")
                if gen.qnn_manifest_path is not None:
                    # The QNN passthrough plugin keys per executable export
                    # symbol → .qnn-ctx path. spec_gen wrote the sidecar; we
                    # forward it here so the plugin's `serializeExecutable`
                    # can find the right blob for each kernel-embedded
                    # `hal.executable.source @kb_<name>`.
                    static_flags.append(f"--iree-hal-qnn-manifest={gen.qnn_manifest_path}")

    print("  🔨 Compiling main model...")
    iree_compile = get_iree_tool("iree-compile", effective_build_dir)
    cmd = [str(iree_compile), str(mlir_file), "-o", str(compile_output_file)] + static_flags + dynamic_flags
    cmd.extend(args.iree_compile_arg)

    iree_rc = utils.run(cmd, dry_run=args.dry_run)
    if iree_rc != 0:
        utils.eprint("❌ Main compilation failed.")
    else:
        print(f"  ✅ Successfully compiled: {compile_output_file}")

    if args.kernels_strict_coverage:
        # Audit: every linalg op in the input must be covered by a kernel
        # rewrite. Walk phase 5 (dispatch-creation) and count linalg ops
        # that aren't inside a `flow.dispatch @kb_*` region.
        phases_dir = output_dir / "phases"
        # Phase filename uses dot-stripped basename (`.q.int8` -> `_q_int8`).
        phase5_candidates = list(phases_dir.glob("*.5.dispatch-creation.mlir"))
        if not phase5_candidates:
            utils.eprint(
                "❌ --kernels-strict-coverage requires phase 5 dump but none " f"was produced under {phases_dir}/."
            )
            return 1
        phase5 = phase5_candidates[0].read_text()
        # By phase 5 (dispatch-creation), unmatched dispatches have been
        # outlined into `flow.dispatch.workgroups(...) = (...) { ... linalg
        # ops ... flow.return }` blocks at the top level. Matched ops are
        # `flow.dispatch @kb_*::@<entry>(...)` direct calls (no body). So
        # the unmatched dispatch count == number of `flow.dispatch.workgroups`
        # occurrences. We classify each by the first linalg op in its body.
        import re as _re
        from collections import Counter as _Counter

        survivors: list[str] = []
        # Split on the dispatch.workgroups openings. Each segment after
        # the first holds the body of one unmatched dispatch up to the
        # next opening or end-of-text — we then scan up to its `flow.return`.
        segments = phase5.split("flow.dispatch.workgroups")[1:]
        for seg in segments:
            end = seg.find("flow.return")
            body = seg[:end] if end >= 0 else seg
            for m in _re.finditer(r"\blinalg\.([a-zA-Z_0-9]+)\b", body):
                op = m.group(1)
                if op in {"yield", "index"}:
                    continue
                survivors.append(f"linalg.{op}")
                break
            else:
                survivors.append("<empty-dispatch>")
        breakdown = _Counter(survivors)
        if breakdown:
            utils.eprint(
                "❌ --kernels-strict-coverage: dispatches survived past "
                "kernel rewrite (these went through IREE codegen, not your "
                "kernels):"
            )
            for op_name, n in breakdown.most_common():
                utils.eprint(f"     {n:5}x  {op_name}")
            utils.eprint(
                f"   Inspect {phase5_candidates[0]} and add matching manifest "
                f"entries (or run `python -m tools.kernels.discover` to "
                f"auto-generate stubs)."
            )
            return 1
        print("  ✅ kernels-strict-coverage: 0 unmatched dispatches " "(100% kernel coverage)")

    # If iree-compile itself failed, propagate the failure now (after audit so
    # the user sees coverage info even when codegen of an unmatched op crashed).
    if iree_rc != 0:
        return 1

    if args.build_benchmarks:
        sources_dir = output_dir / "benchmarks"
        vmfb_out_dir = sources_dir / "vmfb"
        vmfb_out_dir.mkdir(exist_ok=True)

        if sources_dir.exists():
            print("  🧩 Compiling individual dispatch sources...")
            for dispatch_mlir in sources_dir.glob("*.mlir"):
                dispatch_vmfb = vmfb_out_dir / f"{dispatch_mlir.stem}.vmfb"
                d_cmd = [str(iree_compile), str(dispatch_mlir), "-o", str(dispatch_vmfb)] + static_flags
                utils.run(d_cmd, dry_run=args.dry_run)

            zip_name = output_dir / f"{basename}_{args.target}{hw_suffix}_benchmarks.zip"
            zip_artifacts(zip_name, sources_dir, vmfb_out_dir)

    print("=" * 80)
    print(f"🎉 Completed {basename} [{args.target}{hw_suffix}]")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
