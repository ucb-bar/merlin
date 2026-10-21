"""Register native compilation commands on the existing TargetGen CLI."""


def add_io_memory_argument(parser):
    """Keep the provider-selected I/O geometry explicit across build and replay."""
    from argparse import ArgumentTypeError

    def positive_bytes(text):
        try:
            value = int(text)
        except ValueError as error:
            raise ArgumentTypeError("I/O memory size must be a positive integer") from error
        if value <= 0:
            raise ArgumentTypeError("I/O memory size must be a positive integer")
        return value

    parser.add_argument(
        "--io-memory-bytes",
        type=positive_bytes,
        help="device-visible I/O window size; must match the selected snapshot",
    )


def add_native_codegen_commands(sub, handlers):
    """Keep parser construction separate from lazy compiler dependencies."""
    from merlin.semantic_compiler.constant_math_backend import build_command
    from merlin.semantic_compiler.search import SearchAblations

    math_build = sub.add_parser("native-math-build", help="build a pinned compiler-only MPFR constant helper")
    math_build.add_argument("--out", required=True, help="fresh helper build directory")
    math_build.add_argument("--cc", help="provisioned C compiler; MPFR/GMP development files must be installed")
    math_build.set_defaults(func=build_command)

    native_select = sub.add_parser("native-select", help="select and allocate one typed kernel; no target emission")
    native_select.add_argument("--engine", choices=("merlin_native",), required=True)
    native_select.add_argument("--snapshot", required=True)
    select_source = native_select.add_mutually_exclusive_group(required=True)
    select_source.add_argument("--request", help="typed semantic kernel JSON")
    select_source.add_argument("--linalg", help="parsed Linalg MLIR input; admitted semantic subset")
    select_source.add_argument("--compile-inputs", help="verified compiler-only staged capture directory")
    native_select.add_argument("--linalg-entry", help="entry function name for --linalg")
    native_select.add_argument("--physical-layouts", help="source-bound physical layouts for parsed Linalg")
    native_select.add_argument("--support", help="installed target support for a source-bound diagnostic region")
    native_select.add_argument("--diagnostic-region", help="source-bound matmul region; requires --compile-inputs")
    native_select.add_argument("--row-softmax-tile-index", type=int, help="whole-row softmax row-group index")
    native_select.add_argument(
        "--whole-contraction",
        action="store_true",
        help="preserve the source contraction for backend tiling",
    )
    native_select.add_argument("--diagnostic-graph-policy", help="explicit precision policy for a parsed Linalg graph")
    native_select.add_argument(
        "--diagnostic-tile-index", type=int, help="row-major M/N output tile of a source-bound region"
    )
    native_select.add_argument("--mode", choices=("strict-native", "hybrid", "diagnostic"), default="strict-native")
    native_select.add_argument(
        "--abi",
        help="optional fixed_inputs/fixed_outputs JSON with optional reserved physical intervals; no runtime samples",
    )
    native_select.add_argument("--search-limits", help="complete versioned native search-limit JSON")
    native_select.add_argument(
        "--ablation",
        action="append",
        default=[],
        choices=tuple(vars(SearchAblations())),
        help="diagnostic-only native mechanism removal; repeat for multiple ablations",
    )
    native_select.add_argument("--out", required=True, help="selection result JSON")
    add_io_memory_argument(native_select)
    native_select.set_defaults(func=handlers["native_select"])

    native_compile = sub.add_parser(
        "native-compile",
        help="compile a native region or staged application with selected OOT support",
    )
    native_compile.add_argument("--engine", choices=("merlin_native",), required=True)
    native_compile.add_argument("--support", required=True, help="installed target support entry point")
    native_compile.add_argument("--snapshot", required=True)
    native_compile.add_argument(
        "--application", action="store_true", help="compile every supported region of a staged capture"
    )
    native_compile.add_argument(
        "--application-state", help="source-bound JSON mapping state inputs, updates and initialization"
    )
    native_compile.add_argument("--application-name", default="application", help="C application namespace")
    native_compile.add_argument(
        "--application-parameters",
        choices=("embedded", "external"),
        default="embedded",
        help="embed small model parameters or bind a checked external weight blob",
    )
    native_compile.add_argument(
        "--application-memory-bytes", type=int, help="application tensor arena and output-staging budget"
    )
    compile_source = native_compile.add_mutually_exclusive_group(required=True)
    compile_source.add_argument("--request", help="typed semantic kernel JSON")
    compile_source.add_argument("--linalg", help="parsed Linalg MLIR input; admitted semantic subset")
    compile_source.add_argument("--compile-inputs", help="verified compiler-only staged capture directory")
    native_compile.add_argument("--linalg-entry", help="entry function name for --linalg")
    native_compile.add_argument("--physical-layouts", help="source-bound physical layouts for parsed Linalg")
    native_compile.add_argument("--diagnostic-region", help="source-bound matmul region; requires --compile-inputs")
    native_compile.add_argument("--row-softmax-tile-index", type=int, help="whole-row softmax row-group index")
    native_compile.add_argument(
        "--whole-contraction",
        action="store_true",
        help="preserve the source contraction for backend tiling",
    )
    native_compile.add_argument("--diagnostic-graph-policy", help="explicit precision policy for a parsed Linalg graph")
    native_compile.add_argument(
        "--diagnostic-tile-index", type=int, help="row-major M/N output tile of a source-bound region"
    )
    native_compile.add_argument(
        "--ordered-k-tile",
        type=int,
        help="ask selected support to partition one annotated ordered reduction into exact K tiles",
    )
    native_compile.add_argument("--ordered-m-tile", type=int, help="qualified exact output-row tile size")
    native_compile.add_argument("--ordered-n-tile", type=int, help="qualified exact output-column tile size")
    native_compile.add_argument(
        "--outer-tile-index",
        type=int,
        help="row-major output tile to compile; requires both M/N tile sizes",
    )
    native_compile.add_argument(
        "--abi", required=True, help="fixed_inputs/fixed_outputs JSON, or auto for provider layout"
    )
    native_compile.add_argument("--search-limits", help="complete versioned native search-limit JSON")
    native_compile.add_argument("--reduction-unroll", choices=("auto", "1", "2", "4", "8"), default="auto")
    native_compile.add_argument("--tuning-profile", help="measured profile for this exact compilation")
    native_compile.add_argument("--program-size-limit-bytes", type=int, help="optional smaller device code budget")
    add_io_memory_argument(native_compile)
    native_compile.add_argument("--target-source", required=True, help="selected target source checkout")
    native_compile.add_argument("--mode", choices=("strict-native", "hybrid", "diagnostic"), required=True)
    native_compile.add_argument("--out", required=True, help="fresh compilation output directory")
    native_compile.add_argument("--status-file", help="invocation-owned machine-readable status JSON")
    native_compile.set_defaults(func=handlers["native_compile"])

    from .native_profile import command as profile_command

    profile = sub.add_parser("native-profile", help="measure checked reduction-unroll variants with an explicit runner")
    add_io_memory_argument(profile)
    for name in ("support", "snapshot", "target-source", "runner-command", "out"):
        profile.add_argument("--" + name, required=True)
    profile_source = profile.add_mutually_exclusive_group(required=True)
    profile_source.add_argument("--request", help="typed semantic kernel or graph")
    profile_source.add_argument("--linalg", help="parsed Linalg program")
    profile.add_argument("--linalg-entry")
    profile.add_argument("--diagnostic-graph-policy")
    profile.set_defaults(compile_inputs=None)
    profile.add_argument("--abi", default="auto")
    profile.add_argument("--search-limits")
    profile.add_argument("--program-size-limit-bytes", type=int)
    profile.add_argument("--objective", choices=("device_cycles", "warm_host_ticks"), default="device_cycles")
    profile.add_argument("--runner-timeout", type=int, default=300)
    profile.set_defaults(func=profile_command)

    model_regions = sub.add_parser(
        "native-probe-model-regions",
        help="diagnose all source-bound native regions without claiming model composition",
    )
    model_regions.add_argument("--engine", choices=("merlin_native",), required=True)
    model_regions.add_argument("--support", required=True, help="installed target support entry point")
    model_regions.add_argument("--snapshot", required=True)
    model_regions.add_argument("--compile-inputs", required=True, help="verified compiler-only staged capture")
    model_regions_abi = model_regions.add_mutually_exclusive_group(required=True)
    model_regions_abi.add_argument("--abi", help="exact fixed input/output addresses for every tile")
    model_regions_abi.add_argument("--abi-pool", help="diagnostic input/output slot pool for per-tile ABIs")
    model_regions.add_argument("--search-limits", help="complete versioned native search-limit JSON")
    model_regions.add_argument(
        "--max-named-tiles", type=int, default=256, help="maximum source-bound named tiles attempted in this diagnostic"
    )
    model_regions.add_argument(
        "--max-batched-tiles",
        type=int,
        default=256,
        help="maximum source-bound batched tiles attempted in this diagnostic",
    )
    model_regions.add_argument("--target-source", required=True, help="selected target source checkout")
    model_regions.add_argument("--out", required=True, help="fresh diagnostic artifact directory")
    model_regions.set_defaults(func=handlers["native_probe_model_regions"])

    host_object = sub.add_parser(
        "native-check-host-object",
        help="check a lowered host object's calls against an exact native model probe",
    )
    host_object.add_argument("--probe", required=True, help="source-bound native model probe directory")
    host_object.add_argument("--compile-inputs", required=True, help="same verified compiler-only capture")
    host_object.add_argument("--lowered", required=True, help="fresh Merlin host lowering directory")
    host_object.add_argument("--object", required=True, help="host ELF object within --lowered")
    host_object.add_argument("--nm", required=True, help="selected object symbol inspection tool")
    host_object.add_argument("--mlir-opt", required=True, help="selected MLIR tool for audited entry ABI")
    host_object.add_argument("--out", required=True, help="fresh object check artifact directory")
    host_object.set_defaults(func=handlers["native_check_host_object"])

    graph_bridge = sub.add_parser(
        "native-bridge-graph",
        help="outline one exact native whole graph as a typed external host call",
    )
    graph_bridge.add_argument("--linalg", required=True, help="exact source Linalg module")
    graph_bridge.add_argument("--entry", required=True, help="source function to externalize")
    graph_bridge.add_argument("--compiled", required=True, help="native whole-graph compilation directory")
    graph_bridge.add_argument("--device-symbol", required=True, help="C-compatible target callback symbol")
    graph_bridge.add_argument("--out", required=True, help="fresh typed host bridge output directory")
    graph_bridge.set_defaults(func=handlers["native_bridge_graph"])

    native_verify = sub.add_parser("native-verify", help="recheck native source, snapshot and published artifact bytes")
    native_verify.add_argument("--engine", choices=("merlin_native",), required=True)
    native_verify.add_argument("--snapshot", required=True)
    native_verify.add_argument(
        "--application", action="store_true", help="replay complete staged application compilation"
    )
    native_verify.add_argument(
        "--application-state", help="same explicit state interface used during application compilation"
    )
    native_verify.add_argument(
        "--application-parameters",
        choices=("embedded", "external"),
        default="embedded",
        help="same model parameter storage used during application compilation",
    )
    native_verify.add_argument("--target-source", help="selected target source for application C replay")
    verify_source = native_verify.add_mutually_exclusive_group(required=True)
    verify_source.add_argument("--request", help="original typed semantic kernel JSON")
    verify_source.add_argument("--linalg", help="original parsed Linalg MLIR input")
    verify_source.add_argument("--compile-inputs", help="verified compiler-only staged capture directory")
    native_verify.add_argument("--linalg-entry", help="entry function name for --linalg")
    native_verify.add_argument("--physical-layouts", help="source-bound physical layouts for parsed Linalg")
    native_verify.add_argument("--support", help="installed target support for a source-bound diagnostic region")
    native_verify.add_argument("--diagnostic-region", help="source-bound matmul region; requires --compile-inputs")
    native_verify.add_argument("--row-softmax-tile-index", type=int, help="recheck whole-row softmax row-group index")
    native_verify.add_argument(
        "--whole-contraction",
        action="store_true",
        help="recheck the complete source-bound contraction",
    )
    native_verify.add_argument("--diagnostic-graph-policy", help="explicit precision policy for a parsed Linalg graph")
    native_verify.add_argument(
        "--diagnostic-tile-index", type=int, help="recheck the selected row-major M/N source tile"
    )
    native_verify.add_argument("--ordered-k-tile", type=int, help="recheck the selected exact-K source partition")
    native_verify.add_argument("--ordered-m-tile", type=int, help="recheck exact output-row tile size")
    native_verify.add_argument("--ordered-n-tile", type=int, help="recheck exact output-column tile size")
    native_verify.add_argument("--outer-tile-index", type=int, help="recheck selected row-major output tile")
    native_verify.add_argument("--mode", choices=("strict-native", "hybrid", "diagnostic"), required=True)
    native_verify.add_argument("--artifact", required=True, help="published native compilation directory")
    native_verify.add_argument("--out", required=True, help="verification status JSON outside the artifact")
    add_io_memory_argument(native_verify)
    native_verify.set_defaults(func=handlers["native_verify"])
