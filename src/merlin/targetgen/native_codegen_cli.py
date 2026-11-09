"""Register implemented native compiler/preparation/replay command options."""


def preparation_arguments(parser):
    """Expose only transformations executed by select, compile and replay."""
    parser.add_argument("--diagnostic-region", help="source-bound staged contraction region ID")
    parser.add_argument("--diagnostic-tile-index", type=int, help="explicit source-bound output tile index")
    parser.add_argument("--ordered-k-tile", type=int, help="selected backend's qualified ordered reduction geometry")
    parser.add_argument("--ordered-m-tile", type=int, help="selected backend's qualified outer row geometry")
    parser.add_argument("--ordered-n-tile", type=int, help="selected backend's qualified outer column geometry")
    parser.add_argument("--outer-tile-index", type=int, help="selected outer M/N output tile index")


def add_arguments(sub, handlers):
    from merlin.semantic_compiler.search import SearchAblations

    native_build = sub.add_parser("native-build", help="build an offline Merlin-native selection snapshot")
    native_build.add_argument("--engine", choices=("merlin_native",), required=True)
    native_build.add_argument("--profile", help="versioned native target profile JSON")
    native_build.add_argument("--support", help="explicit installed native target support provider")
    native_build.add_argument("--crate", help="override the packaged Merlin egg bridge source directory")
    native_build.add_argument("--cargo-target-dir", required=True)
    native_build.add_argument("--source-revision", required=True)
    native_build.add_argument("--out", required=True, help="fresh snapshot directory")
    native_build.set_defaults(func=handlers["_cmd_native_build"])

    from merlin.semantic_compiler.constant_math_backend import build_command

    math_build = sub.add_parser("native-math-build", help="build a pinned compiler-only MPFR constant helper")
    math_build.add_argument("--out", required=True, help="fresh helper build directory")
    math_build.add_argument("--cc", help="provisioned C compiler with MPFR/GMP development files")
    math_build.set_defaults(func=build_command)

    native_select = sub.add_parser("native-select", help="select and allocate one typed kernel; no target emission")
    native_select.add_argument("--engine", choices=("merlin_native",), required=True)
    native_select.add_argument("--snapshot", required=True)
    select_source = native_select.add_mutually_exclusive_group(required=True)
    select_source.add_argument("--request", help="typed semantic kernel JSON")
    select_source.add_argument("--linalg", help="parsed Linalg MLIR input; admitted integer subset only")
    select_source.add_argument("--compile-inputs", help="verified compiler-only staged capture directory")
    native_select.add_argument("--linalg-entry", help="entry function name for --linalg")
    native_select.add_argument("--support", help="explicit target support for diagnostic graph preparation")
    native_select.add_argument("--diagnostic-graph-policy", help="explicit precision policy for a parsed Linalg graph")
    preparation_arguments(native_select)
    native_select.add_argument("--physical-layouts", help="explicit source/request-bound physical boundary layout JSON")
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
    native_select.set_defaults(func=handlers["_cmd_native_select"])

    native_compile = sub.add_parser("native-compile", help="compile one typed native region with selected OOT support")
    native_compile.add_argument("--engine", choices=("merlin_native",), required=True)
    native_compile.add_argument("--support", required=True, help="installed target support entry point")
    native_compile.add_argument("--snapshot", required=True)
    compile_source = native_compile.add_mutually_exclusive_group(required=True)
    compile_source.add_argument("--request", help="typed semantic kernel JSON")
    compile_source.add_argument("--linalg", help="parsed Linalg MLIR input; admitted integer subset only")
    compile_source.add_argument("--compile-inputs", help="verified compiler-only staged capture directory")
    native_compile.add_argument("--linalg-entry", help="entry function name for --linalg")
    native_compile.add_argument("--diagnostic-graph-policy", help="explicit precision policy for a parsed Linalg graph")
    preparation_arguments(native_compile)
    native_compile.add_argument(
        "--physical-layouts", help="explicit source/request-bound physical boundary layout JSON"
    )
    native_compile.add_argument("--abi", required=True, help="fixed_inputs/fixed_outputs JSON")
    native_compile.add_argument("--search-limits", help="complete versioned native search-limit JSON")
    native_compile.add_argument("--target-source", required=True, help="selected target source checkout")
    native_compile.add_argument("--mode", choices=("strict-native", "hybrid", "diagnostic"), required=True)
    native_compile.add_argument("--out", required=True, help="fresh compilation output directory")
    native_compile.add_argument("--status-file", help="invocation-owned machine-readable status JSON")
    native_compile.set_defaults(func=handlers["_cmd_native_compile"])

    graph_bridge = sub.add_parser(
        "native-bridge-graph",
        help="outline one exact native whole graph as a typed external host call",
    )
    graph_bridge.add_argument("--linalg", required=True, help="exact source Linalg module")
    graph_bridge.add_argument("--entry", required=True, help="source function to externalize")
    graph_bridge.add_argument("--compiled", required=True, help="native whole-graph compilation directory")
    graph_bridge.add_argument("--device-symbol", required=True, help="C-compatible target callback symbol")
    graph_bridge.add_argument("--out", required=True, help="fresh typed host bridge output directory")
    graph_bridge.set_defaults(func=handlers["_cmd_native_bridge_graph"])

    from .native_application_cli import add_arguments as application_arguments

    application_arguments(native_compile, compiling=True)
    native_verify = sub.add_parser("native-verify", help="replay native source/snapshot/artifacts; no execution claim")
    application_arguments(native_verify, compiling=False, require_application=False)
    native_verify.add_argument("--engine", choices=("merlin_native",), required=True)
    native_verify.add_argument("--support", help="explicit support for graph preparation or application replay")
    native_verify.add_argument("--snapshot", required=True)
    verify_source = native_verify.add_mutually_exclusive_group(required=True)
    verify_source.add_argument("--request", help="original typed semantic kernel JSON")
    verify_source.add_argument("--linalg", help="original parsed Linalg MLIR")
    verify_source.add_argument("--compile-inputs", help="verified compiler-only staged capture directory")
    native_verify.add_argument("--linalg-entry", help="entry function for --linalg")
    native_verify.add_argument("--diagnostic-graph-policy", help="same explicit source-bound graph precision policy")
    preparation_arguments(native_verify)
    native_verify.add_argument(
        "--physical-layouts", help="same explicit source/request-bound physical boundary layout JSON"
    )
    native_verify.add_argument("--target-source", help="selected target source for application replay")
    native_verify.add_argument("--artifact", required=True)
    native_verify.add_argument("--mode", choices=("strict-native", "hybrid", "diagnostic"), required=True)
    native_verify.add_argument("--out", required=True, help="external verification status JSON")
    native_verify.set_defaults(func=handlers["_cmd_native_verify"])
