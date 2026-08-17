"""Every stage between a `@triton.jit` kernel and a command buffer an accelerator executes, printed.

This is the inside view of what `merlin-compile-kernel` does in one shot. It writes nothing: each step
prints the artifact it produced plus the reason that step exists, so the pipeline is readable as a
sequence of decisions rather than as one opaque command. `run.sh compile` is the outside view.

Two rules this file follows, both of which matter more than they look:

* every declaration is parsed by the CLI's OWN parser (`cli.parse_arg`), and every verdict comes from
  the library's own entry points. Nothing here re-implements a step in order to narrate it -- an
  example that drifts from the code it explains teaches the wrong thing with total confidence.
* the target is passed in as a PACKAGE PATH, never as a name. That is the whole architecture in one
  argument: this file has no branch on which accelerator it is talking to, and neither does the
  frontend. Point `--package` at a different directory and the same kernel descends somewhere else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = "out/artifacts/targets/gemmini/hand_v0"

# The kernel and its declarations, spelled exactly as `run.sh compile` passes them to the CLI. A
# Triton kernel is not self-describing -- untyped parameters, shapeless pointers, the grid at the call
# site -- so these are the facts Merlin refuses to guess. See README.md "What you have to declare".
KERNEL = "examples/triton/matmul_simple.py:repeated_rhs_matmul"
ARGS = ("a0_ptr=*i8:16x32:read", "a1_ptr=*i8:16x32:read", "w_ptr=*i8:32x16:read",
        "c0_ptr=*i32:16x16:write", "c1_ptr=*i32:16x16:write")
CONSTEXPRS = {"BM": 16, "BN": 16, "BK": 32}
GRID = (1,)


class WalkthroughError(RuntimeError):
    """A prerequisite is missing. The message names the fix, not the symptom."""


def _rule(title: str) -> None:
    print(f"\n\033[1m{'=' * 78}\n{title}\n{'=' * 78}\033[0m")


def _why(text: str) -> None:
    """The reason a step exists. Suppressed by --quiet for readers on a second pass."""
    if not _QUIET:
        for line in text.strip().splitlines():
            print(f"  \033[2m{line.strip()}\033[0m")
        print()


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


# --------------------------------------------------------------------------- steps
def step_toolchain(state: dict) -> None:
    """Is the pinned Triton usable at all?"""
    from merlin.triton import toolchain

    _why("""
        merlin.triton drives triton's compiler-INTERNAL frontend (ASTSource/make_ir), which is not a
        stable API and changes between minor releases. So the version is exact-pinned rather than
        best-effort: a drifted triton either raises deep inside the frontend, or silently emits
        different TTIR and changes your compiled kernel with nothing to point at.
        """)
    probe = toolchain.probe()
    print(f"    triton installed : {probe.installed}")
    print(f"    pinned to        : {probe.pinned}")
    print(f"    usable           : {probe.compatible}")
    for note in probe.notes:
        print(f"    note             : {note}")
    if not probe.compatible:
        raise WalkthroughError(f"{probe.reason}\n  fix: uv pip install -e '.[triton]'")
    state["triton_version"] = probe.installed


def step_package(state: dict) -> None:
    """Load the accelerator from disk, and read what it says it can do."""
    from merlin.targetgen.registry import load_target

    _why("""
        This is the seam that makes the frontend target-independent. The accelerator arrives as a
        DIRECTORY -- a manifest, a dialect module, a lowering table, a contract -- loaded at run time.
        Core does not import it, does not know its name, and has no table of supported targets. The
        directory below is tracked in a fresh clone, so this step needs no toolchain at all.
        """)
    package_dir = state["package_dir"]
    if not package_dir.is_dir():
        raise WalkthroughError(
            f"no target package at {package_dir}\n"
            f"  fix: pass --package <dir>, or see docs/guides/adding_a_target.md to build one")
    pkg = load_target(package_dir)
    state["package"] = pkg
    print(f"    package dir : {package_dir.relative_to(REPO_ROOT)}")
    print(f"    target      : {pkg.name}  (run {pkg.run_id})")
    print(f"    family      : {pkg.contract.get('family', '?')}")
    print(f"    features    : {', '.join(pkg.contract.get('features', ()))}")
    print(f"\n    the dialect it declares ({pkg.dialect_plan().get('dialect_name', '?')}) -- its own")
    print("    compiler-level vocabulary, which is NOT its ISA:")
    for name, summary in _declared_ops(package_dir):
        print(f"      {name:10s} {summary}")
    print("\n    its lowering table (neutral interface op -> its op -> its command opcode):")
    for iface, tgt in pkg.lowering_table.items():
        print(f"      {iface:28s} -> {tgt:16s} -> {pkg.opcode_table.get(tgt, '?')}")


def _declared_ops(package_dir: Path) -> list[tuple[str, str]]:
    """(name, summary) for each op the package declares, read from its own plan file.

    Absent is fine and says so -- a package is not required to carry prose, and inventing a summary
    for it would put words in a target's mouth.
    """
    import yaml

    plan = package_dir / "contracts" / "dialect_plan.yaml"
    if not plan.is_file():
        return [("(none)", f"no {plan.relative_to(package_dir)} in this package")]
    doc = yaml.safe_load(plan.read_text()) or {}
    return [(op.get("name", "?"), op.get("summary", "")) for op in doc.get("ops", ())]


def step_declare(state: dict) -> None:
    """Turn the CLI-style declarations into the spec the frontend consumes."""
    from merlin.triton.cli import load_kernel, parse_arg
    from merlin.triton.spec import GridSpec, TritonKernelSpec

    _why("""
        Each --arg states an element type, a STATIC shape and an effect (read/write/readwrite).
        Effects are declared rather than discovered, and then cross-checked against what the kernel
        actually does: a kernel that writes a buffer the caller believes is read-only is a miscompile,
        so a disagreement in either direction is an error. A pointer with no static shape is refused
        outright -- there is deliberately no dynamic-shape fallback to paper over a missing fact.
        """)
    fn = load_kernel(str(REPO_ROOT / KERNEL.split(":")[0]) + ":" + KERNEL.split(":")[1])
    spec = TritonKernelSpec(function=fn, args=tuple(parse_arg(a) for a in ARGS),
                            grid=GridSpec(dims=GRID), constexprs=dict(CONSTEXPRS))
    state["spec"] = spec
    print(f"    kernel     : {spec.name}")
    for arg in spec.args:
        shape = "x".join(str(d) for d in arg.shape) if arg.shape else "scalar"
        print(f"      {arg.name:10s} {arg.dtype:5s} {shape:10s} {arg.effect or ''}")
    print(f"    constexprs : {spec.constexprs}")
    print(f"    grid       : {spec.grid.dims}")


def step_ttir(state: dict) -> None:
    """Triton's own frontend, used unmodified."""
    from merlin.triton import source

    _why("""
        A STOCK triton wheel produces this. Merlin does not fork triton, does not patch it, and does
        not add a backend to it -- which is what keeps an ordinary Triton kernel ordinary. TTIR is
        also where the GPU assumptions still live: pointers, an SPMD grid, block-shaped tiles.
        """)
    ttir = source.make_ttir(state["spec"])
    state["ttir"] = ttir
    ops: dict[str, int] = {}
    for op in source.walk_ops(ttir):
        name = op.get_name()                   # a triton IR object, so its own accessor
        ops[name] = ops.get(name, 0) + 1
    print(f"    TTIR ops: {sum(ops.values())} total")
    for name, count in sorted(ops.items()):
        print(f"      {count:3d}  {name}")
    print(f"    carries tt.dot: {ttir.has_op('tt.dot')}")


def step_bridge(state: dict) -> None:
    """TTIR -> linalg-on-tensors: the single convergence point."""
    from merlin.triton.bridge import to_linalg

    _why("""
        The one design decision everything else follows from: TTIR is raised to linalg-on-tensors and
        NEVER lowered toward a target dialect. Pointers become whole tensors, the SPMD grid is
        normalized away, and masks are proven to cover each element exactly once and in order.

        Erasing the grid is deliberate. It leaves the parallelism decision UNMADE, so a systolic
        array can sequentialize it and a SIMT machine can map it to warps -- from the same input. The
        cost is that the author's own decomposition is currently discarded with it; see
        docs/design/triton_frontend.md and README.md "What this does not do yet".
        """)
    bridged = to_linalg(state["ttir"], state["spec"])
    state["bridged"] = bridged
    text = bridged.text
    print(f"    no tt.*/ttg.* survives : {'tt.' not in text and 'ttg.' not in text}")
    print(f"    core MLIR ({len(text.splitlines())} lines):\n")
    print(_indent(text))


def step_route(state: dict) -> None:
    """Choose a route from the PAYLOAD and the target's declared coverage -- not from its name."""
    from merlin import compile_core

    _why("""
        Routing asks two questions: what is this computation, and what does this package's own
        dialect plan say it materializes? A matmul on a target that covers matmul takes the staged
        accelerator descent; anything else compiles as generic computation. Neither branch is keyed
        on a target name, and an unreadable plan FAILS CLOSED -- routing cannot tell "accelerates
        nothing" from "the plan is somewhere else", so it refuses instead of guessing the first.
        """)
    result = compile_core.compile_core_mlir(state["bridged"].module, target_package=state["package"])
    state["result"] = result
    route = result.route
    print(f"    payload         : {route.payload}")
    print(f"    target covers   : {route.materializable}")
    print(f"    route           : {route.kind}")
    print(f"    reason          : {route.reason}")
    if route.kind != "staged":
        raise WalkthroughError(
            "this walkthrough follows the staged accelerator descent, and this payload did not take "
            "it.\n  That is a legitimate outcome, not a bug: this package does not declare that it "
            "materializes\n  this payload. `run.sh route` and `run.sh compare` show that refusal "
            "deliberately, side by side.")
    state["lowered"] = result.staged


# What each stage of the descent is FOR. The attribute names are LoweringResult's own fields, so a
# renamed or added stage surfaces here as an AttributeError rather than as a quietly shorter list.
STAGE_MEANING = (
    ("input_module", "what to compute -- the frontend's linalg, unchanged"),
    ("contract_module", "what must be TRUE for the accelerator path to be legal, and its proof"),
    ("schedule_module", "placement, layout, liveness, dispatch grouping"),
    ("interface_module", "target-NEUTRAL accelerator vocabulary: pack / matmul / commit / evict"),
    ("target_module", "this package's own dialect, via its lowering table"),
    ("runtime_module", "device acquisition, command-buffer construction, submit, wait"),
)


def step_stages(state: dict) -> None:
    """The six modules, each a named decision rather than a compiler pass with a number."""
    _why("""
        Read these as an argument: the CONTRACT states what must be true (the weight is immutable,
        it fits in resident storage) and proves it; the SCHEDULE decides placement, layout and
        dispatch grouping; the INTERFACE is target-neutral accelerator vocabulary; and only then does
        anything become this target's own dialect. A claim is proven before it is relied on.
        """)
    lowered = state["lowered"]
    for attr, decides in STAGE_MEANING:
        module = getattr(lowered, attr)
        module.verify()                        # structural validity, checked here as the CLI does
        ops = [op.name for op in module.walk()]
        # The dialects present ARE the stage's identity, so show which ones appeared, not every op.
        dialects = sorted({o.split(".", 1)[0] for o in ops} - {"builtin", "func"})
        print(f"      {attr.removesuffix('_module'):10s} {len(ops):3d} ops  [{', '.join(dialects)}]")
        print(f"      {'':10s} {decides}")
    print(f"\n    this target's dialect after lowering:")
    tgt_ops = sorted({op.name for op in state["lowered"].target_module.walk()}
                     - {"builtin.module", "func.func", "func.return"})
    for name in tgt_ops:
        print(f"      {name}")


def step_command_buffer(state: dict) -> None:
    """The deliverable: what the hardware is actually told to do."""
    _why("""
        Note what the compiler INFERRED rather than what you asked for. The kernel loads one shared
        weight and multiplies two activations against it, so the weight is packed resident ONCE, used
        twice, and released -- RES_PACK / MATMUL x2 / EVICT. Give each matmul its own weight and that
        inference correctly disappears. This is the payoff of proving immutability and capacity in the
        contract: residency is deduced from the program, not requested by an annotation.
        """)
    cb = state["lowered"].command_buffer
    state["cb"] = cb
    print(f"    target   : {cb['target']}")
    print(f"    tensors  : " + ", ".join(
        f"{n}[{'x'.join(str(d) for d in t['shape'])}]:{t['dtype']}" for n, t in cb["tensors"].items()))
    print(f"    commands :")
    for i, cmd in enumerate(cb["commands"]):
        operands = " ".join(f"{k}={v}" for k, v in cmd.get("operands", {}).items())
        print(f"      {i}  {cmd['opcode']:16s} {operands}")


def step_check(state: dict) -> None:
    """L0: is it right? Answered with numpy, so it runs on any machine."""
    import numpy as np

    from merlin.runtime import reference_outputs, simulate
    from merlin.runtime.commandbuffer import materialize_inputs

    _why("""
        Two independent comparisons, because one would only prove Merlin agrees with itself. First
        the command-buffer interpreter against the reference outputs; then every COMMIT against a
        matmul computed here in numpy. Operand names are READ OUT of the command buffer rather than
        assumed, so this keeps checking the right tensors if the naming changes.

        This is L0 -- it localizes a failure to semantics with no toolchain involved. It is NOT a
        hardware result, and nothing here may be reported as one. `run.sh certify` is that.
        """)
    cb = state["cb"]
    outputs = simulate(cb)["outputs"]
    assert outputs == reference_outputs(cb), "interpreter disagrees with the reference"
    print("    interpreter == reference outputs : True")

    tensors = materialize_inputs(cb)
    packed = {c["operands"]["dst"]: c["operands"]["src"]
              for c in cb["commands"] if c["opcode"] == "RES_PACK"}
    lhs_of = {c["operands"]["dst"]: (c["operands"]["lhs"], c["operands"]["rhs"])
              for c in cb["commands"] if c["opcode"].startswith("MATMUL")}
    for commit in [c for c in cb["commands"] if c["opcode"] == "COMMIT"]:
        lhs, rhs = lhs_of[commit["operands"]["src"]]
        activation = np.array(tensors[lhs].to_list(), dtype=np.int64)
        weight = np.array(tensors[packed.get(rhs, rhs)].to_list(), dtype=np.int64)
        got = np.array(outputs[commit["operands"]["dst"]], dtype=np.int64)
        ok = np.array_equal(got, activation @ weight)
        print(f"    {commit['operands']['dst']} == numpy({lhs} @ {rhs})".ljust(45) + f": {ok}")
        assert ok, commit["operands"]["dst"]


STEPS = (
    ("toolchain", "1. Is the pinned Triton usable?", step_toolchain),
    ("package", "2. Load the accelerator from disk", step_package),
    ("declare", "3. Declare what Triton does not say", step_declare),
    ("ttir", "4. Triton's own frontend -> TTIR", step_ttir),
    ("bridge", "5. TTIR -> linalg-on-tensors (the convergence point)", step_bridge),
    ("route", "6. Route on the payload, not the target", step_route),
    ("stages", "7. The staged descent, six named decisions", step_stages),
    ("commands", "8. The command buffer", step_command_buffer),
    ("check", "9. L0: check it against numpy", step_check),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="steps: " + ", ".join(name for name, _, _ in STEPS))
    parser.add_argument("--package", default=DEFAULT_PACKAGE,
                        help=f"target package directory (default: {DEFAULT_PACKAGE})")
    parser.add_argument("--pause", action="store_true",
                        help="stop after each step and wait for Enter")
    parser.add_argument("--quiet", action="store_true",
                        help="artifacts only, without the explanation of why each step exists")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, title, _ in STEPS:
            print(f"  {name:10s} {title}")
        return 0

    global _QUIET
    _QUIET = args.quiet

    package_dir = Path(args.package)
    if not package_dir.is_absolute():
        package_dir = REPO_ROOT / package_dir
    state: dict = {"package_dir": package_dir}

    for i, (_, title, fn) in enumerate(STEPS):
        _rule(title)
        try:
            fn(state)
        except WalkthroughError as exc:
            print(f"\n\033[1mstopped:\033[0m {exc}", file=sys.stderr)
            return 1
        if args.pause and i + 1 < len(STEPS):
            try:
                input("\n  [Enter] for the next step, Ctrl-C to stop ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

    _rule("Done")
    print("    A stock Triton kernel became a command buffer this accelerator executes, and the")
    print("    result was checked against an independent numpy matmul.\n")
    print("    Nothing on that path knew the target's name. To prove that rather than take its word:")
    print("      ./run.sh converge     the same computation hand-written in linalg -> byte-identical")
    print("      ./run.sh certify      the same command buffer on the target's own Verilator RTL\n")
    return 0


_QUIET = False

if __name__ == "__main__":
    raise SystemExit(main())
