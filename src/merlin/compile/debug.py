"""Compile debugging from a command line: list the stages, dump their IR, stop after one, keep a trace.

The same options on ``merlin-compile``, on the whole-model builder's CLI and (as ``build_options``
keys) on a launch config:

* ``--list-stages`` prints every stage the compiler names, read off the pipelines themselves: the
  stages each pipeline declares beside the code that records them
  (:func:`merlin.common.compile_trace.declare`), and the native passes of each upstream pass pipeline
  as the lowering builds it, with this build's optional-pass selection applied;
* ``--dump-ir-after S`` / ``--dump-ir-before S`` (repeatable, comma-separated, ``all``) write the IR at
  those stages; ``S#N`` names the Nth time a stage is reached (``mlir:canonicalize#2``);
* ``--stop-after S`` writes the IR at ``S`` and ends the compile there, exit 0, with no final artifact;
* ``--trace-dir DIR`` holds the dumps, ``trace.json`` (every stage reached, in order, with its file
  and seconds), ``pipeline.txt`` and the xDSL pass log. Any of the dump/stop options without it
  writes the trace under ``out/artifacts/probes/compile-trace/<target>/``.

Nothing here compiles anything or knows a target: it builds the request the recording points read.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common import compile_trace as T

#: The build-option keys a launch config spells these options with.
OPTION_KEYS = ("trace_dir", "dump_ir_after", "dump_ir_before", "stop_after")


def add_arguments(ap: argparse.ArgumentParser, *, list_stages: bool = True) -> None:
    """The compile-debugging options, on any parser."""
    group = ap.add_argument_group("compile debugging (see --list-stages)")
    if list_stages:
        group.add_argument(
            "--list-stages", action="store_true", help="list every stage --stop-after/--dump-ir-* can name, and exit"
        )
    group.add_argument(
        "--dump-ir-after",
        action="append",
        default=[],
        metavar="STAGE",
        help="write the IR after STAGE (repeatable, comma-separated; 'all'; STAGE#N for its Nth occurrence)",
    )
    group.add_argument(
        "--dump-ir-before", action="append", default=[], metavar="STAGE", help="write the IR before STAGE (as above)"
    )
    group.add_argument(
        "--stop-after",
        metavar="STAGE",
        help="write the IR at STAGE and stop there: exit 0, no final artifact",
    )
    group.add_argument(
        "--trace-dir",
        type=Path,
        help="a new directory for the per-stage IR, trace.json, pipeline.txt and the pass log "
        "(default with any dump/stop option: out/artifacts/probes/compile-trace/<target>/<timestamp>)",
    )


# ------------------------------------------------------------------------------------------- stages


def _load_pipelines() -> None:
    """Import every module that declares a pipeline, so the declarations are present."""
    import merlin.compile_cli  # noqa: F401 -- frontend
    import merlin.llvmlower.codegen  # noqa: F401 -- object, link
    import merlin.llvmlower.lower  # noqa: F401 -- llvm route (with its xDSL rewrites)
    import merlin.perf.whole_model_build  # noqa: F401 -- whole-model builder
    import merlin.xdsl_dialects.lowering.pipeline  # noqa: F401 -- staged route


def native_pipelines(features: frozenset[str] | None = None) -> dict[str, list[str]]:
    """Each upstream pass pipeline the LLVM route can run, as its pass names in order: the serial one,
    the OpenMP one, and the vectorizing one, built by the same functions a lowering calls, with the
    active optional-pass selection applied. A package's own features can splice more passes in; those
    are reached as ``mlir:<pass>`` stages all the same."""
    from pathlib import PurePath

    from merlin.llvmlower import pipeline as P
    from merlin.llvmlower.impr_features import normalize
    from merlin.llvmlower.optional_passes import selected_features

    feats = normalize(selected_features(features)) or frozenset()
    return {
        "serial": T.split_pipeline(P._upstream_pipeline(feats)),
        "openmp": T.split_pipeline(P._parallel_pipeline(feats)),
        # The schedule path is only an option value of the transform pass; the pass list is the same.
        "vector": T.split_pipeline(P.build_rvv_pipeline(PurePath("<schedule>"), features=feats)),
    }


#: The order the routes are SHOWN in (front door, the two lowering routes, codegen, the whole-model
#: builder); a pipeline not named here is shown after them. Presentation only: the stages are the
#: pipelines' own.
_SHOWN_FIRST = ("frontend", "staged", "llvm", "codegen", "whole-model")


def stage_rows(features: frozenset[str] | None = None) -> list[dict[str, Any]]:
    """Every nameable stage, in pipeline order: ``{stage, pipeline, entry, summary}``."""
    _load_pipelines()
    rows = []
    shown = sorted(
        T.declared(),
        key=lambda d: _SHOWN_FIRST.index(d["pipeline"]) if d["pipeline"] in _SHOWN_FIRST else len(_SHOWN_FIRST),
    )
    for decl in shown:
        for stage in decl["stages"]:
            rows.append(
                {"stage": stage, "pipeline": decl["pipeline"], "entry": decl["entry"], "summary": decl["summary"]}
            )
        if decl["pipeline"] == "llvm":
            seen: set[str] = set()
            for variant, passes in native_pipelines(features).items():
                for name in T.with_ordinals(passes):
                    if name in seen:
                        continue
                    seen.add(name)
                    rows.append(
                        {
                            "stage": T.MLIR_PREFIX + name,
                            "pipeline": f"llvm ({variant} passes)",
                            "entry": "merlin.llvmlower.pipeline",
                            "summary": "a native pass; its IR is the printer's dump after (or before) it",
                        }
                    )
    return rows


def known_stages() -> set[str]:
    _load_pipelines()
    return {stage for decl in T.declared() for stage in decl["stages"]}


def list_stages(a: argparse.Namespace) -> int:
    rows = stage_rows()
    if getattr(a, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    current = None
    for row in rows:
        if row["pipeline"] != current:
            current = row["pipeline"]
            if not current.startswith("llvm ("):
                print(f"\n{current}: {row['summary']}  [{row['entry']}]")
            else:
                print(f"\n{current}:")
        print(f"  {row['stage']}")
    print(
        "\nName a stage with --stop-after / --dump-ir-after / --dump-ir-before; STAGE#N is its Nth occurrence "
        "(mlir:canonicalize#2). A route reaches only its own stages: the staged route never reaches mlir:*."
    )
    return 0


# ------------------------------------------------------------------------------------------ request


def default_trace_dir(target: str, workload: str) -> Path:
    """``out/artifacts/probes/compile-trace/<target>/<TS>_<sha7>_<workload>``: a one-off diagnostic."""
    from merlin.common.artifacts import git_sha7, utc_stamp
    from merlin.common.paths import artifacts_dir

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in workload) or "compile"
    return artifacts_dir() / "probes" / "compile-trace" / target / f"{utc_stamp()}_{git_sha7()}_{safe}"


def make_request(
    *,
    trace_dir: str | Path | None,
    dump_after: Sequence[str] = (),
    dump_before: Sequence[str] = (),
    stop_after: str | None = None,
    default_dir: Callable[[], Path] | None = None,
) -> T.Request | None:
    """The validated request, or None when no option asks for a trace. Raises :class:`T.TraceError`.

    ``default_dir`` is called only when a trace is asked for without a directory."""
    if not (trace_dir or dump_after or dump_before or stop_after):
        return None
    known = known_stages()
    after = T.validate(dump_after, known, allow_all=True)
    before = T.validate(dump_before, known, allow_all=True)
    stop = T.validate([stop_after], known, allow_all=False) if stop_after else ()
    if len(stop) > 1:
        raise T.TraceError("--stop-after names one stage")
    directory = Path(trace_dir) if trace_dir else (default_dir() if default_dir is not None else None)
    if directory is None:
        raise T.TraceError("no trace directory; pass --trace-dir")
    return T.Request(
        directory=str(directory), dump_after=after, dump_before=before, stop_after=stop[0] if stop else None
    )


def request_from_args(
    ap: argparse.ArgumentParser, a: argparse.Namespace, *, target: str, workload: str
) -> T.Request | None:
    """The request the parsed options name (``ap.error`` on an unknown stage)."""
    try:
        return make_request(
            trace_dir=a.trace_dir,
            dump_after=a.dump_ir_after,
            dump_before=a.dump_ir_before,
            stop_after=a.stop_after,
            default_dir=lambda: default_trace_dir(target, workload),
        )
    except T.TraceError as exc:
        ap.error(str(exc))


def request_from_options(options: Mapping[str, Any], *, target: str, workload: str) -> T.Request | None:
    """The request a launch config's ``build_options`` name (:data:`OPTION_KEYS`)."""

    def listed(key: str) -> list[str]:
        value = options.get(key)
        return [] if value is None else ([value] if isinstance(value, str) else [str(v) for v in value])

    return make_request(
        trace_dir=options.get("trace_dir"),
        dump_after=listed("dump_ir_after"),
        dump_before=listed("dump_ir_before"),
        stop_after=options.get("stop_after"),
        default_dir=lambda: default_trace_dir(target, workload),
    )


def _install_pass_log(request: T.Request) -> dict[str, Any]:
    """Record xDSL pass invocations into the trace's pass log (the existing ``MERLIN_PASS_LOG`` recorder)."""
    try:
        from merlin.xdsl_dialects.lowering import passes

        passes.install_pass_recorder()
    except ImportError as exc:
        return {"pass_log_recorder": f"not installed: {exc}"}
    return {"pass_log_recorder": "installed"}


@contextlib.contextmanager
def opened(request: T.Request | None, command: Sequence[str]) -> Iterator[T.Request | None]:
    """A trace session for ``request`` (with the pass log recorded), or nothing when it is None."""
    if request is None:
        yield None
        return
    with T.session(request, command=list(command), on_open=_install_pass_log) as active:
        yield active


def stopped(tool: str, stop: T.StopAfterStage, *, as_json: bool = False) -> int:
    """Report a requested stop: where the IR is, and that no artifact was produced. Exit status 0."""
    if as_json:
        print(
            json.dumps(
                {
                    "tool": tool,
                    "status": "stopped",
                    "stage": stop.stage,
                    "files": list(stop.files),
                    "trace": str(Path(stop.directory) / T.INDEX),
                    "note": stop.note,
                },
                indent=2,
            )
        )
    else:
        print(stop.message(tool))
    return 0


def not_reached(request: T.Request | None) -> str | None:
    """Why a completed compile is not what was asked: the stop stage was never reached (None otherwise)."""
    if request is None or request.stop_after is None:
        return None
    return (
        f"--stop-after {request.stop_after} was never reached: this compile's route does not pass through it "
        f"(see {Path(request.directory) / T.INDEX} for the stages it did reach)"
    )
