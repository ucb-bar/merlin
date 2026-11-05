"""Original model-check child with the same isolated compiler owner as capsules.

This host entrypoint transports explicit launch selections. It never imports
candidate modules into the evaluator or substitutes host implementations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from merlin_experiments.phase1.compiler_library import selected_child


@selected_child
def main(argv=None):
    from merlin.model_check.cli import main as model_main
    from merlin.targetgen.compiler_library import active_compiler_library
    from merlin_experiments.phase1.context import add_context_arguments, resolve_context
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import selfcheck_execution

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_context_arguments(parser)
    parser.add_argument("--compiler-bundle-manifest", type=Path)
    args, remaining = parser.parse_known_args(argv)
    original = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    original.add_argument("--submission", type=Path, required=True)
    original.add_argument("--work", type=Path, required=True)
    invocation, _ = original.parse_known_args(remaining)
    library = active_compiler_library()
    fork = invocation.submission / "compiler_library"
    if args.compiler_bundle_manifest is None:
        if library is not None or (fork.is_dir() and any(fork.rglob("*.py"))):
            raise ValueError("model compiler requires its original profile and library owner")
        return model_main(remaining)
    context = resolve_context(args, parser)
    with selfcheck_execution(args.compiler_bundle_manifest, context=context,
                             submission=invocation.submission,
                             work_root=invocation.work.resolve() / "compiler_execution"):
        return model_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
