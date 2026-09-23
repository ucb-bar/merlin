"""Compatibility CLI for installed functional engine qualification."""

import sys

from merlin_experiments.phase2.functional_qualification import main as qualification_main

from merlin.common.paths import merlin_dir, repo_root


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--contract-root" or arg.startswith("--contract-root=") for arg in arguments):
        arguments += ["--contract-root", str(merlin_dir() / "contract")]
    if not any(arg == "--source-root" or arg.startswith("--source-root=") for arg in arguments):
        arguments += ["--source-root", str(repo_root())]
    return qualification_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
