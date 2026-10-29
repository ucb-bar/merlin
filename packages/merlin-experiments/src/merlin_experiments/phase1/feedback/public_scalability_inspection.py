"""Source-only inspection of the original selected public scalability members.

The functional suite keeps its engine/tier policy. This adapter calls existing
compiler inspection owners without exposing tensors or issuing a functional grade.
"""

from __future__ import annotations

import copy
import io
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


def requested(request):
    """Admit one explicit source inspection; all other probes retain their refusal."""
    if any(request.get(key) is not None and type(request[key]) is not bool
           for key in ("compile_only", "candidate_diagnostics")):
        raise ValueError("source inspection flags must be booleans")
    modes = [key for key in ("compile_only", "candidate_diagnostics") if request.get(key) is True]
    if not modes:
        return None
    if len(modes) != 1 or any(request.get(key) for key in (
            "caller_layout", "shape_coverage", "offload_census", "model_layers", "dynamic_census", "scale_probe",
            "tuning_overrides", "reference_request", "reference_experiment_request_id", "tuning_experiment",
            "reference_analysis", "mechanism_question")):
        raise ValueError("public scalability source inspection requires one unmodified diagnostic action")
    from .public_scalability import request_policy

    request_policy(request.get("sim"), request.get("tiers"))
    return modes[0]


def cli(args, *, context, mode):
    """Reuse ordinary source/ABI/build owners on freshly verified selected inputs."""
    from . import public_scalability
    from .compiler_profile_runtime import selfcheck_compile_only, selfcheck_diagnostics

    if requested(vars(args)) != mode:
        raise ValueError("public scalability inspection mode changed")
    selection = public_scalability.selected(context)
    if selection is None:
        raise ValueError("no public scalability suite is selected")
    root, record = selection
    names = {value.strip() for value in args.capsules.split(",") if value.strip()}
    if not names or names != {"all"} and not names <= set(record["cases"]):
        raise ValueError("source inspection requested a member outside the selected public suite")
    original = copy.copy(args)
    original.capsules_root = root / "cases"
    temporary_root = os.environ.get("TMPDIR")
    if not temporary_root or not Path(temporary_root).is_absolute():
        raise ValueError("source inspection requires the original task temporary directory")
    # Existing collectors publish to --out and stdout themselves. Hold both
    # channels private until the selected suite's final custody check succeeds.
    with tempfile.TemporaryDirectory(prefix="public-source-inspection-", dir=temporary_root) as directory:
        original.out = str(Path(directory) / "result.json")
        output = io.StringIO()
        with redirect_stdout(output):
            if mode == "compile_only":
                result = selfcheck_compile_only(original, context=context)
            else:
                from .candidate_observations import diagnostics_cli

                with selfcheck_diagnostics(original.compiler_bundle_manifest, context=context, arguments=original):
                    result = diagnostics_cli(original, context=context,
                                             capsules_root=original.capsules_root, contract=original.contract)
            if public_scalability.selected(context) != selection:
                raise ValueError("selected public source inspection inputs changed")
        if getattr(args, "out", None) and Path(original.out).is_file():
            Path(args.out).write_bytes(Path(original.out).read_bytes())
        print(output.getvalue(), end="")
    return result
