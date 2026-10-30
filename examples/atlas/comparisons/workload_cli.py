"""Shared bounded workload CLI reporting; numerical callbacks own their gates."""
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from merlin.common.jsonio import write_pretty_json
from source_relocation import argument_relocator


def run_report(args, callback, *, filename, claim=None, output=None, report=None,
               summary=None, on_error=None, failure_fields=None, reraise=False,
               error_repr=False, memory_unit="kib", atomic=False, allow_nan=None):
    """Run a gate, finalize its receipt, and preserve caller-owned cleanup and exits.

    Callbacks own numerical checks and restoration; ``on_error`` owns any extra
    failure evidence. Legacy CLIs can retain re-raised exceptions, byte RSS units,
    atomic replacement, initial fields and stdout without copying this lifecycle.
    """
    from capture_tinyllama_native import artifact_output, identity
    sys.dont_write_bytecode = True
    if output is None:
        args.output = artifact_output(args.output_dir)
        output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    custom_report = report is not None
    if allow_nan is None:
        allow_nan = not custom_report
    if report is None:
        report = dict(schema_version=1, status="failed", claim=claim, command=sys.argv,
                      cwd=str(Path.cwd()), python_executable=sys.executable,
                      source_script=identity(sys.argv[0]), complete_atlas_execution=False,
                      timing_qualified=False, application_quality_qualified=False,
                      environment={key: os.environ.get(key) for key in
                                   ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    bindings = [identity(__file__)]
    code = 1
    receipt = output / filename
    try:
        if not custom_report or hasattr(args, "relocate"):
            argument_relocator(args)
        callback(args, report)
        for binding in bindings:
            identity(binding["path"], binding["sha256"])
        report["report_helper_bookends"] = bindings
        code = 0
    except Exception as error:
        report.update(status="failed", error=repr(error) if error_repr else dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        report.update(failure_fields or {})
        if on_error is not None:
            on_error(error, report)
        if reraise:
            raise
    finally:
        relocation = getattr(args, "_source_relocation", None)
        if relocation is not None:
            report["source_relocation"] = relocation.report()
        if memory_unit is not None:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report.update(wall_seconds=time.monotonic() - started)
            report["peak_rss_" + memory_unit] = peak * 1024 if memory_unit == "bytes" else peak
        destination = receipt.with_name(receipt.name + ".tmp") if atomic else receipt
        write_pretty_json(destination, report, sort_keys=not custom_report, allow_nan=allow_nan)
        if atomic:
            destination.replace(receipt)
    if summary is None:
        result = dict(status=report["status"], receipt=str(receipt), error=report.get("error"))
    else:
        result = summary(report) if callable(summary) else {key: report[key] for key in summary}
    print(json.dumps(result), flush=True)
    return code
