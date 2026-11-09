"""Execute selected build commands with bounded resources and original receipts.

The caller supplies environment, timeout policy, error type and extra source
dependencies. This owner selects no compiler, target flags or lowering.
"""

import math
import subprocess
from pathlib import Path

from merlin.common import proc as _proc


def run(cmd, *, environment, timeout_limit_s, error, dependencies=(),
        timeout_s=None, inputs=(), outputs=(), stage="object"):
    if timeout_s is not None and (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("explicit compile timeout must be positive")
    limit = timeout_limit_s or None
    if timeout_s is not None:
        limit = min(limit, timeout_s) if limit is not None else timeout_s
    from merlin.common import invocation_record
    from merlin.common.cpu_pool import shared_compute_pool
    from merlin.runtime.build_runtime import run_in_build_view

    if not outputs:
        with shared_compute_pool(timeout_s=limit):
            return _proc.run_checked(
                cmd, error=error, timeout=limit, timeout_hint=" (pathological compile)", env=dict(environment)
            )
    argv = tuple(str(token) for token in cmd)
    cwd, environment = Path.cwd(), dict(environment)
    if "-fstack-usage" in argv:
        outputs = tuple(dict.fromkeys((*map(Path, outputs), Path(outputs[0]).with_suffix(".su"))))
    try:
        with shared_compute_pool(timeout_s=limit):
            selected = run_in_build_view(
                argv,
                inputs=inputs,
                outputs=outputs,
                directory=Path(outputs[0]).parent.resolve(),
                stage=stage,
                cwd=cwd,
                env=environment,
                timeout=limit,
                capture_output=True,
                text=True,
            )
    except subprocess.TimeoutExpired:
        shown = " ".join(argv)[:1000]
        raise error(f"command timed out after {limit}s (pathological compile): {shown}") from None
    if selected is not None:
        if selected.returncode:
            shown = " ".join(argv)[:1000]
            raise error(
                f"command failed (rc {selected.returncode}): {shown}\n"
                f"STDOUT:{(selected.stdout or '')[-2000:]}\nSTDERR:{(selected.stderr or '')[-2000:]}"
            )
        return selected
    from merlin.compiler.build import objects

    with (
        shared_compute_pool(timeout_s=limit),
        invocation_record.observe(
            Path(outputs[0]).parent,
            stage=stage,
            argv=argv,
            cwd=cwd,
            env=environment,
            inputs=inputs,
            outputs=outputs,
            dependencies=(*dependencies, Path(__file__), Path(objects.__file__)),
        ) as record,
    ):
        result = _proc.run_checked(
            argv,
            error=error,
            timeout=limit,
            timeout_hint=" (pathological compile)",
            cwd=cwd,
            env=environment,
        )
        record.complete(result)
        return result

