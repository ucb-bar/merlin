"""Compose source-exact scalar host rewrites at the existing tensor boundary.

The native matchers decide applicability on current IR after ordinary fusion.
The default chooses only a single ordered accumulator, smaller-domain source
math, and squared sums. Packet widths, unrolling, approximation tables and
device schedules require separate cost/capability selections.
"""

from __future__ import annotations

import json
import os

from merlin.compile.host_optimizations import tensor_rewrite_source

FEATURE = "automatic_scalar_host"
MARKER = "__merlin_automatic_scalar_host__"
TOKEN = "OK automatic_scalar_host "
PASS_NAME = "automatic-host-optimizations"


def edit_pipeline(passes):
    hits = [i for i, stage in enumerate(passes) if "one-shot-bufferize" in stage]
    if len(hits) != 1 or MARKER in passes:
        raise ValueError("automatic scalar host requires one bufferization stage")
    at = hits[0]
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered():
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                edit_pipeline=edit_pipeline,
                description="Compose checked source-exact scalar host rewrites after ordinary tensor fusion.",
            )
        )
    return FEATURE


def select(features, *, serial):
    """Select the ordinary scalar route; preserve explicit schedules and opt-outs."""
    from .optional_passes import switched

    selected = frozenset(features or ())
    enabled = serial and switched(PASS_NAME, os.environ.get("MERLIN_AUTOMATIC_HOST_OPTIMIZATIONS", "1") != "0")
    if enabled:
        selected |= {FEATURE}
    return selected


def choices(features):
    """Resolve explicit alternatives before building the child runner."""
    from .broadcast_math_hoist import FEATURE as broadcast
    from .optional_passes import active
    from .scalar_contraction import _OUTPUTS_FEATURES
    from .scalar_squared_sum import FEATURE as squared_sum

    selection = active()
    return {
        "contraction": not bool(_OUTPUTS_FEATURES & features) and selection.state("scalar-contraction") is not False,
        "squared_sum": squared_sum not in features and selection.state("scalar-squared-sum") is not False,
        "broadcast_math": broadcast not in features and selection.state("broadcast-math-hoist") is not False,
    }


def bind_runner(source, *, features):
    return source.replace("__MERLIN_AUTO_HOST_CHOICES__", repr(choices(features)))


def require_report(stdout, *, features):
    rows = [line[len(TOKEN) :] for line in stdout.splitlines() if line.startswith(TOKEN)]
    if len(rows) != 1:
        raise ValueError("automatic host optimization completion is absent or ambiguous")
    report = json.loads(rows[0])
    expected = choices(features)
    hashes = [report.get(key) for key in ("source_sha256", "output_sha256")]
    if (
        report.get("schema") != "merlin.automatic_host_features.v1"
        or report.get("choices") != expected
        or set(report.get("rewrites", {})) != set(expected)
        or any(type(v) is not int or v < 0 for v in report["rewrites"].values())
        or any(report["rewrites"][k] for k, enabled in expected.items() if not enabled)
        or any(
            not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef") for value in hashes
        )
        or report.get("hardware_cycles") != "not_measured"
    ):
        raise ValueError("automatic host rewrite report differs from selected compilation")
    return report


RUNNER_PRELUDE = (
    tensor_rewrite_source()
    + r"""
_AUTO_HOST_CHOICES = __MERLIN_AUTO_HOST_CHOICES__
_AUTO_HOST_ORIGINAL = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    import json as _auto_json
    import hashlib as _auto_hashlib
    passes = [p for p in pipeline.split(',') if p]
    marker = '__merlin_automatic_scalar_host__'
    if marker not in passes:
        return _AUTO_HOST_ORIGINAL(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(marker) != 1:
        raise ValueError('automatic host stage is duplicated')
    at = passes.index(marker)
    head = passes[:at]
    generalizes = any('linalg-generalize-named-ops' in p for p in head)
    _AUTO_HOST_ORIGINAL(ctx, module, ','.join(head), 0, (), (), (),
                        pre_generalize if generalizes else ())
    before = str(module)
    # Call the existing typed rewrites, including their own shape/use/effect
    # refusals. No pattern matching by source name or captured provenance.
    if not generalizes and (_AUTO_HOST_CHOICES['contraction'] or _AUTO_HOST_CHOICES['squared_sum']):
        _AUTO_HOST_ORIGINAL(ctx, module, 'func.func(linalg-generalize-named-ops)',
                            0, (), (), (), pre_generalize)
    counts = _merlin_optimize_host_tensors(ctx, module, _AUTO_HOST_CHOICES)
    report = {
        'schema': 'merlin.automatic_host_features.v1',
        'choices': _AUTO_HOST_CHOICES, 'rewrites': counts,
        'source_sha256': _auto_hashlib.sha256(before.encode()).hexdigest(),
        'output_sha256': _auto_hashlib.sha256(str(module).encode()).hexdigest(),
        'selection_basis': 'typed_source_exactness_and_reduced_recurrence_materialization',
        'unselected_cost_alternatives': ['packet_widths', 'reduction_unrolling', 'lookup_tables'],
        'hardware_cycles': 'not_measured',
    }
    print('OK automatic_scalar_host ' + _auto_json.dumps(report, sort_keys=True))
    _AUTO_HOST_ORIGINAL(ctx, module, ','.join(passes[at+1:]), erase, mid, late, post_openmp, ())
"""
)
