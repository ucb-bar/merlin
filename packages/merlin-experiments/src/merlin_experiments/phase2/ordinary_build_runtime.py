"""Optional original build view and product reuse for retained public revisions.

The original admission and numerical/timing routes remain authoritative. Only
live core owners can restore build products; final and calibration omit reuse.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from .contracts import StageGateError, canonical_json


def binding(policy, *, include=True):
    view, index = policy.build_runtime_view, policy.build_product_index
    return {
        "build_runtime_view": view.verify() if include and view is not None else None,
        "build_product_index": (
            index.verify(build_owner=view, context=policy.artifact_retest_context)
            if include and index is not None
            else None
        ),
    }


def install(policy, view, index=None):
    from merlin.runtime.build_runtime import BuildRuntimeView
    from merlin.targetgen.contract.build_products import BuildProductIndex

    if (
        type(view) is not BuildRuntimeView
        or policy.build_runtime_view is not None
        or policy.evaluation_pool is not None
    ):
        raise StageGateError("normal build view needs its exact live original owner before queue admission")
    view.verify()
    if index is not None:
        if type(index) is not BuildProductIndex:
            raise StageGateError("normal build products need their exact live original index")
        index.verify(build_owner=view, context=policy.artifact_retest_context)
    policy.build_runtime_view, policy.build_product_index = view, index
    policy._selection = canonical_json(policy._binding())
    policy.validate_search_inputs()


@contextmanager
def executing(policy, job_root, *, reuse_products):
    if policy.build_runtime_view is None:
        yield
        return
    from merlin.runtime.build_runtime import executing_build_runtime
    from merlin.targetgen.contract.build_products import executing_build_products

    temporary = Path(os.environ["TMPDIR"]) if os.environ.get("TMPDIR") else None
    storage_roots = (
        (temporary,)
        if temporary is not None
        and temporary.is_dir()
        and temporary.is_relative_to(policy.build_runtime_view.output_root)
        else ()
    )
    with (
        executing_build_runtime(policy.build_runtime_view, storage_roots=storage_roots),
        executing_build_products(policy.build_product_index if reuse_products else None, job_root=Path(job_root)),
    ):
        yield


def finish(policy, job_root, *, completed):
    """The queue parent calls only after real cleanup and durable release."""
    owners = [owner for owner in (policy.artifact_retest_index, policy.build_product_index) if owner is not None]
    key = "published" if completed else "discarded"
    if not owners:
        return {"status": "UNAVAILABLE", key: 0}
    count, failures = 0, []
    for owner in owners:
        try:
            count += (
                owner.publish_completed_job(Path(job_root)) if completed else owner.discard_cleaned_job(Path(job_root))
            )
        except Exception as error:  # noqa: BLE001 - cache failure cannot change the completed job verdict
            failures.append(type(error).__name__)
    return {"status": "UNKNOWN" if failures else key, key: count, "failures": failures}


def close_stage(policy):
    """After final verification, release only owners whose original jobs closed."""
    if policy.build_runtime_view is None:
        return
    pool = policy.evaluation_pool
    if pool is not None and any(not row["terminal"] for row in pool.observations()):
        raise StageGateError("build lookup cleanup UNKNOWN; original owners remain retained")
    if policy.build_product_index is not None:
        policy.build_product_index.close()
    policy.build_runtime_view.close()
