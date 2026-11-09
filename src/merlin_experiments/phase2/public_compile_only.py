"""The inherited ordinary no-data compiler probe outside queued search mode."""

from __future__ import annotations

import tempfile
from pathlib import Path

from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback import compile_only as ORIGINAL

from .broker import BrokerAction
from .contracts import StageGateError

ACTION = "compile-only"


def action():
    return BrokerAction(
        ACTION,
        ("__host_public_compile_only__", "{capsules}"),
        ("capsules",),
        "compile original public scale/frontier tensor sources and inspect actual linked products; "
        "no input values, simulator, numerical/RTL acceptance or timing promotion",
        False,
    )


def service(
    frozen_functional, *, candidate, corpus, target_experiment, contract, facts_path, work_root, sandbox_inputs=None
):
    from .functional_inputs import _verify_private_functional_provenance

    if frozen_functional.host_provenance is None:
        return None
    selected = _verify_private_functional_provenance(frozen_functional.host_provenance)
    if selected["bundle"].get("compiler_tool_profile") is None:
        return None
    owner = CommonTransformService(
        workspace=selected["workspace"], bundle_manifest=selected["bundle_path"], repo=selected["repo"]
    )
    members = tuple(corpus.capsules)

    def invoke(capsules, timeout_s):
        from .corpus import verify_frozen_performance_corpus

        if not isinstance(capsules, str) or not capsules or len(capsules) > 4096:
            raise StageGateError("compile-only selects bounded original public member identifiers")
        verify_frozen_performance_corpus(corpus)
        owner._verify()
        rows = [{**member.descriptor, "__dir__": str(member.source_dir)} for member in members]
        root = Path(work_root)
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="compile-", dir=root) as temporary:
            selected_compiler = ORIGINAL.snapshot_submission(candidate, Path(temporary) / "compiler")
            result = ORIGINAL.observe(
                candidate=selected_compiler,
                target_experiment=target_experiment,
                capsules=rows,
                contract=contract,
                facts_path=facts_path,
                common_service=owner,
                work_root=Path(temporary) / "products",
                timeout_s=min(timeout_s, 600),
                selected=capsules,
                sandbox_inputs=sandbox_inputs,
            )
        owner._verify()
        verify_frozen_performance_corpus(corpus)
        return result

    return invoke
