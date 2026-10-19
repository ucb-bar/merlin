"""Opt-in queued search around the existing normal performance authoring round."""

from . import archive_screen_selection as ARCHIVE
from . import build_view_configuration as BUILD
from .contracts import StageGateError, mapping_file
from .edit_authority import FrozenEditAuthority
from .ordinary_search_policy import prepare_ordinary_search_policy


def prepare_round(
    selection,
    *,
    stage_root,
    candidate,
    baseline,
    corpus,
    edit_contract,
    functional_runs_root,
    functional_run_id,
    functional_submission_sha256,
    target_experiment,
    receipt_path,
    contract_root,
    certificate_path,
    certificate_sha256,
    rtl_facts_path,
    services,
    e2e_sentinel,
    tuning_call_budget,
    waive_functional_gate,
    whole_model_inputs,
    external_boundary=None,
    archive_screen_selection=None,
):
    """Reuse the normal Phase 1 fork, runtime, corpus and final receipt machinery."""
    if edit_contract is None:
        raise StageGateError("queued normal search requires an explicit compiler edit contract")
    authority_root = stage_root / "_search_edit_authority"
    authority_root.mkdir(mode=0o700)
    authority = FrozenEditAuthority(authority_root, guidance_contract=contract_root)
    authority.freeze(candidate, mapping_file(edit_contract), has_iterations=False)
    from .compiler_library_guidance import publish_edit_authority

    publish_edit_authority(authority, candidate)
    policy = prepare_ordinary_search_policy(
        functional_runs_root=functional_runs_root,
        functional_run_id=functional_run_id,
        functional_submission_sha256=functional_submission_sha256,
        corpus=corpus,
        baseline=baseline,
        candidate=candidate,
        target_experiment=target_experiment,
        receipt_path=receipt_path,
        work_root=stage_root / "_development_feedback",
        edit_authority=authority,
        contract_root=contract_root,
        certificate_path=certificate_path,
        certificate_sha256=certificate_sha256,
        rtl_facts_path=rtl_facts_path,
        services=services,
        e2e_sentinel=e2e_sentinel,
        tuning_call_budget=tuning_call_budget,
        waive_functional_gate=waive_functional_gate,
        whole_model_inputs=whole_model_inputs,
    )
    BUILD.install(selection, policy, root=stage_root / "_build_runtime", output_root=stage_root)
    ARCHIVE.install(archive_screen_selection, policy)
    pool = selection.install(
        policy, edit_authority=authority, root=stage_root / "_search", external_boundary=external_boundary
    )
    return policy, pool
