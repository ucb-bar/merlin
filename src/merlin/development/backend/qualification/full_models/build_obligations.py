"""Build-only qualification against operator-private frozen model inputs.

Full-model execution and representative-capsule L3 grades remain independent.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from merlin.development.backend.qualification.full_models import audits as program_checks
from merlin.development.backend.qualification.full_models import (
    private_compilation_inputs as compilation_support,
)
from merlin.development.backend.qualification.full_models import (
    private_source_optimizations as source_optimizations,
)
from merlin.development.backend.qualification.full_models import private_static_products as static_products
from merlin.development.backend.qualification.full_models.audits import (
    audit_built_device_host_compute as _audit_built_device_host_compute,
)
from merlin.development.backend.qualification.full_models.audits import load_diagnostic_receipt
from merlin.development.backend.qualification.full_models.audits import (
    require_linked_symbols as _linked_symbols,
)
from merlin.development.backend.qualification.full_models.audits import (
    require_static_build_inputs as _require_static_build_inputs,
)
from merlin.development.backend.qualification.full_models.completion import (
    BUILD_BOARD_SCOPE,
    RESULT_SCHEMA,
    _complete_capture_identity,
    compile_saved_model,
    complete,
)
from merlin.development.backend.qualification.full_models.private_capture_roster import (
    captured_input_provenance as _captured_input_provenance,
)
from merlin.development.backend.qualification.full_models.private_capture_roster import (
    captured_programs as _captured_programs,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _capture_tree_bindings as _verify_capture_bindings,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _expected_input_provenance,
    _require_selected_input_bindings,
    _selected_admission_views,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _recipe_derivation as _derive_recipe_binding,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _tree as _verify_input_tree,
)
from merlin.development.backend.qualification.full_models.private_source_preflight import (
    _notify as _source_progress,
)
from merlin.development.backend.qualification.full_models.private_source_preflight import (
    source_result as _source_result,
)
from merlin.development.backend.qualification.provenance import private_model_math_policy as math_policy
from merlin.development.backend.qualification.provenance import private_source_demand as source_demand
from merlin.development.backend.qualification.provenance import private_source_freeze as source_freeze_api
from merlin.development.backend.qualification.provenance.groups import (
    convolution_input_preparations,
    source_group_roster,
    verify_routed_source_coverage,
)
from merlin.development.backend.qualification.provenance.groups import (
    eligible_source_identity as _eligible_source_identity,
)
from merlin.development.backend.qualification.provenance.groups import (
    join_routed_source_groups as _join_routed_source_groups,
)
from merlin.development.backend.qualification.provenance.groups import (
    routed_kernel_symbols as _routed_kernel_symbols,
)
from merlin.development.backend.qualification.provenance.private_source_demand import (
    exact_source_groups as _exact_source_groups,
)
from merlin.development.backend.qualification.source_witnesses import bucketize as bucketize_support
from merlin.development.backend.qualification.source_witnesses import control_guards as control_support
from merlin.development.backend.qualification.source_witnesses import data_movement
from merlin.development.backend.qualification.source_witnesses import integer_reductions as integer_support
from merlin.development.backend.qualification.source_witnesses import linalg_bodies as linalg_support
from merlin.development.backend.qualification.source_witnesses import ordered_scan as ordered_scan_support
from merlin.development.backend.qualification.source_witnesses import (
    ordered_scan_admission as _merlin_ordered_scan_admission_1,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_f32_maximum_support as maximum_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_f32_reduction_support as f32_sum_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_host_source_dispatch as host_source_dispatch,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_index_host_support as index_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_linkage_support as linkage_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_linked_elf_selection as linked_policy,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_literal_arange_admission as arange_support,
)
from merlin.development.backend.qualification.source_witnesses import pure_stages as pure_stage
from merlin.development.backend.qualification.source_witnesses.integer_reductions import (
    contraction_zero_init_fill as _contraction_zero_init_fill,
)
from merlin.development.backend.qualification.source_witnesses.private_source_support_join import (
    linked_source_support_complete as _linked_source_support_complete,
)
from merlin.runtime.execution.model_inputs import file_sha256, strict_tree_sha256

# ---- private_full_models -----------------------------------------------------------------------


def _verify_compiled_program(
    receipt: Mapping[str, Any],
    *,
    program: str,
    capture_path: Path | None = None,
    source: dict[str, Any],
    stage_tree: Mapping[str, Any],
    package_digest: Mapping[str, Any],
    target: str,
    board: str,
    catalog: Path,
    dts: Path,
    device_selected: bool,
    host_package: Path | None = None,
    host_package_tree_sha256: str | None = None,
    require_compilation_recipe: bool = True,
) -> dict[str, Any]:
    """Canonical post-build checks shared by freshly built and diagnostic images."""
    from merlin.compiler.lower.device.offload import BY_GROUP

    expected_route = "device_requested_dispatch_unverified" if device_selected else "host_baseline"
    if (
        receipt.get("status") != "compiled"
        or receipt.get("execution_route") != expected_route
        or receipt.get("inputs", {}).get("capture_tree") != stage_tree
    ):
        raise ValueError(f"{program} did not capture/lower/codegen/link the selected source")
    _require_static_build_inputs(receipt, target=target, board=board, catalog=catalog, dts=dts)
    if device_selected and (receipt.get("inputs", {}).get("device") or {}).get("package_tree") != package_digest:
        raise ValueError(f"{program} build used a different candidate compiler tree")
    output = receipt.get("output") or {}
    elf = Path(str(output.get("elf") or ""))
    if not elf.is_file() or file_sha256(elf) != output.get("elf_sha256"):
        raise ValueError(f"{program} linked ELF is absent or changed")
    compilation_binding = compilation_support.verify(output, elf, required=require_compilation_recipe)
    sidecar_sha = None
    linked = 0
    optional_linked = 0
    host_compute_audit = []
    if device_selected:
        sidecar_record = output.get("device_sidecar") or {}
        sidecar = Path(str(sidecar_record.get("path") or ""))
        if not sidecar.is_file() or file_sha256(sidecar) != sidecar_record.get("sha256"):
            raise ValueError(f"{program} candidate device sidecar is absent or changed")
        emitted = json.loads(sidecar.read_text(encoding="utf-8"))
        if emitted.get("granularity") != BY_GROUP or emitted.get("device") != target:
            raise ValueError(f"{program} emitted another device or a partial route")
        assigned = emitted.get("routed")
        joined = verify_routed_source_coverage(emitted, source)
        symbols = _routed_kernel_symbols(emitted)
        _linked_symbols(elf, symbols)
        source["conv_input_preparation"] = convolution_input_preparations(emitted, source)
        host_compute_audit = _audit_built_device_host_compute(
            elf.parent, assigned, source, target, host_artifacts=output.get("host_artifacts")
        )
        sidecar_sha = sidecar_record["sha256"]
        linked = len(set(joined.values()) & set(source["eligible_groups"]))
        optional_linked = len(set(joined.values()) & set(source.get("optional_groups", [])))
    linked_build = {
        "capture_tree_sha256": stage_tree["sha256"],
        "elf_sha256": output["elf_sha256"],
        "candidate_tree_sha256": package_digest["sha256"],
    }
    source_demand.link(source, linked_build, linked)
    transpose_support = source["transpose_data_support"]
    if transpose_support["status"] != "source_structural_data_support_pending_build":
        raise ValueError(f"{program} has no pending source-bound transpose support proof")
    transpose_support["status"] = "source_structural_data_support_linked"
    transpose_support["linked_build"] = linked_build
    generic_copy_support = source["generic_copy_data_support"]
    if generic_copy_support["status"] != "source_structural_data_support_pending_build":
        raise ValueError(f"{program} has no pending source-bound generic copy proof")
    generic_copy_support["status"] = "source_structural_data_support_linked"
    generic_copy_support["linked_build"] = dict(linked_build)
    from merlin.development.backend.qualification.source_witnesses.pooling import LINKED, PENDING

    pool_support = source["pool_value_support"]
    if pool_support["status"] != PENDING:
        raise ValueError(f"{program} has no pending source-bound pool-value proof")
    pool_support["status"] = LINKED
    pool_support["linked_build"] = dict(linked_build)
    pure_stage.link_direct_return(source, linked_build)
    index_lowering = control_support.link_selected_build(source, receipt, linked_build)
    linalg_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    arange_support.link(source, index_lowering, linked_build)
    integer_support.link(source, index_lowering, linked_build)
    f32_sum_support.link(source, index_lowering, linked_build)
    maximum_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    index_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    _merlin_ordered_scan_admission_1.link(source, index_lowering, linked_build)
    bucketize_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    result = {
        "program": program,
        "board": board,
        "status": "capture_lower_codegen_link_verified",
        "capture_tree_sha256": stage_tree["sha256"],
        "source_sha256": source["source_sha256"],
        "elf_sha256": output["elf_sha256"],
        "sidecar_sha256": sidecar_sha,
        "linked_device_groups": linked,
        "linked_optional_device_groups": optional_linked,
        "linked_total_device_groups": linked + optional_linked,
        "static_host_compute_audit": host_compute_audit,
        "candidate_tree_sha256": package_digest["sha256"],
        "index_lowering": index_lowering,
        "compilation_recipe": compilation_binding,
    }
    linkage_support.link_compiled(
        source, result, receipt, host_package, host_package_tree_sha256, catalog, board, dts, linked_build
    )
    return result
