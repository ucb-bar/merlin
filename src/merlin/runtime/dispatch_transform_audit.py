"""Exact retained whole-model transform stages and independent structural replay.

The runtime executes the normalized/outlined module; these helpers only archive
and verify the same stages. Their public compatibility imports remain in
``dispatch_runtime`` so existing callers retain one entry point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _validate_normalization_recipe(recipe: dict) -> dict:
    required = {
        "int8_compute",
        "quant_passes",
        "prequant_gather",
        "selection_policy",
    }
    cpu_fields = {
        "torchao_cpu_avx2_mean",
        "torchao_cpu_avx2_layer_norm",
        "torch_source_commit",
        "torch_cpu_capability",
    }
    if not isinstance(recipe, dict) or not required <= set(recipe) or not set(recipe) <= required | cpu_fields:
        raise ValueError("model normalization recipe is malformed")
    recipe = {
        **{
            "torchao_cpu_avx2_mean": False,
            "torchao_cpu_avx2_layer_norm": False,
            "torch_source_commit": "",
            "torch_cpu_capability": "",
        },
        **recipe,
    }
    if (
        type(recipe["int8_compute"]) is not bool
        or type(recipe["prequant_gather"]) is not bool
        or recipe["selection_policy"] not in {"all", "custom_unreplayable"}
        or (
            recipe["quant_passes"] is not None
            and (
                not isinstance(recipe["quant_passes"], list)
                or any(not isinstance(name, str) or not name for name in recipe["quant_passes"])
            )
        )
        or (recipe["prequant_gather"] and not recipe["int8_compute"])
        or type(recipe["torchao_cpu_avx2_mean"]) is not bool
        or type(recipe["torchao_cpu_avx2_layer_norm"]) is not bool
        or not isinstance(recipe["torch_source_commit"], str)
        or not isinstance(recipe["torch_cpu_capability"], str)
        or (
            (recipe["torchao_cpu_avx2_mean"] or recipe["torchao_cpu_avx2_layer_norm"])
            and not recipe["torch_source_commit"]
        )
        or (recipe["torchao_cpu_avx2_layer_norm"] and not recipe["torch_cpu_capability"])
    ):
        raise ValueError("model normalization recipe has invalid options")
    return dict(recipe)


def record_model_transform_audit(
    source: Path,
    workdir: Path,
    normalized,
    outlined,
    *,
    enabled: bool | str,
    expected_source_sha256: str | None = None,
    normalization_recipe: dict | None = None,
) -> Path | None:
    """Retain exact captured, normalized, and outlined IR for later validation.

    The audit binds bytes and a source-to-output sequence; it does not claim
    value equivalence. Semantic receipts must cite these exact stages and be
    replayed independently before they can make that stronger claim.
    """
    if enabled is False:
        return None
    if enabled == "compact":
        raise ValueError("compact model transform audit keeps hashes but not exact IR, so it cannot be replayed")

    from ..common.ir_audit import IrAudit
    from ..xdsl_dialects._common import text as to_text

    source_text = source.read_text(encoding="utf-8")
    if (
        expected_source_sha256 is not None
        and hashlib.sha256(source_text.encode("utf-8")).hexdigest() != expected_source_sha256
    ):
        raise ValueError("model source changed between parse and transformation audit")
    with IrAudit(
        workdir,
        enabled=enabled,
        producer="merlin.runtime.dispatch_runtime.run_model",
        source=__file__,
        sidecars=(source,),
    ) as audit:
        if normalization_recipe is not None:
            audit.accounting(
                "normalization-recipe",
                {
                    "schema": "merlin.model_normalization_recipe.v1",
                    **_validate_normalization_recipe(normalization_recipe),
                },
            )
        audit.stage("captured-model", source_text)
        # Generic MLIR is the replayable serialization: xDSL's custom printer
        # emits operations this permissive parser does not implement.
        audit.stage("normalized-model", to_text(normalized, generic=True))
        audit.stage("outlined-model", to_text(outlined.module, generic=True))
    return audit.directory / "index.json" if audit.directory is not None else None


def qualify_model_transform_audit(index_path: Path) -> dict[str, Any]:
    """Replay the archived normalization and outline when their recipe is available.

    This checks archival integrity, MLIR validity, and deterministic pass output.
    It does not prove that a pass preserves values.
    """
    from tempfile import TemporaryDirectory

    from ..frontends.linalg_mlir import parse_mlir_file
    from ..xdsl_dialects._common import text as to_text
    from ..xdsl_dialects.lowering.outline import outline_dispatches

    index_path = Path(index_path)
    if index_path.is_symlink() or not index_path.is_file() or index_path.name != "index.json":
        raise ValueError("model transform audit index is absent or indirect")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("producer") != "merlin.runtime.dispatch_runtime.run_model"
        or index.get("outcome") != "completed"
        or index.get("mode") not in {"exact", "both"}
    ):
        raise ValueError("model transform audit did not complete with exact IR")
    stages = index.get("stages")
    expected = ("captured-model", "normalized-model", "outlined-model")
    if (
        not isinstance(stages, list)
        or tuple(stage.get("name") for stage in stages if isinstance(stage, dict)) != expected
        or len(stages) != 3
    ):
        raise ValueError("model transform audit has no exact three-stage sequence")
    sidecars = index.get("sidecars")
    if (
        not isinstance(sidecars, list)
        or len(sidecars) != 1
        or not isinstance(sidecars[0], dict)
        or sidecars[0].get("sha256") != stages[0].get("sha256")
    ):
        raise ValueError("model transform audit does not bind the captured source bytes")
    contents = []
    for ordinal, stage in enumerate(stages):
        name = stage["name"]
        file = stage.get("file")
        digest = stage.get("sha256")
        size = stage.get("bytes")
        if (
            file != f"{ordinal:03d}-{name}.mlir"
            or stage.get("representation") != "exact-ir"
            or stage.get("format") != "mlir"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or type(size) is not int
            or size < 0
        ):
            raise ValueError(f"model transform audit has malformed {name} stage")
        path = index_path.parent / file
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"model transform audit {name} stage is absent or indirect")
        raw = path.read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"model transform audit {name} stage differs from its index")
        contents.append(raw.decode("utf-8"))
    parsed_stages = []
    for name, content in zip(expected, contents, strict=True):
        try:
            # Captured MLIR can use tensor custom syntax that xDSL cannot parse
            # directly. Production uses parse_mlir_file's generic-form fallback;
            # replay the same parser on a private copy of the SHA-checked stage
            # so qualification never writes sidecars into the archived receipt.
            with TemporaryDirectory(prefix="merlin-model-transform-audit-") as scratch:
                stage = Path(scratch) / f"{name}.mlir"
                stage.write_text(content, encoding="utf-8")
                parsed = parse_mlir_file(stage)
                parsed.verify()
                parsed_stages.append(parsed)
        except Exception as exc:
            raise ValueError(f"model transform audit {name} stage is not valid MLIR: {exc}") from exc
    normalization_replay = "not_recorded"
    receipts = index.get("accounting_receipts", [])
    if not isinstance(receipts, list):
        raise ValueError("model transform audit receipts are malformed")
    recipe_descriptors = [
        row for row in receipts if isinstance(row, dict) and row.get("name") == "normalization-recipe"
    ]
    if len(recipe_descriptors) > 1:
        raise ValueError("model transform audit has duplicate normalization recipes")
    if recipe_descriptors:
        descriptor = recipe_descriptors[0]
        if (
            descriptor.get("file") != "normalization-recipe.json"
            or descriptor.get("schema") != "merlin.model_normalization_recipe.v1"
            or type(descriptor.get("bytes")) is not int
            or not isinstance(descriptor.get("sha256"), str)
        ):
            raise ValueError("model transform audit normalization recipe descriptor is malformed")
        recipe_path = index_path.parent / descriptor["file"]
        if recipe_path.is_symlink() or not recipe_path.is_file():
            raise ValueError("model transform audit normalization recipe is absent or indirect")
        raw = recipe_path.read_bytes()
        if len(raw) != descriptor["bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("model transform audit normalization recipe differs from its index")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("schema") != descriptor["schema"]:
            raise ValueError("model transform audit normalization recipe schema differs from its index")
        recipe = _validate_normalization_recipe({key: val for key, val in payload.items() if key != "schema"})
        if recipe["selection_policy"] == "all":
            # The replay invokes the same runtime normalizer, but only on
            # independently re-parsed captured bytes, never on archived IR.
            from ..llvmlower.quant_passes import known
            from .dispatch_runtime import _normalize_model_module

            replay_normalized = parsed_stages[0]
            # Archived v1 None selected all six. New receipts retain their
            # effective pass selection explicitly; never reinterpret old bytes.
            replay_passes = (
                list(known()) if recipe["int8_compute"] and recipe["quant_passes"] is None else recipe["quant_passes"]
            )
            _normalize_model_module(
                replay_normalized,
                int8_compute=recipe["int8_compute"],
                quant_passes=replay_passes,
                prequant_gather=recipe["prequant_gather"],
                torchao_cpu_avx2_mean=recipe["torchao_cpu_avx2_mean"],
                torchao_cpu_avx2_layer_norm=recipe["torchao_cpu_avx2_layer_norm"],
                torch_source_commit=recipe["torch_source_commit"],
                torch_cpu_capability=recipe["torch_cpu_capability"],
            )
            if to_text(replay_normalized, generic=True) != contents[1]:
                raise ValueError("archived normalization differs from replay on the exact captured MLIR")
            normalization_replay = "matched"
        else:
            normalization_replay = "unsupported_custom_selection"
    replay = to_text(outline_dispatches(parsed_stages[1]).module, generic=True)
    if replay != contents[2]:
        raise ValueError("archived outline differs from replay on the exact normalized MLIR")
    return {
        "schema": "merlin.model_transform_audit_qualification.v1",
        "status": "structural_replay_matched",
        "stage_sha256": {name: stage["sha256"] for name, stage in zip(expected, stages, strict=True)},
        "normalization_replay": normalization_replay,
        "semantic_equivalence": "not_proven",
    }
