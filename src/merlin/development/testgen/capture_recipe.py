"""Select exactly one verified software-scoped capture recipe from frozen evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _selected_capture_recipe(evidence_root: Path, *, target: str, operand_dtype: str, accumulator_dtype: str) -> dict:
    """Read exactly one frozen SW-scoped recipe for a live integer model capture.

    Hardware-only recipe derivation is intentionally not a fallback: it can quantize families
    the selected software spec explicitly left on the host.
    """
    from merlin.common import quant_formats
    from merlin.compiler.quantization import recipe as quant_recipe

    index_path = evidence_root / "software" / "quantization-recipes.json"
    index = json.loads(index_path.read_bytes())
    if index.get("schema") != "merlin.phase0.capture_recipes.v1" or index.get("target") != target:
        raise ValueError("frozen capture-recipe index has wrong schema or target")
    wanted_operand = quant_formats.get(operand_dtype).name
    wanted_accumulator = quant_formats.get(accumulator_dtype).name
    matches = []
    for row in index.get("recipes") or []:
        rel = Path(str(row.get("path") or ""))
        if rel.is_absolute() or not rel.parts or ".." in rel.parts or rel.parts[0] != "software":
            raise ValueError("frozen capture recipe has an unsafe relative path")
        path = evidence_root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError("frozen capture recipe is absent or symlinked")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != row.get("sha256") or path.stem != actual:
            raise ValueError("frozen capture recipe bytes differ from their index")
        recipe = json.loads(raw)
        if recipe.get("target") != target or recipe.get("status") != quant_recipe.DERIVED:
            raise ValueError("frozen capture recipe has wrong target or unresolved status")
        digest = quant_recipe.digest(recipe)
        if recipe.get("recipe_sha256") != digest or row.get("recipe_sha256") != digest:
            raise ValueError("frozen capture recipe semantic digest differs from its index")
        if (
            quant_formats.get(recipe["weight"]["dtype"]).name == wanted_operand
            and quant_formats.get(recipe["activation"]["dtype"]).name == wanted_operand
            and quant_formats.get(recipe["accumulator_dtype"]).name == wanted_accumulator
        ):
            matches.append(recipe)
    if len(matches) != 1:
        raise ValueError(
            f"frozen evidence has {len(matches)} unambiguous SW-scoped recipes for "
            f"{wanted_operand}/{wanted_accumulator}; refusing a hardware-only default"
        )
    return matches[0]


def recipe_outputs(candidates, *, target: str, software_review, encode) -> dict[str, bytes]:
    """Frozen-evidence members for derived capture recipes: one file per recipe and their index.

    A reviewed model variant of a methodology recipe (``candidate["model_variants"]``) gets its own
    file, indexed under ``model_variants`` apart from ``recipes`` so that the model-independent
    selection (:func:`_selected_capture_recipe`) stays unambiguous.
    """
    outputs: dict[str, bytes] = {}
    recipes, variants = [], []

    def member(recipe) -> tuple[str, str]:
        payload = encode(recipe)
        digest = hashlib.sha256(payload).hexdigest()
        path = f"software/quantization-recipes/{digest}.json"
        outputs[path] = payload
        return path, digest

    for candidate in candidates:
        path, digest = member(candidate["recipe"])
        recipes.append(
            {key: value for key, value in candidate.items() if key not in ("recipe", "model_variants")}
            | {"path": path, "sha256": digest, "recipe_sha256": candidate["recipe"]["recipe_sha256"]}
        )
        for variant in candidate.get("model_variants") or ():
            path, digest = member(variant["recipe"])
            variants.append(
                {"format_id": candidate["format_id"], "model": variant["model"], "name": variant["name"]}
                | {"path": path, "sha256": digest, "recipe_sha256": variant["recipe"]["recipe_sha256"]}
            )
    outputs["software/quantization-recipes.json"] = encode(
        {
            "schema": "merlin.phase0.capture_recipes.v1",
            "target": target,
            "software_review": software_review,
            "recipes": recipes,
            **({"model_variants": variants} if variants else {}),
            "qualification": "diagnostic transformation inputs; calibrate and verify actual precision before admission",
        }
    )
    return outputs


def _with_selected_model_recipe(
    entry: dict,
    *,
    evidence_root: Path | None,
    target: str,
    binding,
) -> dict:
    """Use the frozen SW-scoped recipe when an integer model has no explicit choice."""
    if (
        evidence_root is None
        or (entry.get("kind") != "model" and entry.get("op") != "model")
        or entry.get("materialized_capture")
        or entry.get("quant_recipe")
        or entry.get("quant_scheme")
        or not binding.integer
    ):
        return entry
    return {
        **entry,
        "quant_recipe": _selected_capture_recipe(
            evidence_root,
            target=target,
            operand_dtype=str(entry.get("operand_dtype") or binding.operand_dtype),
            accumulator_dtype=str(binding.accum_dtype),
        ),
    }


def _prepare_model_capture_entry(
    entry: dict,
    *,
    evidence_root: Path | None,
    target: str,
    binding,
    capture=None,
) -> dict:
    """Check a selected static model capture tool before any capsule writer starts."""
    if entry.get("kind") != "model" and entry.get("op") != "model":
        return entry
    if entry.get("materialized_capture"):
        return entry

    from merlin.capture import numerics as source_numerics
    from merlin.capture import pytorch as source_pytorch
    from merlin.capture import receipts as source_receipts

    try:
        selected = _with_selected_model_recipe(entry, evidence_root=evidence_root, target=target, binding=binding)
    except Exception:  # noqa: BLE001 -- the writer records this entry's recipe failure as before
        return entry
    recipe = selected.get("quant_recipe") or source_numerics.derived_recipe(
        getattr(binding, "target", None), str(selected.get("operand_dtype") or binding.operand_dtype)
    )
    if source_receipts._static_pt2e_model(
        "model",
        scheme=None if recipe is not None else selected.get("quant_scheme"),
        recipe=recipe,
        already_quantized=selected.get("capture_quantization") == "already_materialized",
    ):
        capture = capture if capture is not None else source_pytorch.PytorchRefSource()
        integerizer = capture.m2m_dir / "m2m/capture/pt2e_integerize.py"
        if capture.available() and not integerizer.is_file():
            raise ValueError(
                f"selected model2MLIR checkout {capture.m2m_dir} lacks {integerizer}; "
                "static int8 model capture requires m2m.capture.pt2e_integerize"
            )
        if capture.available() and not entry.get("micro_model"):
            # The PT2E quantizer imports these from the selected checkout only
            # after source capsules start writing. The derived micro-model is
            # the sole exception: its graph is checked at capture time for
            # complete ATen provenance and absence of BatchNorm before Merlin
            # permits a missing fold API. Every other static model must have
            # the API before any writer runs.
            probe = subprocess.run(
                [
                    str(capture.python),
                    "-B",
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from m2m.capture.trace import "
                    "pt2e_conv_bn_fold_candidates, attach_pt2e_conv_bn_folds",
                    str(capture.m2m_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if probe.returncode:
                raise ValueError(
                    f"selected model2MLIR checkout {capture.m2m_dir} lacks the required "
                    "m2m.capture.trace PT2E fold-provenance API "
                    "(pt2e_conv_bn_fold_candidates, attach_pt2e_conv_bn_folds); "
                    "select a compatible checkout and interpreter before generating capsules"
                )
    return selected
