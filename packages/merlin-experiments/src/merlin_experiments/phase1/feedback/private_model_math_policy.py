"""Select native-build host arithmetic from the private spec's sealed references.

This binds policy bytes, not capture execution or supplier qualification. Existing
capture, linked-image, instruction and mathematical-supplier gates remain required.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import yaml

from merlin.common.digest import is_sha256
from merlin.model_check import bundle as B
from merlin.model_check import models as M
from merlin.runtime.host_math import host_math_recipe


def select(document, spec, row, *, target, programs):
    selection = document.get("numerics_bundle")
    if selection is None:
        return {
            "policy": "native",
            "scope": "legacy unselected native arithmetic",
            "pins": [],
            "recipe": recipe_identity("native"),
            "reference_receipts_verified": 0,
        }
    if not isinstance(selection, dict) or not isinstance(selection.get("path"), str):
        raise ValueError("selected numerics_bundle requires a path; host math cannot fall back to native")
    pins = {}

    def read(path, expected=None):
        path = Path(path)
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"selected host-math input differs: {path}")
        pins[str(path)] = digest
        return raw

    if json.loads(json.dumps(document)) != yaml.safe_load(read(spec)):
        raise ValueError("private host-math selection differs from its specification bytes")
    root = Path(selection["path"])
    root = root if root.is_absolute() else Path(spec).parent / root
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(read(manifest_path, selection.get("manifest_sha256")))
    if manifest.get("schema") != B.MANIFEST_SCHEMA or manifest.get("target") != target:
        raise ValueError("selected host-math bundle names a different schema or target")
    seal = manifest.get("seal") or {}
    sums_name = Path(seal.get("sums", "SHA256SUMS"))
    if sums_name.is_absolute() or ".." in sums_name.parts or not is_sha256(seal.get("sums_sha256")):
        raise ValueError("selected host-math bundle has no valid seal")
    sums = read(root / sums_name, seal["sums_sha256"])
    if selection.get("sums_sha256") not in (None, sha256(sums).hexdigest()):
        raise ValueError("selected host-math sums differ from private specification")
    expected = {}
    for line in sums.decode().splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not is_sha256(digest) or name in expected:
            raise ValueError("selected host-math seal has malformed or duplicate members")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("selected host-math seal member escapes bundle")
        expected[name] = digest
    config = root / "configuration/models.yaml"
    config_bytes = read(config, expected.get("configuration/models.yaml") or "missing sealed configuration")
    if selection.get("models_sha256") not in (None, pins[str(config)]):
        raise ValueError("selected host-math models differ from private specification")
    models = M.load(config, source_bytes=config_bytes)
    model = models.model(row["id"])
    if models.target != target or tuple(p.name for p in model.programs) != tuple(programs):
        raise ValueError("selected host-math model target/program roster differs from native build")
    roots = manifest.get("selected_capture_roots") or {}
    if row["id"] in roots and Path(roots[row["id"]]) / "run/capture" != Path(row["capture"]):
        raise ValueError("selected host-math bundle names a different model capture")
    policy = M.selected_host_math_policy(model, models.full)
    references = [
        name
        for name in expected
        if len(Path(name).parts) == 4
        and Path(name).parts[0] == model.name
        and Path(name).parts[1] in {"visible", "sealed"}
        and Path(name).name == "int8.npz"
    ]
    if policy != "native" and not references:
        raise ValueError("portable host math requires sealed reference numerical contracts")
    selected_recipe = recipe_identity(policy)
    verified = 0
    for name in references:
        receipt_name = str(Path(name).with_name("reference_numeric_contract.json"))
        if receipt_name not in expected:
            if policy != "native":
                raise ValueError("portable host math requires its sealed reference numerical contract")
            continue
        path = root / receipt_name
        receipt = json.loads(read(path, expected[receipt_name]))
        contract = receipt.get("numeric_contract") or {}
        if receipt.get("schema") != "merlin.model_check.reference_numeric_contract.v1":
            raise ValueError("selected numerical contract has a different schema")
        actual_policy = (contract.get("host_math") or {}).get("policy", "native")
        if actual_policy != policy or receipt.get("outputs_sha256") != expected[name]:
            raise ValueError("selected models host math differs from sealed reference policy/output identity")
        if policy != "native":
            if contract["host_math"].get("scalar_source_sha256") != selected_recipe["source_sha256"]:
                raise ValueError("sealed reference differs from selected scalar recipe snapshot")
            from merlin.model_check.full_quantized import _numeric_receipt

            _numeric_receipt(path, expected[receipt_name], expected[name], policy)
        verified += 1
    proof = {
        "reference_receipts_verified": verified,
        "recipe": selected_recipe,
        "policy": policy,
        "scope": "private-spec selected sealed policy only; no supplier qualification",
        "pins": [{"path": path, "sha256": digest} for path, digest in pins.items()],
    }
    recheck(proof)
    return proof


def recipe_identity(policy):
    recipe = host_math_recipe(policy)
    return {
        "source_sha256": sha256(recipe.source.encode()).hexdigest() if recipe.source is not None else None,
        "compile_flags": list(recipe.compile_flags),
        "link_flags": list(recipe.link_flags),
    }


def recheck(selection):
    if recipe_identity(selection["policy"]) != selection["recipe"]:
        raise ValueError("selected host-math scalar source/flags changed during native build")
    for pin in selection["pins"]:
        if sha256(Path(pin["path"]).read_bytes()).hexdigest() != pin["sha256"]:
            raise ValueError("selected host-math policy/reference bytes changed during native build")


def build_options(selection, options, symbols):
    recheck(selection)
    policy = selection["policy"]
    if options.get("host_math_policy", policy) != policy:
        raise ValueError("native build host_math_policy conflicts with selected sealed model/reference policy")
    recipe = host_math_recipe(policy)
    wrapped = {flag.removeprefix("-Wl,--wrap=") for flag in recipe.link_flags if flag.startswith("-Wl,--wrap=")}
    conflict = sorted(wrapped.intersection(symbols or ()))
    if conflict:
        raise ValueError(
            f"selected host_math_policy={policy} replaces {conflict}, but source requires "
            "selected_math_archive supplier proof; select a matching portable-provider contract "
            "before building (native fallback is forbidden)"
        )
    return {**options, "host_math_policy": policy}


def bind(document, spec, row, target, program_options, checks):
    selected = select(document, spec, row, target=target, programs=program_options)
    checks["host_math_policy"] = selected
    for program, options in program_options.items():
        program_options[program] = build_options(selected, options, ())
    return selected


def require_sources(selection, sources):
    recheck(selection)
    if selection["policy"] == "native":
        return
    from . import private_linkage_support

    for source in sources.values():
        build_options(selection, {}, private_linkage_support.symbols(source))
