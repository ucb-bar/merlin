"""Prepare explicit author information views without changing private judge data.

Only the new raw-RTL study uses this owner. Receipts stay outside the granted
view directories. The original corpus, semantics and hardware facts remain
host inputs; historical profiles keep their original declarations.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from merlin.common.digest import is_sha256


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_file(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("raw-RTL input must be a canonical regular file")
    return path


def _raw_sources(facts):
    """Read original raw-file pins, excluding extracted HW MLIR and hierarchy."""
    document = json.loads(facts.read_bytes())
    consistency = document.get("source_consistency") or {}
    if consistency.get("status") != "verified":
        raise ValueError("raw hardware lacks verified source consistency")
    rows = (document.get("inputs") or {}).get("firrtl_inputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("raw-RTL profile requires explicit original FIRRTL file pins")
    sources, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("invalid original raw RTL pin")
        path = _canonical_file(row["path"])
        if path.suffix != ".fir" or path.name in seen or _sha(path) != row["sha256"]:
            raise ValueError("raw hardware differs from its original selected pin")
        seen.add(path.name)
        sources.append(path)
    return sources


def prepare(bundle, common, *, profile, experiment, repo, corpus_paths):
    """Return a new base bundle and public grants for one unsealed release."""
    from merlin.targetgen.sandbox.bwrap import resolve_grant

    from . import author_input_views, public_runtime_abi

    if not profile.raw_rtl_study:
        raise ValueError("raw author input preparation requires its explicit versioned profile")
    result = copy.deepcopy(bundle)
    facts = _canonical_file(result["selected_rtl_facts_file"])
    destination = Path(experiment).absolute() / "public_author_inputs"
    destination.mkdir(exist_ok=False)
    raw = destination / "raw-rtl"
    raw.mkdir()
    grants, raw_pins = [], []
    for source in _raw_sources(facts):
        target = raw / source.name
        target.write_bytes(source.read_bytes())
        target.chmod(0o444)
        if _sha(target) != _sha(source):
            raise ValueError("raw hardware copy differs from selected source")
        raw_pins.append(dict(path=str(target), sha256=_sha(target)))
        grants.append(
            {
                "path": str(target),
                "mode": "ro",
                "as": "raw-rtl/" + source.name,
                "note": "selected raw hardware; no extracted facts",
            }
        )
    raw.chmod(0o555)
    result["selected_raw_rtl_files"] = raw_pins
    policy = profile.disclosure
    common, runtime_abi = public_runtime_abi.prepare(
        common, destination=destination / "runtime-protocol", repo=repo, input_format=policy["input_format"]
    )
    corpus_paths = {str(path).rstrip("/") for path in corpus_paths}
    receipts, withheld = [], []
    if policy["input_format"] == "neutral_math_v1":
        from . import neutral_public_contracts

        common, contract_pins = neutral_public_contracts.prepare(
            common, destination=destination / "protocol", repo=repo
        )
        result["neutral_public_contract_files"] = contract_pins
        runtime_abi = public_runtime_abi.bind_protocol(runtime_abi, common)
    result[public_runtime_abi.KEY] = runtime_abi
    for row in common:
        source = resolve_grant(row["path"], Path(repo)).absolute()
        hidden = (
            (not policy["rtl_facts"] and source == facts.parent)
            or (not policy["derived_target_contract"] and row.get("note") == "selected capability contract (ALL arms)")
            or (not policy["software_spec"] and row.get("note") == "admitted software semantics")
        )
        if hidden:
            withheld.append(copy.deepcopy(row))
            continue
        if not policy["derived_test_metadata"] and (
            row["path"].rstrip("/") in corpus_paths or row.get("note") == "explicit public input"
        ):
            view = destination / ("corpus-" + str(len(receipts)))
            transform = None
            if policy["input_format"] == "neutral_math_v1":
                from . import neutral_program_inputs

                transform = neutral_program_inputs.file_transform_for(source)
            receipt = author_input_views.project_tree(source, view, file_transform=transform)
            replacement = copy.deepcopy(row)
            replacement.update(path=str(view), **{"as": row.get("as") or source.name})
            grants.append(replacement)
            receipts.append(receipt)
        else:
            grants.append(copy.deepcopy(row))
    if not policy["rtl_facts"]:
        result.pop("selected_rtl_facts_file")
        result["judge_rtl_facts_file"] = dict(path=str(facts), sha256=_sha(facts))
    # This file contains custody information, never a grant of private metadata.
    receipt_path = Path(experiment).absolute() / "author_input_views.json"
    with receipt_path.open("x") as stream:
        json.dump(
            dict(
                schema="merlin.raw_rtl_author_inputs.v1",
                profile=profile.name,
                disclosure=policy,
                views=receipts,
                withheld=withheld,
                raw_rtl_files=raw_pins,
            ),
            stream,
            sort_keys=True,
            indent=2,
        )
        stream.write("\n")
    receipt_path.chmod(0o400)
    result.setdefault("host_inputs", []).append(
        dict(path=str(receipt_path), note="private author-view custody receipt")
    )
    result["author_input_views"] = [
        {key: receipt[key] for key in ("source", "destination", "sha256")} for receipt in receipts
    ]
    # The software call boundary must agree before any source release seals.
    # This reads only the preparer's own newly generated protocol/private input.
    public_runtime_abi.verify({**result, "allowed": grants})
    destination.chmod(0o555)
    return result, grants


def verify(bundle, *, profile):
    """Fail closed if a raw profile still declares a rich public facts grant."""
    policy = profile.disclosure
    # Native public-contract preflight requires this new binding for all raw
    # profiles. Preserve structural readers of older archived metadata here.
    if "public_runtime_abi" in bundle:
        from .public_runtime_abi import binding

        binding(bundle)
    if not profile.raw_rtl_study:
        raise ValueError("not a raw-RTL disclosure profile")
    pins = bundle.get("selected_raw_rtl_files")
    if not isinstance(pins, list) or not pins:
        raise ValueError("raw-RTL author profile lacks selected raw hardware")
    allowed = bundle.get("allowed", ())
    paths = {row.get("path") for row in allowed}
    for pin in pins:
        if (
            not isinstance(pin, dict)
            or set(pin) != {"path", "sha256"}
            or pin["path"] not in paths
            or not Path(pin["path"]).is_absolute()
            or not is_sha256(str(pin["sha256"]))
            or not any(row.get("path") == pin["path"] and row.get("mode") == "ro" for row in allowed)
        ):
            raise ValueError("raw hardware is not an exact selected public file grant")
    if policy["rtl_facts"]:
        return
    if "selected_rtl_facts_file" in bundle:
        raise ValueError("raw-only author cannot select public extracted RTL facts")
    pin = bundle.get("judge_rtl_facts_file")
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        raise ValueError("raw-only author requires private pinned judge facts")
    hidden = {Path(pin["path"]).parent}
    views = bundle.get("author_input_views")
    if not isinstance(views, list) or not views:
        raise ValueError("restricted author requires explicit public numerical input views")
    for view in views:
        if (
            not isinstance(view, dict)
            or set(view) != {"source", "destination", "sha256"}
            or not is_sha256(str(view["sha256"]))
            or view["destination"] not in paths
        ):
            raise ValueError("public numerical input view differs from the declared grant")
        hidden.add(Path(view["source"]))
    if policy["input_format"] == "neutral_math_v1":
        from . import neutral_public_contracts as N

        pins = bundle.get("neutral_public_contract_files")
        required = N.FILES | {N.INPUT_DOCUMENT} | {"schemas/" + name for name in N.SCHEMAS}
        if not isinstance(pins, list) or len(pins) != len(required) or {row.get("alias") for row in pins} != required:
            raise ValueError("EL1 lacks its neutral external protocol")
        if any(str(row["path"]).startswith("src/merlin/resources/contracts/") for row in allowed):
            raise ValueError("EL1 exposes the original interface input contracts")
    software = ((bundle.get("compiler_tool_profile") or {}).get("semantic_inputs") or {}).get("software_spec")
    if not policy["software_spec"] and isinstance(software, dict):
        hidden.add(Path(software["path"]))
    for row in allowed:
        public = Path(row["path"]).resolve()
        if any(public.is_relative_to(secret.resolve()) or secret.resolve().is_relative_to(public) for secret in hidden):
            raise ValueError("raw-only author grant exposes withheld derived inputs")
        if not policy["derived_target_contract"] and row.get("note") == "selected capability contract (ALL arms)":
            raise ValueError("raw-only author exposes a derived target contract")
        if not policy["software_spec"] and row.get("note") == "admitted software semantics":
            raise ValueError("raw-only author exposes the software specification")
