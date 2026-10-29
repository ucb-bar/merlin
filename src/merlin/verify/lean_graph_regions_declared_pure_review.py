"""Bind the one coordinator-reviewed optional symbol; metadata only.

No process/library is executed and no SDK/runtime file is opened. This is a
finite observational engineering review, not graph/policy/native approval.
"""

from __future__ import annotations

from pathlib import Path


def reviewed_pair(root, document_path, provenance, guard, api):
    review = api["parse_json"](document_path.read_bytes())
    if review.get("schema") != "merlin.frontend.finite-optional-symbol-review.v1":
        raise ValueError("unsupported optional-symbol review")
    owners = [ref["path"] for ref in provenance["original_runtime_refs"] if ref["path"].endswith("/libleanshared.so")]
    if len(owners) != 1:
        raise ValueError("ambiguous original reviewed DSO")
    expected = [owners[0], "l_Lean_Parser_Tactic_simpLemma___boxed"]
    if review["reviewed_pair"] != expected or review["unknown_errors_or_additional_symbols_allowed"] is not False:
        raise ValueError("review is outside the one exact authorized pair")
    input_ref = dict(review["input_ref"])
    relative = Path(input_ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("review input must be a repository-relative documentary reference")
    input_ref["path"] = str(root / relative)
    observation = api["parse_json"](api["read_original"](input_ref))
    if observation["exact_additional_pair"] != expected or observation["all_noninitial_additional_pairs"] != [expected]:
        raise ValueError("reviewed observation has an unlisted additional symbol")
    if (
        observation["passed"] != 5
        or observation["total"] != 5
        or not all(item["passed"] for item in observation["tests"])
    ):
        raise ValueError("reviewed finite positive/refusal evidence is incomplete")
    if observation["process"]["returncode"] != 0 or not observation["postcheck"]["all_bound_bytes_unchanged"]:
        raise ValueError("reviewed case lacks successful process/original binding evidence")
    if observation["postcheck"]["original_sdk_runtime_bindings_checked"] != 3276:
        raise ValueError("review is not under the exact original module/runtime rule")
    recorded = observation["source_guard"]
    # Compare actual guard source bytes as well as their recorded identities.
    # No semantic SHA-injectivity premise is introduced by this host binding.
    if api["read_original"](recorded) != api["read_original"](guard):
        raise ValueError("current guard differs from the exact reviewed implementation")
    return dict(
        pair=tuple(expected),
        input_ref=input_ref,
        review=review,
        guard_identity=recorded,
        source_guard_bytes_equal=True,
        only_when_process_returncode_zero=True,
        qualification=False,
    )
