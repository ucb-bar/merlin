"""Metadata-only exact original Init-module/runtime selection; no file reads.

This NEW reviewed rule includes every five-file family for all653 Init modules
and the original11 runtime files. It does not skip a failing file individually,
refresh a pin, diagnose/promote v9, or modify the old whole-SDK inspector.
"""

from __future__ import annotations

from pathlib import PurePosixPath


class SelectionRefusal(ValueError):
    pass


def index_references(values, strict):
    result = {}
    for value in values:
        reference = strict(value)
        path = reference["path"]
        if path in result and result[path] != reference:
            raise SelectionRefusal("conflicting original expected pins: " + path)
        result[path] = reference
    return result


def select(document, provenance, sdk, original_w, *, strict, references):
    if document.get("schema") != "merlin.frontend.original-module-runtime-selection.v1":
        raise SelectionRefusal("unsupported reviewed dependency selection")
    if (
        document["original_F_manifest"] != provenance["original_F_SDK_manifest"]
        or document["original_W_context"] != provenance["original_W_fixed_context"]
    ):
        raise SelectionRefusal("selection parent identities differ from original provenance")
    modules = sdk["modules"]
    if (
        sdk["module_count"] != 653
        or len(modules) != 653
        or len(set(modules)) != 653
        or document["original_F_modules"] != modules
    ):
        raise SelectionRefusal("every original Init module is required in original order")
    if any(module != "Init" and not module.startswith("Init.") for module in modules):
        raise SelectionRefusal("non-Init/held standard module")
    for module in modules:
        if any(not part or not part.isascii() or not part.replace("_", "a").isalnum() for part in module.split(".")):
            raise SelectionRefusal("invalid original module name")
    f = index_references(sdk["files"].values(), strict)
    w = index_references((reference for _, reference in references(original_w)), strict)
    runtime = index_references(provenance["original_runtime_refs"], strict)
    declared_runtime = index_references(document["runtime_refs"], strict)
    if len(runtime) != 11 or declared_runtime != runtime or len(document["runtime_refs"]) != 11:
        raise SelectionRefusal("exact original11 runtime membership required")
    binary = next((path for path in runtime if path.endswith("/bin/lean")), None)
    if binary is None:
        raise SelectionRefusal("original Lean executable missing")
    toolchain = PurePosixPath(binary).parent.parent
    expected = {}
    roles = {".olean": 0, ".olean.private": 0, ".olean.server": 0, ".ilean": 0, ".lean": 0}
    independently_bound = 0
    for module in modules:
        relative = module.replace(".", "/")
        for suffix in roles:
            directory = "src/lean" if suffix == ".lean" else "lib/lean"
            path = str(toolchain / directory / (relative + suffix))
            if path not in f:
                raise SelectionRefusal("missing original module family member: " + path)
            if path in w and w[path] != f[path]:
                raise SelectionRefusal("overlapping F/W original expected-pin conflict: " + path)
            if suffix in {".olean", ".olean.private", ".olean.server"}:
                if w.get(path) != f[path]:
                    raise SelectionRefusal("compiled Init member missing independent original W binding: " + path)
                independently_bound += 1
            expected[path] = f[path]
            roles[suffix] += 1
    for path, reference in runtime.items():
        if path in f and f[path] != reference or path in w and w[path] != reference:
            raise SelectionRefusal("original runtime overlap conflict: " + path)
        if f.get(path) != reference and w.get(path) != reference:
            raise SelectionRefusal("runtime has no original F/W provenance: " + path)
        if path in expected:
            raise SelectionRefusal("runtime path collides with module file role")
        expected[path] = reference
    supplied = document["selected"]
    if type(supplied) is not dict:
        raise SelectionRefusal("explicit selected path/ref dictionary required")
    for path, reference in supplied.items():
        if strict(reference)["path"] != path:
            raise SelectionRefusal("selected dictionary key/reference path differs")
    if supplied != expected or document["selected_count"] != 3276 or len(supplied) != 3276:
        raise SelectionRefusal("missing/extra/wrong original selected member")
    if independently_bound != 1959 or document["compiled_Init_refs_independently_bound_by_original_W"] != 1959:
        raise SelectionRefusal("incomplete independent compiled-module binding")
    all_original = dict(f)
    all_original.update(runtime)
    excluded = {path: reference for path, reference in all_original.items() if path not in expected}
    if len(excluded) != 18 or document["excluded_auxiliary_refs"] != excluded:
        raise SelectionRefusal("auxiliary complement differs from the module/runtime rule")
    if set(expected) & set(document["held_paths_not_read_or_selected"]):
        raise SelectionRefusal("held path admitted by selection")
    return dict(
        files=expected,
        module_count=653,
        family_counts=roles,
        runtime_count=11,
        independently_bound_compiled=1959,
        selected_count=3276,
        excluded_count=18,
        runtime_or_sdk_files_opened=0,
        new_expected_hashes=False,
        old_failed_profiles_repaired=False,
        qualification=False,
    )
