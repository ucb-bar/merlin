"""Pure documentary archive admission into the unchanged common Selection API.

No images are read. A caller must bind/review all six documentary inputs and the
four controller sources; passing this decoder does NOT authenticate extraction,
approve host closure or authorize native execution. New archive-derived pins
never replace original SDK expectations. All source policies remain trusted.
"""
from pathlib import PurePosixPath
import stat

if __package__:
    from . import lean_graph_original_image_context_data as data
else:
    import lean_graph_original_image_context_data as data

FAMILIES = (".lean", ".olean", ".olean.private", ".olean.server", ".ilean", ".ir", ".ir.sig")
NO_AUTHORITY = {"execute": False, "install": False, "graph_verified": False, "old_holds_cleared": False}


def require(value, reason):
    if not value:
        raise data.ContextRefusal(reason)


def fields(value, names):
    require(type(value) is dict and set(value) == set(names.split()), "closed documentary fields")


def member_path(value, top):
    require(type(value) is str and value and all(ord(c) >= 32 and ord(c) != 127 for c in value)
            and "\\" not in value and ":" not in value, "canonical member path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and str(p) == value and ".." not in p.parts and p.parts[0] == top,
            "member escapes exact top directory")
    return p


def count(value, cap):
    require(type(value) is int and 0 <= value <= cap, "bounded integer, never bool")
    return value


def select(documents, references):
    """Full manifest roster, all object families, explicit runtime/host closure."""
    require(set(documents) == {"request", "manifest", "roster", "selection", "host"}, "exact documents")
    require(set(references) == set(documents) | {"trace"}
            and all(type(r) is data.FileRef for r in references.values()), "typed six documentary references")
    request, manifest, roster, selection, host = (documents[k] for k in ("request", "manifest", "roster", "selection", "host"))
    fields(selection, "schema inputs toolchain members modules sdk_runtime_members forbidden_roots native_authorized")
    require(selection["schema"] == "merlin.fresh-archive-common-selection.v1" and selection["native_authorized"] is False,
            "selection has no native authority")
    require(selection["inputs"] == {k: references[k].record() for k in ("request", "manifest", "roster", "host", "trace")},
            "all input associations must equal caller-bound originals")
    require(request["schema"] == "merlin.fresh-sdk-data-intake.v1" and request["authority"] == NO_AUTHORITY
            and all(v is False for v in request["authority"].values()), "DATA-only request")
    fields(manifest, "schema parent_archive members expanded_bytes new_environment_pins old_expectations_replaced authority request_identity")
    fields(roster, "members parent_sha256")
    require(manifest["schema"] == "merlin.fresh-sdk-authenticated-member-manifest.v1"
            and manifest["new_environment_pins"] is True and manifest["old_expectations_replaced"] is False
            and manifest["authority"] == NO_AUTHORITY and all(v is False for v in manifest["authority"].values()),
            "new DATA manifest, not old expectation refresh")
    require(manifest["request_identity"] == data.identity(request), "manifest/request content identity")
    expected = {"url": request["fetch"]["url"], "sha256": request["fetch"]["expected_sha256"],
                "bytes": request["fetch"]["exact_bytes"], "identity": request["environment_identity"]}
    require(manifest["parent_archive"] == expected and roster["parent_sha256"] == expected["sha256"],
            "official parent archive content association")
    limits = request["extract"]
    top = limits["expected_top_directory"]
    require(type(top) is str and PurePosixPath(top).parts == (top,), "one top-directory component")
    root = data.absolute(request["new_root"]) / "data"
    sdk = root / top
    require(selection["toolchain"] == str(sdk), "exact DATA path mapping, no old SDK alias")
    require(type(roster["members"]) is list and type(manifest["members"]) is list
            and 0 < len(roster["members"]) == len(manifest["members"]) <= limits["member_count_max"], "full bounded roster")
    seen, files, dirs, total = set(), {}, set(), 0
    for original, row in zip(roster["members"], manifest["members"], strict=True):
        fields(original, "path kind declared_bytes crc32 source_mode compression")
        require(all(row.get(k) == v and type(row.get(k)) is type(v) for k, v in original.items()), "exact ordered ZIP entry association")
        path = member_path(original["path"], top)
        require(str(path) not in seen, "duplicate member")
        seen.add(str(path))
        kind = original["kind"]
        require(kind in ("file", "directory"), "regular file or directory only")
        count(original["crc32"], 2**32 - 1)
        count(original["source_mode"], 65535)
        require(type(original["compression"]) is int and original["compression"] in (0, 8), "stored/deflate only")
        require(stat.S_IFMT(original["source_mode"]) == (stat.S_IFREG if kind == "file" else stat.S_IFDIR), "no links or special types")
        size = count(row["measured_bytes"], limits["per_member_bytes_max"])
        require(type(original["declared_bytes"]) is int and size == original["declared_bytes"], "declared/measured size equality")
        if kind == "file":
            fields(row, "path kind declared_bytes crc32 source_mode compression measured_bytes sha256 output_stat")
            files[str(path)] = data.FileRef(str(root / path), row["sha256"], size)
            require(type(row["output_stat"]) is dict, "documentary output stat")
            total += size
        else:
            fields(row, "path kind declared_bytes crc32 source_mode compression measured_bytes sha256")
            require(size == 0 and original["crc32"] == 0 and row["sha256"] is None, "empty directory")
            dirs.add(str(path))
        dirs.update(str(p) for p in path.parents if str(p) != ".")
    require(not set(files) & dirs, "file/directory prefix collision")
    require(type(manifest["expanded_bytes"]) is int and manifest["expanded_bytes"] == total
            and total <= limits["total_uncompressed_bytes_max"], "exact total size")
    require(type(selection["members"]) is list and len(selection["members"]) == len(files)
            and set(selection["members"]) == set(files), "exact entire regular-member roster, no implicit subset")
    modules = selection["modules"]
    require(type(modules) is list and modules and len(set(modules)) == len(modules), "explicit unique module roster")
    actual_modules = set()
    prefix = PurePosixPath(top) / "lib/lean"
    for name in files:
        p = PurePosixPath(name)
        if p.is_relative_to(prefix) and name.endswith(".olean"):
            actual_modules.add(".".join(p.relative_to(prefix).with_suffix("").parts))
    require(set(modules) == actual_modules, "all actual compiled modules explicitly declared")
    for module in modules:
        relative = data.module_name(module).replace(".", "/")
        for suffix in FAMILIES:
            folder = "src/lean" if suffix == ".lean" else "lib/lean"
            require(str(PurePosixPath(top) / folder / (relative + suffix)) in files, "complete source/object/IR signature family")
    fields(host, "schema archive_manifest refs interpreter finite_allowed_misses success_only_misses native_authorized")
    require(host["schema"] == "merlin.fresh-archive-separate-host-closure.v1"
            and host["archive_manifest"] == references["manifest"].record() and host["native_authorized"] is False,
            "separately bound host proposal, not permission")
    require(type(host["refs"]) is list and 0 < len(host["refs"]) <= 64, "bounded explicit host closure")
    runtime = data.unique_refs(host["refs"])
    require(host["interpreter"] in runtime, "host loader absent")
    require(all(data.absolute(p).is_relative_to(PurePosixPath("/usr/lib")) and not data.absolute(p).is_relative_to(sdk)
                for p in runtime), "explicit host-system closure only")
    selected = selection["sdk_runtime_members"]
    require(type(selected) is list and selected and len(set(selected)) == len(selected)
            and set(selected) <= set(files), "explicit unique archive runtime membership")
    for name in selected:
        ref = files[name]
        require(ref.path not in runtime, "host/archive runtime collision")
        runtime[ref.path] = ref
    executable = str(sdk / "bin/lean")
    require(executable in runtime, "actual archive Lean executable absent")
    require(len({PurePosixPath(p).name for p in runtime}) == len(runtime), "ambiguous loader basename")
    misses = data.finite_pairs(host["finite_allowed_misses"], runtime)
    success = data.finite_pairs(host["success_only_misses"], runtime)
    require(set(success) <= set(misses), "success-only diagnostics outside finite allowed list")
    forbidden = selection["forbidden_roots"]
    require(type(forbidden) is list and forbidden and len(set(forbidden)) == len(forbidden), "explicit old held roots")
    for path in [*runtime, *(ref.path for ref in files.values())]:
        require(not any(data.absolute(path).is_relative_to(data.absolute(old)) for old in forbidden), "selected path overlaps held root")
    return data.Selection(str(sdk), tuple(files[k] for k in sorted(files)), tuple(runtime[k] for k in sorted(runtime)),
                          misses, success, executable, host["interpreter"], tuple(forbidden))
