"""Pure NEW context proposal schema. No selection IO, guard, loader or proof API.

Host documentary authenticity is a separate review premise. Valid JSON is not
evidence that archive extraction/host review occurred, and yields no authority.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath

from policy import AUTHORITY, absolute, identity, require, validate


def sha(value):
    require(type(value) is str and len(value) == 64 and not set(value) - set("0123456789abcdef"), "SHA256 string")


def validate_manifest(request, manifest, roster):
    validate(request)
    require(set(manifest) == {"schema", "parent_archive", "members", "expanded_bytes", "new_environment_pins",
                             "old_expectations_replaced", "authority", "request_identity"}, "manifest fields")
    require(manifest["schema"] == "merlin.fresh-sdk-authenticated-member-manifest.v1", "manifest schema")
    require(manifest["authority"] == AUTHORITY and all(v is False for v in manifest["authority"].values()), "no authority")
    require(manifest["new_environment_pins"] is True and manifest["old_expectations_replaced"] is False, "new identity only")
    require(manifest["request_identity"] == identity(request), "request association")
    expected_parent = {"url": request["fetch"]["url"], "sha256": request["fetch"]["expected_sha256"],
                       "bytes": request["fetch"]["exact_bytes"], "identity": request["environment_identity"]}
    require(manifest["parent_archive"] == expected_parent and roster["parent_sha256"] == expected_parent["sha256"], "parent archive association")
    require(type(manifest["members"]) is list and len(manifest["members"]) == len(roster["members"]), "full member roster")
    total, paths = 0, set()
    for member, original in zip(manifest["members"], roster["members"], strict=True):
        require({k: member.get(k) for k in original} == original, "ZIP entry association")
        p = PurePosixPath(member["path"])
        require(not p.is_absolute() and ".." not in p.parts and p.parts[0] == request["extract"]["expected_top_directory"], "member path")
        require(str(p) not in paths, "duplicate member")
        paths.add(str(p))
        size = member["measured_bytes"]
        require(type(size) is int and 0 <= size <= request["extract"]["per_member_bytes_max"]
                and size == original["declared_bytes"], "member byte count")
        if member["kind"] == "file":
            sha(member["sha256"])
            require(set(member) == set(original) | {"measured_bytes", "sha256", "output_stat"}, "file fields")
            total += size
        else:
            require(member["kind"] == "directory" and size == 0 and member["sha256"] is None
                    and set(member) == set(original) | {"measured_bytes", "sha256"}, "directory fields")
    require(type(manifest["expanded_bytes"]) is int and manifest["expanded_bytes"] == total
            and total <= request["extract"]["total_uncompressed_bytes_max"], "full expanded count")


@dataclass(frozen=True)
class ContextProposal:
    """Documentary proposal only, deliberately not a CheckedQuery or runnable Context."""
    archive_identity: str
    manifest_identity: str
    member_refs: tuple
    host_review_identity: str | None
    host_refs: tuple
    status: str


def prepare(request, manifest, roster, selected_paths, host_review=None):
    validate_manifest(request, manifest, roster)
    require(type(selected_paths) is tuple and selected_paths and len(set(selected_paths)) == len(selected_paths), "explicit unique selection")
    files = {x["path"]: x for x in manifest["members"] if x["kind"] == "file"}
    require(set(selected_paths) <= files.keys(), "selected member absent/nonregular")
    refs = tuple((str(PurePosixPath(request["new_root"]) / "data" / p), files[p]["sha256"], files[p]["measured_bytes"])
                 for p in selected_paths)
    host_refs, review_identity = (), None
    if host_review is not None:
        require(set(host_review) == {"schema", "review_ref", "refs", "interpreter", "closure_scope", "native_authorized"}, "host closure fields")
        require(host_review["schema"] == "merlin.separate-host-closure-proposal.v1"
                and host_review["closure_scope"] == "EXPLICIT_FINITE_REVIEW_REQUIRED"
                and host_review["native_authorized"] is False, "documentary host closure only")
        ref = host_review["review_ref"]
        require(set(ref) == {"path", "sha256", "bytes"}, "separate review reference")
        absolute(ref["path"])
        sha(ref["sha256"])
        require(type(ref["bytes"]) is int and ref["bytes"] > 0, "review size")
        require(type(host_review["refs"]) is list and 0 < len(host_review["refs"]) <= 64, "finite host closure")
        host = {}
        for row in host_review["refs"]:
            require(set(row) == {"path", "sha256", "bytes"}, "host reference fields")
            path = absolute(row["path"])
            sha(row["sha256"])
            require(str(path) not in host and not path.is_relative_to(absolute(request["new_root"])), "host duplicate/SDK overlap")
            require(type(row["bytes"]) is int and row["bytes"] > 0, "host size")
            host[str(path)] = (row["sha256"], row["bytes"])
        require(host_review["interpreter"] in host, "interpreter outside closure")
        host_refs = tuple((p, *v) for p, v in sorted(host.items()))
        review_identity = identity(host_review)
    return ContextProposal(identity(manifest["parent_archive"]), identity(manifest), refs,
                           review_identity, host_refs, "SOURCE_ONLY_NOT_RUNNABLE_NO_AUTHORITY")
