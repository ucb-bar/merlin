"""Pure policy for a NEW archive identity; no original SDK pins are reused."""
import hashlib
import json
from pathlib import PurePosixPath
from urllib.parse import urlsplit

AUTHORITY = {"execute": False, "install": False, "graph_verified": False, "old_holds_cleared": False}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out

    def constant(_):
        raise ValueError("nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def absolute(text):
    require(type(text) is str and "\x00" not in text and not text.startswith("//"), "path string")
    p = PurePosixPath(text)
    require(p.is_absolute() and str(p) == text and ".." not in p.parts and ".elan" not in p.parts,
            "canonical new absolute path; installed SDK forbidden")
    return p


def positive(value, maximum):
    require(type(value) is int and 0 < value <= maximum, "finite positive integer bound")


def https_url(url, hosts):
    p = urlsplit(url)
    require(p.scheme == "https" and p.hostname in hosts and p.port in (None, 443)
            and p.username is None and p.password is None and not p.fragment
            and not any(ord(c) < 33 or ord(c) == 127 for c in url), "unapproved HTTPS destination")
    return url


def request_from_plan(plan, plan_ref):
    return {"schema": "merlin.fresh-sdk-data-intake.v1", "plan": plan_ref,
            "environment_identity": plan["identity"], "new_root": plan["new_root"],
            "fetch": plan["archive_fetch"], "extract": plan["extraction_proposal"],
            "attempts": 1, "authority": dict(AUTHORITY),
            "outer_deadline": "POSIX_ITIMER_REAL_MAIN_THREAD_NO_EXISTING_TIMER_V1"}


def validate(request):
    require(set(request) == {"schema", "plan", "environment_identity", "new_root", "fetch", "extract", "attempts", "authority", "outer_deadline"}, "request fields")
    require(request["outer_deadline"] == "POSIX_ITIMER_REAL_MAIN_THREAD_NO_EXISTING_TIMER_V1", "outer timer policy")
    require(request["schema"] == "merlin.fresh-sdk-data-intake.v1", "request schema")
    require(type(request["attempts"]) is int and request["attempts"] == 1, "one attempt")
    require(request["authority"] == AUTHORITY and all(v is False for v in request["authority"].values()), "no execution authority")
    absolute(request["new_root"])
    f, z = request["fetch"], request["extract"]
    require(f["https_hosts"] == ["github.com", "release-assets.githubusercontent.com"], "closed HTTPS hosts")
    https_url(f["url"], f["https_hosts"])
    require(type(f["expected_sha256"]) is str and len(f["expected_sha256"]) == 64
            and not set(f["expected_sha256"]) - set("0123456789abcdef"), "original publisher digest")
    positive(f["exact_bytes"], 1024**3)
    positive(f["chunk_bytes"], 1024**2)
    require(type(f["max_read_bytes_including_sentinel"]) is int
            and f["max_read_bytes_including_sentinel"] == f["exact_bytes"] + 1, "finite oversize sentinel")
    require(type(f["attempts"]) is int and f["attempts"] == 1 and f["resume"] is False, "no retry/resume")
    require(type(f["redirects_max"]) is int and 0 <= f["redirects_max"] <= 3, "redirect cap")
    positive(f["connect_timeout_seconds"], 30)
    positive(f["total_deadline_seconds"], 1800)
    require(f["no_extract_before_original_publisher_digest_matches"] is True, "authentication precedes ZIP")
    require(f["output"] == "download/lean-4.35.0-linux.zip.part", "exclusive fixed archive name")
    require(z["format"] == "zip" and z["expected_top_directory"] == "lean-4.35.0-linux", "exact ZIP family")
    for key, maximum in (("member_count_max", 50000), ("per_member_bytes_max", 1024**3),
                         ("total_uncompressed_bytes_max", 16 * 1024**3), ("disk_reserve_bytes", 20 * 1024**3)):
        positive(z[key], maximum)
    require(z["regular_data_only_first"] is True and z["execute_archive_scripts"] is False
            and z["write_old_sdk_or_elan"] is False, "data-only extraction")
    env = request["environment_identity"]
    require(env["archive_url"] == f["url"] and env["compressed_bytes"] == f["exact_bytes"]
            and env["publisher_platform_sha256"] == f["expected_sha256"], "identity/fetch association")
