"""Pure primary-release metadata interpretation, no download or execution."""
from urllib.parse import urlsplit


def select(release, tag):
    if release.get("tag_name") != "v4.35.0" or release.get("draft") is not False or release.get("prerelease") is not False:
        raise ValueError("exact stable release required")
    obj = tag.get("object", {})
    commit = obj.get("sha", "")
    if tag.get("ref") != "refs/tags/v4.35.0" or obj.get("type") != "commit" or len(commit) != 40 or set(commit) - set("0123456789abcdef"):
        raise ValueError("exact full release commit required")
    wanted = "lean-4.35.0-linux.zip"
    candidates = [a for a in release["assets"] if a.get("name") == wanted]
    if len(candidates) != 1:
        raise ValueError("one exact Linux archive asset required")
    asset = candidates[0]
    checksum = asset.get("digest", "")
    url = asset.get("browser_download_url", "")
    expected_url = "https://github.com/leanprover/lean4/releases/download/v4.35.0/" + wanted
    if (url != expected_url or urlsplit(url).scheme != "https" or asset.get("state") != "uploaded"
            or asset.get("content_type") != "application/zip"):
        raise ValueError("official uploaded versioned zip URL required")
    if type(checksum) is not str or not checksum.startswith("sha256:") or len(checksum) != 71 or set(checksum[7:]) - set("0123456789abcdef"):
        raise ValueError("publisher-platform SHA256 metadata required")
    size = asset.get("size")
    if type(size) is not int or not 0 < size <= 1024**3:
        raise ValueError("finite publisher compressed-size bound")
    return {"version": "4.35.0", "tag": "v4.35.0", "commit": commit,
            "platform": "Linux x86_64 (release matrix/source interpretation)",
            "release_id": release["id"], "asset_id": asset["id"],
            "archive_name": wanted, "archive_url": url, "compressed_bytes": size,
            "publisher_platform_sha256": checksum[7:],
            "checksum_provenance": "GitHub official release API digest over HTTPS; not independent signature verification",
            "old_sdk_identity_reused": False, "download_authorized": False,
            "install_authorized": False, "native_authorized": False, "old_holds_cleared": False}
