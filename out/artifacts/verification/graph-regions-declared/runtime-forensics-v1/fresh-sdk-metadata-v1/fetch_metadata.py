"""Bounded official JSON metadata lookup only; never fetch a release asset."""
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request


URLS = {
    "release": "https://api.github.com/repos/leanprover/lean4/releases/tags/v4.35.0",
    "tag": "https://api.github.com/repos/leanprover/lean4/git/ref/tags/v4.35.0",
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("metadata redirect refused")


def main():
    root = Path(__file__).parent
    output = root / "official-metadata-v1"
    output.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(NoRedirect())
    records = []
    for role, url in URLS.items():
        row = {"role": role, "url": url, "cap_bytes": 1024**2, "timeout_seconds": 20}
        request = urllib.request.Request(url, headers={"User-Agent": "Merlin-source-metadata-review",
                                                       "Accept": "application/vnd.github+json"})
        try:
            with opener.open(request, timeout=20) as response:
                row.update(status=response.status, content_type=response.headers.get("Content-Type"),
                           date=response.headers.get("Date"), etag=response.headers.get("ETag"))
                raw = response.read(1024**2 + 1)
            if len(raw) > 1024**2 or "json" not in (row["content_type"] or ""):
                raise ValueError("metadata type/size bound")
            json.loads(raw)
            (output / (role + ".json")).write_bytes(raw)
            row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error))
        records.append(row)
    (output / "RESULT.json").write_text(json.dumps({"records": records, "archive_downloads": 0,
         "native_launches": 0, "sdk_image_reads": 0}, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
