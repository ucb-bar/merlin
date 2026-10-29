"""Bounded stdlib HTTPS fetch. Real backend is never constructed by tests."""
import hashlib
import os
import sys
import time
from urllib.parse import urljoin

from fdio import write_all
from policy import https_url, require


class StdlibHTTPS:
    def __init__(self):
        import ssl
        import urllib.request
        import urllib.error

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self.request = urllib.request
        self.error = urllib.error
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def open(self, url, timeout):
        request = self.request.Request(url, headers={"User-Agent": "Merlin-reviewed-data-intake",
                                                     "Accept-Encoding": "identity"})
        try:
            return self.opener.open(request, timeout=timeout)
        except self.error.HTTPError as response:
            return response


def fetch(policy, output_fd, state, backend, clock=time.monotonic):
    start = clock()
    url, redirects = policy["url"], 0
    response = None
    digest = hashlib.sha256()
    count = 0

    def remaining():
        left = policy["total_deadline_seconds"] - (clock() - start)
        require(left > 0, "fetch deadline exceeded")
        return left
    try:
        while True:
            https_url(url, policy["https_hosts"])
            state.update(phase="HTTP", url=url, redirects=redirects)
            response = backend.open(url, min(policy["connect_timeout_seconds"], remaining()))
            remaining()
            status = response.status
            if status in (301, 302, 303, 307, 308):
                require(redirects < policy["redirects_max"], "redirect bound exceeded")
                location = response.headers.get("Location")
                require(type(location) is str and bool(location), "redirect has no location")
                url = https_url(urljoin(url, location), policy["https_hosts"])
                redirects += 1
                response.close()
                response = None
                continue
            require(status == 200, "HTTP status is not 200")
            require(response.headers.get("Content-Encoding", "identity") == "identity", "encoded response forbidden")
            lengths = response.headers.get_all("Content-Length", []) if hasattr(response.headers, "get_all") else (
                [response.headers["Content-Length"]] if "Content-Length" in response.headers else [])
            require(len(lengths) <= 1, "ambiguous content length")
            if lengths:
                require(lengths[0].isascii() and lengths[0].isdigit()
                        and int(lengths[0]) == policy["exact_bytes"], "published content length differs")
            require(not (lengths and response.headers.get("Transfer-Encoding")), "ambiguous transfer length")
            break
        while True:
            remaining()
            limit = min(policy["chunk_bytes"], policy["exact_bytes"] + 1 - count)
            require(limit > 0, "finite fetch bound exhausted")
            block = response.read(limit)
            require(type(block) is bytes and len(block) <= limit, "HTTP read contract")
            state["pending"] = block
            state.update(phase="download", pending_offset=count, received_bytes=count + len(block))
            remaining()
            if not block:
                break
            digest.update(block)
            count += len(block)
            write_all(output_fd, block)
            state["pending"] = b""
            require(count <= policy["exact_bytes"], "download exceeds published size")
        state["observed_sha256"] = digest.hexdigest()
        require(count == policy["exact_bytes"], "short download")
        require(digest.hexdigest() == policy["expected_sha256"], "publisher archive digest mismatch")
        os.fsync(output_fd)
        return {"original_publisher_sha256": policy["expected_sha256"], "archive_bytes": count,
                "redirects": redirects, "received_hash_matched": True}
    finally:
        if response is not None:
            primary = sys.exc_info()[0]
            try:
                response.close()
            except BaseException as error:
                state["response_close_error"] = str(error)
                if primary is None:
                    raise
