"""``dashboard --live``: regenerate the page every N seconds and serve it on 127.0.0.1, stdlib only.

The page gets a ``<meta http-equiv="refresh">`` so a browser behind a VS Code or ssh port-forward
follows along without any script.  The server answers GET for the one page and nothing else: it lists
no directory, serves no other file and accepts no upload.  The only file it writes is the page itself,
and it refuses an output path inside any directory it reads (a run directory, a phase-0 derivation),
so a live view can never write into a run.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_INTERVAL = 60.0
DEFAULT_PORT = 8765
HOST = "127.0.0.1"


def with_refresh(page: str, seconds: float) -> str:
    """The page with a meta refresh after its charset declaration (idempotent)."""
    tag = f'<meta http-equiv="refresh" content="{max(1, round(seconds))}">'
    if 'http-equiv="refresh"' in page:
        return page
    anchor = '<meta charset="utf-8">'
    return page.replace(anchor, anchor + tag, 1) if anchor in page else tag + page


def check_output(out: Path, inputs: Iterable[Path | None]) -> None:
    """Refuse an output path inside (or equal to) any directory the view reads."""
    from ..spec import SpecError

    target = Path(out).expanduser().resolve()
    for source in inputs:
        if source is None:
            continue
        root = Path(source).expanduser().resolve()
        if target == root or root in target.parents:
            raise SpecError(f"refusing to write the dashboard inside a directory it reads: {target} is under {root}")


def write_page(destination: Path, page: str) -> None:
    """Write atomically (a reader never sees half a page)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp")
    staging.write_text(page, encoding="utf-8")
    staging.replace(destination)


def _handler(page_path: Path) -> type[BaseHTTPRequestHandler]:
    class OnePage(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 -- the stdlib's name
            request = self.path.partition("?")[0]
            if request in ("/", "/index.html", "/" + page_path.name):
                target = page_path
            else:
                # A page the main page links to: a plain ``<name>.html`` beside it, nothing else.
                name = request[1:]
                if not name.endswith(".html") or "/" in name or "\\" in name or name.startswith("."):
                    self.send_error(404, "this server serves the dashboard pages only")
                    return
                target = page_path.parent / name
            try:
                body = target.read_bytes()
            except OSError:
                self.send_error(503, "the page has not been written yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 -- stdlib signature
            return

    return OnePage


def serve(
    render: Callable[[], str | Mapping[str, str]],
    destination: Path,
    *,
    interval: float = DEFAULT_INTERVAL,
    port: int = DEFAULT_PORT,
    iterations: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    announce: Callable[[str], None] = print,
    ready: Callable[[ThreadingHTTPServer], None] | None = None,
) -> int:
    """Write ``render()`` to ``destination`` every ``interval`` s and serve it on 127.0.0.1:``port``.

    ``render`` returns the page, or ``{"": the page, <name>: a linked page}`` written beside it.

    ``iterations`` bounds the loop (tests); otherwise it runs until Ctrl-C.  A render that raises is
    reported and the previous page stays up."""
    interval = max(1.0, float(interval))
    destination = Path(destination)
    server = ThreadingHTTPServer((HOST, port), _handler(destination))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    announce(f"serving {destination} on http://{HOST}:{server.server_address[1]}/ (refresh every {interval:g} s)")
    if ready is not None:
        ready(server)
    done = 0
    try:
        while iterations is None or done < iterations:
            try:
                rendered = render()
                for name, page in (rendered if isinstance(rendered, Mapping) else {"": rendered}).items():
                    write_page(destination.parent / name if name else destination, with_refresh(page, interval))
            except Exception as exc:  # noqa: BLE001 -- a bad refresh keeps the last page up
                announce(f"refresh failed ({type(exc).__name__}: {exc}); keeping the previous page")
            done += 1
            if iterations is not None and done >= iterations:
                break
            sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


__all__ = ["DEFAULT_INTERVAL", "DEFAULT_PORT", "HOST", "check_output", "serve", "with_refresh", "write_page"]
