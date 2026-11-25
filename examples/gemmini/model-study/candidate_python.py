"""Execute the selected frozen compiler with explicit import roots and origins.

The caller invokes Python with -S, so editable-install .pth files cannot add
another checkout. Third-party packages have explicit raw site-package roots.
This is import provenance checking, not a filesystem sandbox.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import runpy
import sys
import uuid


def main():
    selection = json.loads(Path(sys.argv[1]).read_bytes())
    arguments = sys.argv[2:]
    entry = Path(selection["entrypoint"]).resolve(strict=True)
    if not arguments or Path(arguments[0]).resolve() != entry:
        raise ValueError("candidate invocation does not name the selected entrypoint")
    allowed = [Path(p).resolve(strict=True) for p in selection["compiler_roots"]]

    def admitted(path):
        return any(Path(path).resolve().is_relative_to(root) for root in allowed)

    class Origins(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] not in {"merlin", "backend"}:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                raise ImportError(f"selected frozen compiler does not provide {fullname}")
            locations = list(spec.submodule_search_locations or ())
            if spec.origin and spec.origin not in {"namespace", "built-in", "frozen"}:
                locations.append(spec.origin)
            if not locations or any(not admitted(value) for value in locations):
                raise ImportError(f"compiler import outside frozen roots: {fullname}: {locations}")
            return spec

    sys.meta_path.insert(0, Origins())

    def record():
        rows = []
        for name, module in sorted(sys.modules.items()):
            if name.split(".", 1)[0] in {"merlin", "backend"} and getattr(module, "__file__", None):
                path = Path(module.__file__).resolve()
                rows.append({"module": name, "path": str(path), "allowed": admitted(path),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        output = Path(selection["origins"]) / (uuid.uuid4().hex + ".json")
        output.write_text(json.dumps({"schema": "merlin.model_study.compiler_imports.v1",
                                     "entrypoint": str(entry), "modules": rows}, indent=2) + "\n")

    atexit.register(record)
    sys.argv = arguments
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
