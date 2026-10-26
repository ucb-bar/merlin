"""Own child: load exact authorized source bytes, never candidate pyc."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--authorization-sha256", required=True)
    args = p.parse_args()
    with args.authorization.open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    if len(raw) > 1024**2 or hashlib.sha256(raw).hexdigest() != args.authorization_sha256:
        raise ValueError("exact child authorization binding")
    review = json.loads(raw)
    if review["status"] != "AUTHORIZED_ONE_FRESH_OFFICIAL_SDK_DATA_INTAKE_F_ONLY":
        raise ValueError("child DATA permission required")
    order = ("policy", "fdio", "http_fetch", "zip_intake", "outer_deadline", "intake", "run_intake")
    refs = {Path(x["path"]).stem: x for x in review["source_refs"]}
    if set(refs) != set(order) or len(review["source_refs"]) != len(order):
        raise ValueError("exact child source closure")
    for name in order:
        ref = refs[name]
        path = Path(ref["path"])
        if path.parent != Path(__file__).absolute().parent:
            raise ValueError("same bound child source directory")
        raw = path.read_bytes()
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("child source binding")
        module = types.ModuleType(name)
        module.__file__, module.__package__ = str(path), ""
        sys.modules[name] = module
        exec(compile(raw, str(path), "exec"), module.__dict__)
    sys.argv = [refs["run_intake"]["path"], "--authorization", str(args.authorization),
                "--authorization-sha256", args.authorization_sha256]
    return sys.modules["run_intake"].main()


if __name__ == "__main__":
    raise SystemExit(main())
