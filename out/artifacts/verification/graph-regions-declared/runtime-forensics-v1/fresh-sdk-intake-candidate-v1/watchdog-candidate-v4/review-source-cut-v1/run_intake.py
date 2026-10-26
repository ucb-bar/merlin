"""One future authorized intake; NOT permission to fetch/install/execute now."""
import argparse
import hashlib
import os
from pathlib import Path

import fdio
import intake
from policy import identity, require, request_from_plan, strict_json

MODULES = {"run_intake.py", "intake.py", "fdio.py", "policy.py", "http_fetch.py", "zip_intake.py", "outer_deadline.py"}


def read_source_metadata(ref):
    p = Path(ref["path"])
    require(p.suffix in {".py", ".json"} and ".elan" not in p.parts, "source/metadata only")
    require(type(ref["bytes"]) is int and 0 <= ref["bytes"] <= 64 * 1024**2, "source/metadata cap")
    with p.open("rb") as stream:
        raw = stream.read(ref["bytes"] + 1)
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "source/metadata binding")
    return raw


def failure_record(error):
    return {"status": "FIRST_FAILURE_NO_RETRY", **fdio.error_details(error),
            "evidence": getattr(error, "evidence", None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--authorization-sha256", required=True)
    args = p.parse_args()
    with args.authorization.open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    require(len(raw) <= 1024**2 and hashlib.sha256(raw).hexdigest() == args.authorization_sha256, "authorization binding")
    auth = strict_json(raw)
    require(set(auth) == {"schema", "status", "action", "attempts", "attempt_journal", "source_refs",
                          "plan", "request", "native_authorized", "install_authorized", "old_holds_cleared"}, "closed review fields")
    require(auth["schema"] == "merlin.fresh-sdk-data-intake-review.v1"
            and auth["action"] == "ONE_FRESH_OFFICIAL_SDK_DATA_INTAKE", "exact DATA action")
    require(auth["native_authorized"] is False and auth["install_authorized"] is False
            and auth["old_holds_cleared"] is False, "review cannot escalate authority")
    require(auth["status"] == "AUTHORIZED_ONE_FRESH_OFFICIAL_SDK_DATA_INTAKE_F_ONLY", "proposal is not authorization")
    require(type(auth["attempts"]) is int and auth["attempts"] == 1, "one attempt")
    journal = Path(auth["attempt_journal"])
    parent = fdio.directory(journal.parent)
    try:
        fdio.save(parent, journal.name, {"authorization_sha256": args.authorization_sha256,
                  "attempt_consumed": True, "retry": False, "execute": False})
    finally:
        os.close(parent)
    try:
        refs = auth["source_refs"]
        require(len(refs) == len(MODULES) and {Path(r["path"]).name for r in refs} == MODULES, "full source closure")
        require(all(Path(r["path"]).parent == Path(__file__).absolute().parent for r in refs), "same bound source directory")
        for ref in refs:
            read_source_metadata(ref)
        plan = strict_json(read_source_metadata(auth["plan"]))
        request = strict_json(read_source_metadata(auth["request"]))
        expected = request_from_plan(plan, auth["plan"])
        require(identity(request) == identity(expected), "exact original plan/request association")
        # Only this separately authorized path constructs the real network backend.
        from http_fetch import StdlibHTTPS
        intake.run(request, StdlibHTTPS())
        print('{"status":"NEW_ARCHIVE_DATA_ONLY","native_authorized":false}')
        return 0
    except BaseException as error:
        import json
        print(json.dumps(failure_record(error)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
