"""One future reviewed complete-intake child; unapproved now, no import IO."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

SOURCES = {"policy", "fdio", "http_fetch", "zip_intake", "outer_deadline", "intake", "run_intake",
           "watchdog", "watchdog_runner", "child_bootstrap"}


def bound(ref):
    path = Path(ref["path"])
    if path.suffix not in (".json", ".py") or type(ref["bytes"]) is not int or not 0 <= ref["bytes"] <= 64 * 1024**2:
        raise ValueError("bounded source/metadata ref")
    with path.open("rb") as stream:
        raw = stream.read(ref["bytes"] + 1)
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("source/metadata binding")
    return raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--authorization-sha256", required=True)
    args = p.parse_args()
    with args.authorization.open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    if len(raw) > 1024**2 or hashlib.sha256(raw).hexdigest() != args.authorization_sha256:
        raise ValueError("parent authorization binding")
    review = json.loads(raw)
    fields = {"schema", "status", "action", "attempts", "source_refs", "child_authorization", "workspace",
              "seconds", "terminate_seconds", "reap_seconds", "native_authorized", "install_authorized", "old_holds_cleared"}
    if set(review) != fields or review["schema"] != "merlin.own-child-intake-watchdog-review.v1":
        raise ValueError("closed watchdog review")
    if review["status"] != "AUTHORIZED_ONE_OWN_CHILD_SDK_DATA_INTAKE_F_ONLY" or review["action"] != "ONE_COMPLETE_DATA_INTAKE_CHILD":
        raise ValueError("proposal is not complete-intake permission")
    if type(review["attempts"]) is not int or review["attempts"] != 1 or any(review[k] is not False for k in ("native_authorized", "install_authorized", "old_holds_cleared")):
        raise ValueError("one DATA attempt; no expanded authority")
    if any(type(review[k]) is not int for k in ("seconds", "terminate_seconds", "reap_seconds")) or (review["seconds"], review["terminate_seconds"], review["reap_seconds"]) != (1800, 2, 3):
        raise ValueError("exact parent budget and five-second total grace")
    refs = {Path(r["path"]).stem: r for r in review["source_refs"]}
    if len(review["source_refs"]) != len(SOURCES) or set(refs) != SOURCES:
        raise ValueError("complete parent/child source closure")
    source = {}
    for name, ref in refs.items():
        if Path(ref["path"]).parent != Path(__file__).absolute().parent:
            raise ValueError("one bound source directory")
        source[name] = bound(ref)
    # Only source bytes, never a candidate pyc or installed SDK, supply modules.
    for name in ("policy", "fdio", "watchdog"):
        module = types.ModuleType(name)
        module.__file__, module.__package__ = refs[name]["path"], ""
        sys.modules[name] = module
        exec(compile(source[name], module.__file__, "exec"), module.__dict__)
    child = sys.modules["policy"].strict_json(bound(review["child_authorization"]))
    expected = {k: refs[k] for k in ("policy", "fdio", "http_fetch", "zip_intake", "outer_deadline", "intake", "run_intake")}
    if {Path(r["path"]).stem: r for r in child["source_refs"]} != expected:
        raise ValueError("child source pins differ from parent's bound closure")
    if child["status"] != "AUTHORIZED_ONE_FRESH_OFFICIAL_SDK_DATA_INTAKE_F_ONLY":
        raise ValueError("child authorization is still proposed")
    command = [sys.executable, "-I", "-S", "-B", refs["child_bootstrap"]["path"],
               "--authorization", review["child_authorization"]["path"],
               "--authorization-sha256", review["child_authorization"]["sha256"]]
    outcome = sys.modules["watchdog"].supervise(command, review["workspace"], seconds=1800,
                                               terminate_seconds=2, reap_seconds=3)
    print(json.dumps(outcome), flush=True)
    return 0 if outcome["status"] == "CHILD_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
