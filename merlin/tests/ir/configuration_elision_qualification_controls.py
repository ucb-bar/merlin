"""Live authority and fresh-artifact attacks using one genuinely checked request."""

import copy
import dataclasses
import json
import pickle
from pathlib import Path

from configuration_elision_qualification_process import save

from merlin.verify import configuration_elision_gate as gate
from merlin.verify.configuration_elision_bridge import canonical
from merlin.verify.configuration_elision_emit import emit
from merlin.verify.configuration_elision_intake import prepare


def refused(action):
    try:
        action()
    except (gate.Unavailable, TypeError, ValueError):
        return True
    raise AssertionError("untrusted or stale completion unexpectedly authorized")


def protocol(candidate, output, lean):
    """No mocked positive proofs: each bad request must fail before native work."""
    request = json.loads(candidate.request_json)
    mutations = {
        "source": dataclasses.replace(candidate, source=candidate.source + "\n"),
        "final": dataclasses.replace(candidate, final=candidate.final + "\n"),
        "query": dataclasses.replace(candidate, query=candidate.query + "\n"),
        "missing_root": dataclasses.replace(candidate, roots=candidate.roots[:-1]),
        "root_order": dataclasses.replace(candidate, roots=tuple(reversed(candidate.roots))),
        "domain": dataclasses.replace(candidate, domain="different-domain"),
        "relation": dataclasses.replace(candidate, relation="different-relation"),
        "resolution": dataclasses.replace(candidate, resolution_json="{}"),
        "sources": dataclasses.replace(candidate, sources_json="[]"),
        "proofs": dataclasses.replace(candidate, proof_sources_json="[]"),
        "request_target": dataclasses.replace(candidate, request_json=canonical({**request, "target": "wrong"})),
        "projection": dataclasses.replace(candidate, projection_json="{}"),
    }
    for name in ("forged_value", "forged_erase_flag", "forged_header"):
        projection = json.loads(candidate.projection_json)
        entry = projection["before"]["entries"][0]
        if name == "forged_header":
            entry["header"] += "forged"
        else:
            atom = next(atom for atom in entry["code"] if atom["effect"] and atom["effect"]["pureWrite"])
            if name == "forged_value":
                atom["effect"]["writes"][0][1] += 1
            else:
                atom["eraseAllowed"] = False
        generated = emit(projection)
        mutations[name] = dataclasses.replace(
            candidate, projection_json=canonical(projection), query=generated["query"], roots=tuple(generated["roots"])
        )
    rows = []
    for name, forged in mutations.items():
        destination = output / "forged" / name
        result = gate.check(forged, lean=lean, destination=destination)
        if result.status != "unavailable" or result.completion is not None or destination.exists():
            raise AssertionError("forged configuration candidate was not rejected before native launch: " + name)
        rows.append(dict(case=name, status=result.status, no_native_output=True))
    save(
        output / "PROTOCOL.json", dict(status="FORGED_REQUESTS_REFUSED_BEFORE_NATIVE", cases=rows, human_approved=False)
    )
    return len(rows)


def artifacts(candidate, output, objects, lean):
    output.mkdir(parents=True, exist_ok=False)
    request = json.loads(candidate.request_json)
    # These are new F-owned scratch originals, never sealed corpus or Q sources.
    for key in ("facts", "software"):
        target = output / (key + Path(request[key]).suffix)
        target.write_bytes(Path(request[key]).read_bytes())
        request[key] = str(target.resolve())
    actual = prepare(candidate.source, candidate.final, **request)
    result = gate.check(actual, lean=lean, destination=objects)
    if result.status != "checked":
        raise RuntimeError("dedicated genuine artifact-control proof unavailable: " + result.reason)
    token = result.completion
    delivered = []
    gate.deliver(token, actual, lambda text: delivered.append(text))
    if delivered != [actual.final]:
        raise AssertionError("live exact final consumer failed")
    for forged in (
        copy.copy(token),
        copy.deepcopy(token),
        gate.Completion(token.identity, token.report),
        gate.Completion("invented", "invented"),
    ):
        refused(
            lambda forged=forged: gate.deliver(
                forged, actual, lambda _: (_ for _ in ()).throw(AssertionError("callback ran"))
            )
        )
    refused(lambda: pickle.dumps(token))
    refused(lambda: gate.validate_completion(token, dataclasses.replace(actual)))
    report = json.loads(Path(token.report).read_text())
    targets = {
        "facts_original": Path(request["facts"]),
        "software_original": Path(request["software"]),
        "source_snapshot": Path(report["buffers"][0]["path"]),
        "final_snapshot": Path(report["buffers"][1]["path"]),
        "query": Path(report["query"]["path"]),
        "query_object": Path(report["object"]["path"]),
        "project_object": next(
            Path(row["path"])
            for row in report["library"]
            if "Merlin" in Path(row["path"]).relative_to(Path(report["library_root"])).parts
        ),
        "raw_process_log": Path(report["process_evidence"][0]["path"]),
        "report": Path(token.report),
    }
    rows = []
    for name, path in targets.items():
        if not path.resolve().is_relative_to(output.resolve()) and not path.resolve().is_relative_to(objects.resolve()):
            raise AssertionError("attempted mutation of foreign source/evidence")
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\nF_OWNED_STALE_CONTROL\n")
            refused(lambda: gate.validate_completion(token, actual))
        finally:
            path.write_bytes(original)
        gate.validate_completion(token, actual)
        rows.append(dict(case=name, refused=True, original_bytes_restored=True))
    library = Path(report["library_root"])
    extra = library / "F_extra_unbound_object.txt"
    extra.write_text("unbound membership")
    try:
        refused(lambda: gate.validate_completion(token, actual))
    finally:
        extra.unlink()
    gate.validate_completion(token, actual)
    rows.append(dict(case="extra_object", refused=True))
    path = Path(report["object"]["path"])
    original = path.read_bytes()
    target = output / "own-object-copy.olean"
    target.write_bytes(original)
    path.unlink()
    path.symlink_to(target.resolve())
    try:
        refused(lambda: gate.validate_completion(token, actual))
    finally:
        path.unlink()
        path.write_bytes(original)
    gate.validate_completion(token, actual)
    rows.append(dict(case="object_symlink", refused=True))
    save(
        output / "ARTIFACTS.json",
        dict(
            status="GENUINE_LIVE_ARTIFACT_CONTROLS_PASSED",
            cases=rows,
            constructed_copied_saved_tokens_refused=5,
            copied_candidate_refused=True,
            native_positive_proofs=1,
            no_mock_positive_proof=True,
            human_approved=False,
        ),
    )
    return len(rows)
