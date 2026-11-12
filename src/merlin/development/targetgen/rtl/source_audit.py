"""Compare authored direct RTL audit questions with actual extracted artifacts."""

import argparse
import json
from pathlib import Path

import yaml

from merlin.development.targetgen.rtl.circt_introspect import _module_port_sig
from merlin.development.targetgen.rtl.source_selection import digest, load_selection


def _register_bank_geometry(text: str, check: dict) -> tuple[str, dict]:
    """Check port geometry only; this does not establish register behavior or scale semantics."""
    signature = _module_port_sig(text, check["module"])
    if signature is None:
        return "unknown", {"reason": "module_not_found"}
    ports = {}
    for member in signature.split(","):
        lhs, separator, typ = member.strip().rpartition(" : ")
        if not separator:
            continue
        words = lhs.split()
        if len(words) == 2 and words[0] in {"in", "out"}:
            ports[words[1].lstrip("%")] = {"direction": words[0], "type": typ}
    prefix = check["output_prefix"]
    outputs = {
        name: value
        for name, value in ports.items()
        if name.startswith(prefix)
        and bool(name[len(prefix) :])
        and all("0" <= char <= "9" for char in name[len(prefix) :])
    }
    expected_count = check["expected_entries"]
    index_type = check["expected_index_type"]
    if (
        not isinstance(expected_count, int)
        or not isinstance(index_type, str)
        or not index_type.startswith("i")
        or not index_type[1:]
        or not "1" <= index_type[1] <= "9"
        or not all("0" <= char <= "9" for char in index_type[1:])
    ):
        raise ValueError("invalid register-bank audit expectation")
    if expected_count != 1 << int(index_type[1:]):
        raise ValueError("register-bank entry count must match index width")
    observed = {
        "write_index": ports.get(check["write_index_port"]),
        "write_data": ports.get(check["write_data_port"]),
        "output_names": sorted(outputs),
        "output_types": sorted({(value["direction"], value["type"]) for value in outputs.values()}),
        "entry_count": len(outputs),
    }
    matches = (
        observed["write_index"] == {"direction": "in", "type": index_type}
        and observed["write_data"] == {"direction": "in", "type": check["expected_data_type"]}
        and set(outputs) == {f"{prefix}{index}" for index in range(expected_count)}
        and observed["output_types"] == [("out", check["expected_data_type"])]
    )
    return ("verified" if matches else "mismatch"), observed


def build_source_audit(selection: dict, facts: dict, declaration: dict) -> dict:
    """Audit exact source slices, preserving disagreements instead of writing facts.

    The expected values are human-reviewed audit inputs. Observed source slices
    and extracted values are generated outputs; neither supplies hardware support
    or numerical conformance merely by agreeing.
    """
    checks = declaration.get("audit_checks") or []
    results, seen = [], set()
    text = Path(selection["sources"]["core_hw"]["path"]).read_text()
    for check in checks:
        identity = check["id"]
        if identity in seen:
            raise ValueError("duplicate RTL audit check")
        seen.add(identity)
        if check["kind"] == "hw_register_bank_geometry":
            source_status, observed = _register_bank_geometry(text, check)
            results.append(
                {
                    "id": identity,
                    "source_status": source_status,
                    "source": {
                        **selection["sources"]["core_hw"],
                        "module": check["module"],
                        "observed_geometry": observed,
                    },
                    "authored_expectation": {
                        key: check[key]
                        for key in (
                            "write_index_port",
                            "write_data_port",
                            "output_prefix",
                            "expected_entries",
                            "expected_index_type",
                            "expected_data_type",
                        )
                    },
                    "gap": check.get("gap"),
                    "qualification": check.get("qualification"),
                }
            )
            continue
        if check["kind"] != "hw_port_type":
            raise ValueError("unsupported direct RTL audit question")
        module, port = check["module"], check["port"]
        signature = _module_port_sig(text, module)
        observed, observed_direction, snippet = None, None, None
        if signature:
            for member in signature.split(","):
                lhs, separator, typ = member.strip().rpartition(" : ")
                if separator and lhs.split()[-1].lstrip("%") == port:
                    observed, snippet = typ, member.strip()
                    observed_direction = lhs.split()[0]
        projection = check.get("extraction") or {}
        records = (facts.get("facts") or {}).get(projection.get("collection")) or []
        rows = [row for row in records if row.get("name") == projection.get("name")]
        extracted = rows[0].get(projection.get("field")) if len(rows) == 1 else None
        if extracted is None and projection.get("field") == "elem_bits" and len(rows) == 1:
            from merlin.common.quant_formats import get as quant_format

            try:
                extracted = quant_format(rows[0].get("dtype")).element_bits
            except (KeyError, ValueError, TypeError):
                pass
        direction_matches = check.get("expected_direction") is None or observed_direction == check["expected_direction"]
        source_status = (
            "verified"
            if observed == check.get("expected_type") and direction_matches
            else "unknown"
            if observed is None
            else "mismatch"
        )
        comparable = check.get("comparison", "same_quantity") == "same_quantity"
        bits = int(observed[1:]) if observed and observed.startswith("i") and observed[1:].isdigit() else None
        extraction_status = (
            "different_quantity"
            if not comparable
            else "unknown"
            if bits is None or extracted is None
            else "agrees"
            if bits == extracted
            else "mismatch"
        )
        results.append(
            {
                "id": identity,
                "source_status": source_status,
                "source": {
                    **selection["sources"]["core_hw"],
                    "module": module,
                    "port": port,
                    "observed_type": observed,
                    "observed_direction": observed_direction,
                    "snippet": snippet,
                },
                "authored_expected_type": check.get("expected_type"),
                "authored_expected_direction": check.get("expected_direction"),
                "extraction": {**projection, "value": extracted, "status": extraction_status},
                "gap": check.get("gap"),
                "qualification": check.get("qualification"),
            }
        )
    return {
        "schema": "merlin.rtl_source_audit.v1",
        "target": selection["target"],
        "source_selection_sha256": selection.get("selection_sha256"),
        "status": "verified" if results and all(row["source_status"] == "verified" for row in results) else "unknown",
        "checks": results,
        "historical_hierarchy": selection.get("diagnostics"),
        "qualification": "manual audit checked against exact current RTL; not operation/numerical certification",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-bundle", "facts", "hardware-spec", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    declaration = yaml.safe_load(Path(args.hardware_spec).read_bytes())
    selected = load_selection(args.source_bundle, target=declaration["target"])
    facts = json.loads(Path(args.facts).read_bytes())
    if facts.get("inputs", {}).get("source_bundle_sha256") != selected["selection_sha256"]:
        raise ValueError("audit facts do not bind selected source bundle")
    report = build_source_audit(selected, facts, declaration)
    report["facts_sha256"] = digest(args.facts)
    report["hardware_spec_sha256"] = digest(args.hardware_spec)
    output = Path(args.output)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
