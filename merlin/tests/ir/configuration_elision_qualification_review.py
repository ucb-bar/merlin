"""Independent decoded-field/source association checks, not proof authority."""

import hashlib


def _effect(value, fields):
    if value is None:
        return None

    def field(index):
        if type(index) is not int or not 0 <= index < len(fields):
            raise ValueError("malformed field index")
        return fields[index]

    def assignments(rows):
        if type(rows) is not list:
            raise ValueError("malformed field assignments")
        result = []
        for pair in rows:
            if type(pair) is not list or len(pair) != 2 or type(pair[1]) is not int:
                raise ValueError("malformed field/value pair")
            result.append([field(pair[0]), pair[1]])
        return result

    if type(value["ordered"]) is not bool or type(value["pureWrite"]) is not bool:
        raise ValueError("malformed effect flag")
    return {
        "reads": assignments(value["reads"]),
        "writes": assignments(value["writes"]),
        "clobbers": [field(index) for index in value["clobbers"]],
        "ordered": value["ordered"],
        "pure_write": value["pureWrite"],
    }


def audit_decoding(projection, producer, before, after):
    """Compare every actual original-codec decode to Q's independent projection.

    Retained constants and opaque operations are explicitly distinguished. This
    is a host cross-check of separately exported data; Lean must still check the
    actual complete-program obligation and the effect meaning remains a premise.
    """
    if projection["source_sha256"] != hashlib.sha256(before.encode()).hexdigest():
        raise ValueError("independent source bytes differ from checked projection")
    if projection["final_sha256"] != hashlib.sha256(after.encode()).hexdigest():
        raise ValueError("independent final bytes differ from checked projection")
    fields = projection["fields"]
    if type(fields) is not list or any(type(item) is not str for item in fields) or fields != sorted(set(fields)):
        raise ValueError("noninjective or malformed actual field registry")
    instructions = 0
    entries = 0
    for key, decoded in (("before", producer["source_decoding"]), ("after", producer["final_decoding"])):
        actual = projection[key]["entries"]
        if [entry["name"] for entry in actual] != [entry["function"] for entry in decoded]:
            raise ValueError("independent full defined-entry roster mismatch")
        for entry, original in zip(actual, decoded, strict=True):
            entries += 1
            if len(entry["code"]) != len(original["operations"]):
                raise ValueError("independent complete instruction roster mismatch")
            for atom, row in zip(entry["code"], original["operations"], strict=True):
                if row.get("instruction") is not None:
                    expected = {
                        "instruction": row["instruction"],
                        "arguments": row["bindings"],
                        "word": row["encoded_word"],
                    }
                    if atom["decoded"] != expected or _effect(atom["effect"], fields) != row["effect"]:
                        raise ValueError("actual codec/field/value/effect decoding mismatch")
                    instructions += 1
                elif row["operation"] == "arith.constant":
                    if atom["effect"] != {
                        "reads": [],
                        "writes": [],
                        "clobbers": [],
                        "ordered": True,
                        "pureWrite": False,
                    }:
                        raise ValueError("constant changed declared state tracking")
                elif atom["effect"] is not None or atom["eraseAllowed"]:
                    raise ValueError("opaque operation gained configuration deletion permission")
    return {
        "status": "INDEPENDENT_ALL_DECLARED_INSTRUCTION_DECODES_AGREE",
        "entry_endpoints": entries,
        "instruction_endpoints": instructions,
        "proof_authority": False,
        "hardware_model_approved": False,
    }
