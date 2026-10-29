"""Small synthetic contracts, never actual engine or hardware qualification."""

from merlin_timing import contracts as C


def h(value):
    return C.digest(value)


def sample(*, profile="phase1-public", held_failure=False):
    domain = {name: h(name) for name in C.DOMAIN_KEYS - {"label_engine"}}
    domain["label_engine"] = "gsim"
    data = {
        "schema": C.DATASET,
        "profile": profile,
        "domain": domain,
        "feature_names": ["host_instructions"],
        "cases": [],
        "records": [],
    }
    for split, groups in (("train", range(4)), ("calibration", range(4, 6)), ("test", range(6, 12))):
        for group in groups:
            for variant, count in enumerate((10, 20, 30, 40)):
                identity = f"{split}-{group}-{variant}"
                source = {
                    "visibility": C.PROFILES[profile][0],
                    "phase": C.PROFILES[profile][1],
                    "manifest_sha256": h("manifest"),
                    "source_sha256": h(["source", group]),
                }
                case = {
                    "case_id": identity,
                    "source_group_sha256": h(["group", group]),
                    "workload_sha256": "",
                    "mechanism_family": f"family-{group % 3}",
                    "split": split,
                    "source": source,
                    "elf_sha256": h(identity),
                    "input_sha256": h(["input", group]),
                    "scope_sha256": domain["scope_sha256"],
                    "device_sha256": domain["device_sha256"],
                }
                case["workload_sha256"] = C.workload(case)
                data["cases"].append(case)
                label = 5 + 3 * count
                if held_failure and split == "test":
                    label = 150 - 3 * count
                data["records"].append(
                    {
                        "record_id": identity,
                        "case_id": identity,
                        "features": {"host_instructions": count},
                        "feature_provenance": {
                            "engine_sha256": domain["feature_engine_sha256"],
                            "model_sha256": domain["target_model_sha256"],
                            "trace_sha256": h([identity, "trace"]),
                            "execution_sha256": C.execution_identity(case),
                            "complete": True,
                        },
                        "label": {
                            "engine": domain["label_engine"],
                            "engine_sha256": domain["label_engine_sha256"],
                            "receipt_sha256": h([identity, "measurement"]),
                            "execution_sha256": C.execution_identity(case),
                            "status": "pass",
                            "numeric": "pass",
                            "measured_now": True,
                            "cycles": label,
                        },
                        "event_report": None,
                    }
                )
    return data
