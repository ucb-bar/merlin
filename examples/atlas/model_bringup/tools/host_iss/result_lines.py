"""Validate the UART result transcript emitted by an admitted full host."""

import re


def verify(raw, name, launches, *, progress_launches=None):
    if type(launches) is not int or launches <= 0:
        raise ValueError("positive result launch count required")
    records = []
    progress = []
    completed = 0
    finished = False
    for line in raw.decode("ascii").splitlines():
        if not line.startswith("ATLAS_KERNEL_"):
            continue
        fields = line.split()
        kind = fields[0].removeprefix("ATLAS_KERNEL_")
        if kind not in ("BEGIN", "PROGRESS", "PASS", "FAIL"):
            raise ValueError("unknown full-host UART result line")
        values = {}
        for field in fields[1:]:
            if "=" not in field:
                raise ValueError("malformed full-host UART result field")
            key, value = field.split("=", 1)
            if key in values:
                raise ValueError("duplicate full-host UART result field")
            values[key] = value
        if (values.get("name") != name or values.get("launches_expected") != str(launches)
                or re.fullmatch(r"[0-9]+", values.get("launches_completed", "")) is None):
            raise ValueError("UART workload or launch identity differs")
        count = int(values["launches_completed"])
        if finished or kind == "FAIL":
            raise ValueError("UART reports failure or records after completion: " + line)
        if not records:
            if kind != "BEGIN" or count != 0:
                raise ValueError("UART begin record required first")
        elif kind == "PROGRESS":
            if not completed < count <= launches or not values.get("boundary"):
                raise ValueError("UART progress order or boundary differs")
            completed = count
            progress.append(count)
        elif kind == "PASS":
            if count != completed or count != launches:
                raise ValueError("UART completed launch count differs")
            finished = True
        else:
            raise ValueError("duplicate UART begin record")
        records.append(dict(kind=kind, **values))
    if not finished:
        raise ValueError("UART final PASS record missing")
    if progress_launches is not None and progress != progress_launches:
        raise ValueError("UART workload layer/segment progress coverage differs")
    return dict(begin=True, progress_records=len(records) - 2, final="PASS", launches_completed=completed)
