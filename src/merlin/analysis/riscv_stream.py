"""Record and expand bounded periodic accelerator command streams."""

from __future__ import annotations

from .riscv_values import MASK

_MAX_PERIOD = 8


def _run_step(first, value):
    """The constant step from ``first`` to ``value`` (an int, or a pointer's byte offset), or ``None``."""
    if type(first) is int and type(value) is int:
        return (value - first) & MASK
    if isinstance(first, dict) and isinstance(value, dict) and first.get("pointer_base"):
        same = {k: v for k, v in first.items() if k != "byte_offset"} == {
            k: v for k, v in value.items() if k != "byte_offset"
        }
        return value["byte_offset"] - first["byte_offset"] if same else None
    return 0 if first == value else None


def _advance(value, step, k):
    if not step:
        return value
    if type(value) is int:
        return (value + step * k) & MASK
    return dict(value, byte_offset=value["byte_offset"] + step * k)


def _slot(entry):
    return dict(entry, rs1_step=0, rs2_step=0, next_host=entry["host"])


def _identity(entry):
    return entry["pc"], entry.get("kind"), entry.get("funct")


class _StreamRecorder:
    """Issue-order record as periodic affine runs.

    A run is a period of up to ``_MAX_PERIOD`` issue slots repeated ``count`` times
    (plus ``partial`` leading slots of one more period); each slot's operands advance
    by a constant step per period, and every repeat has the same host gaps. At most
    ``max_runs`` runs are kept; beyond that ``overflow`` is set and recording stops.
    """

    def __init__(self, max_runs):
        self.runs, self.max_runs, self.entries, self.overflow = [], max_runs, 0, False

    def add(self, pc, host, funct=None, rs1=None, rs2=None, kind=None):
        if self.overflow:
            self.entries += 1
            return
        entry = dict(pc=pc, host=host)
        entry.update(dict(kind=kind) if kind is not None else dict(funct=funct, rs1=rs1, rs2=rs2))
        seq, self.entries = self.entries, self.entries + 1
        if self._extend(entry):
            return
        self._split_partial()
        if self._extend(entry):
            return
        self._push(dict(seq=seq, slots=[_slot(entry)], count=1, partial=0))
        self._fold()

    def _expected(self, run):
        slot = run["slots"][run["partial"]]
        period = run["count"]
        return slot, period

    def _extend(self, entry):
        if not self.runs:
            return False
        run = self.runs[-1]
        if run["count"] < 2:
            return False
        slot, period = self._expected(run)
        if _identity(slot) != _identity(entry) or slot["next_host"] != entry["host"]:
            return False
        if "kind" not in slot and (
            _advance(slot["rs1"], slot["rs1_step"], period) != entry["rs1"]
            or _advance(slot["rs2"], slot["rs2_step"], period) != entry["rs2"]
        ):
            return False
        run["partial"] += 1
        if run["partial"] == len(run["slots"]):
            run["count"], run["partial"] = run["count"] + 1, 0
        return True

    def _split_partial(self):
        """A period broken midway leaves its issued slots as single-issue runs."""
        if not self.runs or not self.runs[-1]["partial"]:
            return
        run = self.runs[-1]
        width, period = len(run["slots"]), run["count"]
        seq = run["seq"] + width * period
        tail = []
        for index, slot in enumerate(run["slots"][: run["partial"]]):
            entry = dict(pc=slot["pc"], host=slot["next_host"])
            if "kind" in slot:
                entry["kind"] = slot["kind"]
            else:
                entry.update(
                    funct=slot["funct"],
                    rs1=_advance(slot["rs1"], slot["rs1_step"], period),
                    rs2=_advance(slot["rs2"], slot["rs2_step"], period),
                )
            tail.append(dict(seq=seq + index, slots=[_slot(entry)], count=1, partial=0))
        run["partial"] = 0
        for item in tail:
            self._push(item)

    def _push(self, run):
        if len(self.runs) >= self.max_runs:
            self.overflow = True
            return
        self.runs.append(run)

    def _fold(self):
        """Merge the last ``2w`` single-issue runs into one run of period ``w`` when they repeat."""
        runs = self.runs
        for width in range(1, _MAX_PERIOD + 1):
            if len(runs) < 2 * width:
                return
            tail = runs[-2 * width :]
            if any(r["count"] != 1 or len(r["slots"]) != 1 or r["partial"] for r in tail):
                return
            first, second = [r["slots"][0] for r in tail[:width]], [r["slots"][0] for r in tail[width:]]
            if any(_identity(a) != _identity(b) for a, b in zip(first, second, strict=True)):
                continue
            if any(a["host"] != b["host"] for a, b in zip(first[1:], second[1:], strict=True)):
                continue
            steps = []
            for a, b in zip(first, second, strict=True):
                pair = (0, 0) if "kind" in a else (_run_step(a["rs1"], b["rs1"]), _run_step(a["rs2"], b["rs2"]))
                if None in pair:
                    break
                steps.append(pair)
            if len(steps) != width:
                continue
            slots = [
                dict(a, rs1_step=s1, rs2_step=s2, next_host=b["host"])
                for a, b, (s1, s2) in zip(first, second, steps, strict=True)
            ]
            del runs[-2 * width :]
            runs.append(dict(seq=tail[0]["seq"], slots=slots, count=2, partial=0))
            return


def expand_command_stream(runs, *, limit=None):
    """Entries of a ``command_stream_runs`` record in issue order (at most ``limit``)."""
    produced = 0
    for run in runs:
        width = len(run["slots"])
        for k in range(run["count"] * width + run["partial"]):
            if limit is not None and produced >= limit:
                return
            period, slot = divmod(k, width)
            slot = run["slots"][slot]
            entry = dict(seq=run["seq"] + k, pc=slot["pc"])
            if "kind" in slot:
                entry["kind"] = slot["kind"]
            else:
                entry.update(
                    funct=slot["funct"],
                    rs1=_advance(slot["rs1"], slot["rs1_step"], period),
                    rs2=_advance(slot["rs2"], slot["rs2_step"], period),
                )
            entry["host_instructions_since_prev"] = slot["host"] if period == 0 else slot["next_host"]
            yield entry
            produced += 1
