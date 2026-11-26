"""Compact, image-bound device launch order for generated C regions.

Schedules describe counted runs and repeated groups. They cannot introduce
program paths, binary payloads, or images outside the compiled region. Legacy
single-image multiplicity remains a separate call ABI.
"""

from pathlib import Path

SOURCE_BYTES = Path(__file__).read_bytes()

SCHEMA = "merlin.device_launch_schedule.v1"
MAX_LAUNCHES = (1 << 31) - 1
MAX_IMAGES = 1024
MAX_GROUPS = 1024
MAX_RUNS = 16384


def _positive(value, name):
    if type(value) is not int or not 1 <= value <= MAX_LAUNCHES:
        raise ValueError(f"device launch schedule {name} must be a positive bounded integer")
    return value


def _images(values):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_IMAGES:
        raise ValueError("device launch schedule requires a bounded nonempty image list")
    for value in values:
        if (type(value) is not str or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("device launch schedule image must be a canonical SHA-256")
    return list(values)


def _components(images, groups, *, image_sha256s, placement):
    if placement != "accelerator":
        raise ValueError("device launch schedule requires accelerator placement")
    images, actual = _images(images), _images(image_sha256s)
    if images != actual:
        raise ValueError("device launch schedule images differ from ordered native programs")
    if type(groups) is not list or not 1 <= len(groups) <= MAX_GROUPS:
        raise ValueError("device launch schedule requires bounded nonempty groups")
    canonical, used, total, runs_seen = [], set(), 0, 0
    for group in groups:
        if type(group) is not dict or set(group) != {"repeat", "runs"}:
            raise ValueError("device launch schedule group has missing or unknown fields")
        repeat = _positive(group["repeat"], "group repeat")
        runs = group["runs"]
        if type(runs) is not list or not runs:
            raise ValueError("device launch schedule group requires nonempty runs")
        runs_seen += len(runs)
        if runs_seen > MAX_RUNS:
            raise ValueError("device launch schedule exceeds its compact run bound")
        checked, per_group = [], 0
        for run in runs:
            if type(run) is not dict or set(run) != {"image", "count"}:
                raise ValueError("device launch schedule run has missing or unknown fields")
            image = run["image"]
            if type(image) is not int or not 0 <= image < len(images):
                raise ValueError("device launch schedule image index leaves native programs")
            count = _positive(run["count"], "run count")
            per_group += count
            if per_group > MAX_LAUNCHES:
                raise ValueError("device launch schedule launch total exceeds its bound")
            used.add(image)
            checked.append(dict(image=image, count=count))
        total += repeat * per_group
        if total > MAX_LAUNCHES:
            raise ValueError("device launch schedule launch total exceeds its bound")
        canonical.append(dict(repeat=repeat, runs=checked))
    if used != set(range(len(images))):
        raise ValueError("device launch schedule leaves a native image unused")
    return dict(schema=SCHEMA, images=images, groups=canonical, total_launches=total)


def make_launch_schedule(image_sha256s, groups, *, placement="accelerator"):
    """Build canonical metadata from the actual ordered region image hashes."""
    return _components(image_sha256s, groups, image_sha256s=image_sha256s, placement=placement)


def validate_launch_schedule(schedule, *, image_sha256s, placement):
    """Bind all indices and hashes, validate counts, and return an owned copy."""
    if type(schedule) is not dict or set(schedule) != {"schema", "images", "groups", "total_launches"}:
        raise ValueError("device launch schedule has missing or unknown fields")
    if schedule["schema"] != SCHEMA:
        raise ValueError("device launch schedule schema differs")
    canonical = _components(schedule["images"], schedule["groups"], image_sha256s=image_sha256s, placement=placement)
    if type(schedule["total_launches"]) is not int or schedule["total_launches"] != canonical["total_launches"]:
        raise ValueError("device launch schedule declared total differs from counted runs")
    return canonical


def iter_launch_schedule(schedule, *, image_sha256s, placement="accelerator"):
    """Validate eagerly, then yield image hashes lazily in exact launch order."""
    checked = validate_launch_schedule(schedule, image_sha256s=image_sha256s, placement=placement)

    def launches():
        for group in checked["groups"]:
            for _ in range(group["repeat"]):
                for run in group["runs"]:
                    for _ in range(run["count"]):
                        yield checked["images"][run["image"]]
    return launches()


def expand_launch_schedule(schedule, *, image_sha256s, placement="accelerator", max_launches=1000000):
    """Return a bounded tuple for consumers that need a complete launch ledger."""
    _positive(max_launches, "expansion limit")
    checked = validate_launch_schedule(schedule, image_sha256s=image_sha256s, placement=placement)
    if checked["total_launches"] > max_launches:
        raise ValueError("device launch schedule exceeds the requested expansion limit")
    return tuple(iter_launch_schedule(checked, image_sha256s=image_sha256s, placement=placement))
