#!/usr/bin/env python3
"""One-step setup for the chip examples on a new machine.

    python examples/lib/chip_setup.py [--llvm PREFIX] [--chipyard DIR] [--zephyr-sw DIR] [--zephyr-sdk DIR]
                                      [--m2m DIR] [--support TARGET] [--verify-pin PIN] [--checkout-pin PIN]
                                      [--build-spike PIN=ENV] [--spike-model TARGET] [--no-fetch]

Every chip example calls this through its ``run.sh setup``, which passes what that chip needs on top of
the common prerequisites (``--support`` / ``--verify-pin`` / ``--checkout-pin`` / ``--build-spike`` /
``--spike-model``). Nothing here names a chip.

It sorts the prerequisites by what they cost:

* **Done here, automatically** (minutes, a few GB at most): the model2MLIR capture environment (a Python
  3.12 venv under ``out/build/venv/`` with an exact PyTorch CPU release, ``CAPTURE_TORCH``, and the
  torch-mlir wheel that matches Merlin's LLVM pin, ``TORCH_MLIR``), model2MLIR at its pinned revision
  (cloned when none is configured, and moved onto the pin when it is setup's own clean clone), the chip's
  support package at the exact commit Merlin pins (``--support``, see
  ``merlin.targetgen.support_pins``; its base comes from the same single base pin), the Zephyr west
  workspace, a chip package's spike model of its unit, and a simulator built
  from a VERIFIED source checkout. Each step is skipped when its result is already there and still right.
* **Found and verified, never fetched** (large, or the location is yours to choose): LLVM/MLIR 23,
  chipyard with riscv-tools, the Zephyr SDK, and the pinned hardware checkouts a chip reads its facts
  from. A checkout is verified by CONTENT against ``merlin/contract/hardware_pins.yaml`` (commit,
  required paths, and the gitlinks of the pins nested in it), never by its branch name. For each one
  missing or on the wrong revision it prints the exact commands, and what they cost.

Whatever is found and right is written to the repo's ``.env`` (process environment still wins), so the
examples resolve it from any shell. Paths exported in the environment are adopted the same way. It exits
0 only when nothing is left to do.

It needs ``uv`` and ``git`` on PATH. It writes only under ``out/build/`` and ``.env``, and it never
modifies a checkout it did not create.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: PyTorch's CPU index. It keeps every release; its nightly index keeps about two months, so a capture venv
#: built from a nightly cannot be rebuilt the same way a few weeks later.
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
#: The capture venv's torch stack, at exact releases from :data:`TORCH_INDEX`. torch-mlir's dev wheel links
#: no libtorch (its FX importer runs on torch's public export API), so it works across torch releases; this
#: pair was checked by capturing the one-op GEMM with it and with the nightly setup used to install
#: (2.16.0.dev20261105): model.mlir and every other bundle file identical but for paths. Setup checks the pair
#: again on every venv it builds (:func:`capture_smoke`). Move them together, and re-check.
CAPTURE_TORCH = ("torch==2.14.1", "torchvision==0.29.1", "torchao==0.18.0")
TORCH_MLIR_WHEELS = "https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels"
#: The newest torch-mlir dev wheel at or before Merlin's LLVM pin (llvm-project a47bddcc, 2026-06-03; see
#: docs/guides/llvm_toolchain.md). Lowering prints LLVM-dialect MLIR in-process with this wheel and the
#: pinned mlir-translate reads it; a much newer wheel prints `llvm.load ... <alignment = N>`, which that
#: mlir-translate rejects. Move this together with the LLVM pin.
TORCH_MLIR = "torch-mlir==20260531.828"
#: model2MLIR's own runtime dependencies (installed apart so they do not move torch). Each is installed at
#: the version the model2MLIR checkout's own ``uv.lock`` records (see :func:`capture_deps`): model2MLIR
#: calls their APIs directly, and its ``>=`` floors do not keep them from moving under it. An unpinned
#: ``xdsl`` resolved to 0.74, which no longer has ``Printer.print``, and every capture that printed its
#: module died in model2MLIR with an AttributeError.
CAPTURE_LOCKED_DEPS = ("xdsl", "structlog", "numpy", "PyYAML", "ml_dtypes", "safetensors", "timm")
#: The version of each :data:`CAPTURE_LOCKED_DEPS` entry that model2MLIR's lock does not record (they are
#: not its dependencies, or only optional ones), so the venv is the same on every machine.
CAPTURE_UNLOCKED_PINS = {"ml_dtypes": "0.6.0", "timm": "1.0.31"}
#: Written into the capture venv: what was installed, and the result of the compatibility check.
CAPTURE_RECORD = "merlin-capture-env.json"
#: What the capture item records in .env: the model2MLIR checkout (both spellings), its venv, its python.
CAPTURE_KEYS = ("MERLIN_M2M_DIR", "MERLIN_MODEL2MLIR", "MERLIN_M2M_VENV", "MERLIN_M2M_PYTHON")
#: The software pin model2MLIR captures are verified against (merlin/contract/software_pins.yaml).
M2M_PIN = "model2mlir_capture"
ZEPHYR_SW_URL = "https://github.com/ucb-bar/zephyr-chipyard-sw.git"
ZEPHYR_SW_BRANCH = "dev"
#: The manifest zephyr-chipyard-sw's own installer selects: the RISC-V modules only.
ZEPHYR_MANIFEST = "west-riscv.yml"
ZEPHYR_SDK = "0.17.0"
#: Variables this script adopts from the process environment into .env when they point at something real.
ADOPTED = (
    "MERLIN_CLANG",
    "MERLIN_MLIR_INSTALL",
    "MERLIN_MLIR_TRANSLATE",
    "MERLIN_CHIPYARD",
    "MERLIN_SPIKE",
    "MERLIN_RISCV_GCC",
    "ZEPHYR_BASE",
    "MERLIN_ZEPHYR_SW",
    "ZEPHYR_SDK_INSTALL_DIR",
    "MERLIN_M2M_DIR",
    "MERLIN_M2M_VENV",
)

#: How .env.example spells "set this to your path"; a .env copied from it carries these.
PLACEHOLDER = "/path/to/"
#: ``warn``: usable, but a result built with it is about another revision than the pin (reported, and
#: recorded by every build that reads it); the others that are not ok/built leave something to do.
OK, BUILT, WARN, MISSING, DRIFT, BLOCKED = "ok", "built", "warn", "missing", "drift", "blocked"


@dataclass
class Item:
    """One prerequisite: whether it is satisfied, where, and the exact commands that would satisfy it."""

    name: str
    state: str
    detail: str = ""
    fix: list[str] = field(default_factory=list)
    cost: str = ""
    settings: dict[str, str] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.state in (OK, BUILT, WARN)


# --------------------------------------------------------------------------------------------- helpers


def _run(cmd: list, **kw) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def _dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                out[key.strip()] = value.strip()
    return out


def _upsert_env(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    seen = set()
    for i, line in enumerate(lines):
        key = line.partition("=")[0].strip()
        if key in values and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={values[key]}"
            seen.add(key)
    lines += [f"{k}={v}" for k, v in values.items() if k not in seen]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(values)} setting(s) to {path}")


def _build_dir() -> Path:
    from merlin.common.paths import build_dir

    return build_dir()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def _env_set(values: dict[str, str]) -> Iterator[None]:
    """Set variables in THIS process only, for code that resolves a checkout through its root variable."""
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------------------- pinned checkouts


def _pins(pins_path: Path | None = None) -> dict:
    from merlin.common import provenance as PROV

    return PROV.load_pins(pins_path)


def nested_pins(name: str, pins: dict) -> list:
    """The pins whose checkout lives INSIDE ``name``'s (``nested_in``), e.g. submodules it records."""
    return [p for p in pins.values() if p.nested_in == name]


def fetch_commands(pin, nested: list, dest: Path) -> list[str]:
    """The commands that put ``pin``'s revision (and its nested pins' gitlinks) at ``dest``.

    Derived from the registry entry, so the commands and the verification can never disagree. When the
    pin declares a branch, it is checked out AS that branch at the pinned commit: verification compares
    the branch too, and a detached HEAD would read as drift.
    """
    if not pin.repo_canonical:
        return [f"# {pin.name}: the registry names no repository; see merlin/contract/hardware_pins.yaml"]
    cmds = [f"git clone {pin.repo_observed or pin.repo_canonical} {dest}"]
    if pin.branch:
        cmds.append(f"git -C {dest} checkout -B {pin.branch} {pin.commit}")
    else:
        cmds.append(f"git -C {dest} checkout --detach {pin.commit}")
    paths = sorted({n.nested_path for n in nested if n.nested_path})
    if paths:
        cmds.append(f"git -C {dest} submodule update --init {' '.join(paths)}")
    return cmds


def repair_commands(pin, nested: list, checkout: Path) -> list[str]:
    """The commands that move an existing checkout onto the pin. PRINTED only: this script never moves
    a checkout it did not create (other sessions may be working in it)."""
    if pin.branch:
        cmds = [f"git -C {checkout} fetch origin", f"git -C {checkout} checkout -B {pin.branch} {pin.commit}"]
    else:
        cmds = [f"git -C {checkout} fetch origin", f"git -C {checkout} checkout --detach {pin.commit}"]
    paths = sorted({n.nested_path for n in nested if n.nested_path})
    if paths:
        cmds.append(f"git -C {checkout} submodule update --init {' '.join(paths)}")
    return cmds


def checkout_item(
    name: str,
    candidate: Path | None,
    *,
    default: Path,
    pins_path: Path | None = None,
    verify: Callable | None = None,
    cost: str = "",
) -> Item:
    """Find ``name``'s checkout and verify it against its pin (and every pin nested in it).

    ``candidate`` is the configured location (its ``root_env``), else ``default`` is looked at. Three
    outcomes: verified (recorded in .env), present but on another revision (the drift and the commands
    that would fix it; nothing recorded), or absent (the commands that fetch it to ``default``).
    """
    from merlin.common import provenance as PROV

    verify = verify or PROV.verify
    pins = _pins(pins_path)
    if name not in pins:
        raise SystemExit(f"{name}: no such pin in the registry")
    pin = pins[name]
    if pin.path not in ("", "."):
        raise SystemExit(f"{name}: --checkout-pin takes a pin whose checkout is its root (path '.'), not {pin.path!r}")
    nested = nested_pins(name, pins)
    where = candidate if candidate is not None else default
    if not where.is_dir():
        hint = f"; ${pin.root_env} points there" if candidate is not None else ""
        return Item(
            name=f"{name} ({pin.root_env})",
            state=MISSING,
            detail=f"no checkout at {where}{hint}",
            fix=fetch_commands(pin, nested, default) + ["then run setup again (it looks there)"],
            cost=cost,
        )
    problems: list[str] = []
    # Nested pins find their container through its root variable, so verify with it pointing HERE.
    with _env_set({pin.root_env: str(where)}):
        for p in [pin, *nested]:
            kw = {"checkout": where / p.path if p.path not in ("", ".") else where}
            if pins_path is not None:
                kw["path"] = pins_path
            ver = verify(p.name, **kw)
            if not ver.ok:
                problems += [f"{p.name}: {d}" for d in ver.drift]
                problems += [f"{p.name}: missing {d}" for d in ver.missing_paths]
                problems += [f"{p.name}: must not have {d}" for d in ver.forbidden_present]
    if problems:
        fix = ["either move this checkout onto the pin (if it is yours to move):"]
        fix += [f"  {c}" for c in repair_commands(pin, nested, where)]
        if where != default:
            fix += ["or leave it alone and fetch a separate one where setup looks:"]
            fix += [f"  {c}" for c in fetch_commands(pin, nested, default)]
            fix += [f"  and unset ${pin.root_env} (in the environment and in .env)"]
        return Item(
            name=f"{name} ({pin.root_env})",
            state=DRIFT,
            detail=f"{where} is not the pinned revision:\n      " + "\n      ".join(problems),
            fix=fix,
            cost=cost,
        )
    return Item(
        name=f"{name} ({pin.root_env})",
        state=OK,
        detail=f"{where} at {pin.commit[:12]}" + (f" (+ {', '.join(n.name for n in nested)})" if nested else ""),
        settings={pin.root_env: str(where)},
    )


def verify_pin_item(
    name: str, settings: dict[str, str], *, reads: tuple[str, ...] = (), pins_path: Path | None = None
) -> Item:
    """Verify a pin whose checkout lives inside a prerequisite configured elsewhere (e.g. a submodule of
    the chipyard in MERLIN_CHIPYARD), by what a build would READ from it (``reads``, plus the pin's
    required paths; :func:`merlin.common.provenance.materiality`).

    Another commit or branch whose read bytes are all the pin's is a WARNING: results are right, and every
    build records the revision it read. A read byte that differs (or a required file missing) is DRIFT,
    and blocks: the builds that derive from the checkout refuse it (``merlin.targetgen.unit_pins``) unless
    ``MERLIN_ALLOW_OFF_PIN`` names the pin. Never repaired here: the containing checkout is the user's."""
    from merlin.common import pin_drift as D
    from merlin.targetgen import unit_pins

    pin = _pins(pins_path)[name]
    label = f"{name} (${pin.root_env}/{pin.path})"
    root = settings.get(pin.root_env) or os.environ.get(pin.root_env)
    if not root:
        return Item(name=label, state=BLOCKED, detail=f"${pin.root_env} is not configured")
    checkout = Path(root) / pin.path if pin.path not in ("", ".") else Path(root)
    with _env_set({pin.root_env: root}):
        m = D.materiality(name, checkout=checkout, reads=reads, **({"path": pins_path} if pins_path else {}))
        fix = unit_pins.fix_commands(name, pins_path) if m.material else []
    if m.verdict == D.DRIFT_NONE:
        return Item(name=label, state=OK, detail=f"{checkout} at {pin.commit[:12]}")
    seen = m.verification.observed
    head = f"{checkout} is at {seen.commit[:12]}, not the pin's {pin.commit[:12]}"
    if not m.material:
        return Item(
            name=label,
            state=WARN,
            detail=f"{head}; every file a build reads from it is the pin's, so results are right and record it",
            fix=["to make it the pinned revision too, see the chip example's README"],
        )
    allowed = D.allowed_off_pin(name)
    return Item(
        name=label,
        state=WARN if allowed else DRIFT,
        detail=f"{head}, and a file a build reads differs (builds that use it "
        + ("proceed: you allowed it in $MERLIN_ALLOW_OFF_PIN" if allowed else "REFUSE it")
        + "):\n      "
        + "\n      ".join(m.reasons),
        fix=[
            "put it on the pin (the chip example's README has the exact steps):",
            *(f"  {c}" for c in fix),
            f"or accept it knowingly (each build records the drift): {D.ALLOW_OFF_PIN_ENV}={name} in .env",
        ],
    )


# ---------------------------------------------------------------------------- simulator from source


def spike_build_needed(stamp: dict | None, *, source: Path, commit: str, binary: Path) -> bool:
    """True unless the recorded build is of this exact source revision and its binary is unchanged."""
    if not stamp or not binary.is_file():
        return True
    return not (
        stamp.get("source") == str(source)
        and stamp.get("commit") == commit
        and stamp.get("binary_sha256") == _sha256(binary)
    )


def build_spike(pin_name: str, env_name: str, source_item: Item, *, prefix: Path, jobs: int | None = None) -> Item:
    """Build a riscv-isa-sim from a VERIFIED checkout into ``prefix`` (out of tree; the source is only
    read), and record its binary under ``env_name``. Skipped when ``prefix`` already holds this revision."""
    from merlin.common import provenance as PROV

    name = f"{pin_name} simulator (${env_name})"
    if not source_item.done:
        return Item(name=name, state=BLOCKED, detail=f"needs a verified {pin_name} checkout first (above)")
    pin = _pins()[pin_name]
    source = Path(source_item.settings[pin.root_env])
    binary = prefix / "bin" / "spike"
    stamp_path = prefix / "merlin-build.json"
    stamp = json.loads(stamp_path.read_text()) if stamp_path.is_file() else None
    if not spike_build_needed(stamp, source=source, commit=pin.commit, binary=binary):
        return Item(
            name=name, state=OK, detail=f"{binary} (built from {pin.commit[:12]})", settings={env_name: str(binary)}
        )
    missing = [t for t in ("make", "g++", "dtc") if shutil.which(t) is None]
    if missing:
        return Item(
            name=name,
            state=BLOCKED,
            detail=f"building it needs {', '.join(missing)} on PATH",
            fix=["sudo apt install build-essential device-tree-compiler   # or your distribution's equivalent"],
        )
    build = prefix / "build"
    build.mkdir(parents=True, exist_ok=True)
    _run([source / "configure", f"--prefix={prefix}"], cwd=build)
    _run(["make", f"-j{jobs or os.cpu_count() or 4}"], cwd=build)
    _run(["make", "install"], cwd=build)
    stamp = {
        "pin": pin_name,
        "commit": pin.commit,
        "source": str(source),
        "source_digest": PROV.source_digest([source / p for p in pin.requires_paths]),
        "binary": str(binary),
        "binary_sha256": _sha256(binary),
    }
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return Item(
        name=name, state=BUILT, detail=f"{binary} (built from {pin.commit[:12]})", settings={env_name: str(binary)}
    )


# ------------------------------------------------------------------------------ common prerequisites


def llvm_item(settings: dict[str, str]) -> Item:
    install = settings.get("MERLIN_MLIR_INSTALL")
    prefix = Path(install) if install else REPO / "third_party" / "llvm-install"
    clang = Path(settings.get("MERLIN_CLANG") or prefix / "bin" / "clang-23")
    translate = Path(settings.get("MERLIN_MLIR_TRANSLATE") or prefix / "bin" / "mlir-translate")
    if clang.is_file() and translate.is_file():
        keep = {
            k: settings[k] for k in ("MERLIN_MLIR_INSTALL", "MERLIN_CLANG", "MERLIN_MLIR_TRANSLATE") if k in settings
        }
        return Item(name="LLVM/MLIR 23 (clang-23, mlir-translate)", state=OK, detail=str(clang.parent), settings=keep)
    return Item(
        name="LLVM/MLIR 23 (clang-23, mlir-translate)",
        state=MISSING,
        detail=f"no {clang if not clang.is_file() else translate}",
        fix=[
            "build llvm-project a47bddcc with the two-target recipe in docs/guides/llvm_toolchain.md, or use an",
            "existing LLVM 23 install:  ./run.sh setup --llvm /path/to/llvm-install",
        ],
        cost="~13 GB, tens of minutes on a many-core host (hours on a laptop); no distribution packages it",
    )


def chipyard_item(settings: dict[str, str]) -> Item:
    from merlin.runtime.backends import spike

    with _env_set({k: v for k, v in settings.items() if k in ("MERLIN_CHIPYARD", "MERLIN_SPIKE", "MERLIN_RISCV_GCC")}):
        tools = [spike.spike_path(), spike.gcc_path()]
    missing = [t for t in tools if not t.is_file()]
    if not missing:
        keep = {k: settings[k] for k in ("MERLIN_CHIPYARD", "MERLIN_SPIKE", "MERLIN_RISCV_GCC") if k in settings}
        return Item(
            name="chipyard riscv-tools (spike, riscv64-unknown-elf-gcc)",
            state=OK,
            detail=str(tools[0].parent),
            settings=keep,
        )
    return Item(
        name="chipyard riscv-tools (spike, riscv64-unknown-elf-gcc)",
        state=MISSING,
        detail=f"no {missing[0]}",
        fix=[
            "git clone https://github.com/ucb-bar/chipyard.git && cd chipyard",
            "./build-setup.sh riscv-tools          # conda env + prebuilt riscv-tools, spike, cmake, ninja",
            "then:  ./run.sh setup --chipyard /path/to/chipyard",
        ],
        cost="~15 GB and about an hour (build-setup); the chip README says if this chip needs more in it",
    )


def zephyr_item(settings: dict[str, str], *, fetch: bool) -> Item:
    """The Zephyr west workspace: zephyr-chipyard-sw with its zephyr_ws/zephyr submodule and modules."""
    base = settings.get("ZEPHYR_BASE")
    if base and Path(base, "CMakeLists.txt").is_file():
        keep = {k: settings[k] for k in ("ZEPHYR_BASE", "MERLIN_ZEPHYR_SW") if k in settings}
        return Item(name="Zephyr west workspace (ZEPHYR_BASE)", state=OK, detail=base, settings=keep)
    sw = Path(settings.get("MERLIN_ZEPHYR_SW") or _build_dir() / "ext" / "zephyr-chipyard-sw")
    zws = sw / "zephyr_ws"
    west = ["uv", "tool", "run", "west"]
    steps: list[tuple[bool, list, dict]] = [
        (not (sw / ".git").exists(), ["git", "clone", "-b", ZEPHYR_SW_BRANCH, ZEPHYR_SW_URL, sw], {}),
        (
            not (zws / "zephyr" / "west.yml").is_file(),
            ["git", "-C", sw, "submodule", "update", "--init", "zephyr_ws/zephyr"],
            {},
        ),
        (not (zws / ".west").is_dir(), [*west, "init", "-l", "--mf", ZEPHYR_MANIFEST, zws / "zephyr"], {}),
        (not (zws / ".west").is_dir(), [*west, "update"], {"cwd": zws}),
    ]
    todo = [(cmd, kw) for needed, cmd, kw in steps if needed]
    item = Item(
        name="Zephyr west workspace (ZEPHYR_BASE)",
        state=MISSING,
        detail=f"no workspace at {zws}",
        fix=[" ".join(str(c) for c in cmd) + (f"   # in {kw['cwd']}" if kw else "") for cmd, kw in todo],
        cost="~2 GB, 5-10 minutes",
    )
    if not fetch:
        return item
    try:
        for cmd, kw in todo:
            _run(cmd, **kw)
    except (subprocess.CalledProcessError, OSError) as exc:
        item.detail = f"fetching it failed ({exc}); run the commands below by hand"
        return item
    return Item(
        name="Zephyr west workspace (ZEPHYR_BASE)",
        state=BUILT,
        detail=str(zws / "zephyr"),
        settings={"MERLIN_ZEPHYR_SW": str(sw), "ZEPHYR_BASE": str(zws / "zephyr")},
    )


def sdk_item(settings: dict[str, str]) -> Item:
    sdk = settings.get("ZEPHYR_SDK_INSTALL_DIR")
    if sdk and Path(sdk, "riscv64-zephyr-elf").is_dir():
        return Item(
            name=f"Zephyr SDK {ZEPHYR_SDK} (riscv64)", state=OK, detail=sdk, settings={"ZEPHYR_SDK_INSTALL_DIR": sdk}
        )
    tar = f"zephyr-sdk-{ZEPHYR_SDK}_linux-x86_64_minimal.tar.xz"
    return Item(
        name=f"Zephyr SDK {ZEPHYR_SDK} (riscv64)",
        state=MISSING,
        detail=f"no riscv64-zephyr-elf toolchain under {sdk}" if sdk else "ZEPHYR_SDK_INSTALL_DIR is not configured",
        fix=[
            f"wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v{ZEPHYR_SDK}/{tar}",
            f"tar xf {tar} && zephyr-sdk-{ZEPHYR_SDK}/setup.sh -t riscv64-zephyr-elf -h -c",
            f"then:  ./run.sh setup --zephyr-sdk $PWD/zephyr-sdk-{ZEPHYR_SDK}",
        ],
        cost="~1 GB, a few minutes",
    )


# ------------------------------------------------------------------------- model2MLIR capture venv


def _site_packages(py: Path) -> Path:
    out = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         check=True, capture_output=True, text=True)  # fmt: skip
    return Path(out.stdout.strip())


def _lock_name(name: str) -> str:
    """A distribution name as a lock file spells it (PEP 503: lower case, runs of ``-_.`` as one ``-``)."""
    out: list[str] = []
    sep = False
    for ch in name.lower():
        if ch in "-_.":
            sep = True
            continue
        if sep and out:
            out.append("-")
        sep = False
        out.append(ch)
    return "".join(out)


def capture_deps(m2m: Path) -> list[str]:
    """The capture venv's library specs, each exact: :data:`CAPTURE_LOCKED_DEPS` at the version
    ``m2m/uv.lock`` records, else at :data:`CAPTURE_UNLOCKED_PINS` (unpinned only when neither names it)."""
    import tomllib

    locked: dict[str, str] = {}
    lock = m2m / "uv.lock"
    if lock.is_file():
        try:
            packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
        except (OSError, tomllib.TOMLDecodeError):
            packages = []
        locked = {_lock_name(p["name"]): str(p["version"]) for p in packages if "name" in p and "version" in p}
    out = []
    for d in CAPTURE_LOCKED_DEPS:
        version = locked.get(_lock_name(d)) or CAPTURE_UNLOCKED_PINS.get(d)
        out.append(f"{d}=={version}" if version else d)
    return out


def capture_pins(m2m: Path) -> dict[str, str]:
    """Every exact version the capture venv must carry: the torch stack, torch-mlir and the libraries."""
    specs = [*CAPTURE_TORCH, TORCH_MLIR, *capture_deps(m2m)]
    return dict(s.split("==", 1) for s in specs if "==" in s)


#: Run INSIDE the capture venv: installed versions (local tags such as ``+cpu`` dropped), whether this
#: model2MLIR is the one on the path, and whether torch-mlir imports a tiny exported module with this torch.
_CAPTURE_PROBE = """
import importlib.metadata as md, json, sys
names, m2m = json.loads(sys.argv[1]), sys.argv[2]
out = {"versions": {}, "m2m_on_path": False, "smoke": {"ok": False, "detail": ""}}
for n in names:
    try:
        out["versions"][n] = md.version(n).split("+")[0]
    except md.PackageNotFoundError:
        out["versions"][n] = None
try:
    import m2m.capture, torch
    from torch_mlir import fx
    out["m2m_on_path"] = m2m.__path__[0].startswith(sys.argv[2])
    mod = fx.export_and_import(torch.nn.Linear(4, 3), torch.ones(2, 4), output_type="linalg-on-tensors")
    text = str(mod)
    out["smoke"] = {"ok": "linalg" in text, "detail": f"Linear(4,3) -> {len(text)} chars of linalg-on-tensors"}
except Exception as exc:
    out["smoke"]["detail"] = f"{type(exc).__name__}: {exc}"[:600]
print(json.dumps(out))
"""


def capture_probe(py: Path, m2m: Path) -> dict | None:
    """What the capture venv at ``py`` carries and whether it can capture (:data:`_CAPTURE_PROBE`)."""
    if not py.is_file():
        return None
    got = subprocess.run([str(py), "-c", _CAPTURE_PROBE, json.dumps(sorted(capture_pins(m2m))), str(m2m)],
                         capture_output=True, text=True)  # fmt: skip
    try:
        return json.loads(got.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {"versions": {}, "m2m_on_path": False, "smoke": {"ok": False, "detail": got.stderr[-600:]}}


def capture_problems(probe: dict | None, m2m: Path) -> list[str]:
    """Why the probed venv is not the capture venv setup builds (empty when it is): a version off its pin,
    another model2MLIR on its path, or the torch / torch-mlir pair failing to import a module."""
    if probe is None:
        return ["no venv"]
    have = probe.get("versions") or {}
    out = [f"{n} is {have.get(n)}, not {v}" for n, v in sorted(capture_pins(m2m).items()) if have.get(n) != v]
    if not probe.get("m2m_on_path"):
        out.append(f"{m2m} is not the model2MLIR on its path")
    if not (probe.get("smoke") or {}).get("ok"):
        out.append(f"torch-mlir cannot import an exported module: {(probe.get('smoke') or {}).get('detail')}")
    return out


def _git_out(repo: Path, *args: str) -> str | None:
    got = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return got.stdout.strip() if got.returncode == 0 else None


def m2m_movable(m2m: Path, *, owned: bool) -> str:
    """Why setup must NOT move the model2MLIR checkout at ``m2m`` onto the pin; empty when it may.

    Only a clone setup made (``owned``: its own location under out/build/ext) with no uncommitted change to
    a tracked file and no commit that exists nowhere upstream. Anything else may be somebody's work."""
    if not owned:
        return "it is not the clone setup made (it was named by --m2m or MERLIN_M2M_DIR)"
    status = _git_out(m2m, "status", "--porcelain", "--untracked-files=no")
    if status is None:
        return "it is not a readable git checkout"
    if status:
        return f"it has uncommitted changes ({len(status.splitlines())} file(s))"
    if not _git_out(m2m, "branch", "-r", "--contains", "HEAD"):
        return "its HEAD is on no remote branch (local commits)"
    return ""


def advance_m2m(m2m: Path, commit: str, url: str) -> None:
    """Move a clone setup owns to ``commit``: fetch it if absent, then a detached checkout."""
    if _git_out(m2m, "cat-file", "-e", f"{commit}^{{commit}}") is None:
        try:
            _run(["git", "-C", m2m, "fetch", url, commit])
        except subprocess.CalledProcessError:
            _run(["git", "-C", m2m, "fetch", url])  # a server that refuses a commit want
    _run(["git", "-C", m2m, "checkout", "--detach", commit])


def capture_item(settings: dict[str, str], m2m_arg: Path | None, *, fetch: bool) -> Item:
    """model2MLIR at its pin, and the venv captures and lowering run in.

    The checkout is ``--m2m``, else ``MERLIN_M2M_DIR``, else setup's own clone at out/build/ext/model2MLIR.
    ``ok`` only when it is at the pinned commit (captures from another revision are refused by the chip
    flow): setup moves its own clean clone onto the pin, and for any other checkout prints the commands.
    The venv lives under out/build/, not inside the checkout, and model2MLIR is put on its path with a
    .pth file rather than an editable install, so a shared or read-only checkout works unchanged.
    """
    from merlin.common import provenance as PROV

    pins_path = PROV.software_pins_path()
    pin = PROV.load_pins(pins_path)[M2M_PIN]
    own = (_build_dir() / "ext" / "model2MLIR").resolve()
    m2m = (m2m_arg or Path(settings.get("MERLIN_M2M_DIR") or own)).resolve()
    venv = Path(settings.get("MERLIN_M2M_VENV") or _build_dir() / "venv" / "model2mlir")
    py = venv / "bin" / "python"
    name = f"model2MLIR @{pin.commit[:12]} + capture venv"
    clone = [["git", "clone", pin.repo_canonical, m2m], ["git", "-C", m2m, "checkout", "--detach", pin.commit]]
    if not (m2m / "pyproject.toml").is_file():
        if not fetch:
            return Item(name=name, state=MISSING, detail=f"no model2MLIR at {m2m}",
                        fix=[" ".join(map(str, c)) for c in clone], cost="~50 MB")  # fmt: skip
        for cmd in clone:
            _run(cmd)
    ver = PROV.verify(M2M_PIN, checkout=m2m, path=pins_path)
    if not ver.ok:
        why = m2m_movable(m2m, owned=m2m == own)
        if not why and fetch:
            print(f"{m2m} is off the capture pin ({ver.drift[0]}); it is setup's own clean clone: moving it")
            try:
                advance_m2m(m2m, pin.commit, pin.repo_canonical)
            except subprocess.CalledProcessError as exc:
                why = f"moving it failed ({exc})"
            ver = PROV.verify(M2M_PIN, checkout=m2m, path=pins_path)
        elif not why:
            why = "--no-fetch"
        if not ver.ok:
            fix = [
                f"git -C {m2m} fetch {pin.repo_canonical} {pin.commit}",
                f"git -C {m2m} checkout --detach {pin.commit}",
            ]
            if m2m != own:
                fix += ["or let setup use (and move) its own clone: unset MERLIN_M2M_DIR MERLIN_MODEL2MLIR"]
            return Item(
                name=name,
                state=DRIFT,
                detail=f"{m2m} is not the pinned capture revision ({ver.drift[0]}); setup did not move it: "
                f"{why}. Captures refuse it",
                fix=fix,
            )
    keep = dict(zip(CAPTURE_KEYS, (str(m2m), str(m2m), str(venv), str(py))))

    def write_record(probe: dict | None) -> None:
        if not venv.is_dir():
            return
        record = {"pins": capture_pins(m2m), "model2mlir": {"path": str(m2m), "commit": pin.commit}, **(probe or {})}
        (venv / CAPTURE_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    probe = capture_probe(py, m2m)
    if not capture_problems(probe, m2m):
        write_record(probe)
        return Item(name=name, state=OK, detail=f"{venv} (installed versions in {CAPTURE_RECORD})", settings=keep)
    if shutil.which("uv") is None:
        return Item(name=name, state=MISSING, detail="uv is not on PATH",
                    fix=["curl -LsSf https://astral.sh/uv/install.sh | sh"])  # fmt: skip
    env = {**os.environ, "VIRTUAL_ENV": str(venv)}
    if not py.is_file():
        _run(["uv", "venv", "--python", "3.12", venv])
    _run(["uv", "pip", "install", *CAPTURE_TORCH, "--index-url", TORCH_INDEX], env=env)
    _run(["uv", "pip", "install", "--pre", "--no-deps", TORCH_MLIR, "-f", TORCH_MLIR_WHEELS], env=env)
    # The torch stack again, so resolving the libraries can never move it (it is already satisfied).
    _run(["uv", "pip", "install", *capture_deps(m2m), *CAPTURE_TORCH, "--index-url", "https://pypi.org/simple",
          "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match"], env=env)  # fmt: skip
    (_site_packages(py) / "merlin-model2mlir.pth").write_text(str(m2m) + "\n", encoding="utf-8")
    probe = capture_probe(py, m2m)
    problems = capture_problems(probe, m2m)
    freeze = subprocess.run(["uv", "pip", "freeze"], env=env, capture_output=True, text=True).stdout
    (venv / "merlin-capture-freeze.txt").write_text(freeze, encoding="utf-8")
    write_record(probe)
    if problems:
        return Item(name=name, state=MISSING, detail=f"{venv} was built but is not usable: " + "; ".join(problems),
                    fix=[f"{py} -c 'import torch, torch_mlir, torchao, m2m.capture'   # to see why"])  # fmt: skip
    return Item(name=name, state=BUILT, detail=f"{venv} (installed versions in {CAPTURE_RECORD})", settings=keep)


# ------------------------------------------------------------------------------- support package


def support_item(target: str, *, fetch: bool) -> Item:
    """``target``'s support package, at exactly the commit Merlin pins (and its base, at the base pin).

    Already fetched and verified: nothing to do, no network. Absent: fetched by commit into
    out/build/generated/pinned/<target>@<sha12>/ (``merlin-target-fetch``), unless ``--no-fetch``. A
    checkout that disagrees with its pin (dirty, divergent, or composed against another base) is
    reported with the reason and never moved. With ``MERLIN_TARGET_PATH`` set, that selection is used
    as is; this only reports it.
    """
    from merlin.targetgen import oot_fetch
    from merlin.targetgen import support_pins as SP

    name = f"{target} support package"
    try:
        pins = SP.load_pins()
        pin = pins.get(target)
        if pin is None:
            return Item(name=name, state=MISSING, detail=f"{target!r} is not pinned in {SP.MANIFEST}")
        name = f"{target} support ({pin.repo_name}@{pin.short})"
        if os.environ.get("MERLIN_TARGET_PATH") is not None:
            from merlin.targetgen.plugins import resolve_support

            root = resolve_support(target).base
            return Item(name=name, state=WARN, detail=f"MERLIN_TARGET_PATH selects {root}",
                        fix=["unset MERLIN_TARGET_PATH to use the pinned package"])  # fmt: skip
        present = SP.available_root(pin)
        if present is not None:
            # Verified again (commit, tree, clean, base pin, submodules); no network when it holds.
            oot_fetch.fetch_pinned(target, pins=pins)
            return Item(name=name, state=OK, detail=str(present))
        command = f"{sys.executable} -m merlin.targetgen.oot_fetch {target}"
        if not fetch:
            return Item(name=name, state=MISSING, detail="not fetched", fix=[command], cost="a few MB, seconds")
        print(f"$ {command}", flush=True)
        roots = oot_fetch.fetch_pinned(target, pins=pins)
        return Item(name=name, state=BUILT, detail=str(roots[0]))
    except Exception as exc:  # noqa: BLE001 -- every refusal is reported with its reason, never hidden
        return Item(name=name, state=DRIFT, detail=f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------------------- unit spike model


def spike_model_item(target: str, settings: dict[str, str], chipyard: Item) -> Item:
    """Build ``target``'s functional model of its unit for spike (the package's ``spike/build.py``) into
    out/build/spike-ext/<target>/ and record it under the variable the package declares. Skipped when the
    declared model already resolves (its manifest describes these exact inputs and passed its self-test)."""
    from merlin.targetgen import spike_extension as SE
    from merlin.targetgen.target_registry import resolve

    block = SE.declared_extension(target)
    if not block:
        return Item(name=f"{target} spike model", state=OK, detail="the package declares none")
    env_name = str(block.get("extlib_env"))
    name = f"{target} spike model (${env_name})"
    out = _build_dir() / "spike-ext" / target
    so = out / f"lib{block.get('extension_name')}_model.so"

    def resolves() -> str | None:
        with _env_set({env_name: str(so)}):
            try:
                SE.resolve(target, default_library_dir=out, default_extension_name="")
                return None
            except SE.SpikeExtensionError as exc:
                return str(exc)

    if so.is_file() and resolves() is None:
        return Item(name=name, state=OK, detail=str(so), settings={env_name: str(so)})
    if not chipyard.done:
        return Item(name=name, state=BLOCKED, detail="needs chipyard's spike and headers first (above)")
    build = resolve(target).base / "spike" / "build.py"
    if not build.is_file():
        return Item(name=name, state=MISSING, detail=f"{build.parent} is missing from the selected package")
    env = {**os.environ, **{k: v for k, v in settings.items() if k.startswith(("MERLIN_", "ZEPHYR_"))}}
    try:
        _run([sys.executable, build, "--out", out], env=env)
    except subprocess.CalledProcessError:
        return Item(name=name, state=MISSING, detail="spike/build.py failed (output above)",
                    fix=[f"{sys.executable} {build} --out {out}"])  # fmt: skip
    why = resolves()
    if why:
        return Item(name=name, state=DRIFT, detail=f"built, but not the model the package declares: {why}")
    return Item(name=name, state=BUILT, detail=str(so), settings={env_name: str(so)})


# ----------------------------------------------------------------------------------------- driver


def adopt(current: dict[str, str], args) -> dict[str, str]:
    """Settings from .env, overridden by the process environment, overridden by flags. A ``/path/to/...``
    value (.env.example's placeholder, in a .env copied from it) is not a setting."""
    settings = {k: v for k, v in current.items() if v and not v.startswith(PLACEHOLDER)}
    exported = shell_exports(os.environ)
    settings |= {k: exported[k] for k in ADOPTED if exported.get(k)}
    if args.llvm:
        prefix = args.llvm.resolve()
        settings |= {
            "MERLIN_MLIR_INSTALL": str(prefix),
            "MERLIN_CLANG": str(prefix / "bin/clang-23"),
            "MERLIN_MLIR_TRANSLATE": str(prefix / "bin/mlir-translate"),
        }
    if args.chipyard:
        settings["MERLIN_CHIPYARD"] = str(args.chipyard.resolve())
    if args.zephyr_sw:
        sw = args.zephyr_sw.resolve()
        settings |= {"MERLIN_ZEPHYR_SW": str(sw), "ZEPHYR_BASE": str(sw / "zephyr_ws" / "zephyr")}
    if args.zephyr_sdk:
        settings["ZEPHYR_SDK_INSTALL_DIR"] = str(args.zephyr_sdk.resolve())
    return settings


def _recorded(items: list[Item]) -> dict[str, str]:
    return {k: v for i in items if i.done for k, v in i.settings.items()}


def unit_pin_items(
    explicit: list[str], supports: dict[str, Item], checkouts: dict[str, Item], settings: dict[str, str]
) -> list[Item]:
    """One :func:`verify_pin_item` per pin a fetched support package's matrix units read (with the files
    they read), plus each ``--verify-pin``. A pin whose checkout ``--checkout-pin`` already verified, or
    one nested in such a checkout, is covered there and not repeated."""
    from merlin.targetgen import unit_pins

    reads: dict[str, tuple[str, ...]] = {p: () for p in explicit}
    failed: list[Item] = []
    with _env_set({k: v for k, v in settings.items() if k.startswith(("MERLIN_", "ZEPHYR_"))}):
        for target, item in supports.items():
            if not item.done:
                continue
            try:
                found = unit_pins.unit_pins(target)
            except Exception as exc:  # noqa: BLE001 -- reported, never hidden
                why = f"{type(exc).__name__}: {exc}"
                failed.append(Item(name=f"{target} matrix unit pins", state=DRIFT, detail=why))
                continue
            for up in found:
                reads[up.pin] = tuple(dict.fromkeys([*reads.get(up.pin, ()), *up.reads]))
    pins = _pins()
    covered = set(checkouts) | {p for p in reads if p in pins and pins[p].nested_in in checkouts}
    return failed + [verify_pin_item(p, settings, reads=r) for p, r in reads.items() if p not in covered]


def report(items: list[Item]) -> int:
    width = max(len(i.name) for i in items)
    print("\nSetup:")
    for i in items:
        print(f"  [{i.state:7}] {i.name:<{width}}  {i.detail}")
    for i in (i for i in items if i.state == WARN):
        print(f"\nwarning: {i.name}")
        for line in i.fix:
            print(f"    {line}")
    todo = [i for i in items if not i.done]
    if not todo:
        print("\nEverything the chip example needs is configured. Next: ./run.sh preflight")
        return 0
    print("\nStill needed (run setup again once done; it picks up what you add):")
    for n, i in enumerate((i for i in todo if i.fix), 1):
        print(f"\n{n}. {i.name}" + (f"  [{i.cost}]" if i.cost else ""))
        for line in i.fix:
            print(f"    {line}")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llvm", type=Path, help="an LLVM/MLIR 23 install prefix (has bin/clang-23, bin/mlir-translate)")
    ap.add_argument("--chipyard", type=Path, help="chipyard checkout with riscv-tools in its .conda-env")
    ap.add_argument("--zephyr-sw", type=Path, help="zephyr-chipyard-sw checkout (dev); its west workspace is set up")
    ap.add_argument("--zephyr-sdk", type=Path, help=f"Zephyr SDK {ZEPHYR_SDK} install")
    ap.add_argument(
        "--m2m", type=Path, help="model2MLIR checkout (default: $MERLIN_M2M_DIR, else fetched to out/build/ext)"
    )
    ap.add_argument("--support", action="append", default=[], metavar="TARGET",
                    help="fetch TARGET's support package at the commit Merlin pins (and its base)")  # fmt: skip
    ap.add_argument("--verify-pin", action="append", default=[], metavar="PIN",
                    help="verify this pin's checkout inside a configured prerequisite; report drift")  # fmt: skip
    ap.add_argument("--checkout-pin", action="append", default=[], metavar="PIN",
                    help="find PIN's checkout ($root_env, else out/build/ext/PIN) and verify it")  # fmt: skip
    ap.add_argument("--build-spike", action="append", default=[], metavar="PIN=ENV",
                    help="build spike from PIN's verified checkout into out/build/PIN; record it as ENV")  # fmt: skip
    ap.add_argument("--spike-model", action="append", default=[], metavar="TARGET",
                    help="build TARGET's declared spike model of its unit into out/build/spike-ext/TARGET")  # fmt: skip
    ap.add_argument("--no-fetch", action="store_true", help="print the fetch commands instead of running them")
    ap.add_argument("--skip-capture-env", action="store_true", help="do not check or build the capture venv")
    ap.add_argument("--env-file", type=Path, default=REPO / ".env", help="where settings are written")
    args = ap.parse_args(argv)

    current = _dotenv(args.env_file)
    settings = adopt(current, args)
    fetch = not args.no_fetch
    from merlin.common import provenance as PROV

    items = [llvm_item(settings), chipyard := chipyard_item(settings)]
    items += [zephyr_item(settings, fetch=fetch), sdk_item(settings)]
    if not args.skip_capture_env:
        items.append(capture_item(settings, args.m2m, fetch=fetch))
    supports = {t: support_item(t, fetch=fetch) for t in args.support}
    items += supports.values()
    for t in args.spike_model:
        if t in supports and not supports[t].done:
            items.append(Item(name=f"{t} spike model", state=BLOCKED, detail="needs its support package (above)"))
        else:
            items.append(spike_model_item(t, settings, chipyard))
    checkouts: dict[str, Item] = {}
    for name in args.checkout_pin:
        pin = PROV.pin(name)
        configured = shell_exports(os.environ).get(pin.root_env) or settings.get(pin.root_env)
        checkouts[name] = checkout_item(
            name,
            Path(configured) if configured else None,
            default=_build_dir() / "ext" / name,
        )
        items.append(checkouts[name])
    for spec in args.build_spike:
        pin_name, sep, env_name = spec.partition("=")
        if not sep or pin_name not in checkouts:
            raise SystemExit(f"--build-spike {spec}: expected PIN=ENV with PIN also given to --checkout-pin")
        items.append(build_spike(pin_name, env_name, checkouts[pin_name], prefix=_build_dir() / pin_name))
    items += unit_pin_items(args.verify_pin, supports, checkouts, settings | _recorded(items))

    record = {k: v for i in items if i.done for k, v in i.settings.items()}
    if record:
        _upsert_env(args.env_file, record)
    status = report(items)
    shadowed = shadowed_settings(record, os.environ)
    if shadowed:
        # Setup checked the value it recorded; every later run.sh and merlin command reads the exported one.
        print("\nwarning: this shell exports a different value for what setup just recorded, and the exported")
        print("value wins over .env in every later command, so they will not use what was verified above:")
        for key, exported, recorded in shadowed:
            print(f"    {key}={exported}   (recorded: {recorded})")
        print("  unset them (or export the recorded values), e.g.:  unset " + " ".join(k for k, _, _ in shadowed))
    return status


def shadowed_settings(record: dict[str, str], environ) -> list[tuple[str, str, str]]:
    """``(key, exported, recorded)`` for each recorded setting the process environment overrides with a
    different value. ``run.sh`` and ``merlin.common.paths.env`` both let the environment win over .env, so a
    stale export (a workspace that has since moved) silently replaces the path setup just verified. A value
    ``run.sh`` itself loaded from .env (listed in ``MERLIN_DOTENV_KEYS``) is not an export, and is reloaded."""
    environ = shell_exports(environ)
    return [(k, environ[k], v) for k, v in sorted(record.items()) if environ.get(k) and environ[k] != v]


def shell_exports(environ) -> dict[str, str]:
    """The process environment without what ``run.sh`` loaded from .env (``MERLIN_DOTENV_KEYS``)."""
    loaded = set((environ.get("MERLIN_DOTENV_KEYS") or "").split())
    return {k: v for k, v in environ.items() if k not in loaded}


if __name__ == "__main__":
    raise SystemExit(main())
