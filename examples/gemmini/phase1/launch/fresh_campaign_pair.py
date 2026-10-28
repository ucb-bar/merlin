"""Prepare two matched campaign plans under one existing CPU reservation owner.

Preparation checks bytes and writes scripts; only ``launch --confirm`` executes
the existing resource_reservation wrapper. No compiler or evaluator is added.
The contrast remains CCA off/on; both origins may be fresh or matched reviewed seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FILE_FLAGS = (
    "--descriptor",
    "--bundle-manifest",
    "--corpus-seal",
    "--oracle-timing",
    "--private-full-model-spec",
    "--instruction-selection",
    "--compiler-library-contract",
    "--operator-errata",
    "--joint-campaign",
    "--full-compiler-manifest",
)
NAMESPACE_KEYS = ("MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GSIM_SLOT_ROOT", "MERLIN_BUNDLE_CAS", "TMPDIR")
GUIDANCE_FLAGS = ("--reference-experiments", "--reference-experiments-sha256")
# Guided-arm only: keep the selection's analysis grant while withholding its prior tasks.
TASK_FLAG = "--reference-experiments-tasks"
REFERENCE_MAP_ENV = "MERLIN_REFERENCE_MAP"


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _hash(path)}


def _reopen(pin):
    if _pin(pin["path"]) != pin:
        raise ValueError(f"selected file changed: {pin['path']}")
    return json.loads(Path(pin["path"]).read_text())


def _member_pins(root, members, *, field):
    pins = []
    root = Path(root).resolve()
    for member in members:
        path = (root / member[field]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("selected input member escapes its root")
        pin = _pin(path)
        if pin["sha256"] != member["sha256"]:
            raise ValueError(f"selected input member changed: {path}")
        pins.append(pin)
    return pins


def _module(name):
    spec = importlib.util.spec_from_file_location(f"paired_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _campaign():
    return _module("fresh_campaign")


def _arguments(argv):
    prefix, options, cursor = [], {}, 0
    while cursor < len(argv):
        token = argv[cursor]
        if not token.startswith("--"):
            prefix.append(token)
            cursor += 1
            continue
        if token in options:
            raise ValueError(f"duplicate campaign option: {token}")
        value = True
        if cursor + 1 < len(argv) and not argv[cursor + 1].startswith("--"):
            value = argv[cursor + 1]
            cursor += 1
        options[token] = value
        cursor += 1
    return prefix, options


def _runtime(plan):
    wrapper, inner = plan["reserved_argv"], plan["reserved_argv"].index("--") + 1
    command, argv = wrapper[inner:], plan["argv"]
    if command[:2] != ["/usr/bin/env", "-i"] or command[-len(argv) :] != argv:
        raise ValueError("reservation command differs from recorded campaign argv")
    env = {}
    for item in command[2 : -len(argv)]:
        key, separator, value = item.partition("=")
        if not separator or key in env:
            raise ValueError("reservation requires unique exact environment assignments")
        env[key] = value
    if any(env.get(key) != value for key, value in plan["environment"].items()):
        raise ValueError("reservation environment differs from recorded environment")
    return env


def _arm(plan):
    _, options = _arguments(plan["argv"])
    env = _runtime(plan)
    cpus = tuple(int(value) for value in env["MERLIN_COMPUTE_CPUS"].split(","))
    if not cpus or min(cpus) < 0 or len(set(cpus)) != len(cpus):
        raise ValueError("distinct nonnegative CPU IDs required")
    if sorted(cpus) != sorted(plan["cpu_reservation"]["cpus"]):
        raise ValueError("recorded reservation differs from runtime CPU roster")
    for key in ("MERLIN_CODEX_CPUS", "MERLIN_PREPARATION_CPUS"):
        if env[key] != env["MERLIN_COMPUTE_CPUS"]:
            raise ValueError(f"{key} differs from the arm's compute roster")
    if int(options["--sim-max-jobs"]) > len(cpus):
        raise ValueError("simulator concurrency exceeds the arm's CPU allocation")
    for key in _campaign().WORKER_KEYS:
        if int(env[key]) > len(cpus):
            raise ValueError(f"{key} exceeds the arm's CPU allocation")
    paths = {key: Path(env[key]).resolve() for key in NAMESPACE_KEYS}
    if any(not Path(env[key]).is_absolute() for key in NAMESPACE_KEYS):
        raise ValueError("paired writable namespaces must be absolute")
    paths["PRIVATE_ROOT"] = Path(options["--codex-home-root"]).resolve()
    paths["ARTIFACTS"] = Path(options["--operator-errata"]).resolve().parent
    roots = {
        "RUN_STATE": paths["MERLIN_COMPUTE_CPU_SLOTS"].parent,
        "PRIVATE_ROOT": paths["PRIVATE_ROOT"].parent,
        "ARTIFACTS": paths["ARTIFACTS"],
        "TMPDIR": paths["TMPDIR"],
    }
    replacements = [(str(path), "{" + key + "}") for key, path in roots.items()]
    replacements += [(plan["run_id"], "{RUN_ID}"), (env["MERLIN_COMPUTE_CPUS"], "{CPUS}")]
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    def normalized(value):
        if isinstance(value, str):
            for source, target in replacements:
                value = value.replace(source, target)
        return value

    return options, env, cpus, paths, normalized


def _normalized(value, normalize):
    if isinstance(value, dict):
        return {key: _normalized(item, normalize) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item, normalize) for item in value]
    return normalize(value)


def _seeded_identity(plan, arm):
    """Reopen the existing reviewed-seed boundary; compare content, not arm-local paths."""
    from merlin_experiments.phase1 import joint_seed

    options, env, _, _, normalize = arm
    seed, origin = plan["seed"], plan.get("origin")
    if ("--fresh-backend" in options or "--resume" in options or "--seed-submission" not in options
            or options.get("--sandbox") != "bwrap" or "--allow-unsandboxed" in options):
        raise ValueError("paired seeded runs require isolated reviewed seed origins")
    policy = json.loads(Path(options["--joint-campaign"]).read_text())
    selected = joint_seed.selected(SimpleNamespace(
        joint_campaign=options["--joint-campaign"], seed_submission=options["--seed-submission"],
        sandbox=options["--sandbox"], fresh_backend=False, resume=False,
        joint_epoch=options.get("--joint-epoch", ""),
        qualify_submission=options.get("--qualify-submission", ""),
    ), policy)
    if (not isinstance(selected, dict) or selected != origin or seed.get("source") != selected["source"]
            or seed.get("graded_source_sha256") != selected["sha256"]):
        raise ValueError("paired seed source differs from its recorded origin")
    source = seed["from"]
    if _module("fresh_seed").verify_source(Path(source["path"]), source["sha256"]) != seed["from_digest"]:
        raise ValueError("paired seed digest basis differs from its recorded origin")
    carry = seed["carry"]
    if (set(carry) != {"certificates", "results", "rules", "handoff"}
            or any(type(value) is not bool for value in carry.values())):
        raise ValueError("paired seed carry policy requires explicit booleans")
    carries = carry["certificates"] or carry["results"]
    if (carries != (selected.get(joint_seed.CARRY) is True)
            or carries != (env.get("MERLIN_TIER_CERT_PORTABLE") == "1")
            or carries and env.get("MERLIN_SEEDED_GRADE") != (seed.get("run") or {}).get("run_id")
            or carry["handoff"] != (selected.get(joint_seed.PREVIOUS) is not None)):
        raise ValueError("paired seed carry policy differs from its runtime approval")
    evidence = (seed.get("evidence") or {}).get("pin")
    if evidence != selected.get(joint_seed.EVIDENCE):
        raise ValueError("paired seed evidence differs from its recorded origin")
    review = _reopen(selected["review"])
    if review.get("derived_from") != {
        "path": source["path"], "sha256": source["sha256"], "digest": seed["from_digest"],
        "stripped": seed["stripped"], "run": (seed.get("run") or {}).get("run_id"),
    }:
        raise ValueError("paired seed derivation differs from its operator review")
    # Each arm has its own reviewed copy. Its review's SHA differs because it contains
    # arm-local paths; the existing seed reader has reopened and checked those bytes.
    policy["compiler_origin"] = {**selected, "review": review}
    certificates = seed.get("certificates") or {}
    identity = {key: seed.get(key) for key in (
        "from", "from_digest", "stripped", "graded_source_sha256", "carry", "run", "guidance_addendum",
    )}
    identity.update(policy=policy, evidence_sha256=(evidence or {}).get("sha256"),
                    judge=certificates.get("judge"))
    pins = [selected["review"]]
    if seed.get("guidance_addendum"):
        actual = _pin(seed["guidance_addendum"]["path"])
        if actual != seed["guidance_addendum"]:
            raise ValueError("paired seed guidance changed")
        pins.append(actual)
    return _normalized(identity, normalize), pins


def _validate(raw, cca):
    if raw.get("reference_guidance") != "off" or cca.get("reference_guidance") != "on":
        raise ValueError("paired plans require explicit guidance off and on respectively")
    if raw["run_id"] == cca["run_id"] or any(p.get("execution_started") for p in (raw, cca)):
        raise ValueError("pair requires two distinct, unlaunched run identities")
    shared = (
        "cwd",
        "source_revision",
        "framework_repository",
        "source_release",
        "full_compiler_manifest_sha256",
        "treatment",
        "reference_manifest",
        "inputs_selection",
    )
    for key in shared:
        if key not in raw or raw[key] != cca.get(key):
            raise ValueError(f"paired inputs differ: {key}")
    raw_arm, cca_arm = _arm(raw), _arm(cca)
    seeded = [plan.get("seed") is not None for plan in (raw, cca)]
    if seeded[0] != seeded[1]:
        raise ValueError("paired runs cannot mix fresh and seeded origins")
    seed_pins = []
    if seeded[0]:
        identities = []
        for plan, arm in ((raw, raw_arm), (cca, cca_arm)):
            identity, selected = _seeded_identity(plan, arm)
            identities.append(identity)
            seed_pins += selected
        if identities[0] != identities[1]:
            raise ValueError("paired seed source, carry policy, evidence or effective judge differ")
    left, right = raw_arm[2], cca_arm[2]
    if len(left) != len(right) or set(left) & set(right):
        raise ValueError("paired CPU rosters must be equally sized and disjoint")
    for a in raw_arm[3].values():
        for b in cca_arm[3].values():
            if a == b or a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("paired writable namespaces overlap")
    pins = seed_pins
    for key in ("reference_manifest", "inputs_selection"):
        _reopen(raw[key])
        pins.append(raw[key])
    references = _reopen(raw["reference_manifest"])
    if references.get("schema") != "operator.allowed_reference_sources.v2":
        raise ValueError("paired fresh runs require the reviewed reference origin manifest")
    staged = Path(raw["inputs_selection"]["path"]).parent / "public-references"
    for origin in references["origins"]:
        if not (staged / origin["name"]).resolve().is_relative_to(staged.resolve()):
            raise ValueError("selected reference origin escapes its staged root")
        pins += _member_pins(staged / origin["name"], origin["files"], field="destination")
    for plan, arm in ((raw, raw_arm), (cca, cca_arm)):
        options, env, _, _, normalize = arm
        mode = plan["reference_guidance"]
        if not seeded[0] and (
            "--fresh-backend" not in options or any(flag in options for flag in ("--seed-submission", "--resume"))
        ):
            raise ValueError("paired fresh runs require empty backend origins")
        if mode == "off" and plan.get("reference_experiments"):
            raise ValueError("guidance off cannot retain an enabled selection receipt")
        if any(flag in options for flag in GUIDANCE_FLAGS) != (mode == "on") or (
            mode == "on" and not all(flag in options for flag in GUIDANCE_FLAGS)
        ):
            raise ValueError("guidance options differ from the selected mode")
        if options.get(TASK_FLAG, "on") != plan.get("reference_experiment_tasks", "on") or (
            mode == "off" and TASK_FLAG in options
        ):
            raise ValueError("reference task serving differs from its plan receipt")
        if mode == "on":
            selected = {"path": options[GUIDANCE_FLAGS[0]], "sha256": options[GUIDANCE_FLAGS[1]]}
            if selected != plan.get("reference_experiments"):
                raise ValueError("guidance selection differs from its plan receipt")
            _reopen(selected)
            pins.append(selected)
        for flag in FILE_FLAGS:
            if flag in options:
                pins.append(_pin(options[flag]))
        manifest = _pin(options["--full-compiler-manifest"])
        if manifest["sha256"] != plan["full_compiler_manifest_sha256"]:
            raise ValueError("full compiler manifest differs from the selected identity")
        pins += _member_pins(options["--full-compiler-root"], _reopen(manifest)["members"], field="path")
        contract = _pin(options["--compiler-library-contract"])
        pins += _member_pins(options["--compiler-library-root"], _reopen(contract)["members"], field="path")
        pins += [_pin(plan["reserved_argv"][1]), _pin(plan["argv"][0])]
        policy = json.loads(Path(options["--joint-campaign"]).read_text())
        for key in ("gsim", "gsim_correctness"):
            if int(policy.get(key, {}).get("max_jobs", 1)) > len(left):
                raise ValueError("policy concurrency exceeds the arm's CPU allocation")
        for value in env.values():
            if isinstance(value, str) and value.startswith("{"):
                pin = json.loads(value)
                if isinstance(pin, dict) and set(pin) == {"path", "sha256"}:
                    _reopen(pin)
                    pins.append(pin)
    maps = [
        json.loads(arm[1][REFERENCE_MAP_ENV]) if REFERENCE_MAP_ENV in arm[1] else None for arm in (raw_arm, cca_arm)
    ]
    if any(pin != plan.get("reference_map") for pin, plan in zip(maps, (raw, cca), strict=True)):
        raise ValueError("reference map differs from its plan receipt")
    # The guided arm may additionally select its analysis map. Legacy studies
    # supplying a map to both arms must still use identical map bytes.
    if maps[0] is not None and maps[0].get("sha256") != (maps[1] or {}).get("sha256"):
        raise ValueError("paired reference maps differ: a raw-arm map requires the same guided-arm bytes")
    prefixes = [_arguments(p["argv"])[0] for p in (raw, cca)]
    if prefixes[0] != prefixes[1]:
        raise ValueError("paired controller commands differ")
    for index in (0, 1):
        a, b = raw_arm[index], cca_arm[index]
        keys = set(a) | set(b)
        for key in keys - {*GUIDANCE_FLAGS, TASK_FLAG, REFERENCE_MAP_ENV}:
            if raw_arm[4](a.get(key)) != cca_arm[4](b.get(key)):
                raise ValueError(f"paired settings differ: {key}")
    for flag in FILE_FLAGS:
        if seeded[0] and flag == "--joint-campaign":
            continue  # Compared above after reopening both arm-local reviewed seed origins.
        if flag in raw_arm[0] and _hash(raw_arm[0][flag]) != _hash(cca_arm[0][flag]):
            raise ValueError(f"paired file bytes differ: {flag}")
    if raw["reserved_argv"][:2] != cca["reserved_argv"][:2]:
        raise ValueError("paired reservation implementations differ")
    raw_seconds = raw["reserved_argv"][raw["reserved_argv"].index("--max-seconds") + 1]
    cca_seconds = cca["reserved_argv"][cca["reserved_argv"].index("--max-seconds") + 1]
    if raw_seconds != cca_seconds:
        raise ValueError("paired reservation clocks differ")
    return sorted(set(left) | set(right)), pins


def _supervisor(scripts, logs):
    lines = [
        "#!/bin/bash",
        "set -uo pipefail",
        "pids=()",
        "cleanup() {",
        "  trap '' TERM INT",
        '  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done',
        '  /usr/bin/setsid /bin/bash -c \'sleep 10; for pid in "$@"; do '
        'kill -KILL -- "-$pid" 2>/dev/null || true; done\' pair-watchdog "${pids[@]}" >/dev/null 2>&1 &',
        "  watchdog=$!",
        '  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done',
        "  for (( attempt=0; attempt<100; attempt++ )); do",
        '    kill -0 -- "-$watchdog" 2>/dev/null && break',
        '    kill -0 "$watchdog" 2>/dev/null || break',
        "    sleep 0.01",
        "  done",
        '  kill -KILL -- "-$watchdog" 2>/dev/null || true',
        '  kill -KILL "$watchdog" 2>/dev/null || true',
        '  wait "$watchdog" 2>/dev/null || true',
        "}",
        "trap cleanup EXIT",
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
    ]
    for script, log in zip(scripts, logs, strict=True):
        lines += [
            f"/usr/bin/setsid /bin/bash {shlex.quote(str(script))} >>{shlex.quote(str(log))} 2>&1 &",
            'pids+=("$!")',
        ]
    lines += [
        "first_status=0",
        'wait -n -p finished "${pids[@]}" || first_status=$?',
        'printf \'first_finished=%s status=%s\\n\' "$finished" "$first_status"',
        'if (( first_status != 0 )); then exit "$first_status"; fi',
        'for pid in "${pids[@]}"; do',
        '  if [[ "$pid" != "$finished" ]]; then',
        '    wait "$pid"; status=$?',
        '    printf \'second_finished=%s status=%s\\n\' "$pid" "$status"',
        '    exit "$status"',
        "  fi",
        "done",
    ]
    return "\n".join(lines) + "\n"


def prepare_pair(raw_plan: Path, cca_plan: Path, *, output: Path, unit: str) -> dict:
    """Freeze matched input/settings checks and one supervised outer wrapper argv."""
    plans = [json.loads(Path(path).read_text()) for path in (raw_plan, cca_plan)]
    cpus, pins = _validate(*plans)
    if not unit.startswith("merlinphase1") or not unit.isalnum() or unit.lower() != unit:
        raise ValueError("outer unit must be a unique merlinphase1 lowercase alphanumeric name")
    campaign, output = _campaign(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scripts, logs = [], []
    for label, plan in zip(("raw", "cca"), plans, strict=True):
        script = output / f"run-{label}.sh"
        script.write_text(campaign.preflight_script(plan, run=True))
        script.chmod(0o700)
        scripts.append(script)
        logs.append(Path(plan["argv"][plan["argv"].index("--operator-errata") + 1]).parent / "campaign.log")
    supervisor = output / "run-pair.sh"
    supervisor.write_text(_supervisor(scripts, logs))
    supervisor.chmod(0o700)
    outer = plans[0]["reserved_argv"][: plans[0]["reserved_argv"].index("--") + 1] + ["/bin/bash", str(supervisor)]
    for flag, value in (
        ("--cpus", ",".join(map(str, cpus))),
        ("--unit", unit),
        ("--state", str(output / "cpu-reservation.json")),
        ("--log", str(output / "pair-supervisor.log")),
    ):
        campaign.set_wrapper_option(outer, flag, value)
    pins += [
        _pin(path) for path in (raw_plan, cca_plan, *scripts, supervisor, HERE / "fresh_campaign.py", Path(__file__))
    ]
    record = {
        "schema": "operator.fresh_campaign_pair.v1",
        "execution_started": False,
        "plans": [_pin(path) for path in (raw_plan, cca_plan)],
        "cpus": cpus,
        "unit": unit,
        "artifact_pins": list({pin["path"]: pin for pin in pins}.values()),
        "arms": [
            {
                "run_id": plan["run_id"],
                "reference_guidance": plan["reference_guidance"],
                "cpus": plan["cpu_reservation"]["cpus"],
                "script": _pin(script),
                "log": str(log),
            }
            for plan, script, log in zip(plans, scripts, logs, strict=True)
        ],
        "outer_argv": outer,
        "supervision": "wait both; on failure or signal terminate both process groups and reap",
        "scope": "matched preparation and CPU isolation; author randomness and shared memory/disk remain uncontrolled",
    }
    (output / "pair.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for key in ("raw-plan", "cca-plan", "output"):
        prepare.add_argument("--" + key, type=Path, required=True)
    prepare.add_argument("--unit", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--pair", type=Path, required=True)
    launch.add_argument("--sha256", required=True)
    launch.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        record = prepare_pair(args.raw_plan, args.cca_plan, output=args.output, unit=args.unit)
        print(json.dumps({"pair": _pin(args.output / "pair.json"), "outer_argv": record["outer_argv"]}, indent=2))
        return 0
    record = _reopen({"path": str(args.pair.resolve()), "sha256": args.sha256})
    for pin in record["artifact_pins"]:
        if _pin(pin["path"]) != pin:
            raise ValueError(f"paired launch input changed: {pin['path']}")
    plans = [_reopen(pin) for pin in record["plans"]]
    _validate(*plans)
    if not args.confirm:
        print("refusing to launch without --confirm")
        return 2
    if Path(record["outer_argv"][record["outer_argv"].index("--state") + 1]).exists():
        raise ValueError("paired reservation state already exists; choose a new pair")
    watchdog = _module("run_watchdog")
    log = Path(record["outer_argv"][record["outer_argv"].index("--state") + 1]).with_name("watchdog.log")
    watchdog.start_detached(record["unit"], set(record["cpus"]), log)
    return subprocess.run(record["outer_argv"], check=False, cwd=plans[0]["cwd"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
