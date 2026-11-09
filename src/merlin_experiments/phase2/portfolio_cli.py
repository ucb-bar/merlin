"""Installed portfolio launch over explicitly declared source and sandbox ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath

import yaml

from merlin.common.paths import module_source_path
from merlin.common.source_membership import python_members
from merlin.targetgen.providers import ProviderRole, read_provider
from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments import frozen_python as FP
from merlin_experiments import source_snapshot as SNAP
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import portfolio_launch as LAUNCH
from merlin_experiments.phase2 import portfolio_worker as WORKER
from merlin_experiments.phase2 import qualification_policy as Q
from merlin_experiments.phase2.portfolio_options import parse_invocation

MODULE = "merlin_experiments.phase2.portfolio_cli"
SCHEMA = "merlin.portfolio-deployment.v1"
_FIELDS = {
    "schema",
    "target",
    "source_root",
    "output_root",
    "lease_path",
    "source_roots",
    "python_roots",
    "legacy_roots",
    "internal_aliases",
    "exclude_paths",
    "provider_root",
    "functional_runs_root",
    "contract_root",
    "compiler_shared_source_root",
    "sandbox_root",
    "sandbox_declaration",
    "sandbox_declaration_sha256",
}


def _ordinary_bytes(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("portfolio specification must be an ordinary JSON file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("portfolio specification must be an ordinary JSON file")
        return stream.read()


def _json_bytes(path: Path) -> tuple[dict, bytes]:
    raw = _ordinary_bytes(path)
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("portfolio specification must contain an object")
    return document, raw


def _json(path: Path) -> dict:
    return _json_bytes(path)[0]


def _absolute(value, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"portfolio deployment {field} requires a canonical absolute path")
    path = Path(value)
    # Shape-only decoding permits historical original roots to be absent. Live
    # containment/origins are checked separately before creating a new snapshot.
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"portfolio deployment {field} requires a canonical absolute path")
    return path


def _relative(value, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"portfolio deployment {field} requires a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError(f"portfolio deployment {field} requires a safe relative path")
    return value


def load_deployment(path: Path) -> dict:
    """Decode exact deployment fields without discovering or accessing original roots."""
    return _decode_deployment(_json(path))


def _decode_deployment(document: dict) -> dict:
    if set(document) != _FIELDS or document.get("schema") != SCHEMA:
        raise ValueError("unsupported portfolio deployment schema or fields")
    if not isinstance(document["target"], str) or not document["target"].strip():
        raise ValueError("portfolio deployment requires an explicit target")
    for field in (
        "source_root",
        "output_root",
        "lease_path",
        "functional_runs_root",
        "sandbox_root",
        "sandbox_declaration",
    ):
        _absolute(document[field], field=field)
    if document["provider_root"] is not None:
        _absolute(document["provider_root"], field="provider_root")
    for field in ("contract_root", "compiler_shared_source_root"):
        _relative(document[field], field=field)
    for field in ("source_roots", "python_roots", "legacy_roots", "exclude_paths"):
        values = document[field]
        if not isinstance(values, list) or (not values and field in {"source_roots", "python_roots"}):
            raise ValueError(f"portfolio deployment {field} must be an explicit path list")
        for value in values:
            _relative(value, field=field)
        if len(values) != len(set(values)):
            raise ValueError(f"portfolio deployment {field} has duplicate roots")
    aliases = document["internal_aliases"]
    if not isinstance(aliases, dict):
        raise ValueError("portfolio deployment internal_aliases must be an object")
    for alias, target in aliases.items():
        _relative(alias, field="internal_aliases")
        _relative(target, field="internal_aliases")
    digest = document["sandbox_declaration_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("portfolio deployment requires an exact sandbox declaration SHA-256")
    for value in (*document["legacy_roots"], document["contract_root"], document["compiler_shared_source_root"]):
        if not any(PurePosixPath(value).is_relative_to(root) for root in document["source_roots"]):
            raise ValueError("portfolio deployment input root is outside declared source ownership")
    for value in document["python_roots"]:
        if not any(
            PurePosixPath(value).is_relative_to(root) or PurePosixPath(root).is_relative_to(value)
            for root in document["source_roots"]
        ):
            raise ValueError("portfolio import root has no declared source ownership")
    return document


def _live_path(path: Path, *, directory: bool) -> Path:
    if path.is_symlink() or path.resolve() != path or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"portfolio deployment path is linked, absent or noncanonical: {path}")
    return path


def _source_inputs(path: Path, deployment: dict) -> dict[str, str]:
    source = _live_path(Path(deployment["source_root"]), directory=True)
    for field in ("output_root", "lease_path", "functional_runs_root"):
        selected = Path(deployment[field])
        if selected.is_symlink() or selected.resolve() != selected:
            raise ValueError(f"portfolio deployment {field} is not canonical")
    paths = {"deployment": str(path.absolute()), "interpreter": sys.executable}
    for root in deployment["source_roots"]:
        selected = _live_path(source / root, directory=True)
        paths["source_root:" + root] = str(selected)
    owners = {}
    for root in deployment["python_roots"]:
        import_root = _live_path(source / root, directory=True)
        selected_roots = [
            source / value for value in deployment["source_roots"] if (source / value).is_relative_to(import_root)
        ]
        if any(import_root.is_relative_to(source / value) for value in deployment["source_roots"]):
            selected_roots = [import_root]
        members = {}
        for selected_root in selected_roots:
            for selected in python_members(selected_root, label="portfolio deployment").values():
                relative = selected.relative_to(import_root).as_posix()
                if relative in members:
                    raise ValueError(f"portfolio deployment has duplicate source membership: {relative}")
                members[relative] = selected
        for relative, selected in members.items():
            if any(selected.relative_to(source).is_relative_to(value) for value in deployment["exclude_paths"]):
                continue
            parts = list(PurePosixPath(relative).with_suffix("").parts)
            if parts[0] not in {"merlin", "merlin_experiments"}:
                continue
            if parts[-1] == "__init__":
                parts.pop()
            name = ".".join(parts)
            if name in owners:
                raise ValueError(f"portfolio deployment has duplicate Python ownership: {name}")
            owners[name] = selected
    for name in (
        "merlin",
        "merlin.common.paths",
        "merlin.common.source_membership",
        "merlin_experiments",
        MODULE,
        WORKER.__name__,
    ):
        if owners.get(name) != module_source_path(name).resolve():
            raise ValueError(f"portfolio deployment differs from installed module owner: {name}")
    for name, module in tuple(sys.modules.items()):
        if module is None or not (
            name == "merlin" or name == "merlin_experiments" or name.startswith(("merlin.", "merlin_experiments."))
        ):
            continue
        origin = getattr(module, "__file__", None)
        # Namespace-only contributions have no executable initializer. All their
        # selected locations must still be beneath the explicit import roots.
        if origin is None and getattr(module, "__path__", None) is not None:
            actual = {Path(value).resolve() for value in module.__path__}
            expected = {
                source / root / name.replace(".", "/")
                for root in deployment["python_roots"]
                if (source / root / name.replace(".", "/")).is_dir()
            }
            if actual and actual == expected and getattr(module, "__name__", None) == name:
                continue
        canonical = getattr(module, "__name__", name)
        spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if (
            name not in owners
            or canonical != name
            or origin is None
            or Path(origin).resolve() != owners[name]
            or (spec_origin is not None and Path(spec_origin).resolve() != owners[name])
        ):
            raise ValueError(f"portfolio deployment loaded module origin differs: {name}")
    main = sys.modules.get("__main__")
    if getattr(getattr(main, "__spec__", None), "name", None) == MODULE:
        if Path(main.__file__).resolve() != owners[MODULE]:
            raise ValueError("portfolio deployment executing CLI origin differs")
    _live_path(Path(deployment["sandbox_root"]), directory=True)
    declaration = _live_path(Path(deployment["sandbox_declaration"]), directory=False)
    if C.sha256_file(declaration) != deployment["sandbox_declaration_sha256"]:
        raise ValueError("portfolio sandbox declaration differs from its pin")
    paths["sandbox_declaration"] = str(declaration)
    if deployment["provider_root"] is not None:
        paths["provider_root"] = str(_live_path(Path(deployment["provider_root"]), directory=True))
        _provider(deployment)
    return paths


def deployment_source_inputs(path: Path) -> dict[str, str]:
    """Admit installed owners and return immutable catalog inputs, excluding output roots."""
    return _source_inputs(path, load_deployment(path))


def _provider(deployment: dict) -> dict | None:
    if deployment["provider_root"] is None:
        return None
    selected = read_provider(deployment["provider_root"])
    if selected is None or selected.role != ProviderRole.SUPPORT or selected.target != deployment["target"]:
        raise ValueError("portfolio deployment requires matching external support provider")
    return {
        "target": deployment["target"],
        "resolved_target": selected.target,
        "kind": "external",
        "source": str(selected.root),
    }


def _campaign(document: dict) -> dict:
    config = document["campaigns"][0]["config"] if "campaigns" in document else document.get("config", document)
    if not isinstance(config, dict):
        raise ValueError("portfolio campaign config must be an object")
    return config


def _sealed_directory(snapshot: Path, receipt: dict, relative: str) -> Path:
    path = snapshot / relative
    if relative not in receipt["directories"] or path.resolve() != path or path.is_symlink() or not path.is_dir():
        raise ValueError("portfolio resource root is outside sealed directory ownership")
    return path


def _worker_source(snapshot: Path, receipt: dict) -> Path:
    suffix = WORKER.__name__.replace(".", "/") + ".py"
    matches = [snapshot / root / suffix for root in receipt["python_roots"] if root + "/" + suffix in receipt["files"]]
    if len(matches) != 1 or matches[0].resolve() != Path(WORKER.__file__).resolve():
        raise ValueError("portfolio worker lacks unique active sealed source ownership")
    return matches[0]


def _admit_destinations(args, deployment: dict, declared: dict[str, Path]) -> None:
    immutable = [Path(deployment["source_root"]) / root for root in deployment["source_roots"]]
    immutable.extend(path.resolve() for path in declared.values())
    immutable.append(Path(deployment["sandbox_root"]))
    if deployment["provider_root"] is not None:
        immutable.append(Path(deployment["provider_root"]))
    output = args.output.resolve()
    destinations = (
        output,
        output.with_name(output.name + ".source"),
        output.with_name(output.name + ".transport"),
        args.candidate.resolve(),
        Path(deployment["lease_path"]),
    )
    for destination in destinations:
        if any(destination.is_relative_to(path) or path.is_relative_to(destination) for path in immutable):
            raise ValueError("portfolio destination or candidate overlaps immutable deployment inputs")
    candidate = args.candidate.resolve()
    if any(path.is_relative_to(candidate) or candidate.is_relative_to(path) for path in destinations[:3]):
        raise ValueError("portfolio stage output overlaps its candidate input")


def main(argv: list[str] | None = None) -> int:
    invocation = parse_invocation(argv, description=__doc__)
    args = invocation.args
    if args.deployment is None:
        raise ValueError("installed portfolio launch requires --deployment")
    if not args.source_worker:
        deployment, deployment_raw = _json_bytes(args.deployment)
        deployment = _decode_deployment(deployment)
        _source_inputs(args.deployment, deployment)
        campaign, campaign_raw = _json_bytes(args.campaign_config)
        config = _campaign(campaign)
        descriptor = Path(config["descriptor"])
        _live_path(descriptor, directory=False)
        descriptor_raw = _ordinary_bytes(descriptor)
        descriptor_document = yaml.safe_load(descriptor_raw)
        if not isinstance(descriptor_document, dict) or descriptor_document.get("target") != deployment["target"]:
            raise ValueError("portfolio deployment target differs from campaign descriptor")
        provider = _provider(deployment)
        declared = {
            "deployment": args.deployment.absolute(),
            "campaign_config": args.campaign_config.absolute(),
            "sandbox_declaration": Path(deployment["sandbox_declaration"]),
            "descriptor": descriptor,
        }
        if config.get("telemetry_price_table") is not None:
            declared["telemetry_price_table"] = Path(config["telemetry_price_table"])
        _admit_destinations(args, deployment, declared)
        pins = {
            "deployment": hashlib.sha256(deployment_raw).hexdigest(),
            "campaign_config": hashlib.sha256(campaign_raw).hexdigest(),
            "descriptor": hashlib.sha256(descriptor_raw).hexdigest(),
            "sandbox_declaration": deployment["sandbox_declaration_sha256"],
        }
        if "telemetry_price_table" in declared:
            pins["telemetry_price_table"] = hashlib.sha256(
                _ordinary_bytes(declared["telemetry_price_table"])
            ).hexdigest()
        return LAUNCH.launch(
            invocation,
            deployment=LAUNCH.PortfolioDeployment(
                source_root=Path(deployment["source_root"]),
                output_root=Path(deployment["output_root"]),
                lease_path=Path(deployment["lease_path"]),
                worker_entrypoint=(sys.executable, "-m", MODULE),
                snapshot_options={
                    name: tuple(deployment[name])
                    for name in ("source_roots", "python_roots", "legacy_roots", "exclude_paths")
                }
                | {"internal_aliases": deployment["internal_aliases"]},
                target_name=deployment["target"],
                selected_provider=provider,
                declared_inputs=declared,
                inherited_environment=dict(os.environ),
                worker_python_roots=tuple(deployment["python_roots"] + deployment["legacy_roots"]),
                declared_input_sha256=pins,
            ),
        )
    identity = FP.active_source_identity()
    if identity is None:
        raise ValueError("portfolio worker requires active frozen Python source identity")
    seal = Path(identity["path"])
    snapshot = seal.parent
    receipt = SNAP.verify(snapshot)
    actual_seal, _ = SNAP.load_seal(snapshot, "snapshot")
    if actual_seal != seal or C.sha256_file(seal) != identity["sha256"]:
        raise ValueError("portfolio worker active snapshot seal differs")
    deployment_path = SNAP.remap_input(snapshot, receipt, args.deployment, name="deployment")
    campaign_path = SNAP.remap_input(snapshot, receipt, args.campaign_config, name="campaign_config")
    deployment = load_deployment(deployment_path)
    if (
        receipt["source_root"] != deployment["source_root"]
        or receipt["source_roots"] != deployment["source_roots"]
        or receipt["python_roots"] != deployment["python_roots"]
        or receipt["legacy_roots"] != deployment["legacy_roots"]
        or receipt["internal_aliases"] != deployment["internal_aliases"]
    ):
        raise ValueError("archived portfolio deployment differs from sealed source selection")
    provider = receipt.get("selected_provider")
    if deployment["provider_root"] is None:
        if provider is not None:
            raise ValueError("archived portfolio provider absence differs from sealed selection")
    elif not isinstance(provider, dict) or any(
        provider.get(key) != value
        for key, value in {
            "target": deployment["target"],
            "resolved_target": deployment["target"],
            "kind": "external",
            "source": deployment["provider_root"],
            "root": SNAP.PROVIDER_ROOT,
        }.items()
    ):
        raise ValueError("archived portfolio provider differs from sealed selection")
    config = _campaign(_json(campaign_path))
    declaration_path = SNAP.remap_input(
        snapshot, receipt, Path(deployment["sandbox_declaration"]), name="sandbox_declaration"
    )
    raw = _ordinary_bytes(declaration_path)
    if hashlib.sha256(raw).hexdigest() != deployment["sandbox_declaration_sha256"]:
        raise ValueError("portfolio sandbox declaration differs from its pin")
    declaration = json.loads(raw)
    sandbox_inputs = Q.restore(
        _live_path(Path(deployment["sandbox_root"]), directory=True), declaration["execution_policy"]
    )
    contract = _sealed_directory(snapshot, receipt, deployment["contract_root"])
    shared = _sealed_directory(snapshot, receipt, deployment["compiler_shared_source_root"])
    # The explicit contract follows the existing contract layout: its schema
    # directory is the 'schemas' child, not a separately discovered checkout path.
    os.environ["MERLIN_CONTRACT_DIR"] = str(contract)
    os.environ["MERLIN_SCHEMAS_DIR"] = str(contract / "schemas")
    descriptor = SNAP.remap_input(snapshot, receipt, Path(config["descriptor"]), name="descriptor")
    target = load_target_experiment(descriptor, source_root=snapshot)
    if target.target != deployment["target"]:
        raise ValueError("portfolio deployment target differs from archived descriptor")
    worker_args = argparse.Namespace(**vars(args))
    worker_args.campaign_config = campaign_path
    return WORKER.run(
        replace(invocation, args=worker_args),
        config,
        context=WORKER.PortfolioWorkerContext(
            snapshot_root=snapshot,
            functional_runs_root=Path(deployment["functional_runs_root"]),
            contract_root=contract,
            compiler_shared_source_root=shared,
            controller_source=_worker_source(snapshot, receipt),
            prior_shared_source_relative=Path(deployment["compiler_shared_source_root"]),
            prior_shared_source_fallback=shared,
            guidance_contract=contract,
            sandbox_inputs=sandbox_inputs,
            target_experiment=target,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
