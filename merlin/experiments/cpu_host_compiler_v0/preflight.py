#!/usr/bin/env python3
"""Record the CPU-host experiment GO/NO_GO decision in the canonical AET output tree."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml

from merlin.common.artifacts import finish_run, start_run
from merlin.compare.host_experiment import HostExperimentSpec
from merlin.compare.frozen_environment import capture_frozen_environment


def freeze_protocol(source: Path, output: Path, *, check_environment: bool = True,
                    probe_board: bool = False) -> tuple[HostExperimentSpec, object]:
    """Publish one verified frozen spec without ever replacing another publisher's result."""
    source, output = source.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = output.with_name(f".{output.name}.freeze.lock")
    try:
        descriptor = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"another protocol freezer owns the output reservation: {reservation}") from exc
    environment_capture = None
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")
        spec = HostExperimentSpec.from_yaml(source)
        if spec.status != "draft":
            raise ValueError("freeze-protocol accepts only a draft source spec")
        if check_environment and not probe_board:
            raise ValueError(
                "a live environment freeze requires probe_board=True so K1 identity is captured")
        draft_check = spec.preflight(
            check_environment=check_environment, probe_board=probe_board, require_frozen=False)
        if not draft_check.ready:
            raise ValueError(f"draft protocol preflight is NO_GO: {draft_check.to_dict()}")
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        environment_path = output.with_name(f"{output.stem}.environment.json")
        environment_capture = capture_frozen_environment(
            environment_path,
            source_paths={name: Path(path) for name, path in draft_check.evidence["paths"].items()
                          if name != "environment_manifest" and Path(path).is_file()},
            agent=spec.agent, telemetry=spec.telemetry,
            probe_source=spec._repo_path(spec.grading["k1_probe_source"]),
            include_live_board=bool(check_environment and probe_board))
        raw["environment"]["manifest"] = environment_capture["path"]
        raw["environment"]["sha256"] = environment_capture["sha256"]
        # Recompute the protocol digest from the augmented draft. The environment identity is a
        # protocol input, while protocol_inputs_sha256 itself is intentionally only an output.
        augmented = HostExperimentSpec.parse(raw, source_path=source)
        augmented_check = augmented.preflight(
            check_environment=False, probe_board=False, require_frozen=False)
        if not augmented_check.ready:
            Path(environment_capture["source_bundle_path"]).unlink(missing_ok=True)
            environment_path.unlink(missing_ok=True)
            raise ValueError(
                f"environment-bound draft preflight is NO_GO: {augmented_check.to_dict()}")
        raw["status"] = "protocol_frozen"
        raw["freeze"]["protocol_inputs_sha256"] = augmented_check.evidence[
            "protocol_inputs_sha256"]
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                yaml.safe_dump(raw, stream, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
            frozen = HostExperimentSpec.from_yaml(temporary)
            frozen_check = frozen.preflight(
                check_environment=False, probe_board=False, require_frozen=True)
            if not frozen_check.ready:
                raise ValueError(
                    f"round-tripped frozen protocol is NO_GO: {frozen_check.to_dict()}")
            # link(2) atomically creates the final name and fails if it appeared after the
            # reservation check.  Unlike replace(2), it can never overwrite another result.
            try:
                os.link(temporary, output)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite frozen protocol: {output}") from exc
            published = True
            return HostExperimentSpec.from_yaml(output), frozen_check
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        if environment_capture is not None and not published:
            Path(environment_capture["source_bundle_path"]).unlink(missing_ok=True)
            Path(environment_capture["path"]).unlink(missing_ok=True)
        reservation.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(Path(__file__).with_name("experiment.yaml")))
    parser.add_argument("--probe-board", action="store_true",
                        help="compile and run the architecture probe on K1 (required when freezing)")
    parser.add_argument(
        "--machinery-only", action="store_true",
        help="check implementation/tool readiness without authorizing a live paper campaign")
    parser.add_argument(
        "--freeze-protocol", type=Path, metavar="OUTPUT",
        help="atomically write a new protocol_frozen spec after successful checks")
    args = parser.parse_args(argv)

    spec = HostExperimentSpec.from_yaml(args.spec)
    handle = start_run(
        suite="cpu-host-compiler", method="preflight", target="k1_cpu",
        extra={"experiment": spec.label, "spec": str(Path(args.spec).resolve()),
               "probe_board": args.probe_board},
    )
    status = "error"
    result = None
    try:
        frozen_output = None
        if args.freeze_protocol is not None:
            spec, result = freeze_protocol(
                Path(args.spec), args.freeze_protocol, check_environment=True,
                probe_board=args.probe_board)
            frozen_output = str(args.freeze_protocol.resolve())
        else:
            result = spec.preflight(
                check_environment=True, probe_board=args.probe_board,
                require_frozen=not args.machinery_only)
        out = handle.run_dir / "contracts" / "preflight.yaml"
        out.write_text(yaml.safe_dump(result.to_dict(), sort_keys=False), encoding="utf-8")
        status = "ok" if result.ready else "blocked"
        print(yaml.safe_dump(result.to_dict(), sort_keys=False), end="")
        if frozen_output:
            print(f"frozen_protocol: {frozen_output}")
        print(f"preflight_record: {out}")
        return 0 if result.ready else 2
    finally:
        summary = {"ready": bool(result and result.ready),
                   "errors": len(result.errors) if result else 1,
                   "blockers": len(result.blockers) if result else 1}
        finish_run(handle, status=status, summary=summary)


if __name__ == "__main__":
    raise SystemExit(main())
