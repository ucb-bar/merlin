"""Run one canonical native model build, then reuse its admitted images in the chain."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import launch_phase1 as P
from merlin.common.digest import sha256_file
from merlin.model_check import cli
from merlin_experiments.phase1.feedback import private_model_chain as chain


def owner_alive(pin):
    """A readiness wait is bounded by the actual original child's identity."""
    try:
        fields = (Path('/proc') / str(pin['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        return (fields[0] not in {'Z', 'X'} and int(fields[19]) == pin['start_ticks']
                and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == pin['boot_id'])
    except (OSError, ValueError, KeyError, IndexError):
        return False


def wait_owner(run_dir, pin, deadline):
    """The later native API verifies authority; these files only signal readiness."""
    while True:
        if time.time() >= deadline:
            raise ValueError('native owner readiness deadline expired')
        if not owner_alive(pin):
            raise ValueError('original native owner exited before model readiness')
        required = ('environment.yaml', 'input_bundle_manifest.yaml',
                    'private_full_model_input/spec.yaml', 'private_instruction_input/selection.json')
        if all((run_dir / name).is_file() for name in required):
            # Do not retry a malformed/contradictory owner as a transient wait.
            from merlin_experiments.phase1.feedback.compiler_profile_runtime import functional_service
            if functional_service(run_dir) is None:
                raise ValueError('prepared native owner has no original compiler profile')
            return
        time.sleep(min(1, max(0, deadline - time.time())))


def model_arguments(argv):
    """Accept safe-path/unbuffered interpreter flags without assuming a prefix length."""
    positions = [i for i, arg in enumerate(argv[:-1]) if arg == '-m' and argv[i + 1] == 'merlin.model_check']
    if len(positions) != 1:
        raise ValueError('full-chain job is not the selected model-check invocation')
    return argv[positions[0] + 2:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'owner-pin', 'run-dir', 'descriptor', 'submission', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args(argv)
    job_sha, owner_sha = sha256_file(args.job), sha256_file(args.owner_pin)
    job, owner = json.loads(args.job.read_bytes()), json.loads(args.owner_pin.read_bytes())
    wait_owner(args.run_dir, owner, args.deadline_unix)
    partition = chain.run_static(run_dir=args.run_dir, descriptor=args.descriptor,
                                 submission=args.submission, name=args.model, out=args.out)
    P.save(args.out / 'partition-ready.json', {'static': str(partition.path),
           'sha256': sha256_file(partition.path), 'invocation': str(partition.invocation)})
    result = cli.main(model_arguments(job['argv']), full_build_provider=partition.build)
    partition.verify()
    if sha256_file(args.job) != job_sha or sha256_file(args.owner_pin) != owner_sha:
        raise ValueError('original model/native job identity changed')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
