"""Apply an author-declared pipeline adapter and check its actual emission.

No target adapter is supplied. ``prepare(source, *, package, mechanism)`` must
return session/facts/registry/policy/output for existing pipeline_policy APIs.
``mechanism=None`` is the incumbent. The compiler's declared emit entrypoint
must consume LIBRARY_PASS_SELECTION.json and use that same pipeline adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SELECTION = 'LIBRARY_PASS_SELECTION.json'
# Also consumed by feedback.idle_tuning; sibling exists before variant exposure.
PENDING_PREFIX = '.library-pass-pending-'
IGNORED = frozenset({'.git', 'build', 'out', 'qa', 'selfcheck_out', '__pycache__',
                     '.pytest_cache', '.ruff_cache', '.qa_channel', 'evidence'})


class ConstructionRequired(ValueError):
    pass


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def pin(path):
    return dict(path=str(path.resolve()), sha256=digest(path.read_bytes()))


def files(root, library):
    """Read source only; dereference safe links into regular variant files."""
    def visit(path, relative, ancestors):
        if relative == Path('compiler_library'):
            return  # Selected immutable library is pinned and reused separately.
        real = path.resolve(strict=True)
        if not (real.is_relative_to(root) or real.is_relative_to(library)):
            raise ValueError(f'unsafe source link: {relative}')
        if real.is_dir():
            if real in ancestors:
                raise ValueError(f'cyclic source link: {relative}')
            for child in sorted(path.iterdir()):
                if child.name not in IGNORED:
                    yield from visit(child, relative / child.name, ancestors | {real})
        elif real.is_file():
            yield relative, path
        else:
            raise ValueError(f'unsupported source member: {relative}')
    return tuple(visit(root, Path(), frozenset()))


def snapshot(root, library):
    rows = {str(relative): digest(path.read_bytes()) for relative, path in files(root, library)}
    return digest(json.dumps(rows, sort_keys=True).encode()), rows


def load_adapter(package, relative):
    path = package / relative
    if not path.is_file():
        raise ConstructionRequired(f'Implement author adapter {relative}: prepare(source, *, package, mechanism).')
    if not path.resolve().is_relative_to(package):
        raise ValueError('adapter must be authored inside the selected package')
    spec = importlib.util.spec_from_file_location('library_pass_adapter_' + digest(str(path).encode()), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'prepare', None)):
        raise ConstructionRequired('Adapter must implement prepare(source, *, package, mechanism).')
    return module, path


def prepare(adapter, source, package, mechanism):
    from merlin.transforms.pipeline_policy import PipelinePolicy, PolicyFacts, registry_metadata
    from merlin.transforms.pipeline_state import PipelineSession

    try:
        binding = adapter.prepare(source, package=package, mechanism=mechanism)
    except KeyError as error:
        raise ConstructionRequired(f'Adapter has no declared mapping for mechanism {mechanism!r}.') from error
    if type(binding) is not dict or set(binding) != {'session', 'facts', 'registry', 'policy', 'output'}:
        raise ConstructionRequired('prepare must return session, facts, registry, policy and output.')
    if (type(binding['session']) is not PipelineSession or type(binding['facts']) is not PolicyFacts
            or type(binding['policy']) is not PipelinePolicy):
        raise ConstructionRequired('Adapter must construct the existing typed pipeline session/facts/policy.')
    registry_metadata(binding['registry'])
    return binding


def execute(binding):
    from merlin.transforms.pipeline_policy import execute_policy

    result = execute_policy(**binding)
    artifact = result.artifact
    if type(artifact) is str:
        artifact = artifact.encode()
    if type(artifact) is not bytes or not artifact:
        raise ValueError('Final registered emitter must produce nonempty target text or bytes')
    return artifact, result.receipt


def emit(package, source, command, timeout, directory, library):
    from merlin.common.yaml import load_yaml

    manifest = load_yaml(package / 'manifest.yaml')
    declaration = ((manifest.get('commands') or {}).get(command) or {}).get('argv')
    if (type(declaration) is not list or not declaration
            or any(type(v) is not str or not v for v in declaration)):
        raise ConstructionRequired(f'Declare a stdout-producing manifest command {command!r}.')
    bindings = dict(input_mlir=str(source))
    if any('{tool}' in token for token in declaration):
        build = manifest.get('build') or {}
        relative = build.get('tool_output') or (manifest.get('entrypoints') or {}).get('tool')
        if not isinstance(relative, str) or not relative:
            raise ConstructionRequired('Declare entrypoints.tool (or build.tool_output) for {tool}.')
        tool = (package / relative).resolve()
        if not tool.is_relative_to(package.resolve()) or not tool.is_file():
            raise ConstructionRequired('The declared tool must be built inside the selected package.')
        bindings['tool'] = str(tool)
    try:
        argv = [v.format(**bindings) for v in declaration]
    except (KeyError, ValueError) as error:
        raise ConstructionRequired('Probe command supports declared {tool} and {input_mlir} bindings only.') from error
    executable = shutil.which(argv[0])
    if executable is None:
        raise ConstructionRequired(f'Compiler executable unavailable: {argv[0]}')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    selected = package / 'compiler_library'
    env['PYTHONPATH'] = str(selected if selected.is_dir() else library)
    process = subprocess.run(argv, cwd=package, env=env, capture_output=True, timeout=timeout)
    directory.mkdir(parents=True)
    (directory / 'stdout').write_bytes(process.stdout)
    (directory / 'stderr').write_bytes(process.stderr)
    record = dict(argv=argv, cwd=str(package), executable=pin(Path(executable)),
                  returncode=process.returncode, stdout=pin(directory / 'stdout'), stderr=pin(directory / 'stderr'))
    if process.returncode or not process.stdout:
        raise ValueError(f'Actual compiler emission failed; inspect {directory / "stderr"}')
    return process.stdout, record


def apply(args):
    base, variant = args.submission.resolve(), args.variant.resolve()
    if not base.is_dir() or variant.exists() or variant.is_symlink() or variant.is_relative_to(base):
        raise ValueError('Use an existing submission and a new separate variant directory')
    variant.parent.mkdir(parents=True, exist_ok=True)
    with (variant.parent / (PENDING_PREFIX + variant.name)).open('x') as marker:
        marker.write('Pending or refused library-pass application; not an idle-tuning candidate.\n')
    library = (args.library or base / 'compiler_library').resolve()
    if not library.is_dir():
        raise ConstructionRequired('Select the existing author compiler library with --library.')
    if (base / 'compiler_library').exists() and (base / 'compiler_library').resolve() != library:
        raise ValueError('--library must match the submission selected compiler_library')
    library_before, _ = snapshot(library, library)
    sys.path[:0] = [str(library), str(base / 'tools'), str(base)]
    adapter, adapter_path = load_adapter(base, args.adapter)
    source = args.source
    if source is None:
        locator = getattr(adapter, 'source', None)
        if not callable(locator):
            raise ConstructionRequired('Pass --source OWN_SOURCE.mlir or implement source(case, *, package).')
        source = locator(args.case, package=base)
    source = Path(source).resolve()
    if not source.is_file():
        raise ConstructionRequired('Adapter source locator must return an existing current source file.')
    source_pin = pin(source)
    before, members = snapshot(base, library)
    for relative, path in files(base, library):
        destination = variant / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (variant / 'compiler_library').symlink_to(library, target_is_directory=True)
    (variant / SELECTION).write_text(json.dumps(dict(
        schema='merlin.author_library_pass_selection.v1', adapter=args.adapter,
        mechanism=args.mechanism), indent=2) + '\n')
    other, _ = load_adapter(variant.resolve(), args.adapter)
    baseline = prepare(adapter, source, base, None)
    proposed = prepare(other, source, variant.resolve(), args.mechanism)
    if baseline['session'].stamp() != proposed['session'].stamp() or baseline['facts'] != proposed['facts']:
        raise ValueError('Paired pipelines must start from identical current source and selected facts')
    from merlin.transforms.pipeline_policy import registry_metadata

    if registry_metadata(baseline['registry']) != registry_metadata(proposed['registry']):
        raise ValueError('Both pipelines must use the same registered pass/emitter contracts')
    expected_base, base_receipt = execute(baseline)
    expected_variant, variant_receipt = execute(proposed)
    if base_receipt['passes'][-1]['name'] != variant_receipt['passes'][-1]['name']:
        raise ValueError('Both pipelines must end in the same actual registered target emitter')
    if expected_base == expected_variant:
        raise ValueError('UNCHANGED: declared mechanism produced no emitted product change')
    artifacts = args.receipt.parent / (args.receipt.stem + '-products')
    actual_base, base_process = emit(base, source, args.command, args.timeout, artifacts / 'baseline', library)
    actual_variant, variant_process = emit(variant.resolve(), source, args.command, args.timeout,
                                           artifacts / 'variant', library)
    if actual_base != expected_base or actual_variant != expected_variant:
        raise ValueError('DISCONNECTED_ADAPTER: actual manifest emission differs from registered pipeline output; '
                         'connect the compiler entrypoint to its declared adapter and selection file')
    if actual_base == actual_variant:
        raise ValueError('UNCHANGED: actual compiler emitted identical A/B products')
    if (snapshot(base, library) != (before, members) or pin(source) != source_pin
            or snapshot(library, library)[0] != library_before):
        raise ValueError('Original submission, selected library or source changed during application')
    after, _ = snapshot(variant.resolve(), library)
    return dict(status='APPLIED_EMITTED_CHANGE', schema='merlin.author_library_pass_application.v1',
                mechanism=args.mechanism, case_locator_only=args.case, source=source_pin,
                adapter=pin(adapter_path), baseline_tree_sha256=before, variant_tree_sha256=after,
                selected_library=dict(path=str(library), source_tree_sha256=library_before, reused=True),
                baseline_pipeline=base_receipt, variant_pipeline=variant_receipt,
                baseline_emission=base_process, variant_emission=variant_process,
                variant=str(variant), actual_emitted_products_differ=True,
                numerical_status='NOT_CHECKED', timing_status='UNKNOWN', qualification=
                'Actual compilation-path application only; ordinary same-source numerics, ordering and timing '
                'checks remain required. The mechanism-to-action mapping is authored; declared registry '
                'contracts and a catalogue ID do not independently prove catalogue implementation use. '
                'No target semantics, adapter or hardware law is supplied.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--submission', type=Path, default=Path('submission'))
    parser.add_argument('--variant', type=Path, required=True)
    parser.add_argument('--adapter', default='tools/library_pass_adapter.py')
    parser.add_argument('--mechanism', required=True)
    parser.add_argument('--case', default='', help='Source locator/provenance only, never a pass predicate')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--command', default='lower_target_to_llvm')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 900:
        parser.error('--timeout must be in 1..900 seconds')
    args.receipt = args.receipt.absolute()
    try:
        record, code = apply(args), 0
    except ConstructionRequired as error:
        record, code = dict(status='CONSTRUCTION_REQUIRED', detail=str(error), numerical_status='NOT_CHECKED',
                            timing_status='UNKNOWN'), 2
    except (OSError, ValueError, RuntimeError, TypeError, ImportError, subprocess.SubprocessError) as error:
        record, code = dict(status='REFUSED', detail=str(error), numerical_status='NOT_CHECKED',
                            timing_status='UNKNOWN'), 1
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    if code == 0:
        variant = Path(record['variant'])
        (variant.parent / (PENDING_PREFIX + variant.name)).unlink()
    print(json.dumps(record))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
