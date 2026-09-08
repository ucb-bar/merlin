"""Build a complete SHORT program inside the host's deadline-managed worker.

No simulator, reference computation, candidate import, or sandbox grants live
here. The supplied runner must isolate the worker and kill its entire process
group on timeout, including nested translator/compiler/linker processes. Source
and storage bounds do not establish emitted/source numerical correspondence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

MAX_LLVM_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 256 * 1024
MAX_STORAGE_BYTES = 64 * 1024
MAX_DOMAIN_POINTS = 20000


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("short program input exceeds host byte bound")
    return data


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bounds(source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.perf.model_macs import _domain, _iterators
    from merlin.perf.structural_transitions import _element_bytes
    from xdsl.dialects.builtin import TensorType
    module = parse_mlir_text(source)
    functions = list(module.body.block.ops)
    if (len(functions) != 1 or functions[0].name != "func.func"
            or len(functions[0].body.blocks) != 1):
        raise ValueError("short source must contain exactly one closed single-block entry")
    allowed = {"builtin.module", "func.func", "func.return", "tensor.empty", "tensor.splat",
               "tensor.insert_slice", "tensor.extract_slice", "linalg.generic", "linalg.fill",
               "linalg.matmul", "linalg.yield"}
    if any(not isinstance(arg.type, TensorType) for arg in functions[0].body.block.args):
        raise ValueError("short source boundary must contain only static tensors")
    values, points, scalar_work = set(functions[0].body.block.args), 0, 0
    for op in module.walk():
        if op.name not in allowed and not op.name.startswith("arith."):
            raise ValueError("short source has unsupported work/control flow: " + op.name)
        values.update(op.results)
        if op.name.startswith("linalg.") and op.name not in {"linalg.yield", "linalg.fill"}:
            if op.parent_op() != functions[0]:
                raise ValueError("nested source work is unsupported")
            parallel, reduction, _ = _domain(op, _iterators(op) or ())
            extent = math.prod(parallel + reduction)
            points += extent
            scalar_work += extent * sum(1 for item in op.walk() if item is not op)
        elif op.name == "linalg.fill":
            points += math.prod(op.results[0].type.get_shape())
    tensor_bytes = 0
    for value in values:
        if isinstance(value.type, TensorType):
            shape = value.type.get_shape()
            if any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("short source requires static nonempty tensor extents")
            tensor_bytes += math.prod(shape) * _element_bytes(str(value.type.get_element_type()))
    operation_count = sum(1 for _ in module.walk())
    if (tensor_bytes > MAX_STORAGE_BYTES or points > MAX_DOMAIN_POINTS
            or scalar_work > 100000 or operation_count > 1024):
        raise ValueError("actual short source exceeds tensor/work bound (entry renaming does not reduce work)")
    module.verify()
    return {"entry": functions[0].sym_name.data, "static_ssa_tensor_bytes": tensor_bytes,
            "linalg_domain_points": points, "bounded_scalar_body_operations": scalar_work,
            "source_operations": operation_count}


def _inputs(cb, payloads, *, legacy_dim=None, legacy_format=None):
    from merlin.runtime.storage_binding import resolve_storage_bindings
    from merlin.runtime.tensor import Tensor
    from merlin.perf.storage_encoding import GroupedAxesStorage
    abi = cb.get("kernel_abi", {})
    if (cb.get("target") != "gemmini" or abi.get("kind") != "whole_program"
            or cb.get("canonical_inputs") or not isinstance(abi.get("args"), list)
            or not 1 <= len(abi["args"]) <= 32):
        raise ValueError("explicit short whole_program ABI required, without recorded operands")
    if any(arg.get("access") not in {"read", "write"} for arg in abi["args"]):
        raise ValueError("readwrite warm input restoration is not supported by this renderer")
    reads = {arg["tensor"] for arg in abi["args"] if arg["access"] == "read"}
    names = [arg["tensor"] for arg in abi["args"]]
    if len(set(names)) != len(names) or any(name not in cb.get("tensors", {}) for name in names):
        raise ValueError("short caller arguments require distinct declared tensors")
    if not reads or set(payloads) != reads:
        raise ValueError("every read argument requires exact explicit logical payload bytes")
    records = cb.get("params", {}).get("storage_encodings")
    legacy = records is None
    if legacy and (type(legacy_dim) is not int or legacy_dim <= 0 or legacy_format is None):
        raise ValueError("legacy short caller requires host-pinned physical ABI facts")
    storage = {}
    if legacy:
        total_storage = 0
        for name in names:
            spec = cb["tensors"][name]
            rows, cols = legacy_format.buffer_extent(spec, name=name)
            prows, pcols = (legacy_format.ceil_dim(value, legacy_dim) for value in (rows, cols))
            from merlin.common.quant_formats import storage_bits
            bits = storage_bits(spec["dtype"])
            if bits not in (8, 16, 32, 64):
                raise ValueError("legacy short storage requires byte-addressable scalar elements")
            size = prows*pcols*(bits//8)
            total_storage += size
            storage[name] = {"logical_shape": spec["shape"], "dtype": spec["dtype"],
                "rows": rows, "cols": cols, "physical_rows": prows, "physical_cols": pcols,
                "row_stride_bytes": pcols*(bits//8), "storage_bytes": size,
                "scope": "existing legacy row-major padded caller ABI, not candidate storage metadata"}
        if total_storage > MAX_STORAGE_BYTES:
            raise ValueError("padded legacy caller allocations exceed short storage cap")
    formats = {"i8": "b", "i16": "h", "i32": "i", "i64": "q", "f32": "f"}
    result, total = {}, 0
    for name, payload in payloads.items():
        if legacy:
            dtype, shape = cb["tensors"][name]["dtype"], tuple(cb["tensors"][name]["shape"])
        else:
            encoding = GroupedAxesStorage.from_dict(records.get(name, {}))
            dtype, shape = encoding.dtype, encoding.logical_shape
        if dtype not in formats or type(payload) is not bytes:
            raise ValueError("unsupported logical payload dtype or byte representation")
        fmt = "<" + formats[dtype]  # Explicit witness serialization, not target byte order.
        width = struct.calcsize(fmt)
        if len(payload) != math.prod(shape) * width:
            raise ValueError("logical payload byte count disagrees with source encoding")
        total += len(payload)
        if total > MAX_STORAGE_BYTES:
            raise ValueError("logical payload exceeds host byte bound")
        values = [item[0] for item in struct.iter_unpack(fmt, payload)]
        # Float conversion may quiet signaling NaNs. Refuse instead of silently
        # changing source input bits; the existing renderer retains finite f32.
        if b"".join(struct.pack(fmt, value) for value in values) != payload:
            raise ValueError("logical payload cannot round-trip without changing scalar bits")
        result[name] = Tensor(tuple(shape), values, dtype)
    if legacy:
        return result, storage
    bindings = resolve_storage_bindings(cb, result, max_storage_bytes=MAX_STORAGE_BYTES)
    if bindings is None:
        raise ValueError("complete explicit storage contracts required")
    return result, {name: value.setup_evidence() for name, value in bindings.items()}


def _legacy_dimension(specification, *, host=False):
    """Derive caller geometry from the selected already-pinned recipe headers."""
    headers = [row for row in specification["dependencies"] if row["kind"] == "header"
               and Path(row["source"]).name == "gemmini_params.h"]
    if not headers:
        raise ValueError("legacy caller dimension has no pinned recipe parameter header")
    dimensions = set()
    for row in headers:
        path = Path(row["source" if host else "destination"])
        raw = _read(path, MAX_SOURCE_BYTES)
        if _digest(raw) != row["sha256"]:
            raise ValueError("legacy caller parameter header changed")
        definitions = []
        for line in raw.decode().splitlines():
            tokens = line.split()
            if len(tokens) >= 3 and tokens[:2] == ["#define", "DIM"]:
                definitions.append(int(tokens[2], 0))
        if len(definitions) != 1 or definitions[0] <= 0:
            raise ValueError("legacy caller requires one literal positive header DIM")
        dimensions.add(definitions[0])
    if len(dimensions) != 1:
        raise ValueError("recipe parameter headers disagree on legacy caller dimension")
    dim = next(iter(dimensions))
    if host:
        from . import gemmini_codegen
        if gemmini_codegen.DIM != dim:
            raise ValueError("existing legacy caller dimension differs from selected target header")
    return {"schema": "pinned_legacy_whole_program_abi_v1", "dim": dim,
            "parameter_headers": headers, "candidate_storage_metadata_modified": False,
            "layout": "flatten leading dimensions to rows; pad both axes to DIM; row-major",
            "emitted_access_correspondence": "UNPROVEN; independent execution must check output"}


@dataclass(frozen=True)
class ShortProgramBuild:
    argv: tuple[str, ...]
    request_path: str
    pins_json: str

    @property
    def dependencies(self):
        return tuple(json.loads(Path(self.request_path).read_text())["build_service"]["dependencies"])

    @property
    def request_sha256(self):
        return _file_digest(self.request_path)

    @property
    def argv_sha256(self):
        return _digest(_json(list(self.argv)).encode())

    def revalidate(self):
        for path, expected in json.loads(self.pins_json).items():
            if _digest(Path(path).read_bytes()) != expected:
                raise ValueError("short build input/worker changed: " + path)

    def run(self, run_bounded, *, timeout_s):
        """Callback: run_bounded(argv, *, timeout_s) -> CompletedProcess.

        Use a closure over GlobalPerfExperiment.run_native_probe. Paths are
        obligations, not added mounts. The callback owns the whole process group.
        """
        if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
            raise ValueError("short build requires a positive deadline no larger than 60 seconds")
        started = time.monotonic()
        self.revalidate()
        result = run_bounded(list(self.argv), timeout_s=timeout_s)
        self.revalidate()
        if result.returncode:
            raise ValueError("bounded short build failed: " + str(result.stderr)[-4000:])
        receipt_path = Path(self.request_path).parent / "build_receipt.json"
        receipt = json.loads(_read(receipt_path, 1024*1024))
        if receipt.get("request_sha256") != _digest(Path(self.request_path).read_bytes()):
            raise ValueError("short build receipt belongs to another request")
        for name, pin in receipt["outputs"].items():
            path = receipt_path.parent / "build" / name
            if Path(name).name != name or _digest(path.read_bytes()) != pin:
                raise ValueError("short build output identity changed")
        receipt["bounded_callback_elapsed_seconds"] = time.monotonic() - started
        return receipt


def prepare_short_program_build(*, source_text, lowered_text, command_buffer,
                                logical_payloads, source_evidence, workdir,
                                python_executable, build_namespace_root=None, build_path_bindings=()):
    """Host-only preparation; no candidate-supplied evidence is an authorization.

    source_evidence must bind probe_source_sha256 and lowered_sha256. It is
    preserved as correspondence provenance, NOT promoted to numerical proof.
    Caller must include preparation in its total action budget.
    """
    if len(lowered_text.encode()) > MAX_LLVM_BYTES or len(source_text.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("short source/LLVM byte limit exceeded")
    if (source_evidence.get("probe_source_sha256") != _digest(source_text.encode())
            or source_evidence.get("lowered_sha256") != _digest(lowered_text.encode())):
        raise ValueError("host source evidence does not bind exact source and emitted LLVM")
    bounds = _source_bounds(source_text)
    legacy = command_buffer.get("params", {}).get("storage_encodings") is None
    build_service = _prepare_build_service(build_namespace_root, build_path_bindings) if legacy else None
    legacy_abi = _legacy_dimension(build_service, host=True) if legacy else None
    from merlin.targetgen.contract.build_service import load_build_package
    pure = load_build_package(Path(__file__).resolve().parent.parent/'build_support'/'__init__.py')
    _, storage = _inputs(command_buffer, logical_payloads,
        legacy_dim=legacy_abi["dim"] if legacy_abi else None, legacy_format=pure.format)
    # Reuse the structural pointer-entry check; this is not a compact ABI
    # transformation or a pointer-substitution equivalence claim.
    from .gemmini_compact_caller import _signature
    signature = _signature(lowered_text, {"compact_symbol": "gemmini_kernel",
        "compact_argument_count": len(command_buffer["kernel_abi"]["args"])})
    cb_bytes = _json(command_buffer).encode()
    if len(cb_bytes) > MAX_LLVM_BYTES or len(_json(source_evidence).encode()) > 128*1024:
        raise ValueError("short command-buffer/source evidence byte limit exceeded")
    interpreter = Path(python_executable)
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise ValueError("an already granted absolute Python interpreter is required")
    build_service = build_service or _prepare_build_service(build_namespace_root, build_path_bindings)
    work = Path(workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise ValueError("short build needs a fresh dedicated directory")
    files = {"source.mlir": source_text.encode(), "lowered.mlir": lowered_text.encode(),
             "command_buffer.json": cb_bytes,
             "worker.py": Path(__file__).read_bytes()}
    request = {"schema": "short_complete_program_build_v2", "source_evidence": source_evidence,
        "source_bounds": bounds, "entry_signature": signature, "storage": storage, "payloads": {},
        "files": {name: _digest(data) for name, data in files.items()}, "build_service": build_service}
    if legacy_abi is not None:
        request["legacy_abi"] = legacy_abi
    for i, (name, payload) in enumerate(sorted(logical_payloads.items())):
        filename = f"input_{i}.bin"
        files[filename] = payload
        request["payloads"][name] = filename
        request["files"][filename] = _digest(payload)
    files["request.json"] = _json(request).encode()
    for name, data in files.items():
        (work / name).write_bytes(data)
    pins = {str(work / name): _digest(data) for name, data in files.items()}
    pins[str(interpreter)] = _digest(interpreter.read_bytes())
    for dependency in build_service['dependencies']:
        pins[dependency['source']] = dependency['sha256']
    return ShortProgramBuild((str(interpreter), "-P", str(work / "worker.py"), str(work / "request.json")),
                             str(work / "request.json"), _json(pins))


def _prepare_build_service(namespace_root, path_bindings=()):
    """Host-only recipe/active-source resolution. Produces obligations, never mounts.

    The audited roots below are the existing default LLVM-to-object/build path.
    Import them in a fresh read-only process to collect their transitive loaded
    Python sources without including host registry modules loaded by preparation.
    Neither lowering functions nor tools are executed by this source discovery.
    """
    from merlin.common.paths import repo_root, merlin_dir
    from merlin.runtime.backends.base import harness_build_recipe
    from merlin.llvmlower import toolchain
    if os.environ.get('MERLIN_QUANT_FORMATS'):
        raise ValueError('short pure build forbids numeric format registry overlays')
    root = repo_root().resolve()
    namespace = root if namespace_root is None else Path(namespace_root).absolute()
    if namespace.resolve() != namespace or not namespace.is_dir():
        raise ValueError("build namespace must be an existing real project root")
    recipe = harness_build_recipe('gemmini')
    bindings = []
    for source, destination in path_bindings:
        source, destination = Path(source), Path(destination)
        if (not source.is_absolute() or not destination.is_absolute()
                or source.resolve() != source or destination.resolve() != destination
                or not source.is_dir() or not destination.is_dir()):
            raise ValueError('build path bindings require existing exact host-selected directories')
        bindings.append((source, destination))
    if len({source for source, _ in bindings}) != len(bindings):
        raise ValueError('build path bindings contain duplicate source roots')
    def destination_for(source, kind):
        # Bindings describe already granted build-tool/recipe aliases. They do
        # not mount anything and cannot rename Python or format-data grants.
        selected = [(len(base.parts), base, dest) for base, dest in bindings
                    if (source == base or base in source.parents) and kind not in {'python', 'format_data'}]
        if selected:
            _, base, destination = max(selected)
            return destination/source.relative_to(base)
        return namespace/source.relative_to(root) if source.is_relative_to(root) else source
    pure = Path(__file__).resolve().parent.parent/'build_support'/'__init__.py'
    roots = (
        'merlin.targetgen.contract.compile', 'merlin.targetgen.contract.build_service',
        'merlin.targetgen.contract.build_recipe', 'merlin.targetgen.contract.stack_usage',
        'merlin.targetgen.runtime_build',
        'merlin.targetgen.elf_lanes',
        'merlin.runtime.storage_binding', 'merlin.runtime.tensor',
        'merlin.runtime.commandbuffer', 'merlin.runtime.fp8_formats',
        'merlin.common.quant_formats', 'merlin.llvmlower.codegen', 'merlin.llvmlower.toolchain',
    )
    collector = '''import builtins, importlib, json, sys
from pathlib import Path
request=json.loads(sys.stdin.read())
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.startswith(('merlin.runtime.backends','merlin._oot_backends')) or name in ('merlin.runtime.reference','merlin.runtime.simulator','merlin.targetgen.build_cache'):
        raise ImportError('masked build dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
for name in request['modules']: importlib.import_module(name)
from merlin.targetgen.contract.build_service import load_build_package
load_build_package(Path(request['pure']))
paths=set()
for module in tuple(sys.modules.values()):
    filename=getattr(module,'__file__',None)
    if filename:
        path=Path(filename).absolute()
        if (path.is_relative_to(Path(request['root'])/'merlin'/'python') or path.is_relative_to(Path(request['pure']).parent)) and path.suffix=='.py': paths.add(str(path))
print(json.dumps(sorted(paths)))
'''
    found = subprocess.run([sys.executable, '-P', '-c', collector],
        input=_json({'modules': roots, 'pure': str(pure), 'root': str(root)}),
        env={**os.environ, 'PYTHONPATH': str(merlin_dir()/'python'), 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=15)
    if found.returncode:
        raise ValueError('pure build source discovery failed: '+found.stderr[-2000:])
    sources = [Path(name) for name in json.loads(found.stdout)]
    if not sources:
        raise ValueError('pure build source closure is empty')
    dependencies = []
    def bind(path, kind):
        source = Path(path).absolute()
        if not source.is_file():
            raise ValueError('missing build dependency: '+str(source))
        # Preserve existing tool aliases in argv; Python grants require real paths.
        if kind in {'python', 'format_data'} and (source.is_symlink() or source.resolve() != source):
            raise ValueError('pure Python build dependency has a linked path: '+str(source))
        destination = destination_for(source, kind)
        pin = _file_digest(source)
        # Only exact Python leaves may be supplied separately by the caller's
        # trusted dependency grant. Tools/C sources/data must already exist there.
        if kind not in {'python', 'format_data'} and (not destination.is_file() or _file_digest(destination) != pin):
            raise ValueError('existing build namespace lacks exact non-Python dependency: '+str(destination))
        record = {'source': str(source), 'destination': str(destination), 'sha256': pin, 'kind': kind}
        if record not in dependencies:
            dependencies.append(record)
        return str(destination)
    for source in sources:
        bind(source, 'python')
    # Only these host-selected canonical format leaves may require typed grants.
    # The enclosing host authorizes mounts; declaring them here grants nothing.
    registry_source = merlin_dir()/'schemas'/'quant_formats.registry.yaml'
    schema_source = merlin_dir()/'schemas'/'quant_format.schema.yaml'
    format_validation = _validate_format_data(registry_source, schema_source)
    format_paths = {}
    for role, source in (('numeric_format_registry', registry_source),
                         ('numeric_format_schema', schema_source)):
        destination = bind(source, 'format_data')
        record = next(item for item in dependencies if item['source'] == str(source))
        record['role'] = role
        format_paths[role] = {'source': str(source), 'destination': destination,
                              'sha256': record['sha256']}
    format_relation = {'schema': 'canonical_numeric_format_relation_v1',
        'registry': format_paths['numeric_format_registry'],
        'entry_schema': format_paths['numeric_format_schema'],
        'validation': format_validation, 'overlays_allowed': False}
    relation_sha = _digest(_json(format_relation).encode())
    for record in dependencies:
        if record['kind'] == 'format_data':
            record['format_data_relation_sha256'] = relation_sha
    tools = {'MERLIN_CLANG': bind(toolchain.clang(), 'tool'),
             'MERLIN_MLIR_TRANSLATE': bind(toolchain.mlir_translate(), 'tool')}
    def directory(path):
        source = Path(path).absolute()
        destination = destination_for(source, 'include')
        if not source.is_dir() or not destination.is_dir():
            raise ValueError('existing namespace lacks recipe include directory: '+str(destination))
        return str(destination)
    def recipe_record(mapped):
        transform = (lambda path,kind: bind(path,kind)) if mapped else (lambda path,kind: str(path))
        return {'compiler': transform(recipe.compiler,'tool'),
            'include_roots': [directory(path) if mapped else str(path) for path in recipe.include_roots],
            'support_sources': [transform(path,'support') for path in recipe.support_sources],
            'link_script': transform(recipe.link_script,'support'),
            'load_address': recipe.load_address, 'cflags': list(recipe.cflags), 'ldflags': list(recipe.ldflags),
            'kernel_stack_frame': recipe.require_kernel_stack_frame().record()}
    original_recipe, mapped_recipe = recipe_record(False), recipe_record(True)
    # A different spelling of an existing curated include view must retain the
    # actual header bytes, not just matching support C/assembly files.
    headers = sorted({path for include in recipe.include_roots for path in Path(include).rglob('*.h')})
    for header in headers:
        bind(header, 'header')
    pure_destination = namespace/pure.relative_to(root)
    return {'schema': 'host_resolved_pure_build_service_v1', 'target': 'gemmini',
        'source_root': str(root), 'namespace_root': str(namespace),
        'recipe': mapped_recipe, 'host_recipe': original_recipe,
        'host_recipe_sha256': _digest(_json(original_recipe).encode()),
        'worker_recipe_sha256': _digest(_json(mapped_recipe).encode()),
        'existing_path_bindings': [[str(source), str(destination)] for source, destination in bindings],
        'pure_package_init': str(pure_destination), 'dependencies': dependencies,
        'format_data_relation': format_relation,
        'tool_environment': tools, 'active_source_roots': list(roots),
        'lowering_route': 'verified LLVM/Builtin only; direct pinned mlir-translate',
        'scope': 'exact active Python modules and direct tools/support/data; external headers/tool runtime remain enclosing sandbox obligations'}


def _validate_format_data(registry_path, schema_path):
    """Fresh canonical data validation, separate from cached registry/schema objects."""
    if os.environ.get('MERLIN_QUANT_FORMATS'):
        raise ValueError('short pure build forbids numeric format registry overlays')
    from merlin.common import quant_formats
    import yaml
    registry_bytes, schema_bytes = _read(registry_path, 256*1024), _read(schema_path, 64*1024)
    registry, schema = yaml.safe_load(registry_bytes), yaml.safe_load(schema_bytes)
    required = schema.get('required_top_level_fields') if isinstance(schema, dict) else None
    if (not isinstance(required, list) or not required
            or any(not isinstance(field, str) or not field for field in required)
            or len(set(required)) != len(required)):
        raise ValueError('numeric format schema lacks explicit required fields')
    formats = registry.get('formats') if isinstance(registry, dict) else None
    if (not isinstance(registry, dict) or type(registry.get('version')) is not int
            or registry['version'] <= 0 or not isinstance(formats, dict) or not formats):
        raise ValueError('canonical numeric registry must declare version and nonempty formats')
    for name, fields in formats.items():
        if not isinstance(name, str) or not name or not isinstance(fields, dict):
            raise ValueError('numeric format registry entries must be named mappings')
        missing = set(required)-set({'name': name, **fields})
        if missing:
            raise ValueError('numeric format entry lacks freshly required schema fields: '+','.join(sorted(missing)))
        quant_formats._validate_entry(name, fields)
    if _file_digest(registry_path) != _digest(registry_bytes) or _file_digest(schema_path) != _digest(schema_bytes):
        raise ValueError('numeric format data changed during validation')
    return {'status': 'validated', 'entry_count': len(formats), 'required_fields': required,
        'registry_sha256': _digest(registry_bytes), 'schema_sha256': _digest(schema_bytes),
        'semantics': 'fresh schema presence plus canonical quant_formats._validate_entry'}


def _worker(request_path):
    path = Path(request_path).resolve()
    request_bytes = _read(path, 1024*1024)
    request = json.loads(request_bytes)
    if request.get("schema") != "short_complete_program_build_v2":
        raise ValueError("unknown short build request")
    work = path.parent
    specification = request.get('build_service', {})
    if specification.get('schema') != 'host_resolved_pure_build_service_v1':
        raise ValueError('missing host-resolved pure build service')
    producers, build_pins = {}, {}
    for item in specification['dependencies']:
        destination = Path(item['destination'])
        if (not destination.is_absolute() or not destination.is_file()
                or _file_digest(destination) != item['sha256']):
            raise ValueError('missing or stale pure build dependency: '+str(destination))
        if item['kind'] == 'python':
            if destination.is_symlink() or destination.resolve() != destination:
                raise ValueError('linked pure build source')
            producers[str(destination)] = item['sha256']
        else:
            build_pins[str(destination)] = item['sha256']
    if not producers or _digest(_json(specification['recipe']).encode()) != specification['worker_recipe_sha256']:
        raise ValueError('stale pure build recipe or empty source closure')
    sys.path.insert(0, str(Path(specification['namespace_root'])/'merlin'/'python'))
    relation = specification.get('format_data_relation', {})
    if relation.get('schema') != 'canonical_numeric_format_relation_v1' or relation.get('overlays_allowed') is not False:
        raise ValueError('missing canonical numeric format schema relation')
    format_items = [item for item in specification['dependencies'] if item['kind'] == 'format_data']
    if len(format_items) != 2:
        raise ValueError('pure build requires exactly the canonical numeric registry and schema')
    canonical_roles = {'numeric_format_registry': 'quant_formats.registry.yaml',
                       'numeric_format_schema': 'quant_format.schema.yaml'}
    if {item.get('role') for item in format_items} != set(canonical_roles):
        raise ValueError('unknown numeric format data role')
    for item in format_items:
        expected = Path(specification['namespace_root'])/'merlin'/'schemas'/canonical_roles[item['role']]
        paired = relation['registry' if item['role'] == 'numeric_format_registry' else 'entry_schema']
        if (item['destination'] != str(expected) or expected.is_symlink()
                or item.get('format_data_relation_sha256') != _digest(_json(relation).encode())
                or any(paired.get(key) != item[key] for key in ('source','destination','sha256'))):
            raise ValueError('numeric format dependency differs from canonical paired schema relation')
    validation = _validate_format_data(relation['registry']['destination'], relation['entry_schema']['destination'])
    if validation != relation['validation']:
        raise ValueError('mounted numeric format data validation differs from host preparation')
    tool_env = specification['tool_environment']
    os.environ.update(MERLIN_CACHE_STATE='warm', MERLIN_HW_COUNTERS='0', MERLIN_ELF_BUILD_CACHE='0',
        MERLIN_CLANG=tool_env['MERLIN_CLANG'], MERLIN_MLIR_TRANSLATE=tool_env['MERLIN_MLIR_TRANSLATE'])
    sys.dont_write_bytecode = True
    contents = {}
    for name, expected in request["files"].items():
        if Path(name).name != name:
            raise ValueError("request files must stay in dedicated scratch")
        contents[name] = _read(work / name, MAX_LLVM_BYTES)
        if _digest(contents[name]) != expected:
            raise ValueError("stale short build input: " + name)
    cb = json.loads(contents["command_buffer.json"])
    from merlin.targetgen.contract.compile import compile_lowered_to_elf
    from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
    from merlin.targetgen.contract.build_service import BuildOnlyService, load_build_package
    row = specification['recipe']
    recipe = HarnessBuildRecipe(compiler=Path(row['compiler']),
        include_roots=tuple(map(Path,row['include_roots'])), support_sources=tuple(map(Path,row['support_sources'])),
        link_script=Path(row['link_script']), load_address=row['load_address'],
        cflags=tuple(row['cflags']), ldflags=tuple(row['ldflags']),
        kernel_stack_frame=KernelStackFramePolicy(**row['kernel_stack_frame']))
    pure = load_build_package(Path(specification['pure_package_init']))
    legacy_abi = request.get("legacy_abi")
    if (cb.get("params", {}).get("storage_encodings") is None) != (legacy_abi is not None):
        raise ValueError("short caller legacy ABI mode disagrees with original command buffer")
    if legacy_abi is not None and _legacy_dimension(specification) != legacy_abi:
        raise ValueError("mounted legacy caller ABI differs from host-pinned header")
    inputs, storage = _inputs(cb, {name: contents[filename] for name, filename in request["payloads"].items()},
        legacy_dim=legacy_abi["dim"] if legacy_abi else None, legacy_format=pure.format)
    if storage != request["storage"]:
        raise ValueError("short caller storage differs from host preparation")
    def render(cb, **kwargs):
        return pure.render_whole_program(cb, legacy_dim=legacy_abi["dim"] if legacy_abi else None, **kwargs)
    service = BuildOnlyService(target=specification['target'], recipe=recipe,
        renderer=render, source_pins=tuple(producers.items()))
    recipe_evidence = {**row, 'host_recipe_sha256': specification['host_recipe_sha256'],
        'worker_recipe_sha256': specification['worker_recipe_sha256']}
    build = work / "build"
    if build.exists():
        raise ValueError("short build output directory already exists")
    started = time.monotonic()
    elf = compile_lowered_to_elf(cb, contents["lowered.mlir"].decode(), build,
                               target="gemmini", inputs=inputs, _build_service=service)
    for name, expected in request["files"].items():
        if _digest(_read(work / name, MAX_LLVM_BYTES)) != expected:
            raise ValueError("short build input changed during compilation")
    if any(_file_digest(path) != expected for path, expected in build_pins.items()):
        raise ValueError("short build tool/recipe input changed during compilation")
    if any(_file_digest(path) != expected for path, expected in producers.items()):
        raise ValueError("short build producer changed during compilation")
    active_shared = Path(specification['namespace_root'])/'merlin'/'python'
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith(('merlin.runtime.backends', 'merlin._oot_backends')) or module_name in (
                'merlin.runtime.reference', 'merlin.runtime.simulator', 'merlin.targetgen.build_cache'):
            raise ValueError('build-only worker imported masked runtime service: '+module_name)
        filename = getattr(module, '__file__', None)
        if filename:
            source = Path(filename).absolute()
            if source.is_relative_to(active_shared) and source.suffix == '.py' and str(source) not in producers:
                raise ValueError('unbound active build Python dependency: '+str(source))
    output_names = (elf.name, "kernel.o", "kernel.ll", "kernel.su",
                    "kernel.stack_frame.json", "harness.c")
    receipt = {"schema": "short_complete_program_build_receipt_v1", "status": "built_not_executed",
        "request_sha256": _digest(request_bytes), "source_evidence": request["source_evidence"],
        "source_bounds": request["source_bounds"], "storage": storage,
        "legacy_abi": legacy_abi,
        "entry_signature": request["entry_signature"], "primary_build_pins": build_pins,
        "producer_pins": producers,
        "build_service_schema": specification['schema'],
        "format_data_relation": relation,
        "worker_backend_registry_used": False,
        "recipe": recipe_evidence, "complete_build_dependency_closure": "enclosing host sandbox obligation",
        "outputs": {name: _digest((build/name).read_bytes()) for name in output_names},
        "elf_path": str(elf), "build_elapsed_seconds": time.monotonic()-started,
        "warm_invocations_emitted": 1, "measured_invocations_emitted": 1,
        "measurement_scope": "complete_short_program_including_host_and_device_work",
        "target_executed": False, "full_model_execution": False,
        "numerical_correspondence": "UNPROVEN", "runtime_admission": False,
        "cache_used": False, "process_group_owner": "supplied bounded host callback"}
    (work / "build_receipt.json").write_text(_json(receipt)+"\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("expected one host-prepared request")
    _worker(sys.argv[1])
