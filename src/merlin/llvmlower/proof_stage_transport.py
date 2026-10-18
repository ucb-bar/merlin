"""Hold-only IPC for the actual ordinary native module before bufferization.

The child retains its Module and Context. The parent may acknowledge a checked
snapshot; it cannot return IR or editing instructions. Phase order comes from an
independently authored plan, never from splitting a rendered pass pipeline.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REQUEST_SCHEMA = "merlin.lean.native_hold.request.v1"
ACK_SCHEMA = "merlin.lean.native_hold.ack.v1"
TOKEN = "OK lean_native_tensor_gate"


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def fingerprint(path: Path) -> dict:
    path = Path(path).absolute()
    return {
        "invocation_path": str(path),
        "resolved_path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# Also executed in the actual compiler child. The invocation path matters for
# virtual environments even when two interpreters resolve to the same ELF.
IDENTITY_SOURCE = r"""
def _pg_identity():
    import hashlib, importlib.metadata, sys
    from pathlib import Path
    from torch_mlir._mlir_libs import _mlir
    def file(path):
        path = Path(path).absolute()
        return {"invocation_path": str(path), "resolved_path": str(path.resolve(strict=True)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"compiler_python": file(sys.executable), "mlir_binding": file(_mlir.__file__),
            "mlir_capi": file(Path(_mlir.__file__).with_name("libTorchMLIRAggregateCAPI.so")),
            "torch_mlir_version": importlib.metadata.version("torch-mlir")}
"""


def identity(compiler_python: Path) -> dict:
    """Measure the interpreter and loaded native bindings actually selected for execution."""
    result = subprocess.run(
        [str(compiler_python), "-c", IDENTITY_SOURCE + "\nimport json; print(json.dumps(_pg_identity()))"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@dataclass
class HoldAcknowledgment:
    """One parent proof result; the transport binds it to its exact pending request."""

    request_sha256: str
    report: Path
    chain: Path
    _used: bool = False

    def claim(self, request: dict) -> dict:
        if self._used:
            raise ValueError("native proof acknowledgment was already consumed")
        self._used = True
        report = json.loads(self.report.read_bytes())
        chain = json.loads(self.chain.read_bytes())
        if report.get("verdict") != "verified" or chain.get("verdict") != "verified":
            raise ValueError("native acknowledgment requires verified report and strict chain")
        return {
            "schema": ACK_SCHEMA,
            "run": request["run"],
            "plan_sha256": request["plan_sha256"],
            "request_sha256": self.request_sha256,
            "held_sha256": request["held_sha256"],
            "binding_sha256": request["binding_sha256"],
            "verdict": "verified",
            "report": fingerprint(self.report),
            "chain": fingerprint(self.chain),
        }


NATIVE = r"""
_PG_ORDINARY_RUN_STAGES = _run_stages


def _pg_canonical(value):
    import json
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def _pg_sha(payload):
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _pg_text(module):
    payload = module.operation.get_asm(print_generic_op_form=True).encode('utf-8')
    if len(payload) > _PG_MAX_BYTES:
        raise ValueError('native phase snapshot exceeds explicit byte quota')
    return payload


def _pg_domain(module):
    # Index conversion flags bind the suffix, not earlier optimizer index
    # interpretation (retained S6). The initial 32-bit domain is index-free;
    # width64 still requires the actual strict prefix checker to qualify each use.
    def has_index(t):
        if _PG_BINDING['index_bits'] == 64:
            return False
        if ir.IndexType.isinstance(t):
            return True
        if ir.RankedTensorType.isinstance(t):
            return has_index(ir.RankedTensorType(t).element_type)
        if ir.FunctionType.isinstance(t):
            ft = ir.FunctionType(t)
            return any(has_index(x) for x in (*ft.inputs, *ft.results))
        return False
    def visit(op):
        if any(has_index(v.type) for v in (*op.operands, *op.results)):
            raise ValueError('initial native proof gate refuses index-typed SSA values')
        for a in op.attributes:
            if ir.TypeAttr.isinstance(a.attr) and has_index(ir.TypeAttr(a.attr).value):
                raise ValueError('initial native proof gate refuses index-typed function ABI')
        for region in op.regions:
            for block in region.blocks:
                if any(has_index(a.type) for a in block.arguments):
                    raise ValueError('initial native proof gate refuses index-typed block arguments')
                for child in block.operations:
                    visit(child.operation)
    visit(module.operation)
    if b'dense_resource' in _pg_text(module):
        raise ValueError('initial native proof gate refuses external resource attributes')


def _pg_record(name, entry, before, after):
    import json
    from pathlib import Path
    root = Path(_PG_DIRECTORY) / 'stages'
    ordinal = len(_PG_RECORDS)
    reserved = root / '.reserved'
    reserved.mkdir(parents=True, exist_ok=True)
    (reserved / f'{ordinal:06d}').mkdir()
    writing = root / f'.writing-{ordinal:04d}'
    writing.mkdir()
    (writing / 'before.mlir').write_bytes(before)
    (writing / 'after.mlir').write_bytes(after)
    meta = {'schema': 'merlin.lean.stage.v1', 'run': _PG_RUN, 'boundary': name,
            'pass': entry, 'before_sha256': _pg_sha(before), 'after_sha256': _pg_sha(after)}
    (writing / 'meta.json').write_bytes(_pg_canonical(meta))
    target = root / f'{ordinal:04d}_{name}'
    writing.rename(target)
    _PG_RECORDS.append(name)


def _pg_parse(path, ctx):
    from pathlib import Path
    if _pg_identity() != _PG_BINDING['consumer']:
        raise ValueError('native compiler identity differs before parsing')
    payload = Path(path).read_bytes()
    if len(payload) > _PG_MAX_BYTES or _pg_sha(payload) != _PG_BINDING['input_sha256']:
        raise ValueError('native parse input differs from the verified preprocessing endpoint')
    module = ir.Module.parse(payload.decode('utf-8'), ctx)
    _pg_domain(module)
    _pg_record('native-parse', 'torch_mlir.ir.Module.parse', payload, _pg_text(module))
    return module


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    import json, time
    from pathlib import Path
    if pipeline != _PG_PIPELINE or erase or mid or late or post_openmp or pre_generalize:
        raise ValueError('native gate consumer differs from the authored ordinary plan')
    if _PG_RECORDS != ['native-parse']:
        raise ValueError('native gate requires one actual parsed module and one barrier')
    for name, entry in _PG_PHASES:
        before = _pg_text(module)
        _PG_ORDINARY_RUN_STAGES(ctx, module, entry, 0)
        _pg_domain(module)
        _pg_record(name, entry, before, _pg_text(module))
    held = _pg_text(module)
    owner = Path(_PG_DIRECTORY)
    current = owner / 'current.mlir'
    current.write_bytes(held)
    handoff = {'schema': 'merlin.lean.handoff.v1', 'path': str(current.resolve()),
               'sha256': _pg_sha(held), 'bytes': len(held)}
    (owner / 'stages' / 'handoff.json').write_bytes(_pg_canonical(handoff))
    request = {'schema': 'merlin.lean.native_hold.request.v1', 'run': _PG_RUN,
               'plan_sha256': _PG_PLAN_SHA, 'held_sha256': _pg_sha(held), 'bytes': len(held),
               'binding_sha256': _pg_sha(_pg_canonical(_PG_BINDING)),
               'boundary_count': len(_PG_RECORDS), 'consumer': _pg_identity()}
    encoded = _pg_canonical(request)
    pending = owner / 'request.pending'
    pending.write_bytes(encoded)
    pending.replace(owner / 'request.json')
    deadline = time.monotonic() + _PG_TIMEOUT
    while not (owner / 'ack.json').is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('native tensor proof acknowledgment timed out')
        time.sleep(0.01)
    response = (owner / 'ack.json').read_bytes()
    if len(response) > _PG_MAX_BYTES:
        raise ValueError('native acknowledgment exceeds explicit byte quota')
    ack = json.loads(response)
    keys = {'schema', 'run', 'plan_sha256', 'request_sha256', 'held_sha256',
            'binding_sha256', 'verdict', 'report', 'chain'}
    if (set(ack) != keys or ack['schema'] != 'merlin.lean.native_hold.ack.v1'
        or ack['run'] != _PG_RUN or ack['plan_sha256'] != _PG_PLAN_SHA
        or ack['request_sha256'] != _pg_sha(encoded) or ack['held_sha256'] != _pg_sha(held)
        or ack['binding_sha256'] != request['binding_sha256'] or ack['verdict'] != 'verified'):
        raise ValueError('stale or mismatched native proof acknowledgment')
    for key in ('report', 'chain'):
        ref = ack[key]
        data = Path(ref['invocation_path']).read_bytes()
        if _pg_sha(data) != ref['sha256'] or json.loads(data).get('verdict') != 'verified':
            raise ValueError('acknowledged native proof evidence changed')
    if _pg_text(module) != held or _pg_identity() != request['consumer']:
        raise ValueError('held native module or consumer changed while proof was pending')
    if not module.operation.verify():
        raise ValueError('held native module failed verification before bufferization')
    # No response IR is accepted. Resume on this exact Module and Context.
    _PG_ORDINARY_RUN_STAGES(ctx, module, _PG_SUFFIX, 0)
    print('OK lean_native_tensor_gate', _PG_RUN, _PG_PLAN_SHA, _pg_sha(held), _pg_sha(response))
"""


def bind_runner(source: str, *, directory: Path, run: str, plan, binding: dict, max_bytes: int, timeout: int) -> str:
    """Instrument only the exact ordinary parse/staged call; refuse other runner shapes."""
    tree = ast.parse(source)
    parses = [
        n
        for owner in tree.body
        if isinstance(owner, ast.With)
        for n in ast.walk(owner)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "parse"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "Module"
        and isinstance(n.func.value.value, ast.Name)
        and n.func.value.value.id == "ir"
    ]
    calls = [
        n
        for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "_run_stages"
    ]
    if len(parses) != 1 or len(calls) != 1:
        raise ValueError("native proof transport requires one ordinary parse and staged call")
    parsed = parses[0]
    if ast.dump(parsed) != ast.dump(ast.parse("ir.Module.parse(f.read(), ctx)", mode="eval").body):
        raise ValueError("native proof transport refuses an unrecognized parse consumer")
    owner = next((n for n in tree.body if isinstance(n, ast.With) and parsed in ast.walk(n)), None)
    if owner is None or len(owner.body) != 1 or not isinstance(owner.body[0], ast.Assign):
        raise ValueError("native proof transport requires the ordinary isolated parse statement")
    statement = owner.body[0]
    if (
        len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or statement.targets[0].id != "module"
    ):
        raise ValueError("native proof transport refuses a replaced native module owner")
    config = {
        "_PG_DIRECTORY": str(Path(directory).resolve()),
        "_PG_RUN": run,
        "_PG_PLAN_SHA": plan.sha256,
        "_PG_PHASES": tuple((b.name, b.pass_entry) for b in plan.prefix),
        "_PG_PIPELINE": plan.pipeline(),
        "_PG_SUFFIX": ",".join(plan.suffix),
        "_PG_BINDING": binding,
        "_PG_MAX_BYTES": max_bytes,
        "_PG_TIMEOUT": timeout,
        "_PG_RECORDS": [],
    }
    header = "\n".join(f"{name} = {value!r}" for name, value in config.items())
    lines = source.splitlines(keepends=True)
    start, end = owner.lineno - 1, owner.end_lineno
    return (
        "".join(lines[:start])
        + header
        + "\n"
        + IDENTITY_SOURCE
        + NATIVE
        + "\nmodule = _pg_parse(src_path, ctx)\n"
        + "".join(lines[end:])
    )


def run_command(command, *, directory: Path, callback, run: str, plan, binding: dict, max_bytes: int, timeout: int):
    """Serve exactly one pending proof on the parent thread; terminate only our child on failure."""
    directory = Path(directory)
    request, ack = directory / "request.json", directory / "ack.json"
    if any(directory.iterdir()):
        raise ValueError("native proof transport requires a fresh private owner")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    served = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                break
            except subprocess.TimeoutExpired:
                if request.is_file() and not served:
                    raw = request.read_bytes()
                    if len(raw) > max_bytes:
                        raise ValueError("native hold request exceeds explicit byte quota")
                    descriptor = json.loads(raw)
                    current = (directory / "current.mlir").read_bytes()
                    expected = {
                        "schema": REQUEST_SCHEMA,
                        "run": run,
                        "plan_sha256": plan.sha256,
                        "held_sha256": hashlib.sha256(current).hexdigest(),
                        "bytes": len(current),
                        "binding_sha256": hashlib.sha256(canonical(binding)).hexdigest(),
                        "boundary_count": len(plan.boundaries.boundaries),
                        "consumer": binding["consumer"],
                    }
                    if len(current) > max_bytes or descriptor != expected:
                        raise ValueError("native hold request differs from actual source, plan or consumer")
                    proof = callback(descriptor, directory)
                    digest = hashlib.sha256(raw).hexdigest()
                    if type(proof) is not HoldAcknowledgment or proof.request_sha256 != digest:
                        raise ValueError("native proof callback returned a stale acknowledgment")
                    encoded = canonical(proof.claim(descriptor))
                    if len(encoded) > max_bytes or request.read_bytes() != raw:
                        raise ValueError("native acknowledgment quota or pending request identity changed")
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout)
                    temporary = directory / "ack.pending"
                    temporary.write_bytes(encoded)
                    temporary.replace(ack)
                    served = True
        if process.returncode == 0:
            if not served:
                raise ValueError("native consumer succeeded without requesting its tensor proof barrier")
            expected_token = (
                f"{TOKEN} {run} {plan.sha256} {descriptor['held_sha256']} {hashlib.sha256(encoded).hexdigest()}"
            )
            if [x for x in stdout.splitlines() if x.startswith(TOKEN)] != [expected_token]:
                raise ValueError("native proof barrier completion is absent or inconsistent")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
