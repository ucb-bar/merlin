"""The non-contraction vectorize lever must refuse a SUB-BYTE element type, or it miscompiles.

THE DEFECT, measured on small_llama int8 (host lowering, arms behind ``linalg-specialize-generic-ops``).
The lever tiles an all-parallel ``linalg.generic`` and writes each tile with a
``vector.transfer_write``. For an element narrower than a byte the two sides of that write disagree
about the layout of the SAME buffer: LLVM stores a ``vector<8xi1>`` PACKED -- eight lanes in one byte
-- while the ``memref`` the tile belongs to is addressed one byte per element, which is how every
scalar consumer reads it. The model's causal mask is a ``tensor<8x8xi1>``::

    %3147 = call ptr @malloc(i64 128)                    ; 64 elements, ONE BYTE EACH
    %3163 = getelementptr i1, ptr %3152, i64 %3157       ; %3157 = row * 8
    store <8 x i1> %3162, ptr %3164, align 1             ; ...writes ONE byte
    ...
    %3204 = getelementptr inbounds nuw i1, ptr %3152, i64 %3203
    %3205 = load i1, ptr %3204, align 1                  ; scalar consumer: one byte per element
    %3208 = select i1 %3205, float 0xFFF0000000000000, float %3207

Eight of the sixty-four bytes are written; the other fifty-six are read back UNINITIALISED, straight
into the attention mask's ``select``. MEASURED: cos 0.968247 / rel 0.46352 against a baseline
0.999966 / 0.00836 -- and, from the SAME shared object, two different answers
(``756d00f36c43`` and ``d9d8a01aa32e``) depending only on the host process's memory layout.

Both directions are unsound and both are pinned below: a vectorized WRITE of a sub-byte destination,
and a vectorized READ of a sub-byte input that a scalar loop wrote. Under the arms' current
(pre-specialization) placement the whole producer/consumer chain happened to be vectorized, so the
packed store and a packed load agreed and the output was bit-identical to the baseline. That is the
hazard NOT FIRING, not the hazard being absent, which is why every execution test here runs BOTH
placements.

The fix is a refusal: the arms match ``merlin.vec_r{rank}`` AND an attribute that byte-addressable-
element matchers annotate, so an op whose destination or any input is narrower than a byte is left to
the scalar loop emitter. Refusing a shape is the acceptable outcome; a shape-dependent wrong answer is
not.

Deliberately NOT asserted here: that the lever is a speedup. That is a board measurement.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import artifacts_dir, repo_root
from merlin.llvmlower import lower as _lower_mod  # noqa: F401  (registers runner-gated features)
from merlin.llvmlower import impr_features as impr
from merlin.llvmlower import pipeline as P
from merlin.llvmlower.toolchain import available as _toolchain_available

_needs_m2m = pytest.mark.skipif(not _toolchain_available(),
                                reason="m2m venv / clang not configured")

VEC = impr.VEC_NONCONTRACTION_NAME

#: The WRITE hazard, minimal: a tagged generic whose destination is ``tensor<8x8xi1>``, consumed by a
#: generic the tagging predicate did not tag (so it stays a scalar loop) -- exactly the shape of the
#: causal mask and its ``select`` in small_llama.
MASK_WRITE = """
#p = affine_map<(d0, d1) -> (d0, d1)>
func.func @forward(%a: tensor<8x8xi32>, %x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %z = arith.constant 0 : i32
  %neg = arith.constant -1.000000e+00 : f32
  %m = tensor.empty() : tensor<8x8xi1>
  %mask = linalg.generic {indexing_maps = [#p, #p], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<8x8xi32>) outs(%m : tensor<8x8xi1>) attrs = {merlin.vec_r2} {
  ^bb0(%in: i32, %out: i1):
    %c = arith.cmpi sge, %in, %z : i32
    linalg.yield %c : i1
  } -> tensor<8x8xi1>
  %e = tensor.empty() : tensor<8x8xf32>
  %r = linalg.generic {indexing_maps = [#p, #p, #p], iterator_types = ["parallel", "parallel"]}
      ins(%mask, %x : tensor<8x8xi1>, tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%mi: i1, %xi: f32, %out: f32):
    %s = arith.select %mi, %xi, %neg : f32
    linalg.yield %s : f32
  } -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}
"""

#: The READ hazard, minimal: the same pair with the tags swapped. The mask is written by a scalar
#: loop (one byte per element) and read by a vectorized consumer (``load <8 x i1>``, one byte for
#: eight lanes). Checking only the destination would let this one through.
MASK_READ = (MASK_WRITE
             .replace("outs(%m : tensor<8x8xi1>) attrs = {merlin.vec_r2}",
                      "outs(%m : tensor<8x8xi1>)")
             .replace("outs(%e : tensor<8x8xf32>) {",
                      "outs(%e : tensor<8x8xf32>) attrs = {merlin.vec_r2} {"))

#: The POSITIVE CONTROL: same shape, no sub-byte tensor anywhere. The refusal must not swallow this
#: one, or every assertion above is satisfied by a lever that does nothing.
BYTEWISE_OK = """
#p = affine_map<(d0, d1) -> (d0, d1)>
func.func @forward(%a: tensor<8x8xi32>, %x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %e = tensor.empty() : tensor<8x8xf32>
  %r = linalg.generic {indexing_maps = [#p, #p, #p], iterator_types = ["parallel", "parallel"]}
      ins(%a, %x : tensor<8x8xi32>, tensor<8x8xf32>) outs(%e : tensor<8x8xf32>)
      attrs = {merlin.vec_r2} {
  ^bb0(%ai: i32, %xi: f32, %out: f32):
    %f = arith.sitofp %ai : i32 to f32
    %s = arith.addf %f, %xi : f32
    linalg.yield %s : f32
  } -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}
"""


# ---------------------------------------------------------------------------------------------
# 1. the schedule -- cheap, always runs
# ---------------------------------------------------------------------------------------------

def _armed() -> str:
    return impr.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, impr.normalize([VEC]))


def test_every_arm_is_gated_on_the_bytewise_attribute():
    """An arm that matches the rank tag ALONE vectorizes the sub-byte op. All three must carry the
    refusal, not one or two of them."""
    armed = _armed()
    for rank in (2, 3, 4):
        line = [ln for ln in armed.splitlines()
                if f"merlin.vec_r{rank}" in ln and "structured.match" in ln]
        assert len(line) == 1, (rank, line)
        assert impr.VEC_BYTEWISE_ATTR in line[0], line[0]


def test_the_matchers_are_spliced_and_run_before_the_arms():
    """The annotation has to exist before an arm can match on it."""
    armed = _armed()
    prefix = impr._vec_bytewise_matcher_prefix(P.RVV_TRANSFORM_SCHEDULE)
    assert prefix is not None
    for n in range(impr.VEC_BYTEWISE_MAX_INPUTS + 1):
        assert f"transform.named_sequence @{prefix}{n}(" in armed
        assert f"transform.collect_matching @{prefix}{n} in %arg0" in armed
    assert armed.index("transform.annotate") < armed.index("merlin.vec_r2")


def test_the_check_is_on_the_destination_and_on_every_input():
    """Checking only the destination leaves the READ hazard: a vectorized consumer of a sub-byte
    tensor a scalar loop wrote."""
    armed = _armed()
    assert "match.structured.init %s[0]" in armed
    for k in range(impr.VEC_BYTEWISE_MAX_INPUTS):
        assert f"match.structured.input %s[{k}]" in armed
    # ...and the arity is PINNED, so an input past the enumerated ones cannot go unchecked.
    assert "match.structured.num_inputs" in armed
    assert "match.structured.num_inits" in armed


def test_the_two_libraries_get_distinct_matcher_symbols():
    """``transform-preload-library`` merges the pre-specialization library and the package schedule
    into ONE module; a fixed symbol name would collide there and neither would load."""
    feats = impr.normalize([VEC])
    pre = P.vec_pre_schedule(feats)
    assert pre is not None
    main = _armed()
    pre_syms = {ln.split("@", 1)[1].split("(", 1)[0]
                for ln in pre.splitlines() if "transform.named_sequence @" in ln}
    main_syms = {ln.split("@", 1)[1].split("(", 1)[0]
                 for ln in main.splitlines() if "transform.named_sequence @" in ln}
    assert pre_syms and main_syms
    assert not (pre_syms & main_syms), sorted(pre_syms & main_syms)


def test_a_schedule_with_no_module_to_hold_the_matchers_gets_no_arms():
    """FAIL CLOSED. Arming without the refusal is the miscompile; a lever that stayed off is
    recoverable, a lever that is silently wrong is not."""
    headless = "\n".join(ln for ln in P.RVV_TRANSFORM_SCHEDULE.splitlines()
                         if not ln.strip().startswith("module"))
    out = impr._splice_vec_rank_arms(headless)
    assert out == headless
    assert "merlin.vec_r" not in out


def test_the_baseline_is_untouched():
    """Default-off stays default-off: none of this reaches a build that named no feature."""
    assert impr.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE
    assert impr.VEC_BYTEWISE_ATTR not in P.RVV_TRANSFORM_SCHEDULE
    assert P.vec_pre_schedule(frozenset()) is None


def test_splicing_is_idempotent():
    """``apply_schedule`` runs on a package schedule that may already carry the arms."""
    armed = _armed()
    assert impr.apply_schedule(armed, impr.normalize([VEC])) == armed


def test_the_minimum_width_is_the_byte_not_a_tuning_choice():
    """8 bits is where a ``vector<NxT>`` and a ``memref<...xT>`` start agreeing about layout. It is
    not a knob, and a lower value re-opens the miscompile."""
    assert impr.VEC_BYTEWISE_MIN_BITS == 8
    assert f"constant {impr.VEC_BYTEWISE_MIN_BITS} : i64" in _armed()


# ---------------------------------------------------------------------------------------------
# 2. the emitted code -- is the packed store gone, and is the lever still doing its job?
# ---------------------------------------------------------------------------------------------

def _lower(text: str, features, tmp_path, **kw) -> str:
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual
    ciface, _ = preprocess_text_textual(text)
    work = Path(tempfile.mkdtemp(prefix="subbyte_", dir=str(tmp_path)))
    return P.lower_to_llvm_ir(ciface, workdir=work, vectorize=True,
                              features=impr.normalize(features), **kw)


def _i1_vector_memory_ops(ll_text: str) -> list[str]:
    """Every LLVM load/store whose value type is a VECTOR OF i1.

    Parsed structurally (``split``/``startswith``, no pattern matching). This is the emitted
    signature of the defect: a `<N x i1>` moved to or from memory occupies N BITS, while the buffer
    it names is addressed one byte per element by every scalar access to it.

    It is deliberately used COMPARATIVELY (lever vs the baseline arm, same module) rather than as an
    absolute count. A `memref` whose ELEMENT TYPE is itself `vector<Nxi1>` -- what
    ``convert-vector-to-scf`` allocates for a mask scratch -- is accessed as `<N x i1>` on BOTH sides
    and so is sound, and opaque pointers make the two indistinguishable from the access alone
    (MEASURED: the deepjscc baseline carries 27 such loads with the lever off). What is never sound
    is the lever ADDING one that the baseline did not have.
    """
    out: list[str] = []
    for line in ll_text.splitlines():
        stripped = line.strip()
        body = stripped.split("=", 1)[1].strip() if "=" in stripped else stripped
        for kind in ("store <", "load <"):
            if not body.startswith(kind):
                continue
            if body[len(kind):].split(">", 1)[0].endswith("x i1"):
                out.append(stripped)
    return out


@pytest.mark.parametrize("name,src", [("write", MASK_WRITE), ("read", MASK_READ)],
                         ids=["write", "read"])
@_needs_m2m
def test_a_sub_byte_tensor_is_never_written_or_read_as_a_packed_vector(name, src, tmp_path,
                                                                      monkeypatch):
    """BOTH placements. The arms' position decides which ops they see; it must not decide whether
    the emitted code is sound."""
    for after_specialize in (False, True):
        if after_specialize:
            monkeypatch.setenv("MERLIN_VEC_AFTER_SPECIALIZE", "1")
        else:
            monkeypatch.delenv("MERLIN_VEC_AFTER_SPECIALIZE", raising=False)
        off = _lower(src, frozenset(), tmp_path)
        on = _lower(src, [VEC], tmp_path)
        assert _i1_vector_memory_ops(off) == [], (name, after_specialize)
        assert _i1_vector_memory_ops(on) == [], (name, after_specialize,
                                                 _i1_vector_memory_ops(on))


@_needs_m2m
def test_the_refusal_does_not_swallow_a_byte_wide_op(tmp_path, monkeypatch):
    """NOT VACUOUS. A gate that refused everything would satisfy every assertion above."""
    monkeypatch.delenv("MERLIN_VEC_AFTER_SPECIALIZE", raising=False)
    off = _lower(BYTEWISE_OK, frozenset(), tmp_path)
    on = _lower(BYTEWISE_OK, [VEC], tmp_path)
    assert on != off, "the lever vectorized nothing on an op it is supposed to accept"
    assert on.count("load <8 x float>") > off.count("load <8 x float>"), \
        (on.count("load <8 x float>"), off.count("load <8 x float>"))


# ---------------------------------------------------------------------------------------------
# 3. the numbers -- executed, under more than one initial memory layout
# ---------------------------------------------------------------------------------------------

#: Environment-block sizes the fixture is run under. Shifting the environment block moves the
#: process's initial stack pointer and, with it, what the uninitialised bytes happen to contain.
#: MEASURED on the pre-fix shared object: pads of 0 / 512 / 3072 gave THREE different outputs from
#: the one binary (56, then 14, then 0 of the 64 elements taking the wrong branch). A test that runs
#: once cannot catch this class at all.
_LAYOUT_PADS = (0, 512, 3072)

_RUN_FIXTURE = '''
import hashlib, json, sys
import numpy as np
from merlin.llvmlower.abi import HostModel

so = sys.argv[1]
rng = np.random.default_rng(0)
a = rng.integers(-3, 3, size=(8, 8)).astype(np.int32)
x = rng.standard_normal((8, 8), dtype=np.float32)
out = np.zeros((8, 8), dtype=np.float32)
bufs = [(a.ctypes.data, [8, 8]), (x.ctypes.data, [8, 8]), (out.ctypes.data, [8, 8])]
HostModel.load(so, n_args=len(bufs))(bufs)
expect = np.where(a >= 0, x, np.float32(-1.0))
print(json.dumps({
    "digest": hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest(),
    "matches_reference": bool(np.array_equal(out, expect)),
}))
'''


def _build_so(src: str, features, work: Path) -> Path:
    from merlin.llvmlower.codegen import build_host_shared
    work.mkdir(parents=True, exist_ok=True)
    ll = work / "model.ll"
    ll.write_text(_lower(src, features, work), encoding="utf-8")
    return build_host_shared(ll, work / "model_host.so")


def _run_under_layouts(runner: Path, so: Path, *argv: str) -> list[dict]:
    """Run ``runner`` once per environment-block size. ``ulimit`` is left alone -- changing it also
    perturbs the stack, so the only thing that varies between these runs is the env block."""
    got = []
    for pad in _LAYOUT_PADS:
        env = dict(os.environ, MERLIN_VEC_LAYOUT_PAD="x" * pad)
        proc = subprocess.run([sys.executable, str(runner), str(so), *argv],
                              capture_output=True, text=True, timeout=1800,
                              cwd=str(repo_root()), env=env)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        got.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    return got


@pytest.mark.parametrize("name,src", [("write", MASK_WRITE), ("read", MASK_READ)],
                         ids=["write", "read"])
@_needs_m2m
def test_the_answer_does_not_depend_on_the_initial_memory_layout(name, src, tmp_path, monkeypatch):
    """The whole defect in one assertion, on both arm placements: the lever's output must equal the
    baseline's AND must not move when the process's initial stack layout does."""
    runner = tmp_path / "run_fixture.py"
    runner.write_text(_RUN_FIXTURE, encoding="utf-8")

    for after_specialize in (False, True):
        tag = "old" if after_specialize else "new"
        if after_specialize:
            monkeypatch.setenv("MERLIN_VEC_AFTER_SPECIALIZE", "1")
        else:
            monkeypatch.delenv("MERLIN_VEC_AFTER_SPECIALIZE", raising=False)
        base = _run_under_layouts(runner,
                                  _build_so(src, frozenset(), tmp_path / f"{name}_{tag}_off"))
        lever = _run_under_layouts(runner,
                                   _build_so(src, [VEC], tmp_path / f"{name}_{tag}_on"))
        assert all(r["matches_reference"] for r in base), (name, tag, base)
        assert len({r["digest"] for r in base}) == 1, (name, tag, base)
        assert len({r["digest"] for r in lever}) == 1, \
            f"{name}/{tag}: the lever's output moved with the memory layout: {lever}"
        assert lever[0]["digest"] == base[0]["digest"], (name, tag, lever, base)
        assert all(r["matches_reference"] for r in lever), (name, tag, lever)


# ---------------------------------------------------------------------------------------------
# 4. the whole model -- the measurement the fixtures stand in for
# ---------------------------------------------------------------------------------------------

BUNDLE = artifacts_dir() / "recaptures" / "small_llama_int8_consistent"

_RUN_MODEL = '''
import hashlib, json, resource, sys
import numpy as np
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
from merlin.common.artifacts import recaptures_dir
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args

so, bundle = sys.argv[1], sys.argv[2]
root = recaptures_dir() / bundle
args = resolve_forward_args(root)
golden = np.load(root / "golden.npy")
out = np.zeros(golden.shape, dtype=np.float32)
bufs = [(a.ctypes.data, list(a.shape)) for a in args] + [(out.ctypes.data, list(out.shape))]
HostModel.load(so, n_args=len(bufs))(bufs)
print(json.dumps({
    "digest": hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest(),
}))
'''


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@_needs_m2m
@pytest.mark.skipif(not (BUNDLE / "golden.npy").is_file(), reason="int8 capture bundle absent")
def test_whole_model_is_layout_independent_in_both_placements(tmp_path, monkeypatch):
    """The model the defect was found on. ``ulimit`` is raised inside the runner (the host object
    overflows the default 8 MB stack in ``@forward``) and is IDENTICAL for every arm, so the only
    thing that varies between the runs of one binary is the environment block.

    One model per process: two whole-model ``model_host.so`` in one process abort with
    ``double free or corruption`` (RTLD_GLOBAL interposition), which is why every run is a
    subprocess.
    """
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    runner = tmp_path / "run_model.py"
    runner.write_text(_RUN_MODEL, encoding="utf-8")

    digests: dict[str, set[str]] = {}
    i1_ops: dict[str, int] = {}
    for tag, feats, after in (("base", frozenset(), False),
                              ("lever_new", impr.normalize([VEC]), False),
                              ("lever_old", impr.normalize([VEC]), True)):
        if after:
            monkeypatch.setenv("MERLIN_VEC_AFTER_SPECIALIZE", "1")
        else:
            monkeypatch.delenv("MERLIN_VEC_AFTER_SPECIALIZE", raising=False)
        work = tmp_path / tag
        work.mkdir(parents=True, exist_ok=True)
        prepared, _ = prepare_for_lowering(BUNDLE / "model.mlir", work, int8_compute=True,
                                           features=feats, blocking=False)
        upstream, _ = preprocess_text_textual(prepared.read_text(encoding="utf-8"))
        ll = work / "model.ll"
        ll.write_text(P.lower_to_llvm_ir(upstream, workdir=work, vectorize=True, features=feats),
                      encoding="utf-8")
        i1_ops[tag] = len(_i1_vector_memory_ops(ll.read_text(encoding="utf-8")))
        so = build_host_shared(ll, work / "model_host.so")
        digests[tag] = {r["digest"] for r in _run_under_layouts(runner, so, BUNDLE.name)}

    for tag, seen in digests.items():
        assert len(seen) == 1, f"{tag}: the output moved with the memory layout: {sorted(seen)}"
    assert digests["lever_new"] == digests["base"], digests
    assert digests["lever_old"] == digests["base"], digests
    # ...and the lever added no i1-vector memory access the baseline did not already have.
    assert i1_ops["lever_new"] == i1_ops["base"], i1_ops
    assert i1_ops["lever_old"] == i1_ops["base"], i1_ops


def test_paths_come_from_the_helper():
    """Location-independence: this file must survive being moved."""
    assert (repo_root() / "merlin" / "tests" / "rvv").is_dir()
    assert np.__name__ == "numpy"
    assert hashlib.sha256(b"").hexdigest()
