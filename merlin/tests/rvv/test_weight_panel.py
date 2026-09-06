"""`prepack_weight_panels`: the pack must reach the MAPS, the BYTES and the ABI, or it is wrong.

A bytes-only rewrite produces a model that still links and still runs and is silently wrong, so the
properties asserted here are the three halves that have to agree: the prepared IR's argument types,
the packed blob's shapes, and what `c_runtime.generate` would build its argument table from. Plus the
two that make the lever honest: with the feature off the emitted `.ll` is byte-identical, and with it
on every refusal is counted rather than silently skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import artifacts_dir
from merlin.llvmlower import weight_panel as WP

BUNDLE = artifacts_dir() / "recaptures" / "small_llama_int8_consistent"


def _toolchain_available() -> bool:
    from merlin.llvmlower import toolchain

    return toolchain.m2m_python().is_file() and toolchain.clang().is_file()


def _prepared(work: Path, feats):
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    return prepare_for_lowering(BUNDLE / "model.mlir", work, int8_compute=True,
                                features=frozenset(feats), harts=1, vlen=256)


# ---------------------------------------------------------------------------------------------
# 1. the record: a layout chain replays and inverts
# ---------------------------------------------------------------------------------------------

def test_replay_and_invert_round_trip():
    """The chain the IR rewrite elided is REPLAYED on the bytes, so the two cannot disagree unless
    replay and invert do. A permutation composed with its inverse is the identity, per step kind."""
    steps = (WP.LayoutStep("transpose", perm=(1, 0), in_shape=(7, 5), out_shape=(5, 7)),
             WP.LayoutStep("expand", groups=((0,), (1, 2)), in_shape=(5, 7), out_shape=(5, 7, 1)),
             WP.LayoutStep("collapse", groups=((0, 1), (2,)), in_shape=(5, 7, 1), out_shape=(35, 1)))
    a = np.arange(35, dtype=np.int8).reshape(7, 5)
    b = WP.replay(a, steps)
    assert b.shape == (35, 1)
    assert np.array_equal(WP.replay(b, WP.invert(steps)), a)


def test_pack_bytes_is_the_layout_the_kernel_reads():
    """`Bp[no][k][ni] == B[k][no*NR + ni]`, spelled out independently of the packer."""
    k, n, nr = 6, 8, 4
    stored = np.arange(n * k, dtype=np.int8).reshape(n, k)          # [N][K], as captures store it
    a = WP.PackedArg(arg=0, orig_shape=(n, k), elem="i8",
                     steps=(WP.LayoutStep("transpose", perm=(1, 0),
                                          in_shape=(n, k), out_shape=(k, n)),),
                     m=4, k=k, n=n, mr=4, nr=nr)
    packed = WP.pack_bytes(stored, a)
    assert packed.shape == (n // nr, k, nr)
    b = stored.T
    for no in range(n // nr):
        for kk in range(k):
            for ni in range(nr):
                assert packed[no, kk, ni] == b[kk, no * nr + ni]


def test_line_touch_model_needs_a_line_width_and_reports_both_sides():
    """`line_bytes` is a target fact the caller supplies; there is no default to be wrong about."""
    a = WP.PackedArg(arg=0, orig_shape=(64, 32), elem="i8", steps=(), m=8, k=32, n=64, mr=4, nr=16)
    with pytest.raises(TypeError):
        WP.line_touch_model([a], )                                  # line_bytes is keyword-REQUIRED
    m = WP.line_touch_model([a], line_bytes=64)["total"]
    # one fresh line per k-step before; a contiguous panel after
    assert m["line_touches_before"] == 2 * (64 // 16) * 32
    assert m["line_touches_after"] == 2 * (64 // 16) * (32 * 16 // 64)
    assert m["useful_fraction_after"] == 1.0
    assert m["useful_fraction_before"] < m["useful_fraction_after"]


# ---------------------------------------------------------------------------------------------
# 2. the feature resolves, and it refuses to run without the block request
# ---------------------------------------------------------------------------------------------

def test_feature_resolves_by_name_in_a_fresh_process():
    """`impr_features` must resolve the name in a child that imported nothing else -- the lowering
    runs in one, and `wholemodel_proposer._composes` swallows the KeyError for an unregistered name
    and answers False, so an unresolvable lever is silently unproposable rather than reported."""
    import subprocess
    import sys

    from merlin.common.paths import merlin_dir

    out = subprocess.run(
        [sys.executable, "-c",
         "from merlin.llvmlower.impr_features import get;"
         f"f = get({WP.FEATURE!r}); print(f.name)"],
        cwd=str(Path(merlin_dir()).parent), capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == WP.FEATURE


@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_named_without_the_block_request_is_refused(tmp_path):
    """Named alone it would repack every weight and leave every packed contraction to
    convert-linalg-to-loops -- a lever that reports as applied and measures as a regression."""
    with pytest.raises(ValueError, match="perop_register_block"):
        _prepared(tmp_path, [WP.FEATURE])


# ---------------------------------------------------------------------------------------------
# 3. the pack itself, on a real int8 capture
# ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_every_weight_packs_and_the_argument_types_change(tmp_path):
    prepared, _feats = _prepared(tmp_path, ["perop_register_block", WP.FEATURE])
    args, raw = WP.read_plan(tmp_path)
    assert raw["packed"] == len(args) > 0
    assert raw["refusals"] == {}, raw["refusals"]

    from merlin.llvmlower.model_runner import parse_forward_signature

    sig = parse_forward_signature(prepared)
    for a in args:
        assert a.n % a.nr == 0                                      # never padded
        assert list(sig[a.arg][0]) == list(a.packed_shape), a.arg   # the MAPS moved, not just bytes
        assert list(sig[a.arg][0]) != list(a.orig_shape), a.arg


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_packed_bundle_bytes_manifest_and_abi_all_agree(tmp_path):
    """The three halves. `c_runtime.generate` builds the C runtime's memref descriptors from the
    BUNDLE while the compiled object follows the PREPARED module, so an argument whose rank or extent
    differs between them is a build that links, runs and computes nonsense."""
    prepared, _feats = _prepared(tmp_path, ["perop_register_block", WP.FEATURE])
    args, _raw = WP.read_plan(tmp_path)
    dst, effect = WP.packed_bundle(BUNDLE, args, cache_root=tmp_path / "cache")
    assert effect["weights_packed"] == effect["args_retyped"] == len(args)
    assert WP.assert_abi_agrees(dst, prepared) > 0

    man = json.loads((dst / "weights.safetensors.manifest.json").read_text())
    src_man = json.loads((BUNDLE / "weights.safetensors.manifest.json").read_text())
    header, _ = WP._read_header(dst / "weights.safetensors")
    seen: dict[tuple[int, int], str] = {}
    for a in args:
        name = src_man[str(a.arg)]["weight"]
        assert man[str(a.arg)]["shape"] == list(a.packed_shape)
        assert [int(d) for d in header[name]["shape"]] == list(a.packed_shape)
        # ASSERTED, not assumed: `mining/section_build` dedups weights by NAME, so two arguments CAN
        # share one `data_offsets` range, and packing it for one packs it for the other underneath.
        rng = tuple(header[name]["data_offsets"])
        assert rng not in seen, (name, seen.get(rng))
        seen[rng] = name

    # the bytes are the permutation the plan says, checked against the SOURCE blob independently
    from merlin.llvmlower.weights_pack import load_safetensors_header

    s_hdr, s_off = load_safetensors_header(BUNDLE / "weights.safetensors")
    d_hdr, d_off = load_safetensors_header(dst / "weights.safetensors")
    s_raw = np.fromfile(BUNDLE / "weights.safetensors", dtype=np.uint8)[s_off:]
    d_raw = np.fromfile(dst / "weights.safetensors", dtype=np.uint8)[d_off:]
    for a in args[:4]:
        name = src_man[str(a.arg)]["weight"]
        ss, se = s_hdr[name]["data_offsets"]
        ds, de = d_hdr[name]["data_offsets"]
        src = s_raw[ss:se].view(np.int8).reshape(s_hdr[name]["shape"])
        got = d_raw[ds:de].view(np.int8).reshape(d_hdr[name]["shape"])
        assert np.array_equal(got, WP.pack_bytes(src, a))


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_the_packed_bundle_module_still_parses(tmp_path):
    """Its body reconstructs each argument's original value, so the module still describes the same
    function rather than being a signature over a body that no longer type-checks."""
    from merlin.common import mlir_query as mq

    _prepared(tmp_path, ["perop_register_block", WP.FEATURE])
    args, _raw = WP.read_plan(tmp_path)
    dst, _ = WP.packed_bundle(BUNDLE, args, cache_root=tmp_path / "cache")
    mq.parse((dst / "model.mlir").read_text())


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_the_stock_bundle_is_refused_by_the_abi_guard(tmp_path):
    """The catastrophic case, made loud. A caller that lowers the packed module and hands
    `c_runtime.generate` the STOCK bundle gets an object indexing [N/NR][K][NR] weights against
    [K][N] bytes. It links, it runs, and nothing between there and the result says so."""
    _prepared(tmp_path, ["perop_register_block", WP.FEATURE])
    with pytest.raises(WP.PanelPackRefused, match="unpacked"):
        WP.guard_planned_pack(BUNDLE, tmp_path / "cgen")


# ---------------------------------------------------------------------------------------------
# 4. numerics + the frozen baseline -- whole model, host-side
# ---------------------------------------------------------------------------------------------

def _build_arm(work: Path, feats):
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual
    from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE, lower_to_llvm_ir

    work.mkdir(parents=True, exist_ok=True)
    prepared, concrete = _prepared(work, feats)
    upstream, _ = preprocess_text_textual(prepared.read_text(encoding="utf-8"))
    ll = work / "model.ll"
    ll.write_text(lower_to_llvm_ir(upstream, workdir=work, vectorize=True,
                                   transform_schedule=RVV_TRANSFORM_SCHEDULE,
                                   features=frozenset(concrete or ())), encoding="utf-8")
    return prepared, ll, build_host_shared(ll, work / "model_host.so")


#: The runner, executed in a FRESH PROCESS per (arm, padding). Two reasons, and the first is not
#: hygiene: `HostModel.load(..., n_args=...)` forces RTLD_GLOBAL (its trampoline has to resolve the
#: ciface symbol), so a SECOND whole-model `.so` loaded into the same process has its
#: `_mlir_ciface_forward` resolved to the FIRST one's. Measured here both ways on this capture: with
#: `off` loaded first, `on` returned a different digest; with `on` loaded first, `off` did -- in each
#: case the second arm ran the first arm's code. An in-process A/B therefore compares arm 1 against
#: itself and CANNOT fail. The second reason is the one the padding is for: `MERLIN_AB_PAD` changes
#: the size of the environment block above the initial stack, hence every stack-derived address, so a
#: digest that moves across paddings has an address dependence a single run reads as a pass.
_RUNNER = """
import hashlib, json, sys
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args

so, bundle, ref, dst = sys.argv[1:5]
golden = np.load(ref)
args = resolve_forward_args(bundle)
out = np.zeros(golden.shape, dtype=np.float32)
bufs = [(a.ctypes.data, list(a.shape)) for a in args] + [(out.ctypes.data, list(out.shape))]
HostModel.load(so, n_args=len(bufs))(bufs)
np.save(dst, out)
print(json.dumps({"digest": hashlib.sha256(
    b"".join(float(v).hex().encode() for v in out.ravel())).hexdigest(),
    "n": int(out.size), "arg_shapes": [list(a.shape) for a in args]}))
"""


def _run_fresh(runner: Path, so: Path, bundle: Path, ref: Path, dst: Path, pad: int) -> dict:
    import subprocess
    import sys

    from merlin.common.paths import merlin_dir

    env = dict(os.environ, MERLIN_AB_PAD="X" * pad)
    r = subprocess.run([sys.executable, str(runner), str(so), str(bundle), str(ref), str(dst)],
                       cwd=str(Path(merlin_dir()).parent), env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"), reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "golden_w8a8.npy").is_file(), reason="int8 capture bundle absent")
def test_whole_model_output_is_bit_identical_and_gates(tmp_path):
    """CORRECTNESS IS THE GATE. The packed build must produce the same BYTES, not merely a close
    cosine -- a permutation of a weight's elements cannot change a single value it computes -- and it
    must do so in a fresh process per padding, because a stack-address-dependent bug reads as a pass
    in a single run."""
    from merlin.runtime.backends.zephyr_model import _gate

    runner = tmp_path / "runner.py"
    runner.write_text(_RUNNER, encoding="utf-8")
    golden, golden_w8a8 = np.load(BUNDLE / "golden.npy"), np.load(BUNDLE / "golden_w8a8.npy")
    pads = [1 << (6 + i) for i in range(3)]
    digests: dict[str, set[str]] = {}
    shapes: dict[str, str] = {}
    lls: dict[str, str] = {}
    outs: dict[str, np.ndarray] = {}
    for name, feats in (("off", ["perop_register_block"]),
                        ("on", ["perop_register_block", WP.FEATURE])):
        work = tmp_path / name
        prepared, ll, so = _build_arm(work, feats)
        lls[name] = hashlib.sha256(ll.read_bytes()).hexdigest()
        bundle = BUNDLE
        if name == "on":
            bundle, effect = WP.abi_bundle(BUNDLE, work, prepared)
            assert effect and effect["weights_packed"] > 0
        seen = set()
        for pad in pads:
            r = _run_fresh(runner, so, bundle, BUNDLE / "golden.npy", work / f"out{pad}.npy", pad)
            seen.add(r["digest"])
            shapes[name] = json.dumps(r["arg_shapes"])
        assert len(seen) == 1, f"{name}: digest moved across paddings -- an address dependence: {seen}"
        digests[name] = seen
        outs[name] = np.load(work / f"out{pads[0]}.npy")

    assert lls["off"] != lls["on"], "the lever emitted the same code; it did nothing"
    assert shapes["off"] != shapes["on"], "the two arms bound the same argument shapes; nothing packed"
    assert digests["off"] == digests["on"], "packing the weights changed the output"
    assert np.array_equal(outs["off"], outs["on"])
    for name, arr in outs.items():
        g = _gate(arr, {"fp32": golden, "w8a8": golden_w8a8})
        assert g["ok"], (name, g)
        assert sorted(g["tiers"]) == ["fp32", "w8a8"], (name, g["tiers"])


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"), reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_the_b_operand_pointer_advance_becomes_the_panel_width(tmp_path):
    """The measurement this lever exists for, read off the EMITTED code rather than argued.

    Today the B pointer advances by the contraction's own N per k-step, so a fresh cache line
    delivers NR useful bytes. Packed, it advances by NR. Asserted on the `.ll`'s address arithmetic:
    the N-strided weight walks must be gone and NR-strided ones must appear. This is EVIDENCE about
    the emitted code, NOT a speed claim -- that needs the board.
    """
    _p_off, ll_off, _s = _build_arm(tmp_path / "off", ["perop_register_block"])
    _p_on, ll_on, _s2 = _build_arm(tmp_path / "on", ["perop_register_block", WP.FEATURE])
    args, _raw = WP.read_plan(tmp_path / "on")
    widths = {a.nr for a in args}
    ns = {a.n for a in args}
    assert widths and ns and not (widths & ns), "this capture cannot tell the two strides apart"

    def strides(path: Path) -> set[int]:
        """Constant byte offsets the emitted address arithmetic adds, read structurally."""
        got: set[int] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if " = getelementptr " not in s and " = add " not in s:
                continue
            for tok in s.replace(",", " ").split():
                if tok.lstrip("-").isdigit():
                    got.add(int(tok))
        return got

    off, on = strides(ll_off), strides(ll_on)
    assert ns & off, f"the baseline does not walk B at N; strides seen: {sorted(ns & off)}"
    assert widths <= on, f"the packed build does not walk B at NR={sorted(widths)}"


# ---------------------------------------------------------------------------------------------
# 5. the storage refusals, proven by MUTATION -- a check that cannot fail reports success
# ---------------------------------------------------------------------------------------------

def _tiny_bundle(tmp_path: Path, *, header: dict, manifest: dict, payload: bytes) -> Path:
    import struct

    d = tmp_path / "b"
    d.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    (d / "weights.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
    (d / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    return d


def _arg(i: int, name_shape=(4, 4)) -> WP.PackedArg:
    return WP.PackedArg(arg=i, orig_shape=name_shape, elem="i8",
                        steps=(WP.LayoutStep("transpose", perm=(1, 0),
                                             in_shape=name_shape, out_shape=name_shape[::-1]),),
                        m=4, k=4, n=4, mr=4, nr=2)


def test_two_arguments_naming_one_tensor_are_refused(tmp_path):
    """`mining/section_build` dedups weights by NAME, so two `@forward` arguments CAN name one
    tensor. Packing it for one packs it for the other underneath."""
    d = _tiny_bundle(tmp_path, header={"w": {"dtype": "I8", "shape": [4, 4],
                                             "data_offsets": [0, 16]}},
                     manifest={"0": {"weight": "w", "shape": [4, 4]},
                               "1": {"weight": "w", "shape": [4, 4]}},
                     payload=bytes(range(16)))
    problems = WP.pack_problems(d, [_arg(0), _arg(1)])
    assert any("named by BOTH" in p for p in problems), problems


def test_an_aliased_byte_range_is_refused(tmp_path):
    """Two DISTINCT names can index overlapping bytes (a tied head, an aliased view). Rewriting one
    rewrites the other's data underneath it."""
    d = _tiny_bundle(tmp_path, header={"w1": {"dtype": "I8", "shape": [4, 4],
                                              "data_offsets": [0, 16]},
                                       "w2": {"dtype": "I8", "shape": [4, 4],
                                              "data_offsets": [8, 24]}},
                     manifest={"0": {"weight": "w1", "shape": [4, 4]},
                               "1": {"weight": "w2", "shape": [4, 4]}},
                     payload=bytes(range(24)))
    problems = WP.pack_problems(d, [_arg(0), _arg(1)])
    assert any("share bytes" in p for p in problems), problems


def test_a_stubbed_weight_is_refused(tmp_path):
    """A quantized-subclass weight is STUBBED in the manifest and has no safetensors entry at all;
    packing only the manifest shape would describe a permutation nobody performed."""
    d = _tiny_bundle(tmp_path, header={"w1": {"dtype": "I8", "shape": [4, 4],
                                              "data_offsets": [0, 16]}},
                     manifest={"0": {"weight": "w1", "shape": [4, 4]},
                               "1": {"weight": "gone", "shape": [4, 4], "stub": True}},
                     payload=bytes(range(16)))
    assert WP.storage_gate(d)(1, (4, 4)) == "refused_stub_weight"
    assert WP.storage_gate(d)(0, (4, 4)) is None
    assert WP.storage_gate(d)(9, (4, 4)) == "refused_no_manifest_weight"
    assert WP.storage_gate(d)(0, (5, 5)) == "refused_stored_shape_disagrees"
    assert any("no bytes" in p for p in WP.pack_problems(d, [_arg(0), _arg(1)]))
