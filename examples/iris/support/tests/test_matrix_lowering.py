"""The VME route's provider pieces: adapter, contract, kernel emission and the stand-in's arithmetic.

Nothing here runs the unit. The derivation test runs only when the pinned sources are present
($MERLIN_CHIPYARD_IRIS); the stand-in test compiles the emitted kernel for the HOST with the scalar tile
body and checks the tiling loop (super-tiles, clamped edge tiles, short reduction steps) numerically.
"""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    from merlin.targetgen.plugins import load_declared

    module = load_declared("iris", "matrix_lowering")
    assert Path(module.__file__).resolve() == ROOT / "matrix_lowering.py"
    return module


def _fake_encodings(spec_roles):
    """A synthetic table with distinct words per role. Test data only: the real one is derived."""
    from merlin.targetgen.rtl.matrix_isa import MaskedEncoding

    out = {}
    for i, name in enumerate(sorted(spec_roles.values())):
        match = (0x57 | (i << 26)) | (1 << 25)
        mask = 0x7F | (0x3F << 26) | (1 << 25) | (0x7 << 12)
        fixed_rs2 = name in ("SF_VSETTN", "SF_VSETTM", "SF_VSETTK", "SF_VTZERO_T", "SF_VTMV_V_T", "SF_VTDISCARD")
        if fixed_rs2:
            match |= (i & 0x1F) << 20
            mask |= 0x1F << 20
        if name in ("SF_VTZERO_T", "SF_VTDISCARD"):
            mask |= 0x1F << 15
        if name == "SF_VTDISCARD":
            mask |= 0x1F << 7
        out[name] = MaskedEncoding(mnemonic=name, match=match, mask=mask, origin="test")
    return out


def _spec():
    import yaml

    from merlin.targetgen.plugins import load_module

    shim = load_module(ROOT, "vme_shim.py", package_name="iris_test")
    block = yaml.safe_load((ROOT / "contracts" / "matrix_units.yaml").read_text())["units"]["saturn_vme"]
    k = shim.vme_kernel
    spec = k.VmeKernelSpec(
        roles=block["kernel_roles"],
        operand_type=(8, 4),
        readout_type=(32, 1),
        kstep=4,
        layout=k.OperandLayout(**block["operand_layout"]),
    )
    return k, spec


def test_geometry_build_and_simulator_delegate(adapter, monkeypatch, tmp_path):
    shim = adapter.vme_shim if hasattr(adapter, "vme_shim") else adapter.shim
    calls = []
    monkeypatch.setattr(
        shim,
        "load_contract",
        lambda unit: calls.append(unit) or SimpleNamespace(geometry=lambda config: calls.append(config) or (32, 16)),
    )
    assert adapter.geometry(unit="u", config="c") == (32, 16)
    assert calls == ["u", "c"]
    result = object()
    monkeypatch.setattr(shim, "build_object", lambda *a, **k: calls.append((a, k)) or result)
    sigs = {"s": (64, 64, 64)}
    assert (
        adapter.build_object(
            sigs, tmp_path, unit="u", config="c", cc="cc", cflags=["-O2"], scalar_tile=True, parallel_tiles=False
        )
        is result
    )
    assert calls[-1] == (
        (sigs, tmp_path),
        dict(unit="u", config="c", cc="cc", cflags=["-O2"], scalar_tile=True, parallel_tiles=False),
    )
    monkeypatch.setattr(shim, "device_simulator", lambda **k: {"available": False, "reason": "x", **k})
    assert adapter.device_simulator(unit="u", config="c")["available"] is False


def test_selector_and_sidecar_are_the_vme_ones(adapter, tmp_path):
    assert adapter.selector(32)(SimpleNamespace(parallel=(32, 64)))
    assert not adapter.selector(32)(SimpleNamespace(parallel=(31, 64)))
    assert adapter.SIDECAR_NAME == "vme_signatures.json"


def test_contract_names_only_addresses():
    """No encoding, tile edge or kmax may be written in the contract; they are derived."""
    import yaml

    block = yaml.safe_load((ROOT / "contracts" / "matrix_units.yaml").read_text())["units"]["saturn_vme"]

    def numbers(node, path=()):
        if isinstance(node, bool):
            return
        if isinstance(node, int):
            yield path
        elif isinstance(node, dict):
            for k, v in node.items():
                yield from numbers(v, (*path, k))
        elif isinstance(node, list):
            for v in node:
                yield from numbers(v, path)

    # The only numbers are register-operand POSITIONS (operand_layout); every encoding, edge and depth is read.
    assert {p[0] for p in numbers(block)} == {"operand_layout"}
    assert set(block["kernel_roles"]) == {"matmul", "zero", "readout", "release", "set_type", "set_n", "set_m", "set_k"}


def test_kernel_refuses_unresolved_derivations_and_missing_roles():
    k, spec = _spec()
    enc = _fake_encodings(spec.roles)
    with pytest.raises(ValueError, match="refusing"):
        k.emit_kernel(enc, spec, derivation_ok=False)
    enc.pop(spec.roles["matmul"])
    with pytest.raises(ValueError, match="refusing to guess"):
        k.emit_kernel(enc, spec, derivation_ok=True)


def test_spec_rejects_a_layout_the_unit_cannot_read():
    k, spec = _spec()
    with pytest.raises(ValueError, match="8-register window"):
        k.VmeKernelSpec(
            roles=spec.roles, operand_type=(8, 4), readout_type=(32, 1), kstep=4, layout=spec.layout, a_regs=(2, 8)
        )
    with pytest.raises(ValueError, match="accumulating"):
        k.VmeKernelSpec(roles=spec.roles, operand_type=(8, 2), readout_type=(32, 1), kstep=4, layout=spec.layout)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")
def test_stand_in_tiling_matches_the_reference(tmp_path):
    """The scalar stand-in keeps the kernel's tiling loop; sweep tile edges so every edge case is hit."""
    k, spec = _spec()
    src = k.emit_kernel(_fake_encodings(spec.roles), spec, derivation_ok=True)
    (tmp_path / "kernel.h").write_text(src)
    (tmp_path / "main.c").write_text(r"""
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
static int8_t r8(unsigned *s) { *s = *s * 1664525u + 1013904223u; return (int8_t)(*s >> 24); }
int main(void) {
  static const size_t shapes[][3] = {{64,64,64},{96,80,200},{7,9,5},{33,65,17},{1,1,1},{40,3,9},{3,40,2}};
  unsigned s = 1; int bad = 0;
  for (size_t t = 0; t < sizeof shapes / sizeof shapes[0]; t++) {
    size_t m = shapes[t][0], n = shapes[t][1], kk = shapes[t][2];
    int8_t *a = malloc(m * kk), *b = malloc(kk * n);
    int32_t *c = malloc(4 * m * n), *r = malloc(4 * m * n), *bias = malloc(4 * n);
    for (size_t i = 0; i < m * kk; i++) a[i] = r8(&s);
    for (size_t i = 0; i < kk * n; i++) b[i] = r8(&s);
    for (size_t j = 0; j < n; j++) bias[j] = r8(&s) * 100;
    for (int wb = 0; wb < 2; wb++) {
      merlin_vme_gemm_i8(c, a, b, wb ? bias : 0, m, n, kk);
      for (size_t i = 0; i < m; i++) for (size_t j = 0; j < n; j++) {
        int32_t acc = wb ? bias[j] : 0;
        for (size_t q = 0; q < kk; q++) acc += (int32_t)a[q * m + i] * (int32_t)b[q * n + j];
        bad += c[i * n + j] != acc;
      }
    }
    free(a); free(b); free(c); free(r); free(bias);
  }
  printf("bad=%d\n", bad);
  return bad != 0;
}
""")
    for edge in (4, 8, 32):
        exe = tmp_path / f"t{edge}"
        subprocess.run(
            [
                "cc",
                "-O1",
                "-DMERLIN_MATRIX_SCALAR_TILE",
                f"-DMERLIN_MATRIX_TILE_EDGE={edge}",
                "-I",
                str(tmp_path),
                str(tmp_path / "main.c"),
                "-o",
                str(exe),
            ],
            check=True,
        )
        got = subprocess.run([str(exe)], capture_output=True, text=True)
        assert got.returncode == 0, (edge, got.stdout, got.stderr)


@pytest.fixture
def pinned_sources():
    from merlin.common.paths import env

    root = env("MERLIN_CHIPYARD_IRIS")
    if not root or not (Path(root) / "generators" / "saturn" / "benchmarks" / "common" / "vme.h").is_file():
        pytest.skip("MERLIN_CHIPYARD_IRIS with the saturn_vme sources is not configured")
    return Path(root)


def test_derivation_cross_checks_and_geometry_follow_the_enabled_unit(pinned_sources):
    from merlin.targetgen.plugins import load_module

    shim = load_module(ROOT, "vme_shim.py", package_name="iris_test_rtl")
    contract = shim.load_contract("saturn_vme")
    derived = shim.derive_encodings(contract)
    assert derived.ok, derived.to_dict()
    assert set(derived.encodings) >= set(contract.kernel_roles.values())
    # IrisConfig's FIRST vector unit is Rocket's DMA unit (dLen 256, no matrix unit); the geometry must
    # come from the Shuttle unit that enables it (dLen 128).
    assert contract.geometry("IrisConfig") == (32, 16)
    assert contract.kmax(8, 4) >= 1
    src = shim.vme_kernel.emit_kernel(derived.encodings, contract.kernel_spec(), derivation_ok=True)
    assert src.count(".insn r") >= len(contract.kernel_roles)
