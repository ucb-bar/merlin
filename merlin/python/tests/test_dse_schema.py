"""Every emitted DSE dict validates against its schema."""
from merlin.common import schemas
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.synthesize import FEATURE_RESIDENT
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
from merlin.dse.exploitability import compute_exploitability, row_for
from merlin.dse.harness import evaluate_feature, run_matrix


def _rpv(H=8):
    return compute_rpv(build_region(H=H, reuse_count=H, epilogue=True, K=256, M=1, N=256))


def test_dse_result_validates():
    res = evaluate_feature("vla_action_chunk_decode", _rpv(), FEATURE_RESIDENT)
    assert schemas.validate(res, "dse_result") == []
    assert set(res["results"]) == {"baseline", "software_visible", "hardware_managed", "oracle"}


def test_exploitability_report_validates():
    rows = [row_for(_rpv(r), FEATURE_RESIDENT, r) for r in (1, 2, 4, 8)]
    report = compute_exploitability("vla_action_chunk_decode", FEATURE_RESIDENT, "reuse_count", rows)
    assert schemas.validate(report, "exploitability_report") == []


def test_run_matrix_writes_artifacts(tmp_path):
    cells = [("vla_action_chunk_decode", _rpv())]
    out = run_matrix(cells, [FEATURE_RESIDENT], out_base=tmp_path)
    assert len(out) == 1
    assert (tmp_path / "vla_action_chunk_decode" / FEATURE_RESIDENT / "dse_result.yaml").is_file()
