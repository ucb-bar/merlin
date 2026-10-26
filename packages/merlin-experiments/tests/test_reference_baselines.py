"""Reference-compiler baselines: built from operator studies, staged read-only, compared same-engine only.

Fixtures are shaped like the operator's measurement study (per layer, one object per implementation with
``case``, ``label`` and ``gsim_cycles``), a corpus case directory (``capsule.yaml`` + interface), a run's
``qa/tier_state.json`` certificate and ``.qa_channel`` gSIM response, and a self-check response row.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from merlin_experiments.corpus import performance_baselines as B
from merlin_experiments.corpus import public_references
from merlin_experiments.phase1.feedback import performance_brief as PB
from merlin_experiments.phase1.feedback import performance_headline as HL
from merlin_experiments.phase1.feedback import performance_headroom as H
from merlin_experiments.phase1.feedback import reference_baselines as RB
from merlin_experiments.spec import SpecError

PEAK = 256
BIG, SMALL = "big_matmul_m32k2048n4096", "small_matmul_m16k64n64"


def _capsule(root, name, m, k, n):
    case = root / "_public_scalability" / "cases" / name
    case.mkdir(parents=True)
    (case / "capsule.yaml").write_text(
        "\n".join(
            [
                f"name: {name}",
                "interface_mlir: capsule.interface.mlir",
                "inputs:",
                f"- {{name: W, role: weight, shape: [{k}, {n}], dtype: i8}}",
                f"- {{name: A0, role: input, shape: [{m}, {k}], dtype: i8}}",
                "operation:",
                "  op: matmul",
                "  attributes: {lhs: A0, weight: W, out: Y0, epilogue: [], output_dtype: i32}",
            ]
        )
    )
    (case / "capsule.interface.mlir").write_text(f"// interface {name}\n")


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    _capsule(root, BIG, 32, 2048, 4096)
    _capsule(root, SMALL, 16, 64, 64)
    _capsule(root / "hidden", "hidden_case", 8, 8, 8)
    (root / "MANIFEST.yaml").write_text("cases: 2\n")
    return root


def _study(path, *, big=(1_000_000, 1_250_000), firesim=None):
    """A five-way-shaped study: one layer on a corpus case, one by shape only, one not in the corpus."""
    fsm, merlin = big
    layers = [
        {
            "layer": "mm_big",
            "useful_macs": 32 * 2048 * 4096,
            "exo": {"case": "EXO_mm_big", "label": "exo:mm_big", "gsim_cycles": 900_000},
            "fsm": {
                "case": BIG,
                "label": "fsm:lib@1234abcd",
                "gsim_cycles": fsm,
                "elf_sha256": "e" * 64,
                **({"firesim_cycles": firesim} if firesim else {}),
            },
            "merlin": {"case": BIG, "label": "merlin:reviewed:d6076f36", "gsim_cycles": merlin},
            "agent": {"case": BIG, "label": "final:frozen", "gsim_cycles": 2_000_000},
        },
        {
            "layer": "mm_small_by_shape",
            "op": "matmul",
            "output_dtype": "i32",
            "epilogue": "",
            "tensors": {
                "W": [[64, 64], "i8", "weight"],
                "A0": [[16, 64], "i8", "input"],
                "Y0": [[16, 64], "i32", "output"],
            },
            "fsm": {"case": "X_scratch_small", "label": "fsm:lib@1234abcd", "gsim_cycles": 5_000},
        },
        {"layer": "mm_other", "fsm": {"case": "X_not_in_corpus", "label": "fsm:lib@1234abcd", "gsim_cycles": 7}},
    ]
    document = {
        "schema": "study.v1",
        "instrument": {"gsim_emulator_sha256": "7" * 64, "harness": "cold call, mcycle window around the call"},
        "layers": layers,
    }
    path.write_text(json.dumps(document))
    return path


def _built(tmp_path, corpus, **kwargs):
    return B.build(
        [_study(tmp_path / "study.json", **kwargs)],
        corpus,
        sources={"fsm": "fsm-library", "merlin": "merlin-reference"},
    )


def test_builder_maps_cases_by_id_and_shape_and_never_reads_excluded_or_author_columns(tmp_path, corpus):
    document = _built(tmp_path, corpus, firesim=1_100_000)
    rows = {(r["case"], r["source"], r["engine"]): r for r in document["baselines"]}
    assert set(rows) == {
        (BIG, "fsm-library", "gsim"),
        (BIG, "fsm-library", "firesim"),
        (BIG, "merlin-reference", "gsim"),
        (SMALL, "fsm-library", "gsim"),  # mapped by its exact shape+semantics key
    }
    big = rows[(BIG, "fsm-library", "gsim")]
    assert big["cycles"] == 1_000_000 and big["revision"] == "fsm:lib@1234abcd"
    assert big["interface_sha256"] == hashlib.sha256(f"// interface {BIG}\n".encode()).hexdigest()
    assert big["provenance"]["study"] == "study.json" and big["provenance"]["row"] == "mm_big"
    assert document["instruments"][big["instrument"]]["engine"] == "gsim"
    assert document["unmapped"] == [
        {
            "row": "mm_other",
            "source": "fsm-library",
            "reason": "not a public corpus case and no exact shape+semantics match",
        }
    ]
    text = json.dumps(document).lower()
    assert {r["cycles"] for r in document["baselines"]} == {1_000_000, 1_100_000, 1_250_000, 5_000}
    assert "exo:mm_big" not in text and "exo_mm_big" not in text and "final:frozen" not in text
    assert document["excluded_sources"] == ["exo: withheld from author inputs"]
    assert "hidden_case" not in text
    assert "emulator_sha256" not in document["instruments"][rows[(BIG, "fsm-library", "firesim")]["instrument"]]
    assert document["sources"]["fsm-library"]["display"] == "the FSM software library"
    with pytest.raises(SpecError):
        B.build([], corpus, sources={"exo": "exo-kernels"})


def test_validate_admits_only_numbers_and_one_line_labels(tmp_path, corpus):
    document = _built(tmp_path, corpus)
    assert RB.validate(json.loads(json.dumps(document))) == document
    for bad in (
        {**document, "schedule": "for i in range(4): mvin"},
        {**document, "baselines": [{**document["baselines"][0], "code": "x"}]},
        {**document, "baselines": [{**document["baselines"][0], "revision": "line one\nline two"}]},
        {**document, "baselines": [{**document["baselines"][0], "source": "undeclared"}]},
        {**document, "baselines": [{**document["baselines"][0], "engine": "spike"}]},
        {**document, "baselines": [{**document["baselines"][0], "cycles": -3}]},
    ):
        with pytest.raises(ValueError):
            RB.validate(bad)


def _stage(tmp_path, document, *, pin=None):
    """Stage ``document`` the way a release stages public references (``reference-kernels/``): one
    positively reviewed operator-authorized origin whose single file is the baselines document."""
    import zipfile

    source = tmp_path / "refs"
    source.mkdir(parents=True)
    payload = (json.dumps(document) + "\n").encode()
    archive = source / "baselines.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(RB.STAGED_DESTINATION, payload)
    origin = {
        "name": RB.STAGED_ORIGIN,
        "kind": "user_permitted_archive",
        "container": archive.name,
        "identity": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "license": "operator measurements",
        "user_authorization": "operator-measured reference compiler cycles; numbers and labels only",
        "allowed_purpose": "reference_reading",
        "files": [
            {
                "member": RB.STAGED_DESTINATION,
                "destination": RB.STAGED_DESTINATION,
                "sha256": pin or hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    manifest = {"schema": public_references.SCHEMA, "review_id": "test-operator-baselines", "origins": [origin]}
    (source / "reference-manifest.json").write_text(json.dumps(manifest))
    return source


def test_release_staging_pins_validates_and_checks_the_corpus(tmp_path, corpus):
    document = _built(tmp_path, corpus)
    staged = tmp_path / "experiment" / "reference-kernels"
    record = public_references.stage(_stage(tmp_path, document) / "reference-manifest.json", staged)
    assert record["members"] == 1 and (staged / RB.STAGED).stat().st_mode & 0o222 == 0
    assert B.verify_corpus(staged, corpus)["baselines"] == len(document["baselines"])
    (corpus / "_public_scalability" / "cases" / BIG / "capsule.interface.mlir").write_text("changed\n")
    with pytest.raises(SpecError, match="another interface"):
        B.verify_corpus(staged, corpus)

    malformed = {**document, "baselines": [{**document["baselines"][0], "kernel": "mvin; compute"}]}
    with pytest.raises(SpecError, match="performance baselines member is malformed"):
        public_references.stage(_stage(tmp_path / "bad", malformed) / "reference-manifest.json", tmp_path / "x")


def _workspace(tmp_path, corpus, document=None, *, pin=None):
    """A run workspace: BIG certified on gSIM (1.5M cycles), SMALL with a model estimate only."""
    ws = tmp_path / "workspace"
    (ws / "qa").mkdir(parents=True)
    (ws / ".qa_channel").mkdir()
    digest = "d" * 64
    cert = {"status": "pass", "digest": digest[:16], "execution_digest": digest, "components": {"p": "1"}}
    (ws / "qa" / "tier_state.json").write_text(json.dumps({BIG: {"L2": cert, "L3": cert}}))
    roofline = {
        "useful_macs": 32 * 2048 * 4096,
        "structural_peak_macs_per_cycle": PEAK,
        "structural_compute_floor_cycles": 32 * 2048 * 4096 // PEAK,
    }
    (ws / ".qa_channel" / "simresp_1.json").write_text(
        json.dumps(
            {
                "sim": "gsim",
                "per_capsule": [
                    {
                        "capsule": BIG,
                        "execution_digest": digest,
                        "barrier_cycles": 1_500_000,
                        "pass": True,
                        "performance": {"structural_roofline": roofline},
                    }
                ],
            }
        )
    )
    if document is not None:
        source = _stage(tmp_path / "src", document, pin=pin)
        public_references.stage(source / "reference-manifest.json", ws / RB.REFERENCE_ROOT)
    small = {
        "status": "PROVISIONAL",
        "complete_event_accounting": True,
        "estimated_whole_soc_cycles": 9_000,
        "structural_roofline": {
            "useful_macs": 16 * 64 * 64,
            "structural_peak_macs_per_cycle": PEAK,
            "structural_compute_floor_cycles": 64,
        },
    }
    verdict = {
        "per_capsule": [
            {"capsule": BIG, "status": "pass", "tiers": {"L2": "pass", "L3": "pass"}},
            {"capsule": SMALL, "status": "pass", "tiers": {"L2": "pass"}, "performance": small},
        ],
        "graded_source_sha256": "a" * 64,
        "graded_at": "2026-10-26T03:00:00+00:00",
        "feedback_brief": {"performance_priority": {"advisory_only": True}},
    }
    return ws, verdict


def test_headroom_and_headline_compare_measured_cycles_on_the_same_engine_only(tmp_path, corpus):
    ws, verdict = _workspace(tmp_path, corpus, _built(tmp_path, corpus, firesim=1_100_000))
    document = H.attach(ws, verdict)
    block = document["feedback_brief"]["performance_priority"]["development_headroom"]
    entries = {e["test_case"]: e for e in block["largest_headroom"]}
    big = {(c["source"], c["engine"]): c for c in entries[BIG]["reference_baselines"]}
    assert big[("fsm-library", "gsim")] == {
        "source": "fsm-library",
        "engine": "gsim",
        "baseline_cycles": 1_000_000,
        "ratio": 1.5,
    }
    assert big[("merlin-reference", "gsim")]["ratio"] == 1.2
    assert big[("fsm-library", "firesim")]["ratio"] is None  # gSIM cycles are never divided by FireSim
    assert "FireSim" in big[("fsm-library", "firesim")]["not_compared"]
    small = entries[SMALL]["reference_baselines"][0]
    assert small["ratio"] is None and "model estimate" in small["not_compared"]
    summary = block["reference_baselines"]
    assert summary["cases_with_baseline"] == 2
    assert summary["by_label"]["fsm-library"]["geomean_ratio"] == 1.5
    assert summary["by_label"]["fsm-library"]["not_compared"] == 2

    headline = document[HL.KEY]
    line = headline["reference_baselines"]["line"]
    assert line.startswith("vs reference compilers on 2 large case(s): geomean ×1.50 slower than the FSM software")
    assert f"worst: `{BIG}` ×1.50 vs the FSM software library" in line
    top = {t["test_case"]: t for t in headline["top_headroom"]}
    assert top[BIG]["vs_reference"] == {"fsm-library": 1.5, "merlin-reference": 1.2}  # same-engine ratio wins
    assert top[SMALL]["vs_reference"] == {"fsm-library": "5000 gSIM cycles (not compared)"}
    rendered = "\n".join(HL.render(headline))
    assert "- vs reference compilers on 2 large case(s)" in rendered and "advisory targets" in rendered
    assert "vs merlin-reference ×1.20" in rendered


def test_no_baselines_leave_every_block_unchanged(tmp_path, corpus):
    ws, verdict = _workspace(tmp_path, corpus)
    document = H.attach(ws, verdict)
    assert "reference_baselines" not in document[HL.KEY]
    assert "reference_baselines" not in document["feedback_brief"]["performance_priority"]["development_headroom"]


def test_a_file_that_differs_from_its_manifest_pin_is_ignored(tmp_path, corpus):
    ws, _ = _workspace(tmp_path, corpus, _built(tmp_path, corpus))
    assert RB.load(ws) is not None
    manifest = ws / RB.REFERENCE_ROOT / RB.REFERENCE_MANIFEST
    manifest.chmod(0o644)
    pinned = json.loads(manifest.read_text())
    pinned["origins"][0]["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(pinned))
    assert RB.load(ws) is None


def test_selfcheck_brief_carries_baselines_for_its_cases(tmp_path, corpus):
    ws, _ = _workspace(tmp_path, corpus, _built(tmp_path, corpus))
    roofline = {
        "useful_macs": 500_000 * PEAK,
        "structural_peak_macs_per_cycle": PEAK,
        "structural_compute_floor_cycles": 500_000,
    }
    rows = [
        {
            "capsule": BIG,
            "pass": True,
            "barrier_tier": "L3",
            "barrier_cycles": 800_000,
            "performance": {"structural_roofline": roofline},
        }
    ]
    brief = PB.build(rows, workspace=ws)
    entry = brief["largest_headroom"][0]
    assert {c["source"]: c["ratio"] for c in entry["reference_baselines"]} == {
        "fsm-library": 0.8,
        "merlin-reference": 0.64,
    }
    assert "faster than Merlin's reference compiler" in brief["reference_baselines"]["line"]


def test_guidance_paragraph_names_the_file_and_says_targets_not_code(tmp_path, corpus):
    text = RB.guidance(_built(tmp_path, corpus))
    assert f"{RB.REFERENCE_ROOT}/{RB.STAGED}" in text
    assert "the FSM software library and Merlin's reference compiler" in text
    assert "2 large test case(s)" in text and "general mechanisms" in text and "no code or schedules" in text


def test_builder_cli_writes_the_file_and_prints_the_member_stanza(tmp_path, corpus, capsys):
    study_dir = tmp_path / "studies"
    study_dir.mkdir()
    _study(study_dir / "five_way.json")
    out = tmp_path / "out" / RB.STAGED
    assert (
        B.main(
            [
                "--corpus",
                str(corpus),
                "--study-dir",
                str(study_dir),
                "--out",
                str(out),
                "--guidance-out",
                str(tmp_path / "guidance.md"),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert hashlib.sha256(out.read_bytes()).hexdigest() in printed
    assert RB.validate(json.loads(out.read_text()))["baselines"]
    assert "general mechanisms" in (tmp_path / "guidance.md").read_text()
