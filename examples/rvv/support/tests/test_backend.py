"""Executable RVV provider: pass ordering and published schedules."""

import hashlib
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("rvv_provider")
PACKAGE.__path__ = [str(ROOT)]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    "rvv_provider.backend", ROOT / "backend.py"
)
assert SPEC is not None and SPEC.loader is not None
BACKEND = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BACKEND
SPEC.loader.exec_module(BACKEND)


def test_backend_pipeline_matches_frozen_compiler_pass_order(monkeypatch):
    for key in (
        "MERLIN_FUSE_POST",
        "MERLIN_PAR_RESIDUE_PARALLEL",
        "MERLIN_NO_DEALLOC",
        "MERLIN_GENERALIZE_BEFORE_FUSE",
    ):
        monkeypatch.delenv(key, raising=False)
    assert BACKEND.API_VERSION == 1
    assert BACKEND.DEFAULT_CFLAGS == (
        "-march=rv64gcv",
        "-fno-vectorize",
        "-fno-slp-vectorize",
    )
    assert BACKEND.vector_cflags(
        ["-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]
    ) == [
        *BACKEND.DEFAULT_CFLAGS,
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
    ]
    assert BACKEND.spike_cflags() == [
        "-march=rv64gcv",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
    ]
    assert BACKEND.spike_harness_cflags(
        ["-march=rv64gcv_zvl256b", "-mabi=lp64d", "-mcmodel=medany"]
    ) == [
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-march=rv64gcv_zvl256b",
        "-mabi=lp64d",
        "-mcmodel=medany",
    ]
    serial = BACKEND.build_pipeline("/RVV/schedule.mlir")
    parallel = BACKEND.build_pipeline(
        "/RVV/schedule.mlir", par_sched_path="/RVV/parallel.mlir", perop_parallel=True
    )
    assert (
        hashlib.sha256(serial.encode()).hexdigest()
        == "7f7c9e0df67a937a73d357f9817975268a1255ae65c1b4000f6dd51f1c05aecb"
    )
    assert (
        hashlib.sha256(parallel.encode()).hexdigest()
        == "dcc6464b56db7ca402b0b736a74e3458d4cf37ed14a52a2e705cde73439b2875"
    )
    assert "transform-interpreter{entry-point=__transform_main}" in serial
    assert "convert-vector-to-llvm" in serial


def test_active_schedules_are_byte_identical_to_the_published_payloads():
    champion = ROOT / "schedules/impr_tuned_wholemodel_vf_int8/schedule.mlir"
    prepack = ROOT / "schedules/kodiak_prepack_w8a8_v1/schedule.mlir"
    assert hashlib.sha256(champion.read_bytes()).hexdigest() == (
        "928064a1264cf3dd4e70c1f6d162f989c3cc67160c36e2c19a94e731c10867b4"
    )
    assert prepack.read_bytes() == champion.read_bytes()
    assert BACKEND.schedule_path().name == BACKEND.DEFAULT_SCHEDULE
    assert BACKEND.baseline_schedule_text() == champion.read_text(encoding="utf-8")
    fp32 = (
        BACKEND.schedule_path("rvv_tuned_v1_d1_vfmacc_outerproduct") / "schedule.mlir"
    )
    assert hashlib.sha256(fp32.read_bytes()).hexdigest() == (
        "dec409ddd8a08eec90f4aefea8e4f73d5f02770b5d358bc6c340c73add9fc3c1"
    )
    assert BACKEND.schedule_path("hand_v0_int8").is_dir()


def test_multicore_schedule_variants_preserve_frozen_bytes():
    cases = (
        ({}, "43f9187c5d1d00ccdafc741116a266a3aad96cd61d247611697130f76a02420b"),
        (
            {"tile_aligned": True},
            "56897f03d76cbc3e4457698f0c55907a3c684d2d20de8ab8cfe3e70868c5197c",
        ),
        (
            {"chunks": []},
            "433d3ea18b98b3fedee2b2a0d1a49c51fc4df6cbb8a218b8b6112b149d980aa8",
        ),
    )
    for kwargs, expected in cases:
        actual = BACKEND.parallel_transform_schedule(2, **kwargs)
        assert hashlib.sha256(actual.encode()).hexdigest() == expected
    assert BACKEND.vec_pre_schedule(frozenset()) is None


def test_published_wholemodel_feature_is_owned_here_and_matches_frozen_bytes():
    for names in (
        {"accumulator_resident_wholemodel_vf"},
        {"accumulator_resident_wholemodel_vf", "prepack_weight_layout"},
    ):
        features = BACKEND.normalize_features(names)
        schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
        pipeline = BACKEND.build_pipeline("/RVV/schedule.mlir", features=features)
        assert hashlib.sha256(schedule.encode()).hexdigest() == (
            "93f7ea831e998e27ef8c52c9f93d6c3ec5bc033db3fdf484e7cd14156a1fddac"
        )
        normalized = pipeline.replace(tempfile.gettempdir(), "/TMP")
        assert hashlib.sha256(normalized.encode()).hexdigest() == (
            "e76ed9039baa36dd06a551ebe5ea4ff6a5b359f0aebd5db2dff4c5955a4f1dcb"
        )


def test_prepack_off_does_not_read_or_rewrite_bundle(tmp_path):
    source = tmp_path / "bundle"
    source.mkdir()
    work = tmp_path / "work"
    assert BACKEND.prepare_build_bundle(source, work, frozenset()) == source.resolve()
    assert not work.exists()


def test_published_fused_vfmacc_schedule_is_provider_owned():
    features = BACKEND.normalize_features({"fused_vfmacc_tiled"})
    schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
    assert hashlib.sha256(schedule.encode()).hexdigest() == (
        "393312d4e20b63c4644d01bd1c5a2601b36ab9fd071766abbdf627761a6d9356"
    )


def test_published_accumulator_resident_wholemodel_is_provider_owned():
    features = BACKEND.normalize_features({"accumulator_resident_wholemodel"})
    schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
    assert hashlib.sha256(schedule.encode()).hexdigest() == (
        "ac776529966ab3ddae691639c9c228316630d2743b5fe721ae81e547ea8e2bbf"
    )
    passes = [
        "canonicalize",
        "one-shot-bufferize{bufferize-function-boundaries}",
        "convert-vector-to-scf",
    ]
    actual = BACKEND.apply_pipeline(passes, features)
    normalized = str(actual).replace(tempfile.gettempdir(), "/TMP")
    assert hashlib.sha256(normalized.encode()).hexdigest() == (
        "e1df4bc8342c9713f63aeef9b12a94955fe906c196c5547afdc8fa8ebc90633c"
    )
    assert BACKEND.normalize_features({"fuse_quantize_round_convert"}) == frozenset(
        {"fuse_quantize_round_convert"}
    )


def test_noncontraction_vector_arms_and_name_derived_points_keep_frozen_bytes():
    cases = (
        (
            "vectorize_non_contraction_generics",
            "ae2044e97143c4eccc8e6417d2666c8622dd3d6073fd7ad9ea6b1c3124fe972e",
        ),
        (
            "vectorize_non_contraction_generics_l16_r5",
            "2b9e77fd23a354fc55feac4be47c4bf8bd7c04a9faf9f0bca3094a571c5ad43f",
        ),
    )
    for name, expected in cases:
        features = BACKEND.normalize_features({name})
        assert "erase_self_copy" in features
        schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
        assert hashlib.sha256(schedule.encode()).hexdigest() == expected


def test_rvv_register_tile_packing_and_lmul_grid_keep_frozen_outputs():
    cases = {
        "vfmacc_t_8_32_64": "70e09ba144da7a01e63bb92aac23bcc8c508ddd11c6c1bcae6c56bac8ee3760f",
        "vfmacc_packed": "581def2d44503f5d5a8cee300a381021af6e02ba55b26fd0812ef637ff73160d",
        "vfmacc_packed_8_32_64": "164a195257410b9bafdb4ed5cd4df0c54b69f7e3397db27f805a7285e4968738",
        "fused_vfmacc_scalable": "393312d4e20b63c4644d01bd1c5a2601b36ab9fd071766abbdf627761a6d9356",
    }
    for name, expected in cases.items():
        schedule = BACKEND.apply_schedule(
            BACKEND.baseline_schedule_text(), BACKEND.normalize_features({name})
        )
        assert hashlib.sha256(schedule.encode()).hexdigest() == expected
    assert BACKEND.apply_cflags(
        ["-O2"], BACKEND.normalize_features({"lmul_group_m4"})
    ) == [
        "-O2",
        "-mllvm",
        "-riscv-v-register-bit-width-lmul=4",
    ]


def test_accumulator_resident_v1_v2_grids_keep_frozen_outputs():
    cases = {
        "accumulator_resident_ntail": "8b89d5ba126269f011f3acb2cb4db148e2e93c8789d11856daada1b0373e04b0",
        "accumulator_resident_v2": "654746f6dbbecdc5fc4f863a88573e295fcda3cf8249dbb35c1aa73c1faab881",
        "accum_resident_v2_8_16_16": "9f43e294e6fcf644b18480fb0707f064f3a32168f6ed797ec8ebf727d7fd28b5",
    }
    for name, expected in cases.items():
        features = BACKEND.normalize_features({name})
        schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
        assert hashlib.sha256(schedule.encode()).hexdigest() == expected


def test_accumulator_resident_v3_padding_variants_keep_frozen_outputs():
    cases = {
        "accumulator_resident_wholemodel_vf_bmmpad": "662d7ec4d9a11a0ce1120c1129d58f179804e7ed219ef5a792c35981da7ac7e3",
        "accumulator_resident_wholemodel_vf_mrpad_i8_i8_i32_mr1_nr64_nb16": "a1c928ee69c01ef59aa470c860db261a4f9c4b03468c96d8f62934241b319cb2",
        "accum_resident_v3_8_16_16": "4a06e3b87535e29ecf992aad6e86ad0f6d4c236c28c24b01cc1ffdd85c6c4340",
    }
    for name, expected in cases.items():
        features = BACKEND.normalize_features({name})
        schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
        assert hashlib.sha256(schedule.encode()).hexdigest() == expected


def test_accumulator_microkernel_runner_source_is_provider_owned():
    assert BACKEND.accum_microkernel_marker() == "__merlin_scalarize_a__"
    cases = {
        False: "d845c343b5ccb5713d523e9a275af1b5d15e8ea972ec40471fdff0534efb8f20",
        True: "c185f6dc1f5f468e7e5aba6f69eecf25ad6f099b27c6f0cc27b07985222c0be5",
    }
    for tag_bmm_tails, expected in cases.items():
        source = BACKEND.accum_microkernel_runner_source(tag_bmm_tails=tag_bmm_tails)
        assert hashlib.sha256(source.encode()).hexdigest() == expected


def test_other_vector_schedule_variants_keep_frozen_outputs():
    cases = {
        "fused_vfmacc_contraction": "f1fba77bb3cc9c6259e1a516c1fc99db61722ecfdd826cdf2cd2865883978f0f",
        "vectorized_transcendental_activation": "c848760b45d688fb685874416b929b15e84a91d985015d504fc1044fe6579f73",
        "vectorize_reduction": "7d45088421a3616ffd0cd5730d4b321e7beace473ed7f58f611c909f463be134",
        "vectorize_amax_reduction": "7a53f10063e4150caea8a543b2f050741881fc3348f2245fc9bd6dc8b240d5c3",
    }
    for name, expected in cases.items():
        schedule = BACKEND.apply_schedule(
            BACKEND.baseline_schedule_text(), BACKEND.normalize_features({name})
        )
        assert hashlib.sha256(schedule.encode()).hexdigest() == expected
    assert BACKEND.apply_cflags(
        ["-march=rv64gcv", "-fno-vectorize", "-fno-slp-vectorize", "-O2"],
        BACKEND.normalize_features({"vectorize_scalar_residue"}),
    ) == ["-march=rv64gcv", "-O2"]


def test_perop_request_and_concrete_schedule_are_provider_owned():
    assert BACKEND.vec_noncontraction_shape(
        BACKEND.normalize_features({"vectorize_non_contraction_generics_l16_r5"})
    ) == (16, 5)
    assert BACKEND.perop_mr_sentinel(8) == "perop_register_block_mr8"
    assert BACKEND.perop_nr_sentinel(32) == "perop_register_block_nr32"
    assert BACKEND.parse_perop_mr_sentinel("perop_register_block_mr8") == 8
    assert BACKEND.parse_perop_nr_sentinel("perop_register_block_nr32") == 32
    table = {
        "linalg.matmul:4x16x16": (4, 16),
        "linalg.batch_matmul:1x4x8x16": (4, 8),
    }
    name = BACKEND.ensure_perop_block(table, 16)
    assert name == "perop_register_block_2b_16_71170452a5cb"
    features = BACKEND.normalize_features({name})
    assert "erase_self_copy" in features
    schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
    assert hashlib.sha256(schedule.encode()).hexdigest() == (
        "dc83ea37069613f14560900c9225d7b1708f3e091631a6143ed4a1a0b3228353"
    )
    assert BACKEND.normalize_features({"conv_register_block"}) == frozenset(
        {"conv_register_block"}
    )
    from rvv_provider.requant_fuse import fused_arms

    assert hashlib.sha256(fused_arms([[0, 4, 8, 2, 0, 8]]).encode()).hexdigest() == (
        "bea207e9cd39acf29a8da02f0e5006024d9b0cbc4db028cc7f74afd697320ab4"
    )
    assert BACKEND.normalize_features(
        {"fuse_requant_into_contraction_vec"}
    ) == frozenset({"fuse_requant_into_contraction_vec"})


def test_dynamic_v3_register_blocks_resolve_from_names_and_keep_frozen_schedules():
    cases = {
        "accum_resident_v3_3_16_16": "f60ff4b7c4600c39955b95926eaee3f93eaf24530addeaab79405e43f16352bb",
        "accum_resident_v3kb_4_16_32": "ec25e2616a9b4fd2eff05a464e8674f28c2f68a0f4641847d70d19180419734e",
        "accum_resident_v3u_7_16_16": "21bc0b6d7ebb215dbd70f4281f81cd001d5f244e3ede5d76bd0b44f0caacc08d",
        "accum_resident_v3vl_4_16_16": "a8e75ae043c8d47f873bbd477c2a58601770335e889cccfc788fd773ee131251",
        "accum_resident_v3p_x_x_4_8_16": "d85a384b432fc14544172d83c2eb6a476abac94c64ff39b41ae59732d4b14af6",
    }
    for name, expected in cases.items():
        features = BACKEND.normalize_features({name})
        schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
        assert hashlib.sha256(schedule.encode()).hexdigest() == expected
    assert BACKEND.unclaimed_op_classes("accum_resident_v3p_x_x_4_8_16") == [
        "linalg.matmul"
    ]
