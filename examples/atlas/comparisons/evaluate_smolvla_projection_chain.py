"""Source-bound actual QKV(/O) experiment using a patched frozen native replay.

The full original archive must close before candidates. Native policy, masks,
RoPE, later caches and ten-step integration remain the original implementation.
Actual K/V receipts and a new QKV-conditional O source are required inputs, never
inferred from future jobs. This edge supplies no target arithmetic or dispatch.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

import capture_smolvla_projections as projections
from capture_tinyllama_native import (array_identity, bf16_words_to_float, identity, safe_path,
                                      load_provider_validator)
from merlin.common.paths import artifacts_dir


BASE_SHA = "373e6a07a73282b36f83b66916cc466e71493bbbf2ea3107d4656d8fd29c81fb"
PATCH_SHA = "73dd58ef923d8e1550b21f45d5b372986be5dad59c263935b4c717f7f86c4498"
GENERATED_SHA = "8aa128dbdd784cc58a341b5aebadb86eadd164c021e6170b1a048e9bcc9a045a"
PROJECTION_HELPER_SHA = "7254d161ac1252b50a7cc38f7481cb74fd02dd41a98084a56224ea1fd870740f"
PREFIX = "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn."
HELPER_HASHES = {**projections.HELPER_HASHES,
                 "evaluate_smolvla_attention": "4e2e0798723db98227c7131e94f4625c698b799b48eb400ae77680e48719dcdd"}


def role_names(roles):
    if set(roles) == {"q", "k", "v"}:
        return "qkv"
    if set(roles) == {"q", "k", "v", "o"}:
        return "qkvo"
    raise ValueError("Require exactly QKV, with optional O only as the complete fourth role")


def words_from_actual(target, width):
    if target.shape != (192, width) or target.dtype != np.float32 or not np.isfinite(target).all():
        raise ValueError("Wrong actual projection shape/dtype/finite values")
    if np.any(target.view(np.uint32) & 65535) or np.any(target[177:].view(np.uint32)):
        raise ValueError("Actual projection must be exact BF16 with positive-zero export padding")
    return (target[:177].view(np.uint32) >> 16).astype(np.uint16)


def require_observation(actual_a, actual_c, arrays):
    projections.same(actual_a, arrays["A_bf16"][:177], "Native projection input differs from admitted source")
    projections.same(actual_c, arrays["native_C_bf16"][:177], "Native projection reference differs from admitted source")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectionPlan:
    def __init__(self, roles, args, validator, bindings):
        self.roles_name = role_names(roles)
        self.args, self.validator, self.bindings = args, validator, bindings
        self.roles, self.records, self.handles, self.observations = {}, [], [], []
        for role in self.roles_name:
            item = roles[role]
            if set(item) != {"native_bundle", "native_manifest_sha256", "provider_receipt", "provider_receipt_sha256"}:
                raise ValueError("Unexpected projection role fields")
            manifest, arrays, target, records, details = validator.admit_contraction(
                receipt_path=safe_path(item["provider_receipt"]), receipt_sha256=item["provider_receipt_sha256"],
                native_bundle=safe_path(item["native_bundle"]), native_manifest_sha256=item["native_manifest_sha256"],
                producer_root=safe_path(args.producer_root), model_root=safe_path(args.model_root))
            width = 320 if role in ("k", "v") else 960
            site = manifest["site"]
            if (manifest["status"] != "native_selective_capture_passed" or manifest["source_dtype"] != "bfloat16"
                    or site["module"] != PREFIX + role + "_proj" or site["matrix_shape"] != [192, 960, width]
                    or site["positions"] != [0, 177] or site["k_range"] != [0, 960] or site["output_channels"] != [0, width]
                    or site["native_rows"] != 177 or site["valid_rows"] != 177 or site["attention_valid_positions"] != 141
                    or site["original_masked_positions"] != 36 or site["padding"]["native_masks_or_positions_extended"] is not False):
                raise ValueError("Wrong complete native projection role/source extent: " + role)
            actual = words_from_actual(target, width)
            projections.validate_export(arrays, arrays["A_bf16"][:177], arrays["B_bf16"], arrays["native_C_bf16"][:177])
            manifest_id = identity(safe_path(item["native_bundle"]) / "manifest.json", item["native_manifest_sha256"])
            provider_id = identity(item["provider_receipt"], item["provider_receipt_sha256"])
            self.roles[role] = dict(manifest=manifest, arrays=arrays, actual=actual, details=details,
                                    manifest_id=manifest_id, provider_id=provider_id, actual_id=array_identity(actual),
                                    array_ids={name: array_identity(value) for name, value in arrays.items()})
            self.records.extend([manifest_id, provider_id, *records])
        self.chain = {role: {"native_manifest": self.roles[role]["manifest_id"],
                             "provider_receipt": self.roles[role]["provider_id"],
                             "corrected_actual_words": array_identity(self.roles[role]["actual"])} for role in "qkv"}
        qsource = self.roles["q"]["manifest"]["source"]
        for role in "kv":
            source = self.roles[role]["manifest"]["source"]
            if {key: value for key, value in source.items() if key != "projection_capture"} != qsource:
                raise ValueError("K/V belongs to another original checkpoint/prefix/source epoch")
            if source["projection_capture"]["native_query_manifest"] != self.roles["q"]["manifest_id"]:
                raise ValueError("K/V does not bind this original Q capture")
            if role == "k" and self.roles[role]["manifest"]["site"]["before_RoPE"] is not True:
                raise ValueError("K must be captured before native RoPE")
            projections.same(self.roles[role]["arrays"]["A_bf16"], self.roles["q"]["arrays"]["A_bf16"], "Q/K/V normalized inputs differ")
        if "o" in self.roles:
            o = self.roles["o"]["manifest"]
            if (o["site"].get("conditional_context") != "actual_QKV"
                    or o["source"].get("candidate_chain") != self.chain
                    or {key: value for key, value in o["source"].items() if key != "candidate_chain"} != qsource):
                raise ValueError("O must be recaptured from this exact actual-QKV context/source chain")
            if o["native_context"] != array_identity(self.roles["o"]["arrays"]["A_bf16"][:177]):
                raise ValueError("O native context identity differs from its captured input")
            if not o["source_bookends"]:
                raise ValueError("Conditional O lacks its native capture source bookends")
            for record in o["source_bookends"]:
                identity(record["path"], record["sha256"])
            self.records.extend(o["source_bookends"])

    def bind_prefix(self, model, torch, manifest, arrays, actual_q, prefix, native_rope):
        # Admission only: no forward calls, weight writes, hooks or RNG changes.
        if self.roles["q"]["manifest"] != manifest:
            raise ValueError("Base replay and plan use different native Q sources")
        projections.same(self.roles["q"]["actual"], actual_q, "Base replay and plan use different actual Q")
        for name in ("A_bf16", "B_bf16", "native_C_bf16"):
            projections.same(self.roles["q"]["arrays"][name], arrays[name], "Base replay Q operands changed")
        self.torch, self.rope = torch, native_rope
        self.attention = model.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn
        self.positions = (torch.cumsum(prefix[1], dim=1) - 1).clone()
        if self.positions.shape != (1, 177) or int(prefix[1].sum()) != 141:
            raise ValueError("Original native mask/position extent changed")
        self.positions_id = array_identity(self.positions.numpy())
        self.active = {role: projections.parameter_identity(getattr(self.attention, role + "_proj").weight, torch.uint16) for role in "qkvo"}
        for role, item in self.roles.items():
            module = getattr(self.attention, role + "_proj")
            if module.weight.dtype != torch.bfloat16:
                raise ValueError("Active native projection dtype changed")
            projections.same(module.weight.detach().t().contiguous().view(torch.uint16).numpy(), item["arrays"]["B_bf16"], "Active projection weight differs from admitted B")
        self.o_weight = self.attention.o_proj.weight.detach().t().contiguous().view(torch.uint16).numpy().copy()

    def begin_candidate(self):
        if self.handles:
            raise ValueError("Candidate projection hooks already installed")
        self.counts, self.current_o = {role: 0 for role in "kv"}, None
        self.counts["o"] = 0
        try:
            for role in "kvo":
                self.handles.append(getattr(self.attention, role + "_proj").register_forward_hook(self._hook(role)))
        except BaseException:
            self.remove_candidate_hooks()
            raise

    def _hook(self, role):
        def observe(module, incoming, output):
            torch = self.torch
            width = 320 if role in "kv" else 960
            if (len(incoming) != 1 or incoming[0].shape != (1, 177, 960) or incoming[0].dtype != torch.bfloat16
                    or output.shape != (1, 177, width) or output.dtype != torch.bfloat16 or self.counts[role] != 0):
                raise ValueError("Native projection callback extent/dtype/count changed: " + role)
            projections.require_active_parameters(self.attention, self.active, torch.uint16)
            a = incoming[0].detach().clone().contiguous().view(torch.uint16).numpy()[0].copy()
            c = output.detach().clone().contiguous().view(torch.uint16).numpy()[0].copy()
            if not np.isfinite(bf16_words_to_float(a)).all() or not np.isfinite(bf16_words_to_float(c)).all():
                raise ValueError("Nonfinite native projection callback")
            if role in self.roles:
                item = self.roles[role]
                require_observation(a, c, item["arrays"])
                projections.same(module.weight.detach().t().contiguous().view(torch.uint16).numpy(), item["arrays"]["B_bf16"], "Native active B changed")
                replacement = torch.from_numpy(item["actual"].copy()).view(torch.bfloat16).reshape(1, 177, width)
                projections.same(replacement.view(torch.uint16).numpy()[0], item["actual"], "Actual replacement adaptation changed words")
            else:
                replacement = None
            if role == "o":
                self.current_o = {"A": a, "B": self.o_weight.copy(), "native_C": c}
            self.counts[role] += 1
            return replacement
        return observe

    def remove_candidate_hooks(self):
        while self.handles:
            self.handles.pop().remove()

    def finish_candidate(self, captures):
        if self.handles or self.counts != {"k": 1, "v": 1, "o": 1} or self.current_o is None:
            raise ValueError("Incomplete candidate K/V/O callbacks or cleanup")
        projections.require_active_parameters(self.attention, self.active, self.torch.uint16)
        self.check_array_bookends()
        if self.positions_id != array_identity(self.positions.numpy()):
            raise ValueError("Original native positions changed")
        if self.observations:
            for key in self.current_o:
                projections.same(self.current_o[key], self.observations[0][key], "Repeated actual-QKV O context/reference changed")
        self.observations.append(self.current_o)
        # Core captures are read-only here; the patched base revalidates every
        # archived field and closed Euler chain immediately after this call.

    def check_array_bookends(self):
        for item in self.roles.values():
            if (item["actual_id"] != array_identity(item["actual"])
                    or item["array_ids"] != {name: array_identity(value) for name, value in item["arrays"].items()}):
                raise ValueError("Admitted source/replacement array bytes changed")

    def check_layer0_caches(self, candidates):
        torch = self.torch
        key = torch.from_numpy(self.roles["k"]["actual"].copy()).view(torch.bfloat16).reshape(1, 177, 5, 64)
        with torch.no_grad():
            expected_k = self.rope(key, self.positions).view(torch.uint16).numpy()
        expected_v = self.roles["v"]["actual"].reshape(1, 177, 5, 64)
        for captures in candidates:
            projections.same(captures["prefix_key_cache"][0], expected_k, "Candidate cache0 K differs from native RoPE(actual K)")
            projections.same(captures["prefix_value_cache"][0], expected_v, "Candidate cache0 V differs from actual V")

    def recheck(self):
        self.check_array_bookends()
        if self.positions_id != array_identity(self.positions.numpy()):
            raise ValueError("Original native positions changed")
        self.validator.recheck(self.records)
        for record in self.bindings:
            identity(record["path"], record["sha256"])
        projections.require_active_parameters(self.attention, self.active, self.torch.uint16)

    def extend_report(self, report):
        if len(self.observations) != 2:
            raise ValueError("Need two closed candidate repeats before reporting/export")
        if self.args.capture_conditional_o:
            self._export_o(report)
        report["checks"].pop("layer0_KV_preserved_under_Q_only_substitution")
        report["checks"]["layer0_KV_matches_native_RoPE_actual_K_and_actual_V"] = True
        report.update(actual_projection_roles=self.roles_name, actual_qkv_chain=self.chain,
                      additional_physical_admissions={role: item["details"] for role, item in self.roles.items() if role != "q"})
        report["source_bookends"].extend(self.bindings)
        report["provider_bookends"].extend(self.records)
        report["claim"] = "Actual " + self.roles_name.upper() + " in retained-prefix native policy replay; original masks/RoPE/later caches/residual/ten-step policy remain native. No fresh vision, physical-unit, task-quality, whole-target or timing qualification."
        for flag in ("complete_atlas_execution", "timing_qualified", "application_quality_qualified", "vision_prefix_recomputed", "physical_units_qualified"):
            if report[flag] is not False:
                raise ValueError("Projection plan changed a qualification limit")
        self.recheck()

    def _export_o(self, report):
        if self.roles_name != "qkv":
            raise ValueError("Capture conditional native O before any O substitution")
        out = self.args.output / "conditional-o"
        out.mkdir(exist_ok=False)
        native = self.observations[0]
        a, c = np.pad(native["A"], ((0, 15), (0, 0))), np.pad(native["native_C"], ((0, 15), (0, 0)))
        arrays = {"A_bf16": a, "B_bf16": native["B"], "native_C_bf16": c}
        arrays.update({name: bf16_words_to_float(arrays[name + "_bf16"]) for name in ("A", "B", "native_C")})
        projections.validate_export(arrays, native["A"], native["B"], native["native_C"])
        np.savez(out / "operands.npz", **arrays)
        source = copy.deepcopy(self.roles["q"]["manifest"]["source"])
        source["candidate_chain"] = self.chain
        manifest = dict(schema_version=1, status="native_selective_capture_passed", source_dtype="bfloat16", source=source,
                        npz_sha256=identity(out / "operands.npz")["sha256"], arrays={key: array_identity(value) for key, value in arrays.items()},
                        site={"module": PREFIX + "o_proj", "positions": [0, 177], "k_range": [0, 960], "output_channels": [0, 960],
                              "matrix_shape": [192, 960, 960], "native_rows": 177, "valid_rows": 177, "attention_valid_positions": 141,
                              "original_masked_positions": 36, "conditional_context": "actual_QKV", "before_residual": True,
                              "padding": {"kind": "exported_matrix_zero_rows_only", "rows": [177, 192], "native_masks_or_positions_extended": False}},
                        native_context=array_identity(native["A"]), source_bookends=[*report["source_bookends"], *self.bindings],
                        provider_bookends=[*report["provider_bookends"], *self.records],
                        capture={"route": "genuine_current_original_forward_with_actual_QKV_before_O_residual",
                                 "native_context_O_repeats_raw_exact": True, "core_trajectory_revalidated_after_adapter": True},
                        claim_limits=self.roles["q"]["manifest"]["claim_limits"] + ["genuine_current_QKV_context_before_native_O_residual"])
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        report["conditional_o_capture"] = identity(out / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("role-map", "base-replay", "producer-root", "model-root", "historical-source-root", "observation-input", "output-dir"):
        parser.add_argument("--" + name, type=safe_path, required=True)
    parser.add_argument("--validator-sha256", required=True)
    parser.add_argument("--capture-conditional-o", action="store_true")
    args = parser.parse_args()
    args.output = args.output_dir
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("Output must be new and beneath the configured artifact root")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="failed", claim=__doc__, source_script=identity(__file__), command=sys.argv, complete_atlas_execution=False,
                  timing_qualified=False, application_quality_qualified=False, vision_prefix_recomputed=False, physical_units_qualified=False)
    code = 1
    try:
        early_helpers = [identity(sys.modules[name].__file__, sha) for name, sha in projections.HELPER_HASHES.items()]
        role_map_id = identity(args.role_map)
        role_map_bytes = args.role_map.read_bytes()
        if hashlib.sha256(role_map_bytes).hexdigest() != role_map_id["sha256"]:
            raise ValueError("Role map changed before decoding")
        roles = json.loads(role_map_bytes)
        if args.capture_conditional_o and role_names(roles) != "qkv":
            raise ValueError("Conditional O capture requires QKV without actual O")
        patch = safe_path(Path(__file__).with_name("smolvla_qkv_replay.patch"))
        bindings = [*early_helpers, identity(__file__), identity(projections.__file__, PROJECTION_HELPER_SHA), role_map_id,
                    identity(args.base_replay, BASE_SHA), identity(patch, PATCH_SHA),
                    identity(args.producer_root / "tools/validate_model_contraction.py", args.validator_sha256)]
        q = roles["q"]
        validator, _ = load_provider_validator(
            args.producer_root, args.validator_sha256, q["provider_receipt"], q["provider_receipt_sha256"])
        plan = ProjectionPlan(roles, args, validator, bindings)
        generated = args.output / "replay.py"
        subprocess.run(["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "--output", str(generated), str(args.base_replay), str(patch)],
                       check=True, capture_output=True, text=True)
        bindings.append(identity(generated, GENERATED_SHA))
        replay = load_module("smolvla_generated_qkv_replay", generated)
        args.workload_helper_bindings = [identity(sys.modules[name].__file__, sha) for name, sha in HELPER_HASHES.items()]
        bindings.extend(args.workload_helper_bindings)
        q = roles["q"]
        args.native_bundle, args.provider_receipt = safe_path(q["native_bundle"]), safe_path(q["provider_receipt"])
        args.native_manifest_sha256, args.provider_receipt_sha256 = q["native_manifest_sha256"], q["provider_receipt_sha256"]
        replay.evaluate(args, report, candidate_plan=plan)
        plan.recheck()
        code = 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    report.update(wall_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "receipt": str(args.output / "receipt.json"), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
