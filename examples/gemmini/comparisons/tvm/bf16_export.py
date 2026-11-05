"""Source-bound scalar/reduction exports and opt-in fused BF16 math.

The independent scalar hook preserves original Python values in typed DOUBLE
constants; other ordinary scalar dtypes use their built-in translations.
The Smol mean hook preserves a separately pinned native reduction order.
The language-scale adapter preserves native FP32 scalar opmath in a copied
export graph before ordinary ONNX preparation.
The denoiser rotary adapter specializes the complete bounded integer-position domain
with authenticated native tables, preserving the original RoPE operations.

Original operands and results remain BF16. FP32 internal arithmetic avoids adding
BF16 rounding points inside a mathematical fused operation. Native linear can
choose an unfused layout-dependent kernel; native attention can use blocked BF16
exponential storage. LayerNorm retains its single BF16 output boundary, without
claiming native reduction order or auxiliary statistics. These are FP32-math
candidates, not reproductions of every native CPU kernel, and remain source-reference gated.
"""
import math
from typing import Optional, Sequence


SOURCE_MEAN_ARTIFACTS = {
    "lib/libtorch_cpu.so": "c960407c516ec84d7994fd2c46cf2adc282d0fe6d3d0e928e1d64cc66598e5e4",
    "include/ATen/native/DispatchStub.h": "45cde3f5f24a89acc536a064a17f007189255d6998321a7ac534c7b6f14ddb27",
    "include/ATen/AccumulateType.h": "93381e925f39f8f7f689df6a07f528db2e50c97c4c18d71a9971becafb9e8746",
    "include/ATen/native/cpu/Reduce.h": "f7286c1fa97e96597b8728968a9f639d1d17f1913c64926f8e485ed570df854f",
    "include/ATen/cpu/vec/vec256/vec256_float.h": "62cd6915726701b7b62dea00c16406eb1792c140c197ae1a12e4d35b9c6d81dd",
}

SOURCE_SOFTMAX_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/DispatchStub.h": SOURCE_MEAN_ARTIFACTS["include/ATen/native/DispatchStub.h"],
    "include/ATen/cpu/vec/functional_base.h": "f60163d76edecb5bc3871b46af52f6148be9cb12d2d378bde86ccef6b528853a",
    "include/ATen/cpu/vec/vec512/vec512_float.h": "0150e7eebb07a5235f19ce5c759368ac3c3bf93f65ff459632a30bcf99d5896c",
    "include/ATen/native/cpu/SoftmaxKernel.h": "2e5d98a57e767e29ace4db183eb6d3f3281f669e7093b0b00bdfbb946118a536",
    "include/sleef.h": "2955f7429d04a05c5534a634f1aa77519510c90c603b4cbb4afa29d01e9a5923",
}

SOURCE_LAYERNORM_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/DispatchStub.h": SOURCE_MEAN_ARTIFACTS["include/ATen/native/DispatchStub.h"],
    "include/ATen/native/cpu/moments_utils.h": "0225459908f720e74b4d6814e4b61734042b3d7db22fc8c9d98a889ed1126dff",
    "include/ATen/cpu/vec/vec256/vec256_float.h": SOURCE_MEAN_ARTIFACTS["include/ATen/cpu/vec/vec256/vec256_float.h"],
    "include/ATen/cpu/vec/vec256/vec256_bfloat16.h": "a337093b5269dd1349f15a3dec8c6113f1af9e11714a669b6273139a67f72242",
    "include/ATen/cpu/vec/vec256/vec256_16bit_float.h": "aa2fd19c04c46964061cea8395d292613827e77572c6e8f95d105d5cd2bc3e63",
    "include/ATen/cpu/vec/functional.h": "01672298ad7826eef730bec0d2b1172ed6380bd907daa0de3748cb09d5699258",
    "include/ATen/cpu/vec/functional_base.h": SOURCE_SOFTMAX_ARTIFACTS["include/ATen/cpu/vec/functional_base.h"],
    "include/ATen/cpu/vec/functional_bfloat16.h": "6d40c9bd4b21750b0140bdeaf93cd8f455c4f0466b4501c91e6218b420747ef9",
}

SOURCE_CONV_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/dnnl_version.h": "f51d7a673319a0519ace0a799ee718687a6b0571bb42a9b627aec41d850367af",
    "include/oneapi/dnnl/dnnl_version.h": "99128af2408719334bee22737425b5b4fad774ebba550906f93a6f259a55a6e8",
}

SOURCE_SGEMM_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/CPUBlas.h": "a7f1b8b3bcd6669902170c730917ceefa0da1aa18489dbea6d5e086f706987b1",
}


SOURCE_LANGUAGE_SCALE_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/TensorIterator.h": "06a7456c42b538e2e40429ffb194e9ad1b786735092814a9e242fd8d40877712",
    "include/ATen/OpMathType.h": "0d978d4f2e9b1ecba574b24b2e59ea1ddff96ad09262ddc15c1c359bf6dad5cc",
    "include/ATen/native/cpu/Loops.h": "a6f2ab2e6e7175454f1c4226994177cd2e934d844fe89b09dc049e7a77a7872d",
    "include/c10/util/BFloat16.h": "b403700bb442c764384d0a59c5e6eeb423a573b5c8890debcc2882f4600bc569",
}

SOURCE_ROTARY_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/DispatchStub.h": SOURCE_MEAN_ARTIFACTS["include/ATen/native/DispatchStub.h"],
    "include/ATen/cpu/vml.h": "fdb8944b3720b693b1f9ab63d0c4bc40e085950c4764353686b83b1905a04f4c",
}

SOURCE_ROTARY_MODEL_ARTIFACTS = {
    "lerobot/policies/smolvla/smolvlm_with_expert.py": "b70356145870c7da1e92a2195ee4626f7e9f9387576c6bed5ba2bfefae2a38d9",
    "lerobot/policies/smolvla/modeling_smolvla.py": "be4463140f1abc5bce61f8a8c849ba6ba23f37777d19ad9159c2e1ffff60c1c5",
}

SOURCE_ROTARY_TABLE_WORDS = {
    "sine": "06aee69b79688669a0b603005d0a1c3a083e399e5f5f6da9edca3dcbcb1b2538",
    "cosine": "32911b60b68ad0bdcbe7198e8fd2f8583a7ed2d34f892c69672912aa2c46b4ac",
}

def _source_cpu_bindings(torch, artifacts, operation):
    if torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0" or torch.backends.cpu.get_cpu_capability() != "AVX512" or torch.get_num_threads() != 1:
        raise ValueError(f"{operation} requires the pinned PyTorch CPU build, AVX512 capability and one native thread")
    from pathlib import Path
    from verify_resnet import file_identity, safe_path
    root = safe_path(Path(torch.__file__).parent)
    bindings = [file_identity(root / relative, safe_path) for relative in artifacts]
    if any(record["sha256"] != expected for record, expected in zip(bindings, artifacts.values())):
        raise ValueError(f"{operation} requires the independently proved native library and headers")
    return bindings

SOURCE_FLASH_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/DispatchStub.h": SOURCE_MEAN_ARTIFACTS["include/ATen/native/DispatchStub.h"],
    "include/ATen/native/CPUBlas.h": SOURCE_SGEMM_ARTIFACTS["include/ATen/native/CPUBlas.h"],
    "include/ATen/native/transformers/attention.h": "33d9145503bdd5a37cde7e6925668745cd098312bb7ac792eaf579f27c223852",
    "include/ATen/cpu/vec/vec512/vec512_float.h": SOURCE_SOFTMAX_ARTIFACTS["include/ATen/cpu/vec/vec512/vec512_float.h"],
    "include/ATen/cpu/vec/functional_base.h": SOURCE_SOFTMAX_ARTIFACTS["include/ATen/cpu/vec/functional_base.h"],
    "include/c10/util/BFloat16.h": SOURCE_LANGUAGE_SCALE_ARTIFACTS["include/c10/util/BFloat16.h"],
}
SOURCE_FLASH_LIBM_SHA256 = "f0734a09a0602ccde6b98ea7fbb3c9cfc62b3a4cf4a7b17ccd36d774ca22b8df"


SOURCE_SILU_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/DispatchStub.h": SOURCE_MEAN_ARTIFACTS["include/ATen/native/DispatchStub.h"],
    "include/ATen/native/Activation.h": "e63501cd2f4cae15db83d8b471cd32592fe61b409360e77c17210b12aa528fd1",
    "include/ATen/TensorIterator.h": SOURCE_LANGUAGE_SCALE_ARTIFACTS["include/ATen/TensorIterator.h"],
    "include/ATen/native/cpu/Loops.h": SOURCE_LANGUAGE_SCALE_ARTIFACTS["include/ATen/native/cpu/Loops.h"],
    "include/ATen/cpu/vec/vec512/vec512_float.h": SOURCE_SOFTMAX_ARTIFACTS["include/ATen/cpu/vec/vec512/vec512_float.h"],
    "include/sleef.h": SOURCE_SOFTMAX_ARTIFACTS["include/sleef.h"],
}


def source_silu_contract(torch, policy="native"):
    """Bind the pinned one-thread FP32 vector SiLU without changing native defaults."""
    if policy == "native":
        return None
    if policy != "torch_cpu_silu_fp32_v1":
        raise ValueError("Unknown source SiLU export policy")
    import os
    if "ATEN_CPU_CAPABILITY" in os.environ:
        raise ValueError("Source SiLU requires original CPU dispatch without an ISA override")
    bindings = _source_cpu_bindings(torch, SOURCE_SILU_ARTIFACTS, "Source SiLU")
    return {
        "name": policy,
        "status": "opt_in_source_preserving_export_not_policy_qualified",
        "domain": "merlin.source_preserving",
        "version": 1,
        "operator": "TorchSiluAvx512F32",
        "torch_git": torch.version.git_version,
        "cpu_capability": "AVX512",
        "cpu_threads": 1,
        "source": "pytorch449b_Activation.cpp_silu_kernel_FP32_branch_L1115_1125",
        "native_loop": "positive_dense_contiguous_CPU_FP32_coalesced_flat_count_divisible32;paired16lane_vectors;no_scalar_tail",
        "arithmetic": "pinned_SLEEF_u10_eight_explicit_FP32_FMA;separate_one_plus_exp_negative_x;direct_FP32_divide",
        "export_admission": "original_EP_FP32_SiLU_static_positive_shape_contiguous_offset0_count_divisible32;BF16_sites_keep_existing_fused_recipe",
        "proof_geometry": [[1, 50, 720], [640]],
        "IEEE_admission": "ambient_RNE_FTZ_DAZ_clear_verified_per_call;not_enforced_by_export_shape_guard",
        "proof_admission": "actual_and_controls_have_finite_source_exp_denominator_output;general_runtime_value_admission_not_enforced_by_export_shape_guard",
        "dispatch_evidence": "actual_readonly_DispatchStubImpl_chooser_selected_AVX512_ELF8290270;not_internal_callback_trace",
        "workspace": "original_shape_FP32_output;flatten_and_restore_views;no_private_heap_scratch",
        "lowering": "existing_TorchSiluAvx512F32_flat_helper_explicit_llvm_fma;no_Torch_MKL_SLEEF_exp_or_SiLU_callout;backend_may_call_libm_fmaf_floorf;process_primitive_resolution_not_recorded",
        "qualification": "bounded_actual_and_controls_only;full_prefix_policy_and_target_unqualified;no_universal_correctly_rounded_exp_claim",
        "bindings": bindings,
    }


def prepare_source_silu_export(torch, source, policy="native"):
    """Admit original FP32 operands; retain the source EP and all explicit casts."""
    record = source_silu_contract(torch, policy)
    if record is None:
        return source, None
    if not isinstance(source, torch.export.ExportedProgram):
        raise ValueError("Source SiLU requires an original ExportedProgram")
    sites = []
    for node in source.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.silu.default:
            continue
        if len(node.args) != 1 or node.kwargs or not isinstance(node.args[0], torch.fx.Node):
            raise ValueError("Source SiLU original operand signature changed")
        value = node.args[0].meta.get("val")
        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            continue
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Source SiLU requires original dense CPU FP32 input metadata")
        shape = tuple(value.shape)
        if not shape or any(type(dim) is not int or dim <= 0 for dim in shape) or not value.is_contiguous() or value.storage_offset() != 0:
            raise ValueError("Source SiLU requires positive static contiguous input geometry with zero storage offset")
        count = math.prod(shape)
        if count % 32 or count > ((1 << 63) - 1) // 4:
            raise ValueError("Source SiLU requires a paired32 element count within the FP32 byte span")
        result = node.meta.get("val")
        if not isinstance(result, torch.Tensor) or result.dtype != value.dtype or result.device != value.device or tuple(result.shape) != shape or not result.is_contiguous():
            raise ValueError("Source SiLU original result metadata changed")
        sites.append({
            "node": node.name,
            "input_node": node.args[0].name,
            "shape": list(shape),
            "dtype": "torch.float32",
            "strides": list(value.stride()),
            "storage_offset": value.storage_offset(),
            "flat_count": count,
        })
    record.update(site_count=len(sites), sites=sites, graph_mutated=False)
    return source, record


def source_flash_contract(torch, policy="native"):
    """Bind the proved host-only vision Flash arithmetic and native dispatch."""
    if policy == "native":
        return None
    if policy != "torch_cpu_flash_bf16_vision1024_v1":
        raise ValueError("Unknown source Flash export policy")
    import ctypes
    import os
    from pathlib import Path
    from verify_resnet import file_identity, safe_path
    overrides = ("MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS", "MKL_DEBUG_CPU_TYPE", "ATEN_CPU_CAPABILITY", "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "ONEDNN_DEFAULT_FPMATH_MODE", "DNNL_DEFAULT_FPMATH_MODE")
    if not torch.backends.mkl.is_available() or not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled or torch.get_float32_matmul_precision() != "highest" or any(name in os.environ for name in overrides):
        raise ValueError("Source Flash requires original highest-precision MKL/oneDNN dispatch without ISA or consistency overrides")
    if torch.cpu._is_avx512_bf16_supported() or torch.cpu._is_amx_tile_supported():
        raise ValueError("Source Flash requires the proved AVX512 software-BF16 CPU family")
    bindings = _source_cpu_bindings(torch, SOURCE_FLASH_ARTIFACTS, "Source vision Flash")
    root = safe_path(Path(torch.__file__).parent.parent)
    model_bindings = [file_identity(root / relative, safe_path) for relative in SOURCE_ROTARY_MODEL_ARTIFACTS]
    if any(record["sha256"] != expected for record, expected in zip(model_bindings, SOURCE_ROTARY_MODEL_ARTIFACTS.values())):
        raise ValueError("Source Flash requires the independently bound original Smol model sources")
    cpu = ctypes.CDLL(bindings[0]["path"])
    packing = getattr(cpu, "_ZN2at6native7cpublas10could_packEN3c1010ScalarTypeE")
    packing.argtypes, packing.restype = [ctypes.c_int8], ctypes.c_bool
    version = cpu.mkl_blas_avx512_get_bkernel_api_version
    version.argtypes, version.restype = [], ctypes.c_int
    consistency = cpu.mkl_serv_cbwr_get
    consistency.argtypes, consistency.restype = [ctypes.c_int], ctypes.c_int
    if packing(15) or version() != 1 or consistency(-1) != 1:
        raise ValueError("Source Flash requires original nonpacking software-BF16 MKL dispatch and default CBWR")

    class DlInfo(ctypes.Structure):
        _fields_ = [("fname", ctypes.c_char_p), ("fbase", ctypes.c_void_p), ("sname", ctypes.c_char_p), ("saddr", ctypes.c_void_p)]

    loader = ctypes.CDLL(None)
    loader.dladdr.argtypes, loader.dladdr.restype = [ctypes.c_void_p, ctypes.POINTER(DlInfo)], ctypes.c_int
    loader.dlvsym.argtypes, loader.dlvsym.restype = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_void_p
    owner = DlInfo()
    anchor = ctypes.cast(cpu.cblas_gemm_bf16bf16f32, ctypes.c_void_p).value
    if not loader.dladdr(anchor, ctypes.byref(owner)) or safe_path(owner.fname.decode()) != safe_path(bindings[0]["path"]) or anchor - owner.fbase != 0x90af260:
        raise ValueError("Source Flash native library symbol ownership changed")
    original_expf = ctypes.c_void_p.from_address(owner.fbase + 0x14d515f8).value
    exp_owner = DlInfo()
    if not original_expf or not loader.dladdr(original_expf, ctypes.byref(exp_owner)):
        raise ValueError("Source Flash original expf binding is unavailable")
    libm_path = safe_path(exp_owner.fname.decode())
    libm_binding = file_identity(libm_path, safe_path)
    if libm_binding["sha256"] != SOURCE_FLASH_LIBM_SHA256:
        raise ValueError("Source Flash requires the independently proved host libm")
    libm = ctypes.CDLL(str(libm_path))
    expf_symbols = (loader.dlvsym(libm._handle, b"expf", b"GLIBC_2.27"), ctypes.cast(libm.expf, ctypes.c_void_p).value, ctypes.cast(loader.expf, ctypes.c_void_p).value)
    if any(address != original_expf for address in expf_symbols):
        raise ValueError("Source Flash requires the original GLIBC_2.27 expf family")
    return {
        "name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
        "domain": "merlin.source_preserving", "version": 1, "operator": "TorchFlashBf16Vision1024",
        "native": "original_BF16_CPU_vision_Flash_QKV_1x12x1024x64_positive_zero_mask",
        "native_threads": 1, "native_qkv_strides": [786432, 64, 768, 1],
        "source_shapes": {"qkv": [1, 12, 1024, 64], "mask": [1, 1, 1024, 1024]},
        "native_dispatch": "AVX512_Flash_query256_key512_with_pack_false;MKL_software_BF16_widen_to_FP32_SGEMM;default_CBWR",
        "algorithm": "QK64_serial_FP32_FMA;source_scale_mask_FMA;source_fexp_u20_BF16_probability_store;16lane_max_sum;online_expf_rescale;PV512_two256_serial_FMA_partials_with_per_partial_beta_blend;FP32_reciprocal_product_oneBF16_RNE_store",
        "scratch": {"fp32_words": 180832, "bf16_words": 131072, "bytes": 985472, "explicit_call_tir_tuple_outputs": True, "reuse": "serial_head_query_tiles"},
        "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp",
        "expf": {"version": "GLIBC_2.27", "original_caller_GOT_and_default_symbol_equal": True, "library": libm_binding},
        "source_admission": "original_static_CPU_BF16_QKV_geometry_stride_offset0;dropout0_noncausal_noGQA_scaleNone_or0.125;bounded_input_independent_original_constant_mask_DAG_all_positive_zero",
        "wholeprefix_expected_site_count": 24,
        "IEEE_admission": "finite_source_intermediates_RNE_FTZ_DAZ_clear_checked_by_external_proofs;no_runtime_value_or_IEEE_state_guard_and_no_floating_state_setters",
        "external_full_kernel_proof_sha256": "e139164d14961170dadd88968d9d2a463be4eb129305c7e6a17260db7fe8e6bb",
        "external_QK_proof_sha256": "a6de37875c1e4f7e471747f904b2a840a6e08d518937ab7fb062ff855ac3ac01",
        "external_PV_proof_sha256": "72068427b43f3cf02c91af6fd3887fd9f766eb07f882e131cfeea3c8b7fb06e7",
        "limits": ["private_MKL_selected_inner_entry_and_internal_packing_not_observed;code_conditioned_route_and_bounded_word_proofs", "host_only;fullprefix_session_trajectory_policy_quality_and_target_unqualified", "unsupported_dynamic_nonzero_or_negative_zero_mask_source_refused"],
        "unchanged": ["original_EP", "masks", "parameters", "public_inputs_outputs", "generic_attention_elsewhere", "native_reference", "acceptance_gates"],
        "bindings": bindings + model_bindings + [libm_binding],
    }


def source_flash_sites(torch, program, expected_count=None):
    """Prove the original bounded constant-zero mask without changing the EP."""
    if expected_count is not None and (type(expected_count) is not int or expected_count < 0):
        raise ValueError("Source Flash expected site count must be a nonnegative integer")
    from torch.export.graph_signature import InputKind, TensorArgument
    if not isinstance(program, torch.export.ExportedProgram):
        raise ValueError('Source Flash requires an original ExportedProgram')
    geometry = (1, 12, 1024, 64)
    source_stride = (786432, 64, 768, 1)
    specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs if isinstance(spec.arg, TensorArgument)}
    constructors = {torch.ops.aten.ones.default, torch.ops.aten.zeros.default}
    operations = constructors | {torch.ops.aten.to.device, torch.ops.aten.to.dtype, torch.ops.aten.view.default, torch.ops.aten.unsqueeze.default, torch.ops.aten.expand.default, torch.ops.aten.lift_fresh_copy.default, torch.ops.aten.detach_.default, torch.ops.aten.sub.Tensor, torch.ops.aten.masked_fill.Scalar, torch.ops.aten.alias.default}
    cached = {}

    # Producer evaluation alone cannot prove a mask if an off-path writer changes
    # its storage. Track schema aliases across the whole graph, then check writes
    # before each attention call against the mask and all of its producer inputs.
    import operator
    nodes = list(program.graph.nodes)
    order = {node: index for index, node in enumerate(nodes)}
    parents = {node: node for node in nodes}
    writes = []
    unknown_aliases = []

    def root(node):
        while parents[node] is not node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def merge(left, right):
        parents[root(left)] = root(right)

    def tensor_nodes(value):
        if isinstance(value, torch.fx.Node):
            return [value]
        if isinstance(value, (tuple, list)):
            return [node for item in value for node in tensor_nodes(item)]
        if isinstance(value, dict):
            return [node for item in value.values() for node in tensor_nodes(item)]
        return []

    # Distinct lifted names can still refer to the same bound storage.
    storages = {}
    for node in nodes:
        if node.op != 'placeholder':
            continue
        spec = specs.get(node.name)
        value = program.constants.get(spec.target) if spec is not None and spec.kind == InputKind.CONSTANT_TENSOR else program.state_dict.get(spec.target) if spec is not None else None
        if isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided:
            storage = value.untyped_storage()._cdata
            if storage in storages:
                merge(node, storages[storage])
            else:
                storages[storage] = node

    for node in nodes:
        if node.op in ('placeholder', 'output', 'get_attr'):
            continue
        arguments = tensor_nodes((node.args, node.kwargs))
        if node.op == 'call_function' and node.target is operator.getitem:
            for argument in tensor_nodes(node.args[0]):
                merge(node, argument)
            continue
        schema = getattr(node.target, '_schema', None) if node.op == 'call_function' else None
        if schema is None:
            for argument in arguments:
                merge(node, argument)
            unknown_aliases.append((node, arguments))
            continue
        bindings = [(argument, node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)) for index, argument in enumerate(schema.arguments)]
        aliases = [(argument.alias_info, tensor_nodes(actual)) for argument, actual in bindings if argument.alias_info is not None]
        for info, actual_nodes in aliases:
            if info.is_write and node.target != torch.ops.aten.detach_.default:
                writes.append((node, actual_nodes))
        for returned in schema.returns:
            info = returned.alias_info
            if info is None:
                continue
            labels = info.before_set | info.after_set
            # Tensor-list annotations and wildcard aliases lack a unique label;
            # conservatively include every input storage rather than assume fresh.
            for argument_info, actual_nodes in aliases:
                argument_labels = argument_info.before_set | argument_info.after_set
                if not labels or not argument_labels or '*' in labels or '*' in argument_labels or labels & argument_labels:
                    for argument in actual_nodes:
                        merge(node, argument)
            if not aliases:
                for argument in arguments:
                    merge(node, argument)
                unknown_aliases.append((node, arguments))

    def prove_no_mask_writes(mask_node, attention_node):
        closure = set()
        pending = [mask_node]
        while pending:
            node = pending.pop()
            if not isinstance(node, torch.fx.Node) or node in closure:
                continue
            closure.add(node)
            pending.extend(node.all_input_nodes)
        families = {root(node) for node in closure}
        for writer, targets in writes + unknown_aliases:
            if order[writer] < order[attention_node] and any(root(target) in families for target in targets):
                raise ValueError('Source Flash mask has an unproved writable or unknown storage alias')

    def meta(node):
        return node.meta.get('val') if isinstance(node, torch.fx.Node) else None

    def evaluate(value):
        if not isinstance(value, torch.fx.Node):
            if isinstance(value, (tuple, list)):
                return type(value)(evaluate(item) for item in value)
            if value is None or type(value) in (int, float, bool, str) or isinstance(value, (torch.dtype, torch.device, torch.layout)):
                return value
            raise ValueError('Source Flash masks reject inline tensors or nonprimitive literals')
        if value in cached:
            return cached[value]
        info = meta(value)
        if not isinstance(info, torch.Tensor) or info.device.type != 'cpu' or info.layout != torch.strided or info.dtype not in (torch.bool, torch.float32, torch.bfloat16) or any(type(dim) is not int for dim in info.shape) or math.prod(info.shape) > 1048576:
            raise ValueError('Unproved source Flash mask metadata')
        if value.op == 'placeholder':
            spec = specs.get(value.name)
            constant = program.constants.get(spec.target) if spec is not None and spec.kind == InputKind.CONSTANT_TENSOR else None
            if not isinstance(constant, torch.Tensor) or constant.device.type != 'cpu' or constant.layout != torch.strided or constant.dtype != torch.bfloat16 or constant.shape != () or int(constant.view(torch.uint16)) != 0x3f80:
                raise ValueError('Source Flash masks require original scalar-one constants')
            result = constant.detach().clone()
        elif value.op == 'call_function' and value.target in operations:
            args = evaluate(value.args)
            kwargs = {name: evaluate(item) for name, item in value.kwargs.items()}
            if value.target in constructors:
                shape = args[0]
                if not isinstance(shape, (tuple, list)) or any(type(dim) is not int or dim <= 0 for dim in shape) or math.prod(shape) > 1048576:
                    raise ValueError('Source Flash mask constructor is not bounded')
                kwargs.setdefault('dtype', info.dtype)
                if kwargs['dtype'] != info.dtype or kwargs.get('layout', torch.strided) != torch.strided or kwargs.get('requires_grad', False) is not False:
                    raise ValueError('Source Flash mask constructor dtype/layout changed')
                if kwargs.get('device', torch.device('cpu')) != torch.device('cpu') or kwargs.get('pin_memory', False) is not False:
                    raise ValueError('Source Flash masks require ordinary CPU constants')
            if value.target == torch.ops.aten.to.device and (len(args) < 3 or args[1] != torch.device('cpu') or args[2] not in (torch.bool, torch.float32, torch.bfloat16)):
                raise ValueError('Source Flash mask cast requires the original CPU dtype')
            if value.target == torch.ops.aten.to.dtype and (len(args) < 2 or args[1] not in (torch.bool, torch.float32, torch.bfloat16)):
                raise ValueError('Source Flash mask cast dtype changed')
            if value.target == torch.ops.aten.expand.default:
                if any(type(dim) is not int or dim <= 0 for dim in args[1]) or math.prod(args[1]) > 1048576:
                    raise ValueError('Source Flash mask expansion is not bounded')
            result = value.target(*args, **kwargs)
        else:
            raise ValueError('Source Flash mask must be derived solely from original constants')
        if not isinstance(result, torch.Tensor) or result.device.type != 'cpu' or result.dtype != info.dtype or tuple(result.shape) != tuple(info.shape):
            raise ValueError('Source Flash mask result differs from original metadata')
        cached[value] = result
        return result

    records = []
    with torch.no_grad():
        for node in program.graph.nodes:
            if node.op != 'call_function' or node.target != torch.ops.aten.scaled_dot_product_attention.default:
                continue
            q = meta(node.args[0])
            if not isinstance(q, torch.Tensor) or q.dtype != torch.bfloat16:
                continue
            if any(type(dim) is not int for dim in q.shape):
                raise ValueError('Source Flash requires static query dimensions')
            if tuple(q.shape) != geometry:
                continue
            schema = node.target._schema.arguments
            if len(node.args) > sum(not argument.kwarg_only for argument in schema):
                raise ValueError('Source Flash positional argument inventory changed')
            bound = {argument.name: argument.default_value for argument in schema if argument.has_default_value()}
            for argument, actual in zip(schema, node.args):
                if argument.kwarg_only:
                    raise ValueError('Source Flash argument signature changed')
                bound[argument.name] = actual
            for name, actual in node.kwargs.items():
                if name not in {argument.name for argument in schema} or name in {argument.name for argument in schema[:len(node.args)]}:
                    raise ValueError('Source Flash argument signature changed')
                bound[name] = actual
            if not {'query', 'key', 'value'} <= set(bound) or type(bound.get('dropout_p')) not in (int, float) or type(bound.get('is_causal')) is not bool or type(bound.get('enable_gqa')) is not bool or bound.get('scale') is not None and type(bound['scale']) is not float:
                raise ValueError('Source Flash flags must be original compile-time literals')
            if bound.get('dropout_p') != 0.0 or bound.get('is_causal') is not False or bound.get('enable_gqa') is not False or bound.get('scale') not in (None, 0.125):
                raise ValueError('Source Flash requires original inference flags and scale')
            for name in ('query', 'key', 'value'):
                operand = meta(bound[name])
                if not isinstance(operand, torch.Tensor) or operand.device.type != 'cpu' or operand.layout != torch.strided or operand.dtype != torch.bfloat16 or any(type(dim) is not int for dim in (*operand.shape, *operand.stride())) or type(operand.storage_offset()) is not int or operand.storage_offset() != 0 or tuple(operand.shape) != geometry or tuple(operand.stride()) != source_stride:
                    raise ValueError('Source Flash requires original vision Q/K/V geometry and strides')
            prove_no_mask_writes(bound.get('attn_mask'), node)
            mask = evaluate(bound.get('attn_mask'))
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bfloat16 or tuple(mask.shape) != (1, 1, 1024, 1024) or bool(torch.any(mask.contiguous().view(torch.uint16) != 0)):
                raise ValueError('Source Flash requires a proved original positive-zero mask')
            records.append({'node': node.name, 'mask': bound['attn_mask'].name, 'mask_proof': 'bounded_input_independent_original_constant_DAG_positive_zero', 'preceding_mask_alias_writes_refused': True, 'native_qkv_strides': list(source_stride)})
    if expected_count is not None and len(records) != expected_count:
        raise ValueError('Original vision Flash inventory changed')
    return records


def prepare_source_flash_export(torch, source, policy="native", expected_count=None):
    """Validate the original Flash source before any policy translation is used."""
    record = source_flash_contract(torch, policy)
    if record is None:
        return source, None
    sites = source_flash_sites(torch, source, expected_count)
    record.update(site_count=len(sites), sites=sites, source_EP_mask_proof=True)
    return source, record


def source_rotary_contract(torch, policy="native"):
    """Bind source domain specialization for the original bounded Smol denoiser."""
    if policy == "native":
        return None
    if policy != "torch_cpu_denoise_rotary64_table_v1":
        raise ValueError("Unknown source rotary export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_ROTARY_ARTIFACTS, "Source denoiser rotary")
    if not torch.backends.mkl.is_available():
        raise ValueError("Source denoiser rotary requires the pinned native MKL build")
    from pathlib import Path
    from verify_resnet import file_identity, safe_path
    root = safe_path(Path(torch.__file__).parent.parent)
    model_bindings = [file_identity(root / relative, safe_path) for relative in SOURCE_ROTARY_MODEL_ARTIFACTS]
    if any(record["sha256"] != expected for record, expected in zip(model_bindings, SOURCE_ROTARY_MODEL_ARTIFACTS.values())):
        raise ValueError("Source denoiser rotary requires the independently proved original model sources")
    return {
        "name": policy,
        "status": "opt_in_source_preserving_export_not_policy_qualified",
        "native": "original_CPU_FP32_D64_RoPE_with_integer_denoiser_positions",
        "native_threads": 1,
        "native_dispatch": "AVX2_sin_cos_registration;FP32_oneMKL_VML_HA_FTZDAZ_OFF",
        "algorithm": "whole_domain_original_FP32_frequency_division_and_native_trig_source_domain_specialization;runtime_integer_Gather",
        "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp#L770",
        "integer_position_table_domain": [-1, 226],
        "velocity_runtime_position_domain": [0, 226],
        "index": "original_integer_position_plus1;no_clipping_wrapping_or_fallback",
        "source_shapes": {"prefix_mask": [1, 177], "suffix_positions": [1, 50], "trig": [1, 50, 1, 32]},
        "table_bindings": [{"name": label, "dtype": "float32", "shape": [228, 32], "bytes": 29184, "sha256": digest} for label, digest in SOURCE_ROTARY_TABLE_WORDS.items()],
        "constant_runtime_bytes": 58368,
        "runtime_index_bytes": 400,
        "runtime_trig_output_bytes": 12800,
        "export": "copied_original_EP_plus_two_authenticated_lifted_constants;ordinary_public_ONNX_export",
        "admission": "exact_Boolean_prefix177_sum_plus_source_ones50_cumsum_minus1;or_original_subtract_min;D64_wavelength10000_FP32_frequency_DAG",
        "IEEE_admission": "RNE_and_FTZ_DAZ_clear_checked_in_external_proofs;adapter_does_not_set_or_guard_runtime_floating_state",
        "external_native_context_proof": {"receipt_sha256": "f5105723be6eef825777fa558fd7ff76e1b9435299c67ae8c02a2772f5e58959", "all_Boolean_prefix_counts": [0, 177], "alignment_float_offsets": [0, 1, 7, 15], "scalar_vector_tail_controls": True},
        "external_compiled_cut_proof": {"receipt_sha256": "2eb4b921f1199ad0b0958110f8637d70d33d508706f0354ee4b8703316a859de", "both_modes_fresh_VM_repeats": True},
        "unchanged": ["original_EP", "original_parameters_and_constants", "RoPE_Mul_Add_Sub_BF16_stores", "prefix_trig", "FP64_time_trig", "native_reference", "acceptance_gates"],
        "qualification": "bounded_velocity_source_specialization;not_portable_sine_algorithm_or_full_velocity_prefix_session_qualification",
        "bindings": bindings + model_bindings,
    }


def _source_rotary_tables(torch):
    """Recompute pure original source invariants; authenticate every FP32 word."""
    import hashlib
    exponents = (2.0 / 64) * torch.arange(32, dtype=torch.float32, device="cpu")
    timescale = 10000 ** exponents
    positions = torch.arange(-1, 227, dtype=torch.int64, device="cpu")
    angles = positions[:, None].to(torch.float32) / timescale[None, :].to(torch.float32)
    tables = {"sine": torch.sin(angles), "cosine": torch.cos(angles)}
    for label, table in tables.items():
        if table.dtype != torch.float32 or table.device.type != "cpu" or table.layout != torch.strided or tuple(table.shape) != (228, 32) or not table.is_contiguous():
            raise ValueError("Native rotary table geometry changed")
        if hashlib.sha256(table.detach().view(torch.uint8).numpy().tobytes()).hexdigest() != SOURCE_ROTARY_TABLE_WORDS[label]:
            raise ValueError("Native rotary table words do not match the independently proved CPU dispatch")
    return tables


def prepare_source_rotary_export(torch, source, policy="native"):
    """Lift native source invariants only for the exact original denoiser DAG."""
    record = source_rotary_contract(torch, policy)
    if record is None:
        return source, None
    import copy
    import operator
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument
    from torch.fx.immutable_collections import immutable_list
    if not isinstance(source, torch.export.ExportedProgram):
        raise ValueError("Source rotary export requires an original ExportedProgram")

    def tensor(node, shape, dtype):
        value = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
        return (isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.layout == torch.strided
                and value.dtype == dtype and len(value.shape) == len(shape)
                and all(type(actual) is int and actual == expected for actual, expected in zip(value.shape, shape))
                and all(type(stride) is int for stride in value.stride()) and value.is_contiguous())

    def call(node, target, argc=None):
        return isinstance(node, torch.fx.Node) and node.op == "call_function" and node.target == target and not node.kwargs and (argc is None or len(node.args) == argc)

    def integer(value, expected):
        return type(value) is int and value == expected

    def axis(node, target, dimension):
        return call(node, target, 2) and integer(node.args[1], dimension)

    def suffix_positions(node):
        if not tensor(node, (1, 50), torch.int64) or not call(node, torch.ops.aten.sub.Tensor, 2):
            return False
        if not integer(node.args[1], 1):
            base, minimum = node.args
            if not call(minimum, operator.getitem, 2) or not integer(minimum.args[1], 0):
                return False
            reduction = minimum.args[0]
            return call(reduction, torch.ops.aten.min.dim, 3) and reduction.args[0] is base and integer(reduction.args[1], 1) and reduction.args[2] is True and suffix_positions(base)
        addition = node.args[0]
        if not call(addition, torch.ops.aten.add.Tensor, 2) or not tensor(addition, (1, 50), torch.int64):
            return False
        count, cumulative = addition.args
        if not axis(count, torch.ops.aten.unsqueeze.default, 1) or not tensor(count, (1, 1), torch.int64):
            return False
        total = count.args[0]
        if not (call(total, torch.ops.aten.sum.dim_IntList, 2) and type(total.args[1]) in (list, immutable_list)
                and len(total.args[1]) == 1 and integer(total.args[1][0], -1) and tensor(total, (1,), torch.int64)):
            return False
        mask = total.args[0]
        if not tensor(mask, (1, 177), torch.bool) or mask.op != "placeholder":
            return False
        specs = [spec for spec in source.graph_signature.input_specs if isinstance(spec.arg, TensorArgument) and spec.arg.name == mask.name]
        if len(specs) != 1 or specs[0].kind != InputKind.USER_INPUT:
            return False
        if not axis(cumulative, torch.ops.aten.cumsum.default, 1) or not tensor(cumulative, (1, 50), torch.int64):
            return False
        concatenated = cumulative.args[0]
        if not (axis(concatenated, torch.ops.aten.cat.default, 1) and type(concatenated.args[0]) in (tuple, list, immutable_list)
                and len(concatenated.args[0]) == 1 and tensor(concatenated, (1, 50), torch.bool)):
            return False
        ones = concatenated.args[0][0]
        return (isinstance(ones, torch.fx.Node) and ones.op == "call_function" and ones.target == torch.ops.aten.ones.default
                and len(ones.args) == 1 and type(ones.args[0]) in (tuple, list, immutable_list) and len(ones.args[0]) == 2
                and integer(ones.args[0][0], 1) and integer(ones.args[0][1], 50)
                and ones.kwargs == {"dtype": torch.bool, "device": torch.device("cpu"), "pin_memory": False}
                and tensor(ones, (1, 50), torch.bool))

    def position_of(trig):
        if not (call(trig, torch.ops.aten.sin.default, 1) or call(trig, torch.ops.aten.cos.default, 1)):
            return None
        if not tensor(trig, (1, 50, 1, 32), torch.float32):
            return None
        outer = trig.args[0]
        if not axis(outer, torch.ops.aten.unsqueeze.default, 2) or not tensor(outer, (1, 50, 1, 32), torch.float32):
            raise ValueError("Rotary output-axis derivation changed")
        division = outer.args[0]
        if not call(division, torch.ops.aten.div.Tensor, 2) or not tensor(division, (1, 50, 32), torch.float32):
            raise ValueError("Rotary angle division changed")
        numerator, denominator = division.args
        if any(not call(value, torch.ops.aten.to.dtype, 2) or value.args[1] != torch.float32 for value in (numerator, denominator)):
            raise ValueError("Original FP32 angle casts changed")
        if not tensor(numerator, (1, 50, 1), torch.float32) or not tensor(denominator, (1, 1, 32), torch.float32):
            raise ValueError("Original FP32 angle operand geometry changed")
        unsqueezed = numerator.args[0]
        if not axis(unsqueezed, torch.ops.aten.unsqueeze.default, 2) or not suffix_positions(unsqueezed.args[0]):
            raise ValueError("Unproved rotary integer-position route")
        frequency = denominator.args[0]
        if not axis(frequency, torch.ops.aten.unsqueeze.default, 1) or not tensor(frequency, (1, 1, 32), torch.float32):
            raise ValueError("Rotary frequency axis changed")
        frequency = frequency.args[0]
        if not axis(frequency, torch.ops.aten.unsqueeze.default, 0) or not tensor(frequency, (1, 32), torch.float32):
            raise ValueError("Rotary frequency axis changed")
        power = frequency.args[0]
        if not call(power, torch.ops.aten.pow.Scalar, 2) or not integer(power.args[0], 10000) or not tensor(power, (32,), torch.float32):
            raise ValueError("Original rotary wavelength changed")
        exponent = power.args[1]
        if not (call(exponent, torch.ops.aten.mul.Tensor, 2) and type(exponent.args[1]) is float
                and exponent.args[1] == 0.03125 and tensor(exponent, (32,), torch.float32)):
            raise ValueError("Original rotary exponent changed")
        arange = exponent.args[0]
        if not (isinstance(arange, torch.fx.Node) and arange.op == "call_function" and arange.target == torch.ops.aten.arange.default
                and len(arange.args) == 1 and integer(arange.args[0], 32)
                and arange.kwargs == {"dtype": torch.float32, "device": torch.device("cpu"), "pin_memory": False}
                and tensor(arange, (32,), torch.float32)):
            raise ValueError("Original rotary frequency range changed")
        return unsqueezed.args[0]

    sites = [(node, position_of(node)) for node in source.graph.nodes]
    sites = [(node, position) for node, position in sites if position is not None]
    record.update(site_count=len(sites), added_constant_names=[], sites=[])
    if not sites:
        return source, record
    if len(sites) != 48:
        raise ValueError("Full denoiser rotary inventory changed")
    groups = {}
    for trig, position in sites:
        groups.setdefault(trig.args[0], []).append(trig)
    if len(groups) != 24 or any(len(pair) != 2 or {node.target for node in pair} != {torch.ops.aten.sin.default, torch.ops.aten.cos.default} for pair in groups.values()):
        raise ValueError("Original rotary sin/cos pairs changed")
    original_graph, original_signature = str(source.graph), str(source.graph_signature)
    original_user_specs = copy.deepcopy([spec for spec in source.graph_signature.input_specs if spec.kind == InputKind.USER_INPUT])
    original_output_specs = copy.deepcopy(source.graph_signature.output_specs)
    original_constant_names = set(source.constants)
    tables = _source_rotary_tables(torch)
    prepared = copy.deepcopy(source)
    graph = prepared.graph
    by_name = {node.name: node for node in graph.nodes}
    specs = copy.deepcopy(prepared.graph_signature.input_specs)
    first_user = next(index for index, spec in enumerate(specs) if spec.kind == InputKind.USER_INPUT)
    before = by_name[specs[first_user].arg.name]
    fake_mode = getattr(before.meta.get("val"), "fake_mode", None)
    if fake_mode is None:
        raise ValueError("Original EP FakeTensor metadata missing")
    constants = dict(prepared.constants)
    lifted = {}
    for label, table in tables.items():
        name = "merlin_native_denoise_rotary64_" + label
        if name in constants or name in prepared.state_dict or name in by_name:
            raise ValueError("Rotary constant name collision")
        with graph.inserting_before(before):
            placeholder = graph.placeholder(name)
        placeholder.meta["val"] = fake_mode.from_tensor(table)
        specs.insert(first_user, InputSpec(InputKind.CONSTANT_TENSOR, TensorArgument(placeholder.name), name))
        first_user += 1
        constants[name] = table.detach().clone()
        lifted[label] = placeholder
    for original, position in sites:
        node, position = by_name[original.name], by_name[position.name]
        label = "sine" if original.target == torch.ops.aten.sin.default else "cosine"
        with graph.inserting_before(node):
            index = graph.call_function(torch.ops.aten.add.Tensor, (position, 1))
            values = graph.call_function(torch.ops.aten.index.Tensor, (lifted[label], [index]))
        with fake_mode:
            index.meta["val"] = position.meta["val"] + 1
            values.meta["val"] = lifted[label].meta["val"][index.meta["val"]]
        node.target, node.args, node.kwargs = torch.ops.aten.unsqueeze.default, (values, 2), {}
        record["sites"].append({"node": original.name, "original_target": str(original.target), "position_node": position.name, "table": label})
    graph.lint()
    prepared.graph_module.recompile()
    signature = copy.deepcopy(prepared.graph_signature)
    signature.input_specs[:] = specs
    result = torch.export.ExportedProgram(prepared.graph_module, graph, signature, prepared.state_dict, prepared.range_constraints, prepared.module_call_graph, prepared.example_inputs, constants=constants, verifiers=prepared.verifiers)
    result.validate()
    added = sorted(set(constants) - original_constant_names)
    if added != sorted("merlin_native_denoise_rotary64_" + label for label in tables):
        raise AssertionError("Source rotary must lift exactly two authenticated constants")
    if [spec for spec in result.graph_signature.input_specs if spec.kind == InputKind.USER_INPUT] != original_user_specs or result.graph_signature.output_specs != original_output_specs:
        raise AssertionError("Original user/output graph signature changed")
    if str(source.graph) != original_graph or str(source.graph_signature) != original_signature or set(source.constants) != original_constant_names:
        raise AssertionError("Original EP changed")
    record.update(added_constant_names=added, original_user_input_count=len(original_user_specs), original_user_and_output_specs_preserved=True)
    return result, record


def source_scalar_contract(torch, policy="native"):
    """Bind the narrowly proved original CPU language scalar multiplication."""
    if policy == "native":
        return None
    if policy != "torch_cpu_language_scale960_v1":
        raise ValueError("Unknown source scalar export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_LANGUAGE_SCALE_ARTIFACTS, "Source language scale")
    return {
        "name": policy,
        "status": "opt_in_source_preserving_export_not_policy_qualified",
        "native": "original_BF16_CPU_embedding_1x48x960_times_original_Pythonfloat_sqrt960",
        "native_threads": 1,
        "export_shape": [1, 48, 960],
        "original_scalar_FP32_word": "0x41f7def6",
        "algorithm": "original_scalar_value_float_opmath;BF16_input_widen_FP32_Mul_single_BF16_store",
        "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L131-L147",
        "export": "copied_original_EP_explicit_opmath_then_ordinary_public_ONNX_export",
        "unchanged": ["original_EP", "parameters", "Tensor_other", "original_explicit_casts", "native_reference", "acceptance_gates"],
        "qualification": "actual_saved_language_branch_only;fullprefix_and_policy_unqualified",
        "value_admission": "finite_original_language_rows_checked_by_external_receipt;metadata_guard_does_not_check_runtime_values",
        "bindings": bindings,
    }


def prepare_source_scalar_export(torch, source, policy="native"):
    """Copy only the admitted language scalar site; keep native/no-site EPs."""
    record = source_scalar_contract(torch, policy)
    if record is None:
        return source, None
    if not isinstance(source, torch.export.ExportedProgram):
        raise ValueError("Source scalar export requires an original ExportedProgram")
    sites = []
    for node in source.graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.mul.Tensor:
            continue
        if len(node.args) != 2 or node.kwargs or type(node.args[1]) is not float or node.args[1] != math.sqrt(960):
            continue
        left = node.args[0]
        operand = left.meta.get("val") if isinstance(left, torch.fx.Node) else None
        if isinstance(operand, torch.Tensor) and operand.dtype == torch.bfloat16 and operand.device.type == "cpu" and tuple(operand.shape) == (1, 48, 960):
            sites.append(node)
    if len(sites) > 1:
        raise ValueError("Ambiguous source language scalar sites")
    record.update({
        "site_count": len(sites),
        "sites": [],
    })
    if not sites:
        return source, record
    original = sites[0]
    embedding = original.args[0]
    if embedding.op != "call_function" or embedding.target != torch.ops.aten.embedding.default or len(embedding.args) < 2:
        raise ValueError("Source language scale requires the original embedding producer")
    weight_node, token_node = embedding.args[:2]
    weight = weight_node.meta.get("val") if isinstance(weight_node, torch.fx.Node) else None
    tokens = token_node.meta.get("val") if isinstance(token_node, torch.fx.Node) else None
    if not (isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16 and weight.device.type == "cpu" and tuple(weight.shape) == (49280, 960)
            and isinstance(tokens, torch.Tensor) and tokens.dtype == torch.int64 and tokens.device.type == "cpu" and tuple(tokens.shape) == (1, 48)):
        raise ValueError("Source language embedding operand geometry changed")
    original_result = original.meta.get("val")
    original_operand = embedding.meta.get("val")
    if not (isinstance(original_result, torch.Tensor) and original_result.dtype == torch.bfloat16
            and original_result.device == original_operand.device and tuple(original_result.shape) == tuple(original_operand.shape)):
        raise ValueError("Original BF16 scalar Mul result metadata changed")
    original_mode = getattr(original_operand, "fake_mode", None)
    if original_mode is None or getattr(original_result, "fake_mode", None) is not original_mode:
        raise ValueError("Qualified original EP FakeTensor metadata missing")

    import copy
    original_graph = str(source.graph_module.graph)
    original_signature = str(source.graph_signature)
    prepared = copy.deepcopy(source)
    graph = prepared.graph_module.graph
    copied = [node for node in graph.nodes if node.name == original.name]
    if len(copied) != 1:
        raise ValueError("Copied source language scalar site inventory changed")
    node = copied[0]
    if node.op != "call_function" or node.target != original.target or node.kwargs or len(node.args) != 2:
        raise ValueError("Copied source language scalar operation changed")
    left, literal = node.args
    if not isinstance(left, torch.fx.Node) or left.name != embedding.name or type(literal) is not float or literal != original.args[1]:
        raise ValueError("Copied source language scalar operands changed")
    operand = left.meta.get("val")
    result = node.meta.get("val")
    fake_mode = getattr(operand, "fake_mode", None)
    if not (isinstance(operand, torch.Tensor) and operand.dtype == original_operand.dtype and operand.device == original_operand.device
            and tuple(operand.shape) == tuple(original_operand.shape) and isinstance(result, torch.Tensor)
            and result.dtype == original_result.dtype and result.device == original_result.device and tuple(result.shape) == tuple(original_result.shape)):
        raise ValueError("Copied source language scalar metadata changed")
    if fake_mode is None or getattr(result, "fake_mode", None) is not fake_mode:
        raise ValueError("Copied EP FakeTensor mode changed")
    users = tuple(node.users)
    with graph.inserting_before(node):
        widened = graph.call_function(torch.ops.aten._to_copy.default, (left,), {"dtype": torch.float32})
        scalar = graph.call_function(torch.ops.aten.scalar_tensor.default, (literal,), {
            "dtype": torch.float32,
            "device": operand.device,
        })
        product = graph.call_function(torch.ops.aten.mul.Tensor, (widened, scalar))
    with fake_mode:
        widened.meta["val"] = operand.to(torch.float32)
        scalar.meta["val"] = torch.scalar_tensor(literal, dtype=torch.float32, device=operand.device)
        product.meta["val"] = widened.meta["val"] * scalar.meta["val"]
    for inserted in (widened, scalar, product):
        for key in ("stack_trace", "nn_module_stack", "source_fn_stack"):
            if key in node.meta:
                inserted.meta[key] = copy.deepcopy(node.meta[key])
    # Retain the original BF16 result node and its users.
    node.target = torch.ops.aten._to_copy.default
    node.args = (product,)
    node.kwargs = {"dtype": torch.bfloat16}
    if tuple(node.users) != users or widened.args[0] is not left:
        raise ValueError("Original source language scalar users changed")
    record["sites"].append({
        "original_mul": node.name,
        "original_left_producer": left.name,
        "original_literal": literal,
        "original_users": [user.name for user in users],
        "original_explicit_cast_removed": False,
        "original_output_name_preserved": True,
        "inserted": [widened.name, scalar.name, product.name],
    })
    graph.lint()
    prepared.graph_module.recompile()
    prepared.validate()
    if str(source.graph_module.graph) != original_graph or str(source.graph_signature) != original_signature:
        raise ValueError("Source EP changed")
    if str(prepared.graph_signature) != original_signature:
        raise ValueError("Copied EP signature changed")
    return prepared, record

def source_mean_contract(torch, policy="native"):
    """Bind the native CPU mean used by the scoped Smol prefix/expert export."""
    if policy == "native":
        return None
    if policy not in ("torch_cpu_mean720_v1", "torch_cpu_mean720_960_v1"):
        raise ValueError("Unknown source mean export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_MEAN_ARTIFACTS, "Source mean")
    record = {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchMean720Fp32",
            "native": "unchanged_Tensor_mean_float32_last_axis_keepdim_Smol_expert_1x50x720_contiguous",
            "native_dispatch": "pinned_AVX2_eight_lane_sum;AVX512_sum_registration_null", "native_threads": 1,
            "export_shape": [1, 50, 720], "workspace": "65_FP32_private_output_words_260_bytes_plus_mean_output_and_reshape",
            "algorithm": "source_cascade_FP32_sum_then_direct_FP32_division_by720;no_reassociation_or_reciprocal_rewrite",
            "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/SumKernel.cpp",
            "unchanged": ["square", "epsilon", "rsqrt", "affine", "BF16_casts", "native_reference", "acceptance_gates"],
            "bindings": bindings}
    if policy == "torch_cpu_mean720_960_v1":
        record.pop("operator")
        record.pop("export_shape")
        record.update({"operators": ["TorchMean720Fp32", "TorchMean960Fp32"], "export_shapes": [[1, 50, 720], [1, 177, 960]],
                       "native": "unchanged_Tensor_mean_FP32_last_axis_keepdim_contiguous_Smol_expert720_or_prefix960",
                       "algorithm": "source_AVX2_four_stream_eight_lane_cascade;720_16plus6_groups_and_two_remainder_vectors;960_16plus14_groups_no_remainder;direct_FP32_division",
                       "qualification": "bounded_actual_and_independent_controls;full_prefix_and_policy_unqualified"})
    return record


def source_mean_translations(torch, policy="native"):
    """Export only the proved Smol mean geometry; preserve other translations."""
    if source_mean_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.onnx_types import FLOAT
    from onnxscript.values import Opset, Op
    source = Opset("merlin.source_preserving", 1)
    geometries = {(1, 50, 720): 720}
    if policy == "torch_cpu_mean720_960_v1":
        geometries[(1, 177, 960)] = 960
    operators = {}
    for width in geometries.values():
        name = f"TorchMean{width}Fp32"
        schema = defs.OpSchema(name, "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter("a", "T")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(float)"], "Bound source FP32 mean")])
        operators[width] = Op(source, name, schema)

    def mean_dim(self: FLOAT, dim: Optional[Sequence[int]] = None, keepdim: bool = False, dtype: int = -1) -> FLOAT:
        shape = tuple(self.shape) if self.shape is not None else ()
        if shape in geometries and self.dtype == ir.DataType.FLOAT and isinstance(dim, (tuple, list)) and tuple(dim) in ((-1,), (2,)) and keepdim and dtype in (None, -1, ir.DataType.FLOAT):
            return operators[geometries[shape]](self)
        return core.aten_mean_dim(self, dim, keepdim, dtype)

    return {torch.ops.aten.mean.dim: mean_dim}


def source_softmax_contract(torch, policy="native"):
    """Bind the scoped native CPU softmax; leave QK/PV and masks unchanged."""
    if policy == "native":
        return None
    if policy not in ("torch_cpu_softmax177_v1", "torch_cpu_softmax177_227_v1", "torch_cpu_softmax177_227_prefix_v1"):
        raise ValueError("Unknown source softmax export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_SOFTMAX_ARTIFACTS, "Source softmax")
    record = {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchSoftmaxAvx512F32LastDim177",
            "native": "unchanged_Tensor_softmax_float32_last_axis_Smol_expert_1x15x50x177_contiguous",
            "native_dispatch": "pinned_AVX512_sixteen_lane_softmax", "native_threads": 1,
            "export_shape": [1, 15, 50, 177],
            "algorithm": "source_lane_reduction_SLEEF_exp_explicit_FMA_then_FP32_reciprocal_multiply",
            "qualification": "bounded_saved_actual_and_controls;complete_native_attention_and_policy_unqualified",
            "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/SoftMaxKernel.cpp",
            "unchanged": ["QK", "PV", "masks", "BF16_casts", "native_reference", "acceptance_gates"],
            "bindings": bindings}
    if policy in ("torch_cpu_softmax177_227_v1", "torch_cpu_softmax177_227_prefix_v1"):
        record.pop("operator")
        record.pop("export_shape")
        record.update({"operators": ["TorchSoftmaxAvx512F32LastDim177", "TorchSoftmaxAvx512F32LastDim227"], "export_shapes": [[1, 15, 50, 177], [1, 15, 50, 227]], "native": "unchanged_FP32_Smol_manual_attention_softmax_width177_or227"})
    if policy == "torch_cpu_softmax177_227_prefix_v1":
        record["export_shapes"].append([1, 15, 177, 177])
        record.update({"native": "unchanged_FP32_Smol_expert_and_prefix_manual_attention_softmax", "prefix_native_strides": [469935, 31329, 177, 1], "prefix_native_storage_offset": 0, "qualification": "bounded_saved_actual_prefix_Q177_K177_and_independent_controls_raw_exact;complete_prefix_attention_and_policy_unqualified", "proof_admission": "source_contiguous_CPU_FP32;finite_logits_and_shifted_subtractions;positive_finite_sum;RNE_gradual_underflow"})
    return record


def source_softmax_translations(torch, policy="native"):
    """Export only the proved FP32 Smol shape and last axis."""
    if source_softmax_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.onnx_types import FLOAT
    from onnxscript.values import Opset, Op
    widths = (177,) if policy == "torch_cpu_softmax177_v1" else (177, 227)
    ops = {}
    for width in widths:
        name = f"TorchSoftmaxAvx512F32LastDim{width}"
        schema = defs.OpSchema(name, "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter("a", "T")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(float)"], "Bound source FP32 softmax")])
        ops[(1, 15, 50, width)] = Op(Opset("merlin.source_preserving", 1), name, schema)
    if policy == "torch_cpu_softmax177_227_prefix_v1":
        ops[(1, 15, 177, 177)] = ops[(1, 15, 50, 177)]

    def softmax(self: FLOAT, dim: int, dtype: int = -1) -> FLOAT:
        selected = ops.get(tuple(self.shape or ()))
        if selected is not None and dim in (-1, 3) and dtype in (None, -1, ir.DataType.FLOAT):
            return selected(self)
        return core.aten_softmax(self, dim, dtype)

    def internal_softmax(self: FLOAT, dim: int, half_to_float: bool) -> FLOAT:
        selected = ops.get(tuple(self.shape or ()))
        if selected is not None and dim in (-1, 3) and not half_to_float:
            return selected(self)
        return core.aten__softmax(self, dim, half_to_float)

    return {torch.ops.aten.softmax.int: softmax, torch.ops.aten._softmax.default: internal_softmax}


def source_layernorm_contract(torch, policy="native"):
    """Bind the single-output BF16 vision LayerNorm's native CPU arithmetic."""
    if policy == "native":
        return None
    if policy != "torch_cpu_layernorm768_v1":
        raise ValueError("Unknown source LayerNorm export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_LAYERNORM_ARTIFACTS, "Source LayerNorm")
    return {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchLayerNormAvx2Bf16Width768",
            "native": "unchanged_single_output_BF16_Smol_vision_LayerNorm_1x1024x768_original_affines_epsilon1e-6",
            "native_dispatch": "pinned_AVX2_BF16_sixteen_lane_FP32_eight_lane_moments;AVX512_registration_null",
            "native_threads": 1, "export_shape": [1, 1024, 768], "epsilon_fp32_word": "0x358637bd",
            "algorithm": "source_Welford_cascade_scalar_merge_sqrt_reciprocal_and_two_affine_FMAs",
            "qualification": "bounded_saved_actual_and_controls;complete_vision_prefix_and_policy_unqualified",
            "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp",
            "unchanged": ["parameters", "epsilon", "native_reference", "acceptance_gates", "ordinary_ONNX_LayerNormalization"],
            "bindings": bindings}


def source_layernorm_translations(torch, policy="native"):
    """Preserve only the proved native BF16 vision shape and original epsilon."""
    if source_layernorm_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir, opset18 as op
    from onnxscript.onnx_types import BFLOAT16
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.values import Opset, Op
    schema = defs.OpSchema("TorchLayerNormAvx2Bf16Width768", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter(name, "T" if name != "epsilon" else "F") for name in ("input", "weight", "bias", "epsilon")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(bfloat16)"], "Original BF16 operands"), ("F", ["tensor(float)"], "Original FP32 kernel epsilon")])
    layernorm_op = Op(Opset("merlin.source_preserving", 1), "TorchLayerNormAvx2Bf16Width768", schema)
    fallback = translations(torch)[torch.ops.aten.layer_norm.default]

    def layer_norm(input: BFLOAT16, normalized_shape: Sequence[int], weight: Optional[TFloat] = None, bias: Optional[TFloat] = None, eps: float = 1e-5, cudnn_enable: bool = True) -> BFLOAT16:
        if tuple(input.shape or ()) == (1, 1024, 768) and tuple(normalized_shape) == (768,) and eps == 1e-6 and all(value is not None and value.dtype == ir.DataType.BFLOAT16 and tuple(value.shape or ()) == (768,) for value in (weight, bias)):
            epsilon = op.Constant(value=ir.tensor(eps, dtype=ir.DataType.FLOAT))
            return layernorm_op(input, weight, bias, epsilon)
        return fallback(input, normalized_shape, weight, bias, eps, cudnn_enable)

    return {torch.ops.aten.layer_norm.default: layer_norm}


def source_conv_contract(torch, policy="native"):
    """Bind the proved software-BF16 native patch convolution dispatch."""
    if policy == "native":
        return None
    if policy != "torch_cpu_conv3x16_v1":
        raise ValueError("Unknown source convolution export policy")
    import os
    overrides = ("ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "ONEDNN_DEFAULT_FPMATH_MODE", "DNNL_DEFAULT_FPMATH_MODE", "ATEN_CPU_CAPABILITY")
    if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled or torch.cpu._is_avx512_bf16_supported() or torch.cpu._is_amx_tile_supported() or any(name in os.environ for name in overrides):
        raise ValueError("Source convolution requires the proved software-BF16 oneDNN dispatch without ISA/fpmath overrides")
    bindings = _source_cpu_bindings(torch, SOURCE_CONV_ARTIFACTS, "Source convolution")
    return {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchConv2dAvx512Bf16IC3Kernel16Stride16",
            "native": "unchanged_BF16_Smol_patch_convolution_1x3x512x512_original_768x3x16x16_weight_and_768_bias",
            "native_dispatch": "oneDNN3.7.1_jit_bf16_avx512_core_software_BF16", "native_threads": 1,
            "export_shape": [1, 3, 512, 512], "workspace": "32_FP32_private_words_128_bytes_plus_output",
            "algorithm": "KH_KW_channel1_channel0_zero_partner_channel2_explicit_FP32_FMA_then_original_bias_add_and_one_BF16_RNE_cast",
            "qualification": "bounded_actual_and_controls;finite_normal_product_scope;complete_vision_prefix_and_policy_unqualified",
            "source": "https://github.com/oneapi-src/oneDNN/blob/8d263e693366ef8db40acc569cc7d8edf644556d/src/cpu/x64/jit_avx512_core_bf16_conv_kernel.cpp",
            "unchanged": ["parameters", "stride", "padding", "dilation", "groups", "native_reference", "acceptance_gates", "ordinary_ONNX_Conv"],
            "bindings": bindings}


def source_conv_translations(torch, policy="native"):
    """Export the proved patch geometry; keep other convolutions ordinary."""
    if source_conv_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.onnx_types import BFLOAT16
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.values import Opset, Op
    schema = defs.OpSchema("TorchConv2dAvx512Bf16IC3Kernel16Stride16", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter(name, "T") for name in ("input", "weight", "bias")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(bfloat16)"], "Original BF16 patch operands")])
    conv_op = Op(Opset("merlin.source_preserving", 1), schema.name, schema)

    def pair(value):
        if isinstance(value, int):
            return (value, value)
        values = tuple(value) if isinstance(value, (tuple, list)) else ()
        return values * 2 if len(values) == 1 else values

    def admitted(input, weight, bias, stride, padding, dilation, groups):
        return all(value is not None and value.dtype == ir.DataType.BFLOAT16 and tuple(value.shape or ()) == shape for value, shape in ((input, (1, 3, 512, 512)), (weight, (768, 3, 16, 16)), (bias, (768,)))) and pair(stride) == (16, 16) and pair(padding) == (0, 0) and pair(dilation) == (1, 1) and groups == 1

    def conv2d(input: BFLOAT16, weight: TFloat, bias: Optional[TFloat] = None, stride: Sequence[int] = (1, 1), padding: Sequence[int] = (0, 0), dilation: Sequence[int] = (1, 1), groups: int = 1) -> BFLOAT16:
        if admitted(input, weight, bias, stride, padding, dilation, groups):
            return conv_op(input, weight, bias)
        return core.aten_conv2d(input, weight, bias, stride, padding, dilation, groups)

    def convolution(input: BFLOAT16, weight: TFloat, bias: Optional[TFloat] = None, stride: Sequence[int] = (1,), padding: Sequence[int] = (0,), dilation: Sequence[int] = (1,), transposed: bool = False, output_padding: Sequence[int] = (0,), groups: int = 1) -> BFLOAT16:
        if not transposed and pair(output_padding) == (0, 0) and admitted(input, weight, bias, stride, padding, dilation, groups):
            return conv_op(input, weight, bias)
        return core.aten_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

    return {torch.ops.aten.conv2d.default: conv2d, torch.ops.aten.convolution.default: convolution}


def source_sgemm_contract(torch, policy="native"):
    """Bind the scoped native attention kernels, leaving scaling intact."""
    if policy == "native":
        return None
    if policy not in ("torch_cpu_sgemm177_v1", "torch_cpu_attention177_227_v1", "torch_cpu_attention177_227_prefix_v1"):
        raise ValueError("Unknown source SGEMM export policy")
    import os
    if not torch.backends.mkl.is_available() or torch.get_float32_matmul_precision() != "highest" or any(name in os.environ for name in ("MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS", "MKL_DEBUG_CPU_TYPE", "ATEN_CPU_CAPABILITY")):
        raise ValueError("Source SGEMM requires the proved highest-precision MKL dispatch without ISA/consistency overrides")
    artifacts = SOURCE_SGEMM_ARTIFACTS
    if policy in ("torch_cpu_attention177_227_v1", "torch_cpu_attention177_227_prefix_v1"):
        overrides = ("ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "ONEDNN_DEFAULT_FPMATH_MODE", "DNNL_DEFAULT_FPMATH_MODE")
        if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled or torch.cpu._is_avx512_bf16_supported() or torch.cpu._is_amx_tile_supported() or any(name in os.environ for name in overrides):
            raise ValueError("Source attention requires the proved software-BF16 oneDNN dispatch without ISA/fpmath overrides")
        artifacts = {**artifacts, **SOURCE_CONV_ARTIFACTS}
    bindings = _source_cpu_bindings(torch, artifacts, "Source SGEMM")
    record = {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1,
            "operators": ["TorchMklSgemmF32QK177", "TorchMklSgemmF32PV177"],
            "native": "unchanged_FP32_manual_Smol_denoiser_Q50_K177_QK_and_PV_matmul",
            "native_dispatch": "observed_AVX512_copied_QK_kernel_0_b0_and_nocopy_NN_PV", "native_threads": 1,
            "qk_native_strides": [[48000, 64, 960, 1], [169920, 64, 1, 960]],
            "pv_native_strides": [[132750, 8850, 177, 1], [169920, 64, 960, 1]],
            "algorithm": "positive_zero_then_increasing_k_explicit_FP32_FMA;native_alpha1_final_multiply_is_identity_in_admitted_scope",
            "qualification": "bounded_actual_and_controls;finite_RNE_gradual_underflow;native_physical_layouts_bound_by_capture;complete_attention_and_policy_unqualified",
            "unchanged": ["parameters", "QK_scale", "masks", "softmax", "BF16_casts", "native_reference", "acceptance_gates", "ordinary_ONNX_MatMul"],
            "bindings": bindings}
    if policy in ("torch_cpu_attention177_227_v1", "torch_cpu_attention177_227_prefix_v1"):
        record["operators"] += ["TorchMklSgemmF32QK227", "TorchOnednnGemmBf16PV227"]
        record.update({"native": "unchanged_manual_Smol_Q50_K177_or227_attention_contractions", "native_dispatch": "MKL_AVX512_FP32_QK177_QK227_PV177;oneDNN3.7.1_gemm_jit_software_BF16_PV227", "qk227_native_strides": [[48000, 64, 960, 1], [217920, 64, 1, 960]], "pv227_native_strides": [[170250, 11350, 227, 1], [217920, 64, 960, 1]], "pv227_algorithm": "positive_zero_then_pair_high_low_explicit_FP32_FMA;odd_tail_FMA_positive_zero_times_last_probability_then_last_product;one_BF16_RNE_store", "pv227_evidence": "all_four_installed_JIT_variants_and_isolated_copy_packing;original_selected_variant_and_internal_buffers_not_observed;driver_alpha1_beta0_inferred_from_ordinary_matmul_semantics"})
    if policy == "torch_cpu_attention177_227_prefix_v1":
        record["operators"].append("TorchOnednnGemmBf16PV177")
        record.update({
            "native": "unchanged_manual_Smol_expert_and_prefix_attention_contractions",
            "native_dispatch": "MKL_AVX512_FP32_QK177_QK227_PV177;oneDNN3.7.1_software_BF16_source_family_PV227_and_prefix_PV177",
            "prefix_pv_native_strides": [[469935, 31329, 177, 1], [169920, 64, 960, 1]],
            "prefix_pv_native_storage_offsets": [0, 0],
            "prefix_pv_shape": [[1, 15, 177, 177], [1, 15, 177, 64]],
            "prefix_pv_value_origin": "original_cat7_BF16_1x177x5x64_unsqueeze3_expand_heads3_reshape15_permute0213",
            "prefix_pv_algorithm": "FP32_positive_zero;88pairs_highK_then_lowK_explicit_FMA;odd_tail_FMA_positive_zero_times_last_probability_then_last_product;one_BF16_RNE_store",
            "prefix_pv_evidence": "all_four_installed_JIT_variants_and_isolated_copy_packing;original_selected_variant_and_internal_buffers_unobserved;driver_alpha1_beta0_inferred_from_ordinary_matmul_semantics",
            "proof_admission": "finite_BF16_operands_FP32_products_and_every_FMA_intermediate;RNE_gradual_underflow;proof_conditions_not_exported_runtime_guards",
            "prefix_pv_layout_evidence": "captured_original_source_route_not_symbolic_IR_physical_layout_enforcement",
            "qualification": "bounded_actual_prefix_BF16_PV177_and_controls_raw_exact;finite_RNE_gradual_underflow;captured_original_physical_layouts;complete_prefix_attention_and_policy_unqualified",
        })
    return record


def source_sgemm_translations(torch, policy="native"):
    """Preserve only the proved manual-attention contraction geometries."""
    record = source_sgemm_contract(torch, policy)
    if record is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.values import Opset, Op
    source = Opset("merlin.source_preserving", 1)
    ops = {}
    for name in record["operators"]:
        dtype = "tensor(bfloat16)" if name in ("TorchOnednnGemmBf16PV177", "TorchOnednnGemmBf16PV227") else "tensor(float)"
        schema = defs.OpSchema(name, source.domain, 1, inputs=[defs.OpSchema.FormalParameter(value, "T") for value in ("left", "right")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", [dtype], "Original contraction operands")])
        ops[name] = Op(source, name, schema)
    geometries = {
        (ir.DataType.FLOAT, (1, 15, 50, 64), (1, 15, 64, 177)): "TorchMklSgemmF32QK177",
        (ir.DataType.FLOAT, (1, 15, 50, 177), (1, 15, 177, 64)): "TorchMklSgemmF32PV177",
        (ir.DataType.FLOAT, (1, 15, 50, 64), (1, 15, 64, 227)): "TorchMklSgemmF32QK227",
        (ir.DataType.BFLOAT16, (1, 15, 50, 227), (1, 15, 227, 64)): "TorchOnednnGemmBf16PV227",
        (ir.DataType.BFLOAT16, (1, 15, 177, 177), (1, 15, 177, 64)): "TorchOnednnGemmBf16PV177",
    }

    def matmul(self: TFloat, other: TFloat) -> TFloat:
        key = (self.dtype, tuple(self.shape or ()), tuple(other.shape or ()))
        selected = ops.get(geometries.get(key)) if self.dtype == other.dtype else None
        return selected(self, other) if selected is not None else core.aten_matmul(self, other)

    return {torch.ops.aten.matmul.default: matmul}


def bf16_linear_error_diagnostic(input, weight, bias, torch):
    """A conditional worst-case CPU arithmetic diagnostic, never an active gate.

    Covers BF16 operands, FP32 multiply/add/reduction, optional FP32 bias add,
    and ONE final BF16 rounding. It does not infer that a native source kernel
    follows this operational contract. FP64 evaluation is a labeled diagnostic
    and its own roundoff is included; it is not an exact real-valued oracle.
    """
    operands = (input, weight) + (() if bias is None else (bias,))
    if any(value.device.type != "cpu" or value.dtype != torch.bfloat16
           or not torch.isfinite(value).all() for value in operands):
        raise ValueError("Linear diagnostic requires finite original BF16 CPU operands")
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1] or \
            bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("Linear diagnostic shape mismatch")
    width = weight.shape[1]
    if width <= 0:
        raise ValueError("Linear diagnostic requires a nonempty reduction")
    operations = 2 * width + (1 if bias is not None else 0)

    def gamma(dtype):
        unit = torch.finfo(dtype).eps / 2
        if operations * unit >= 1:
            raise ValueError("Reduction length exceeds this floating-point error model")
        return operations * unit / (1 - operations * unit)

    with torch.no_grad():
        left, right = input.double(), weight.double().T
        ideal = left @ right
        magnitude = left.abs() @ right.abs()
        if bias is not None:
            ideal = ideal + bias.double()
            magnitude = magnitude + bias.double().abs()
        g32, g64 = gamma(torch.float32), gamma(torch.float64)
        # An upper bound on sum(abs(products))+abs(bias), accounting for FP64
        # roundoff in the diagnostic's magnitude computation.
        magnitude_upper = magnitude / (1 - g64)
        if not torch.isfinite(magnitude_upper).all() or \
                torch.any(magnitude_upper * (1 + g32) >= torch.finfo(torch.float32).max):
            raise ValueError("Diagnostic cannot exclude FP32 intermediate overflow")
        minimum_subnormal32 = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
        underflow = operations * minimum_subnormal32 / (2 * (1 - operations * torch.finfo(torch.float32).eps / 2))
        arithmetic = (g32 + g64) * magnitude_upper + underflow
        # Relative rounding plus a subnormal absolute term. This avoids an
        # arbitrary near-zero floor or a fixed number of BF16 representable steps.
        bf16 = torch.finfo(torch.bfloat16)
        rounding = (bf16.eps / 2) * (ideal.abs() + arithmetic) + bf16.tiny * bf16.eps / 2
        bound = arithmetic + rounding
        rounded = ideal.bfloat16()
        if not torch.isfinite(rounded).all():
            raise ValueError("Diagnostic reference overflows BF16 at the declared output cast")
    return {
        "higher_precision_value": ideal, "bf16_rounded_diagnostic": rounded,
        "absolute_error_bound_to_fp64_diagnostic": bound,
        "fp32_arithmetic_error_bound": arithmetic, "sum_absolute_products_and_bias_upper": magnitude_upper,
        "diagnostic_only": True, "acceptance_contract_selected": False,
        "assumptions": "BF16 exact operands; FP32 products/reduction/bias; one BF16 output cast; IEEE rounding; no overflow; gradual underflow",
        "roundoff_model": "gamma_(2K+bias) sum(abs(products))+abs(bias), plus FP64 diagnostic roundoff and subnormal terms",
        "reduction_width": width, "operation_bound": operations,
        "placement": "CPU; regular Gemmini INT8/INT32 is outside this diagnostic",
    }


def scalar_translations(torch):
    """Preserve original Python scalars when materializing DOUBLE constants.

    The scalar attribute annotation avoids the built-in TensorType argument's
    implicit FLOAT constant before its DOUBLE cast. Other ordinary dtypes keep
    the built-in translation. This does not select floating arithmetic order.
    """
    from onnxscript import ir, opset18 as op
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.onnx_types import TensorType

    def scalar_tensor(s: float, dtype: int = -1, layout: str = "", device: str = "", pin_memory: bool = False) -> TensorType:
        # This annotation marks an attribute; the exporter retains Python int
        # and bool arguments too. Never numerically convert those through float.
        if type(s) not in (int, float, bool):
            raise ValueError("Scalar translation requires an original static Python number")
        if layout not in (None, "", 0, "strided", "torch.strided"):
            raise ValueError("Scalar translation requires default strided layout")
        selected_device = str(torch.get_default_device()) if device in (None, "") else str(device)
        if selected_device not in ("cpu", "cpu:0"):
            raise ValueError("Scalar translation requires default or explicit CPU device")
        if pin_memory:
            raise ValueError("Scalar translation cannot preserve pinned allocation")
        if dtype in (None, -1):
            defaults = {torch.float16: ir.DataType.FLOAT16, torch.bfloat16: ir.DataType.BFLOAT16, torch.float32: ir.DataType.FLOAT, torch.float64: ir.DataType.DOUBLE}
            dtype = defaults[torch.get_default_dtype()]
        selected = ir.DataType(dtype)
        if selected not in (ir.DataType.FLOAT16, ir.DataType.FLOAT, ir.DataType.DOUBLE, ir.DataType.BFLOAT16, ir.DataType.INT8, ir.DataType.INT16, ir.DataType.INT32, ir.DataType.INT64, ir.DataType.UINT8, ir.DataType.BOOL):
            raise ValueError("Scalar translation requires a supported ordinary real dtype")
        if selected == ir.DataType.DOUBLE:
            return op.Constant(value=ir.tensor(s, dtype=selected))
        return core.aten_scalar_tensor(s, dtype=selected, layout=layout, device=device, pin_memory=False)

    return {torch.ops.aten.scalar_tensor.default: scalar_tensor}


def translations(torch, source_flash_policy="native", source_silu_policy="native"):
    # The StageCompiler must authenticate the original EP mask DAG before export.
    flash_record = source_flash_contract(torch, source_flash_policy)
    silu_record = source_silu_contract(torch, source_silu_policy)
    # Optional exporter dependencies are loaded only when this recipe is selected.
    from onnxscript import ir, opset18 as op
    from onnxscript.function_libs.torch_lib.ops import nn
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.onnx_types import BFLOAT16, TensorType
    flash_op = None
    if flash_record is not None:
        from onnx import defs
        from onnxscript.values import Opset, Op
        schema = defs.OpSchema("TorchFlashBf16Vision1024", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter(name, "T") for name in ("query", "key", "value", "mask")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(bfloat16)"], "Original BF16 vision Flash operands")])
        flash_op = Op(Opset("merlin.source_preserving", 1), "TorchFlashBf16Vision1024", schema)

    def silu(self: BFLOAT16) -> BFLOAT16:
        # Fused BF16 SiLU has one logical output cast, not a BF16 sigmoid store.
        if self.dtype != ir.DataType.BFLOAT16:
            raise ValueError("Fused SiLU translation requires BF16 dispatcher input")
        widened = op.Cast(self, to=1)
        return op.Cast(op.Mul(widened, op.Sigmoid(widened)), to=16)

    source_silu_op = None
    if silu_record is not None:
        from onnx import defs
        from onnxscript.values import Opset, Op
        schema = defs.OpSchema("TorchSiluAvx512F32", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter("self", "T")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(float)"], "Original FP32 vector SiLU operand")])
        source_silu_op = Op(Opset("merlin.source_preserving", 1), "TorchSiluAvx512F32", schema)

    def source_silu(self: TFloat) -> TFloat:
        if self.dtype == ir.DataType.BFLOAT16:
            return silu(self)
        if self.dtype != ir.DataType.FLOAT or self.shape is None or not self.shape or any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise ValueError("Source SiLU translation requires the admitted static FP32 tensor")
        count = math.prod(self.shape)
        if count % 32 or count > ((1 << 63) - 1) // 4:
            raise ValueError("Source SiLU translation requires a paired32 element count within the FP32 byte span")
        flat = op.Reshape(self, op.Constant(value_ints=[count]))
        return op.Reshape(source_silu_op(flat), op.Constant(value_ints=list(self.shape)))

    def linear(input: TFloat, weight: TFloat, bias: Optional[TFloat] = None) -> TFloat:
        if input.dtype != ir.DataType.BFLOAT16:
            return nn.aten_linear(input, weight, bias)
        if weight.dtype != input.dtype or bias is not None and bias.dtype != input.dtype:
            raise ValueError("Fused BF16 linear requires matching operand dtypes")
        if len(weight.shape) != 2 or len(input.shape) < 1:
            raise ValueError("Fused BF16 linear requires a matrix weight and a non-scalar input")
        product = op.MatMul(op.Cast(input, to=1), op.Transpose(op.Cast(weight, to=1), perm=[1, 0]))
        if bias is not None:
            product = op.Add(product, op.Cast(bias, to=1))
        return op.Cast(product, to=16)

    def layer_norm(input: BFLOAT16, normalized_shape: Sequence[int], weight: Optional[TFloat] = None, bias: Optional[TFloat] = None, eps: float = 1e-5, cudnn_enable: bool = True) -> BFLOAT16:
        # A BF16-only signature lets the exporter retain its built-in FP32 path.
        # Register the observed single-output ATen op; native_layer_norm's CPU
        # auxiliary statistics can be BF16 and are outside this translation.
        if input.dtype != ir.DataType.BFLOAT16:
            raise ValueError("LayerNorm translation requires BF16 dispatcher input")
        if not normalized_shape or any(type(value) is not int or value <= 0 for value in normalized_shape) or input.shape is None or len(input.shape) < len(normalized_shape):
            raise ValueError("BF16 LayerNorm requires a known rank and positive static normalized shape")
        trailing_shape = tuple(input.shape)[-len(normalized_shape):]
        if any(type(value) is not int for value in trailing_shape) or trailing_shape != tuple(normalized_shape):
            raise ValueError("BF16 LayerNorm normalized shape must equal the static trailing input dimensions")
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("BF16 LayerNorm requires a finite nonnegative epsilon")
        if any(value is not None and value.dtype != ir.DataType.BFLOAT16 for value in (weight, bias)):
            raise ValueError("BF16 LayerNorm requires original BF16 scale/bias")
        if any(value is not None and (value.shape is None or tuple(value.shape) != tuple(normalized_shape)) for value in (weight, bias)):
            raise ValueError("BF16 LayerNorm affine shapes must equal the normalized shape")
        # Explicit FLOAT standardization and affine have no intervening BF16
        # normalization cast. Original BF16 parameters widen exactly. The
        # decomposition also supports omitted affine and all trailing axes.
        axes = list(range(-len(normalized_shape), 0))
        widened = op.Cast(input, to=1)
        centered = op.Sub(widened, op.ReduceMean(widened, axes, keepdims=1))
        variance = op.ReduceMean(op.Mul(centered, centered), axes, keepdims=1)
        output = op.Div(centered, op.Sqrt(op.Add(variance, eps)))
        if weight is not None:
            output = op.Mul(output, op.Cast(weight, to=1))
        if bias is not None:
            output = op.Add(output, op.Cast(bias, to=1))
        return op.Cast(output, to=16)

    def attention(query: TFloat, key: TFloat, value: TFloat, attn_mask: Optional[TensorType] = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False) -> TFloat:
        if query.dtype != ir.DataType.BFLOAT16:
            return nn.aten_scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("Fused BF16 attention requires matching Q/K/V dtypes")
        if flash_record is not None and query.shape is not None and tuple(query.shape) == (1, 12, 1024, 64):
            if any(tensor.shape is None or tuple(tensor.shape) != (1, 12, 1024, 64) for tensor in (key, value)) or attn_mask is None or attn_mask.dtype != ir.DataType.BFLOAT16 or tuple(attn_mask.shape) != (1, 1, 1024, 1024):
                raise ValueError("Source Flash requires proved vision operands and the original zero-mask geometry")
            if dropout_p != 0.0 or is_causal or enable_gqa or scale not in (None, 0.125):
                raise ValueError("Source Flash inference flags or scale changed")
            return flash_op(query, key, value, attn_mask)
        if any(len(tensor.shape) != 4 for tensor in (query, key, value)):
            raise ValueError("Fused BF16 attention requires rank-four Q/K/V")
        if dropout_p != 0.0 or enable_gqa:
            raise ValueError("Fused BF16 attention supports inference without dropout or grouped-query expansion")
        if is_causal and attn_mask is not None:
            raise ValueError("Fused BF16 attention cannot combine a causal flag and an explicit mask")
        width = query.shape[-1]
        if scale is None and (not isinstance(width, int) or width <= 0):
            raise ValueError("Fused BF16 attention requires a static positive head width for its default scale")
        factor = 1.0 / math.sqrt(width) if scale is None else scale
        logits = op.Mul(op.MatMul(op.Cast(query, to=1), op.Transpose(op.Cast(key, to=1), perm=[0, 1, 3, 2])), factor)
        if is_causal:
            rows = op.Gather(op.Shape(query), 2, axis=0)
            columns = op.Gather(op.Shape(key), 2, axis=0)
            row = op.Unsqueeze(op.Range(0, rows, 1), [1])
            column = op.Unsqueeze(op.Range(0, columns, 1), [0])
            logits = op.Where(op.LessOrEqual(column, row), logits, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == ir.DataType.BOOL:
                logits = op.Where(attn_mask, logits, float("-inf"))
            elif attn_mask.dtype in (ir.DataType.BFLOAT16, ir.DataType.FLOAT):
                logits = op.Add(logits, op.Cast(attn_mask, to=1))
            else:
                raise ValueError("Fused BF16 attention requires a boolean, BF16, or FP32 mask")
        if is_causal or attn_mask is not None:
            # Native SDPA returns zero for fully masked rows rather than NaN.
            valid = op.Cast(op.Not(op.Equal(logits, float("-inf"))), to=7)
            blocked = op.Equal(op.ReduceSum(valid, [-1], keepdims=1), 0)
            logits = op.Where(blocked, 0.0, logits)
        probabilities = op.Softmax(logits, axis=-1)
        if is_causal or attn_mask is not None:
            probabilities = op.Where(blocked, 0.0, probabilities)
        return op.Cast(op.MatMul(probabilities, op.Cast(value, to=1)), to=16)

    return {torch.ops.aten.linear.default: linear, torch.ops.aten.scaled_dot_product_attention.default: attention, torch.ops.aten.layer_norm.default: layer_norm, torch.ops.aten.silu.default: source_silu if silu_record is not None else silu}
