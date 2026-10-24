"""Capture native FP32 ResNet contractions from the frozen development stream.

This workload edge preserves the original source. Target encodings, arithmetic,
launch support and performance evaluation belong to the companion provider.
"""

from merlin.common.digest import sha256_bytes
from merlin.common.jsonio import write_pretty_json

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

from merlin.common.paths import artifacts_dir, repo_root

# This explicit example sibling owns shared source admission; no target code is imported.
sys.path.insert(0, str(Path(__file__).parent.parent))
from capture_tinyllama_native import identity, safe_path
import source_relocation
from source_relocation import SourceRelocation


RECEIPT_SHA256 = "7630b69889495a657a2396fcfe2cee0a909cd6f6b7aa92e5390095e47c1d9f8a"
# This identifies the historical verifier declared by the frozen receipt. It is
# separate from the current native helper source identity recorded at admission.
HISTORICAL_VERIFIER_SHA256 = "5ca2ca510969d9f8237183ae028d49bb9a8d1ba9c7fbfb8cff11e4ffbe8a406b"
PREPARATION_SHA256 = "8c0876d4092485b3e6686500c0dd2b833767861acfba75f452926ec3ceff1c85"


def source_relocator(args):
    relocation = getattr(args, '_source_relocation', None)
    if relocation is None:
        relocation = SourceRelocation(getattr(args, 'relocate', ()))
        args._source_relocation = relocation
    return relocation


def read_source_metadata(args, report, path, digest):
    relocation = source_relocator(args)
    document, _ = relocation.read_json(path, digest)
    report['source_relocation'] = relocation.report()
    return document


def relocation_source_identity():
    return identity(source_relocation.__file__)


def require_identity(record):
    actual = identity(record["path"])
    if actual["sha256"] != record["sha256"]:
        raise ValueError("Source binding changed: " + actual["path"])
    return actual


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_native_admission(receipt, records, report):
    """Preserve historical receipt provenance and bind the current native helper."""
    historical = receipt["verifier_source"]
    if historical["sha256"] != HISTORICAL_VERIFIER_SHA256:
        raise ValueError("Frozen receipt declares a different historical verifier")
    helper_path = Path(__file__).with_name("resnet_native.py")
    helper_id = identity(helper_path)
    helper = load_module("atlas_resnet_native_admission", helper_path)
    require_identity(helper_id)
    records.append(helper_id)
    report["native_admission"] = {"historical_verifier_source": dict(historical),
                                "current_helper": helper_id}
    return helper


def git_head(path):
    return subprocess.check_output(["git", "-C", str(safe_path(path)), "rev-parse", "HEAD"], text=True).strip()


def tensor_identity(value):
    array = value.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "raw_sha256": sha256_bytes(array.tobytes())}


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    write_pretty_json(temporary, manifest, allow_nan=False)
    temporary.replace(path)


def patch_rows(activation, start, count, np, padding=1):
    """A[p,(c*3+kh)*3+kw], with explicit out-of-image zeros."""
    if activation.ndim != 4 or activation.shape[0] != 1:
        raise ValueError("Patch extraction requires one NCHW image")
    _, channels, height, width = activation.shape
    if start < 0 or count <= 0 or start + count > height * width:
        raise ValueError("Patch rows lie outside the native spatial output")
    positions = np.arange(start, start + count)
    h, w = positions // width, positions % width
    result = np.zeros((count, channels * 9), dtype=activation.dtype)
    for channel in range(channels):
        for kh in range(3):
            for kw in range(3):
                ih, iw = h + kh - padding, w + kw - padding
                valid = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
                result[valid, (channel * 3 + kh) * 3 + kw] = activation[0, channel, ih[valid], iw[valid]]
    return result

def require_same_patches(actual, expected, np):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual.view(np.uint32),
        expected.view(np.uint32)):
        raise ValueError("Explicit patch layout differs from independent native unfold")

def packing_controls(activation, expected, start, count, np):
    controls = []
    wrong_padding = patch_rows(activation, start, count, np, padding=0)
    correct = patch_rows(activation, start, count, np)
    channels = activation.shape[1]
    wrong_order = correct.reshape(count, channels, 3, 3).transpose(0, 2, 3, 1).reshape(count, channels * 9)
    for name, changed in (("missing_zero_padding", wrong_padding), ("kernel_major_instead_of_channel_major", wrong_order)):
        try:
            require_same_patches(changed.copy(), expected, np)
        except ValueError:
            controls.append({"control": name, "refused": True})
        else:
            raise AssertionError("Padding/layout negative control did not disagree: " + name)
    return controls

def require_tensor(actual, expected, message):
    if tensor_identity(actual) != tensor_identity(expected):
        raise ValueError(message)

def restore_native_source(source, receipt, records, report, *, sample_index):
    """Restore the exact source environment, checkpoint and selected image once."""
    helper = load_native_admission(receipt, records, report)
    import numpy as np
    import torch
    import torchvision
    import torchvision.models.resnet as implementation
    versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                "torch": torch.__version__, "torchvision": torchvision.__version__}
    if (versions != source["versions"] or
            identity(implementation.__file__)["sha256"] != receipt["torchvision_resnet_source"]["sha256"]):
        raise ValueError("Native source environment differs from the admitted capture")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model = torchvision.models.resnet50(weights=None).eval()
    helper.load_checkpoint(model, source["checkpoint"]["path"], torch)
    loader = load_module("resnet_source_loader", receipt["loader_source"]["path"])
    state_before = loader._state_dict_sha256(model)
    if state_before != source["state_dict_sha256"]:
        raise ValueError("Native checkpoint parameters or buffers differ")
    images, _ = helper.load_images(source["input_stream"]["path"], np)
    image = torch.from_numpy(images[sample_index].copy())
    if tensor_identity(image) != source["input"]:
        raise ValueError("Native source image differs")
    return torch, model, loader, state_before, image, versions


def capture_native_site(model, image, module, original):
    """Observe one unchanged native invocation, then replay and remove the hook."""
    captured = []

    def hook(_, inputs, output):
        captured.append((inputs[0].detach().clone(), output.detach().clone()))

    handle = module.register_forward_hook(hook)
    try:
        instrumented = model(image).detach().clone()
    finally:
        handle.remove()
    if len(captured) != 1 or tensor_identity(instrumented) != tensor_identity(original):
        raise AssertionError("Capture hook changed native invocation count or terminal")
    activation, convolution = captured[0]
    if tensor_identity(module(activation)) != tensor_identity(convolution):
        raise AssertionError("Native same-input convolution replay differs from captured output")
    if tensor_identity(model(image)) != tensor_identity(original):
        raise AssertionError("Native repeat after hook removal changed the terminal")
    return activation, convolution


def _admit_original(args, report):
    if args.receipt_sha256 != RECEIPT_SHA256 or args.preparation_sha256 != PREPARATION_SHA256:
        raise ValueError("Require the original frozen receipt and preparation digests")
    relocation = source_relocator(args)
    receipt_path = safe_path(args.receipt)
    receipt, receipt_id = relocation.read_json(receipt_path, args.receipt_sha256)
    if receipt["status"] != "passed" or receipt["evaluation_role"] != "development":
        raise ValueError("Expected the frozen passing development receipt")
    preparation, preparation_id = relocation.read_json(args.preparation, args.preparation_sha256)
    report["source_relocation"] = relocation.report()
    records = [receipt_id, preparation_id, identity(__file__), identity(source_relocation.__file__)]
    records += [require_identity(receipt["artifacts"][name]) for name in ("checkpoint", "inputs", "labels")]
    records += [require_identity(receipt[name]) for name in ("loader_source", "torchvision_resnet_source")]
    helper = load_native_admission(receipt, records, report)
    import numpy as np
    import torch
    import torchvision
    import torchvision.models.resnet as resnet_source

    expected_versions = receipt["versions"]
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "torchvision": torchvision.__version__, "numpy": np.__version__}
    if any(versions[name] != expected_versions[name] for name in versions):
        raise ValueError("Native environment differs from the frozen source versions")
    if identity(resnet_source.__file__)["sha256"] != receipt["torchvision_resnet_source"]["sha256"]:
        raise ValueError("Imported torchvision implementation differs from the frozen source")
    prepared_bindings = {str(safe_path(record["path"])): record for record in preparation["source_bindings"]}
    for path in (torch.__file__, np.__file__, Path(torch.__file__).parent / "lib/libtorch_cpu.so"):
        record = prepared_bindings.get(str(safe_path(path)))
        if record is None:
            raise ValueError("Imported native dependency is outside the frozen preparation bindings: " + str(path))
        records.append(require_identity(record))
    if not 0 <= args.sample_index < len(receipt["images"]):
        raise ValueError("Sample index is outside the frozen development stream")
    sample = preparation["samples"]["evaluation"][args.sample_index]
    if sample["id"] != receipt["label_declarations"]["samples"][args.sample_index]["id"]:
        raise ValueError("Development sample identity differs between receipts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(194)
    model = torchvision.models.resnet50(weights=None).eval()
    helper.load_checkpoint(model, receipt["artifacts"]["checkpoint"]["path"], torch)
    loader = load_module("refresh_resnet_source_loader", receipt["loader_source"]["path"])
    state_before = loader._state_dict_sha256(model)
    if state_before != receipt["model"]["state_dict_sha256"]:
        raise ValueError("Loaded native model state differs from the frozen model")
    module = model.get_submodule(args.site)
    if (type(module) is not torch.nn.Conv2d or module.kernel_size != (1, 1)
            or module.stride != (1, 1) or module.padding != (0, 0)
            or module.dilation != (1, 1) or module.groups != 1 or module.bias is not None):
        raise ValueError("Capture requires a native bias-free stride-one 1x1 convolution")
    images, _ = helper.load_images(receipt["artifacts"]["inputs"]["path"], np)
    image = images[args.sample_index].copy()
    tagged = sha256_bytes(str(image.dtype).encode() + str(image.shape).encode() + image.tobytes())
    if tagged != receipt["images"][args.sample_index]["input_sha256"]:
        raise ValueError("Selected native input tensor differs from its frozen digest")
    native_path = receipt_path.parent / ("image_" + str(args.sample_index) + ".npz")
    saved_native_id = identity(native_path)
    records.append(saved_native_id)
    with np.load(native_path, allow_pickle=False) as archive:
        historical_logits = archive["torch"].copy()
        if not np.array_equal(archive["image"], image):
            raise ValueError("Historical native image differs from the selected stream input")
    tensor = torch.from_numpy(image.copy())
    input_before = tensor_identity(tensor)
    logits = model(tensor).detach().clone()
    if not np.array_equal(logits.numpy(), historical_logits):
        raise AssertionError("Native source differs from the currently saved native logits")
    activation, convolution = capture_native_site(model, tensor, module, logits)
    if input_before != tensor_identity(tensor) or loader._state_dict_sha256(model) != state_before:
        raise AssertionError("Native capture changed input, parameter or buffer bytes")
    if activation.dtype != torch.float32 or convolution.dtype != torch.float32:
        raise ValueError("Native contraction must retain FP32 source dtype")
    return {
        'activation': activation,
        'convolution': convolution,
        'weight': module.weight,
        'records': records,
        'source': {
            'workload': 'torchvision ResNet50 v1.5 IMAGENET1K_V2; modern pinned comparison, not original-paper reproduction',
            'receipt': receipt_id, 'preparation': preparation_id,
            'checkpoint': receipt['artifacts']['checkpoint'],
            'input_stream': receipt['artifacts']['inputs'], 'sample_index': args.sample_index, 'sample': sample,
            'split': 'development',
            'selection': 'frozen class-table/lexical development order; index selected before candidate execution',
            'transform': preparation['transform'], 'state_dict_sha256': state_before,
            'native_logits': tensor_identity(logits),
            'input': input_before, 'saved_native_archive': saved_native_id,
            'saved_native_archive_provenance': (
                'Archive hash is bound by this capture, not authenticated by the original numerical receipt'
            ),
            'immutable_bindings': records, 'original_preparation_bindings': preparation['source_bindings'],
            'versions': versions,
            'merlin_revision': git_head(repo_root()),
            'baseline_merlin_revision': git_head(safe_path(args.baseline_root)) if args.baseline_root else None,
            'source_relocation': relocation.report(),
        },
        'site': {
            'module': args.site, 'boundary': 'native convolution before BatchNorm; no bias, cast or folding',
            'operation': 'C=A@B',
            'mapping': 'A[p,c]=X[0,c,h,w], B[c,o]=W[o,c,0,0], native_C[p,o]=Y[0,o,h,w]; h,w=divmod(spatial_start+p,W)',
        },
        'checks': {
            'saved_native_logits_raw_exact': True, 'instrumented_terminal_raw_exact': True,
            'native_site_replay_raw_exact': True,
            'repeat_after_hook_removal_raw_exact': True, 'input_state_preserved': True,
            'bindings_unchanged': True,
            'matrix_decomposition_rtol': 0.0001, 'matrix_decomposition_atol': 0.0001, 'cpu_threads': 1,
            'cpu_interop_threads': 1,
        },
        'report_fields': {},
        'bookends': records + preparation['source_bindings'],
    }


def _admit_original_3x3(args, report):
    import numpy as np
    source_dir = safe_path(args.source_bundle)
    source_id = require_identity({"path": str(source_dir / "manifest.json"), "sha256": args.source_manifest_sha256})
    previous = read_source_metadata(args, report, source_id["path"], source_id["sha256"])
    if previous.get("status") != "passed" or previous.get("source_dtype") != "float32":
        raise ValueError("Require the original passing FP32 native source capture")
    source = previous["source"]
    if source["split"] != "development" or source["sample_index"] != args.sample_index:
        raise ValueError("Require the same frozen development image")
    receipt_id = require_identity(source["receipt"])
    if receipt_id["sha256"] != RECEIPT_SHA256:
        raise ValueError("Wrong frozen native source receipt")
    receipt = read_source_metadata(args, report, receipt_id["path"], receipt_id["sha256"])
    records = [source_id, receipt_id, relocation_source_identity(), identity(__file__), identity(Path(__file__).with_name(
        "capture_resnet.py"))]
    records += [require_identity(record) for record in source["immutable_bindings"]]
    records += [require_identity(record) for record in source["original_preparation_bindings"]]
    torch, model, loader, state_before, image, versions = restore_native_source(
        source, receipt, records, report, sample_index=args.sample_index)
    module = model.get_submodule(args.site)
    if (type(module) is not torch.nn.Conv2d or module.kernel_size != (3, 3) or module.stride != (1, 1)
            or module.padding != (1, 1) or module.padding_mode != "zeros" or module.dilation != (1,
        1) or module.groups != 1 or module.bias is not None):
        raise ValueError("Capture requires native bias-free stride-one zero-padded 3x3 convolution")
    original = model(image).detach().clone()
    if tensor_identity(original) != source["native_logits"]:
        raise AssertionError("Unchanged native terminal differs from the frozen capture")
    activation, convolution = capture_native_site(model, image, module, original)
    if activation.dtype != torch.float32 or convolution.dtype != torch.float32:
        raise ValueError("Native convolution must retain original FP32 arithmetic")
    if tensor_identity(image) != source["input"] or loader._state_dict_sha256(model) != state_before:
        raise AssertionError("Native repeat/input/state preservation failed")
    return {
        'activation': activation,
        'convolution': convolution,
        'weight': module.weight,
        'records': records,
        'source': {
            **source, 'source_capture': source_id, 'immutable_bindings': records,
            'merlin_revision': git_head(repo_root()),
            'candidate_chain': None,
            'activation_role': 'Unchanged native model prefix; no earlier target substitutions',
        },
        'site': {
            'module': args.site, 'boundary': 'Native 3x3 convolution before BatchNorm; no bias, cast or folding',
            'operation': 'native_C=Conv2d(X,W); A=explicit_host_im2col(X), B=flatten_native_weights(W)',
            'mapping': (
                'p=spatial_start+row; h,w=divmod(p,W); q=(c*3+kh)*3+kw; A[row,q]=X[0,c,h+kh-1,w+kw-1] or '
                'zero outside image; B[q,o]=weight[output_start+o,c,kh,kw]; '
                'native_C[row,o]=actual_Y[0,output_start+o,h,w]'
            ),
        },
        'checks': {
            'native_terminal_raw_exact': True, 'instrumented_terminal_raw_exact': True,
            'native_site_replay_raw_exact': True,
            'repeat_after_hook_removal_raw_exact': True, 'input_state_preserved': True,
            'source_bookends_unchanged': True,
            'patches_vs_native_unfold_raw_exact': True, 'cpu_threads': 1, 'cpu_interop_threads': 1,
        },
        'report_fields': {
            'schema_version': 1,
        },
    }


def _admit_after_conv1(args, report):
    from evaluate_resnet import load_results
    import numpy as np
    # Reuse full physical-output admission, including producer/source hashes,
    # complete tile coverage, payload linkage and its actual negative controls.
    admission = {}
    native, operands, target, records = load_results(args, admission)
    source = native["source"]
    if args.site != "layer1.0.conv1" or native["site"]["module"] != args.site:
        raise ValueError("This bounded candidate chain starts at layer1.0.conv1")
    records += [identity(__file__), identity(Path(__file__).with_name("capture_resnet.py")),
                identity(Path(__file__).with_name("evaluate_resnet.py"))]
    receipt_id = require_identity(source["receipt"])
    if receipt_id["sha256"] != RECEIPT_SHA256:
        raise ValueError("Wrong frozen original development receipt")
    receipt = read_source_metadata(args, report, receipt_id["path"], receipt_id["sha256"])
    records += [require_identity(record) for record in source["immutable_bindings"]]
    records += [require_identity(record) for record in source["original_preparation_bindings"]]
    torch, model, loader, state_before, image, versions = restore_native_source(
        source, receipt, records, report, sample_index=args.sample_index)
    block = model.layer1[0]
    conv1, bn1, conv2 = block.conv1, block.bn1, block.conv2
    if (type(conv1) is not torch.nn.Conv2d or type(bn1) is not torch.nn.BatchNorm2d
            or type(block.relu) is not torch.nn.ReLU or type(conv2) is not torch.nn.Conv2d
            or conv1.kernel_size != (1, 1) or conv1.stride != (1, 1) or conv1.bias is not None
            or conv2.kernel_size != (3, 3) or conv2.stride != (1, 1) or conv2.padding != (1, 1)
            or conv2.padding_mode != "zeros" or conv2.dilation != (1, 1) or conv2.groups != 1 or conv2.bias is not None):
        raise ValueError("Pinned native bottleneck boundary has unexpected operations")
    original_capture = []

    def original_hook(_, inputs, output):
        original_capture.append((inputs[0].detach().clone(), output.detach().clone()))

    handle = conv2.register_forward_hook(original_hook)
    try:
        original = model(image).detach().clone()
    finally:
        handle.remove()
    if len(original_capture) != 1 or tensor_identity(original) != source["native_logits"]:
        raise AssertionError("Unchanged original reference differs from the frozen source")
    height, width = native["site"]["input_shape"][2:]
    replacement = torch.from_numpy(target.copy()).t().reshape(1, target.shape[1], height, width).contiguous()
    if not np.array_equal(replacement[0].flatten(1).t().contiguous().numpy().view(np.uint32), target.view(np.uint32)):
        raise AssertionError("Actual target words changed during matrix-to-NCHW reshape")
    traces, captures, batchnorms, terminals = [], [], [], []
    current_trace, current_bn = [], []

    def inject(_, inputs, output):
        a = inputs[0][0].flatten(1).t().contiguous().numpy()
        c = output[0].flatten(1).t().contiguous().numpy()
        if (list(inputs[0].shape) != native["site"]["input_shape"] or output.shape != replacement.shape
                or not np.array_equal(a.view(np.uint32), operands["A"].view(np.uint32))
                or not np.array_equal(c.view(np.uint32), operands["native_C"].view(np.uint32))):
            raise ValueError("Conv1 callback no longer receives the admitted original invocation")
        current_trace.append("actual_conv1_output")
        return replacement.clone()

    def observe_bn(_, inputs, output):
        if current_trace != ["actual_conv1_output"]:
            raise ValueError("Native BatchNorm callback occurred outside the candidate chain")
        require_tensor(inputs[0], replacement, "Native BatchNorm input is not the actual target conv1 result")
        current_bn.append(output.detach().clone())
        current_trace.append("native_bn1")

    def observe_conv2(_, inputs, output):
        if current_trace != ["actual_conv1_output", "native_bn1"] or len(current_bn) != 1:
            raise ValueError("Conv2 callback occurred outside the admitted prefix")
        require_tensor(inputs[0], torch.relu(current_bn[0]), "Conv2 input is not native ReLU(native BN(actual conv1))")
        captures.append((inputs[0].detach().clone(), output.detach().clone()))
        batchnorms.append(current_bn[0])
        current_trace.append("native_relu_to_conv2")

    handles = [conv1.register_forward_hook(inject), bn1.register_forward_hook(observe_bn), conv2.register_forward_hook(
        observe_conv2)]
    try:
        for repeat in range(2):
            current_trace.clear()
            current_bn.clear()
            terminals.append(model(image).detach().clone())
            if current_trace != ["actual_conv1_output", "native_bn1", "native_relu_to_conv2"]:
                raise AssertionError("Candidate callbacks must occur exactly once and in order")
            traces.append(list(current_trace))
    finally:
        for handle in handles:
            handle.remove()
    if (len(captures) != 2 or tensor_identity(terminals[0]) != tensor_identity(terminals[1])
            or not all(torch.isfinite(value).all() for value in terminals)):
        raise AssertionError("Candidate capture count or repeated terminal changed")
    activation, convolution = captures[0]
    for left, right in zip(captures[0], captures[1]):
        require_tensor(left, right, "Candidate activation/native output changed across repeats")
    if tensor_identity(conv2(activation)) != tensor_identity(convolution):
        raise AssertionError("Native conv2 replay on candidate activation differs")
    restored = model(image).detach().clone()
    if (tensor_identity(restored) != tensor_identity(original) or tensor_identity(image) != source["input"]
            or loader._state_dict_sha256(model) != state_before):
        raise AssertionError("Original terminal/input/state not restored after hook removal")
    # Invoke the actual callbacks with wrong data, without running extra graphs.
    wrong_replacement = replacement.clone()
    wrong_replacement.view(torch.int32).flatten()[0] ^= 1
    current_trace[:] = ["actual_conv1_output"]
    current_bn.clear()
    try:
        observe_bn(bn1, (wrong_replacement,), batchnorms[0])
    except ValueError:
        wrong_bn_control = {"control": "changed_actual_conv1_words_at_bn_callback", "refused": True}
    else:
        raise AssertionError("BatchNorm callback admitted changed target output words")
    current_trace[:] = ["actual_conv1_output", "native_bn1"]
    current_bn[:] = [batchnorms[0]]
    try:
        observe_conv2(conv2, (original_capture[0][0],), original_capture[0][1])
    except ValueError:
        original_activation_control = {"control": "original_later_site_activation", "refused": True}
    else:
        raise AssertionError("Selected candidate did not distinguish the original later-site activation")
    return {
        'activation': activation,
        'convolution': convolution,
        'weight': conv2.weight,
        'records': records,
        'source': {
            **source, 'source_capture': admission['source_bundle'], 'immutable_bindings': records,
            'merlin_revision': git_head(repo_root()),
            'activation_role': 'Native BN/ReLU applied to actual prior target-model conv1 physical output',
            'candidate_chain': {
                'sites': ['layer1.0.conv1', 'layer1.0.bn1', 'native ReLU', 'layer1.0.conv2'],
                'provider_receipt': admission['provider_receipt'],
                'provider_result': admission['provider_result'], 'provider_recipe': admission['provider_recipe'],
                'provider_output_formatter': admission['provider_output_formatter'],
                'actual_conv1_output': tensor_identity(replacement),
                'native_bn1_output': tensor_identity(batchnorms[0]),
                'conv2_activation': tensor_identity(activation),
                'conv2_native_output': tensor_identity(convolution),
                'original_conv2_activation': tensor_identity(original_capture[0][0]),
                'original_conv2_native_output': tensor_identity(original_capture[0][1]),
                'original_native_logits': tensor_identity(original),
                'candidate_prefix_native_logits': tensor_identity(terminals[0]),
            },
        },
        'site': {
            'module': 'layer1.0.conv2',
            'boundary': 'Native conv2 before BatchNorm, on the explicitly admitted candidate prefix',
            'operation': 'native_C=Conv2d(candidate_X,W); explicit_host_im2col(candidate_X)@flatten_native_weights(W)',
            'mapping': (
                'q=(c*3+kh)*3+kw; h,w=divmod(spatial_start+p,W); A[p,q]=candidate_X[0,c,h+kh-1,w+kw-1] or '
                'zero outside image; B[q,o]=weight[output_start+o,c,kh,kw]; '
                'native_C[p,o]=actual_native_conv2(candidate_X)[0,output_start+o,h,w]'
            ),
        },
        'checks': {
            'original_native_reference_raw_exact': True, 'actual_conv1_output_physically_admitted': True,
            'native_bn_relu_callback_chain_exact': True,
            'callback_traces': traces, 'candidate_repeats_raw_exact': True,
            'candidate_native_conv2_replay_raw_exact': True,
            'original_terminal_restored_raw_exact': True, 'source_input_state_bookends': True,
            'patches_vs_unfold_raw_exact': True,
            'provider_admission_controls': admission['negative_controls'],
        },
        'report_fields': {
            'complete_atlas_execution': False,
        },
        'extra_controls': [wrong_bn_control, original_activation_control],
        'archives': {
            'terminals': ('terminals.npz', {
                'original_native_logits': original.numpy(),
                'candidate_prefix_native_logits': terminals[0].numpy()}),
            'original_reference': ('original-reference.npz', {
                'native_logits': original.numpy(),
                'native_C': original_capture[0][1][0].flatten(1).t().contiguous().numpy()}),
        },
    }


def _admit_after_conv2(args, report):
    from evaluate_resnet import evaluate_chain as evaluate
    import numpy as np
    import torch
    sink, hybrid = {}, {"status": "failed", "claim": "Two actual model-result convolutions inside unchanged native ResNet",
                       "complete_atlas_execution": False, "timing_qualified": False, "application_quality_qualified": False,
                       "command": sys.argv, "source_script": identity(Path(__file__).with_name("evaluate_resnet.py"))}
    evaluate(args, hybrid, capture_sink=sink)
    activation, convolution = sink["candidate"]
    original_activation, original_convolution = sink["original"]
    weight = sink["weight"]
    if (list(activation.shape) != [1, 64, 56, 56] or list(convolution.shape) != [1, 256, 56, 56]
            or list(weight.shape) != [256, 64, 1, 1] or not sink["replay_raw_exact"]
            or any(value.dtype != torch.float32 for value in (activation, convolution, weight))):
        raise ValueError("Unexpected candidate conv3 geometry/dtype/replay")
    source_records = hybrid['source_bookends'] + hybrid['conv2_admission_bookends']
    source_records += [identity(__file__), identity(Path(__file__).with_name('evaluate_resnet.py'))]
    second_receipt = require_identity({
        'path': str(args.conv2_provider_receipt), 'sha256': args.conv2_provider_receipt_sha256})
    second = read_source_metadata(args, report, second_receipt['path'], second_receipt['sha256'])
    second_result = require_identity({
        'path': str(Path(second_receipt['path']).parent / 'result.npz'),
        'sha256': second['artifacts']['result.npz']})
    with np.load(args.output / 'outputs.npz', allow_pickle=False) as archive:
        original_logits = archive['original_native_logits'].copy()
    return {
        'activation': activation,
        'convolution': convolution,
        'weight': weight,
        'records': source_records,
        'source': {
            **hybrid['source'], 'immutable_bindings': source_records, 'merlin_revision': git_head(repo_root()),
            'activation_role': 'Native BN2/ReLU after actual admitted conv1 and coupled conv2 model outputs',
            'candidate_chain': {
                'conv1_admission': hybrid['first_operator_admission'],
                'conv2_native_manifest': {
                    'path': str(safe_path(args.conv2_native_bundle) / 'manifest.json'),
                    'sha256': args.conv2_native_manifest_sha256,
                },
                'conv2_provider_receipt': second_receipt, 'conv2_provider_result': second_result,
                'conv2_admission': hybrid['second_operator_admission'],
                'actual_outputs': sink['actual_outputs'],
                'native_bn2_output': hybrid['native_suffix_observations'][0]['bn2'],
                'conv3_activation': tensor_identity(activation),
                'conv3_native_output': tensor_identity(convolution),
                'original_conv3_activation': tensor_identity(original_activation),
                'original_conv3_native_output': tensor_identity(original_convolution),
                'original_native_logits': hybrid['original_native_logits'],
                'two_operator_native_logits': hybrid['hybrid_logits'][0],
            },
        },
        'site': {
            'module': 'layer1.0.conv3',
            'boundary': 'Native convolution before BN3/residual, on admitted two-operator candidate prefix',
            'operation': 'C=A@B',
            'mapping': (
                'h,w=divmod(spatial_start+p,56); A[p,k]=candidate_X[0,k,h,w]; '
                'B[k,o]=W[output_start+o,k,0,0]; '
                'native_C[p,o]=actual_native_conv3(candidate_X)[0,output_start+o,h,w]'
            ),
        },
        'checks': {
            **hybrid['checks'], 'candidate_native_conv3_replay_raw_exact': True,
            'two_operator_callback_negatives': hybrid['negative_controls'],
        },
        'report_fields': {},
        'original_activation': original_activation,
        'archives': {'original_reference': ('original-reference.npz', {
            'native_logits': original_logits,
            'A': original_activation[0].flatten(1).t().contiguous().numpy(),
            'native_C': original_convolution[0].flatten(1).t().contiguous().numpy()})},
        'hybrid_receipt': hybrid,
    }


def capture(args, report, mode="original"):
    """Capture any admitted convolution through the same packing/report path.

    Admission adapters retain the distinct frozen-source and physical-output
    callback checks. Only the original 1x1 capture gates on GEMM agreement;
    later captures retain the unchanged native output and report diagnostics.
    """
    adapters = {"original": _admit_original, "original-3x3": _admit_original_3x3,
                "after-conv1": _admit_after_conv1, "after-conv2": _admit_after_conv2}
    state = adapters[mode](args, report)
    import numpy as np
    import torch

    activation, convolution, weight = (state[key] for key in ("activation", "convolution", "weight"))
    _, channels, height, width = activation.shape
    stop_row, stop_channel = args.spatial_start + args.rows, args.output_channel_start + args.columns
    patch_capture = mode in ("original-3x3", "after-conv1")
    if patch_capture:
        if (args.output_channel_start < 0 or args.columns <= 0 or stop_channel > weight.shape[0]
                or (mode == "original-3x3" and tuple(convolution.shape) != (1, weight.shape[0], height, width))):
            raise ValueError("Requested output columns or native output shape are invalid")
    elif (activation.shape[0] != 1 or args.spatial_start < 0 or stop_row > height * width
          or args.output_channel_start < 0 or stop_channel > weight.shape[0]):
        raise ValueError("Requested tile lies outside the native contraction")
    rows, outputs = slice(args.spatial_start, stop_row), slice(args.output_channel_start, stop_channel)
    controls = None
    if patch_capture:
        a = patch_rows(activation.numpy(), args.spatial_start, args.rows, np)
        unfold = torch.nn.functional.unfold(activation, kernel_size=3, padding=1)[0].t().contiguous().numpy()
        require_same_patches(a, unfold[rows], np)
        b = weight[outputs].flatten(1).t().contiguous().numpy()
        c = convolution[0].flatten(1).t()[rows, outputs].contiguous().numpy()
        matrix = torch.mm(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    else:
        a = activation[0].flatten(1).t()[rows].contiguous()
        b = weight[outputs, :, 0, 0].t().contiguous()
        c = convolution[0].flatten(1).t()[rows, outputs].contiguous()
        matrix = torch.mm(a, b)
        if mode == "original" and not torch.allclose(matrix, c, rtol=1e-4, atol=1e-4):
            raise AssertionError("Matrix decomposition fails the unchanged native floating gate")
    values = (a, b, c) if mode == "original" else (a, b, c, matrix)
    finite = np.isfinite if patch_capture else torch.isfinite
    if any(not finite(value).all() for value in values):
        raise ValueError("Nonfinite native contraction operands or outputs")
    if patch_capture:
        # Exercise the actual top/left edge, independently of selected rows.
        edge_count = min(32, height * width) if mode == "original-3x3" else 32
        controls = packing_controls(activation.numpy(), unfold[:edge_count], 0, edge_count, np)
        controls += state.get("extra_controls", [])
    elif mode == "after-conv2":
        wrong = a.numpy().astype(np.float64) @ b.numpy()[::-1].astype(np.float64)
        controls = {
            "wrong_k_order_detected": not bool(np.allclose(wrong, c.numpy(), rtol=1e-4, atol=1e-4)),
            "original_later_site_input_distinguished": tensor_identity(activation) != tensor_identity(state[
            "original_activation"]),
            "wrong_output_order_detected": not bool(np.array_equal(
                c.numpy().view(np.uint32), np.roll(c.numpy(), 1, axis=1).view(np.uint32))),
        }
        if not all(controls.values()):
            raise AssertionError("Candidate conv3 layout/source control failed: " + str(controls))
    for record in state.get("bookends", state["records"]):
        require_identity(record)
    arrays = (a, b, c) if patch_capture else tuple(value.numpy() for value in (a, b, c))
    np.savez(args.output / "operands.npz", **dict(zip(("A", "B", "native_C"), arrays)))
    extra = dict(state["report_fields"])
    for key, (filename, tensors) in state.get("archives", {}).items():
        np.savez(args.output / filename, **tensors)
        extra[key] = identity(args.output / filename)
    if "hybrid_receipt" in state:
        write_manifest(args.output / "hybrid-receipt.json", state["hybrid_receipt"])
        extra["hybrid_receipt"] = identity(args.output / "hybrid-receipt.json")
    site = {
        **state["site"], "activation_layout": "NCHW", "weight_layout": "OIHW",
        "input_shape": list(activation.shape), "weight_shape": list(weight.shape),
        "spatial_flat_range": [args.spatial_start, stop_row],
        "output_channel_range": [args.output_channel_start, stop_channel],
        "k_range": [0, channels * (9 if patch_capture else 1)], "complete_k": True,
        "matrix_shapes": {name: list(value.shape) for name, value in zip(("A", "B", "native_C"), arrays)},
    }
    if mode != "original":
        site["output_shape"] = list(convolution.shape)
    if patch_capture:
        site.update(stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1)
    checks = dict(state["checks"])
    if patch_capture:
        checks["negative_controls"] = controls
    elif mode == "after-conv2":
        extra["negative_controls"] = controls
    else:
        checks["matrix_decomposition_max_abs"] = float((matrix - c).abs().max())
    if mode != "original":
        roles = {
            "original-3x3": (
                'Diagnostic only: separate GEMM accumulation may differ; actual native convolution output '
                'is retained unchanged'
            ),
            "after-conv1": "Diagnostic only; native conv2 on the candidate activation is retained unchanged",
            "after-conv2": "Diagnostic only; actual native conv3 on candidate input retained unchanged",
        }
        close = np.isclose(matrix, c, rtol=1e-4, atol=1e-4) if patch_capture else torch.isclose(matrix, c, rtol=1e-4, atol=1e-4)
        delta = matrix.astype(np.float64) - c.astype(np.float64) if patch_capture else matrix - c
        diagnostic = {"role": roles[mode], "rtol": 1e-4, "atol": 1e-4,
                      "allclose": bool(close.all()), "failed_values": int((~close).sum()),
                      "max_abs": float(np.abs(delta).max()) if patch_capture else float(delta.abs().max())}
        if mode == "original-3x3":
            diagnostic.update(raw_exact=bool(np.array_equal(matrix.view(np.uint32), c.view(np.uint32))),
                              rms=float(np.sqrt(np.mean(delta * delta))))
        extra["fp32_decomposition"] = diagnostic
    report.update(
        **extra, status="passed", source_dtype="float32", npz_sha256=identity(args.output / "operands.npz")["sha256"],
        source=state["source"], site=site, checks=checks,
        tensors={name: {"shape": list(value.shape), "dtype": str(value.dtype),
                        "raw_sha256": sha256_bytes(value.tobytes())}
                 for name, value in zip(("A", "B", "native_C"), arrays)},
        target_arithmetic_qualified=False, target_or_timing_qualified=False, application_quality_qualified=False)


def capture_original(args, report):
    return capture(args, report, "original")


def capture_original_3x3(args, report):
    return capture(args, report, "original-3x3")


def capture_after_conv1(args, report):
    return capture(args, report, "after-conv1")


def capture_after_conv2(args, report):
    return capture(args, report, "after-conv2")


def main():
    root = argparse.ArgumentParser(description=__doc__)
    modes = root.add_subparsers(dest="mode", required=True)
    specifications = {
        "original": (("receipt", "preparation"), ("receipt-sha256", "preparation-sha256"), "layer1.0.conv1"),
        "original-3x3": (("source-bundle",), ("source-manifest-sha256",), "layer1.0.conv2"),
        "after-conv1": (("native-bundle", "provider-receipt", "provider-arithmetic"),
                        ("native-manifest-sha256", "provider-receipt-sha256"), "layer1.0.conv1"),
        "after-conv2": (("conv1-hybrid-receipt", "conv2-native-bundle", "conv2-provider-receipt", "producer-root", "model-root"),
                        ("conv1-hybrid-receipt-sha256", "conv2-native-manifest-sha256", "conv2-provider-receipt-sha256",
        "validator-sha256"), None),
    }
    for mode, (paths, digests, site) in specifications.items():
        parser = modes.add_parser(mode)
        for name in (*paths, "output-dir"):
            parser.add_argument("--" + name, type=Path, required=True)
        for name in digests:
            parser.add_argument("--" + name, required=True)
        for name, default in (("rows", 32), ("columns", 32), ("spatial-start", 0), ("output-channel-start", 0)):
            parser.add_argument("--" + name, type=int, default=default)
        parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
        if site:
            options = {"choices": [site]} if mode == "after-conv1" else {}
            parser.add_argument("--site", default=site, **options)
            parser.add_argument("--sample-index", type=int, default=0)
        if mode == "original":
            parser.add_argument("--baseline-root", type=Path, help="Optional original Git checkout for provenance only")
    args = root.parse_args()
    if args.rows <= 0 or args.columns <= 0:
        root.error("Rows and columns must be positive")
    from workload_cli import run_report
    return run_report(args, lambda args, report: capture(args, report, args.mode),
                      filename="manifest.json", claim=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
