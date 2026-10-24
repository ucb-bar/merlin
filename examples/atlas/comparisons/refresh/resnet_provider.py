"""Explicit target-provider adapter; no task-quality workflow."""

from functools import lru_cache
from resnet_native import safe_path, file_identity
from atlas_provider_files import PACKING_FILE, PACKING_FILES, packing_paths


class Provider:
    """Call only the independently reviewed target provider's arithmetic."""

    def __init__(self, vector, scalar, np):
        self.vector, self.scalar, self.np = vector, scalar, np
        self.packed = lru_cache(maxsize=1048576)(scalar.pack_bf16)
        self.weights = {}

    def encode(self, value):
        np = self.np
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Require finite FP32 source operands")
        raw = value.view(np.uint32)
        high = (raw >> 16).astype(np.uint32)
        low = raw & 65535
        high += ((low > 32768) | ((low == 32768) & ((high & 1) != 0))).astype(np.uint32)
        bits = high.astype(np.uint16)
        if np.any((bits & 0x7F80) == 0x7F80):
            raise ValueError("Source overflows finite BF16")
        maximum = int(np.max(bits & 0x7FFF))
        exponent = self.scalar.tensor_scale_exponent([maximum])
        unique, inverse = np.unique(bits, return_inverse=True)
        lut = np.array([self.packed(int(x), 127 + exponent) for x in unique], dtype=np.uint8)
        encoded = lut[inverse].reshape(value.shape)
        return encoded, exponent

    def contraction(self, a, b, name):
        np = self.np
        ea, sa = self.encode(a)
        if name not in self.weights:
            self.weights[name] = self.encode(b)
        eb, sb = self.weights[name]
        result = self.vector.ordered_matmul(ea, eb)
        exponent = sa + sb
        if not -126 <= exponent <= 127:
            raise ValueError("Combined rescale outside original finite range")
        # Only output rescaling is vectorized; the provider must expose the same
        # signed-zero/overflow-refusal contract as scalar rescale_bf16.
        if hasattr(self.vector, "rescale_bf16_array"):
            corrected = self.vector.rescale_bf16_array(result, exponent)
        else:
            unique, inverse = np.unique(result, return_inverse=True)
            lut = np.array([self.scalar.rescale_bf16(int(x), exponent) for x in unique], dtype=np.uint16)
            corrected = lut[inverse].reshape(result.shape)
        return (corrected.astype(np.uint32) << 16).view(np.float32), {"a_scale": sa, "b_scale": sb, "rescale": exponent}


def read_source(args, torch, torchvision, np):
    """Admit frozen metadata before constructing a relocated path view."""
    import sys

    if sys.flags.optimize:
        raise ValueError("Source admission requires Python assertions enabled")
    from merlin.common.paths import repo_root

    sys.path.insert(0, str(repo_root() / "examples/atlas/comparisons"))
    from source_relocation import SourceRelocation
    from capture_tinyllama_native import verify_declared_sources


    relocation = SourceRelocation(args.relocate)
    document, original = relocation.read_json(args.source_manifest, args.source_manifest_sha256)
    source = document["source"]
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    if versions != source["versions"]:
        raise ValueError("Native environment differs from the frozen source")
    # These preserve original source/helper/native-library admission even though
    # path spelling changes. No digest or version field is rewritten.
    verify_declared_sources(source.get("immutable_bindings", []))
    verify_declared_sources(source.get("original_preparation_bindings", []))
    return source, original, relocation


def load_provider(name, path, expected):
    """Import only a caller-pinned target implementation, retaining its identity."""
    import importlib.util
    import sys

    path = safe_path(path)
    before = file_identity(path, safe_path)
    if before["sha256"] != expected:
        raise ValueError("Selected provider digest differs")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if file_identity(module.__file__, safe_path) != before or file_identity(path, safe_path) != before:
        raise ValueError("Provider source changed during import")
    return module, before


def load_packing(args):
    """Bind the actual imported target dependency tuple before using its API."""
    import sys

    names = tuple(name[:-3] for name in PACKING_FILES)
    dependencies = [file_identity(path, safe_path) for path in packing_paths(args.atlas_tools)]
    before = {name: file_identity(args.atlas_tools / (name + ".py"), safe_path) for name in names}
    for name in names[1:]:
        if name in sys.modules and safe_path(sys.modules[name].__file__) != safe_path(before[name]["path"]):
            raise ValueError("Target helper cached from a different provider: " + name)
    module, own = load_provider(
        "full_export_packing", args.atlas_tools / PACKING_FILE, args.packing_provider_sha256
    )
    for name in names[1:]:
        if name not in sys.modules or safe_path(sys.modules[name].__file__) != safe_path(before[name]["path"]):
            raise ValueError("Actual imported target helper differs: " + name)
    for item in dependencies:
        if file_identity(item["path"], safe_path) != item:
            raise ValueError("Target helper changed during import")
    return module, own, dependencies


def validate_plan(plan, source, sourceid, providers):
    """Refuse a same-shape plan bound to other native parameters or providers."""
    if (
        plan.get("status") != "exported_compile_plan_only"
        or plan["source_manifest"]["sha256"] != sourceid["sha256"]
        or plan["checkpoint"] != source["checkpoint"]
        or plan["input_stream"] != source["input_stream"]
        or len(plan["providers"]) not in (2, 3)
    ):
        raise ValueError("Host plan source/checkpoint differs")
    for actual, expected in zip(plan["providers"], providers):
        if actual != expected:
            raise ValueError("Host plan provider differs")


def add_export_arguments(parser, reference=False):
    parser.add_argument("--source-manifest", type=safe_path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--relocate", action="append", default=[])
    for name in ("vector", "scalar", "atlas-tools"):
        parser.add_argument("--" + name, type=safe_path, required=True)
    for name in ("vector-sha256", "scalar-sha256", "packing-provider-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--out", type=safe_path, required=True)
    if reference:
        parser.add_argument("--host-plan", type=safe_path, required=True)
        parser.add_argument("--host-plan-sha256", required=True)
    else:
        parser.add_argument("--inventory", type=safe_path, required=True)
        parser.add_argument("--inventory-sha256", required=True)
        parser.add_argument("--memory-plan", type=safe_path)
        parser.add_argument("--memory-plan-sha256")
        parser.add_argument(
            "--tile-shape",
            type=int,
            nargs=3,
            required=True,
            metavar=("M", "K", "N"),
            help="Explicit provider-declared physical matrix alignment",
        )


def artifact_output(path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from capture_tinyllama_native import artifact_output as checked

    return checked(path)


def source_bookend(args, records, source):
    """Freeze current producers and every original source binding before work."""
    from pathlib import Path

    root = Path(__file__).parent
    admitted = list(records) + [source[key] for key in ("checkpoint", "input_stream")]
    for key in ("immutable_bindings", "original_preparation_bindings"):
        admitted.extend(source.get(key, []))
    # Compare to already admitted digests rather than replacing their authority
    # with whatever bytes happen to be present when this snapshot is taken.
    bound = {}
    for item in admitted:
        current = file_identity(item["path"], safe_path)
        if current["sha256"] != item["sha256"] or bound.setdefault(current["path"], current) != current:
            raise ValueError("An admitted source changed before export: " + current["path"])
    paths = [x["path"] for x in admitted]
    paths.extend(
        root / name
        for name in (
            "resnet_provider.py",
            "resnet_graph_inputs.py",
            "resnet_native.py",
            "partition_resnet.py",
            "export_full_host_plan.py",
            "export_full_reference.py",
        )
    )
    paths.extend(root.parent / name for name in ("source_relocation.py", "capture_tinyllama_native.py",
                                                "atlas_provider_files.py"))
    paths.extend(source[key]["path"] for key in ("checkpoint", "input_stream"))
    paths.extend(
        record["path"]
        for key in ("immutable_bindings", "original_preparation_bindings")
        for record in source.get(key, [])
    )
    if hasattr(args, "atlas_tools"):
        paths.extend(packing_paths(args.atlas_tools))
    for key in ("inventory", "memory_plan", "host_plan"):
        if getattr(args, key, None) is not None:
            paths.append(getattr(args, key))
    for path in paths:
        current = file_identity(path, safe_path)
        if bound.setdefault(current["path"], current) != current:
            raise ValueError("An admitted source changed before export: " + current["path"])
    return bound


def finish_bookend(before):
    after = {path: file_identity(path, safe_path) for path in before}
    if after != before:
        raise ValueError("An admitted source changed during export")
    return {"before": before, "after": after, "unchanged": True}
