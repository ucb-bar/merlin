"""Evaluate Voyager's own quantization recipes on full validation sets (ImageNet-1k, GLUE SST-2).

Runs inside Voyager's environment (``out/build/voyager-venv``) and imports nothing from merlin, like
``voyager_h2h/scripts/voyager_export.py``: this arm is Voyager's stock code. The merlin-side driver
(``run_table3.py``) creates the product dir, records provenance and compares against the paper.

Reused from the pinned Voyager compiler checkout by import, never copied:
  * the flag parser -- ``voyager_compiler.add_quantization_args`` / ``add_compile_args``;
  * model loaders, calibration, dataset dumps and evaluate loops -- ``test/utils/models/{
    torchvision_models,bert,mobilebert}.py`` and ``test/utils/dataset/{imagenet,glue}.py``;
  * the quantizer -- ``get_default_quantizer`` then ``prepare_pt2e`` / ``convert_pt2e``;
  * the ``run_ci`` recipe -- ``test/run_ci.py::SCHEME_ARGS``;
  * image decoding -- the ``datasets`` webdataset builder ``retrieve_dataset`` streams ImageNet through
    (it applies EXIF orientation, so a hand-rolled tar reader would not be byte-identical).

Transcribed, because the module cannot be imported against this compiler (it needs ``param_pb2``,
retired at compiler a884ff3): the ``accelerator`` recipe, i.e. the per-datatype quantization flags of
voyager-accelerator ``run_regression.py::run_accuracy`` -- the flow whose gold table matches the paper's
Table 3 (INT8 calibrates 10 steps there, 3 in run_ci; MXINT8 blocks by max(IC, OC)).

Mirrored, because upstream fuses it with the hardware compile: the pre-compile half of
``torchvision_models.quantize_and_dump_model``, exported with a dynamic batch dimension so 50k images can
be evaluated in batches (``--check-static`` measures what that changes against the batch-1 graph). The
BERT-family helpers are called as-is with only their ``transform``/``compile`` hardware passes stubbed.

Redirected, and recorded in each result: Voyager's ``load_dataset("timm/imagenet-1k-wds")`` reads the
same shards from the local cache, and ``load_dataset("glue", task)`` is loaded as ``nyu-mll/glue`` (the
repo the hub redirects ``glue`` to; the bare id no longer parses under huggingface_hub 1.x).

Modes:
  index         key/label list of the ImageNet validation stream in Voyager's order (no image decode)
  quantized     ImageNet: evaluate FP32/BF16/quantized columns on a key selection, batched
  glue          SST-2: Voyager's own evaluate / evaluate_gm on the full validation split, per column
  test_codegen  Voyager's test/test_codegen.py main() as-is with --evaluate (ImageNet or SST-2)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import shlex
import shutil
import sys
import time
from pathlib import Path

import torch

COLUMNS = ("FP32", "BF16", "E4M3", "Posit8", "INT8", "MXINT8")
QUANTIZED = ("E4M3", "Posit8", "INT8", "MXINT8")
#: Table 3 column -> key into Voyager's run_ci.SCHEME_ARGS / SCHEME_INPUT_BYTES.
CI_SCHEME = {"E4M3": "E4M3", "Posit8": "P8_1", "INT8": "INT8", "MXINT8": "MXINT8"}
#: voyager-accelerator@e3a725db run_regression.py::run_accuracy, DATATYPE -> quantization_args
#: (CFLOAT is its FP32 run). ``{bs}`` is max(IC_DIMENSION, OC_DIMENSION).
ACCELERATOR_RECIPE = {
    "FP32": [],
    "BF16": ["--bf16"],
    "E4M3": ["--activation", "fp8_e4m3", "--weight", "fp8_e4m3", "--bf16"],
    "Posit8": ["--activation", "posit8_1", "--weight", "posit8_1", "--bf16"],
    "INT8": ["--activation", "int8,qs=per_tensor_symmetric", "--weight",
             "int8,qs=per_tensor_symmetric", "--bias", "int24", "--bf16", "--calibration_steps", "10"],
    "MXINT8": ["--force_scale_power_of_two", "--activation", "int8,qs=microscaling,bs={bs}",
               "--weight", "int8,qs=microscaling,bs={bs}", "--bf16"],
}
#: run_regression.py::run_accuracy model_path per (model, dataset); "accelerator:" = in that checkout.
GLUE_CHECKPOINTS = {"bert": "JeremiahZ/bert-base-uncased-sst2",
                    "mobilebert": "accelerator:models/mobilebert/mobilebert-tiny-sst2-bf16"}
HUB_IMAGENET = "timm/imagenet-1k-wds"  # what imagenet.retrieve_dataset streams
HUB_GLUE = "nyu-mll/glue"  # where the hub redirects the legacy "glue" id
SHARD_GLOB = "imagenet1k-validation-*.tar"
LFS_POINTER = b"version https://git-lfs"


def _voyager_root() -> Path:
    return Path(os.environ["MERLIN_EXT_VOYAGER_COMPILER"]).resolve()


def _voyager_modules():
    test_dir = str(_voyager_root() / "test")
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    import run_ci
    from utils.dataset import imagenet as v_imagenet
    from utils.models import torchvision_models as v_tv
    return run_ci, v_imagenet, v_tv


def _shards(shard_dir: str) -> list[str]:
    paths = sorted(str(p) for p in Path(shard_dir).glob(SHARD_GLOB))
    if not paths:
        raise SystemExit(f"no {SHARD_GLOB} under {shard_dir}")
    return paths


def _stream(shards: list[str], decode: bool = True):
    from datasets import Image, load_dataset
    ds = load_dataset("webdataset", data_files={"validation": shards}, split="validation",
                      streaming=True)
    if not decode:
        ds = ds.cast_column("jpg", Image(decode=False))
    return ds


def _redirect_imagenet(v_imagenet, shards: list[str]) -> None:
    upstream = v_imagenet.load_dataset

    def load(path, *args, **kwargs):
        if path != HUB_IMAGENET:
            raise RuntimeError(f"unexpected dataset {path!r}")
        return upstream("webdataset", data_files={"validation": shards},
                        split=kwargs.get("split", "validation"), streaming=True)

    v_imagenet.load_dataset = load


def _redirect_glue(v_glue) -> None:
    upstream = v_glue.load_dataset

    def load(path, *args, **kwargs):
        if path != "glue":
            raise RuntimeError(f"unexpected dataset {path!r}")
        return upstream(HUB_GLUE, *args, **kwargs)

    v_glue.load_dataset = load


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))


@contextlib.contextmanager
def _to_log(log: Path):
    """Voyager prints whole graph tables; keep them out of the worker's own stdout."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield


class ArgmaxRecorder(torch.nn.Module):
    """Wraps a model so Voyager's own batch-1 evaluate loop runs unchanged while each top-1 is kept.

    ``cast`` converts floating positional inputs to a dtype; used only where an upstream loop hands a
    model inputs of another dtype and raises (recorded by the caller)."""

    def __init__(self, inner, sink: list, cast=None):
        super().__init__()
        self.inner, self.sink, self.cast = inner, sink, cast

    def forward(self, *args, **kwargs):
        if self.cast is not None:
            args = tuple(x.to(self.cast) if torch.is_tensor(x) and x.is_floating_point() else x
                         for x in args)
        out = self.inner(*args, **kwargs)
        logits = out.logits if hasattr(out, "logits") else out
        self.sink.append(int(torch.argmax(logits, dim=-1).reshape(-1)[0].item()))
        return out


def _label(item) -> int:
    value = item["label"] if "label" in item else item["labels"]
    return int(value)


def _float_dtype(model):
    param = next((t for t in model.parameters() if t.is_floating_point()), None)
    return param.dtype if param is not None else None


# ---------------------------------------------------------------------------------------- recipes
def recipe_flags(column: str, recipe: str, run_ci, array: str) -> list[str]:
    ic, oc = (int(v) for v in array.split(","))
    if recipe == "accelerator":
        return [f.format(bs=max(ic, oc)) for f in ACCELERATOR_RECIPE[column]]
    if recipe != "ci":
        raise SystemExit(f"unknown recipe {recipe!r}")
    if column == "FP32":
        return []
    if column == "BF16":
        return ["--bf16"]
    return shlex.split(run_ci.SCHEME_ARGS[CI_SCHEME[column]])


def voyager_args(model: str, column: str, run_ci, recipe: str, array: str, extra=()):
    """Voyager's own argparse namespace for one column."""
    from voyager_compiler import add_compile_args, add_quantization_args
    p = argparse.ArgumentParser()
    add_quantization_args(p)
    add_compile_args(p)
    # test_codegen.py's own flags that its model helpers read.
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--task_name", default="sst2")
    p.add_argument("--quantize_fc", action="store_true")
    p.add_argument("--use_maxpool_2x2", action="store_true")
    p.add_argument("--conv2d_im2col", action="store_true")
    p.add_argument("--num_hidden_layers", type=int, default=None)
    p.add_argument("--model_output_dir", default=None)
    p.add_argument("--dump_tensors", action="store_true")
    p.add_argument("--debug", action="store_true")
    flags = recipe_flags(column, recipe, run_ci, array)
    args = p.parse_args(flags + list(extra))
    args.model = model
    return args, flags


def _quantizer(args):
    from voyager_compiler import get_default_quantizer
    return get_default_quantizer(input_activation=args.activation,
                                 output_activation=args.output_activation, weight=args.weight,
                                 bias=args.bias,
                                 force_scale_power_of_two=args.force_scale_power_of_two)


def _versions() -> dict:
    import datasets
    import PIL
    import torchao
    import torchvision
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "torchao": torchao.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__,
            "pillow": PIL.__version__}


# ---------------------------------------------------------------------------------------- ImageNet
def quantize_torchvision(model, quantizer, calib_images, args, dtype, *, dynamic: bool,
                         batch_max: int):
    """The pre-compile half of Voyager's torchvision_models.quantize_and_dump_model (resnet path)."""
    from voyager_compiler import (DerivedQuantizationSpec, QuantizationConfig, QuantizationSpec,
                                  convert_pt2e, derive_bias_qparams_fn, export_model, prepare_pt2e,
                                  replace_conv2d_with_im2col)
    from voyager_compiler.quantization.quantize import get_conv_bn_layers

    pairs = get_conv_bn_layers(model)
    if pairs:
        model = torch.ao.quantization.fuse_modules(model, pairs, inplace=True)
    if args.use_maxpool_2x2:
        raise NotImplementedError("no Table 3 recipe sets --use_maxpool_2x2")
    if not args.quantize_fc:
        quantizer.set_module_name("fc", None)
    if args.residual is not None:
        qspec = QuantizationSpec.from_str(f"{args.residual},qs=per_tensor_symmetric")
        qconfig = QuantizationConfig(qspec, None, None, None)
        quantizer.set_object_type(torch.ops.aten.add.Tensor, qconfig)
        quantizer.set_object_type(torch.ops.aten.add_.Tensor, qconfig)
    if args.activation is not None and "microscaling" in args.activation:
        # Upstream: conv1 uses per-tensor instead of microscaling; nfA_B maps to intB.
        dtype_name = args.activation.split(",")[0]
        head, _, tail = dtype_name.lower().partition("_")
        if head.startswith("nf") and head[2:].isdigit() and tail.isdigit():
            dtype_name = f"int{tail}"
        qspec = QuantizationSpec.from_str(f"{dtype_name},qs=per_tensor_symmetric")
        bias_qspec = DerivedQuantizationSpec(derived_from=None,
                                             derive_qparams_fn=derive_bias_qparams_fn, dtype=None)
        quantizer.set_module_name("^conv1$", QuantizationConfig(qspec, None, qspec, bias_qspec))

    if dynamic:
        example = (torch.randn(2, 3, 224, 224, dtype=dtype),)
        shapes = ({0: torch.export.Dim("bs", min=1, max=batch_max)},)
    else:
        example = (torch.randn(1, 3, 224, 224, dtype=dtype),)
        shapes = None
    gm = export_model(model, example, dynamic_shapes=shapes)
    if args.conv2d_im2col:
        replace_conv2d_with_im2col(gm)  # upstream: must precede prepare_pt2e
    gm = prepare_pt2e(gm, quantizer)
    for image in calib_images[: args.calibration_steps]:
        with torch.no_grad():
            gm(image.to(dtype))
    convert_pt2e(gm, args.bias)
    return gm


def _weights_record(model_name: str, weights_arg: str) -> dict:
    """Which torchvision checkpoint ``models.<name>(weights=weights_arg)`` resolved to, and its bytes."""
    import hashlib
    from torchvision.models import get_model_weights
    enum = get_model_weights(model_name)
    w = enum.DEFAULT if weights_arg == "DEFAULT" else enum[weights_arg]
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(w.url).name
    return {"enum": f"{enum.__name__}.{w.name}", "url": w.url, "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "torchvision_acc1": w.meta["_metrics"]["ImageNet-1K"]["acc@1"],
            "torchvision_eval_transforms": str(w.transforms())}


def build_torchvision_column(a, column, run_ci, v_tv, calib_images, *, dynamic: bool):
    args, flags = voyager_args(a.model, column, run_ci, a.recipe, a.array)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = v_tv.load_model(args)  # weights=args.model_name_or_path ("DEFAULT"); .bfloat16() on --bf16
    info = {"flags": flags, "weights_arg": args.model_name_or_path,
            "weights": _weights_record(a.model, args.model_name_or_path), "dtype": str(dtype),
            "calibration_steps": args.calibration_steps, "bias": args.bias,
            "quantize_fc": args.quantize_fc, "conv2d_im2col": args.conv2d_im2col}
    if column not in QUANTIZED:
        return model.eval(), dtype, info
    gm = quantize_torchvision(model, _quantizer(args), calib_images, args, dtype, dynamic=dynamic,
                              batch_max=a.batch)
    return gm, dtype, info


def _first_images(shards: list[str], n: int, transform):
    """The first ``n`` stream items exactly as Voyager's retrieve_dataset makes them."""
    return [(item["__key__"], transform(item["jpg"]).unsqueeze(0), int(item["cls"]))
            for item in _stream(shards).take(n)]


class _Collate:
    """Decode + preprocess in the DataLoader worker; drops keys outside the selection."""

    def __init__(self, transform, selected: set[str]):
        from datasets import Image
        self.transform, self.selected, self.image = transform, selected, Image()

    def __call__(self, examples):
        xs, ys, ks = [], [], []
        for ex in examples:
            if ex["__key__"] not in self.selected:
                continue
            xs.append(self.transform(self.image.decode_example(ex["jpg"])))
            ys.append(int(ex["cls"]))
            ks.append(ex["__key__"])
        if not xs:
            return None
        return torch.stack(xs), torch.tensor(ys), ks


def mode_index(a) -> int:
    shards = _shards(a.shards)
    keys, labels, counts = [], [], {}
    t0 = time.time()
    for ex in _stream(shards, decode=False):
        keys.append(ex["__key__"])
        labels.append(int(ex["cls"]))
        name = Path(ex["__url__"]).name
        counts[name] = counts.get(name, 0) + 1
    _write_json(Path(a.out), {"keys": keys, "labels": labels, "shard_counts": counts,
                              "shards": [Path(s).name for s in shards], "seconds": time.time() - t0})
    print(json.dumps({"index": a.out, "n": len(keys)}))
    return 0


def _check_static(a, column, run_ci, v_tv, calib_images, dyn_gm, dtype, probe):
    """Does the dynamic-batch export change anything against the batch-1 graph upstream builds?"""
    static_gm, _, _ = build_torchvision_column(a, column, run_ci, v_tv, calib_images, dynamic=False)
    x = torch.cat([img for _, img, _ in probe]).to(dtype)
    ref = torch.cat([static_gm(x[i:i + 1]) for i in range(len(x))]).float()
    dyn_b1 = torch.cat([dyn_gm(x[i:i + 1]) for i in range(len(x))]).float()
    dyn_bn = dyn_gm(x).float()
    return {"n": len(x),
            "batch1_max_abs_diff": (dyn_b1 - ref).abs().max().item(),
            "batchN_max_abs_diff": (dyn_bn - ref).abs().max().item(),
            "batch1_top1_agree": int((dyn_b1.argmax(1) == ref.argmax(1)).sum()),
            "batchN_top1_agree": int((dyn_bn.argmax(1) == ref.argmax(1)).sum()),
            "ref_logit_absmax": ref.abs().max().item()}


def mode_quantized(a) -> int:
    run_ci, v_imagenet, v_tv = _voyager_modules()
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    shards = _shards(a.shards)
    index = json.loads(Path(a.index).read_text())
    selected = set(json.loads(Path(a.select).read_text()))
    transform = v_imagenet.get_transforms("resnet")
    out = Path(a.out)

    columns = a.columns.split(",")
    n_calib = max([voyager_args(a.model, c, run_ci, a.recipe, a.array)[0].calibration_steps
                   for c in columns] + [a.check_static])
    first = _first_images(shards, n_calib, transform)
    if [k for k, _, _ in first] != index["keys"][:len(first)]:
        raise SystemExit("stream order differs from the index; rebuild the index")
    calib_images = [img for _, img, _ in first]

    models, info = {}, {}
    for col in columns:
        t0 = time.time()
        m, dtype, meta = build_torchvision_column(a, col, run_ci, v_tv, calib_images, dynamic=True)
        meta["build_seconds"] = time.time() - t0
        meta["calibration_keys"] = index["keys"][:meta["calibration_steps"]]
        if a.check_static and col in QUANTIZED:
            meta["static_check"] = _check_static(a, col, run_ci, v_tv, calib_images, m, dtype,
                                                 first[:a.check_static])
        models[col], info[col] = (m, dtype), meta
        print(json.dumps({"built": col, "flags": meta["flags"],
                          "build_seconds": round(meta["build_seconds"], 1)}), flush=True)

    loader = torch.utils.data.DataLoader(_stream(shards, decode=False), batch_size=a.batch,
                                         num_workers=a.workers,
                                         collate_fn=_Collate(transform, selected))
    preds, top5 = {c: {} for c in columns}, {c: {} for c in columns}
    labels, seconds = {}, {c: 0.0 for c in columns}
    t_start, seen = time.time(), 0
    for batch in loader:
        if batch is None:
            continue
        x, y, keys = batch
        for col, (m, dtype) in models.items():
            t0 = time.time()
            logits = m(x.to(dtype)).float()
            seconds[col] += time.time() - t0
            for k, row, lab in zip(keys, logits.topk(5, dim=1).indices.tolist(), y.tolist()):
                preds[col][k] = row[0]
                top5[col][k] = lab in row
        for k, lab in zip(keys, y.tolist()):
            labels[k] = lab
        seen += len(keys)
        if seen // a.progress != (seen - len(keys)) // a.progress:
            print(json.dumps({"seen": seen, "elapsed": round(time.time() - t_start, 1)}), flush=True)

    order = [k for k in index["keys"] if k in labels]
    missing = sorted(selected - set(labels))
    result = {
        "task": "imagenet", "model": a.model, "recipe": a.recipe, "array": a.array,
        "n": len(order), "n_missing": len(missing), "missing_selected_keys": missing[:20],
        "threads": a.threads, "workers": a.workers, "batch": a.batch,
        "wall_seconds": time.time() - t_start, "versions": _versions(),
        "preprocessing": str(transform),
        "columns": {c: {**info[c],
                        "top1_correct": sum(preds[c][k] == labels[k] for k in order),
                        "top5_correct": sum(top5[c][k] for k in order),
                        "model_seconds": seconds[c],
                        "images_per_second": len(order) / seconds[c] if seconds[c] else None}
                    for c in columns},
    }
    _write_json(out / "results.json", result)
    _write_json(out / "preds.json", {"keys": order, "labels": [labels[k] for k in order],
                                     "top1": {c: [preds[c][k] for k in order] for c in columns}})
    print(json.dumps({"done": str(out), "n": len(order)}))
    return 0


# ---------------------------------------------------------------------------------------- GLUE
def _glue_checkpoint(model: str) -> str:
    ref = GLUE_CHECKPOINTS[model]
    if not ref.startswith("accelerator:"):
        return ref
    root = os.environ.get("MERLIN_EXT_VOYAGER_ACCELERATOR")
    if not root:
        raise SystemExit(f"{model}: needs MERLIN_EXT_VOYAGER_ACCELERATOR for {ref}")
    path = Path(root) / ref.partition(":")[2]
    for f in sorted(path.glob("*")):
        if f.is_file() and f.read_bytes()[:len(LFS_POINTER)] == LFS_POINTER:
            raise SystemExit(f"BLOCKED: {path} holds unfetched git-lfs pointers ({f.name})")
    return str(path)


def _hf_revision(ref: str) -> str | None:
    """The commit a hub checkpoint id resolved to in the local cache (None for a local path)."""
    if Path(ref).exists():
        return None
    from huggingface_hub import scan_cache_dir
    for repo in scan_cache_dir().repos:
        if repo.repo_id == ref and repo.repo_type == "model":
            return ",".join(sorted(r.commit_hash for r in repo.revisions))
    return None


def mode_glue(a) -> int:
    run_ci, _, _ = _voyager_modules()
    import test_codegen  # its VECTOR_PIPELINE, handed to the helper exactly as its main() does
    from utils.dataset import glue as v_glue
    from utils.models import bert as v_bert
    from utils.models import mobilebert as v_mobilebert
    helper = {"bert": v_bert, "mobilebert": v_mobilebert}[a.model]
    _redirect_glue(v_glue)
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    ckpt = _glue_checkpoint(a.model)
    out, work = Path(a.out), Path(a.work)
    log = out / "voyager_stdout.log"
    columns = a.columns.split(",")
    result = {"task": "sst2", "model": a.model, "checkpoint": ckpt, "checkpoint_revision":
              _hf_revision(ckpt), "recipe": a.recipe, "array": a.array, "threads": a.threads,
              "dataset": {"repo": HUB_GLUE, "config": "sst2", "split": "validation",
                          "redirected_from": "glue"},
              "versions": _versions(), "columns": {}}
    preds = {}
    labels = None
    for col in columns:
        t0 = time.time()
        args, flags = voyager_args(a.model, col, run_ci, a.recipe, a.array,
                                   extra=["--model_name_or_path", ckpt, "--task_name", "sst2"])
        meta = {"flags": flags, "calibration_steps": args.calibration_steps,
                "calibration_source": "train split, first calibration_steps rows, batch 1"}
        try:
            sink = _glue_column(a, col, args, helper, v_glue, test_codegen, log, work, meta)
        except Exception as exc:  # a native failure is this column's result; keep the others
            meta["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            meta["seconds"] = time.time() - t0
            result["columns"][col] = meta
            preds[col] = None
            print(json.dumps({"column": col, "error": meta["error"]}), flush=True)
            continue
        col_labels = meta.pop("labels")
        labels = labels or col_labels
        if col_labels != labels:
            raise SystemExit("validation labels differ between columns")
        meta["seconds"] = time.time() - t0
        meta["n"] = len(sink)
        meta["top1_correct"] = sum(p == l for p, l in zip(sink, labels))
        result["columns"][col] = meta
        preds[col] = sink
        print(json.dumps({"column": col, "correct": meta["top1_correct"], "n": meta["n"],
                          "seconds": round(meta["seconds"])}), flush=True)
    _write_json(out / "results.json", result)
    _write_json(out / "preds.json", {"labels": labels, "top1": preds})
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"done": str(out)}))
    return 0


def _glue_column(a, col, args, helper, v_glue, test_codegen, log, work, meta) -> list[int]:
    """One SST-2 column through Voyager's own helpers; returns per-sentence top-1."""
    with _to_log(log):
        model, tokenizer = helper.load_model(args)
        eval_ds, train_ds = v_glue.retrieve_dataset(model, tokenizer, args)
    meta["labels"] = [int(v) for v in eval_ds["labels"]]
    sink: list[int] = []
    if col not in QUANTIZED:
        with _to_log(log):
            helper.evaluate(ArgmaxRecorder(model, sink), eval_ds)
        meta["evaluated"] = "Voyager evaluate(model, eval_dataset): unquantized HF model"
        return sink
    dump = work / f"{a.model}_{col}"
    with _to_log(log):
        # test_codegen dumps the evaluation set before quantizing; keep that order.
        prepared = v_glue.dump_dataset(str(dump), eval_ds, model)
        saved = helper.transform, helper.compile
        helper.transform = helper.compile = lambda *x, **k: None  # hardware passes only
        try:
            gm, _, _ = helper.quantize_and_dump_model(
                model=model, quantizer=_quantizer(args), calibration_data=train_ds,
                vector_stages=test_codegen.VECTOR_PIPELINE, args=args)
        finally:
            helper.transform, helper.compile = saved
        # Upstream defect at f9d4c498: dump_dataset builds the attention mask in fp32
        # ((1 - mask) * finfo(float32).min) while a --bf16 graph was traced on a bf16 mask
        # (get_extended_attention_mask, finfo(bf16).min); the fp32 scores then meet bf16 V and
        # evaluate_gm raises "expected m1 and m2 to have the same dtype". Cast the dumped floats
        # to the model dtype -- the values the traced path itself produces.
        cast = torch.bfloat16 if args.bf16 else None
        meta["input_cast"] = str(cast) if cast is not None else None
        helper.evaluate_gm(ArgmaxRecorder(gm, sink, cast), prepared)
    shutil.rmtree(dump, ignore_errors=True)
    meta["evaluated"] = ("Voyager evaluate_gm(gm, dump_dataset(...)): the converted "
                         "(pre-transform) quantized graph; transform/compile stubbed")
    return sink


# ---------------------------------------------------------------------------------------- as-is
def mode_test_codegen(a) -> int:
    """Voyager's test/test_codegen.py --evaluate, run as-is; returns what its evaluate loops saw."""
    run_ci, v_imagenet, v_tv = _voyager_modules()
    from utils.dataset import glue as v_glue
    from utils.models import bert as v_bert
    from utils.models import mobilebert as v_mobilebert
    root = _voyager_root()
    if a.column not in QUANTIZED:
        raise SystemExit("test_codegen mode needs a quantized column; the unquantized model is its "
                         "first evaluate()")
    # 1. Dataset sources (recorded): local ImageNet shards; "glue" -> nyu-mll/glue.
    if a.shards:
        _redirect_imagenet(v_imagenet, _shards(a.shards))
    _redirect_glue(v_glue)
    # 2. Thread cap: test_codegen calls torch.set_num_threads(32); this host is shared.
    upstream_threads = torch.set_num_threads
    torch.set_num_threads = lambda n: upstream_threads(min(int(n), a.threads))
    # 3. Record per-example predictions around each upstream evaluate loop.
    stages = []
    flags = recipe_flags(a.column, a.recipe, run_ci, a.array)
    model_dtype = torch.bfloat16 if "--bf16" in flags else None
    gm_loops = {v_bert.evaluate_gm, v_mobilebert.evaluate_gm}
    tv_eval = v_tv.evaluate  # captured before it is replaced below

    def recording(upstream):
        def evaluate(model, dataset):
            first = not stages
            stage = {"stage": "unquantized_model" if first else "lowered_graph",
                     "labels": [_label(item) for item in dataset], "preds": []}
            # Two upstream dtype defects at f9d4c498 under --bf16, each cast to the model dtype
            # and recorded: torchvision_models.evaluate() hands retrieve_dataset's fp32 images to
            # the bf16 model ("Input type (torch.FloatTensor) and weight type (CPUBFloat16Type)
            # should be the same"), and glue.dump_dataset builds an fp32 attention mask for a
            # graph traced on a bf16 one, so {bert,mobilebert}.evaluate_gm raises "expected m1
            # and m2 to have the same dtype". The ImageNet lowered stage is already cast by
            # dump_imagenet.
            cast = None
            if (first and upstream is tv_eval) or upstream in gm_loops:
                cast = model_dtype
            stage["input_cast"] = str(cast) if cast is not None else None
            t0 = time.time()
            upstream(ArgmaxRecorder(model, stage["preds"], cast), dataset)
            stage["seconds"] = time.time() - t0
            stage["n"] = len(stage["preds"])
            stage["top1_correct"] = sum(p == l for p, l in zip(stage["preds"], stage["labels"]))
            stages.append(stage)
        return evaluate

    v_tv.evaluate = recording(v_tv.evaluate)
    for helper in (v_bert, v_mobilebert):
        helper.evaluate = recording(helper.evaluate)
        helper.evaluate_gm = recording(helper.evaluate_gm)

    out, work = Path(a.out), Path(a.work)
    compile_dir, dump_dir = work / "compile", work / "dataset_dump"
    for d in (compile_dir, dump_dir):
        d.mkdir(parents=True, exist_ok=True)
    cols = int(a.array.split(",")[1])
    argv = ["test_codegen.py", a.model, "--debug"]
    if a.model in GLUE_CHECKPOINTS:
        argv += ["--model_name_or_path", _glue_checkpoint(a.model), "--task_name", "sst2"]
    argv += recipe_flags(a.column, a.recipe, run_ci, a.array)
    argv += ["--pe_array_size", a.array]
    # Compile-side flags exactly as run_ci builds them for this compiler.
    if "--layout_policy" not in argv:
        argv += ["--layout_policy", "systolic"]
    for flag, value in run_ci.DEFAULT_TILER_ARGS.items():
        if flag not in argv:
            argv += [flag, value]
    if "--bank_width" not in argv:
        argv += ["--bank_width", str(int(cols * run_ci.SCHEME_INPUT_BYTES[CI_SCHEME[a.column]]))]
    argv += ["--model_output_dir", str(compile_dir), "--evaluate", "--dump_dataset",
             "--dataset_output_dir", str(dump_dir)]

    log_path = out / "test_codegen.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0, error = time.time(), None
    saved_argv, sys.argv = sys.argv, argv
    try:
        with open(log_path, "w") as log, contextlib.redirect_stdout(log), \
                contextlib.redirect_stderr(log):
            runpy.run_path(str(root / "test" / "test_codegen.py"), run_name="__main__")
    except BaseException as exc:  # a native failure is the arm's result; record it
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.argv = saved_argv
    printed = [line.strip() for line in log_path.read_text().splitlines() if " Accuracy: " in line]
    _write_json(out / "results.json", {
        "model": a.model, "column": a.column, "recipe": a.recipe, "argv": argv[1:],
        "error": error, "wall_seconds": time.time() - t0, "printed_accuracy_lines": printed,
        "stages": stages, "threads": a.threads,
        "note": "stage preds follow the evaluation stream order (ImageNet: first n index keys)"})
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"done": str(out), "error": error,
                      "stages": [(s["stage"], s["top1_correct"], s["n"]) for s in stages]}))
    return 0 if error is None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("index")
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    for name in ("quantized", "glue", "test_codegen"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--threads", type=int, default=12)
        p.add_argument("--recipe", default="accelerator", choices=("accelerator", "ci"))
        p.add_argument("--array", default="16,16", help="IC,OC: MXINT8 block size and the compile")
        if name == "quantized":
            p.add_argument("--columns", default="FP32,BF16,INT8,MXINT8")
            p.add_argument("--shards", required=True)
            p.add_argument("--index", required=True)
            p.add_argument("--select", required=True, help="JSON list of keys to evaluate")
            p.add_argument("--workers", type=int, default=3)
            p.add_argument("--batch", type=int, default=50)
            p.add_argument("--check-static", type=int, default=0)
            p.add_argument("--progress", type=int, default=2000)
        elif name == "glue":
            p.add_argument("--columns", default="FP32,BF16,INT8,MXINT8")
            p.add_argument("--work", required=True, help="scratch dir for the dataset dump")
        else:
            p.add_argument("--column", required=True)
            p.add_argument("--shards", default=None, help="ImageNet shards (vision models)")
            p.add_argument("--work", required=True, help="scratch dir for the compile and dump")
            p.add_argument("--keep-work", action="store_true")
    a = parser.parse_args(argv)
    return {"index": mode_index, "quantized": mode_quantized, "glue": mode_glue,
            "test_codegen": mode_test_codegen}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
