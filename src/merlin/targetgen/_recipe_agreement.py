"""Calibration agreement over every tensor result, without granting numerical admission."""

from __future__ import annotations

from contextlib import nullcontext
from itertools import islice


class AgreementError(ValueError):
    """The result trees have no complete, finite tensor correspondence."""


class NonfiniteReference(AgreementError):
    """The floating-point reference itself is NaN or infinite, so no agreement exists to report.

    Distinct from a nonfinite candidate: the quantized path did not introduce the special value,
    the source arithmetic did (a nonfinite input, an overflow, a degenerate normalization).
    """

    def __init__(self, *, sample, output, nonfinite_count):
        super().__init__(f"agreement reference output {output} of sample {sample} is not finite")
        self.details = {"sample": sample, "output_index": output, "nonfinite_count": nonfinite_count}


class _Metrics:
    def __init__(self):
        self.count = self.agree = 0
        self.cosines = []
        self.worst = 0.0

    def add(self, expected, actual):
        import torch

        self.cosines.append(float(torch.nn.functional.cosine_similarity(expected, actual, dim=0)))
        self.worst = max(self.worst, float((expected - actual).abs().max()))
        self.agree += int(int(expected.argmax()) == int(actual.argmax()))
        self.count += 1

    def result(self, empty=0):
        if not self.count:
            return {"samples": empty, "empty_samples": empty} if empty else {"samples": 0}
        return {
            "samples": self.count + empty,
            **({"empty_samples": empty} if empty else {}),
            "argmax_agreement": self.agree / self.count,
            "cosine_min": min(self.cosines),
            "cosine_mean": sum(self.cosines) / self.count,
            "max_abs_error": self.worst,
        }


def compare(reference, candidate, samples, *, limit, as_tuple):
    """Compare complete result trees; retain the single-result receipt's numeric behavior.

    Multiple results have a concatenated summary and individual output receipts,
    so a large output cannot hide a smaller output's error. Shape correspondence
    is checked per sample; dynamic extents may change between samples. An output
    empty on both sides has nothing to compare; a sample of only such outputs is
    counted, not scored. Each sample is restored after the reference call and the
    candidate runs on its own copy, so an in-place operation can affect neither.
    """
    import copy

    import torch

    try:
        from _capture_integer_mm import exact_integer_mm
    except ImportError:
        from merlin.targetgen._capture_integer_mm import exact_integer_mm

    if type(limit) is not int or limit < 0:
        raise AgreementError("agreement sample limit must be a nonnegative integer")
    summary, outputs = _Metrics(), []
    topology = None
    empty = 0
    with torch.no_grad():
        for sample_index, sample in enumerate(islice(samples, limit)):
            inputs = as_tuple(sample)
            pristine = copy.deepcopy(inputs)
            leaves = [leaf for leaf in torch.utils._pytree.tree_flatten(inputs)[0] if isinstance(leaf, torch.Tensor)]
            versions = [leaf._version for leaf in leaves]
            # The reference sees the caller's own objects, so a recorded forward still matches them.
            expected, expected_tree = torch.utils._pytree.tree_flatten(reference(*inputs))
            if [leaf._version for leaf in leaves] != versions:
                originals = torch.utils._pytree.tree_flatten(pristine)[0]
                for leaf, original in zip(leaves, (o for o in originals if isinstance(o, torch.Tensor)), strict=True):
                    leaf.copy_(original)
            candidate_inputs = copy.deepcopy(pristine)
            eager_graph = (
                isinstance(candidate, torch.fx.GraphModule)
                and hasattr(candidate, "_compiled_call_impl")
                and candidate._compiled_call_impl is None
            )
            evaluation = exact_integer_mm(torch) if eager_graph else nullcontext()
            with evaluation:
                actual, actual_tree = torch.utils._pytree.tree_flatten(candidate(*candidate_inputs))
            if not expected or expected_tree != actual_tree or len(expected) != len(actual):
                raise AgreementError("agreement requires matching nonempty result trees")
            if topology is None:
                topology = expected_tree
                outputs = [_Metrics() for _ in expected]
            elif topology != expected_tree:
                raise AgreementError("agreement result topology changed between samples")
            pairs = []
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                    raise AgreementError(f"agreement output {index} is not a tensor")
                if left.shape != right.shape:
                    raise AgreementError(f"agreement output {index} has different shapes")
                if left.is_complex() or right.is_complex():
                    raise AgreementError(f"agreement output {index} is complex")
                if left.numel() == 0:
                    pairs.append(None)
                    continue
                if left.is_floating_point() and not bool(torch.isfinite(left).all()):
                    count = int((~torch.isfinite(left)).sum())
                    raise NonfiniteReference(sample=sample_index, output=index, nonfinite_count=count)
                left, right = left.float().flatten(), right.float().flatten()
                if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                    raise AgreementError(f"agreement output {index} is not finite")
                if len(expected) > 1:
                    left, right = left.double(), right.double()
                pairs.append((left, right))
            scored = [pair for pair in pairs if pair is not None]
            if not scored:
                empty += 1
            elif len(pairs) == 1:
                summary.add(*pairs[0])
            else:
                for metrics, pair in zip(outputs, pairs, strict=True):
                    if pair is not None:
                        metrics.add(*pair)
                summary.add(torch.cat([pair[0] for pair in scored]), torch.cat([pair[1] for pair in scored]))
    result = summary.result(empty)
    if len(outputs) > 1:
        result["multi_output_metric_policy"] = "fp32_values_fp64_accumulation_v1"
        result["outputs"] = [{"output_index": index, **metrics.result()} for index, metrics in enumerate(outputs)]
    return result
