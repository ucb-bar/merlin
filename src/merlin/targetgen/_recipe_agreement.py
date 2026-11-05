"""Calibration agreement over every tensor result, without granting numerical admission."""

from __future__ import annotations

from contextlib import nullcontext
from itertools import islice


class AgreementError(ValueError):
    """The result trees have no complete, finite tensor correspondence."""


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

    def result(self):
        if not self.count:
            return {"samples": 0}
        return {
            "samples": self.count,
            "argmax_agreement": self.agree / self.count,
            "cosine_min": min(self.cosines),
            "cosine_mean": sum(self.cosines) / self.count,
            "max_abs_error": self.worst,
        }


def compare(reference, candidate, samples, *, limit, as_tuple):
    """Compare complete result trees; retain the single-result receipt's numeric behavior.

    Multiple results have a concatenated summary and individual output receipts,
    so a large output cannot hide a smaller output's error. Shape correspondence
    is checked per sample; dynamic extents may change between samples.
    """
    import torch

    try:
        from _capture_integer_mm import exact_integer_mm
    except ImportError:
        from merlin.targetgen._capture_integer_mm import exact_integer_mm

    if type(limit) is not int or limit < 0:
        raise AgreementError("agreement sample limit must be a nonnegative integer")
    summary, outputs = _Metrics(), []
    topology = None
    with torch.no_grad():
        for sample in islice(samples, limit):
            inputs = as_tuple(sample)
            expected, expected_tree = torch.utils._pytree.tree_flatten(reference(*inputs))
            eager_graph = (
                isinstance(candidate, torch.fx.GraphModule)
                and hasattr(candidate, "_compiled_call_impl")
                and candidate._compiled_call_impl is None
            )
            evaluation = exact_integer_mm(torch) if eager_graph else nullcontext()
            with evaluation:
                actual, actual_tree = torch.utils._pytree.tree_flatten(candidate(*inputs))
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
                if left.shape != right.shape or left.numel() == 0:
                    raise AgreementError(f"agreement output {index} has different or empty shapes")
                if left.is_complex() or right.is_complex():
                    raise AgreementError(f"agreement output {index} is complex")
                left, right = left.float().flatten(), right.float().flatten()
                if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                    raise AgreementError(f"agreement output {index} is not finite")
                if len(expected) > 1:
                    left, right = left.double(), right.double()
                pairs.append((left, right))
            if len(pairs) == 1:
                summary.add(*pairs[0])
            else:
                for metrics, pair in zip(outputs, pairs, strict=True):
                    metrics.add(*pair)
                summary.add(torch.cat([pair[0] for pair in pairs]), torch.cat([pair[1] for pair in pairs]))
    result = summary.result()
    if len(outputs) > 1:
        result["multi_output_metric_policy"] = "fp32_values_fp64_accumulation_v1"
        result["outputs"] = [{"output_index": index, **metrics.result()} for index, metrics in enumerate(outputs)]
    return result
