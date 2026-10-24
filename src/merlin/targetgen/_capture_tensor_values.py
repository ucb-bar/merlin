"""JSON-safe tensor storage values for isolated framework capture workers."""


def to_native(t):
    """A torch tensor -> JSON-safe lists without rounding integer or boolean values."""
    import torch

    if isinstance(t, torch.Tensor):
        value = t.detach().cpu()
        if value.is_complex():

            def encode(value):
                if isinstance(value, list):
                    return [encode(item) for item in value]
                return {"kind": "complex", "real": value.real, "imag": value.imag}

            return encode(value.tolist())
        return (value if not (value.is_floating_point() or value.is_complex()) else value.to(torch.float64)).tolist()
    if isinstance(t, (list, tuple)):
        return [to_native(x) for x in t]
    return t
