"""Level 3-5 model workloads: full PyTorch models exported via Turbine AOT."""

from __future__ import annotations

from dataclasses import dataclass, field

from workloads.base import Workload


@dataclass
class TorchModelWorkload(Workload):
    """Workload backed by a PyTorch nn.Module.

    For IREE: exports via iree.turbine.aot → MLIR → iree-compile.
    For torch.compile: uses the module directly.

    MLIR source is generated lazily on first call via Turbine export.
    """

    _model_factory: object = None

    def mlir_source(self, size: dict) -> str:
        raise NotImplementedError(
            f"{self.name}: use compile_from_torch() instead of mlir_source()"
        )

    def input_specs(self, size: dict) -> list[str]:
        shapes = self._get_input_shapes(size)
        return [
            "x".join(str(d) for d in shape) + "xf32" for shape in shapes
        ]

    def _get_input_shapes(self, size: dict) -> list[tuple[int, ...]]:
        raise NotImplementedError

    def torch_module(self, size: dict) -> tuple | None:
        if self._model_factory is None:
            return None
        return self._model_factory(size)


def _simple_mlp_factory(size):
    import torch
    import torch.nn as nn

    B = size.get("B", 1)
    I, H, O = size["I"], size["H"], size["O"]

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(I, H)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(H, O)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    return SimpleMLP(), [torch.randn(B, I)]


def _dronet_factory(size):
    import torch
    import torch.nn as nn

    B = size.get("B", 1)

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.downsample = None
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

        def forward(self, x):
            identity = x
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample:
                identity = self.downsample(x)
            return torch.relu(out + identity)

    class DroNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.block1 = ResBlock(32, 32, stride=2)
            self.block2 = ResBlock(32, 64, stride=2)
            self.block3 = ResBlock(64, 128, stride=2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 2)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    return DroNet().eval(), [torch.randn(B, 3, 224, 224)]


def _mobilenetv2_factory(size):
    import torch
    import torchvision.models as models

    B = size.get("B", 1)
    model = models.mobilenet_v2(weights=None).eval()
    return model, [torch.randn(B, 3, 224, 224)]


def _alexnet_factory(size):
    import torch
    import torchvision.models as models

    B = size.get("B", 1)
    model = models.alexnet(weights=None).eval()
    return model, [torch.randn(B, 3, 224, 224)]


@dataclass
class SimpleMLP(TorchModelWorkload):
    name: str = "simple_mlp"
    level: int = 3
    _model_factory: object = field(default=_simple_mlp_factory, repr=False)
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"B": 1, "I": 10, "H": 64, "O": 2, "desc": "tiny"},
            {"B": 32, "I": 256, "H": 512, "O": 128, "desc": "medium"},
            {"B": 128, "I": 1024, "H": 2048, "O": 512, "desc": "large"},
        ]
    )

    def _get_input_shapes(self, size):
        return [(size.get("B", 1), size["I"])]


@dataclass
class DroNetModel(TorchModelWorkload):
    name: str = "dronet"
    level: int = 3
    _model_factory: object = field(default=_dronet_factory, repr=False)
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"B": 1, "desc": "batch1"},
            {"B": 8, "desc": "batch8"},
            {"B": 32, "desc": "batch32"},
        ]
    )

    def _get_input_shapes(self, size):
        B = size.get("B", 1)
        return [(B, 3, 224, 224)]


@dataclass
class MobileNetV2Model(TorchModelWorkload):
    name: str = "mobilenet_v2"
    level: int = 4
    _model_factory: object = field(default=_mobilenetv2_factory, repr=False)
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"B": 1, "desc": "batch1"},
            {"B": 8, "desc": "batch8"},
            {"B": 32, "desc": "batch32"},
        ]
    )

    def _get_input_shapes(self, size):
        B = size.get("B", 1)
        return [(B, 3, 224, 224)]


@dataclass
class AlexNetModel(TorchModelWorkload):
    name: str = "alexnet"
    level: int = 4
    _model_factory: object = field(default=_alexnet_factory, repr=False)
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"B": 1, "desc": "batch1"},
            {"B": 8, "desc": "batch8"},
            {"B": 32, "desc": "batch32"},
        ]
    )

    def _get_input_shapes(self, size):
        B = size.get("B", 1)
        return [(B, 3, 224, 224)]


ALL_MODELS: list[Workload] = [
    SimpleMLP(),
    DroNetModel(),
    MobileNetV2Model(),
    AlexNetModel(),
]
