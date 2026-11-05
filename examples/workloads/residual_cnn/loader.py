"""Independent residual-CNN iteration model, not a ResNet50 implementation.

An optional adjacent ``profile.json`` (written by ``merlin experiment corpus variants``) is a
selected capture input: the sealed capture inventories its bytes with this loader before the
model is built. Without one, the model and its inputs are exactly the original single-block,
16x16 development workload.

Beyond the required ``channels``/``spatial_side``/``blocks``, a profile may opt into further
independently authored patterns; each is off when its key is absent:

* ``stem``: ``"conv3"`` (default) or ``"conv7s2"``, a 7x7 stride-2 pad-3 stem on the 3 input channels;
* ``stem_pool``: a 3x3 stride-2 pad-1 max pool after the stem activation;
* ``downsample``: a final stride-2 block that doubles the channels, with a 3x3 stride-2 pad-1
  convolution on the branch and a 1x1 stride-2 projection on the shortcut;
* ``bottleneck``: residual branches of 1x1 reduce (to a quarter of the channels), 3x3, 1x1 expand;
* ``batchnorm``: an eval-mode ``BatchNorm2d`` with deterministic non-trivial running statistics
  after every convolution (whose own bias is then dropped).
"""

import hashlib
import json
from pathlib import Path

import torch

PROFILE_SCHEMA = "merlin.iteration_workload_profile.v1"
REQUIRED_KEYS = frozenset({"schema", "workload_id", "channels", "spatial_side", "blocks"})
DEFAULTS = {
    "channels": 8,
    "spatial_side": 16,
    "blocks": 1,
    "stem": "conv3",
    "stem_pool": False,
    "downsample": False,
    "bottleneck": False,
    "batchnorm": False,
}
OPTIONAL_KEYS = frozenset(DEFAULTS) - REQUIRED_KEYS
STEMS = ("conv3", "conv7s2")


class ResidualCNN(torch.nn.Module):
    def __init__(
        self,
        channels=8,
        blocks=1,
        *,
        stem="conv3",
        stem_pool=False,
        downsample=False,
        bottleneck=False,
        batchnorm=False,
    ):
        super().__init__()

        def conv(in_channels, out_channels, kernel, stride=1):
            layer = torch.nn.Conv2d(
                in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=not batchnorm
            )
            return torch.nn.Sequential(layer, torch.nn.BatchNorm2d(out_channels)) if batchnorm else layer

        width = max(1, channels // 4) if bottleneck else channels
        branch = ((channels, width, 1), (width, width, 3)) if bottleneck else ((channels, channels, 3),) * 2
        self.stem_pool = stem_pool
        self.bottleneck = bottleneck
        self.downsample = downsample
        self.stem = conv(3, channels, 7, 2) if stem == "conv7s2" else conv(3, channels, 3)
        self.conv1 = conv(*branch[0])
        self.conv2 = conv(*branch[1])
        self.head = torch.nn.Linear(2 * channels if downsample else channels, 4)
        # Registered after the original modules so the single-block model keeps its module order,
        # state keys and seeded parameters.
        self.residuals = torch.nn.ModuleList(
            torch.nn.ModuleList([conv(*layer) for layer in branch] + ([conv(width, channels, 1)] if bottleneck else []))
            for _ in range(blocks - 1)
        )
        # Optional patterns register after everything above, so a profile without them is unchanged.
        if bottleneck:
            self.expand = conv(width, channels, 1)
        if downsample:
            self.down_conv1 = conv(channels, 2 * channels, 3, 2)
            self.down_conv2 = conv(2 * channels, 2 * channels, 3)
            self.down_shortcut = conv(channels, 2 * channels, 1, 2)

    @staticmethod
    def _branch(layers, features):
        for index, layer in enumerate(layers):
            features = layer(features)
            if index + 1 < len(layers):
                features = torch.relu(features)
        return features

    def forward(self, image):
        features = torch.relu(self.stem(image))
        if self.stem_pool:
            features = torch.nn.functional.max_pool2d(features, 3, stride=2, padding=1)
        first = (self.conv1, self.conv2, self.expand) if self.bottleneck else (self.conv1, self.conv2)
        residual = self._branch(first, features)
        features = torch.relu(features + residual)
        for layers in self.residuals:
            residual = self._branch(layers, features)
            features = torch.relu(features + residual)
        if self.downsample:
            residual = self.down_conv2(torch.relu(self.down_conv1(features)))
            features = torch.relu(self.down_shortcut(features) + residual)
        pooled = features.mean(dim=(-2, -1))
        return self.head(pooled)


def validate_profile(profile):
    """The complete profile with defaults filled in, or ``ValueError`` for any unsupported field."""
    if (
        not isinstance(profile, dict)
        or not REQUIRED_KEYS <= set(profile)
        or set(profile) - REQUIRED_KEYS - OPTIONAL_KEYS
    ):
        raise ValueError("residual CNN profile has an unsupported shape")
    if profile["schema"] != PROFILE_SCHEMA or profile["workload_id"] != "residual_cnn":
        raise ValueError("residual CNN profile has an unsupported identity")
    for key, maximum in (("channels", 256), ("spatial_side", 512), ("blocks", 16)):
        if type(profile[key]) is not int or not 1 <= profile[key] <= maximum:
            raise ValueError(f"residual CNN profile has an invalid {key}")
    if "stem" in profile and (type(profile["stem"]) is not str or profile["stem"] not in STEMS):
        raise ValueError("residual CNN profile has an invalid stem")
    for key in ("stem_pool", "downsample", "bottleneck", "batchnorm"):
        if key in profile and type(profile[key]) is not bool:
            raise ValueError(f"residual CNN profile has an invalid {key}")
    return {**DEFAULTS, **profile}


def _selected_profile():
    path = Path(__file__).with_name("profile.json")
    if not path.exists():
        return dict(DEFAULTS), None
    if path.is_symlink() or not path.is_file():
        raise ValueError("residual CNN profile must be an adjacent ordinary file")
    raw = path.read_bytes()
    return validate_profile(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def _seed_batchnorm(model):
    """Deterministic non-identity scale, shift and running statistics for every eval-mode norm."""
    for ordinal, module in enumerate(model.modules()):
        if isinstance(module, torch.nn.BatchNorm2d):
            index = torch.arange(module.num_features, dtype=torch.int64)
            module.weight.copy_(1 + (((index * 7 + ordinal) % 9) - 4).to(torch.float32) / 16)
            module.bias.copy_((((index * 5 + ordinal * 3) % 11) - 5).to(torch.float32) / 64)
            module.running_mean.copy_((((index * 3 + ordinal) % 13) - 6).to(torch.float32) / 128)
            module.running_var.copy_(0.5 + ((index * 11 + ordinal * 5) % 17).to(torch.float32) / 16)


def get_model_and_inputs():
    profile, profile_sha256 = _selected_profile()
    side = profile["spatial_side"]
    elements = 3 * side * side
    image = torch.arange(elements, dtype=torch.float32).reshape(1, 3, side, side) / elements
    model = ResidualCNN(
        profile["channels"],
        profile["blocks"],
        stem=profile["stem"],
        stem_pool=profile["stem_pool"],
        downsample=profile["downsample"],
        bottleneck=profile["bottleneck"],
        batchnorm=profile["batchnorm"],
    ).eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
        _seed_batchnorm(model)
    if profile_sha256 is not None:
        model.session_provenance = {
            "workload_id": "residual_cnn",
            "workload_role": "iteration",
            "input_source": "seeded_synthetic",
            "profile_sha256": profile_sha256,
        }
    return model, (image,)
