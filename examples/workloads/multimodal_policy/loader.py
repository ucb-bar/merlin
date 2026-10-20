"""Independent vision/text/state policy pattern, not a SmolVLA checkpoint or session.

An optional adjacent ``profile.json`` (schema ``merlin.iteration_workload_profile.v1``,
``workload_id: multimodal_policy``) is a selected capture input. Without one, the model and its
inputs are exactly the original 4-patch, 4-token, single-query development workload. Each
optional key keeps its default when absent:

* ``tokens`` (4, at most 32): text tokens; any count, so the cross-attention key/value length
  (4 vision patches plus the text tokens) need not be a multiple of a tile;
* ``queries`` (1, at most 16): query positions, the projected state plus a learned per-position
  offset, so the cross-attention query length differs from its key/value length and the output
  becomes a ``(1, queries, 4)`` action chunk;
* ``gelu_mlp`` (false): a residual tanh-GELU MLP block over the normalized context;
* ``time_embedding`` (false): an extra scalar timestep input whose sinusoidal (exp-spaced
  frequencies, sin and cos) embedding is projected and added to the projected state.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

PROFILE_SCHEMA = "merlin.iteration_workload_profile.v1"
REQUIRED_KEYS = frozenset({"schema", "workload_id"})
DEFAULTS = {"tokens": 4, "queries": 1, "gelu_mlp": False, "time_embedding": False}
OPTIONAL_KEYS = frozenset(DEFAULTS)
WIDTH = 16


class MultimodalPolicy(torch.nn.Module):
    def __init__(self, queries=1, gelu_mlp=False, time_embedding=False):
        super().__init__()
        self.queries, self.gelu_mlp, self.time_embedding = queries, gelu_mlp, time_embedding
        self.vision = torch.nn.Conv2d(3, 16, 4, stride=4)
        self.text = torch.nn.Embedding(64, 16)
        self.state = torch.nn.Linear(4, 16)
        self.norm = torch.nn.LayerNorm(16)
        self.query = torch.nn.Linear(16, 16)
        self.key = torch.nn.Linear(16, 16)
        self.value = torch.nn.Linear(16, 16)
        self.action = torch.nn.Linear(16, 4)
        # Optional patterns register after the original modules, so the default model is unchanged.
        if gelu_mlp:
            self.context_up = torch.nn.Linear(WIDTH, 2 * WIDTH)
            self.context_down = torch.nn.Linear(2 * WIDTH, WIDTH)
        if time_embedding:
            self.time = torch.nn.Linear(WIDTH, WIDTH)
            half = WIDTH // 2
            exponents = torch.arange(half, dtype=torch.float32) * (-math.log(1000.0) / half)
            self.register_buffer("frequency_exponents", exponents)
        if queries > 1:
            # A parameter rather than an embedding of constant positions, which a frontend may fold away.
            self.query_offsets = torch.nn.Parameter(torch.zeros(1, queries, WIDTH))

    def _act(self, image, tokens, state, timestep):
        vision = self.vision(image).flatten(2).transpose(1, 2)
        context = self.norm(torch.cat((vision, self.text(tokens)), dim=1))
        if self.gelu_mlp:
            context = context + self.context_down(
                torch.nn.functional.gelu(self.context_up(context), approximate="tanh")
            )
        state_feature = self.state(state)
        if timestep is not None:
            angles = timestep.reshape(1, 1) * torch.exp(self.frequency_exponents)
            state_feature = state_feature + self.time(torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1))
        if self.queries == 1:
            query = self.query(state_feature.unsqueeze(1))
        else:
            query = self.query(state_feature.unsqueeze(1) + self.query_offsets)
        scores = (query @ self.key(context).transpose(-2, -1)) / 4
        attended = torch.softmax(scores, dim=-1) @ self.value(context)
        if self.queries == 1:
            return state + self.action(attended.squeeze(1))
        return state.unsqueeze(1) + self.action(attended)

    def forward(self, image, tokens, state):
        return self._act(image, tokens, state, None)


class TimedMultimodalPolicy(MultimodalPolicy):
    """The policy with an extra ``(1,)`` floating timestep input."""

    def forward(self, image, tokens, state, timestep):
        return self._act(image, tokens, state, timestep)


def validate_profile(profile):
    """The complete profile with defaults filled in, or ``ValueError`` for any unsupported field."""
    if (
        not isinstance(profile, dict)
        or not REQUIRED_KEYS <= set(profile)
        or set(profile) - REQUIRED_KEYS - OPTIONAL_KEYS
    ):
        raise ValueError("multimodal policy profile has an unsupported shape")
    if profile["schema"] != PROFILE_SCHEMA or profile["workload_id"] != "multimodal_policy":
        raise ValueError("multimodal policy profile has an unsupported identity")
    selected = {**DEFAULTS, **{key: value for key, value in profile.items() if key not in REQUIRED_KEYS}}
    for key, maximum in (("tokens", 32), ("queries", 16)):
        if type(selected[key]) is not int or not 1 <= selected[key] <= maximum:
            raise ValueError(f"multimodal policy profile has an invalid {key}")
    for key in ("gelu_mlp", "time_embedding"):
        if type(selected[key]) is not bool:
            raise ValueError(f"multimodal policy profile has an invalid {key}")
    return selected


def _selected_profile():
    path = Path(__file__).with_name("profile.json")
    if not path.exists():
        return dict(DEFAULTS), None
    if path.is_symlink() or not path.is_file():
        raise ValueError("multimodal policy profile must be an adjacent ordinary file")
    raw = path.read_bytes()
    return validate_profile(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def get_model_and_inputs():
    profile, profile_sha256 = _selected_profile()
    image = torch.arange(192, dtype=torch.float32).reshape(1, 3, 8, 8) / 192
    tokens = torch.arange(profile["tokens"], dtype=torch.int64).reshape(1, profile["tokens"])
    state = torch.linspace(-1, 1, 4).reshape(1, 4)
    timed = profile["time_embedding"]
    model = (TimedMultimodalPolicy if timed else MultimodalPolicy)(
        profile["queries"], profile["gelu_mlp"], timed
    ).eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
        if profile["queries"] > 1:
            values = torch.arange(model.query_offsets.numel(), dtype=torch.int64).reshape(model.query_offsets.shape)
            model.query_offsets.copy_((((values * 29 + 11) % 61) - 30).to(torch.float32) / 64)
    if profile_sha256 is not None:
        model.session_provenance = {
            "workload_id": "multimodal_policy",
            "workload_role": "iteration",
            "input_source": "seeded_synthetic",
            "profile_sha256": profile_sha256,
        }
    if timed:
        return model, (image, tokens, state, torch.tensor([0.625], dtype=torch.float32))
    return model, (image, tokens, state)
