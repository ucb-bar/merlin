"""SmolVLA (LeRobot, SO100 policy) with every Linear and the vision patch Conv2d on the emulated MXU.

Host operations keep the model's native Torch arithmetic. The observation NPZ holds images, image_masks,
language_tokens, language_mask, state, and explicit noise, so sampling is deterministic. --reference is a
default-recipe trajectory NPZ whose final_actions must match byte for byte.
"""

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from . import arithmetic


def projection(linear, name, module, x):
    if isinstance(module, torch.nn.Conv2d):
        # The only convolution is the unpadded, nonoverlapping patch embedding.
        batch, _, height, width = x.shape
        patches = F.unfold(x, module.kernel_size, stride=module.stride)
        matrix = patches.transpose(1, 2).reshape(-1, module.weight[0].numel())
        shape = (batch, height // module.kernel_size[0], width // module.kernel_size[1], module.out_channels)
        out = linear(name, matrix, lambda: module.weight.flatten(1).t()).to(x.dtype).reshape(shape)
    else:
        out = linear(name, x.reshape(-1, module.in_features), lambda: module.weight.t())
        out = out.to(x.dtype).reshape(*x.shape[:-1], module.out_features)
    if module.bias is not None:
        out = out + module.bias
    return out.permute(0, 3, 1, 2).contiguous() if isinstance(module, torch.nn.Conv2d) else out


def run(args):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(str(args.checkpoint), local_files_only=True)
    config.device, config.vlm_model_name = "cpu", str(args.backbone)
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint), config=config, local_files_only=True,
                                           strict=True).eval()
    with np.load(args.observation) as archive:
        inputs = {name: torch.from_numpy(archive[name]) for name in archive.files}

    def sample():
        images, masks = inputs["images"], inputs["image_masks"]
        actions = policy.model.sample_actions(
            [images[:, i] for i in range(images.shape[1])], [masks[:, i] for i in range(masks.shape[1])],
            inputs["language_tokens"], inputs["language_mask"], inputs["state"], noise=inputs["noise"])
        return actions[..., :config.action_feature.shape[0]].contiguous().numpy()

    native = sample()
    linear = arithmetic.Linear(args)
    for name, module in policy.model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            module.register_forward_hook(lambda module, arguments, _, name=name: projection(
                linear, name, module, arguments[0]))
    actions = sample()
    error = actions.astype(np.float64) - native.astype(np.float64)
    report = dict(model="smol", action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                  relative_l2=float(np.linalg.norm(error) / np.linalg.norm(native.astype(np.float64))),
                  max_abs=float(np.abs(error).max()))
    if args.reference:
        with np.load(args.reference) as reference:
            report["reference_exact"] = reference["final_actions"].tobytes() == actions.tobytes()
    return report
