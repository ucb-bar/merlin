"""RDT (RoboticsDiffusionTransformer) loop-preserving capture wrapper (P21).

THE HARDEST family. Unlike smolVLA / pi0.5 (flow-matching: x_t = x_t + dt*v_t, a
single-step Euler with NO solver state), RDT's few-step denoise uses a
``DPMSolverMultistepScheduler`` -- a *multistep* DPM-solver that carries solver
state (the previous converted model output) across iterations and picks the solver
order per step. The naive ``while_loop`` around the bare ``model.forward`` failed
two ways (see STATUS.md HAZARD): (1) aliasing -- the body returned loop-invariant
conditioning; (2) shape mismatch -- the bare model's input is hidden-projected
(65x2048) but its output is action-space (64x128). The REAL loop carries the
ACTION LATENT (action-space) and RE-EMBEDS it each step + applies the DPM-solver
step. That re-embed + scheduler lives in the upstream SAMPLER
(``RDTRunner.conditional_sample``), not the bare ``model.forward``.

Faithful to ``RDTRunner.conditional_sample`` (models/rdt_runner.py:119-162):
    noisy_action = noise                                          # (b, H, A) action-space
    for t in scheduler.timesteps:                                 # K = num_inference_timesteps = 5
        action_traj       = cat([noisy_action, action_mask], -1)  # RE-EMBED: action-space -> hidden
        action_traj       = state_adaptor(action_traj)
        state_action_traj = cat([state_traj, action_traj], 1)     # prepend the state token
        model_output      = model(state_action_traj, ctrl_freqs, t, lang_cond, img_cond, lang_mask)
        noisy_action      = scheduler.step(model_output, t, noisy_action).prev_sample   # DPM update
    return noisy_action * action_mask

The carried iter_args are therefore (i, noisy_action, m_prev) -- ALL shape-invariant
in action-space (b, H, A). The action latent round-trips action-space -> hidden ->
model -> action-space every step, so its shape is invariant (this fixes the
65x2048-vs-64x128 hazard).

-- How the DPM-solver multistep state is handled (the crux of this family) --
With RDT's config (algorithm_type=dpmsolver++, solver_type=midpoint, solver_order=2,
prediction_type=sample, final_sigmas_type=zero, K=5<15) the per-step solver order is
STATICALLY known: [1st, 2nd, 2nd, 2nd, 1st]
  step 0: lower_order_nums<1            -> first-order  (D1 unused)
  steps 1-3: second-order (midpoint)
  step 4: lower_order_final (sigma=0)  -> first-order  (D1 unused)
The second-order midpoint update is the first-order update PLUS a D1 correction
term  ( -0.5*alpha_t*(exp(-h)-1)*D1 , D1 = (1/r0)*(m0-m1) ). So we compute the full
second-order formula EVERY step and gate the D1 term by a per-step scalar
use_2nd[i] = [0,1,1,1,0] -> exact, single straight-line body (no data-dependent
Python branch inside the while_loop). `m_prev` (m1, the previous converted model
output) is carried so D1 is available; on first-order steps its contribution is
zeroed. prediction_type=sample => convert_model_output is the identity, so the
"converted model output" carried IS the raw model output.

-- RULE 4 (hoist every in-body tensor constant) --
The whole DPM coefficient schedule (sigma_t, sigma_s0, sigma_s1, alpha_t, the
exp(-h)-1 factor, the 1/r0 ratio, use_2nd) depends ONLY on the step index, so it is
precomputed OUTSIDE the loop as length-K tensors and indexed by the carried counter
i (via torch.index_select with i -- a tensor, no Python int). The diffusion timestep
t fed to the model likewise comes from a precomputed timesteps table indexed by i.
No torch.tensor([...]) / linspace / arange survives inside the body.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

# diffusers 0.27.2 imports a symbol removed from modern huggingface_hub; shim it.
import huggingface_hub as _hh  # noqa: E402

if not hasattr(_hh, "cached_download"):
    _hh.cached_download = _hh.hf_hub_download

from diffusers.schedulers.scheduling_dpmsolver_multistep import (  # noqa: E402
    DPMSolverMultistepScheduler,
)
from torch._higher_order_ops.while_loop import while_loop  # noqa: E402

_RDT_REPO = "/scratch/agustin/projects/RoboticsDiffusionTransformer"
if _RDT_REPO not in sys.path:
    sys.path.insert(0, _RDT_REPO)
sys.path.insert(0, "/scratch/agustin/projects/model2MLIR/workloads/rdt")

from loader import get_model_and_inputs  # noqa: E402

K = 5  # num_inference_timesteps (configs/base.yaml) -> the IR loop constant


def _build_scheduler():
    s = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
    )
    s.set_timesteps(K)
    return s


def _solver_tables(scheduler):
    """Precompute, for each of the K steps, every DPM-solver++ midpoint coefficient
    so the loop body is constant-free. Returns length-K Python float lists, indexed
    in-body by the carried counter i. Mirrors dpm_solver_first_order_update /
    multistep_dpm_solver_second_order_update with algorithm_type=dpmsolver++,
    solver_type=midpoint, solver_order=2, final_sigmas_type=zero, len(timesteps)=K<15.

    Computed in PURE PYTHON math (concrete scheduler constants) so NOTHING here is
    traced -- the tables are built once in __init__ and become module buffers (closed
    over), never in-body ops (RULE 4).
    """
    import math as _m

    sig = [float(x) for x in scheduler.sigmas.tolist()]  # length K+1 (last = 0.0)

    def a_s(sigma):
        alpha = 1.0 / _m.sqrt(sigma * sigma + 1.0)
        return alpha, sigma * alpha

    c_sigma_ratio, c_D0, c_D1_half, c_invr0, use_2nd = [], [], [], [], []
    for k in range(K):
        sigma_t_raw, sigma_s0_raw = sig[k + 1], sig[k]
        sigma_s1_raw = sig[k - 1] if k >= 1 else sig[k]  # unused on step 0 (gated off)
        alpha_t, sigma_t = a_s(sigma_t_raw)
        alpha_s0, sigma_s0 = a_s(sigma_s0_raw)
        alpha_s1, sigma_s1 = a_s(sigma_s1_raw)

        def _lam(alpha, sigma):
            # lambda = log(alpha) - log(sigma); diffusers uses torch.log so sigma==0
            # -> log(0) = -inf -> lambda = +inf (the final_sigmas_type='zero' step).
            return _m.inf if sigma == 0.0 else (_m.log(alpha) - _m.log(sigma))

        lambda_t = _lam(alpha_t, sigma_t)
        lambda_s0 = _lam(alpha_s0, sigma_s0)
        lambda_s1 = _lam(alpha_s1, sigma_s1)
        h = lambda_t - lambda_s0
        h0 = lambda_s0 - lambda_s1
        # exp(-h): h=+inf (sigma_t==0) -> exp(-inf)=0 -> expm1=-1 (the limiting value).
        expm1 = (_m.exp(-h) - 1.0) if _m.isfinite(h) else (-1.0 if h > 0 else _m.inf)
        # order schedule (K=5<15, final sigma==0): first, 2nd, 2nd, 2nd, first
        lower_order_final = k == (K - 1)
        first_order = (k == 0) or lower_order_final  # lower_order_nums<1 on step0
        # 1/r0 = h/h_0; on first-order steps the D1 term is gated off (use_2nd=0) and
        # h_0 may be 0 (last step, sigma=0) -> set invr0=0 so 0*inf never appears.
        invr0 = 0.0 if (first_order or h0 == 0.0) else (h / h0)
        c_sigma_ratio.append(sigma_t / sigma_s0)
        c_D0.append(-(alpha_t * expm1))
        c_D1_half.append(-0.5 * (alpha_t * expm1))
        c_invr0.append(invr0)
        use_2nd.append(0.0 if first_order else 1.0)

    return {
        "sigma_ratio": c_sigma_ratio,
        "D0": c_D0,
        "D1_half": c_D1_half,
        "invr0": c_invr0,
        "use_2nd": use_2nd,
        "timesteps": [float(x) for x in scheduler.timesteps.tolist()],
    }


class RDTWhileLoopSampler(nn.Module):
    """``RDTRunner.conditional_sample`` as a single ``torch.while_loop`` over K DPM steps.

    Conditioning (lang_cond, img_cond, state_traj, action_mask, ctrl_freqs,
    lang_mask) + the state_adaptor weights are closed over (invariant
    additional_inputs). Carried iter_args = (i, noisy_action, m_prev), all
    action-space (b, H, A) and shape-invariant.
    """

    def __init__(self, model: nn.Module, state_adaptor: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.state_adaptor = state_adaptor
        self.K = K
        # Precompute the full DPM-solver++ midpoint coefficient schedule ONCE, as
        # concrete float32 buffers (closed over -> never in-body constants). RULE 4.
        tb = _solver_tables(_build_scheduler())
        for name, lst in tb.items():
            self.register_buffer(f"_tb_{name}",
                                 torch.tensor(lst, dtype=torch.float32),
                                 persistent=False)

    def forward(self, noise, ctrl_freqs, state_traj, action_mask, lang_cond, img_cond, lang_mask):
        m = self.model
        sa = self.state_adaptor
        dtype = noise.dtype

        # --- DPM-solver coefficient schedule + timesteps: closed-over buffers ---
        sigma_ratio = self._tb_sigma_ratio
        cD0 = self._tb_D0
        cD1h = self._tb_D1_half
        cinvr0 = self._tb_invr0
        use2 = self._tb_use_2nd
        ts = self._tb_timesteps
        K_local = self.K

        # action_mask broadcast over the horizon, computed once (invariant).
        action_mask_H = action_mask.expand(-1, noise.shape[1], -1)  # (b, H, A)

        def gather(table, i):
            # index a length-K constant table by the carried counter i (a tensor)
            return torch.index_select(table, 0, i.reshape(1)).reshape(())

        def cond_fn(i, x_t, m_prev):
            return i < K_local

        def body_fn(i, x_t, m_prev):
            # ---- RE-EMBED: action-space latent -> hidden trajectory ----
            action_traj = torch.cat([x_t, action_mask_H], dim=2)     # (b, H, A+A) -> state_adaptor in
            action_traj = sa(action_traj)                            # (b, H, hidden)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)  # (b, H+1, hidden)

            # ---- diffusion timestep for this step, from the precomputed table ----
            t = gather(ts, i).reshape(1)                             # (1,)

            # ---- denoise transformer (the captured heavy graph) ----
            model_output = m(
                state_action_traj, ctrl_freqs, t,
                lang_cond, img_cond, lang_mask,
            )  # (b, H, A) -- prediction_type=sample => already the converted model output
            # the model returns a SLICE of the (b, H+3, A) sequence (model.py:164
            # `x = x[:, -horizon:]`) -> non-standard stride (8576 not 8192). Force a
            # true standard-contiguous copy so the carried m_prev layout is invariant
            # across iterations (while_loop meta-consistency requires matching strides).
            model_output = model_output.clone(memory_format=torch.contiguous_format)

            # ---- DPM-solver++ midpoint step (constant-free; per-step coeffs gathered) ----
            sr = gather(sigma_ratio, i)
            d0c = gather(cD0, i)
            d1hc = gather(cD1h, i)
            invr0 = gather(cinvr0, i)
            u2 = gather(use2, i)
            sample32 = x_t.to(torch.float32)
            D0 = model_output
            D1 = invr0 * (model_output - m_prev)                     # (1/r0)*(m0-m1)
            prev_sample = sr * sample32 + d0c * D0 + u2 * (d1hc * D1)
            x_next = prev_sample.to(dtype)

            return (i + 1, x_next, model_output)

        i0 = torch.zeros((), dtype=torch.int64, device=noise.device)
        m0 = torch.zeros_like(noise)  # m_prev seed; unused on step 0 (use_2nd[0]=0)
        _, x_final, _ = while_loop(cond_fn, body_fn, (i0, noise, m0))

        return x_final * action_mask_H


# ----------------------------------------------------------------------------
# Building / reference
# ----------------------------------------------------------------------------

def build():
    """Construct the wrapper, a state_adaptor, and matching inputs.

    The m2m loader gives the bare RDT denoise step with hidden-size conditioning.
    The sampler additionally needs the state_adaptor (action-space+mask -> hidden)
    and action-space inputs. We build a small state_adaptor (linear, the base.yaml
    'linear' adaptor) and action-space noise/state/mask consistent with the loader's
    dims (hidden=2048, horizon=64, action_dim=128).
    """
    step_mod, _ = get_model_and_inputs()
    model = step_mod.model

    # The loader zero-inits the model's final layer (model.py:120-121) for a clean
    # smoke trace; that makes every model_output 0 and the denoise loop trivially
    # all-zeros (cosine undefined). Randomize the final layer so the numeric check
    # against the real DPM-solver loop is meaningful. (Capture topology is unaffected.)
    torch.manual_seed(1234)
    with torch.no_grad():
        fl = model.final_layer
        fl.ffn_final.fc2.weight.normal_(0.0, 0.02)
        fl.ffn_final.fc2.bias.normal_(0.0, 0.02)

    hidden_size = 2048
    horizon = 64
    action_dim = 128
    b = 1
    lang_len = 32
    img_cond_len = 4096

    # state_adaptor: in = action_dim*2 (action + mask indicator) -> hidden_size (base.yaml: linear)
    state_adaptor = nn.Linear(action_dim * 2, hidden_size).eval()

    torch.manual_seed(0)
    noise = torch.randn(b, horizon, action_dim)
    ctrl_freqs = torch.tensor([25.0] * b)
    state_traj = torch.randn(b, 1, hidden_size)          # already adapted state token
    action_mask = torch.ones(b, 1, action_dim)           # 0-1 float mask
    lang_cond = torch.randn(b, lang_len, hidden_size)
    img_cond = torch.randn(b, img_cond_len, hidden_size)
    lang_mask = torch.ones(b, lang_len, dtype=torch.bool)

    inputs = (noise, ctrl_freqs, state_traj, action_mask, lang_cond, img_cond, lang_mask)
    return RDTWhileLoopSampler(model, state_adaptor).eval(), inputs


def ref_unrolled(wrapper, inputs):
    """Reference: the REAL RDTRunner.conditional_sample loop, unrolled eagerly with
    the actual stateful DPMSolverMultistepScheduler."""
    (noise, ctrl_freqs, state_traj, action_mask, lang_cond, img_cond, lang_mask) = inputs
    m = wrapper.model
    sa = wrapper.state_adaptor
    sched = _build_scheduler()
    noisy_action = noise
    am_H = action_mask.expand(-1, noise.shape[1], -1)
    for t in sched.timesteps:
        action_traj = torch.cat([noisy_action, am_H], dim=2)
        action_traj = sa(action_traj)
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)
        model_output = m(
            state_action_traj, ctrl_freqs, t.unsqueeze(-1).to(noise.dtype),
            lang_cond, img_cond, lang_mask,
        )
        noisy_action = sched.step(model_output, t, noisy_action).prev_sample
        noisy_action = noisy_action.to(noise.dtype)
    return noisy_action * am_H


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs = build()
    with torch.no_grad():
        out = wrapper(*inputs)
        ref = ref_unrolled(wrapper, inputs)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    maxabs = (out - ref).abs().max().item()
    print("out shape:", tuple(out.shape), "| cos vs real DPM-solver loop:", f"{cos:.7f}",
          "| maxabs:", f"{maxabs:.3e}")

    print("\n=== torch.export.export(strict=False) ===")
    ep = torch.export.export(wrapper, inputs, strict=False)
    gm = ep.graph_module
    wl = [n for n in gm.graph.nodes
          if n.op == "call_function" and "while_loop" in str(n.target)]
    print("export OK; while_loop call_function nodes:", len(wl))

    print("\n=== m2m.convert(backend='fx_importer', level='linalg-on-tensors') ===")
    import m2m
    res = m2m.convert(wrapper, inputs, backend="fx_importer",
                      quantization=None, level="linalg-on-tensors")
    s = res.mlir_text or ""
    print("path_taken:", res.path_taken, "| MLIR length:", len(s))
    print("  scf.for:", s.count("scf.for"), "| scf.while:", s.count("scf.while"))
    if res.diagnostics:
        print("  diag[-1]:", str(res.diagnostics[-1]).splitlines()[0][:240])
    if s and "scf.for" in s:
        out_path = ("/scratch/agustin/projects/oscar-merlin/merlin/benchmarks/"
                    "dse_guidance/recaptures_loop/rdt/model.mlir")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(s)
        print("  wrote", out_path)
