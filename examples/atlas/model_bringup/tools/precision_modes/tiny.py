"""TinyLlama with every projection on the emulated MXU and the Atlas host arithmetic for everything else.

The PR #26 arithmetic capture supplies the weight panels, rotary tables, and default prompt. --inputs takes
{"prompts": [{"id": ..., "token_ids": [...]}]}; nothing is tokenized or downloaded. Native FP32 Transformers
logits give the free-greedy and forced-history comparisons.
"""

import hashlib
import json

import torch
from safetensors import safe_open

from . import arithmetic

SITES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
         "mlp.up_proj", "mlp.down_proj")


def bf(x):
    return x.to(torch.bfloat16).float()


def fma(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def norm(x, weight):
    """RMSNorm with the host's blocked FP32 sum of squares over 2048 lanes."""
    squares = (x * x).reshape(x.shape[0], 64, 4, 8)
    partial = torch.zeros((x.shape[0], 4, 8))
    for block in range(4):
        local = torch.zeros_like(partial)
        for i in range(16):
            local = local + squares[:, block * 16 + i]
        partial = partial + local
    lanes = partial[:, 0]
    for group in range(1, 4):
        lanes = lanes + partial[:, group]
    total = torch.zeros(x.shape[0])
    for lane in range(8):
        total = total + lanes[:, lane]
    return bf(bf(x * (1.0 / torch.sqrt(total / 2048.0 + 1e-5))[:, None]) * weight)


def rotary(x, cosine, sine):
    half = x.shape[-1] // 2
    return bf(bf(x * cosine) + bf(torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sine))


def exp_nonpositive(x):
    """Host exp for x <= 0: two-constant ln 2 reduction, an FP32 FMA polynomial, and two ldexp steps."""
    safe = torch.clamp(x, min=-104.0)
    q = torch.round(safe * torch.tensor(1.4426950408889634, dtype=torch.float32))
    s = fma(q, torch.tensor(-0.693145751953125), safe)
    s = fma(q, torch.tensor(-1.428606765330187045e-06), s)
    u = torch.full_like(s, 0.000198527617612853646278381)
    for constant in (0.00139304355252534151077271, 0.00833336077630519866943359, 0.0416664853692054748535156,
                     0.166666671633720397949219, 0.5):
        u = fma(u, s, torch.tensor(constant, dtype=torch.float32))
    u = 1.0 + fma(s * s, u, s)
    qi = q.to(torch.int32)
    half = torch.div(qi, 2, rounding_mode="floor")
    return torch.where(x < -104.0, 0.0, torch.ldexp(torch.ldexp(u, half), qi - half))


def softmax(x):
    ex = exp_nonpositive(x - x.amax(dim=-1, keepdim=True))
    lanes = torch.zeros((*x.shape[:-1], 16))
    for i in range(0, x.shape[-1], 16):
        size = min(16, x.shape[-1] - i)
        lanes[..., :size] = lanes[..., :size] + ex[..., i:i + size]
    for delta in (8, 4, 2, 1):
        lanes[..., :delta] = lanes[..., :delta] + lanes[..., delta:2 * delta]
    return bf(ex * (1.0 / lanes[..., :1]))


def pair_bmm(a, b, transposed, decode):
    """Host attention matmul in its FP32 FMA order; BF16 inputs are exact in FP32."""
    heads, rows, reduction = a.shape
    right = b.transpose(-1, -2) if transposed else b
    columns = right.shape[-1]
    if decode and transposed and reduction == 64:
        lanes = torch.zeros((heads, columns, 16))
        for start in range(0, 64, 16):
            lanes = fma(a[:, 0, start:start + 16, None].transpose(-1, -2), b[:, :, start:start + 16], lanes)
        for distance in (8, 4, 2, 1):
            lanes = lanes + lanes[..., torch.arange(16) ^ distance]
        return bf(lanes[..., 0].unsqueeze(1))
    total = torch.zeros((heads, rows, columns))
    if decode and not transposed and columns == 64:
        start = 0
        for block in (8, 4, 2, 1):
            while reduction - start >= block:
                partial = torch.zeros_like(total)
                for pair in range(start, start + block, 2):
                    if pair + 1 < start + block:
                        partial = fma(a[:, :, pair + 1, None], right[:, None, pair + 1, :], partial)
                    partial = fma(a[:, :, pair, None], right[:, None, pair, :], partial)
                total = total + partial
                start += block
    else:
        for pair in range(0, reduction, 2):
            if pair + 1 < reduction:
                total = fma(a[:, :, pair + 1, None], right[:, None, pair + 1, :], total)
            total = fma(a[:, :, pair, None], right[:, None, pair, :], total)
    return bf(total)


def attention(q, k, v, cosine, sine, past):
    """Grouped-query attention, 32 query and 4 KV heads of 64, with a KV cache."""
    rows = q.shape[0]
    offset = 0 if past is None else past[0].shape[1]
    cosine, sine = cosine[offset:offset + rows], sine[offset:offset + rows]
    q = rotary(q.reshape(rows, 32, 64).transpose(0, 1), cosine, sine)
    k = rotary(k.reshape(rows, 4, 64).transpose(0, 1), cosine, sine)
    v = v.reshape(rows, 4, 64).transpose(0, 1)
    if past is not None:
        k, v = torch.cat((past[0], k), dim=1), torch.cat((past[1], v), dim=1)
    scores = bf(pair_bmm(q, k.repeat_interleave(8, dim=0), True, rows == 1) * 0.125)
    causal = torch.arange(k.shape[1])[None, :] > torch.arange(offset, offset + rows)[:, None]
    scores = bf(scores + torch.where(causal, torch.tensor(-3.3895313892515355e38), torch.tensor(0.0)))
    attended = pair_bmm(softmax(scores), v.repeat_interleave(8, dim=0), False, rows == 1)
    return attended.transpose(0, 1).contiguous().reshape(rows, 2048), (k, v)


def silu_mul(gate, up):
    # exp and division run in FP32, and SiLU rounds before the MLP multiply.
    return bf(bf(gate / (1.0 + torch.exp(-gate))) * up)


def sha(logits):
    return hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()


class Engine:
    def __init__(self, args):
        self.receipt = json.loads((args.capture / "receipt.json").read_text())
        self.panels = {call["site"]: call["panels"] for call in self.receipt["calls"] if call["step"] == 0}
        tables = [self.receipt["host_parameters"][key] for key in ("rotary_cosine", "rotary_sine")]
        self.cos, self.sin = (torch.frombuffer(bytearray((args.capture / table["file"]).read_bytes()), dtype=torch.int16)
                              .reshape(table["shape"]).view(torch.bfloat16).float() for table in tables)
        self.weights = safe_open(str(args.checkpoint / "model.safetensors"), framework="pt", device="cpu")
        self.layers = json.loads((args.checkpoint / "config.json").read_text())["num_hidden_layers"]
        self.embedding = self.weights.get_tensor("model.embed_tokens.weight")
        self.linear = arithmetic.Linear(args)

    def weight(self, site):
        return self.weights.get_tensor(site + ".weight").float()

    def project(self, x, site):
        """One contraction per prefill weight panel, each with its own tensor scale."""
        return torch.cat([self.linear(site, x, lambda: self.weight(site)[lo:hi].T, key=(site, lo))
                          for lo, hi in (panel["columns"] for panel in self.panels[site])], dim=1)

    def forward(self, ids, cache):
        x = self.embedding[torch.tensor(ids)].float()
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            n = norm(x, self.weight(prefix + "input_layernorm"))
            q, k, v = (self.project(n, prefix + site) for site in SITES[:3])
            attended, cache[layer] = attention(q, k, v, self.cos, self.sin, cache[layer])
            residual = bf(x + self.project(attended, prefix + SITES[3]))
            n = norm(residual, self.weight(prefix + "post_attention_layernorm"))
            gate, up = (self.project(n, prefix + site) for site in SITES[4:6])
            x = bf(residual + self.project(silu_mul(gate, up), prefix + SITES[6]))
        return self.project(norm(x, self.weight("model.norm")), "lm_head")[-1], cache


def native(model, ids, steps):
    """Greedy native FP32 decode: tokens and last-position logits per step."""
    past, tokens, logits = None, [], []
    for _ in range(steps):
        output = model(input_ids=torch.tensor([tokens[-1:] or ids]), past_key_values=past, use_cache=True)
        past = output.past_key_values
        logits.append(output.logits[0, -1].float())
        tokens.append(int(logits[-1].argmax()))
    return tokens, logits


def run(args):
    from transformers import LlamaForCausalLM

    engine = Engine(args)
    prompts = (json.loads(args.inputs.read_text())["prompts"] if args.inputs else
               [{"id": "capture", "token_ids": engine.receipt["target_trace"][0]["input_tokens"][0]}])
    model = LlamaForCausalLM.from_pretrained(str(args.checkpoint), torch_dtype=torch.float32,
                                             attn_implementation="eager", local_files_only=True).eval()
    references = [native(model, prompt["token_ids"], args.steps) for prompt in prompts]
    del model
    report = dict(model="tiny", prompts=[])
    for prompt, (native_tokens, native_logits) in zip(prompts, references):
        tokens, steps, free = [], [], [None] * engine.layers
        for step in range(args.steps):
            logits, free = engine.forward(tokens[-1:] or prompt["token_ids"], list(free))
            if tokens == native_tokens[:step]:
                forced_logits, forced = logits, free
            else:
                forced_logits, forced = engine.forward([native_tokens[step - 1]], list(forced))
            tokens.append(int(logits.argmax()))
            error = (forced_logits.double() - native_logits[step].double()).square().mean().sqrt()
            steps.append(dict(token=tokens[-1], logits_sha256=sha(logits), forced_token=int(forced_logits.argmax()),
                              forced_rmse=float(error)))
        report["prompts"].append(dict(id=prompt["id"], tokens=tokens, native_tokens=native_tokens,
                                      forced_top1=sum(s["forced_token"] == t for s, t in zip(steps, native_tokens)),
                                      steps=steps))
    if not (args.inputs or args.finer_scales or args.k_split or args.two_term):
        report["capture_exact"] = all(
            step["logits_sha256"] == target["logits"]["sha256"] and step["token"] == target["next_token"]
            for step, target in zip(report["prompts"][0]["steps"], engine.receipt["target_trace"]))
    return report
