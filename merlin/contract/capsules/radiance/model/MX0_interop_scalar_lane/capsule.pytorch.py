"""A small but COMPLETE LLaMA-style transformer for fast end-to-end RVV/spike
verification. Same op surface as a real LLaMA (RMSNorm, RoPE, causal attention +
softmax, SwiGLU MLP, tied-free lm_head) at tiny dims so a spike functional-sim run
finishes in seconds. The Merlin pipeline that runs this is identical to the one that
runs tiny_llama-1.1B / smolVLA; only the dims differ."""
from __future__ import annotations
import math
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.w


def rope(x, pos, dim):
    # x: [B,H,T,D]
    half = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos[:, None].float() * freq[None, :]           # [T, half]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
    def forward(self, x, pos):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        q = rope(q, pos, self.dh); k = rope(k, pos, self.dh)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.full((T, T), float("-inf")).triu(1)
        att = (att + mask).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.g = nn.Linear(d, hidden, bias=False); self.u = nn.Linear(d, hidden, bias=False)
        self.dn = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.dn(torch.nn.functional.silu(self.g(x)) * self.u(x))


class Block(nn.Module):
    def __init__(self, d, h, hidden):
        super().__init__()
        self.n1 = RMSNorm(d); self.attn = Attn(d, h)
        self.n2 = RMSNorm(d); self.mlp = MLP(d, hidden)
    def forward(self, x, pos):
        x = x + self.attn(self.n1(x), pos)
        return x + self.mlp(self.n2(x))


class SmallLlama(nn.Module):
    def __init__(self, vocab=256, d=128, h=4, layers=2, hidden=344):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, hidden) for _ in range(layers)])
        self.norm = RMSNorm(d); self.lm = nn.Linear(d, vocab, bias=False)
    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T)
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, pos)
        return self.lm(self.norm(x))


def get_model_and_inputs():
    m = SmallLlama().eval()
    ids = torch.randint(0, 256, (1, 8), dtype=torch.long)
    return m, (ids,)
