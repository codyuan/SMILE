import torch
import torch.nn as nn
import math

from util.functions import exists,extract,default


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim))
        self.b = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class LinearDenoiser(nn.Module):

    def __init__(self, dim, dim_out, *, activate=None, time_emb_dim=None, norm=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Tanh(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Linear(dim, dim_out),
            activate() if activate else nn.Identity(),
        )

    def forward(self, x, time_emb=None):
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            x = x + condition

        res = self.net(x)

        return res


class Denoiser(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            in_dim=None,
            hidden_dim=None,
            out_dim=None,
            with_time_emb=True
    ):
        super().__init__()

        self.state_dim=state_dim
        self.action_dim=action_dim

        in_dim = default(in_dim, state_dim + action_dim)
        hidden_dim = default(hidden_dim, 128)
        out_dim = default(out_dim, action_dim)
        dims = [in_dim, hidden_dim, hidden_dim, out_dim]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = in_dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(in_dim),
                nn.Linear(in_dim - 1 if in_dim % 2 != 0 else in_dim, in_dim * 4),
                nn.Tanh(),
                nn.Linear(in_dim * 4, in_dim)
            )
        else:
            time_dim = 0
            self.time_mlp = None

        self.denoiser = nn.ModuleList([])
        num_layers = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_layers - 1)

            self.denoiser.append(
                LinearDenoiser(dim_in, dim_out, activate=nn.Tanh if not is_last else None, time_emb_dim=time_dim,
                               norm=ind != 0),
            )

    def forward(self, s, a_t, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        x = torch.cat([s, a_t], dim=-1).float()

        for linear in self.denoiser:
            x = linear(x, t)
        return x