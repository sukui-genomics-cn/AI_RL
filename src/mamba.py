import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    # dt 是 discrete_time_step 的实例
    dt = dt + dt_bias if dt_bias is not None else dt
    if dt_softplus:
        dt = torch.nn.functional.softplus(dt)

    # 离散化 A 和 B
    A_bar = torch.matrix_exp(dt * A)  # discrete_time_step 的作用
    B_bar = (torch.linalg.solve(A, A_bar - torch.eye(A.shape[-1])) @ B)

    # 状态更新
    state = A_bar @ state + B_bar @ x
    out = C @ state

    # 跳跃连接（如果有）
    if D is not None and z is not None:
        out = out + D * z

    return out

def selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    assert D.shape == (nheads, dim)

    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "d h d -> d h d 1")*A) # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
    state.copy_(state*dA + dB*rearrange(x, "b h d -> b h d 1")) # (batch, dim, dstate
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out