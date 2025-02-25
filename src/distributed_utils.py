from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 4 lines are for backward compatibility with
# older PyTorch.

if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base


# Raw operation, does not support autograd, but does support async
def all_gather_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    output = torch.empty(
        world_size * input_.shape[0], *input_.shape[1:], dtype=input_.dtype, device=input_.device
    )
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Raw operation, does not support autograd, but does support async
def reduce_scatter_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(
        input_.shape[0] // world_size, *input_.shape[1:], dtype=input_.dtype, device=input_.device
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Raw operation, does not support autograd, but does support async
def all_reduce_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(input_, group=process_group, async_op=async_op)
    return input_, handle


class AllGatherFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""
    @staticmethod
    def forward(ctx, input_:Tensor, process_group:ProcessGroup) -> Tensor:
        """
        自定义前向和反向传播操作
        :param ctx: 上下文对象, 用于在前向传播中保存信息共反向传播使用
        :param input_: 输入张量, 通常是在多个进程中分布的数据
        :param process_group: 分布式进程组, 定义了参与通信的GPU或设备集合
        :return:
        """
        ctx.process_group = process_group
        # 每个进程提供一个输入张量, 所有进程的输入张量先进行归约, 然后将结果分割成等份, 分布散布到每个进程
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        # Reduce-Scatter的"逆操作", 因为前向传播将数据分散, 反向传播需要将梯度重新聚合.
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return grad_input, None

# Supports autograd, but does not support async
all_gather = AllGatherFunc.apply


class ReduceScatterFunc(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = reduce_scatter_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group)
        return grad_input, None

# Supports autograd, but does not support async
reduce_scatter = ReduceScatterFunc.apply


class AllReduceFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_reduce_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None

# Supports autograd, but does not support async
all_reduce = AllReduceFunc.apply

def sync_shared_params(model:torch.nn.Module, process_group:ProcessGroup):
    # We want to iterate over parameters with _share_params=True in the same order,
    # as different ranks might have different number of parameters (e.g, only rank o has bias)
    pamams_shared = {
        name:p for name, p in model.named_parameters() if getattr(p, "_shared_params", False)
    }
    for _, p in sorted(pamams_shared.items()):
        with torch.no_grad():
            # Broadcast needs src to be global rank, not group rank
            torch.distributed.broadcast(
                p, src=torch.distributed.get_global_rank(process_group, 0), group=process_group
            )

# Ref: https://github.com/NVIDIA/Megatron-LM/blob/52e636888cccc41e931251c417a7181fc36de926/megatron/optimizer/optimizer.py#L256
def allreduce_sequence_parallel_grad(model:torch.nn.Module, process_group:ProcessGroup):
    # We want to iterate over parameters with _sequence_parallel=True in the same order,
    # as different ranks might have different number of parameters (e.g., only rank 0 has bias).
    params_seqparallel = {
        name: p for name, p in model.named_parameters() if getattr(p, "_sequence_parallel", False)
    }
    grads = [p.grad for _, p in sorted(params_seqparallel.items())]
    if grads:
        with torch.no_grad():
            coalesced = torch._utils._flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=process_group)
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

def get_dim_for_local_rank(dim:int, world_size:int, local_rank:int, multiple_of:int = 1) -> int:
    """
    Get the dim for the local rank derived from splitting dim on world_size process.
    the split may not be even across the world_size process.
    :param dim:
    :param world_size:
    :param local_rank:
    :param multiple_of:
    :return:
    """
    multiple = dim // multiple_of
    div = multiple // world_size
    mod = multiple % world_size
    local_multiple = div + int(local_rank < mod)
    return local_multiple * multiple_of