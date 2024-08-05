import triton
import torch
import triton.language as tl
import numpy as np

def triton_quantize_and_pack_along_last_dim_dummy(data: torch.Tensor, group_size: int, bit: int):
    assert len(data.shape) == 4
    shape = data.shape
    B, nh, D, T = shape
    # B : batch size, nh : number of heads
    # D : key 의 경우 head dim, value 의 경우에는 seq len
    # T : key 의 경우 key len, value 의 경우에는 head dim
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B * nh * D, num_groups, group_size)
    scale_mn_shape = B, nh, D, num_groups
    # 양자화 실시
    data = data.reshape(new_shape)
    mx = torch.empty(scale_mn_shape, device=data.device, dtype=data.dtype)
    mn = torch.empty(scale_mn_shape, device=data.device, dtype=data.dtype)
    BLOCK_SIZE_N = 128
    grid = lambda x: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
    _minmax_along_last_dim[grid](
        data,
        mn, mx,
        data.numel(),
        data.shape[0], # B * nh * D
        num_groups,
        group_size,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=8
    )
    scale = (mx - mn) / (2 ** bit - 1)
    quant_data = data - mn.unsqueeze(-1)
    quant_data.div_(scale.unsqueeze(-1))
    quant_data = quant_data.clamp_(0, 2 ** bit - 1).round_()

    # 차원 복구
    quant_data = quant_data.view(-1, T).to(torch.int32)

    feat_per_int = 32 // bit
    packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int) # (B * nh * D, 32//bit)
    code = torch.zeros(*packshape, device=quant_data.device, dtype=torch.int32)
    grid = lambda x: (triton.cdiv(quant_data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
    _pack_along_last_dim[grid](bit, quant_data, code, quant_data.shape[0],
                               quant_data.shape[1], feat_per_int,
                               BLOCK_SIZE_N=BLOCK_SIZE_N,
                               num_warps=8)
    return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

@triton.jit
def _minmax_along_last_dim(
    x_ptr,
    mn_ptr, mx_ptr,
    total_elements: tl.constexpr,
    N: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    bid = tl.program_id(axis=0)
    offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
    mask = offsets < total_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mx_val = tl.max(x, axis=1)
    mn_val = tl.min(x, axis=1)
    tl.store(mn_ptr + offsets_b, mn_val, mask=offsets_b<N*num_groups)
    tl.store(mx_ptr + offsets_b, mx_val, mask=offsets_b<N*num_groups)

@triton.jit
def _pack_along_last_dim(
    bits: tl.constexpr,
    intensor_ptr,
    code_ptr,
    N,
    num_feats: tl.constexpr,
    feat_per_int: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    num_int_per_y_dim = num_feats // feat_per_int
    bid = tl.program_id(axis=0)
    yid = tl.program_id(axis=1)
    offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int
    packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    for i in range(feat_per_int):
        ptr = block_start + i
        element = tl.load(ptr, mask=offs_N<N, other=0.)
        element = element << (i * bits)
        packed = packed | element
    tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)