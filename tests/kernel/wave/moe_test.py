# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import wave_lang.kernel as tk
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
#from .common.utils import (
#    require_e2e,
#)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    enable_scheduling_barriers,
    dump_generated_mlir,
    check_individual_kernels,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.templates.moe import (
    get_gemm_kernel,
    get_silu_and_mul_kernel,
)
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang import DataType
import torch.nn.functional as F

torch.manual_seed(0)


def silu_and_mul_torch_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def silu_and_mul_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate * up / (1 + torch.exp(-gate))


def softmax(
    x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    # Convert input to float32 for stable calculations
    x_float = x.to(dtype)

    # Subtract max for numerical stability
    max_vals = torch.max(x_float, dim=dim, keepdim=True).values
    x_exp = torch.exp(x_float - max_vals)

    # Compute softmax
    sum_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    result = x_exp / sum_exp

    return result


def get_wave_silu_and_mul_kernel(
    m: int,
    n: int,
    datatype: DataType,
):
    assert datatype in [
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported datatype: {datatype}"
    silu_and_mul_kernel, symbols = get_silu_and_mul_kernel(
        m, n, tkl.f16 if datatype == torch.float16 else tkl.bf16
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        wave_runtime=False,
    )
    options = set_default_run_config(options)
    silu_and_mul = wave_compile(options, silu_and_mul_kernel)
    return silu_and_mul


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    assert len(gate.shape) == len(up.shape) == 2
    assert gate.shape[0] == up.shape[0] and gate.shape[1] == up.shape[1]
    wave_kernel = get_wave_silu_and_mul_kernel(gate.shape[0], gate.shape[1], gate.dtype)

    out = torch.zeros(gate.shape, dtype=gate.dtype, device=gate.device)
    wave_kernel(gate, up, out)
    ref = silu_and_mul_ref(gate, up)
    if check_individual_kernels:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, check_device=False)
    return out


def get_wave_gemm_kernel(
    m: int,
    k: int,
    n: int,
    block_m: int,
    block_k: int,
    block_n: int,
    mfma_variant: MMAType,
    datatype: DataType,
    get_double_gemm: bool,
):
    print(f"M {m} K {k} N {n}")
    gemm, symbols = get_gemm_kernel(
        m,
        k,
        n,
        block_m,
        block_k,
        block_n,
        mfma_variant,
        datatype,
        get_double_gemm,
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)
    return gemm


def torch_ref_moe(a, w1, w2, score, topk):
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    out = torch.zeros(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            x = a[mask] @ w1[i].transpose(0, 1)
            d = x.shape[-1] // 2
            out[mask] = silu_and_mul_torch_ref(x[..., :d], x[..., d:]) @ w2[i].transpose(0, 1)
    return (
        out.view(m, -1, w2.shape[1]) * topk_weight.view(m, -1, 1).to(out.dtype)
    ).sum(dim=1)


@torch.compile(  # Requires PyTorch 2.0+
    fullgraph=False,  # Allow dynamic control flow
    mode="max-autotune",  # Aggressive optimizations
)
def torch_ref_moe_split_w1(a, w1_gate, w1_up, w2, score, topk):
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)  # [m * topk, k]
    out = torch.zeros(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)  # [m * topk, k]
    score = torch.softmax(score, dim=-1, dtype=torch.float32)  # [m, e]
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)  # [m * topk]
    topk_ids = topk_ids.view(-1)  # [m * topk]
    for i in range(w1_gate.shape[0]):
        mask = (
            topk_ids == i
        )  # num_selected (which of the m * topk tokens selected this expert)
        if mask.sum():
            # Split into gate and up projections
            gate = a[mask] @ w1_gate[i].transpose(0, 1) # [num_selected, n]
            up = a[mask] @ w1_up[i].transpose(0, 1)  # [num_selected, n]
            lhs = torch.zeros(m, w2.shape[-1], dtype=a.dtype, device=a.device)
            lhs = silu_and_mul_torch_ref(gate, up)
            out[mask] = lhs @ w2[i].transpose(0, 1)  # [num_selected, k]
    return (
        out.view(m, -1, w2.shape[1]) * topk_weight.view(m, -1, 1).to(out.dtype)
    ).sum(
        dim=1
    )  # [m, k]


def tkw_moe_split_w1(a, w1_gate, w1_up, w2, score, topk, block_m=64, block_n=64, block_k=64):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    dtype = tkl.f16 if a.dtype == torch.float16 else tkl.bf16
    assert w1_gate.shape[0] == w1_up.shape[0] == w2.shape[0]
    e = w1_gate.shape[0]
    for i in range(e):
        mask = topk_ids == i
        if mask.sum():
            m = int(mask.sum())
            gate = torch.zeros(
                m, w1_gate[i].shape[0], dtype=torch.float32, device=a.device
            )
            up = torch.zeros(
                m, w1_gate[i].shape[0], dtype=torch.float32, device=a.device
            )
            assert w1_gate[i].shape == w1_up[i].shape
            gemm_kernel_gate_up = get_wave_gemm_kernel(
                m,  # M
                w1_gate[i].shape[-1],  # K
                w1_gate[i].shape[0],  # N
                block_m,
                block_n,
                block_k,
                MMAType.F32_16x16x16_F16,
                dtype,
                False,
            )
#           gemm_kernel_gate_up(a[mask], w1_gate[i], w1_up[i], gate, up)
            gemm_kernel_gate_up(a[mask], w1_gate[i], gate)
            gemm_kernel_gate_up(a[mask], w1_up[i], up)
            gate = gate.to(dtype=a.dtype)
            up = up.to(dtype=a.dtype)
#           check_individual_kernels = True
            if check_individual_kernels:
                torch.testing.assert_close(
                    gate,
                    a[mask] @ w1_gate[i].transpose(0, 1),
                    rtol=rtol,
                    atol=atol,
                    check_device=False,
                )
                torch.testing.assert_close(
                    up,
                    a[mask] @ w1_up[i].transpose(0, 1),
                    rtol=rtol,
                    atol=atol,
                    check_device=False,
                )
            lhs = torch.zeros(m, w2.shape[-1], dtype=a.dtype, device=a.device)
          # lhs = silu_and_mul_torch_ref(gate, up)
            lhs = silu_and_mul(gate, up)
            rhs = w2[i]
            partial_out = torch.zeros(
                m, w2.shape[1], dtype=torch.float32, device=a.device
            )
            gemm_kernel_out = get_wave_gemm_kernel(
                m,  # M
                w2[i].shape[-1],  # K
                w2[i].shape[0],  # N
                block_m,
                block_n,
                block_k,
                MMAType.F32_16x16x16_F16,
                dtype,
                False,
            )
            gemm_kernel_out(lhs, rhs, partial_out)
            partial_out = partial_out.to(dtype=a.dtype)
            if check_individual_kernels:
                torch.testing.assert_close(
                    partial_out, lhs @ rhs.transpose(0, 1), rtol=rtol, atol=atol, check_device=False
                )
            out[mask] = partial_out
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


num_experts = [8, 64]
top_ks = [2]
m_values = [1, 33]
n_values = [128, 1024]
k_values = [511]
dtypes = [torch.float16, torch.bfloat16]
rtol, atol = 1e-1, 1e-2


#@require_e2e
@pytest.mark.parametrize("m", m_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("e", num_experts)
@pytest.mark.parametrize("topk", top_ks)
@pytest.mark.parametrize("dtype", dtypes)
def testReferenceMoe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: DataType,
):
    device = "cuda"

    if dtype == torch.float16 and k == 1024:
        pytest.skip("This combination generates NaNs and INFs")

    # TODO: investigate why using torch.randn would have precision issue in silu computation
    a = torch.rand((m, k), dtype=dtype, device=device)
    w1 = torch.rand((e, 2 * n, k), dtype=dtype, device=device)
    w2 = torch.rand((e, k, n), dtype=dtype, device=device)
    score = torch.rand((m, e), dtype=dtype, device=device)

    ref_output = torch_ref_moe(a, w1, w2, score, topk)

    # TODO: remove manual splitting
    # We need to manually split w1 into 2 halves, since this is
    # required by `silu_and_mul` kernel, and currently we can't
    # do this in Wave.
    w1_gate = w1[:, :n, :]  # First half for gate
    w1_up = w1[:, n:, :]  # Second half for up projection

    # Make sure the algorithm with w1 splitting works in PyTorch.
    ref_split_output = torch_ref_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)
    torch.testing.assert_close(ref_split_output, ref_output, rtol=rtol, atol=atol)

    # The implementation in Wave should also work.
    tkw_output = tkw_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)
    torch.testing.assert_close(tkw_output, ref_output, rtol=rtol, atol=atol)


from time import perf_counter
from torch.autograd import DeviceType
from torch.profiler import profile as torch_profile, ProfilerActivity
import numpy as np

m_values = [1, 33, 64, 222]
m_values = [16384 * 4]
m_values = [5120]
n_values = [128, 1024]
n_values = [8192]
k_values = [128, 511, 1024]
k_values = [5120]
e_values = [8, 64]
e_values = [16]
topk_values = [2, 6]
topk_values = [2]
dtypes = [torch.float16, torch.bfloat16]
dtypes = [torch.float16]
block_m_values = [32, 64, 128, 256]
block_n_values = [32, 64, 128, 256]
block_k_values = [32, 64, 128, 256]

block_m_values = [32]
block_n_values = [64]
block_k_values = [16]

block_m_values = [32]
block_n_values = [128]
block_k_values = [32]

# Best, slightly better than the above 2
block_m_values = [32]
block_n_values = [64]
block_k_values = [32]

block_m_values = [64]
block_n_values = [128]
block_k_values = [32]

# llama4
# ~2xx for all
block_m_values = [64]
block_n_values = [64]
block_k_values = [64]

# 2xx, 3xx but also 19x
block_m_values = [128]
block_n_values = [64]
block_k_values = [64]

block_m_values = [128]
block_n_values = [32]
block_k_values = [64]

for m in m_values:
    for n in n_values:
        for k in k_values:
            for e in e_values:
                for topk in topk_values:
                    for dtype in dtypes:
                        for block_m in block_m_values:
                            for block_n in block_n_values:
                                for block_k in block_k_values:
                                    device = "cuda"
                                    a = torch.rand((m, k), dtype=dtype, device=device)
                                    w1 = torch.rand((e, 2 * n, k), dtype=dtype, device=device)
                                    w2 = torch.rand((e, k, n), dtype=dtype, device=device)
                                    score = torch.rand((m, e), dtype=dtype, device=device)
                                    w1_gate = w1[:, :n, :]  # First half for gate
                                    w1_up = w1[:, n:, :]  # Second half for up projection

                                    num_its = 15
                                    print_profile = True
                                    row_limit = 20

            #                       print(f"Num tokens {m}")
            #                       print(f"Num experts {e}")
            #                       print(f"Top K {topk}")

              #                     start = perf_counter()
              #                     ref_split_output = torch_ref_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)

                                    tkw_output = tkw_moe_split_w1(a, w1_gate, w1_up, w2, score, topk, block_m, block_n, block_k)
              #                     tkw_output = tkw_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)

            # #                     ref_output = torch_ref_moe(a, w1, w2, score, topk)
              #                     torch.cuda.synchronize()  # Wait for all GPU operations to complete
              #                     end = perf_counter()
              #                     print(f"Execution took {end - start:.6f} seconds")

#                                   torch.cuda.synchronize()
#                                   with torch_profile(
#                                       activities=[ProfilerActivity.CUDA],
#                                       # Discard 15 iterations to gain stability.
#                                       schedule=torch.profiler.schedule(
#                                           wait=5, warmup=5, active=num_its, skip_first=5
#                                       ),
#                                     # record_shapes=True,
#                                   ) as prof:
#                                       for i in range(num_its + 15):
#                                           ref_split_output = torch_ref_moe_split_w1(a, w1_gate, w1_up, w2, score, topk, block_m, block_n, block_k)
#                                         # ref_output = torch_ref_moe(a, w1, w2, score, topk)
#                                         # tkw_output = tkw_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)
#                                           torch.cuda.synchronize()
#                                           prof.step()
#                                   events = prof.key_averages(group_by_input_shape=True)
#                                   if print_profile:
#                                       print(
#                                           events.table(sort_by="cuda_time_total", row_limit=row_limit),
#                                           flush=True,
#                                       )
#                                       print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#                                   # Return the average device time in milliseconds
#                                   print(np.array(
#                                       [
#                                           event.self_device_time_total
#                                           for event in events
#                                           if event.device_type == DeviceType.CUDA and ("emcpy" not in event.key)
#                                       ]
#                                   ).sum() / (num_its * 1000.0))

