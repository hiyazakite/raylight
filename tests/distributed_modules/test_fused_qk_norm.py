import os
import torch
import torch.distributed as dist
from raylight.distributed_modules.tensor_parallel import (
    TPRMSNormAcrossHeads,
    TPFusedQKNorm,
)


def init_dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group("gloo", rank=0, world_size=1)


def test_fused_matches_separate():
    # Prefer torch.distributed collectives for this unit test; disable
    # the newer funcol path inside the module so it doesn't require a
    # resolved group name object.
    import raylight.distributed_modules.tensor_parallel as _tpmod
    _tpmod._HAS_FUNCOL = False
    init_dist()
    full = 64
    local = 64
    eps = 1e-5
    torch.manual_seed(0)

    q = torch.randn(2, 3, local, dtype=torch.float32)
    k = torch.randn(2, 4, local, dtype=torch.float32)

    sep_q = TPRMSNormAcrossHeads(full, local, eps)
    sep_k = TPRMSNormAcrossHeads(full, local, eps)
    fused = TPFusedQKNorm(full, local, eps)

    # Align weights
    with torch.no_grad():
        fused.q_weight.copy_(sep_q.weight)
        fused.k_weight.copy_(sep_k.weight)

    out_q_sep = sep_q(q)
    out_k_sep = sep_k(k)

    out_q_f, out_k_f = fused(q, k)

    assert torch.allclose(out_q_sep, out_q_f, atol=1e-6, rtol=1e-4)
    assert torch.allclose(out_k_sep, out_k_f, atol=1e-6, rtol=1e-4)


def test_fused_q_only_matches_separate_q():
    import raylight.distributed_modules.tensor_parallel as _tpmod
    _tpmod._HAS_FUNCOL = False
    init_dist()
    full = 48
    local = 48
    eps = 1e-5
    torch.manual_seed(1)

    q = torch.randn(1, 5, local, dtype=torch.float32)

    sep_q = TPRMSNormAcrossHeads(full, local, eps)
    fused = TPFusedQKNorm(full, local, eps)

    with torch.no_grad():
        fused.q_weight.copy_(sep_q.weight)

    out_q_sep = sep_q(q)
    (out_q_f,) = fused(q, None)

    assert torch.allclose(out_q_sep, out_q_f, atol=1e-6, rtol=1e-4)
