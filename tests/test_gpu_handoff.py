"""GPU handoff tests -- skipped unless CUDA is present.  NOT run during the review (no GPU available);
they exist so GPU testing starts from executable checks instead of prose.

    python -m pytest tests/test_gpu_handoff.py -q -s          # -s shows the throughput report

Finding F-21 (perf, dtype, masking kernels) can only be closed with these numbers.
"""
import time

import pytest
import torch

from helpers import small_lm_config
from llm.model import TransformerModel
from llm.model_config import ModelConfig

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def _pair(cfg=None, dtype=torch.float32):
    torch.manual_seed(0)
    cpu = TransformerModel(cfg or small_lm_config()).eval()
    gpu = TransformerModel(cfg or small_lm_config()).eval()
    gpu.load_state_dict(cpu.state_dict())
    return cpu, gpu.to("cuda", dtype)


def test_cpu_gpu_logit_parity_fp32():
    cpu, gpu = _pair()
    ids = torch.randint(0, 1722, (2, 32))
    with torch.no_grad():
        assert torch.allclose(cpu(ids), gpu(ids.cuda()).cpu(), atol=2e-3, rtol=1e-3)


def test_kv_cache_decode_matches_full_forward_on_gpu():
    _, gpu = _pair()
    p = torch.randint(0, 1722, (2, 6), device="cuda")
    assert torch.equal(gpu.generate(p, max_new_tokens=8, temperature=0.0, use_cache=True),
                       gpu.generate(p, max_new_tokens=8, temperature=0.0, use_cache=False))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_logits_stay_close_to_fp32(dtype):
    """Relative error budget only meaningful for a TRAINED checkpoint (logit scale ~ O(1-10));
    re-run with the retrained weights before trusting the number."""
    cpu, gpu = _pair(dtype=dtype)
    ids = torch.randint(0, 1722, (2, 32))
    with torch.no_grad():
        ref, low = cpu(ids), gpu(ids.cuda()).float().cpu()
    assert torch.isfinite(low).all()
    # Tolerance is a starting point (5% of the largest logit); tune against the trained checkpoint.
    assert (ref - low).abs().max() < 0.05 * ref.abs().max()


def test_left_padded_batch_has_no_nan_with_fused_kernels():
    """Fully-masked query rows (left padding) can yield NaN in fused SDPA kernels on some versions."""
    _, gpu = _pair()
    ids = torch.randint(0, 1722, (1, 6), device="cuda")
    mask = torch.tensor([[0, 0, 1, 1, 1, 1]], device="cuda")
    with torch.no_grad():
        assert torch.isfinite(gpu(ids, attention_mask=mask)).all()


def test_long_context_beyond_trained_length_is_finite_with_yarn():
    cfg = small_lm_config(yarn_scale_factor=4.0, max_position_embeddings=1024, yarn_original_max_position=256)
    _, gpu = _pair(cfg)
    ids = torch.randint(0, 1722, (1, 900), device="cuda")
    with torch.no_grad():
        assert torch.isfinite(gpu(ids)).all()


def test_report_decode_throughput_and_kv_memory():
    """Prints numbers; only asserts they are sane. Full-size 51.5M config, random weights."""
    torch.manual_seed(0)
    m = TransformerModel(ModelConfig()).cuda().eval()
    p = torch.randint(0, 8000, (1, 128), device="cuda")
    m.generate(p, max_new_tokens=4, temperature=0.0)                      # warm-up
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    m.generate(p, max_new_tokens=128, temperature=0.0)
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"\n[GPU REPORT] decode {128 / dt:.1f} tok/s (batch 1, 128 prompt + 128 new); "
          f"peak mem {torch.cuda.max_memory_allocated() / 1e6:.0f} MB; {torch.cuda.get_device_name()}")
    assert dt > 0
