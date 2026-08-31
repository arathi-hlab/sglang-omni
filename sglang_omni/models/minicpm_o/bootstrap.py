# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o-specific scheduler construction."""

from __future__ import annotations

from typing import Any


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
):
    """Create the MiniCPM-o thinker scheduler (text output only)."""
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.minicpm_o.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
    )
    from sglang_omni.models.minicpm_o.thinker_model_runner import (
        MiniCPMOThinkerModelRunner,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="MiniCPMO",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=None,
    )
    model_runner = MiniCPMOThinkerModelRunner(model_worker, output_proc)

    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)
    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=make_thinker_stream_output_builder(),
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )
