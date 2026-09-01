# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for MiniCPM-o stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import xxhash

from sglang_omni.models.minicpm_o.payload_types import (
    MiniCPMOPipelineState,
    ThinkerOutput,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

logger = logging.getLogger(__name__)

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
MM_AGGREGATE_STAGE = "mm_aggregate"


def _resolve_seed(params: dict[str, Any]) -> int | None:
    for key in ("seed", "sampling_seed"):
        value = params.get(key)
        if value is not None:
            return int(value)
    return None


# ---------------------------------------------------------------------------
# Routing / projection edges
# ---------------------------------------------------------------------------


def resolve_preprocessing_next_stages(
    request_id: str, output: StagePayload
) -> list[str]:
    del request_id
    state = MiniCPMOPipelineState.from_dict(output.data)
    return [
        *_encoder_stages_with_model_inputs(state.encoder_inputs),
        MM_AGGREGATE_STAGE,
    ]


def resolve_mm_aggregate_wait_sources(
    request_id: str,
    from_stage: str,
    payload: StagePayload,
) -> list[str] | None:
    del request_id
    if from_stage != "preprocessing":
        return None
    state = MiniCPMOPipelineState.from_dict(payload.data)
    return [
        "preprocessing",
        *_encoder_stages_with_model_inputs(state.encoder_inputs),
    ]


def project_preprocessing_to_image_encoder(payload: StagePayload) -> StagePayload:
    return _project_preprocessing_to_encoder(payload, stage_name=IMAGE_STAGE)


def project_preprocessing_to_audio_encoder(payload: StagePayload) -> StagePayload:
    return _project_preprocessing_to_encoder(payload, stage_name=AUDIO_STAGE)


def project_preprocessing_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    projected = MiniCPMOPipelineState(
        prompt=dict(state.prompt) if isinstance(state.prompt, dict) else None,
        mm_inputs=dict(state.mm_inputs),
        encoder_inputs=_project_encoder_input_metadata(state.encoder_inputs),
        stream_state=dict(state.stream_state),
    )
    return _payload_with_state(payload, projected)


def project_encoder_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    if len(state.encoder_outs) != 1:
        raise ValueError(
            "Expected exactly one encoder output in payload, got "
            f"{sorted(state.encoder_outs)}"
        )
    stage_name = next(iter(state.encoder_outs))
    projected = MiniCPMOPipelineState(
        encoder_outs={stage_name: state.encoder_outs[stage_name]}
    )
    return _payload_with_state(payload, projected)


def project_thinker_to_decode(payload: StagePayload) -> StagePayload:
    """Keep decode payload focused on text detokenization state."""
    state = MiniCPMOPipelineState.from_dict(payload.data)
    state.thinker_inputs = {}

    if isinstance(state.thinker_out, dict):
        thinker_out = dict(state.thinker_out)
        thinker_out.pop("extra_model_outputs", None)
        state.thinker_out = thinker_out

    if state.engine_outputs:
        engine_outputs = dict(state.engine_outputs)
        thinker_engine_out = engine_outputs.get(THINKER_STAGE)
        if isinstance(thinker_engine_out, dict):
            thinker_engine_out = dict(thinker_engine_out)
            thinker_engine_out.pop("extra_model_outputs", None)
            engine_outputs[THINKER_STAGE] = thinker_engine_out
        state.engine_outputs = engine_outputs

    return _payload_with_state(payload, state)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _project_preprocessing_to_encoder(
    payload: StagePayload,
    *,
    stage_name: str,
) -> StagePayload:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    stage_inputs = state.encoder_inputs.get(stage_name)
    encoder_inputs = (
        {stage_name: dict(stage_inputs)} if isinstance(stage_inputs, dict) else {}
    )
    projected = MiniCPMOPipelineState(encoder_inputs=encoder_inputs)
    return _payload_with_state(payload, projected)


def _payload_with_state(
    payload: StagePayload, state: MiniCPMOPipelineState
) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _project_encoder_input_metadata(
    encoder_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    projected: dict[str, dict[str, Any]] = {}
    for stage_name, stage_inputs in encoder_inputs.items():
        if not isinstance(stage_inputs, dict):
            continue
        stage_metadata: dict[str, Any] = {}
        cache_key = stage_inputs.get("cache_key")
        if cache_key is not None:
            stage_metadata["cache_key"] = cache_key
        if _has_encoder_model_input(stage_name, stage_inputs):
            stage_metadata["_active"] = True
        if stage_metadata:
            projected[stage_name] = stage_metadata
    return projected


def _encoder_stages_with_model_inputs(
    encoder_inputs: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        stage_name
        for stage_name in (IMAGE_STAGE, AUDIO_STAGE)
        if _has_encoder_model_input(stage_name, encoder_inputs.get(stage_name))
    ]


def _has_encoder_model_input(stage_name: str, stage_inputs: Any) -> bool:
    if not isinstance(stage_inputs, dict):
        return False
    if stage_inputs.get("_active") is not None:
        return stage_inputs.get("_active") is True
    if stage_name == IMAGE_STAGE:
        return stage_inputs.get("pixel_values") is not None
    if stage_name == AUDIO_STAGE:
        return stage_inputs.get("audio_features") is not None
    return False


# ---------------------------------------------------------------------------
# Encoder request builders
# ---------------------------------------------------------------------------


@dataclass
class EncoderRequestData:
    """Prepared inputs for one encoder stage forward."""

    model_inputs: dict[str, Any]
    cache_key: str | None = None
    skip_result: dict[str, Any] | None = None


def build_encoder_request(
    state: MiniCPMOPipelineState, *, stage_name: str
) -> EncoderRequestData:
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(model_inputs={}, skip_result={})
    if inputs.get("_skip"):
        skip_result = inputs.get("_result")
        return EncoderRequestData(
            model_inputs={},
            skip_result=skip_result if isinstance(skip_result, dict) else {},
        )
    cache_key = inputs.get("cache_key")
    model_inputs = {
        k: v for k, v in inputs.items() if k not in ("cache_key", "_active")
    }
    return EncoderRequestData(
        model_inputs=model_inputs,
        cache_key=str(cache_key) if cache_key is not None else None,
    )


def apply_encoder_result(
    state: MiniCPMOPipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    encoder_out = result if isinstance(result, dict) else {"result": result}
    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def _bounds_to_positions(bounds: Any, device: Any = None) -> torch.Tensor | None:
    """Flatten ``(N, 2)`` [start, end) bound rows into a 1D position tensor."""
    if not isinstance(bounds, torch.Tensor) or bounds.numel() == 0:
        return None
    return torch.cat(
        [torch.arange(int(r[0]), int(r[1]), device=device) for r in bounds]
    )


def _apply_mm_pad_values(
    input_ids: torch.Tensor,
    *,
    mm_inputs: dict[str, Any],
    model_inputs: dict[str, Any],
    vocab_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Rewrite placeholder ``<unk>`` runs to per-modality pad values.

    MiniCPM-o marks all placeholders with the same ``<unk>`` token; the
    ``image_bound`` / ``audio_bounds`` intervals disambiguate modalities. The
    base thinker runner injects embeddings by matching ``pad_values``, so we
    rewrite the ids inside each modality's intervals to a cache-key-derived
    pad value and record the positions.
    """
    pad_values: dict[str, int] = {}
    empty = torch.empty(0, dtype=torch.long)
    # The base runner iterates image/video/audio unconditionally, so every
    # modality needs a positions entry even when absent from the prompt.
    mm_positions: dict[str, torch.Tensor] = {
        "image": empty,
        "video": empty,
        "audio": empty,
    }
    has_any = False
    input_ids = input_ids.clone()
    for modality in ("image", "audio"):
        info = mm_inputs.get(modality)
        if not isinstance(info, dict):
            continue
        positions = _bounds_to_positions(info.get("bounds"))
        if positions is None:
            continue
        has_any = True
        cache_key = str(info.get("cache_key") or modality)
        h = xxhash.xxh3_64(cache_key.encode()).intdigest()
        pad_val = vocab_size + h % (1 << 62)
        pad_values[modality] = pad_val
        input_ids[positions] = pad_val
        mm_positions[modality] = positions
    if not has_any:
        return input_ids, None
    model_inputs["pad_values"] = pad_values
    return input_ids, mm_positions


# ---------------------------------------------------------------------------
# Thinker request builders
# ---------------------------------------------------------------------------


def build_sglang_thinker_request(
    state: MiniCPMOPipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    request_id: str | None = None,
) -> SGLangARRequestData:
    """Build SGLangARRequestData from pipeline state.

    Constructs a SGLang Req with normalized SamplingParams, then wraps it in
    SGLangARRequestData. MiniCPM-o uses plain 1D RoPE, so no
    multimodal_inputs/mrope positions get attached; multimodal embeddings are
    injected in the model runner via ``req.omni_model_inputs`` at the
    placeholder positions derived from ``image_bound`` / ``audio_bounds``.
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prompt = state.prompt
    input_ids = prompt["input_ids"]
    attention_mask = prompt.get("attention_mask")

    thinker_inputs = state.thinker_inputs or {}
    model_inputs = thinker_inputs.get("model_inputs")
    if model_inputs is None:
        model_inputs = {}
    elif not isinstance(model_inputs, dict):
        raise TypeError("MiniCPM-o thinker model_inputs must be a dict when provided")
    else:
        model_inputs = dict(model_inputs)

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", -1),
        min_p=params.get("min_p", 0.0),
        repetition_penalty=params.get("repetition_penalty", 1.0),
        stop=params.get("stop") or [],
        stop_token_ids=params.get("stop_token_ids") or [],
        sampling_seed=_resolve_seed(params),
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    input_ids = input_ids.to(dtype=torch.long)
    mm_positions = None
    if model_inputs:
        input_ids, mm_positions = _apply_mm_pad_values(
            input_ids,
            mm_inputs=state.mm_inputs,
            model_inputs=model_inputs,
            vocab_size=vocab_size,
        )
    req = Req(
        rid=request_id or "req-0",
        origin_input_text="",
        origin_input_ids=input_ids.tolist(),
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )
    req.tokenizer = tokenizer

    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None
    req._codec_suppress_tokens = None
    req._omni_mm_positions = mm_positions

    data = SGLangARRequestData(
        input_ids=input_ids,
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    data.return_logprob = bool(params.get("return_logprob"))
    return data


def apply_thinker_result(
    state: MiniCPMOPipelineState,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    output_ids = list(result.output_ids)
    thinker_out: ThinkerOutput = {
        "output_ids": output_ids,
        "step": len(output_ids),
        "is_final": True,
        "extra_model_outputs": dict(result.extra_model_outputs),
    }

    for attr in ("finish_reason", "weight_version", "output_token_logprobs"):
        value = getattr(result, attr, None)
        if value is not None:
            thinker_out[attr] = value

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


def make_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    stage_name: str = THINKER_STAGE,
):
    """Build model-specific StagePayload <-> scheduler adapters for thinker."""

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        req_data = build_sglang_thinker_request(
            state,
            params=payload.request.params or {},
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        state = MiniCPMOPipelineState.from_dict(payload.data)
        apply_thinker_result(state, stage_name=stage_name, result=data)
        return _payload_with_state(payload, state)

    return request_builder, result_adapter


def make_thinker_stream_output_builder():
    def _build_stream_output(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        req = getattr(req_data, "req", None)
        if req is not None and req.inflight_middle_chunks > 0:
            # Suppress emission while chunked prefill still consumes prompt
            # tokens; those steps carry no generated token.
            return []
        if req_output.data is None:
            return []

        stage_payload = req_data.stage_payload
        is_streaming = bool(
            stage_payload is not None
            and (stage_payload.request.params or {}).get("stream", False)
        )
        if not is_streaming:
            return []

        token_id = int(req_output.data)
        # Wrap int; stream transport only accepts tensors.
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=torch.tensor([token_id], dtype=torch.long),
                target=DECODE_STAGE,
                metadata={"token_id": token_id},
            )
        ]

    return _build_stream_output
