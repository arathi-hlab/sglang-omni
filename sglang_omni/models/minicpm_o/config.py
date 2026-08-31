# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniCPM-o."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.minicpm_o"
THINKER_STAGE = "thinker"


def _preprocessing_stage(*, process: str) -> StageConfig:
    # Encoder stages join in Phase 2; until then preprocessing feeds
    # mm_aggregate directly and the multi-branch route_fn stays unused.
    return StageConfig(
        name="preprocessing",
        process=process,
        factory_path=f"{_PKG}.stages.create_preprocessing_executor",
        factory=FactoryArgs(max_seq_len=8192),
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": (
                f"{_PKG}.request_builders.project_preprocessing_to_mm_aggregate"
            ),
        },
    )


def _aggregate_stage(*, process: str, gpu: int) -> StageConfig:
    return StageConfig(
        name="mm_aggregate",
        process=process,
        factory_path=f"{_PKG}.stages.create_aggregate_executor",
        gpu=gpu,
        wait_for=["preprocessing"],
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
        next="thinker",
        disable_direct_cuda_ipc_payload=True,
    )


def _thinker_stage(*, gpu: int, process: str) -> StageConfig:
    return EngineStageConfig(
        name="thinker",
        process=process,
        factory_path=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory=FactoryArgs(max_seq_len=8192, enable_async_decode=True),
        gpu=gpu,
        next="decode",
        stream_to=["decode"],
        project_payload={
            "decode": f"{_PKG}.request_builders.project_thinker_to_decode",
        },
    )


def _decode_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="decode",
        process=process,
        factory_path=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _text_stages() -> list[StageConfig]:
    # image_encoder / audio_encoder stages join in Phase 2; until then the
    # preprocessing route_fn never selects them for text-only requests.
    return [
        _preprocessing_stage(process="pipeline"),
        _aggregate_stage(process="pipeline", gpu=0),
        _thinker_stage(gpu=0, process="pipeline"),
        _decode_stage(process="pipeline"),
    ]


class MiniCPMOPipelineConfig(PipelineConfig):
    """Text-only pipeline: preprocessing → mm_aggregate → thinker → decode."""

    architecture: ClassVar[str] = "MiniCPMO"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = Field(default_factory=_text_stages)


EntryClass = MiniCPMOPipelineConfig
