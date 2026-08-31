# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o preprocessing: chat template + tokenization.

Text-only for now. Image slicing (slice_mode) and audio feature extraction
land with the encoder stages; they will populate ``encoder_inputs`` here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _resolve_local_model_dir(model_path: str) -> str:
    if Path(model_path).exists():
        return model_path
    return str(resolve_model_path(model_path, local_files_only=False))


class MiniCPMOPreprocessor:
    def __init__(self, model_path: str, *, max_seq_len: int | None = None):
        local_dir = _resolve_local_model_dir(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, trust_remote_code=True
        )
        self.max_seq_len = max_seq_len

    async def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            if inputs.get("images") or inputs.get("audio") or inputs.get("audios"):
                raise NotImplementedError(
                    "MiniCPM-o image/audio inputs are not wired up yet"
                )
        else:
            messages = inputs

        if (
            isinstance(messages, list)
            and messages
            and all(isinstance(token, int) for token in messages)
        ):
            # Pre-tokenized prompt ids (rollout path): use them verbatim so
            # serving tokens match the caller's exactly.
            prompt_text = ""
            input_ids = torch.tensor(messages, dtype=torch.long)
        else:
            if isinstance(messages, str):
                prompt_text = messages
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"][0].to(dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        state = MiniCPMOPipelineState(
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        # Downstream projections consume the canonical state.
        payload.request.inputs = None
        return payload
