# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

from typing import Any

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


class MiniCPMOThinkerModelRunner(ThinkerModelRunner):
    """Thinker runner over the MiniCPM-o text backbone.

    The base class resolves the embedding table through
    ``model.thinker.model.embed_tokens`` (satisfied by the wrapper's ``thinker``
    / ``model`` properties) but reads modality token ids from a Qwen-style
    ``hf_config.thinker_config``, which MiniCPM-o's flat config does not have.
    MiniCPM-o marks multimodal spans with ``<unk>`` runs plus bound intervals
    instead of dedicated placeholder tokens, so the id-based injection path is
    unused; the ids are set to -1 (matching no token).
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        # Skip ThinkerModelRunner.__init__ (it requires hf_config.thinker_config)
        # but keep its grandparent initialization.
        super(ThinkerModelRunner, self).__init__(tp_worker, output_processor)

        model = self.model
        self._outer_model = model.thinker
        self._text_model = self._outer_model.model
        self._embed_tokens = self._text_model.embed_tokens
        self._th_host_bufs = None
        self._th_slot = 0

        self._image_token_id = -1
        self._video_token_id = -1
        self._audio_token_id = -1
