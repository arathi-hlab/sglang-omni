# SPDX-License-Identifier: Apache-2.0
"""The shared ServerArgs builder completes a SGLang-derived prefill ladder."""

from __future__ import annotations

from typing import Any

from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig

from sglang_omni.scheduling.sglang_backend import server_args_builder


class _ResolvedServerArgs:
    """Stands in for a ServerArgs whose resolution pipeline has run."""

    locked: frozenset[tuple[str, str]] = frozenset()

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.enable_dp_attention = False
        self.startup_weight_load_mode = "serial"
        self.cuda_graph_config = CudaGraphConfig(
            prefill=PhaseConfig(
                backend=kwargs["cuda_graph_backend_prefill"],
                bs=[3840, 4096],
                max_bs=4592,
            )
        )
        self._cuda_graph_config_locked = set(type(self).locked)


class _DeclaredListServerArgs(_ResolvedServerArgs):
    locked = frozenset({("prefill", "bs")})


def _build(monkeypatch, server_args_cls: type, **extra: Any) -> Any:
    monkeypatch.setattr(server_args_builder, "ServerArgs", server_args_cls)
    return server_args_builder.build_sglang_server_args(
        "model",
        context_length=128,
        cuda_graph_backend_prefill="breakable",
        **extra,
    )


def test_builder_completes_a_derived_ladder_to_the_resolved_cap(monkeypatch) -> None:
    server_args = _build(monkeypatch, _ResolvedServerArgs)

    assert server_args.cuda_graph_config.prefill.bs == [3840, 4096, 4592]
    assert server_args.cuda_graph_config.prefill.max_bs == 4592


def test_builder_leaves_a_declared_ladder_alone(monkeypatch) -> None:
    server_args = _build(monkeypatch, _DeclaredListServerArgs)

    assert server_args.cuda_graph_config.prefill.bs == [3840, 4096]


def test_builder_skips_the_fill_for_a_disabled_backend(monkeypatch) -> None:
    server_args = _build(
        monkeypatch, _ResolvedServerArgs, cuda_graph_backend_prefill="disabled"
    )

    assert server_args.cuda_graph_config.prefill.bs == [3840, 4096]


def test_builder_returns_an_unresolved_dummy_config_untouched() -> None:
    server_args = server_args_builder.build_sglang_server_args(
        "dummy", context_length=128
    )

    assert server_args.cuda_graph_config is None
