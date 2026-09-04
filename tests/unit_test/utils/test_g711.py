# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
import warnings

import numpy as np
import pytest

from sglang_omni.utils import g711
from sglang_omni.utils.g711 import (
    ALAW,
    MULAW,
    decode_g711,
    resolve_g711_encoding,
    wrap_g711_as_wav,
)


def _reference_table(encoding: str) -> np.ndarray:
    # Note (Jeffro)audioop ships the reference G.711 tables until Python 3.13 removes it the project pins <3.13,
    # so we compare against it while we can.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        audioop = pytest.importorskip("audioop")
    decode = audioop.ulaw2lin if encoding == MULAW else audioop.alaw2lin
    pcm = np.frombuffer(decode(bytes(range(256)), 2), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


@pytest.mark.parametrize("encoding", [MULAW, ALAW])
def test_tables_match_the_reference_implementation(encoding: str) -> None:
    np.testing.assert_array_equal(g711._TABLES[encoding], _reference_table(encoding))


def test_decode_yields_one_sample_per_byte_in_unit_range() -> None:
    samples = decode_g711(bytes(range(256)), MULAW)

    assert samples.shape == (256,)
    assert samples.dtype == np.float32
    assert samples.min() >= -1.0 and samples.max() <= 1.0
    # 0xFF is µ-law silence; 0x00 is the loudest negative code.
    assert samples[0xFF] == 0.0
    assert samples[0x00] == pytest.approx(-32124 / 32768.0)


def test_decode_rejects_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="Unsupported G.711 encoding"):
        decode_g711(b"\x00", "pcm16")


@pytest.mark.parametrize(
    ("content_type", "filename", "expected"),
    [
        ("audio/basic", None, MULAW),
        ("audio/PCMU", "call.bin", MULAW),
        ("audio/x-mulaw; rate=8000", None, MULAW),
        ("audio/PCMA", None, ALAW),
        ("audio/pcma; rate=8000", "call.wav", ALAW),
        # Spellings nobody registered or documents are not accepted.
        ("audio/x-alaw", "call.alaw", None),
        ("audio/mulaw", None, None),
        # Generic media types defer to the filename.
        (None, "call.ulaw", MULAW),
        ("", "CALL.UL", MULAW),
        ("application/octet-stream", "call.alaw", ALAW),
        ("audio/*", "call.al", ALAW),
        # A concrete non-G.711 media type wins over the extension.
        ("audio/wav", "call.ulaw", None),
        ("audio/mpeg", None, None),
        (None, "call.wav", None),
        (None, None, None),
    ],
)
def test_resolve_encoding_from_media_type_then_filename(
    content_type, filename, expected
) -> None:
    assert resolve_g711_encoding(content_type, filename) == expected


def test_wrap_produces_a_wav_that_the_fast_path_decodes() -> None:
    from sglang_omni.preprocessing.audio import _parse_wav_bytes

    payload = bytes(range(256)) * 4
    wav = wrap_g711_as_wav(payload, MULAW)

    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    fmt_tag, channels, sample_rate, _, block_align, bits = struct.unpack(
        "<HHIIHH", wav[20:36]
    )
    assert (fmt_tag, channels, sample_rate, block_align, bits) == (7, 1, 8000, 1, 8)
    assert wav.endswith(payload)

    audio, decoded_rate = _parse_wav_bytes(wav)
    assert decoded_rate == 8000
    np.testing.assert_array_equal(audio, decode_g711(payload, MULAW))


def test_wrap_pads_odd_payloads_to_keep_riff_chunks_aligned() -> None:
    wav = wrap_g711_as_wav(b"\xff" * 3, ALAW)

    riff_size = struct.unpack("<I", wav[4:8])[0]
    assert riff_size == len(wav) - 8
    assert len(wav) % 2 == 0


def test_wrap_leaves_existing_wav_untouched() -> None:
    wav = wrap_g711_as_wav(b"\xff" * 10, MULAW)

    assert wrap_g711_as_wav(wav, MULAW) is wav


def test_wrap_rejects_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="Unsupported G.711 encoding"):
        wrap_g711_as_wav(b"\x00", "pcm16")
