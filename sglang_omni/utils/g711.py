# SPDX-License-Identifier: Apache-2.0
"""G.711 (µ-law / A-law) telephony audio helpers."""

from __future__ import annotations

import struct

import numpy as np

from sglang_omni.utils.audio import is_riff_wav

# ITU-T G.711 fixes the telephony sample rate and both companding laws:
# https://www.itu.int/rec/T-REC-G.711
G711_SAMPLE_RATE = 8000

MULAW = "mulaw"
ALAW = "alaw"

# WAVE format tags, registered in RFC 2361 (WAVE_FORMAT_ALAW = 0x0006,
# WAVE_FORMAT_MULAW = 0x0007): https://www.rfc-editor.org/rfc/rfc2361
WAV_FORMAT_ALAW = 6
WAV_FORMAT_MULAW = 7

_WAV_FORMAT_TAGS = {MULAW: WAV_FORMAT_MULAW, ALAW: WAV_FORMAT_ALAW}

# Media types telephony providers and SIP stacks attach to raw G.711 bytes.
# audio/basic is 8 kHz µ-law per RFC 2046 section 4.3:
# https://www.rfc-editor.org/rfc/rfc2046#section-4.3
# audio/PCMU and audio/PCMA are the RTP payload names from RFC 4856:
# https://www.rfc-editor.org/rfc/rfc4856
# audio/x-mulaw is what Twilio Media Streams put in their mediaFormat:
# https://www.twilio.com/docs/voice/media-streams/websocket-messages
_MULAW_CONTENT_TYPES = frozenset({"audio/basic", "audio/pcmu", "audio/x-mulaw"})
_ALAW_CONTENT_TYPES = frozenset({"audio/pcma"})

# Media types that carry no format information, so the filename decides.
_GENERIC_CONTENT_TYPES = frozenset({"", "application/octet-stream", "audio/*"})

# Extensions for headerless G.711: .ul/.al are what ffmpeg and sox use
# (`ffmpeg -h demuxer=mulaw`), .ulaw/.alaw are Asterisk's recording formats:
# https://docs.asterisk.org/Getting-Started/Installing-Asterisk/Installing-Asterisk-From-Source/Exploring-Sound-Prompts/
_MULAW_EXTENSIONS = frozenset({".ulaw", ".ul"})
_ALAW_EXTENSIONS = frozenset({".alaw", ".al"})


# Both tables follow the expansion formulas in G.711 section 4. The bit
# layout matches the widely copied Sun Microsystems g711.c, which CPython's
# audioop and sox also use:
# https://github.com/python/cpython/blob/3.12/Modules/audioop.c
def _build_mulaw_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.int32)
    inverted = ~codes & 0xFF
    mantissa = inverted & 0x0F
    exponent = (inverted & 0x70) >> 4
    magnitude = ((mantissa << 3) + 0x84) << exponent
    signed = np.where(inverted & 0x80, 0x84 - magnitude, magnitude - 0x84)
    return (signed.astype(np.float32) / 32768.0).astype(np.float32)


def _build_alaw_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.int32) ^ 0x55
    mantissa = (codes & 0x0F) << 4
    segment = (codes & 0x70) >> 4
    magnitude = np.where(
        segment == 0,
        mantissa + 8,
        (mantissa + 0x108) << np.maximum(segment - 1, 0),
    )
    signed = np.where(codes & 0x80, magnitude, -magnitude)
    return (signed.astype(np.float32) / 32768.0).astype(np.float32)


_TABLES: dict[str, np.ndarray] = {
    MULAW: _build_mulaw_table(),
    ALAW: _build_alaw_table(),
}


def decode_g711(data: bytes, encoding: str) -> np.ndarray:
    """Turn one G.711 byte per sample into float32 amplitudes in [-1, 1]."""
    try:
        table = _TABLES[encoding]
    except KeyError:
        raise ValueError(f"Unsupported G.711 encoding: {encoding!r}") from None
    codes = np.frombuffer(data, dtype=np.uint8)
    return table[codes]


def resolve_g711_encoding(
    content_type: str | None, filename: str | None = None
) -> str | None:
    """Work out whether a caller declared raw G.711 audio."""
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized in _MULAW_CONTENT_TYPES:
        return MULAW
    if normalized in _ALAW_CONTENT_TYPES:
        return ALAW
    if normalized not in _GENERIC_CONTENT_TYPES:
        return None

    name = (filename or "").strip().lower()
    dot = name.rfind(".")
    extension = name[dot:] if dot >= 0 else ""
    if extension in _MULAW_EXTENSIONS:
        return MULAW
    if extension in _ALAW_EXTENSIONS:
        return ALAW
    return None


def wrap_g711_as_wav(
    data: bytes, encoding: str, sample_rate: int = G711_SAMPLE_RATE
) -> bytes:
    """Put a WAV header in front of headerless G.711 bytes."""
    if is_riff_wav(data):
        return data
    try:
        fmt_tag = _WAV_FORMAT_TAGS[encoding]
    except KeyError:
        raise ValueError(f"Unsupported G.711 encoding: {encoding!r}") from None
    channels = 1
    bits_per_sample = 8
    block_align = channels * bits_per_sample // 8
    fmt_chunk = struct.pack(
        "<HHIIHHH",
        fmt_tag,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits_per_sample,
        0,
    )
    padding = b"\x00" if len(data) % 2 else b""
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_chunk))
        + fmt_chunk
        + b"data"
        + struct.pack("<I", len(data))
        + data
        + padding
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body
