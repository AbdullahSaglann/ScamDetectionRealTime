"""TTS engine adapters."""

from __future__ import annotations

from typing import Any

from .base import TTSEngine


AVAILABLE_ENGINES = ("coqui_xtts", "fish_speech", "chatterbox", "f5_tts", "edge_openvoice", "edge_tts")

def get_engine(name: str, **kwargs: Any) -> TTSEngine:
    """Factory: return the engine adapter matching *name*."""
    if name == "edge_tts":
        from .edge_tts_engine import EdgeTTSEngine
        return EdgeTTSEngine(**kwargs)
    if name == "edge_openvoice":
        from .edge_openvoice import EdgeTTSOpenVoice
        return EdgeTTSOpenVoice(**kwargs)
    if name == "f5_tts":
        from .f5_tts_engine import F5TTS
        return F5TTS(**kwargs)
    if name == "coqui_xtts":
        from .coqui_xtts import CoquiXTTS
        return CoquiXTTS(**kwargs)
    if name == "fish_speech":
        from .fish_speech import FishSpeech
        return FishSpeech(**kwargs)
    if name == "chatterbox":
        from .chatterbox import Chatterbox
        return Chatterbox(**kwargs)
    raise ValueError(
        f"Unknown engine {name!r}. Available: {', '.join(AVAILABLE_ENGINES)}"
    )
