"""F5-TTS engine adapter.

This adapter wraps the modern F5-TTS (Flow Matching) model, which
provides excellent zero-shot voice cloning and is highly stable in Colab.

Install:
    pip install git+https://github.com/SWivid/F5-TTS.git
    pip install soundfile numpy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import SpeechTurn, SynthesisResult, TTSEngine


class F5TTS(TTSEngine):
    name = "f5_tts"

    def __init__(
        self,
        *,
        device: str = "cuda",
        language: str = "tr",
        agent_speaker_wav: str | None = None,
        victim_speaker_wav: str | None = None,
        default_speaker_wav: str | None = None,
        speaker_map: dict[str, Any] | None = None,
        tts_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not any((agent_speaker_wav, victim_speaker_wav, default_speaker_wav, speaker_map)):
            raise RuntimeError(
                "F5-TTS needs a reference WAV for voice cloning."
            )

        self.speaker_map = self._normalize_speaker_map(speaker_map or {})
        self.language = language
        self.device = device
        self.agent_speaker_wav = agent_speaker_wav or default_speaker_wav
        self.victim_speaker_wav = victim_speaker_wav or default_speaker_wav
        self.tts_kwargs = tts_kwargs or {}

        # Lazy model loading
        self._model_obj = None
        self._vocoder = None

    def _ensure_loaded(self) -> None:
        if self._model_obj is not None:
            return

        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise RuntimeError(
                "F5-TTS is not installed. Install with:\n"
                "  pip install git+https://github.com/SWivid/F5-TTS.git"
            ) from exc

        # Load F5-TTS using high-level API
        print("Loading F5-TTS model via High-Level API...")
        self._model_obj = F5TTS(device=self.device)
        print("F5-TTS loaded successfully.")

    def synthesize_turn(self, turn: SpeechTurn, output_path: Path) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_loaded()

        speaker_wav = self._speaker_wav_for(turn)
        ref_text = "" # F5-TTS automatically analyzes the reference audio

        # Sentez (high-level API returns wav, sr, spec)
        audio, sample_rate, _ = self._model_obj.infer(
            ref_file=speaker_wav,
            ref_text=ref_text,
            gen_text=turn.text,
        )

        import soundfile as sf
        sf.write(str(output_path), audio, sample_rate)

        return SynthesisResult(
            row_id=turn.row_id,
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            output_path=output_path,
            engine=self.name,
        )

    def _speaker_wav_for(self, turn: SpeechTurn) -> str | None:
        mapped = self._mapped_speaker_wav_for(turn)
        if mapped:
            return mapped
        if turn.speaker == "agent":
            return self.agent_speaker_wav
        if turn.speaker == "victim":
            return self.victim_speaker_wav
        return self.agent_speaker_wav or self.victim_speaker_wav

    def _mapped_speaker_wav_for(self, turn: SpeechTurn) -> str | None:
        value = self.speaker_map.get(turn.speaker)
        if isinstance(value, str):
            return value
        if not isinstance(value, dict):
            return None
        for key in (turn.emotion, turn.style, turn.role, "default"):
            if key and key in value:
                return str(value[key])
        return None

    @staticmethod
    def _normalize_speaker_map(raw_map: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for speaker, value in raw_map.items():
            if isinstance(value, str):
                normalized[speaker] = str(value)
            if isinstance(value, dict):
                normalized[speaker] = {k: str(v) for k, v in value.items()}
        return normalized
