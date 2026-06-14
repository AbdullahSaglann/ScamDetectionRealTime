"""Chatterbox (Resemble AI) TTS adapter.

This adapter wraps the Chatterbox model for high-quality multilingual
voice synthesis with zero-shot voice cloning and emotion control.
Chatterbox supports 23 languages including Turkish and provides
emotion exaggeration control.

Install:
    pip install -r scripts/tts/requirements-chatterbox.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import SpeechTurn, SynthesisResult, TTSEngine


# Emotion → exaggeration mapping for Chatterbox's emotion control.
# Higher values = more intense emotion.
EMOTION_EXAGGERATION = {
    "neutral": 0.3,
    "calm": 0.3,
    "friendly": 0.5,
    "cautious": 0.4,
    "worried": 0.6,
    "urgent": 0.7,
    "persuasive": 0.6,
    "threatening": 0.8,
    "pressuring": 0.7,
}


class Chatterbox(TTSEngine):
    name = "chatterbox"

    def __init__(
        self,
        *,
        device: str = "cpu",
        language: str = "tr",
        agent_speaker_wav: str | None = None,
        victim_speaker_wav: str | None = None,
        default_speaker_wav: str | None = None,
        speaker_map: dict[str, Any] | None = None,
        tts_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not any((agent_speaker_wav, victim_speaker_wav, default_speaker_wav, speaker_map)):
            raise RuntimeError(
                "Chatterbox needs a reference WAV for voice cloning. "
                "Pass --default-speaker-wav, --agent-speaker-wav, --victim-speaker-wav, "
                "or --speaker-map."
            )
        for wav_path in (agent_speaker_wav, victim_speaker_wav, default_speaker_wav):
            if wav_path and not Path(wav_path).exists():
                raise RuntimeError(f"Reference WAV not found: {wav_path}")

        self.speaker_map = self._normalize_speaker_map(speaker_map or {})
        self.language = language
        self.device = device
        self.agent_speaker_wav = agent_speaker_wav or default_speaker_wav
        self.victim_speaker_wav = victim_speaker_wav or default_speaker_wav
        self.tts_kwargs = tts_kwargs or {}

        # Lazy-load model
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the Chatterbox model."""
        if self._model is not None:
            return

        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise RuntimeError(
                "Chatterbox is not installed. Install with:\n"
                "  pip install chatterbox-tts\n"
                "Or follow: https://github.com/resemble-ai/chatterbox"
            ) from exc

        self._model = ChatterboxTTS.from_pretrained(device=self.device)

    def synthesize_turn(self, turn: SpeechTurn, output_path: Path) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_loaded()

        speaker_wav = self._speaker_wav_for(turn)

        # Determine emotion exaggeration from turn metadata
        exaggeration = EMOTION_EXAGGERATION.get(turn.emotion, 0.5)
        if "exaggeration" in self.tts_kwargs:
            exaggeration = self.tts_kwargs["exaggeration"]

        # Build synthesis kwargs
        kwargs = {
            "text": turn.text,
            "audio_prompt_path": speaker_wav,
            "exaggeration": exaggeration,
        }

        # Apply any extra kwargs (temperature, cfg_weight, etc.)
        for k, v in self.tts_kwargs.items():
            if k not in ("exaggeration",):
                kwargs[k] = v

        wav = self._model.generate(**kwargs)

        # Save output
        import torchaudio

        # Chatterbox returns a tensor; ensure 2D shape [1, samples]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        sample_rate = self._model.sr
        torchaudio.save(str(output_path), wav.cpu(), sample_rate)

        return SynthesisResult(
            row_id=turn.row_id,
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            output_path=output_path,
            engine=self.name,
        )

    # --- Speaker WAV resolution (same logic as other adapters) ---

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
                path = Path(value)
                if not path.exists():
                    raise RuntimeError(f"Reference WAV not found for {speaker}: {value}")
                normalized[speaker] = str(path)
                continue
            if isinstance(value, dict):
                normalized[speaker] = {}
                for key, wav_path in value.items():
                    path = Path(str(wav_path))
                    if not path.exists():
                        raise RuntimeError(
                            f"Reference WAV not found for {speaker}.{key}: {wav_path}"
                        )
                    normalized[speaker][key] = str(path)
        return normalized
