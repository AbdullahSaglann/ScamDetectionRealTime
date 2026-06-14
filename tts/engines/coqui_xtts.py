"""Coqui/XTTS adapter.

This adapter intentionally keeps Coqui as an optional dependency. Install it
only when running audio synthesis experiments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import SpeechTurn, SynthesisResult, TTSEngine


DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


class CoquiXTTS(TTSEngine):
    name = "coqui_xtts"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        language: str = "tr",
        agent_speaker_wav: str | None = None,
        victim_speaker_wav: str | None = None,
        default_speaker_wav: str | None = None,
        speaker_map: dict[str, Any] | None = None,
        speaker: str | None = None,
        tts_kwargs: dict[str, Any] | None = None,
        agree_cpml: bool = False,
    ) -> None:
        if not any((agent_speaker_wav, victim_speaker_wav, default_speaker_wav, speaker_map, speaker)):
            raise RuntimeError(
                "XTTS needs either a consented reference WAV "
                "(--default-speaker-wav, --agent-speaker-wav, --victim-speaker-wav) "
                "a --speaker-map, or a valid Coqui preset --speaker."
            )

        for wav_path in (agent_speaker_wav, victim_speaker_wav, default_speaker_wav):
            if wav_path and not Path(wav_path).exists():
                raise RuntimeError(f"Reference WAV not found: {wav_path}")
        self.speaker_map = self._normalize_speaker_map(speaker_map or {})

        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError(
                "Coqui TTS is not installed. Install the maintained fork with: "
                "pip install coqui-tts"
            ) from exc

        if agree_cpml:
            os.environ["COQUI_TOS_AGREED"] = "1"

        self.language = language
        self.agent_speaker_wav = agent_speaker_wav or default_speaker_wav
        self.victim_speaker_wav = victim_speaker_wav or default_speaker_wav
        self.speaker = speaker
        self.tts_kwargs = tts_kwargs or {}
        self.model = TTS(model_name).to(device)

    def synthesize_turn(self, turn: SpeechTurn, output_path: Path) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "text": turn.text,
            "file_path": str(output_path),
            "language": self.language,
            "split_sentences": True,
        }

        speaker_wav = self._speaker_wav_for(turn)
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif self.speaker:
            kwargs["speaker"] = self.speaker

        kwargs.update(self.tts_kwargs)
        self.model.tts_to_file(**kwargs)
        return SynthesisResult(
            row_id=turn.row_id,
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            output_path=output_path,
            engine=self.name,
        )

    def _speaker_wav_for(self, turn: SpeechTurn) -> str | None:
        speaker = turn.speaker
        mapped = self._mapped_speaker_wav_for(turn)
        if mapped:
            return mapped
        if speaker == "agent":
            return self.agent_speaker_wav
        if speaker == "victim":
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
                        raise RuntimeError(f"Reference WAV not found for {speaker}.{key}: {wav_path}")
                    normalized[speaker][key] = str(path)
        return normalized
