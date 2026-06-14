"""Fish Speech TTS adapter.

This adapter wraps the Fish Speech (fish-speech) model for high-quality
multilingual voice synthesis with zero-shot voice cloning.  Fish Speech
uses a DualAR transformer architecture and consistently ranks at the top
of TTS quality benchmarks.

Install:
    pip install -r scripts/tts/requirements-fish-speech.txt
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .base import SpeechTurn, SynthesisResult, TTSEngine


class FishSpeech(TTSEngine):
    name = "fish_speech"

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
                "Fish Speech needs a reference WAV for voice cloning. "
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

        # Lazy-load model on first synthesis call to keep import lightweight
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the Fish Speech model."""
        if self._model is not None:
            return

        try:
            from fish_speech.models import TTSModel
        except ImportError:
            # Try alternative import paths for different fish-speech versions
            try:
                import fish_speech
                self._use_cli = True
                return
            except ImportError as exc:
                raise RuntimeError(
                    "Fish Speech is not installed. Install with:\n"
                    "  pip install fish-speech\n"
                    "Or follow: https://github.com/fishaudio/fish-speech"
                ) from exc

        self._model = TTSModel.from_pretrained(device=self.device)
        self._use_cli = False

    def synthesize_turn(self, turn: SpeechTurn, output_path: Path) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_loaded()

        speaker_wav = self._speaker_wav_for(turn)

        if getattr(self, "_use_cli", False):
            self._synthesize_via_cli(turn.text, speaker_wav, output_path)
        else:
            self._synthesize_via_api(turn.text, speaker_wav, output_path)

        return SynthesisResult(
            row_id=turn.row_id,
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            output_path=output_path,
            engine=self.name,
        )

    def _synthesize_via_api(self, text: str, speaker_wav: str | None, output_path: Path) -> None:
        """Synthesize using the Fish Speech Python API."""
        import soundfile as sf

        kwargs = {
            "text": text,
            "reference_audio": speaker_wav,
            "language": self.language,
        }
        kwargs.update(self.tts_kwargs)

        result = self._model.synthesize(**kwargs)
        audio = np.array(result.audio, dtype=np.float32)
        sample_rate = getattr(result, "sample_rate", 44100)
        sf.write(str(output_path), audio, sample_rate)

    def _synthesize_via_cli(self, text: str, speaker_wav: str | None, output_path: Path) -> None:
        """Fallback: synthesize using the Fish Speech CLI tools."""
        import subprocess

        cmd = [
            "python", "-m", "fish_speech.inference",
            "--text", text,
            "--output", str(output_path),
        ]
        if speaker_wav:
            cmd.extend(["--reference-audio", speaker_wav])

        extra = self.tts_kwargs
        if "temperature" in extra:
            cmd.extend(["--temperature", str(extra["temperature"])])
        if "top_p" in extra:
            cmd.extend(["--top-p", str(extra["top_p"])])

        subprocess.run(cmd, check=True, capture_output=True)

    # --- Speaker WAV resolution (same logic as Coqui adapter) ---

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
