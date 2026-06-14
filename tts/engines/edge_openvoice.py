"""Edge-TTS + OpenVoice v2 engine adapter.

Pipeline:
  1. Edge-TTS  → high-quality Turkish synthesis (Microsoft Azure neural voices)
  2. OpenVoice v2 → zero-shot voice cloning (transfers reference voice timbre)

This combination provides the best Turkish TTS quality available
while maintaining voice identity from reference recordings.

Install:
    pip install edge-tts openvoice-cli soundfile
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from .base import SpeechTurn, SynthesisResult, TTSEngine


# Edge-TTS Turkish voice presets
TURKISH_VOICES = {
    "agent_male": "tr-TR-AhmetNeural",
    "agent_female": "tr-TR-EmelNeural",
    "victim_male": "tr-TR-AhmetNeural",
    "victim_female": "tr-TR-EmelNeural",
}

# Emotion → SSML prosody mapping
EMOTION_SSML = {
    "neutral":     {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
    "friendly":    {"rate": "+5%",  "pitch": "+5Hz",  "volume": "+0%"},
    "worried":     {"rate": "-5%",  "pitch": "+10Hz", "volume": "-5%"},
    "urgent":      {"rate": "+15%", "pitch": "+5Hz",  "volume": "+10%"},
    "threatening": {"rate": "-10%", "pitch": "-10Hz", "volume": "+15%"},
    "persuasive":  {"rate": "+5%",  "pitch": "+0Hz",  "volume": "+5%"},
    "cautious":    {"rate": "-10%", "pitch": "+5Hz",  "volume": "-5%"},
}


class EdgeTTSOpenVoice(TTSEngine):
    name = "edge_openvoice"

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
        self.speaker_map = self._normalize_speaker_map(speaker_map or {})
        self.language = language
        self.device = device
        self.agent_speaker_wav = agent_speaker_wav or default_speaker_wav
        self.victim_speaker_wav = victim_speaker_wav or default_speaker_wav
        self.tts_kwargs = tts_kwargs or {}

        self._openvoice_model = None

    def _ensure_loaded(self) -> None:
        if self._openvoice_model is not None:
            return

        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
        except ImportError as exc:
            raise RuntimeError(
                "OpenVoice is not installed. Install with:\n"
                "  pip install openvoice-cli"
            ) from exc

        from openvoice.api import ToneColorConverter
        import torch

        ckpt_path = self.tts_kwargs.get("openvoice_ckpt", None)
        self._openvoice_model = ToneColorConverter(ckpt_path, device=self.device)
        print("OpenVoice v2 loaded successfully.")

    def synthesize_turn(self, turn: SpeechTurn, output_path: Path) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Edge-TTS synthesis
        edge_path = output_path.with_suffix(".edge.wav")
        voice = self._voice_for(turn)
        ssml_params = EMOTION_SSML.get(turn.emotion, EMOTION_SSML["neutral"])
        asyncio.run(self._edge_synthesize(turn.text, voice, ssml_params, str(edge_path)))

        # Step 2: OpenVoice voice cloning (optional)
        speaker_wav = self._speaker_wav_for(turn)
        if speaker_wav and self._openvoice_model is not None:
            self._clone_voice(str(edge_path), speaker_wav, str(output_path))
        else:
            import shutil
            shutil.copy2(str(edge_path), str(output_path))

        return SynthesisResult(
            row_id=turn.row_id,
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            output_path=output_path,
            engine=self.name,
        )

    async def _edge_synthesize(self, text: str, voice: str, ssml_params: dict, output_path: str) -> None:
        import edge_tts
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=ssml_params.get("rate", "+0%"),
            pitch=ssml_params.get("pitch", "+0%"),
            volume=ssml_params.get("volume", "+0%"),
        )
        await communicate.save(output_path)

    def _clone_voice(self, source_path: str, reference_path: str, output_path: str) -> None:
        from openvoice import se_extractor
        source_se = se_extractor.get_se(source_path, self._openvoice_model, vad=False)
        target_se = se_extractor.get_se(reference_path, self._openvoice_model, vad=True)
        self._openvoice_model.convert(
            audio_src_path=source_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
        )

    def _voice_for(self, turn: SpeechTurn) -> str:
        if turn.speaker == "victim":
            return self.tts_kwargs.get("victim_voice", TURKISH_VOICES["victim_female"])
        return self.tts_kwargs.get("agent_voice", TURKISH_VOICES["agent_male"])

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
