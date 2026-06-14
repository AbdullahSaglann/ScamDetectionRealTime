"""Pure Edge-TTS engine adapter.

Generates high-quality Turkish audio using Microsoft Azure neural voices.
No voice cloning, no GPU required. Super fast and reliable.

Install:
    pip install edge-tts
"""

from __future__ import annotations

import asyncio
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


class EdgeTTSEngine(TTSEngine):
    name = "edge_tts"

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
        self.language = language
        self.tts_kwargs = tts_kwargs or {}

    def synthesize_turn(self, turn: SpeechTurn, output_path: Path) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        voice = self._voice_for(turn)
        ssml_params = EMOTION_SSML.get(turn.emotion, EMOTION_SSML["neutral"])
        
        asyncio.run(self._edge_synthesize(turn.text, voice, ssml_params, str(output_path)))

        return SynthesisResult(
            row_id=turn.row_id,
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            output_path=output_path,
            engine=self.name,
        )

    async def _edge_synthesize(self, text: str, voice: str, ssml_params: dict, output_path: str) -> None:
        import edge_tts
        import time
        from aiohttp.client_exceptions import ClientError
        
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate=ssml_params.get("rate", "+0%"),
                    pitch=ssml_params.get("pitch", "+0Hz"),
                    volume=ssml_params.get("volume", "+0%"),
                )
                await communicate.save(output_path)
                # Başarılı olursa çok ufak bir bekleme ekle (Rate limit yememek için)
                time.sleep(0.5)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Edge-TTS Bağlantı Hatası (503/429). {base_delay} saniye bekleniyor... (Deneme {attempt+1}/{max_retries})")
                time.sleep(base_delay)
                base_delay *= 2  # Exponential backoff (2, 4, 8, 16 saniye)

    def _voice_for(self, turn: SpeechTurn) -> str:
        # Default: agent = Ahmet, victim = Emel
        if turn.speaker == "victim":
            return self.tts_kwargs.get("victim_voice", TURKISH_VOICES["victim_female"])
        return self.tts_kwargs.get("agent_voice", TURKISH_VOICES["agent_male"])

