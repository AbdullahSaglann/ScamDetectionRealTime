"""Shared types for TTS engine adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeechTurn:
    row_id: int
    turn_index: int
    speaker: str
    text: str
    role: str
    style: str
    emotion: str
    rate: str
    pitch: str


@dataclass(frozen=True)
class SynthesisResult:
    row_id: int
    turn_index: int
    speaker: str
    output_path: Path
    engine: str


class TTSEngine:
    name = "base"

    def synthesize_turn(self, turn: SpeechTurn, output_path: Path) -> SynthesisResult:
        raise NotImplementedError

