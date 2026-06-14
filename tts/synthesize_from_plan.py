#!/usr/bin/env python3
"""Synthesize audio turn files from a JSONL speech plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engines import AVAILABLE_ENGINES, get_engine
from engines.base import SpeechTurn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TTS engine over JSONL speech plans.")
    parser.add_argument("--plan", required=True, help="Input JSONL plan file.")
    parser.add_argument("--output-dir", default="outputs/tts_audio", help="Audio output folder.")
    parser.add_argument("--engine", default="coqui_xtts", choices=list(AVAILABLE_ENGINES))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or another torch device.")
    parser.add_argument("--language", default="tr", help="XTTS language code.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum dialogue plans to synthesize.")
    parser.add_argument(
        "--max-turns-per-plan",
        type=int,
        help="Maximum turns to synthesize from each dialogue plan.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without loading TTS.")
    parser.add_argument("--default-speaker-wav", help="Reference WAV used for both speakers.")
    parser.add_argument("--agent-speaker-wav", help="Reference WAV for agent turns.")
    parser.add_argument("--victim-speaker-wav", help="Reference WAV for victim turns.")
    parser.add_argument(
        "--speaker-map",
        help=(
            "JSON file mapping speaker/emotion/style/role to reference WAVs. "
            "Example: {'agent': {'pressuring': 'agent_urgent.wav'}, 'victim': {'worried': 'victim_worried.wav'}}"
        ),
    )
    parser.add_argument(
        "--tts-kwargs",
        help='Extra JSON kwargs passed to Coqui TTS, for example {"temperature":0.75,"speed":0.92}.',
    )
    parser.add_argument("--tts-kwargs-file", help="JSON file with extra kwargs passed to TTS engine.")
    parser.add_argument("--speaker", help="Coqui preset speaker name, if no speaker_wav is used.")
    parser.add_argument(
        "--agree-coqui-cpml",
        action="store_true",
        help="Confirm Coqui XTTS CPML/non-commercial terms or commercial-license coverage.",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply audio post-processing (EQ, compression) to reduce robotic artifacts.",
    )
    return parser.parse_args()


def iter_plans(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def make_turn(plan: dict, turn: dict) -> SpeechTurn:
    return SpeechTurn(
        row_id=int(plan["row_id"]),
        turn_index=int(turn["turn_index"]),
        speaker=str(turn["speaker"]),
        text=str(turn["text"]),
        role=str(turn.get("role", "")),
        style=str(turn.get("style", "")),
        emotion=str(turn.get("emotion", "")),
        rate=str(turn.get("rate", "")),
        pitch=str(turn.get("pitch", "")),
    )


def build_engine(args: argparse.Namespace):
    speaker_map = None
    if args.speaker_map:
        speaker_map = json.loads(Path(args.speaker_map).read_text(encoding="utf-8"))
    tts_kwargs = None
    if args.tts_kwargs_file:
        tts_kwargs = json.loads(Path(args.tts_kwargs_file).read_text(encoding="utf-8"))
    elif args.tts_kwargs:
        tts_kwargs = json.loads(args.tts_kwargs)

    # Common kwargs shared by all engines
    common_kwargs = dict(
        device=args.device,
        language=args.language,
        default_speaker_wav=args.default_speaker_wav,
        agent_speaker_wav=args.agent_speaker_wav,
        victim_speaker_wav=args.victim_speaker_wav,
        speaker_map=speaker_map,
        tts_kwargs=tts_kwargs,
    )

    # Engine-specific kwargs
    if args.engine == "coqui_xtts":
        common_kwargs["speaker"] = args.speaker
        common_kwargs["agree_cpml"] = args.agree_coqui_cpml

    return get_engine(args.engine, **common_kwargs)


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan)
    output_dir = Path(args.output_dir)

    try:
        engine = None if args.dry_run else build_engine(args)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    manifest = []

    for plan in iter_plans(plan_path, args.limit):
        row_id = int(plan["row_id"])
        dialogue_dir = output_dir / f"row_{row_id:05d}"
        dialogue_dir.mkdir(parents=True, exist_ok=True)
        turns = plan["turns"]
        if args.max_turns_per_plan is not None:
            turns = turns[: args.max_turns_per_plan]

        for raw_turn in turns:
            turn = make_turn(plan, raw_turn)
            output_path = dialogue_dir / f"turn_{turn.turn_index:02d}_{turn.speaker}.wav"

            if output_path.exists():
                print(f"Skipping {output_path} (already exists)")
                continue

            if args.dry_run:
                print(
                    f"{args.engine}: row={turn.row_id} turn={turn.turn_index} "
                    f"speaker={turn.speaker} emotion={turn.emotion} style={turn.style} "
                    f"text={turn.text!r} -> {output_path}"
                )
                manifest.append({"row_id": row_id, "turn_index": turn.turn_index, "path": str(output_path)})
                continue

            result = engine.synthesize_turn(turn, output_path)

            # Optional post-processing to reduce robotic artifacts
            if args.postprocess and result.output_path.exists():
                from postprocess import postprocess_wav
                postprocess_wav(result.output_path)
                print(f"Wrote {result.output_path} (post-processed)")
            else:
                print(f"Wrote {result.output_path}")

            manifest.append(
                {
                    "row_id": result.row_id,
                    "turn_index": result.turn_index,
                    "speaker": result.speaker,
                    "engine": result.engine,
                    "path": str(result.output_path),
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
