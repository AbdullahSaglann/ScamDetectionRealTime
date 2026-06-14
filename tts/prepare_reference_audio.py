#!/usr/bin/env python3
"""Convert consented reference recordings to XTTS-friendly WAV files."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


SUPPORTED_INPUTS = {".mp4", ".m4a", ".mp3", ".wav", ".flac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reference voice WAV files.")
    parser.add_argument("--input-dir", default="data/test_audio", help="Folder with source recordings.")
    parser.add_argument(
        "--output-dir",
        default="data/reference_voice",
        help="Folder for converted 24 kHz mono WAV files.",
    )
    parser.add_argument("--sample-rate", type=int, default=24000, help="Output WAV sample rate.")
    return parser.parse_args()


def collect_inputs(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUTS
    )


def convert_to_wav(source: Path, output_dir: Path, sample_rate: int) -> Path:
    output_path = output_dir / f"{source.stem}.wav"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")

    sources = collect_inputs(input_dir)
    if not sources:
        raise SystemExit(f"No supported audio/video files found in: {input_dir}")

    for source in sources:
        output_path = convert_to_wav(source, output_dir, args.sample_rate)
        print(f"{source} -> {output_path}")


if __name__ == "__main__":
    main()
