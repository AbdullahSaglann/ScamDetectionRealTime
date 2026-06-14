"""Audio post-processing utilities for TTS output.

Applies simple audio enhancements to reduce metallic / robotic artifacts
commonly produced by neural TTS systems.  All filters use scipy so there
are no heavy external dependencies.

Usage:
    from scripts.tts.postprocess import postprocess_wav
    postprocess_wav("input.wav", "output.wav")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _highpass(audio: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    """Remove low-frequency rumble below *cutoff* Hz."""
    from scipy.signal import butter, sosfilt

    sos = butter(4, cutoff, btype="high", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _warmth_eq(audio: np.ndarray, sr: int, gain_db: float = 2.5) -> np.ndarray:
    """Boost 200-400 Hz band slightly to add vocal warmth."""
    from scipy.signal import butter, sosfilt

    sos = butter(2, [200, 400], btype="band", fs=sr, output="sos")
    band = sosfilt(sos, audio)
    gain = 10.0 ** (gain_db / 20.0)
    return (audio + band * (gain - 1.0)).astype(np.float32)


def _demetallic(audio: np.ndarray, sr: int, cut_db: float = -3.0) -> np.ndarray:
    """Attenuate 3-5 kHz to reduce metallic artifacts."""
    from scipy.signal import butter, sosfilt

    sos = butter(2, [3000, 5000], btype="band", fs=sr, output="sos")
    band = sosfilt(sos, audio)
    gain = 10.0 ** (cut_db / 20.0)
    # Subtract the difference to attenuate the band
    return (audio - band * (1.0 - gain)).astype(np.float32)


def _normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Peak-normalize audio to *target_db* dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_amp = 10.0 ** (target_db / 20.0)
    return (audio * (target_amp / peak)).astype(np.float32)


def _soft_compress(
    audio: np.ndarray,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
) -> np.ndarray:
    """Simple soft-knee compressor for more even dynamics."""
    threshold = 10.0 ** (threshold_db / 20.0)
    out = audio.copy()
    mask = np.abs(out) > threshold
    above = np.abs(out[mask])
    compressed = threshold + (above - threshold) / ratio
    out[mask] = np.sign(out[mask]) * compressed
    return out.astype(np.float32)


def postprocess_audio(
    audio: np.ndarray,
    sr: int,
    *,
    highpass: bool = True,
    warmth: bool = True,
    demetallic: bool = True,
    compress: bool = True,
    normalize: bool = True,
    highpass_cutoff: float = 80.0,
    warmth_gain_db: float = 2.5,
    demetallic_cut_db: float = -3.0,
    compress_threshold_db: float = -18.0,
    compress_ratio: float = 3.0,
    normalize_target_db: float = -1.0,
) -> np.ndarray:
    """Apply a chain of audio post-processing to a float32 ndarray."""
    if highpass:
        audio = _highpass(audio, sr, highpass_cutoff)
    if warmth:
        audio = _warmth_eq(audio, sr, warmth_gain_db)
    if demetallic:
        audio = _demetallic(audio, sr, demetallic_cut_db)
    if compress:
        audio = _soft_compress(audio, compress_threshold_db, compress_ratio)
    if normalize:
        audio = _normalize_peak(audio, normalize_target_db)
    return audio


def postprocess_wav(
    input_path: str | Path,
    output_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """Read a WAV file, apply post-processing, and write the result.

    If *output_path* is ``None`` the original file is overwritten.
    """
    import soundfile as sf

    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    audio, sr = sf.read(str(input_path), dtype="float32")
    processed = postprocess_audio(audio, sr, **kwargs)
    sf.write(str(output_path), processed, sr)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process TTS WAV output.")
    parser.add_argument("input", help="Input WAV file.")
    parser.add_argument("-o", "--output", help="Output WAV file (default: overwrite input).")
    parser.add_argument("--no-highpass", action="store_true")
    parser.add_argument("--no-warmth", action="store_true")
    parser.add_argument("--no-demetallic", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    result = postprocess_wav(
        args.input,
        args.output,
        highpass=not args.no_highpass,
        warmth=not args.no_warmth,
        demetallic=not args.no_demetallic,
        compress=not args.no_compress,
        normalize=not args.no_normalize,
    )
    print(f"Post-processed: {result}")
