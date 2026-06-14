#!/usr/bin/env python3
"""Merge separate TTS turn audio files into full dialogue recordings."""

import os
import glob
import soundfile as sf
import numpy as np
import argparse

def merge_dialogues(input_dir, output_dir, pause_ms=500):
    os.makedirs(output_dir, exist_ok=True)
    
    row_dirs = glob.glob(os.path.join(input_dir, "row_*"))
    if not row_dirs:
        print(f"No row directories found in {input_dir}")
        return

    processed_count = 0
    for row_dir in row_dirs:
        row_name = os.path.basename(row_dir)
        turn_files = sorted(glob.glob(os.path.join(row_dir, "turn_*.wav")))
        
        if not turn_files:
            continue
            
        print(f"Processing {row_name} ({len(turn_files)} turns)...")
        
        combined_audio = []
        sample_rate = None
        
        for i, f in enumerate(turn_files):
            audio, sr = sf.read(f, dtype="float32")
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                print(f"Warning: Sample rate mismatch in {f}. Expected {sample_rate}, got {sr}. Skipping row.")
                combined_audio = []
                break
                
            combined_audio.append(audio)
            
            # Add a small natural pause after each turn (except the last one)
            if i < len(turn_files) - 1:
                pause_samples = int(sample_rate * (pause_ms / 1000.0))
                if len(audio.shape) > 1: # Stereo
                    silence = np.zeros((pause_samples, audio.shape[1]), dtype="float32")
                else: # Mono
                    silence = np.zeros(pause_samples, dtype="float32")
                combined_audio.append(silence)
                
        if combined_audio:
            final_audio = np.concatenate(combined_audio, axis=0)
            out_file = os.path.join(output_dir, f"{row_name}_full.wav")
            sf.write(out_file, final_audio, sample_rate)
            processed_count += 1
            
    print(f"\nDone! Successfully merged {processed_count} dialogues into {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge separate TTS turns into a single dialogue audio file.")
    parser.add_argument("--input", required=True, help="Input directory containing row_* folders")
    parser.add_argument("--output", required=True, help="Output directory to save merged WAVs")
    parser.add_argument("--pause", type=int, default=500, help="Pause between turns in milliseconds (default: 500)")
    
    args = parser.parse_args()
    merge_dialogues(args.input, args.output, args.pause)
