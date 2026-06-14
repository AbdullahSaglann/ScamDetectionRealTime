#!/usr/bin/env python3
"""
TTS Engine Comparison — Colab Test Script
==========================================

This script is designed to be run on Google Colab.
It allows you to test and compare three different TTS engines (Coqui XTTS v2, Fish Speech, Chatterbox) using the same sentences and reference voices.

Usage:
    1. Upload this file and your project folder to Google Drive
    2. Open in Colab, Runtime > Change runtime type > GPU (T4 or A100)
    3. Run the cells sequentially

Note: This file is saved as .py. To use in Colab, you can copy the cells below to a Colab notebook or run it directly as a script.
"""

# ===========================================================================
# CELL 1: Google Drive connection and project setup
# ===========================================================================

SETUP_CELL = """
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set project folder - change according to location on Drive
import os
PROJECT_DIR = "/content/drive/MyDrive/BITIRME"
# Alternative: upload project directly to Colab
# PROJECT_DIR = "/content/BITIRME"

os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}")
print(f"GPU check:")
!nvidia-smi
"""

# ===========================================================================
# CELL 2: Install dependencies
# ===========================================================================

INSTALL_CELL = """
# --- Common dependencies ---
!pip install -q soundfile scipy numpy torchaudio

# --- Engine 1: Coqui XTTS v2 (current engine) ---
!pip install -q coqui-tts

# --- Engine 2: Chatterbox (Resemble AI) ---
!pip install -q chatterbox-tts

# --- Engine 3: Fish Speech ---
!pip install -q fish-speech

print("\\n✅ All dependencies installed!")
"""

# ===========================================================================
# CELL 3: Prepare test sentences and reference voices
# ===========================================================================

TEST_SETUP = """
import os
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from IPython.display import Audio, display, HTML

PROJECT_DIR = os.getcwd()

# Test sentences - different emotions and situations
TEST_SENTENCES = {
    "neutral": "Merhaba, ben bankadan arıyorum. Hesabınızla ilgili bir bilgilendirme yapmak istiyorum.",
    "urgent": "Dikkat! Hesabınızda şüpheli bir işlem tespit ettik. Hemen doğrulama yapmanız gerekiyor.",
    "friendly": "İyi günler, size özel bir kampanyamız var. Birkaç dakikanızı alabilir miyim?",
    "worried": "Bir dakika, bu biraz garip geldi. Gerçekten bankadan mı arıyorsunuz?",
    "threatening": "Eğer beş dakika içinde bilgilerinizi doğrulamazsanız, hesabınız kalıcı olarak kapatılacak.",
}

# Reference voice files
AGENT_REF = os.path.join(PROJECT_DIR, "data/reference_voice/agent.wav")
VICTIM_REF = os.path.join(PROJECT_DIR, "data/reference_voice/victim.wav")

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs/tts_engine_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check reference voices
for ref_name, ref_path in [("Agent", AGENT_REF), ("Victim", VICTIM_REF)]:
    if os.path.exists(ref_path):
        info = sf.info(ref_path)
        print(f"✅ {ref_name}: {ref_path} ({info.duration:.1f}s, {info.samplerate}Hz)")
    else:
        print(f"❌ {ref_name}: {ref_path} — NOT FOUND!")

print(f"\\n📁 Output directory: {OUTPUT_DIR}")
print(f"📝 {len(TEST_SENTENCES)} test sentences ready")
"""

# ===========================================================================
# CELL 4: Engine 1 — Coqui XTTS v2 (Optimized)
# ===========================================================================

COQUI_CELL = """
print("=" * 60)
print("ENGINE 1: Coqui XTTS v2 (Optimized Parameters)")
print("=" * 60)

import os
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS
import torch
import soundfile as sf
import time

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
coqui_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Optimized parameters (old: only temperature=0.85)
COQUI_KWARGS = {
    "temperature": 0.72,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 0.95,
}

coqui_results = {}
for emotion, text in TEST_SENTENCES.items():
    print(f"\\n🔊 [{emotion}] {text[:50]}...")
    out_path = os.path.join(OUTPUT_DIR, f"coqui_{emotion}.wav")

    start = time.time()
    coqui_model.tts_to_file(
        text=text,
        file_path=out_path,
        speaker_wav=AGENT_REF if emotion != "worried" else VICTIM_REF,
        language="tr",
        split_sentences=True,
        **COQUI_KWARGS,
    )
    elapsed = time.time() - start

    info = sf.info(out_path)
    coqui_results[emotion] = {"path": out_path, "duration": info.duration, "time": elapsed}
    print(f"   ✅ {info.duration:.1f}s audio, took {elapsed:.1f}s")
    display(Audio(out_path))

print("\\n✅ Coqui XTTS completed!")
"""

# ===========================================================================
# CELL 5: Engine 2 — Chatterbox
# ===========================================================================

CHATTERBOX_CELL = """
print("=" * 60)
print("ENGINE 2: Chatterbox (Resemble AI)")
print("=" * 60)

from chatterbox.tts import ChatterboxTTS
import torchaudio
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
cb_model = ChatterboxTTS.from_pretrained(device=device)

# Exaggeration values by emotion
EMOTION_EXAGGERATION = {
    "neutral": 0.3,
    "urgent": 0.7,
    "friendly": 0.5,
    "worried": 0.6,
    "threatening": 0.8,
}

cb_results = {}
for emotion, text in TEST_SENTENCES.items():
    print(f"\\n🔊 [{emotion}] {text[:50]}...")
    out_path = os.path.join(OUTPUT_DIR, f"chatterbox_{emotion}.wav")
    ref_wav = AGENT_REF if emotion != "worried" else VICTIM_REF

    start = time.time()
    wav = cb_model.generate(
        text=text,
        audio_prompt_path=ref_wav,
        exaggeration=EMOTION_EXAGGERATION.get(emotion, 0.5),
    )
    elapsed = time.time() - start

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(out_path, wav.cpu(), cb_model.sr)

    info = sf.info(out_path)
    cb_results[emotion] = {"path": out_path, "duration": info.duration, "time": elapsed}
    print(f"   ✅ {info.duration:.1f}s audio, took {elapsed:.1f}s")
    display(Audio(out_path))

print("\\n✅ Chatterbox completed!")
"""

# ===========================================================================
# CELL 6: Engine 3 — Fish Speech
# ===========================================================================

FISH_CELL = """
print("=" * 60)
print("ENGINE 3: Fish Speech")
print("=" * 60)

import time

# Fish Speech installation method may vary by version.
# We try the two most common methods below.

fish_results = {}
fish_available = False

try:
    # Method 1: Python API
    from fish_speech.models import TTSModel
    fish_model = TTSModel.from_pretrained(device=device)

    for emotion, text in TEST_SENTENCES.items():
        print(f"\\n🔊 [{emotion}] {text[:50]}...")
        out_path = os.path.join(OUTPUT_DIR, f"fish_{emotion}.wav")
        ref_wav = AGENT_REF if emotion != "worried" else VICTIM_REF

        start = time.time()
        result = fish_model.synthesize(
            text=text,
            reference_audio=ref_wav,
            language="tr",
        )
        elapsed = time.time() - start

        audio = np.array(result.audio, dtype=np.float32)
        sr = getattr(result, "sample_rate", 44100)
        sf.write(out_path, audio, sr)

        info = sf.info(out_path)
        fish_results[emotion] = {"path": out_path, "duration": info.duration, "time": elapsed}
        print(f"   ✅ {info.duration:.1f}s audio, took {elapsed:.1f}s")
        display(Audio(out_path))

    fish_available = True
    print("\\n✅ Fish Speech completed!")

except ImportError:
    print("⚠️  Fish Speech Python API not found.")
    print("Try installing Fish Speech via CLI:")
    print("  !git clone https://github.com/fishaudio/fish-speech.git")
    print("  !cd fish-speech && pip install -e .")
except Exception as e:
    print(f"⚠️  Fish Speech error: {e}")
    print("Check Fish Speech installation.")
"""

# ===========================================================================
# CELL 7: Apply post-processing
# ===========================================================================

POSTPROCESS_CELL = """
print("=" * 60)
print("POST-PROCESSING: Robotic Voice Improvement")
print("=" * 60)

import sys
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts/tts"))
from postprocess import postprocess_wav

# Apply post-processing to outputs of each engine
for engine in ["coqui", "chatterbox", "fish"]:
    for emotion in TEST_SENTENCES:
        in_path = os.path.join(OUTPUT_DIR, f"{engine}_{emotion}.wav")
        out_path = os.path.join(OUTPUT_DIR, f"{engine}_{emotion}_pp.wav")
        if os.path.exists(in_path):
            postprocess_wav(in_path, out_path)
            print(f"✅ {engine}_{emotion} → post-processed")

print("\\n✅ Post-processing completed!")
print("You can listen to the files after post-processing:")
"""

# ===========================================================================
# CELL 8: Comparison table and side-by-side listening
# ===========================================================================

COMPARISON_CELL = """
print("=" * 60)
print("COMPARISON: All Engines Side by Side")
print("=" * 60)

from IPython.display import HTML, display, Audio

for emotion in TEST_SENTENCES:
    print(f"\\n{'='*40}")
    print(f"🎭 Emotion: {emotion.upper()}")
    print(f"📝 \\"{TEST_SENTENCES[emotion][:60]}...\\"")
    print(f"{'='*40}")

    for engine_name, results in [("Coqui XTTS v2", coqui_results),
                                  ("Chatterbox", cb_results),
                                  ("Fish Speech", fish_results)]:
        if emotion in results:
            r = results[emotion]
            print(f"\\n  🔊 {engine_name} ({r['duration']:.1f}s, {r['time']:.1f}s inference)")
            display(Audio(r["path"]))

            # Post-processed version
            pp_path = r["path"].replace(".wav", "_pp.wav")
            if os.path.exists(pp_path):
                print(f"  🔊 {engine_name} + Post-Processing")
                display(Audio(pp_path))

# Summary table
print("\\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Engine':<20} {'Avg. Inference (s)':<20} {'Avg. Audio Duration (s)':<20}")
print("-" * 60)

for engine_name, results in [("Coqui XTTS v2", coqui_results),
                              ("Chatterbox", cb_results),
                              ("Fish Speech", fish_results)]:
    if results:
        avg_time = np.mean([r["time"] for r in results.values()])
        avg_dur = np.mean([r["duration"] for r in results.values()])
        print(f"{engine_name:<20} {avg_time:<20.1f} {avg_dur:<20.1f}")

print()
print("Next steps:")
print("1. Listen to the audio above and determine the most natural one")
print("2. Integrate your chosen engine into the main pipeline")
print("3. Synthesize all dialogue plans with the chosen engine")
"""

# ===========================================================================
# CELL 9: Save results
# ===========================================================================

SAVE_CELL = """
# Save results to JSON
comparison = {
    "test_sentences": TEST_SENTENCES,
    "engines": {},
}

for engine_name, results in [("coqui_xtts_v2", coqui_results),
                              ("chatterbox", cb_results),
                              ("fish_speech", fish_results)]:
    if results:
        comparison["engines"][engine_name] = {
            emotion: {
                "path": r["path"],
                "duration_s": round(r["duration"], 2),
                "inference_s": round(r["time"], 2),
            }
            for emotion, r in results.items()
        }

result_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)

print(f"✅ Results saved: {result_path}")
"""


# ===========================================================================
# Main execution: Print all cells
# ===========================================================================

if __name__ == "__main__":
    cells = [
        ("1. Drive Connection", SETUP_CELL),
        ("2. Dependency Installation", INSTALL_CELL),
        ("3. Test Preparation", TEST_SETUP),
        ("4. Coqui XTTS v2", COQUI_CELL),
        ("5. Chatterbox", CHATTERBOX_CELL),
        ("6. Fish Speech", FISH_CELL),
        ("7. Post-Processing", POSTPROCESS_CELL),
        ("8. Comparison", COMPARISON_CELL),
        ("9. Save Results", SAVE_CELL),
    ]

    print("=" * 60)
    print("TTS Engine Comparison — Colab Notebook Content")
    print("=" * 60)
    print()
    print("Copy the cells below to a Colab notebook.")
    print("Each '# CELL' should be a separate Colab cell.")
    print()

    for title, cell in cells:
        print(f"\\n{'#' * 60}")
        print(f"# {title}")
        print(f"{'#' * 60}")
        print(cell)
