# TTS Agent Scripts

This folder contains the new model-agnostic Turkish emotional TTS agent work.

Legacy free-TTS audio generation scripts remain in `scripts/legacy/` only for
historical reference. They should not be used as training references.

## Available Engines

| Engine | Model | Voice Cloning | Turkish | Install |
|--------|-------|:---:|:---:|---------|
| **Coqui XTTS v2** | `tts_models/multilingual/multi-dataset/xtts_v2` | ✅ Zero-shot | ✅ | `pip install -r requirements-coqui.txt` |
| **Chatterbox** | Resemble AI Chatterbox Multilingual | ✅ Zero-shot + Emotion control | ✅ | `pip install -r requirements-chatterbox.txt` |
| **Fish Speech** | Fish Audio S2 Pro / DualAR | ✅ Zero-shot | ✅ | `pip install -r requirements-fish-speech.txt` |

## First Tool

`plan_dialogue_styles.py` reads dataset dialogues and writes structured speech
plans as JSONL.

Example:

```powershell
python scripts/tts/plan_dialogue_styles.py --input data/final/DataSetV4.csv --limit 20 --output outputs/tts_plans/sample_plans.jsonl
```

The JSONL output can later be consumed by any engine adapter via the `--engine`
flag.

## Quick Start

Prepare MP4 reference recordings as WAV:

```powershell
python scripts/tts/prepare_reference_audio.py --input-dir data/test_audio --output-dir data/reference_voice
```

Dry-run any engine without loading models:

```powershell
python scripts/tts/synthesize_from_plan.py --plan outputs/tts_plans/sample_plans.jsonl --limit 1 --dry-run
```

## Coqui/XTTS Adapter

```powershell
pip install -r scripts/tts/requirements-coqui.txt

python scripts/tts/synthesize_from_plan.py \
    --plan outputs/tts_plans/sample_plans.jsonl \
    --engine coqui_xtts \
    --device cuda \
    --default-speaker-wav data/reference_voice/agent.wav \
    --tts-kwargs-file scripts/tts/xtts_natural_kwargs.json \
    --agree-coqui-cpml \
    --postprocess
```

## Chatterbox Adapter

```powershell
pip install -r scripts/tts/requirements-chatterbox.txt

python scripts/tts/synthesize_from_plan.py \
    --plan outputs/tts_plans/sample_plans.jsonl \
    --engine chatterbox \
    --device cuda \
    --default-speaker-wav data/reference_voice/agent.wav \
    --postprocess
```

Chatterbox automatically maps emotion metadata to `exaggeration` values:
neutral=0.3, urgent=0.7, threatening=0.8, etc.

## Fish Speech Adapter

```powershell
pip install -r scripts/tts/requirements-fish-speech.txt

python scripts/tts/synthesize_from_plan.py \
    --plan outputs/tts_plans/sample_plans.jsonl \
    --engine fish_speech \
    --device cuda \
    --default-speaker-wav data/reference_voice/agent.wav \
    --postprocess
```

## Post-Processing

Add `--postprocess` to any synthesis command to apply audio enhancements:
- High-pass filter (remove rumble below 80Hz)
- Vocal warmth EQ (boost 200-400Hz)
- De-metallic filter (attenuate 3-5kHz)
- Soft compression (even dynamics)
- Peak normalization

Standalone usage:

```powershell
python scripts/tts/postprocess.py input.wav -o output.wav
```

## Colab Comparison

Use `colab_tts_comparison.py` to test all engines side-by-side on Google Colab:

1. Upload your project to Google Drive
2. Open a Colab notebook with GPU runtime
3. Copy cells from `colab_tts_comparison.py` into the notebook
4. Run all cells and compare audio outputs

## Speaker Map

Use `--speaker-map` with a JSON file to map emotions to specific reference
voices (see `speaker_map.example.json`).

```powershell
python scripts/tts/synthesize_from_plan.py \
    --plan outputs/tts_plans/sample_plans.jsonl \
    --engine chatterbox \
    --device cuda \
    --speaker-map scripts/tts/speaker_map.example.json \
    --postprocess
```
