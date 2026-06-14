# ScamDetectionRealTime - TTS Engines

This repository contains the Text-to-Speech (TTS) architecture and related Python scripts used for the Scam Detection Real-Time project. It demonstrates how to utilize various modern open-source TTS engines to generate realistic, expressive voice recordings.

## Contents
- **`tts/`**: Contains synthesis scripts, planning logic, and engine adapters.
  - `colab_tts_comparison.py`: A Colab notebook script to compare different TTS engines (Coqui XTTS, Chatterbox, Fish Speech) with reference voices.
  - `create_expressive_tts_plan.py`: Prepares emotional speech generation plans based on text.
  - `synthesize_from_plan.py`: Main synthesis pipeline using the chosen TTS engine.
  - `postprocess.py`: Audio enhancement utilities (EQ, soft-compression) to reduce robotic artifacts.
  - `engines/`: Adapters for different TTS engines.

## Usage
The scripts are designed to be integrated into a larger data generation pipeline or run independently.

*Note: For the mobile application and real-time backend, please refer to the main application repository.*
