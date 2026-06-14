# ScamDetectionRealTime - TTS Engines

This repository contains the Text-to-Speech (TTS) architecture and related Python scripts used for the Scam Detection Real-Time project. It demonstrates how to utilize various modern open-source TTS engines to generate realistic, expressive voice recordings.

## Contents
- **`tts/`**: Contains synthesis scripts, planning logic, and engine adapters.
  - `colab_tts_comparison.py`: A Colab notebook script to compare different TTS engines (Coqui XTTS, Chatterbox, Fish Speech) with reference voices.
  - `create_expressive_tts_plan.py`: Prepares emotional speech generation plans based on text.
  - `synthesize_from_plan.py`: Main synthesis pipeline using the chosen TTS engine.
  - `postprocess.py`: Audio enhancement utilities (EQ, soft-compression) to reduce robotic artifacts.
  - `engines/`: Adapters for different TTS engines.

- **`dataset_scripts/`**: Contains python scripts used for generating, cleaning, and expanding the text datasets. Includes logic for creating balanced real-time scam and safe dialogue datasets.

- **`data/final/`**: Contains the final, cleaned, and balanced text datasets (in CSV format) used for training and evaluating the Scam Detection models.

## Audio Dataset (19.4 GB)
Due to GitHub's size limitations, the complete generated TTS audio dataset (19.4 GB of WAV files) is hosted externally on Google Drive. 
**[Download Audio Dataset (Google Drive)](https://drive.google.com/drive/folders/1SymchcT0u3mvM2OMDGi5SvF4EwzR34qp?usp=sharing)**

## Usage
The scripts are designed to be integrated into a larger data generation pipeline or run independently.

*Note: For the mobile application and real-time backend, please refer to the main application repository.*
