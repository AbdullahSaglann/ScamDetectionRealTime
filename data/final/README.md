# Final Text Datasets

This directory contains the final, cleaned, and processed text datasets (in CSV format) used for training and evaluating the Scam Detection Real-Time models.

## Dataset Structure

Each dataset iteration improves upon the previous one by adding more balanced data, realistic noise, and diverse scam scenarios.

- **`DataSetV4.csv` - `DataSetV5.csv`**: Early iterations of the dataset containing basic scam and safe dialogues.
- **`DataSetV6.csv`**: A balanced dataset focusing on real-time conversational scenarios. Includes a more robust mixture of safe (non-scam) daily interactions and malicious scam patterns.
- **`DataSetV7.csv`**: An expanded dataset introducing advanced scam types (e.g., romantic fraud, fake charity, cargo scams).
- **`DataSetV7_1.csv`**: The most robust and final version. This dataset includes "noisy" scam scenarios and difficult edge cases to improve the AI model's real-world resilience.

## Data Schema

The CSV files typically contain the following key columns:
- `text`: The dialogue or message content (e.g., "[AGENT]: Hello, you won a prize! [VICTIM]: I don't believe you.").
- `label`: The classification label. 
  - `1` = Scam / Malicious
  - `0` = Safe / Normal Conversation
- `scam_type`: The specific category of the scam (e.g., `banka_kart_hesap` for banking scams, `kargo_gumruk` for cargo scams, or `safe` for non-scam).

## Generation Process

The datasets are generated synthetically using Large Language Models (LLMs) instructed with specific scam and non-scam personas. The raw generated data is then processed through a pipeline of Python scripts located in the `dataset_scripts/` directory at the root of this repository.

The pipeline generally involves:
1. **Scenario Generation**: Generating raw dialogues based on predefined scam vectors.
2. **Expansion**: Augmenting the dataset with daily safe negatives to prevent false positives (`add_daily_safe_negatives.py`).
3. **Cleaning & Formatting**: Removing markdown artifacts, normalizing text, and structuring it into a strict `[AGENT]` and `[VICTIM]` conversational format (`clean_augmented_dataset.py`).
4. **Balancing**: Ensuring an equal distribution of `label 0` (safe) and `label 1` (scam) data points to prevent model bias (`create_dataset_v6_realtime_balanced.py`).

These text datasets are the foundation for the text-based NLP classification models and serve as the base scripts from which the Audio TTS Datasets are generated.
