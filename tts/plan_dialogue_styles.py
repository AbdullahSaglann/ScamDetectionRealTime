#!/usr/bin/env python3
"""Create model-agnostic emotional speech plans from dialogue rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TURN_PATTERN = re.compile(r"\[(AGENT|VICTIM)\]:\s*([^[\[]*?)(?=\[|$)", re.IGNORECASE)


SCAM_STYLE_BY_TYPE = {
    "polis_emniyet_jandarma": ("official", "threatening"),
    "savcilik_adliye": ("official", "threatening"),
    "teror_tehdidi": ("official", "threatening"),
    "banka_kart_hesap": ("urgent", "pressuring"),
    "oltalama_link_kod": ("urgent", "pressuring"),
    "kargo_gumruk": ("formal", "pressuring"),
    "edevlet_kamu_vergi": ("official", "pressuring"),
    "tuketici_hakem_iade": ("formal", "persuasive"),
    "sosyal_medya_whatsapp": ("casual", "persuasive"),
    "pazar_yeri_ilan": ("casual", "persuasive"),
    "yatirim_kripto": ("confident", "persuasive"),
    "sahte_is_teklifi": ("friendly", "persuasive"),
    "abonelik_odeme": ("formal", "urgent"),
    "operator_sim": ("formal", "urgent"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured TTS style plans from dataset dialogues."
    )
    parser.add_argument("--input", default="data/final/DataSetV4.csv", help="Input CSV path.")
    parser.add_argument(
        "--output",
        default="outputs/tts_plans/dialogue_style_plans.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Maximum rows to process.")
    parser.add_argument(
        "--include-safe",
        action="store_true",
        help="Include legitimate rows as well as scam rows.",
    )
    return parser.parse_args()


def extract_turns(dialogue: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for match in TURN_PATTERN.finditer(dialogue or ""):
        speaker = match.group(1).lower()
        text = normalize_text(match.group(2))
        if text:
            turns.append({"speaker": speaker, "text": text})
    return turns


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.strip("\"'")


def plan_turn_style(
    speaker: str,
    text: str,
    label: str,
    scam_type: str,
    turn_index: int,
) -> dict[str, str]:
    if label == "0" or scam_type == "legit":
        if speaker == "agent":
            return {
                "role": "advisor",
                "style": "reassuring",
                "emotion": "calm",
                "rate": "-3%",
                "pitch": "0Hz",
            }
        return {
            "role": "citizen",
            "style": "attentive",
            "emotion": "neutral",
            "rate": "-2%",
            "pitch": "+1Hz",
        }

    if speaker == "agent":
        style, emotion = SCAM_STYLE_BY_TYPE.get(scam_type, ("persuasive", "pressuring"))
        return {
            "role": "scam_agent",
            "style": style,
            "emotion": emotion,
            "rate": "-5%" if turn_index == 0 else "+0%",
            "pitch": "-2Hz",
        }

    return {
        "role": "victim",
        "style": "hesitant",
        "emotion": infer_victim_emotion(text),
        "rate": "-8%",
        "pitch": "+1Hz",
    }


def infer_victim_emotion(text: str) -> str:
    lowered = text.lower()
    panic_terms = ("ne demek", "kork", "emin", "anlamad", "nasıl", "hemen")
    refusal_terms = ("paylaşmam", "vermiyorum", "güvenmiyorum", "aramam", "resmi")
    if any(term in lowered for term in refusal_terms):
        return "cautious"
    if any(term in lowered for term in panic_terms):
        return "worried"
    return "uncertain"


def build_plan(row_id: int, row: dict[str, str]) -> dict[str, object] | None:
    label = str(row.get("label", "")).strip()
    scam_type = str(row.get("scam_type", "unknown")).strip() or "unknown"
    turns = extract_turns(row.get("text", ""))
    if not turns:
        return None

    planned_turns = []
    for index, turn in enumerate(turns):
        style = plan_turn_style(turn["speaker"], turn["text"], label, scam_type, index)
        planned_turns.append(
            {
                "turn_index": index,
                "speaker": turn["speaker"],
                "text": turn["text"],
                **style,
            }
        )

    return {
        "row_id": row_id,
        "label": int(label) if label in {"0", "1"} else label,
        "scam_type": scam_type,
        "turn_count": len(planned_turns),
        "turns": planned_turns,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    scanned = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row_id, row in enumerate(reader):
                scanned += 1
                label = str(row.get("label", "")).strip()
                if not args.include_safe and label == "0":
                    continue

                plan = build_plan(row_id, row)
                if not plan:
                    continue

                output_file.write(json.dumps(plan, ensure_ascii=False) + "\n")
                written += 1

                if written >= args.limit:
                    break

    print(f"Scanned rows: {scanned}")
    print(f"Written plans: {written}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
