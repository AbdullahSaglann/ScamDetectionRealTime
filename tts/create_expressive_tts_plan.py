#!/usr/bin/env python3
"""Create expressive, performance-oriented TTS plans from scam datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TURN_PATTERN = re.compile(r"\[(AGENT|VICTIM)\]:\s*([^[\[]*?)(?=\[|$)", re.IGNORECASE)


SCAM_EMOTION_BY_TYPE = {
    "banka_kart_hesap": ("pressuring", "urgent"),
    "scam_noisy_bank": ("pressuring", "urgent"),
    "kargo_gumruk": ("pressuring", "formal"),
    "scam_noisy_cargo_gov": ("pressuring", "official"),
    "edevlet_kamu_vergi": ("pressuring", "official"),
    "polis_emniyet_jandarma": ("threatening", "official"),
    "savcilik_adliye": ("threatening", "official"),
    "yatirim_kripto": ("persuasive", "confident"),
    "scam_noisy_job_invest": ("persuasive", "confident"),
    "pazar_yeri_ilan": ("persuasive", "casual"),
    "scam_noisy_market_social": ("persuasive", "casual"),
    "sosyal_medya_whatsapp": ("persuasive", "casual"),
    "sahte_is_teklifi": ("friendly_pressure", "friendly"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/final/DataSetV7_1.csv")
    parser.add_argument("--output", default="outputs/tts_plans/expressive_plans.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--include-safe", action="store_true")
    parser.add_argument("--max-chars", type=int, default=220)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text.strip("\"'")


def extract_turns(text: str) -> list[dict[str, str]]:
    turns = []
    for match in TURN_PATTERN.finditer(text or ""):
        speaker = match.group(1).lower()
        turn_text = normalize(match.group(2))
        if turn_text:
            turns.append({"speaker": speaker, "text": turn_text})
    if turns:
        return turns
    clean = normalize(text)
    return [{"speaker": "agent", "text": clean}] if clean else []


def expressive_text(text: str, speaker: str, label: str, scam_type: str) -> str:
    text = normalize(text)
    text = text[:260]
    text = soften_for_speech(text)

    if label == "0" or scam_type.startswith("legit") or scam_type == "legit":
        return make_safe_expression(text, speaker)
    if speaker == "victim":
        return make_victim_expression(text)
    return make_scam_expression(text, scam_type)


def soften_for_speech(text: str) -> str:
    replacements = {
        ";": ",",
        "  ": " ",
        " iban ": " IBAN ",
        " sms ": " SMS ",
        " edevlet ": " e-devlet ",
        " whatsapp ": " WhatsApp ",
    }
    padded = f" {text} "
    for old, new in replacements.items():
        padded = padded.replace(old, new)
    text = normalize(padded)
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def make_safe_expression(text: str, speaker: str) -> str:
    if speaker == "victim":
        prefixes = ("Tamam, ", "Anladım, ", "")
    else:
        prefixes = ("Merhaba. ", "Kısa bir bilgilendirme yapalım. ", "")
    prefix = prefixes[len(text) % len(prefixes)]
    if any(term in text.lower() for term in ("kod", "şifre", "sifre", "link", "para", "kart")):
        return f"{prefix}{text} Lütfen acele etmeden, resmi kanaldan kontrol edelim."
    return f"{prefix}{text}"


def make_victim_expression(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("resmi", "paylaşmam", "paylasmam", "güvenmiyorum", "guvenmiyorum")):
        return f"Bir dakika... {text} Ben bunu resmi kanaldan kontrol etmek istiyorum."
    if any(term in lowered for term in ("neden", "nasıl", "nasil", "zorunda", "emin")):
        return f"Şey... {text} Bundan tam emin olamadım."
    return f"Anlamadım... {text}"


def make_scam_expression(text: str, scam_type: str) -> str:
    lowered = text.lower()
    greeting = "" if lowered.startswith(("merhaba", "selam", "günaydın", "gunaydin", "iyi günler", "iyi gunler")) else "Merhaba. "
    if any(term in lowered for term in ("hemen", "acil", "bugün", "bugun", "kapan", "bloke")):
        return f"Dikkatli dinleyin. {text} Bu işlemi şimdi tamamlamamız gerekiyor."
    if any(term in lowered for term in ("kod", "şifre", "sifre", "kart", "iban", "IBAN")):
        return f"{greeting}{text} İşlemi hızlandırmak için bilgiyi şimdi teyit edelim."
    if "kargo" in lowered:
        return f"Teslimat biriminden arıyorum. {text} Gecikme olmaması için hemen ilerleyelim."
    if "yatırım" in lowered or "yatirim" in lowered or "kripto" in lowered:
        return f"Size özel bir fırsat var. {text} Bu kontenjan uzun süre açık kalmayacak."
    return f"{greeting}{text} Lütfen bu işlemi geciktirmeyelim."


def style_for_turn(speaker: str, text: str, label: str, scam_type: str, turn_index: int) -> dict[str, str]:
    if label == "0" or scam_type.startswith("legit") or scam_type == "legit":
        return {
            "role": "advisor" if speaker == "agent" else "citizen",
            "style": "reassuring" if speaker == "agent" else "attentive",
            "emotion": "calm" if speaker == "agent" else "neutral",
            "rate": "-6%",
            "pitch": "0Hz",
        }

    if speaker == "victim":
        emotion = "cautious" if "resmi" in text.lower() else "worried"
        return {
            "role": "victim",
            "style": "hesitant",
            "emotion": emotion,
            "rate": "-10%",
            "pitch": "+1Hz",
        }

    emotion, style = SCAM_EMOTION_BY_TYPE.get(scam_type, ("pressuring", "persuasive"))
    return {
        "role": "scam_agent",
        "style": style,
        "emotion": emotion,
        "rate": "-4%" if turn_index == 0 else "+2%",
        "pitch": "-2Hz",
    }


def build_plan(row_id: int, row: dict[str, str], max_chars: int) -> dict[str, object] | None:
    label = str(row.get("label", "")).strip()
    scam_type = normalize(row.get("scam_type", "unknown")) or "unknown"
    turns = extract_turns(row.get("text", ""))
    if not turns:
        return None

    planned = []
    for index, turn in enumerate(turns):
        spoken = expressive_text(turn["text"], turn["speaker"], label, scam_type)
        if len(spoken) > max_chars:
            spoken = spoken[:max_chars].rsplit(" ", 1)[0] + "."
        planned.append(
            {
                "turn_index": index,
                "speaker": turn["speaker"],
                "source_text": turn["text"],
                "text": spoken,
                **style_for_turn(turn["speaker"], spoken, label, scam_type, index),
            }
        )

    return {
        "row_id": row_id,
        "label": int(label) if label in {"0", "1"} else label,
        "scam_type": scam_type,
        "turn_count": len(planned),
        "turns": planned,
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
                if label == "0" and not args.include_safe:
                    continue
                plan = build_plan(row_id, row, args.max_chars)
                if not plan:
                    continue
                output_file.write(json.dumps(plan, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.limit:
                    break

    print(f"Scanned rows: {scanned}")
    print(f"Written expressive plans: {written}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
