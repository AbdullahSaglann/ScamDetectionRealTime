#!/usr/bin/env python3
"""Create DataSetV4 by adding safe security-warning examples.

These examples are intentionally "hard negatives": they contain words such as
password, SMS code, phishing, fraud, bank, and e-Devlet, but their intent is a
protective warning rather than a scam request.
"""

from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/intermediate/Sonhali_augmented_15k20k_typed.csv")
OUTPUT_FILE = Path("data/final/DataSetV4.csv")
REPORT_FILE = Path("data/final/DataSetV4_report.txt")
SEED = 496
TARGET_SAFE_WARNINGS = 300


OPENERS = [
    "[AGENT]: Güvenlik bilgilendirmesi:",
    "[AGENT]: Önemli güvenlik uyarısı:",
    "[AGENT]: Banka güvenlik duyurusu:",
    "[AGENT]: Kamu güvenliği bilgilendirmesi:",
    "[AGENT]: Siber güvenlik hatırlatması:",
    "[AGENT]: Dolandırıcılık uyarısı:",
]

WARNING_SENTENCES = [
    "Şifrenizi kimseyle paylaşmayın.",
    "SMS doğrulama kodunuzu hiçbir görevliye söylemeyin.",
    "Banka çalışanları sizden SMS kodu veya kart şifresi istemez.",
    "e-Devlet şifrenizi asla telefonla veya mesajla paylaşmayın.",
    "Kart bilgilerinizi yalnızca resmi uygulama üzerinden kontrol edin.",
    "Kargo ödeme linklerine tıklamadan önce resmi uygulamadan doğrulama yapın.",
    "Tanımadığınız kişilerden gelen yatırım vaatlerine para göndermeyin.",
    "Kendini polis, savcı veya banka görevlisi olarak tanıtan kişilere para aktarmayın.",
    "Dolandırıcılık girişimlerine karşı dikkatli olun.",
    "Kimlik bilgilerinizi e-posta veya mesaj bağlantılarına yazmayın.",
    "Tek kullanımlık kodlar sadece sizin kullanımınız içindir.",
    "Şüpheli bir arama alırsanız görüşmeyi sonlandırıp resmi çağrı merkezini arayın.",
    "Resmi kurumlar telefonla para transferi talep etmez.",
    "Güvenliğiniz için bağlantı adresini kendiniz yazarak giriş yapın.",
    "Bu mesaj bilgilendirme amaçlıdır, sizden ödeme veya şifre istenmemektedir.",
]

VICTIM_RESPONSES = [
    "[VICTIM]: Anladım, şifremi ve kodumu kimseyle paylaşmayacağım.",
    "[VICTIM]: Teşekkürler, işlemleri yalnızca resmi uygulamadan kontrol edeceğim.",
    "[VICTIM]: Bilgilendirme için teşekkür ederim, şüpheli linklere tıklamayacağım.",
    "[VICTIM]: Tamam, bir talep gelirse kurumu kendi numarasından arayacağım.",
    "[VICTIM]: Güvenlik uyarısını dikkate alacağım.",
]

CONTEXTS = [
    "Bu bir güvenlik bilgilendirmesidir; herhangi bir ödeme talebi içermez.",
    "Bu uyarı yalnızca farkındalık amacıyla gönderilmiştir.",
    "Bu görüşmede sizden kişisel bilgi, şifre veya kod talep edilmeyecektir.",
    "Şüpheli bir durum yaşarsanız resmi kanallardan destek alabilirsiniz.",
    "Amacımız hesap güvenliği konusunda sizi bilgilendirmektir.",
]


def normalize(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_warning(rng: random.Random, index: int) -> str:
    opener = rng.choice(OPENERS)
    warnings = rng.sample(WARNING_SENTENCES, k=rng.choice([2, 2, 3]))
    context = rng.choice(CONTEXTS)
    response = rng.choice(VICTIM_RESPONSES)

    if index % 5 == 0:
        return " ".join([opener, context, *warnings, response])
    if index % 5 == 1:
        return " ".join([opener, *warnings, context, response])
    if index % 5 == 2:
        return " ".join([opener, warnings[0], response, context, *warnings[1:]])
    if index % 5 == 3:
        return " ".join([opener, context, response, *warnings])
    return " ".join([opener, *warnings, response])


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    required = {"text", "label", "scam_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"{INPUT_FILE} must contain columns: {sorted(required)}")

    rng = random.Random(SEED)
    seen = set(df["text"].map(normalize))
    new_rows = []
    attempts = 0
    while len(new_rows) < TARGET_SAFE_WARNINGS:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("Could not generate enough unique safe warnings.")

        text = generate_warning(rng, attempts)
        norm = normalize(text)
        if norm in seen:
            continue

        seen.add(norm)
        new_rows.append({"text": text, "label": 0, "scam_type": "legit"})

    v3 = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    v3 = v3.drop_duplicates(subset=["text"], keep="first").sample(frac=1, random_state=SEED).reset_index(drop=True)
    v3.to_csv(OUTPUT_FILE, index=False)

    safe = v3[v3["label"] == 0]["text"].map(normalize)
    checks = {
        "safe contains sifre": safe.str.contains("şifre", regex=False).sum(),
        "safe contains sms kod": safe.str.contains("sms", regex=False).sum(),
        "safe contains dolandiricilik": safe.str.contains("dolandırıcılık", regex=False).sum(),
        "safe contains e-devlet": safe.str.contains("e-devlet", regex=False).sum(),
        "safe contains banka": safe.str.contains("banka", regex=False).sum(),
    }

    report = [
        f"Input file: {INPUT_FILE}",
        f"Output file: {OUTPUT_FILE}",
        f"Original rows: {len(df)}",
        f"Added safe warning rows: {len(new_rows)}",
        f"Final rows: {len(v3)}",
        f"Unique text: {v3['text'].nunique()}",
        f"Duplicate text: {len(v3) - v3['text'].nunique()}",
        "",
        "Label counts:",
        v3["label"].value_counts().sort_index().to_string(),
        "",
        "Safe warning keyword coverage:",
        "\n".join(f"{key}: {value}" for key, value in checks.items()),
    ]
    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
