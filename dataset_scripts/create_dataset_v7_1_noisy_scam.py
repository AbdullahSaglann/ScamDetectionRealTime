#!/usr/bin/env python3
"""Create V7.1 by adding noisy scam robustness examples to V7."""

from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/final/DataSetV7.csv")
OOD_PATHS = [
    Path("data/eval/OODTestSetV1.csv"),
    Path("data/eval/OODTestSetV2_realtime_noise.csv"),
]
OUTPUT_PATH = Path("data/final/DataSetV7_1.csv")
REPORT_PATH = Path("data/final/DataSetV7_1_report.txt")

TARGET_SAFE_ADD = 1_000
TARGET_SCAM_ADD = 3_000


def normalize(text: str) -> str:
    return " ".join(str(text).split())


def dialogue(*turns: tuple[str, str]) -> str:
    return " ".join(f"[{speaker.upper()}]: {normalize(text)}" for speaker, text in turns)


def add_unique(
    rows: list[dict[str, object]],
    seen: set[str],
    blocked: set[str],
    text: str,
    label: int,
    scam_type: str,
) -> None:
    normalized = normalize(text)
    if not normalized:
        return
    key = normalized.lower()
    if key in seen or key in blocked:
        return
    seen.add(key)
    rows.append({"text": normalized, "label": label, "scam_type": scam_type})


def build_noisy_scam(existing: set[str], blocked: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen = set(existing)

    noisy_openers = [
        "",
        "mrb",
        "slm",
        "acil",
        "bilgi",
        "son uyari",
        "lutfen dikkat",
        "sistem mesaji",
    ]
    noisy_tails = [
        "",
        "tsk",
        "lutfen",
        "bugun",
        "hemen",
        "acele",
        "gec kalmayin",
        "simdi yapin",
    ]
    typo_tokens = [
        ("sifrenizi", ["sifreni", "sifrenizi", "sifreyi", "parolani", "parolanizi"]),
        ("kodunu", ["kodu", "kodunu", "sms kodunu", "dogrulama kodunu", "6 haneli kodu"]),
        ("linke", ["linke", "lnke", "baglantiya", "forma", "sayfaya"]),
        ("kart", ["kart", "karti", "kredi karti", "banka karti"]),
    ]

    base_scam = [
        (
            "scam_noisy_bank",
            [
                "{kart} supheli islem verdi iptal icin {linke} girip {sifrenizi} yaz",
                "{kart} islem askida kaldi iptal etmek icin {kodunu} bize soyle",
                "hesap kapanacak guvenlik icin {kodunu} oku",
                "bankadan ariyoruz paran riskte guvenli hesaba aktar",
                "mobil banka kilitlendi acmak icin {sifrenizi} teyit et",
                "iade alman icin {kart} bilgilerini forma gir",
            ],
        ),
        (
            "scam_noisy_cargo_gov",
            [
                "kargo adres eksik teslimat icin {kodunu} soyle",
                "kargo gumrukte kaldi odeme yapmazsan geri gider",
                "edevlet dosya kapanacak {sifrenizi} teyit et",
                "vergi iaden hazir almak icin {kart} bilgilerini yaz",
                "icra dosyan acilmis uzlasma icin ibana para at",
                "resmi evrak ucreti icin hemen havale yap",
            ],
        ),
        (
            "scam_noisy_market_social",
            [
                "alisveris odemen askida {kart} bilgilerini tekrar gir",
                "ilan sana ayrildi kapora gonder urunu baskasina vermeyeyim",
                "satici guven ucreti icin ibana havale yap",
                "whatsapp hesabin kapanacak {kodunu} bana yaz",
                "instagram sikayet aldi {linke} giris yap",
                "cekilis kazandin kargo ucretini ode hediyeyi al",
            ],
        ),
        (
            "scam_noisy_job_invest",
            [
                "ise alindin ekipman ucreti yatirman lazim",
                "is kaydi icin kimlik fotografi ve banka bilgisi at",
                "kripto kazancin hazir cekmek icin once vergi ode",
                "yatirim firsati bugun son ibana para aktar",
                "hesap blokesi kalkmasi icin islem ucreti gonder",
                "evden is icin depozito ode sonra baslayacaksin",
            ],
        ),
    ]

    replacement_sets = []
    for _, values in typo_tokens:
        replacement_sets.append(values)

    for scam_type, templates in base_scam:
        for template in templates:
            for opener, tail, replacements in product(noisy_openers, noisy_tails, product(*replacement_sets)):
                text = template
                for (placeholder, _), value in zip(typo_tokens, replacements):
                    text = text.replace("{" + placeholder + "}", value)
                message = " ".join(part for part in [opener, text, tail] if part)
                add_unique(rows, seen, blocked, message, 1, scam_type)

                add_unique(
                    rows,
                    seen,
                    blocked,
                    dialogue(
                        ("agent", message),
                        ("victim", "resmi yerden kontrol edebilir miyim"),
                        ("agent", "olmaz sistem simdi kapanacak hemen yapmaniz lazim"),
                    ),
                    1,
                    scam_type,
                )

    return rows


def build_noisy_safe(existing: set[str], blocked: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen = set(existing)

    safe_subjects = [
        "kartimi kaybettim bankayi kendim arayacagim",
        "kargo mesajini resmi uygulamadan kontrol ederim",
        "sms kodunu kimseye soylemem",
        "edevlet sifremi sadece resmi sitede kullanirim",
        "kapora isteyen ilani once kontrol edecegim",
        "kripto reklamina para yatirmadim",
        "alisveris linki supheli geldi tiklamadim",
        "is basvurusu icin ucret istenirse kabul etmem",
        "ibanimi sadece kendi hesabima transfer icin kullandim",
        "telefonla sifre isteyen aramayi kapattim",
    ]
    noisy_prefixes = ["", "mrb", "slm", "bilgi", "not", "hatirlatma", "bence", "bugun"]
    safe_tails = [
        "",
        "guvenli olan bu",
        "kimseye bilgi vermedim",
        "para gondermedim",
        "linke tiklamadim",
        "resmi kanaldan bakicam",
        "sadece bilgilendirme",
        "dolandiricilik olmasin diye",
    ]
    for subject, prefix, tail in product(safe_subjects, noisy_prefixes, safe_tails):
        text = " ".join(part for part in [prefix, subject, tail] if part)
        add_unique(rows, seen, blocked, text, 0, "legit_noisy_hard_safe")
        add_unique(
            rows,
            seen,
            blocked,
            dialogue(
                ("victim", text),
                ("agent", "dogru, gizli bilgi veya para paylasmadan resmi kanaldan kontrol et"),
            ),
            0,
            "legit_noisy_hard_safe",
        )

    return rows


def take_balanced(candidates: list[dict[str, object]], target: int) -> list[dict[str, object]]:
    if len(candidates) < target:
        raise ValueError(f"Not enough candidates: {len(candidates)} < {target}")
    groups: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        groups.setdefault(str(row["scam_type"]), []).append(row)
    keys = sorted(groups)
    per_group = target // len(keys)
    remainder = target % len(keys)
    selected: list[dict[str, object]] = []
    leftovers: list[dict[str, object]] = []
    for index, key in enumerate(keys):
        quota = per_group + (1 if index < remainder else 0)
        selected.extend(groups[key][:quota])
        leftovers.extend(groups[key][quota:])
    if len(selected) < target:
        selected.extend(leftovers[: target - len(selected)])
    return selected[:target]


def load_blocked() -> set[str]:
    blocked: set[str] = set()
    for path in OOD_PATHS:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        blocked.update(df["text"].astype(str).map(normalize).str.lower())
    return blocked


def main() -> None:
    base_df = pd.read_csv(INPUT_PATH)
    base_df["text"] = base_df["text"].astype(str).map(normalize)
    existing = set(base_df["text"].str.lower())
    blocked = load_blocked()

    safe_candidates = build_noisy_safe(existing, blocked)
    seen_after_safe = existing | {row["text"].lower() for row in safe_candidates}
    scam_candidates = build_noisy_scam(seen_after_safe, blocked)

    safe_rows = take_balanced(safe_candidates, TARGET_SAFE_ADD)
    scam_rows = take_balanced(scam_candidates, TARGET_SCAM_ADD)

    add_df = pd.DataFrame(safe_rows + scam_rows)
    out_df = pd.concat([base_df, add_df], ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    leaked = sorted(set(out_df["text"].str.lower()) & blocked)
    if leaked:
        raise ValueError(f"OOD leakage detected: {len(leaked)} rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    label_counts = Counter(out_df["label"])
    scam_type_counts = Counter(out_df["scam_type"])
    report_lines = [
        "DataSetV7.1 noisy scam generation report",
        f"Input: {INPUT_PATH}",
        f"Output: {OUTPUT_PATH}",
        f"OOD blocked rows: {len(blocked)}",
        f"Base rows: {len(base_df)}",
        f"Generated noisy safe candidates: {len(safe_candidates)}",
        f"Generated noisy scam candidates: {len(scam_candidates)}",
        f"Added noisy safe rows: {len(safe_rows)}",
        f"Added noisy scam rows: {len(scam_rows)}",
        f"Final rows: {len(out_df)}",
        f"Duplicate texts: {int(out_df.duplicated(subset=['text']).sum())}",
        f"Label counts: {dict(label_counts)}",
        "Top scam_type counts:",
    ]
    for key, value in scam_type_counts.most_common(35):
        report_lines.append(f"  {key}: {value}")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
