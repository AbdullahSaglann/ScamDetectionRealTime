#!/usr/bin/env python3
"""Add a scam_type column to the 15k-20k dialogue dataset."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/intermediate/Sonhali_augmented_15k20k.csv")
OUTPUT_FILE = Path("data/intermediate/Sonhali_augmented_15k20k_typed.csv")
REPORT_FILE = Path("data/archive/reports/Sonhali_augmented_15k20k_type_report.txt")


TYPE_RULES: list[tuple[str, list[str]]] = [
    (
        "teror_tehdidi",
        [
            r"\bter[oö]r\b",
            r"ter[oö]r[üu]st",
            r"pkk",
            r"fet[oö]",
            r"örg[üu]t",
        ],
    ),
    (
        "savcilik_adliye",
        [
            r"savc[ıi]",
            r"savc[ıi]l[ıi]k",
            r"adliye",
            r"dava dosya",
            r"soruşturma",
            r"gizli dosya",
        ],
    ),
    (
        "polis_emniyet_jandarma",
        [
            r"\bpolis\b",
            r"emniyet",
            r"jandarma",
            r"siber suç",
            r"kolluk",
            r"kimlik bilgileriniz bir dosyada",
        ],
    ),
    (
        "banka_kart_hesap",
        [
            r"banka",
            r"bankas[ıi]",
            r"kart",
            r"hesap",
            r"hesab[ıi]n[ıi]z",
            r"iban",
            r"eft",
            r"havale",
            r"para çıkışı",
            r"şüpheli işlem",
            r"güvenli hesap",
            r"sms kod",
            r"tek kullanımlık kod",
        ],
    ),
    (
        "kargo_gumruk",
        [
            r"kargo",
            r"ptt",
            r"yurtiçi",
            r"aras",
            r"mng",
            r"dhl",
            r"ups",
            r"gümrük",
            r"paketiniz",
            r"teslim",
            r"dağıtım",
        ],
    ),
    (
        "edevlet_kamu_vergi",
        [
            r"e-devlet",
            r"edevlet",
            r"vergi",
            r"gümrük birimi",
            r"belediye",
            r"kamu",
            r"t\.c\.",
            r"kimlik numarası",
            r"kayıt güncellemesi",
        ],
    ),
    (
        "tuketici_hakem_iade",
        [
            r"tüketici",
            r"hakem",
            r"iade",
            r"başvurunuz sonuçlandı",
            r"kart bilgilerinizi doğrulamamız",
            r"hakkınız yanar",
        ],
    ),
    (
        "sosyal_medya_whatsapp",
        [
            r"whatsapp",
            r"instagram",
            r"mavi tik",
            r"sosyal medya",
            r"yeni numaram",
            r"cüzdanımı kaybettim",
            r"arkadaş",
            r"eski numara",
            r"iban atıyorum",
            r"para gönderebilir misin",
        ],
    ),
    (
        "yatirim_kripto",
        [
            r"kripto",
            r"yatırım",
            r"forex",
            r"spk",
            r"garantili yüksek kazanç",
            r"cüzdan adres",
            r"transfer yaparsanız",
            r"danışmanınız",
        ],
    ),
    (
        "abonelik_odeme",
        [
            r"netflix",
            r"spotify",
            r"abonelik",
            r"aboneliğiniz",
            r"elektrik",
            r"fatura",
            r"kesinti",
            r"üyeliğiniz",
            r"ödeme bilgisi",
            r"bulut depolama",
            r"oyun üyeliği",
        ],
    ),
    (
        "operator_sim",
        [
            r"turkcell",
            r"vodafone",
            r"türk telekom",
            r"operatör",
            r"\bsim\b",
            r"hattınız",
            r"hat güvenliği",
            r"sim bloke",
        ],
    ),
    (
        "sahte_is_teklifi",
        [
            r"evden çalışma",
            r"iş başvur",
            r"iş garantisi",
            r"kayıt ücreti",
            r"yurtdışı iş",
            r"güvence bedeli",
            r"sözleşme çıkacak",
            r"başvurunuz onaylandı",
        ],
    ),
    (
        "pazar_yeri_ilan",
        [
            r"ilanınızdaki",
            r"alıcı ödemeyi yaptı",
            r"platform dışı",
            r"satışta para almak",
            r"doğrulama ücreti",
        ],
    ),
    (
        "sahte_kredi_destek",
        [
            r"kredi",
            r"kobi",
            r"destek ve kredi",
            r"sıfır faiz",
            r"ön onay bedeli",
        ],
    ),
    (
        "oltalama_link_kod",
        [
            r"link",
            r"bağlantı",
            r"onay kod",
            r"şifre",
            r"oturum bilg",
            r"giriş bilg",
            r"kimlik doğrulama",
            r"bilgilerinizi doğrul",
            r"yetkisiz .*giriş",
            r"erişiminiz",
            r"güvenlik ekranı",
            r"kısa form",
            r"kart son",
        ],
    ),
    (
        "sahte_ceza_arac",
        [
            r"arac[ıi]n[ıi]z",
            r"araç bilg",
            r"suç ihbar",
            r"suç dosyas",
            r"cezai işlem",
            r"ceza alman",
        ],
    ),
    (
        "sahte_teknik_destek",
        [
            r"zararlı yazılım",
            r"güvenlik yazılım",
            r"çalışma istasyon",
            r"sisteminizi korumak",
            r"indirip kur",
            r"bt ekibi",
        ],
    ),
    (
        "romantik_yardim_bagis",
        [
            r"romantik",
            r"ilişki",
            r"cüzdan[ıi]m çal[ıi]nd[ıi]",
            r"otelde ödeme",
            r"geçici borç",
            r"yardım galası",
            r"ritüel",
            r"maddi enerji",
            r"dul olduğum",
            r"para göndermeyeceğim",
        ],
    ),
    (
        "asker_yardim_bahanesi",
        [
            r"askeri personel",
            r"sınırda görevli",
            r"irak.taki görev",
            r"aileme ulaşamıyorum",
            r"güvenlik güçleri",
        ],
    ),
    (
        "noter_tebligat_kimlik",
        [
            r"noter",
            r"tebligat",
            r"anne kızlık soyad",
            r"sahte kira",
            r"kira sözleşmesi",
        ],
    ),
    (
        "icra_haciz_tehdit",
        [
            r"icra",
            r"haciz",
            r"borcum yok",
            r"para cezası",
            r"ödeme yapmadan kapatmayın",
        ],
    ),
]


def normalize(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_scam(text: object, label: int) -> str:
    if int(label) == 0:
        return "legit"

    normalized = normalize(text)

    priority_rules = [
        "teror_tehdidi",
        "savcilik_adliye",
        "polis_emniyet_jandarma",
        "kargo_gumruk",
        "edevlet_kamu_vergi",
        "tuketici_hakem_iade",
        "operator_sim",
        "sosyal_medya_whatsapp",
        "sahte_is_teklifi",
        "pazar_yeri_ilan",
        "sahte_kredi_destek",
    ]
    rule_map = dict(TYPE_RULES)
    for scam_type in priority_rules:
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in rule_map[scam_type]):
            return scam_type

    for scam_type, patterns in TYPE_RULES:
        if scam_type in priority_rules:
            continue
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            return scam_type

    return "diger_scam"


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    typed = df.copy()
    typed["scam_type"] = [
        classify_scam(text, label) for text, label in zip(typed["text"], typed["label"])
    ]
    typed.to_csv(OUTPUT_FILE, index=False)

    fraud_counts = typed[typed["label"] == 1]["scam_type"].value_counts()
    all_counts = typed["scam_type"].value_counts()

    report = [
        f"Input file: {INPUT_FILE}",
        f"Output file: {OUTPUT_FILE}",
        f"Rows: {len(typed)}",
        f"Unique text: {typed['text'].nunique()}",
        f"Duplicate text: {len(typed) - typed['text'].nunique()}",
        "",
        "Label counts:",
        typed["label"].value_counts().sort_index().to_string(),
        "",
        "Fraud type counts:",
        fraud_counts.to_string(),
        "",
        "All type counts:",
        all_counts.to_string(),
    ]
    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
