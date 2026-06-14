#!/usr/bin/env python3
"""Create a stronger V7 dataset from V6 with OOD-focused hard examples."""

from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/final/DataSetV6.csv")
OOD_TEST_PATH = Path("data/eval/OODTestSetV1.csv")
OUTPUT_PATH = Path("data/final/DataSetV7.csv")
REPORT_PATH = Path("data/final/DataSetV7_report.txt")

TARGET_SAFE_ADD = 7_500
TARGET_SCAM_ADD = 7_500


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


def build_hard_safe(existing: set[str], blocked: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen = set(existing)

    actors = [
        "banka calisani",
        "polis memuru",
        "kargo gorevlisi",
        "e-devlet destek hatti",
        "sosyal medya destek ekibi",
        "kripto yatirim danismani",
        "ilan saticisi",
        "ise alim sorumlusu",
    ]
    protected_items = [
        "SMS kodu",
        "kart sifresi",
        "e-devlet sifresi",
        "internet bankaciligi parolasi",
        "WhatsApp dogrulama kodu",
        "kimlik fotografi",
        "kart bilgisi",
        "tek kullanimlik sifre",
    ]
    safe_actions = [
        "kimseyle paylasilmamali",
        "telefonda soylenmemeli",
        "sadece resmi uygulamada kullanilmali",
        "mesajla gonderilmemeli",
        "sahte linklere yazilmamali",
        "resmi numara aranarak kontrol edilmeli",
        "guvenilmeyen kisilere verilmemeli",
        "supheli durumda islem durdurulmali",
    ]
    endings = [
        "Bu bir guvenlik bilgilendirmesidir.",
        "Bu konusmada para transferi talebi yoktur.",
        "Herhangi bir kod veya sifre istemiyoruz.",
        "Lutfen islemi resmi kanaldan kendiniz kontrol edin.",
        "Amac sadece farkindalik olusturmaktir.",
    ]
    for actor, item, action, ending in product(actors, protected_items, safe_actions, endings):
        add_unique(
            rows,
            seen,
            blocked,
            dialogue(
                ("agent", f"{actor} gibi arayan biri {item} isterse dikkatli olun."),
                ("agent", f"{item} {action}. {ending}"),
                ("victim", "Anladim, resmi kanaldan kontrol edecegim."),
            ),
            0,
            "legit_hard_security_warning",
        )
        add_unique(
            rows,
            seen,
            blocked,
            f"Guvenlik notu: {actor} oldugunu soyleyen biri {item} isterse {action}. {ending}",
            0,
            "legit_hard_security_warning",
        )

    safe_finance_subjects = [
        "Kartimi kaybettim",
        "Hesap hareketlerimi kontrol edecegim",
        "IBAN bilgisini kendi hesabima para aktarmak icin kullandim",
        "Kredi karti limitimi dusurmek istiyorum",
        "Bankanin resmi numarasini arayacagim",
        "Mobil uygulamada bildirim gordum",
        "Subeden hesap dokumu alacagim",
        "Faturami resmi uygulamadan odeyecegim",
    ]
    safe_finance_contexts = [
        "sifre veya kod paylasmadan",
        "mesajdaki linke tiklamadan",
        "resmi uygulamayi kullanarak",
        "bankanin kendi sitesinden numarayi bularak",
        "kimseye para aktarmadan",
        "kart bilgisi vermeden",
        "musteri hizmetlerini kendim arayarak",
        "subeye giderek",
    ]
    safe_finance_results = [
        "islemi kontrol edecegim",
        "guvenli sekilde tamamlayacagim",
        "gerekirse karti iptal ettirecegim",
        "supheli bir durum varsa bankaya bildirecegim",
        "herhangi bir gizli bilgi paylasmayacagim",
    ]
    for subject, context, result in product(safe_finance_subjects, safe_finance_contexts, safe_finance_results):
        add_unique(
            rows,
            seen,
            blocked,
            f"{subject}; {context} {result}. Bu normal bir kullanici islemidir.",
            0,
            "legit_hard_finance",
        )
        add_unique(
            rows,
            seen,
            blocked,
            dialogue(
                ("victim", subject + "."),
                ("agent", f"Guvenli yol {context} ilerlemek."),
                ("victim", f"Tamam, {result}."),
            ),
            0,
            "legit_hard_finance",
        )

    safe_public_warnings = [
        "Polis telefonda para isteyen kisilere guvenilmemesi konusunda uyarida bulundu",
        "Belediye sahte odeme linklerine karsi bilgilendirme yapti",
        "Banka SMS kodu isteyen aramalara dikkat edilmesini duyurdu",
        "Kargo firmasi ek odeme isteyen sahte mesajlari acikladi",
        "E-devlet sifresinin hicbir gorevliye soylenmemesi gerektigi hatirlatildi",
        "Kripto yatirim vaadiyle arayan kisilere para gonderilmemesi onerildi",
        "Ilanlarda kapora gondermeden once saticinin kontrol edilmesi tavsiye edildi",
        "Sosyal medya hesaplari icin dogrulama kodunun paylasilmamasi istendi",
    ]
    qualifiers = [
        "Bu haber sadece bilgilendirme amaclidir",
        "Bu cumlede herhangi bir odeme talebi yoktur",
        "Kullanici sadece dikkatli olmaktan bahsediyor",
        "Islem yapmak icin resmi kanala yonlendirme vardir",
        "Gizli bilgi isteme veya para aktarma talebi bulunmaz",
    ]
    for warning, qualifier in product(safe_public_warnings, qualifiers):
        add_unique(
            rows,
            seen,
            blocked,
            f"{warning}. {qualifier}.",
            0,
            "legit_hard_public_warning",
        )
        add_unique(
            rows,
            seen,
            blocked,
            dialogue(
                ("agent", warning + "."),
                ("victim", "Tamam, resmi kaynaklardan kontrol ederim."),
                ("agent", qualifier + "."),
            ),
            0,
            "legit_hard_public_warning",
        )

    official_topics = [
        "kargo teslimati",
        "vergi borcu",
        "banka bildirimi",
        "kredi karti ekstresi",
        "e-devlet basvurusu",
        "sosyal medya guvenligi",
        "ikinci el ilan gorusmesi",
        "is basvurusu",
        "kripto haberleri",
        "fatura odemesi",
    ]
    official_channels = [
        "resmi uygulamadan",
        "kurumun kendi web sitesinden",
        "subeye giderek",
        "cagri merkezi numarasini kendim bularak",
        "e-devlet uzerinden",
        "kargo firmasinin uygulamasindan",
        "bankanin mobil uygulamasindan",
    ]
    no_risk_phrases = [
        "linke tiklamadan kontrol edecegim",
        "kod veya sifre paylasmayacagim",
        "para transferi yapmayacagim",
        "kart bilgisi girmeyecegim",
        "once kaynagin dogrulugunu kontrol edecegim",
        "supheli gorursem islemi durduracagim",
    ]
    for topic, channel, phrase in product(official_topics, official_channels, no_risk_phrases):
        add_unique(
            rows,
            seen,
            blocked,
            f"{topic} hakkinda mesaj geldi; {channel} {phrase}.",
            0,
            "legit_hard_official_channel",
        )
        add_unique(
            rows,
            seen,
            blocked,
            dialogue(
                ("victim", f"{topic} ile ilgili bir bildirim gordum."),
                ("agent", f"En dogrusu {channel} kontrol etmek."),
                ("victim", phrase + "."),
            ),
            0,
            "legit_hard_official_channel",
        )

    negative_intent_subjects = [
        "Bana para gonder diyen kisiyi reddettim",
        "SMS kodu isteyen aramayi kapattim",
        "Sahte oldugunu dusundugum linke tiklamadim",
        "Kapora isteyen ilan sahibine odeme yapmadim",
        "E-devlet sifremi isteyen kisiye cevap vermedim",
        "Kripto kazanc vaadine inanmadim",
        "Kart bilgisi isteyen forma bilgi girmedim",
        "Kargo ucreti isteyen mesaji sildim",
    ]
    explanations = [
        "cunku resmi kanal disinda islem yapmak istemiyorum",
        "cunku gizli bilgilerimi korumam gerekiyor",
        "cunku bu sadece bir guvenlik onlemiydi",
        "cunku once kurumun gercek numarasindan teyit edecegim",
        "cunku hicbir acil baskiya gore hareket etmeyecegim",
    ]
    for subject, explanation in product(negative_intent_subjects, explanations):
        add_unique(
            rows,
            seen,
            blocked,
            f"{subject}, {explanation}.",
            0,
            "legit_hard_refusal",
        )
        add_unique(
            rows,
            seen,
            blocked,
            dialogue(
                ("victim", subject + "."),
                ("agent", explanation + "."),
                ("victim", "Tamam, resmi kanaldan teyit edecegim."),
            ),
            0,
            "legit_hard_refusal",
        )

    education_topics = [
        "dolandiricilik farkindaligi egitimi",
        "siber guvenlik dersi",
        "aile icin guvenlik hatirlatmasi",
        "is yerinde guvenli internet sunumu",
        "okulda banka dolandiriciligi konusu",
        "mahalle bilgilendirme toplantisi",
        "telefon guvenligi semineri",
        "sosyal medya hesap guvenligi anlatimi",
    ]
    risk_terms = [
        "SMS kodu",
        "kart sifresi",
        "para transferi",
        "sahte kargo linki",
        "e-devlet sifresi",
        "kapora odemesi",
        "kripto vaadi",
        "kimlik fotografi",
        "IBAN'a para gonderme",
        "dogrulama kodu",
    ]
    learning_actions = [
        "asla paylasilmamasi gerektigini anlattik",
        "resmi kanaldan kontrol edilmesi gerektigini konustuk",
        "supheli durumda islemi durdurmayi ogrendik",
        "kimseye guvenerek verilmemesi gerektigini belirttik",
        "acil baski yapilsa bile dikkatli olunmasini vurguladik",
        "telefonla istenirse reddedilmesi gerektigini soyledik",
    ]
    for topic, term, action in product(education_topics, risk_terms, learning_actions):
        add_unique(
            rows,
            seen,
            blocked,
            f"{topic} sirasinda {term} konusunda {action}. Bu egitim amacli guvenli bir cumledir.",
            0,
            "legit_hard_education",
        )
        add_unique(
            rows,
            seen,
            blocked,
            dialogue(
                ("agent", f"Bugunku konu {topic}."),
                ("agent", f"{term} icin {action}."),
                ("victim", "Anladim, bunu kimseyle paylasmayacagim."),
            ),
            0,
            "legit_hard_education",
        )

    return rows


def build_robust_scam(existing: set[str], blocked: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen = set(existing)

    openings = [
        "Merhaba",
        "Iyi gunler",
        "Acil islem gerekiyor",
        "Sistem uzerinden ariyoruz",
        "Basvurunuz hakkinda ulasiyoruz",
        "Teslimat biriminden ariyoruz",
    ]
    politeness = ["", "lutfen", "tesekkurler", "islemi geciktirmeyin", "bugun icinde tamamlayin"]

    scam_templates = [
        (
            "scam_bank_robust",
            [
                "kartinizda supheli islem var, iptal icin tek kullanimlik kodu bize okuyun",
                "hesabiniz bloke olacak, guvenlik icin sifrenizi ve gelen SMS kodunu teyit edin",
                "paraniz riskte, guvenli hesaba gecici transfer yapmaniz gerekiyor",
                "kredi basvurunuz onaylandi, dosya ucreti icin verilen IBAN'a odeme yapin",
                "mobil bankacilik erisiminiz kapanacak, linke girip bilgilerinizi yenileyin",
                "kart aidati iadesi icin kart numarasi ve son kullanma tarihini yazin",
            ],
        ),
        (
            "scam_cargo_gov_robust",
            [
                "kargonuz gumrukte bekliyor, teslimat icin kisa linkten odeme yapin",
                "adresiniz eksik, teslimat icin gelen dogrulama kodunu bize iletin",
                "vergi iadeniz hazir, almak icin kart bilgilerinizi forma girin",
                "e-devlet dosyaniz kapanacak, sifrenizi soyleyerek kimlik dogrulayin",
                "icra islemi baslamadan uzlasma bedelini bu hesaba yatirin",
                "resmi evrak ucreti icin havale yaparsaniz dosyaniz bugun kapanir",
            ],
        ),
        (
            "scam_invest_job_robust",
            [
                "evden is basvurunuz onaylandi, ekipman icin depozito gonderin",
                "ise giris kaydi icin kimlik fotografinizi ve banka bilgilerinizi yollayin",
                "yuksek getirili fon icin sadece bugun bu IBAN'a para aktarabilirsiniz",
                "kripto kazancinizi cekmek icin once bloke kaldirma ucreti odeyin",
                "yatirim firsati sinirli, para gonderirseniz kazanciniz ikiye katlanacak",
                "hesabinizdaki bonusu almak icin vergi odemesini hemen tamamlayin",
            ],
        ),
        (
            "scam_social_market_robust",
            [
                "ilani size ayiririm ama once kaporayi bu hesaba gondermeniz gerekiyor",
                "satici koruma ucreti icin havale yapin, sonra urunu gonderecegiz",
                "alisveris odemeniz askida kaldi, kart bilgilerinizi tekrar girin",
                "cekilis kazandiniz, odulu almak icin kargo ucretini hemen odeyin",
                "hesabiniz sikayet edildi, kapanmamasi icin bu linkten giris yapin",
                "WhatsApp hesabinizi korumak icin gelen dogrulama kodunu bize yazin",
            ],
        ),
    ]
    victims = [
        "Bu islem zorunlu mu",
        "Resmi uygulamadan kontrol edebilir miyim",
        "Neden kodu size soylemem gerekiyor",
        "Bu odemeyi simdi yapmak zorunda miyim",
        "Link yerine resmi kanaldan yapamaz miyim",
    ]
    pressure = [
        "aksi halde hesabiniz kapanir",
        "islem bugun tamamlanmazsa hakkiniz yanar",
        "gec kalirsaniz dosya isleme alinir",
        "sistem sadece simdi izin veriyor",
        "guvenlik icin telefonda tamamlamamiz gerekiyor",
    ]
    for scam_type, templates in scam_templates:
        for opening, base, polite, victim, press in product(openings, templates, politeness, victims, pressure):
            tail = f" {polite}" if polite else ""
            add_unique(
                rows,
                seen,
                blocked,
                f"{opening}, {base}; {press}{tail}.",
                1,
                scam_type,
            )
            add_unique(
                rows,
                seen,
                blocked,
                dialogue(
                    ("agent", f"{opening}, {base}."),
                    ("victim", victim + "?"),
                    ("agent", f"{press}, bu yuzden hemen yapmaniz gerekiyor{tail}."),
                ),
                1,
                scam_type,
            )

    return rows


def take_rows(candidates: list[dict[str, object]], target: int) -> list[dict[str, object]]:
    if len(candidates) < target:
        raise ValueError(f"Not enough candidates: {len(candidates)} < {target}")
    return candidates[:target]


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
        chunk = groups[key]
        selected.extend(chunk[:quota])
        leftovers.extend(chunk[quota:])

    if len(selected) < target:
        selected.extend(leftovers[: target - len(selected)])
    return selected[:target]


def main() -> None:
    base_df = pd.read_csv(INPUT_PATH)
    base_df["text"] = base_df["text"].astype(str).map(normalize)
    existing = set(base_df["text"].str.lower())

    blocked: set[str] = set()
    if OOD_TEST_PATH.exists():
        ood_df = pd.read_csv(OOD_TEST_PATH)
        blocked = set(ood_df["text"].astype(str).map(normalize).str.lower())

    safe_candidates = build_hard_safe(existing, blocked)
    after_safe_seen = existing | {row["text"].lower() for row in safe_candidates}
    scam_candidates = build_robust_scam(after_safe_seen, blocked)

    safe_rows = take_balanced(safe_candidates, TARGET_SAFE_ADD)
    scam_rows = take_balanced(scam_candidates, TARGET_SCAM_ADD)

    add_df = pd.DataFrame(safe_rows + scam_rows)
    out_df = pd.concat([base_df, add_df], ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    leaked = sorted(set(out_df["text"].str.lower()) & blocked)
    if leaked:
        raise ValueError(f"OOD test leakage detected: {len(leaked)} rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    label_counts = Counter(out_df["label"])
    scam_type_counts = Counter(out_df["scam_type"])
    report_lines = [
        "DataSetV7 robust generation report",
        f"Input: {INPUT_PATH}",
        f"Output: {OUTPUT_PATH}",
        f"OOD blocked rows: {len(blocked)}",
        f"Base rows: {len(base_df)}",
        f"Generated hard safe candidates: {len(safe_candidates)}",
        f"Generated robust scam candidates: {len(scam_candidates)}",
        f"Added hard safe rows: {len(safe_rows)}",
        f"Added robust scam rows: {len(scam_rows)}",
        f"Final rows: {len(out_df)}",
        f"Duplicate texts: {int(out_df.duplicated(subset=['text']).sum())}",
        f"Label counts: {dict(label_counts)}",
        "Top scam_type counts:",
    ]
    for key, value in scam_type_counts.most_common(30):
        report_lines.append(f"  {key}: {value}")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
