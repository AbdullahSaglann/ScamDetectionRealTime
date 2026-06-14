#!/usr/bin/env python3
"""Create DataSetV5 by adding daily and hard-negative safe dialogues."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/final/DataSetV4.csv")
OUTPUT_PATH = Path("data/final/DataSetV5.csv")
REPORT_PATH = Path("data/final/DataSetV5_report.txt")


def dialogue(*turns: tuple[str, str]) -> str:
    return " ".join(f"[{speaker.upper()}]: {text.strip()}" for speaker, text in turns)


def make_daily_dialogues() -> list[str]:
    people = ["Anne", "Mert", "Ayşe", "Burak", "Elif", "Can", "Zeynep", "Ömer"]
    activities = [
        "kahve içmeye",
        "markete",
        "yürüyüşe",
        "kütüphaneye",
        "spor salonuna",
        "sinemaya",
        "toplantıya",
        "ders çalışmaya",
    ]
    times = ["saat üçte", "akşam yedide", "yarın sabah", "öğleden sonra", "cumartesi günü"]
    places = ["Kadıköy'de", "okulun önünde", "ofiste", "evin yakınında", "metro çıkışında"]

    rows: list[str] = []
    for person in people:
        for activity in activities:
            for time in times:
                rows.append(
                    dialogue(
                        ("agent", f"{person}, {time} {activity} gidelim mi?"),
                        ("victim", f"Olur, {places[(len(rows) + 1) % len(places)]} buluşuruz."),
                        ("agent", "Tamam, yola çıkınca haber veririm."),
                    )
                )
                rows.append(
                    dialogue(
                        ("victim", f"{person}, bugün biraz yoğunum."),
                        ("agent", f"Sorun değil, {time} tekrar konuşuruz."),
                        ("victim", "Tamamdır, teşekkür ederim."),
                    )
                )
    return rows


def make_work_school_dialogues() -> list[str]:
    topics = [
        "sunum dosyasını",
        "rapor taslağını",
        "ders notlarını",
        "toplantı gündemini",
        "proje planını",
        "anket sonuçlarını",
    ]
    actions = [
        "mail ile paylaşabilir misin",
        "akşama kadar kontrol eder misin",
        "yarın birlikte gözden geçirelim mi",
        "son halini klasöre ekler misin",
    ]
    rows: list[str] = []
    for topic in topics:
        for action in actions:
            rows.append(
                dialogue(
                    ("agent", f"Merhaba, {topic} {action}?"),
                    ("victim", "Tabii, birazdan bakıp dönüş yaparım."),
                    ("agent", "Teşekkürler, acil değil."),
                )
            )
            rows.append(
                dialogue(
                    ("victim", f"{topic.capitalize()} üzerinde küçük bir düzeltme yaptım."),
                    ("agent", "Gördüm, gayet iyi olmuş."),
                    ("victim", "Eksik yer varsa haber verirsin."),
                )
            )
    return rows


def make_legit_service_hard_negatives() -> list[str]:
    rows: list[str] = []

    cargo_companies = ["PTT Kargo", "Yurtiçi Kargo", "Aras Kargo", "MNG Kargo", "Sürat Kargo"]
    for company in cargo_companies:
        rows.extend(
            [
                dialogue(
                    ("agent", f"{company} bilgilendirme hattından arıyorum."),
                    ("agent", "Paketiniz yarın dağıtıma çıkacak, ödeme veya link gerekmiyor."),
                    ("victim", "Teşekkür ederim, resmi uygulamadan da kontrol ederim."),
                ),
                dialogue(
                    ("agent", f"{company} teslimat bilgilendirmesidir."),
                    ("agent", "Adresiniz sistemde kayıtlı, sizden şifre veya kart bilgisi istenmez."),
                    ("victim", "Anladım, kapıda kimlik teyidi yapılabilir."),
                ),
                dialogue(
                    ("victim", "Kargo durumunu öğrenmek istiyorum."),
                    ("agent", "Resmi uygulama veya çağrı merkezi üzerinden gönderi numarasıyla kontrol edebilirsiniz."),
                    ("victim", "Tamam, herhangi bir linke tıklamayacağım."),
                ),
            ]
        )

    banks = ["Ziraat Bankası", "İş Bankası", "Garanti BBVA", "Akbank", "Yapı Kredi"]
    for bank in banks:
        rows.extend(
            [
                dialogue(
                    ("agent", f"{bank} güvenlik bilgilendirmesidir."),
                    ("agent", "Bankamız sizden telefonla kart şifresi, SMS kodu veya internet bankacılığı parolası istemez."),
                    ("victim", "Bilgilendirme için teşekkür ederim."),
                ),
                dialogue(
                    ("victim", "Mobil bankacılığa girişte sorun yaşıyorum."),
                    ("agent", "Lütfen yalnızca resmi uygulamadan işlem yapın, şifrenizi bizimle paylaşmayın."),
                    ("victim", "Tamam, müşteri hizmetlerini resmi numaradan arayacağım."),
                ),
                dialogue(
                    ("agent", f"{bank} işlem bilgilendirmesi gönderdi."),
                    ("agent", "Bu mesaj sadece bilgilendirme amaçlıdır, ödeme veya link talebi içermez."),
                    ("victim", "Anladım, işlem bana ait değilse resmi kanaldan bildireceğim."),
                ),
            ]
        )

    public_services = ["e-Devlet", "MHRS", "belediye", "vergi dairesi", "nüfus müdürlüğü"]
    for service in public_services:
        rows.extend(
            [
                dialogue(
                    ("agent", f"{service} işlemleri için resmi site veya uygulamayı kullanmanız önerilir."),
                    ("agent", "Kimseyle şifre, doğrulama kodu veya kişisel bilgilerinizi paylaşmayın."),
                    ("victim", "Teşekkürler, resmi adresten kontrol edeceğim."),
                ),
                dialogue(
                    ("victim", f"{service} randevumun saatini kontrol etmek istiyorum."),
                    ("agent", "Randevu bilgisi resmi uygulamada görünür, telefonla ödeme istenmez."),
                    ("victim", "Tamam, uygulamadan bakacağım."),
                ),
            ]
        )

    return rows


def make_safe_keyword_variants() -> list[str]:
    risky_words = [
        "şifre",
        "SMS kodu",
        "kart bilgisi",
        "IBAN",
        "link",
        "ödeme",
        "kargo",
        "banka",
        "e-Devlet",
        "dolandırıcılık",
    ]
    contexts = [
        "Bu konuda güvenlik eğitimi yapıyoruz.",
        "Bu sadece bilgilendirme amaçlı bir uyarıdır.",
        "Resmi kanallar dışında işlem yapılmamalıdır.",
        "Kimse sizden bu bilgiyi telefonla istememelidir.",
        "Şüpheli durumda kurumun resmi numarası aranmalıdır.",
    ]
    rows: list[str] = []
    for word in risky_words:
        for context in contexts:
            rows.append(
                dialogue(
                    ("agent", f"{word} hakkında kısa bir güvenlik hatırlatması yapıyoruz."),
                    ("agent", context),
                    ("victim", "Anladım, kişisel bilgilerimi kimseyle paylaşmayacağım."),
                )
            )
            rows.append(
                dialogue(
                    ("victim", f"{word} geçen bir mesaj aldım ama emin olamadım."),
                    ("agent", "Mesajdaki bağlantıya tıklamayın, resmi uygulamadan kontrol edin."),
                    ("victim", "Tamam, bu bir işlem talebi değil sadece güvenlik kontrolü."),
                )
            )
    return rows


def make_plain_transcript_safe_rows() -> list[str]:
    openings = [
        "Bugün hava güzel, dışarı çıkıp biraz yürüyelim.",
        "Akşam yemeği için ne hazırlayalım?",
        "Toplantı saatini yarına alabilir miyiz?",
        "Ders notlarını paylaştım, uygun olunca bakarsın.",
        "Kahve molasına çıkıyorum, bir şey ister misin?",
        "Otobüse bindim, yaklaşık yirmi dakikaya oradayım.",
        "Market listesini kontrol ettim, eksik bir şey kalmadı.",
        "Hafta sonu aile ziyaretine gideceğiz.",
        "Film saatini kontrol ettim, biletler hazır.",
        "Bugünkü egzersiz programını biraz hafif tuttum.",
    ]
    responses = [
        "Tamam, haber verdiğin için teşekkür ederim.",
        "Olur, bana uygun.",
        "Ben de birazdan çıkıyorum.",
        "Gördüm, akşam cevap yazarım.",
        "Sorun değil, sonra konuşuruz.",
    ]
    rows: list[str] = []
    for opening in openings:
        for response in responses:
            rows.append(f"{opening} {response}")
            rows.append(dialogue(("agent", opening), ("victim", response)))
    return rows


def make_broad_daily_safe_rows() -> list[str]:
    subjects = [
        "Ben",
        "Biz",
        "Annem",
        "Babam",
        "Kardeşim",
        "Arkadaşım",
        "Hocamız",
        "Takım lideri",
        "Komşumuz",
        "Kuzenim",
    ]
    actions = [
        "bugün biraz geç kalacağımı söyledi",
        "yarın erken çıkmamız gerektiğini hatırlattı",
        "akşam yemeği için alışveriş listesi hazırladı",
        "toplantı saatinin değiştiğini yazdı",
        "hafta sonu planını netleştirmek istedi",
        "ders notlarını kontrol etmemi istedi",
        "otobüs saatini tekrar kontrol etti",
        "doktor randevusuna gideceğini söyledi",
        "spor programını hafifleteceğini belirtti",
        "evde eksik olan malzemeleri listeledi",
    ]
    closings = [
        "Ben de tamam diye cevap verdim.",
        "Uygun olduğumu söyledim.",
        "Sonra tekrar konuşacağımızı yazdım.",
        "Gerekirse haberleşiriz dedim.",
        "Bu konuşmada ödeme veya kişisel bilgi talebi yoktu.",
    ]
    rows: list[str] = []
    for subject in subjects:
        for action in actions:
            for closing in closings:
                rows.append(f"{subject} {action}. {closing}")
                rows.append(
                    dialogue(
                        ("agent", f"{subject} {action}."),
                        ("victim", closing),
                    )
                )
    return rows


def make_legit_keyword_conversations() -> list[str]:
    institutions = [
        "banka",
        "kargo firması",
        "hastane",
        "okul",
        "belediye",
        "operatör",
        "sigorta şirketi",
        "apartman yönetimi",
    ]
    safe_actions = [
        "resmi uygulamadan kontrol etmemi önerdi",
        "kişisel bilgi paylaşmamam gerektiğini söyledi",
        "ödeme yapmamı istemedi",
        "linke tıklamamı istemedi",
        "SMS kodumu kimseyle paylaşmamamı hatırlattı",
        "yalnızca bilgilendirme yaptığını belirtti",
    ]
    topics = [
        "randevu saati",
        "teslimat durumu",
        "hesap özeti",
        "aidat bilgilendirmesi",
        "başvuru sonucu",
        "fatura dönemi",
        "güvenlik uyarısı",
        "çalışma saatleri",
    ]
    rows: list[str] = []
    for institution in institutions:
        for topic in topics:
            for action in safe_actions:
                rows.append(
                    dialogue(
                        ("agent", f"{institution.capitalize()} {topic} hakkında bilgilendirme yaptı."),
                        ("agent", action.capitalize() + "."),
                        ("victim", "Ben de resmi kanallardan kontrol edeceğimi söyledim."),
                    )
                )
    return rows


def make_customer_support_safe_rows() -> list[str]:
    products = [
        "internet paketi",
        "telefon faturası",
        "kargo teslimatı",
        "market siparişi",
        "kitap siparişi",
        "otobüs bileti",
        "sinema bileti",
        "randevu kaydı",
        "abonelik planı",
        "garanti kaydı",
    ]
    requests = [
        "durumunu öğrenmek istiyorum",
        "tarihini değiştirmek istiyorum",
        "iptal koşullarını öğrenmek istiyorum",
        "resmi uygulamada görünmüyor, kontrol eder misiniz",
        "bilgilendirme mesajı geldi, doğruluğunu kontrol etmek istiyorum",
    ]
    replies = [
        "Size şifre veya SMS kodu sormadan yardımcı olabilirim.",
        "Resmi uygulamada görünen bilgileri esas almanız gerekir.",
        "Bu işlem için ödeme bağlantısı göndermiyoruz.",
        "Kimlik doğrulaması sadece resmi kanal üzerinden yapılır.",
        "Kişisel bilgilerinizi burada paylaşmanıza gerek yok.",
    ]
    rows: list[str] = []
    for product in products:
        for request in requests:
            for reply in replies:
                rows.append(
                    dialogue(
                        ("victim", f"{product.capitalize()} {request}."),
                        ("agent", reply),
                        ("victim", "Tamam, teşekkür ederim."),
                    )
                )
    return rows


def make_augmented_safe_rows() -> pd.DataFrame:
    texts: list[str] = []
    texts.extend(make_daily_dialogues())
    texts.extend(make_work_school_dialogues())
    texts.extend(make_legit_service_hard_negatives())
    texts.extend(make_safe_keyword_variants())
    texts.extend(make_plain_transcript_safe_rows())
    texts.extend(make_broad_daily_safe_rows())
    texts.extend(make_legit_keyword_conversations())
    texts.extend(make_customer_support_safe_rows())

    unique_texts = list(dict.fromkeys(texts))
    return pd.DataFrame(
        {
            "text": unique_texts,
            "label": 0,
            "scam_type": "legit",
        }
    )


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    new_safe = make_augmented_safe_rows()

    before_count = len(df)
    before_labels = Counter(df["label"].tolist())
    combined = pd.concat([df, new_safe], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    added_count = len(combined) - before_count
    after_labels = Counter(combined["label"].tolist())
    duplicate_count = int(combined["text"].duplicated().sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    report = [
        f"Input file: {INPUT_PATH.name}",
        f"Output file: {OUTPUT_PATH.name}",
        f"Original rows: {before_count}",
        f"Generated safe rows: {len(new_safe)}",
        f"Actually added rows after de-duplication: {added_count}",
        f"Final rows: {len(combined)}",
        f"Duplicate text count: {duplicate_count}",
        "",
        "Label counts before:",
        f"label 0: {before_labels.get(0, 0)}",
        f"label 1: {before_labels.get(1, 0)}",
        "",
        "Label counts after:",
        f"label 0: {after_labels.get(0, 0)}",
        f"label 1: {after_labels.get(1, 0)}",
        "",
        "Added safe row families:",
        "- daily casual dialogues",
        "- work and school conversations",
        "- legitimate cargo/banking/public-service hard negatives",
        "- safe keyword warnings containing scam-trigger words",
        "- plain transcript style safe utterances without [AGENT]/[VICTIM] tags",
        "- broader family, appointment, shopping, and commute transcripts",
        "- customer-support conversations with safe handling of risky keywords",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))


if __name__ == "__main__":
    main()
