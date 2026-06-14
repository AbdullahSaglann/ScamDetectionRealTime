#!/usr/bin/env python3
"""Create a 30k real-time oriented dataset with better safe/scam coverage."""

from __future__ import annotations

from collections import Counter
from itertools import cycle
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/final/DataSetV4.csv")
OUTPUT_PATH = Path("data/final/DataSetV6.csv")
REPORT_PATH = Path("data/final/DataSetV6_report.txt")

TARGET_TOTAL = 30_000
TARGET_SAFE = 15_500
TARGET_SCAM = TARGET_TOTAL - TARGET_SAFE


def dialogue(*turns: tuple[str, str]) -> str:
    return " ".join(f"[{speaker.upper()}]: {text.strip()}" for speaker, text in turns)


def add_unique(rows: list[dict[str, object]], seen: set[str], text: str, label: int, scam_type: str) -> None:
    normalized = " ".join(text.split())
    if normalized and normalized not in seen:
        seen.add(normalized)
        rows.append({"text": normalized, "label": label, "scam_type": scam_type})


def safe_candidates(existing: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen = set(existing)

    people = ["Mert", "Ayşe", "Elif", "Can", "Zeynep", "Burak", "Deniz", "Ece", "Kerem", "Selin"]
    plan_times = {
        "kahve içmek": ["bugün akşam", "öğleden sonra", "cumartesi günü", "saat altıda"],
        "markete uğramak": ["bugün akşam", "yarın sabah", "öğleden sonra", "cumartesi günü"],
        "ders notlarına bakmak": ["yarın sabah", "öğleden sonra", "cumartesi günü", "saat altıda"],
        "sunumu gözden geçirmek": ["yarın sabah", "öğleden sonra", "saat altıda"],
        "yürüyüş yapmak": ["bugün akşam", "yarın sabah", "öğleden sonra", "cumartesi günü"],
        "akşam yemeği hazırlamak": ["bugün akşam", "cumartesi günü", "saat altıda"],
        "kütüphanede buluşmak": ["yarın sabah", "öğleden sonra", "cumartesi günü"],
        "film saatini kontrol etmek": ["bugün akşam", "öğleden sonra", "cumartesi günü"],
        "spor programını konuşmak": ["yarın sabah", "öğleden sonra", "cumartesi günü"],
        "hafta sonu planını netleştirmek": ["bugün akşam", "öğleden sonra", "cumartesi günü"],
    }
    places = ["evin yakınında", "okulun önünde", "ofiste", "metro çıkışında", "kafenin önünde"]
    replies = ["Uygun olur", "Benim için tamam", "Biraz geç gelebilirim", "Olur, haberleşiriz"]

    for person in people:
        for plan, times in plan_times.items():
            for time in times:
                for reply in replies:
                    place = places[(len(rows) + len(person)) % len(places)]
                    add_unique(
                        rows,
                        seen,
                        dialogue(
                            ("agent", f"{person}, {time} {plan} ister misin?"),
                            ("victim", f"{reply}. {place.capitalize()} buluşabiliriz."),
                            ("agent", "Tamam, yola çıkınca haber veririm."),
                        ),
                        0,
                        "legit",
                    )
                    add_unique(
                        rows,
                        seen,
                        f"{person} ile {time} {plan} için konuştuk. {reply}. Herhangi bir ödeme, link veya kişisel bilgi talebi olmadı.",
                        0,
                        "legit",
                    )

    service_templates = [
        (
            "banka",
            "Hesap özetiniz hazır olduğunda resmi uygulamadan görüntüleyebilirsiniz.",
            "Kart şifresi, SMS kodu veya internet bankacılığı parolası istenmez.",
        ),
        (
            "kargo",
            "Paketiniz yarın dağıtıma çıkacak.",
            "Bu bilgilendirme için ödeme bağlantısı gönderilmez.",
        ),
        (
            "hastane",
            "Randevunuz sistemde görünüyor.",
            "Randevu değişikliği yalnızca resmi uygulama veya çağrı merkeziyle yapılır.",
        ),
        (
            "belediye",
            "Başvurunuzun durumu resmi portalda güncellendi.",
            "Telefonla ödeme veya şifre talebi yapılmaz.",
        ),
        (
            "operatör",
            "Fatura döneminiz yaklaşmıştır.",
            "İşlem yapmak isterseniz resmi uygulamayı kullanabilirsiniz.",
        ),
    ]
    names = ["Ziraat", "Garanti", "PTT", "MHRS", "Turkcell", "Vodafone", "İSKİ", "e-Devlet"]
    for institution, info, safety in service_templates:
        for name in names:
            for channel in ["resmi uygulama", "çağrı merkezi", "web sitesi", "şube"]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{name} {institution} bilgilendirme hattından arıyorum."),
                        ("agent", info),
                        ("agent", safety),
                        ("victim", f"Teşekkürler, gerekirse {channel} üzerinden kontrol ederim."),
                    ),
                    0,
                    "legit",
                )
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("victim", f"{name} ile ilgili bir bilgilendirme mesajı aldım, doğruluğunu kontrol etmek istiyorum."),
                        ("agent", f"En güvenli yol {channel} üzerinden kontrol etmektir."),
                        ("agent", "Biz sizden şifre, kod veya kart bilgisi istemiyoruz."),
                        ("victim", "Anladım, bağlantıya tıklamadan resmi kanaldan bakacağım."),
                    ),
                    0,
                    "legit",
                )

    support_topics = [
        "internet paketi",
        "telefon faturası",
        "kargo teslimatı",
        "market siparişi",
        "kitap siparişi",
        "otobüs bileti",
        "sinema bileti",
        "doktor randevusu",
        "abonelik planı",
        "garanti kaydı",
        "aidat bilgilendirmesi",
        "ders kaydı",
    ]
    support_requests = [
        "durumunu öğrenmek istiyorum",
        "tarihini değiştirmek istiyorum",
        "iptal koşullarını öğrenmek istiyorum",
        "resmi uygulamada görünmüyor, kontrol eder misiniz",
        "bilgilendirme mesajı geldi, doğruluğunu kontrol etmek istiyorum",
    ]
    safe_replies = [
        "Size şifre veya SMS kodu sormadan yardımcı olabilirim.",
        "Resmi uygulamada görünen bilgileri esas almanız gerekir.",
        "Bu işlem için ödeme bağlantısı göndermiyoruz.",
        "Kimlik doğrulaması sadece resmi kanal üzerinden yapılır.",
        "Kişisel bilgilerinizi burada paylaşmanıza gerek yok.",
    ]
    for topic in support_topics:
        for request in support_requests:
            for reply in safe_replies:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("victim", f"{topic.capitalize()} hakkında {request}."),
                        ("agent", reply),
                        ("victim", "Tamam, teşekkür ederim."),
                    ),
                    0,
                    "legit",
                )

    awareness_words = ["şifre", "SMS kodu", "kart bilgisi", "IBAN", "link", "ödeme", "kargo", "banka", "e-Devlet"]
    awareness_contexts = [
        "Bu sadece güvenlik bilgilendirmesidir.",
        "Resmi kurumlar bu bilgiyi telefonla istemez.",
        "Şüpheli durumda resmi numarayı kendiniz arayın.",
        "Mesajdaki bağlantıya tıklamadan uygulamadan kontrol edin.",
        "Bu konuşmada para transferi veya gizli bilgi talebi yoktur.",
    ]
    for word in awareness_words:
        for context in awareness_contexts:
            for ending in ["Bilgilendirme için teşekkürler.", "Anladım, dikkat edeceğim.", "Resmi kanaldan kontrol ederim."]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{word} konusunda kısa bir hatırlatma yapmak istiyoruz."),
                        ("agent", context),
                        ("victim", ending),
                    ),
                    0,
                    "legit",
                )

    daily_subjects = [
        "annem",
        "babam",
        "kardeşim",
        "arkadaşım",
        "hocam",
        "komşum",
        "kuzenim",
        "takım arkadaşım",
        "iş arkadaşım",
        "ev arkadaşım",
        "servis şoförü",
        "apartman görevlisi",
    ]
    daily_events = [
        "akşam yemeğine biraz geç geleceğini söyledi",
        "market listesini tamamladığını yazdı",
        "otobüsün geciktiğini haber verdi",
        "toplantı saatinin değiştiğini söyledi",
        "doktor randevusuna gideceğini hatırlattı",
        "ders notlarını paylaştığını söyledi",
        "kütüphanede yer ayırdığını belirtti",
        "spor salonuna bugün gitmeyeceğini söyledi",
        "kahve içmek için müsait olup olmadığımı sordu",
        "evde eksik olan malzemeleri kontrol etti",
        "hafta sonu planını tekrar konuşmak istedi",
        "dosyanın son halini e-postayla gönderdi",
        "kargo teslim saatini resmi uygulamadan kontrol etti",
        "fatura tarihini takvimine ekledi",
        "sinemaya gitmek için uygun seansı baktı",
    ]
    daily_responses = [
        "Ben de tamam dedim.",
        "Uygun olduğumu söyledim.",
        "Sonra tekrar konuşacağımızı yazdım.",
        "Gerekirse haberleşiriz dedim.",
        "Herhangi bir ödeme veya şifre konuşulmadı.",
        "Bu sıradan bir günlük konuşmaydı.",
        "Kişisel bilgi paylaşımı istenmedi.",
        "Resmi işlem gerektiren bir konu değildi.",
    ]
    for subject in daily_subjects:
        for event in daily_events:
            for response in daily_responses:
                add_unique(rows, seen, f"{subject.capitalize()} {event}. {response}", 0, "legit")
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{subject.capitalize()} {event}."),
                        ("victim", response),
                    ),
                    0,
                    "legit",
                )

    neutral_transcript_starts = [
        "Bugün toplantıdan sonra kısa bir yürüyüş yaptık",
        "Ders çıkışı kafede oturup proje planını konuştuk",
        "Akşam eve gelmeden önce markete uğradım",
        "Sabah otobüs biraz geciktiği için derse zor yetiştim",
        "Hafta sonu aile ziyareti için saat belirledik",
        "Ofiste raporun son bölümünü birlikte kontrol ettik",
        "Kütüphanede kaynak listesini düzenledik",
        "Spor sonrası yemek planını değiştirdik",
        "Arkadaşım yeni taşındığı ev için birkaç öneri istedi",
        "Bugünkü hava durumuna göre planı erteledik",
    ]
    neutral_transcript_ends = [
        "konuşma tamamen günlük bir plan hakkındaydı",
        "kimse para ya da kod istemedi",
        "linke tıklama veya ödeme yapma konusu geçmedi",
        "kişisel bilgi paylaşımı istenmedi",
        "sadece saat ve yer bilgisi netleştirildi",
        "güvenlik riski oluşturacak bir talep yoktu",
        "normal bir arkadaş sohbeti olarak tamamlandı",
        "resmi işlem veya banka konusu açılmadı",
    ]
    for start in neutral_transcript_starts:
        for end in neutral_transcript_ends:
            for detail in ["sonra eve döndüm", "biraz daha konuştuk", "planı yarına aldık", "herkes uygun olduğunu söyledi"]:
                add_unique(rows, seen, f"{start}; {detail}. {end}.", 0, "legit")

    schedule_topics = [
        "toplantı saati",
        "ders programı",
        "servis saati",
        "doktor randevusu",
        "market alışverişi",
        "aile ziyareti",
        "spor dersi",
        "sinema planı",
        "kütüphane buluşması",
        "proje görüşmesi",
        "fatura son ödeme tarihi",
        "kargo teslim aralığı",
        "apartman toplantısı",
        "yemek rezervasyonu",
        "otobüs bileti",
    ]
    schedule_updates = [
        "bir saat ileri alındı",
        "yarına ertelendi",
        "bugün için uygun görünüyor",
        "resmi uygulamada güncellendi",
        "takvime eklendi",
        "katılımcılara haber verildi",
        "iptal edilmedi, sadece saati değişti",
        "herkesin uygun olduğu saate çekildi",
    ]
    schedule_replies = [
        "Tamam, ben de not aldım.",
        "Uygun, o saatte orada olurum.",
        "Teşekkürler, hatırlattığın iyi oldu.",
        "Ben de uygulamadan kontrol ederim.",
        "Sorun değil, plana uyarım.",
        "Gerekirse tekrar haberleşiriz.",
    ]
    for topic in schedule_topics:
        for update in schedule_updates:
            for reply in schedule_replies:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{topic.capitalize()} {update}."),
                        ("victim", reply),
                        ("agent", "Bu sadece bilgilendirme, herhangi bir ödeme veya kod gerekmiyor."),
                    ),
                    0,
                    "legit",
                )
                add_unique(rows, seen, f"{topic.capitalize()} {update}. {reply}", 0, "legit")

    checkin_topics = [
        "bugünkü ders",
        "yarınki toplantı",
        "akşam yemeği",
        "hafta sonu planı",
        "spor programı",
        "market alışverişi",
        "ev temizliği",
        "proje teslimi",
        "randevu saati",
        "otobüs güzergahı",
        "hava durumu",
        "kitap kulübü buluşması",
    ]
    checkin_questions = [
        "sana uygun mu",
        "bir değişiklik var mı",
        "yardım ister misin",
        "saatini netleştirelim mi",
        "benim hazırlamam gereken bir şey var mı",
        "bunu yarına alsak olur mu",
    ]
    checkin_answers = [
        "Uygun, sorun yok.",
        "Şimdilik değişiklik yok.",
        "Gerekirse haber veririm.",
        "Evet, saati netleştirelim.",
        "Hayır, her şey hazır.",
        "Yarına almak daha iyi olur.",
        "Olur, ben de takvime eklerim.",
        "Tamam, birazdan tekrar bakarım.",
        "Benim için uygun görünüyor.",
    ]
    for topic in checkin_topics:
        for question in checkin_questions:
            for answer in checkin_answers:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{topic.capitalize()} için {question}?"),
                        ("victim", answer),
                        ("agent", "Tamam, sadece planı netleştirmek istemiştim."),
                    ),
                    0,
                    "legit",
                )

    return rows


def scam_candidates(existing: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen = set(existing)

    banks = ["Ziraat", "Garanti", "Akbank", "Yapı Kredi", "İş Bankası"]
    amounts = ["4.850 TL", "12.300 TL", "28.750 TL", "45.000 TL", "97.500 TL"]
    for bank in banks:
        for amount in amounts:
            for issue in ["şüpheli işlem", "kart harcaması", "hesap bloke uyarısı", "internet bankacılığı girişi"]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{bank} güvenlik biriminden arıyorum, hesabınızda {amount} tutarında {issue} görünüyor."),
                        ("victim", "Ben böyle bir işlem yapmadım."),
                        ("agent", "İptal için size gelen SMS kodunu hemen söylemeniz gerekiyor."),
                        ("victim", "SMS kodumu paylaşmam, bankayı kendim arayacağım."),
                    ),
                    1,
                    "banka_kart_hesap",
                )
                add_unique(
                    rows,
                    seen,
                    f"{bank} adına arayan kişi {amount} tutarında {issue} olduğunu söyledi ve işlemi iptal etmek için SMS kodunu istedi.",
                    1,
                    "banka_kart_hesap",
                )

    cargo = ["PTT Kargo", "Yurtiçi Kargo", "Aras Kargo", "MNG Kargo"]
    fees = ["18,90 TL", "32,50 TL", "74,99 TL", "96,50 TL"]
    for company in cargo:
        for fee in fees:
            for reason in ["adres teyidi", "gümrük onayı", "teslimat güncellemesi", "depo bekleme ücreti"]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{company} adına arıyorum, paketiniz için {reason} gerekiyor."),
                        ("agent", f"Teslimatın iptal olmaması için {fee} ödemeyi gönderdiğim linkten yapın."),
                        ("victim", "Resmi uygulama dışında ödeme yapmam."),
                        ("agent", "Link süresi kısa, şimdi ödemezseniz paket iade edilir."),
                    ),
                    1,
                    "kargo_gumruk",
                )

    public_names = ["e-Devlet", "CİMER", "vergi dairesi", "SGK", "belediye"]
    claims = ["iade hakkınız var", "başvurunuz onaylandı", "ceza kaydınız görünüyor", "hesabınızda şüpheli giriş var"]
    for name in public_names:
        for claim in claims:
            for credential in ["e-Devlet şifrenizi", "SMS doğrulama kodunu", "kimlik bilgilerinizi", "kart numaranızı"]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{name} işlem merkezi gibi arıyorum, sistemde {claim}."),
                        ("agent", f"İşlemi tamamlamak için {credential} paylaşmanız gerekiyor."),
                        ("victim", "Resmi kurumlar bu bilgileri telefonda istemez."),
                        ("agent", "Dosya kapanmadan hemen işlem yapmalısınız."),
                    ),
                    1,
                    "edevlet_kamu_vergi",
                )

    social_apps = ["WhatsApp", "Instagram", "Facebook", "Telegram"]
    social_claims = ["hesabınız kapanacak", "çekiliş kazandınız", "doğrulama yapmanız gerekiyor", "şikayet kaydı oluştu"]
    for app in social_apps:
        for claim in social_claims:
            for action in ["gelen kodu söyleyin", "linkten giriş yapın", "kart bilgisiyle kimlik doğrulayın"]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{app} destek ekibinden yazıyorum, {claim}."),
                        ("agent", f"Hesabınızı korumak için {action}."),
                        ("victim", "Bunu resmi uygulamadan kontrol edeceğim."),
                        ("agent", "Geç kalırsanız hesabınız askıya alınabilir."),
                    ),
                    1,
                    "sosyal_medya_whatsapp",
                )

    products = ["telefon", "oyun konsolu", "laptop", "koltuk takımı", "bisiklet", "kamera"]
    deposits = ["500 TL", "1.000 TL", "1.750 TL", "2.500 TL"]
    for product in products:
        for deposit in deposits:
            for excuse in ["ilanı sizin için ayırmam", "kargoya vermem", "kapora kaydı açmam", "indirimli fiyatı sabitlemem"]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{product.capitalize()} ilanı için çok kişi yazıyor."),
                        ("agent", f"{excuse} için önce {deposit} kapora göndermeniz gerekiyor."),
                        ("victim", "Ürünü görmeden kapora göndermem."),
                        ("agent", "Hemen göndermezseniz başkasına satacağım."),
                    ),
                    1,
                    "pazar_yeri_ilan",
                )

    investment_assets = ["kripto paketi", "altın fonu", "borsa robotu", "döviz sinyali", "yatırım uygulaması"]
    returns = ["bir haftada yüzde 30", "ayda iki kat", "garantili günlük kazanç", "risksiz yüksek getiri"]
    for asset in investment_assets:
        for ret in returns:
            for first_payment in ["2.000 TL", "5.000 TL", "10.000 TL", "25.000 TL"]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{asset.capitalize()} ile {ret} kazanç sağlıyoruz."),
                        ("victim", "Garanti getiri gerçekçi gelmiyor."),
                        ("agent", f"Başlamak için bugün {first_payment} yatırmanız yeterli."),
                        ("victim", "Resmi lisans ve sözleşme olmadan ödeme yapmam."),
                    ),
                    1,
                    "yatirim_kripto",
                )

    jobs = ["evden paketleme", "çağrı merkezi", "anket doldurma", "veri girişi", "kurye başvurusu"]
    fees = ["350 TL", "750 TL", "1.250 TL", "2.000 TL"]
    for job in jobs:
        for fee in fees:
            for promise in ["hemen işe başlatacağız", "öncelikli kayıt açacağız", "ekipman göndereceğiz"]:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"{job.capitalize()} işi için başvurunuz onaylandı."),
                        ("agent", f"{promise}, sadece {fee} kayıt ücreti ödemeniz gerekiyor."),
                        ("victim", "İş başvurusu için ücret ödemem."),
                        ("agent", "Kontenjan dolmadan ödeme yapmalısınız."),
                    ),
                    1,
                    "sahte_is_teklifi",
                )

    legal_roles = ["polis", "savcı", "jandarma", "icra memuru"]
    accusations = ["adınız soruşturmada geçiyor", "hesabınız suç gelirleriyle ilişkilendirildi", "kimliğiniz yasa dışı işlemde kullanıldı"]
    valuables = ["altınlarınızı", "birikiminizi", "hesabınızdaki parayı", "dövizlerinizi"]
    for role in legal_roles:
        for accusation in accusations:
            for valuable in valuables:
                add_unique(
                    rows,
                    seen,
                    dialogue(
                        ("agent", f"Ben {role}, {accusation}."),
                        ("agent", f"Dosyanın temizlenmesi için {valuable} güvenli hesaba aktarmanız gerekiyor."),
                        ("victim", "Resmi kurum para transferi istemez, karakolu kendim arayacağım."),
                        ("agent", "Kimseye haber vermeyin, operasyon gizli yürütülüyor."),
                    ),
                    1,
                    "polis_emniyet_jandarma",
                )

    scenario_blocks = [
        {
            "scam_type": "banka_kart_hesap",
            "callers": ["banka güvenlik birimi", "kart işlem merkezi", "mobil bankacılık destek hattı", "hesap güvenlik servisi"],
            "issues": [
                "hesabınızda şüpheli giriş tespit edildi",
                "kartınızdan yüksek tutarlı işlem denendi",
                "internet bankacılığınız geçici olarak askıya alındı",
                "adınıza yeni cihazdan giriş yapıldı",
                "hesabınız güvenlik incelemesine alındı",
            ],
            "requests": ["SMS kodunu söyleyin", "kartınızın arkasındaki kodu okuyun", "şifrenizi geçici olarak paylaşın", "gönderdiğimiz linkten giriş yapın"],
            "pressures": ["işlemi hemen durdurmamız gerekiyor", "gecikirseniz tutar hesabınızdan düşer", "dosya kısa süre içinde kapanacak"],
        },
        {
            "scam_type": "kargo_gumruk",
            "callers": ["kargo teslimat merkezi", "gümrük işlem birimi", "depo takip servisi", "adres doğrulama hattı"],
            "issues": [
                "paketiniz teslim edilemedi",
                "gönderiniz gümrükte bekliyor",
                "adres bilgileriniz eksik görünüyor",
                "teslimat dosyanız askıya alındı",
                "paketiniz için ek işlem açıldı",
            ],
            "requests": ["küçük ücreti linkten ödeyin", "kart bilginizi doğrulayın", "adres formunu bağlantıdan doldurun", "ödeme dekontunu bize gönderin"],
            "pressures": ["bugün yapılmazsa paket iade olur", "süre dolmadan işlem yapmalısınız", "teslimat ancak bu şekilde devam eder"],
        },
        {
            "scam_type": "edevlet_kamu_vergi",
            "callers": ["e-Devlet destek hattı", "vergi iade birimi", "kamu başvuru merkezi", "CİMER işlem hattı"],
            "issues": [
                "adınıza iade kaydı oluştu",
                "başvurunuz eksik doğrulama nedeniyle bekliyor",
                "hesabınızda şüpheli giriş görünüyor",
                "kimlik doğrulaması tamamlanmadı",
                "dosyanız kapanmadan onay gerekiyor",
            ],
            "requests": ["e-Devlet şifrenizi söyleyin", "SMS kodunu paylaşın", "kimlik ve kart bilgilerinizi girin", "gönderdiğimiz bağlantıdan giriş yapın"],
            "pressures": ["aksi halde başvurunuz iptal edilir", "süre bitmeden onay vermelisiniz", "bu işlem telefonla tamamlanacak"],
        },
        {
            "scam_type": "sosyal_medya_whatsapp",
            "callers": ["sosyal medya destek ekibi", "WhatsApp güvenlik servisi", "hesap doğrulama merkezi", "çekiliş destek hattı"],
            "issues": [
                "hesabınız şikayet nedeniyle kapanacak",
                "çekiliş ödülünüz hazırlandı",
                "profiliniz doğrulama bekliyor",
                "hesabınıza farklı cihazdan giriş yapıldı",
                "ödül teslimi için son aşamadasınız",
            ],
            "requests": ["doğrulama kodunu gönderin", "linkten hesabınıza giriş yapın", "kart bilgisiyle kimlik doğrulayın", "ödül vergisini ödeyin"],
            "pressures": ["gecikirseniz hakkınız yanar", "hesabınız askıya alınabilir", "işlem sadece bugün geçerli"],
        },
        {
            "scam_type": "pazar_yeri_ilan",
            "callers": ["ilan sahibi", "alıcı", "satıcı destek hesabı", "kurye yönlendirme servisi"],
            "issues": [
                "ürünü sizin için ayırabilirim",
                "kargo işlemini hemen başlatabilirim",
                "güvenli alışveriş kaydı açacağım",
                "indirimli fiyatı sabitleyebilirim",
                "başka alıcılar da bekliyor",
            ],
            "requests": ["kapora gönderin", "ödeme bağlantısından işlem yapın", "IBAN'a ön ödeme yapın", "kart bilginizi forma girin"],
            "pressures": ["hemen göndermezseniz başkasına satılır", "süre kısa", "kampanya birazdan kapanıyor"],
        },
        {
            "scam_type": "yatirim_kripto",
            "callers": ["yatırım danışmanı", "kripto işlem uzmanı", "borsa sinyal ekibi", "finans koçu"],
            "issues": [
                "garantili kazanç fırsatı açıldı",
                "düşük riskli özel portföy hazır",
                "bugüne özel yüksek getiri var",
                "otomatik alım satım sistemi aktif",
                "sınırlı kontenjanlı yatırım grubu kuruldu",
            ],
            "requests": ["ilk yatırımı yapın", "uygulamayı indirip para aktarın", "hesap açılışı için ödeme gönderin", "cüzdan adresine transfer yapın"],
            "pressures": ["fırsat bugün kapanıyor", "gecikirseniz kazancı kaçırırsınız", "kontenjan dolmadan katılmalısınız"],
        },
        {
            "scam_type": "sahte_is_teklifi",
            "callers": ["insan kaynakları", "evden çalışma ekibi", "başvuru destek hattı", "işe alım danışmanı"],
            "issues": [
                "başvurunuz ön onay aldı",
                "evden çalışma kaydınız açıldı",
                "ekipman gönderimi hazırlanıyor",
                "öncelikli aday listesine alındınız",
                "eğitim kaydınız oluşturuldu",
            ],
            "requests": ["kayıt ücretini ödeyin", "ekipman depozitosu gönderin", "IBAN'a işlem bedeli yatırın", "kimlik fotoğrafınızı ve ücret dekontunu gönderin"],
            "pressures": ["kontenjan bitmeden tamamlamalısınız", "bugün ödemezseniz başvuru düşer", "süreç hemen başlatılacak"],
        },
        {
            "scam_type": "polis_emniyet_jandarma",
            "callers": ["polis", "savcı", "jandarma", "adliye görevlisi"],
            "issues": [
                "adınız soruşturmada geçiyor",
                "kimliğiniz yasa dışı işlemde kullanıldı",
                "hesabınız suç gelirleriyle ilişkilendirildi",
                "dosyanız gizli operasyon kapsamında inceleniyor",
                "üzerinize kayıtlı hat suçta kullanıldı",
            ],
            "requests": ["parayı güvenli hesaba aktarın", "altınları belirtilen kişiye teslim edin", "kimseye haber vermeden talimatı uygulayın", "hesap bilgilerinizi paylaşın"],
            "pressures": ["operasyon gizli yürütülüyor", "gecikirseniz hakkınızda işlem yapılır", "şimdi iş birliği yapmanız gerekiyor"],
        },
    ]

    channels_by_type = {
        "banka_kart_hesap": ["telefonla", "SMS ile"],
        "kargo_gumruk": ["SMS ile", "WhatsApp üzerinden"],
        "edevlet_kamu_vergi": ["telefonla", "SMS ile", "WhatsApp üzerinden"],
        "sosyal_medya_whatsapp": ["WhatsApp üzerinden", "sosyal medya mesajıyla"],
        "pazar_yeri_ilan": ["pazar yeri mesajıyla", "WhatsApp üzerinden"],
        "yatirim_kripto": ["telefonla", "WhatsApp üzerinden", "sosyal medya mesajıyla"],
        "sahte_is_teklifi": ["telefonla", "WhatsApp üzerinden", "e-posta ile"],
        "polis_emniyet_jandarma": ["telefonla"],
    }
    for block in scenario_blocks:
        channels = channels_by_type.get(block["scam_type"], ["telefonla", "SMS ile"])
        for caller in block["callers"]:
            for issue in block["issues"]:
                for request in block["requests"]:
                    for pressure in block["pressures"]:
                        for channel in channels:
                            add_unique(
                                rows,
                                seen,
                                dialogue(
                                    ("agent", f"{caller.capitalize()} olarak {channel} ulaşıyorum, {issue}."),
                                    ("victim", "Bu konuda emin değilim, resmi kanaldan kontrol etmek istiyorum."),
                                    ("agent", f"{request.capitalize()}; {pressure}."),
                                    ("victim", "Bu talep güvenli görünmüyor, işlemi durduruyorum."),
                                ),
                                1,
                                block["scam_type"],
                            )
                            add_unique(
                                rows,
                                seen,
                                f"{caller.capitalize()} gibi konuşan kişi {channel} ulaşıp {issue} dedi ve {request.lower()} talebinde bulundu; ayrıca {pressure}.",
                                1,
                                block["scam_type"],
                            )

    return rows


def take_needed(candidates: list[dict[str, object]], needed: int) -> list[dict[str, object]]:
    if len(candidates) < needed:
        raise RuntimeError(f"Not enough candidates: needed={needed}, available={len(candidates)}")
    return candidates[:needed]


def take_balanced_by_type(candidates: list[dict[str, object]], needed: int) -> list[dict[str, object]]:
    if len(candidates) < needed:
        raise RuntimeError(f"Not enough candidates: needed={needed}, available={len(candidates)}")

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        grouped.setdefault(str(row["scam_type"]), []).append(row)

    selected: list[dict[str, object]] = []
    type_names = sorted(grouped)
    indexes = {name: 0 for name in type_names}
    while len(selected) < needed:
        progressed = False
        for name in type_names:
            index = indexes[name]
            if index < len(grouped[name]):
                selected.append(grouped[name][index])
                indexes[name] += 1
                progressed = True
                if len(selected) >= needed:
                    break
        if not progressed:
            break

    if len(selected) < needed:
        raise RuntimeError(f"Balanced selection failed: needed={needed}, selected={len(selected)}")
    return selected


def main() -> None:
    base = pd.read_csv(INPUT_PATH)
    base = base.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    safe_count = int((base["label"] == 0).sum())
    scam_count = int((base["label"] == 1).sum())
    needed_safe = max(TARGET_SAFE - safe_count, 0)
    needed_scam = max(TARGET_SCAM - scam_count, 0)

    existing = set(base["text"].astype(str).tolist())
    safe_rows = safe_candidates(existing)
    existing_after_safe = existing | {str(row["text"]) for row in safe_rows}
    scam_rows = scam_candidates(existing_after_safe)

    selected_safe = take_needed(safe_rows, needed_safe)
    selected_scam = take_balanced_by_type(scam_rows, needed_scam)

    additions = pd.DataFrame(selected_safe + selected_scam)
    combined = pd.concat([base, additions], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    if len(combined) > TARGET_TOTAL:
        # Keep all original rows and trim only generated additions deterministically.
        original_texts = set(base["text"].astype(str).tolist())
        generated = combined[~combined["text"].isin(original_texts)]
        keep_generated = generated.head(TARGET_TOTAL - len(base))
        combined = pd.concat([base, keep_generated], ignore_index=True)

    final_counts = Counter(combined["label"].tolist())
    duplicate_count = int(combined["text"].duplicated().sum())

    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    report_lines = [
        f"Input file: {INPUT_PATH.name}",
        f"Output file: {OUTPUT_PATH.name}",
        f"Target rows: {TARGET_TOTAL}",
        f"Original rows: {len(base)}",
        f"Generated safe candidates: {len(safe_rows)}",
        f"Generated scam candidates: {len(scam_rows)}",
        f"Added safe rows: {len(selected_safe)}",
        f"Added scam rows: {len(selected_scam)}",
        f"Final rows: {len(combined)}",
        f"Duplicate text count: {duplicate_count}",
        "",
        "Label counts:",
        f"label 0: {final_counts.get(0, 0)}",
        f"label 1: {final_counts.get(1, 0)}",
        "",
        "Scam type counts:",
    ]
    for scam_type, count in combined["scam_type"].value_counts().sort_index().items():
        report_lines.append(f"{scam_type}: {count}")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
