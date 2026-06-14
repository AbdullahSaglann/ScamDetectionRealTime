#!/usr/bin/env python3
"""Expand the cleaned dialogue dataset to a controlled 15k-20k training set.

The generated scenarios are inspired by public fraud-awareness warnings from
EGM and the Turkish Ministry of Trade, but the dialogue text is newly written.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/intermediate/Sonhali_augmented_clean.csv")
OUTPUT_FILE = Path("data/intermediate/Sonhali_augmented_15k20k.csv")
REPORT_FILE = Path("data/archive/reports/Sonhali_augmented_15k20k_report.txt")
TARGET_COUNTS = {0: 8000, 1: 10000}
SEED = 495496
SIMILARITY_THRESHOLD = 0.94
MAX_HAMMING_DISTANCE = 16


AGENT_OPENERS = [
    "Merhaba",
    "İyi günler",
    "Günaydın",
    "Kolay gelsin",
    "Selamlar",
]

VICTIM_CHECKS = [
    "Bunu resmi uygulamadan kontrol edebilir miyim?",
    "Telefonla böyle bilgi paylaşmak istemiyorum.",
    "Önce kurumun kendi numarasını arayıp teyit edeceğim.",
    "Bu işlem için yazılı bildirim var mı?",
    "Neden bu kadar acele etmem gerekiyor?",
]

BANKS = ["Ziraat", "Garanti", "Akbank", "İş Bankası", "Yapı Kredi", "Halkbank", "VakıfBank"]
CARGOES = ["PTT", "Yurtiçi Kargo", "Aras Kargo", "MNG", "Sürat Kargo", "DHL", "UPS"]
OPERATORS = ["Turkcell", "Vodafone", "Türk Telekom"]
PUBLIC_UNITS = ["emniyet", "savcılık", "vergi dairesi", "gümrük birimi", "belediye"]
AMOUNTS = ["750 TL", "1.250 TL", "2.900 TL", "6.500 TL", "18.000 TL", "42.000 TL"]
SMALL_FEES = ["19,90 TL", "34,75 TL", "48 TL", "72 TL", "96,50 TL", "143 TL"]
DEADLINES = ["10 dakika", "yarım saat", "bugün 17.00", "mesai bitimine kadar", "hemen şimdi"]
CHANNELS = ["SMS linki", "WhatsApp bağlantısı", "kısa form", "güvenlik ekranı", "ödeme sayfası"]
NAMES = ["Murat", "Ayşe", "Elif", "Mehmet", "Zeynep", "Can", "Fatma", "Emre"]
PRODUCTS = ["telefon", "kulaklık", "tablet", "akıllı saat", "oyun konsolu", "kahve makinesi"]
DISTRICTS = ["Kadıköy", "Çankaya", "Nilüfer", "Konak", "Gebze", "Seyhan", "Keçiören", "Muratpaşa"]
DAYS = ["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi"]
SAFE_CHANNELS = ["resmi uygulama", "kurumsal portal", "e-posta duyurusu", "çağrı merkezi", "randevu sistemi"]
REFERENCE_PREFIXES = ["BLG", "RND", "KRG", "DST", "TKP", "BAS", "MHS", "ORD"]


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text, flags=re.UNICODE)


def word_shingles(words: list[str], size: int = 3) -> list[str]:
    if len(words) < size:
        return words
    return [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]


def simhash(features: list[str], bits: int = 64) -> int:
    vector = [0] * bits
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        hashed = int.from_bytes(digest, "big")
        for bit in range(bits):
            vector[bit] += 1 if (hashed >> bit) & 1 else -1

    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def build_candidates(texts: list[str], hashes: list[int], labels: list[int]) -> set[tuple[int, int]]:
    buckets: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for row_index, hashed in enumerate(hashes):
        length_bucket = len(texts[row_index]) // 80
        for band in range(4):
            band_value = (hashed >> (band * 16)) & 0xFFFF
            buckets[(labels[row_index], band, band_value, length_bucket)].append(row_index)

    candidates: set[tuple[int, int]] = set()
    for rows in buckets.values():
        if len(rows) < 2 or len(rows) >= 500:
            continue
        for left_pos, left in enumerate(rows):
            for right in rows[left_pos + 1 :]:
                max_len = max(len(texts[left]), len(texts[right]))
                if abs(len(texts[left]) - len(texts[right])) <= max(90, 0.25 * max_len):
                    candidates.add((left, right) if left < right else (right, left))
    return candidates


def near_duplicate_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    working = df.copy()
    working["_norm"] = working["text"].map(normalize_text)
    working = working.drop_duplicates(subset=["_norm"], keep="first").reset_index(drop=True)

    texts = working["_norm"].tolist()
    labels = working["label"].tolist()
    feature_sets = []
    for text in texts:
        shingles = word_shingles(tokenize(text), size=3)
        feature_sets.append(shingles if shingles else [text])
    hashes = [simhash(features) for features in feature_sets]

    remove: set[int] = set()
    for left, right in sorted(build_candidates(texts, hashes, labels)):
        if right in remove or labels[left] != labels[right]:
            continue
        if hamming_distance(hashes[left], hashes[right]) > MAX_HAMMING_DISTANCE:
            continue
        similarity = SequenceMatcher(None, texts[left], texts[right], autojunk=False).ratio()
        if similarity >= SIMILARITY_THRESHOLD:
            remove.add(right)

    clean = working.loc[[idx not in remove for idx in range(len(working))], ["text", "label"]]
    return clean.reset_index(drop=True), len(remove)


def join_turns(*turns: str) -> str:
    return " ".join(turns)


def reference_code(rng: random.Random) -> str:
    return f"{rng.choice(REFERENCE_PREFIXES)}-{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGH')}"


def diversify_text(text: str, label: int, rng: random.Random) -> str:
    """Add small semantic variations without changing the class label."""
    if label == 1:
        additions = [
            f"[AGENT]: Dosya kodunuz {reference_code(rng)}, gecikme olursa sistem otomatik işlem başlatır.",
            f"[AGENT]: Bu görüşmeyi kapatmadan {rng.choice(CHANNELS)} üzerinden onay vermeniz gerekiyor.",
            f"[VICTIM]: Bu baskı güven vermiyor, resmi numarayı kendim arayacağım.",
            f"[AGENT]: {rng.choice(DEADLINES)} içinde dönüş yapmazsanız bloke ve yasal takip görünebilir.",
            f"[AGENT]: Kimseyle paylaşmayın, sadece bu hatta işlem açabiliyoruz.",
        ]
    else:
        additions = [
            f"[AGENT]: Bilgilendirme kodunuz {reference_code(rng)}, işlemi {rng.choice(SAFE_CHANNELS)} üzerinden görebilirsiniz.",
            "[AGENT]: Bu görüşmede şifre, kart bilgisi veya tek kullanımlık kod talep edilmeyecek.",
            f"[VICTIM]: Anladım, gerekirse {rng.choice(SAFE_CHANNELS)} üzerinden kendim işlem yaparım.",
            f"[AGENT]: Detaylar {rng.choice(DAYS)} günü kayıtlı iletişim adresinize de gönderilecek.",
            "[VICTIM]: Kişisel bilgi istemediğiniz için teşekkür ederim.",
        ]

    selected = rng.sample(additions, k=rng.choice([1, 1, 2]))
    if rng.choice([True, False]):
        return join_turns(text, *selected)
    return join_turns(selected[0], text, *selected[1:])


def fraud_dialogue(rng: random.Random) -> tuple[str, str]:
    scenario = rng.choice(
        [
            "bank_otp",
            "police_savci",
            "cargo_customs",
            "hakem_refund",
            "social_friend",
            "investment",
            "subscription",
            "job_offer",
            "operator_sim",
            "marketplace",
            "edevlet",
            "ministry_credit",
        ]
    )
    opener = rng.choice(AGENT_OPENERS)
    check = rng.choice(VICTIM_CHECKS)

    if scenario == "bank_otp":
        bank = rng.choice(BANKS)
        amount = rng.choice(AMOUNTS)
        channel = rng.choice(CHANNELS)
        text = join_turns(
            f"[AGENT]: {opener}, {bank} güvenlik biriminden arıyorum.",
            f"[AGENT]: Hesabınızdan {amount} tutarında şüpheli işlem başlatılmış.",
            f"[VICTIM]: {check}",
            f"[AGENT]: İşlemi durdurmak için {channel} üzerinden kart son haneleri ve SMS kodunu girmeniz gerekiyor.",
            "[VICTIM]: SMS kodunu kimseyle paylaşmam, bankayı kendim arayacağım.",
            f"[AGENT]: Beklerseniz para çıkışı onaylanır, {rng.choice(DEADLINES)} içinde yapmalısınız.",
        )
    elif scenario == "police_savci":
        unit = rng.choice(["polis", "savcı", "jandarma komutanlığı", "siber suçlar"])
        amount = rng.choice(AMOUNTS)
        text = join_turns(
            f"[AGENT]: {opener}, ben {unit} adına arıyorum.",
            "[AGENT]: Kimlik bilgileriniz bir dosyada kullanılmış, soruşturma gizli yürütülüyor.",
            "[VICTIM]: Resmi tebligat olmadan işlem yapmam.",
            f"[AGENT]: Gizlilik nedeniyle kimseye söylemeyin, hesabınızdaki {amount} güvenli hesaba aktarılacak.",
            "[VICTIM]: Kamu görevlisi telefonda para istemez, görüşmeyi kapatıyorum.",
        )
    elif scenario == "cargo_customs":
        cargo = rng.choice(CARGOES)
        fee = rng.choice(SMALL_FEES)
        reason = rng.choice(["gümrükte bekliyor", "adres teyidi yapılamadı", "teslim edilemedi", "içerik kontrolüne takıldı"])
        text = join_turns(
            f"[AGENT]: {opener}, {cargo} gönderi merkezinden arıyorum.",
            f"[AGENT]: Paketiniz {reason}; teslim için {fee} ödeme çıkıyor.",
            f"[VICTIM]: {check}",
            f"[AGENT]: Size gönderdiğim linkten ödeme yaparsanız dosya kapanır.",
            "[VICTIM]: Kargo ödemesini yalnızca resmi uygulamadan kontrol ederim.",
        )
    elif scenario == "hakem_refund":
        amount = rng.choice(AMOUNTS)
        text = join_turns(
            f"[AGENT]: {opener}, tüketici başvurunuz sonuçlandı.",
            f"[AGENT]: {amount} iade alacaksınız, kart bilgilerinizi doğrulamamız gerekiyor.",
            "[VICTIM]: Hakem heyeti iadesi için kart şifresi istenmez.",
            f"[AGENT]: İşlem bugün kapanıyor, {rng.choice(DEADLINES)} içinde bilgileri vermezseniz hakkınız yanar.",
            "[VICTIM]: Resmi başvuru ekranından bakacağım, telefonda bilgi vermiyorum.",
        )
    elif scenario == "social_friend":
        name = rng.choice(NAMES)
        amount = rng.choice(AMOUNTS)
        text = join_turns(
            f"[AGENT]: Ben {name}, yeni numaram bu.",
            f"[AGENT]: Cüzdanımı kaybettim, {amount} acil gönderebilir misin?",
            "[VICTIM]: Sesini duymadan para göndermem.",
            "[AGENT]: Arayamıyorum, telefonum bozuk; IBAN atıyorum, hemen gönder.",
            "[VICTIM]: Seni eski numarandan arayıp teyit edeceğim.",
        )
    elif scenario == "investment":
        amount = rng.choice(AMOUNTS)
        text = join_turns(
            f"[AGENT]: {opener}, kripto yatırım kulübümüzden ulaşıyorum.",
            f"[AGENT]: {amount} yatıran üyeler bu hafta garantili yüksek kazanç alıyor.",
            "[VICTIM]: Garanti kazanç vaadi gerçekçi değil.",
            "[AGENT]: Kontenjan kapanmadan cüzdan adresine transfer yaparsanız danışmanınız işlemi açar.",
            "[VICTIM]: Lisanslı olmayan hesaba para göndermem.",
        )
    elif scenario == "subscription":
        service = rng.choice(["Netflix", "Spotify", "bulut depolama", "oyun üyeliği", "internet paketi"])
        text = join_turns(
            f"[AGENT]: {opener}, {service} üyeliğiniz askıya alınacak.",
            f"[AGENT]: Ücretsiz kullanım hakkınız var, {rng.choice(CHANNELS)} ile ödeme bilgisi güncelleyin.",
            f"[VICTIM]: {check}",
            "[AGENT]: Güncelleme yapmazsanız hesabınız ve indirim hakkınız silinir.",
            "[VICTIM]: Uygulamaya kendim girip kontrol edeceğim.",
        )
    elif scenario == "job_offer":
        amount = rng.choice(SMALL_FEES)
        text = join_turns(
            f"[AGENT]: {opener}, evden çalışma başvurunuz onaylandı.",
            f"[AGENT]: Evrak açılışı için {amount} güvence bedeli yatırmanız gerekiyor.",
            "[VICTIM]: İş başvurusunda benden önce para istenmesi normal değil.",
            "[AGENT]: Ücreti yatırınca kargo ve sözleşme çıkacak, aksi halde hakkınız iptal olur.",
            "[VICTIM]: Şirketi resmi kanaldan teyit etmeden ödeme yapmam.",
        )
    elif scenario == "operator_sim":
        operator = rng.choice(OPERATORS)
        text = join_turns(
            f"[AGENT]: {opener}, {operator} hat güvenliği biriminden arıyorum.",
            "[AGENT]: Hattınız kopyalanmış görünüyor, e-Devlet ve banka erişiminiz riskte.",
            f"[VICTIM]: {check}",
            "[AGENT]: SIM bloke kaldırma için gelen tek kullanımlık kodu hemen söyleyin.",
            "[VICTIM]: Operatörü kendim arayacağım, kod paylaşmam.",
        )
    elif scenario == "marketplace":
        product = rng.choice(PRODUCTS)
        fee = rng.choice(SMALL_FEES)
        text = join_turns(
            f"[AGENT]: {opener}, ilanınızdaki {product} için alıcı ödemeyi yaptı.",
            f"[AGENT]: Parayı hesabınıza almak için {fee} doğrulama ücreti çıkıyor.",
            "[VICTIM]: Satışta para almak için ödeme yapmam gerekmemeli.",
            "[AGENT]: Sistem böyle çalışıyor, linkten kartınızı doğrulayınca tutar serbest kalır.",
            "[VICTIM]: Platform dışı link kullanmayacağım.",
        )
    elif scenario == "edevlet":
        unit = rng.choice(PUBLIC_UNITS)
        text = join_turns(
            f"[AGENT]: {opener}, {unit} kayıt güncellemesi için arıyorum.",
            "[AGENT]: e-Devlet hesabınızda eksik doğrulama var, işlem yapılmazsa hizmetleriniz kapanır.",
            f"[VICTIM]: {check}",
            "[AGENT]: Gönderdiğim bağlantıdan T.C. kimlik numarası, şifre ve onay kodunu yazmanız yeterli.",
            "[VICTIM]: e-Devlet şifremi hiçbir bağlantıya girmem.",
        )
    else:
        amount = rng.choice(AMOUNTS)
        text = join_turns(
            f"[AGENT]: {opener}, KOBİ destek ve kredi biriminden arıyorum.",
            f"[AGENT]: İşletmeniz için sıfır faizli {amount} kredi hakkı tanımlandı.",
            "[VICTIM]: Böyle bir birime başvuru yapmadım.",
            "[AGENT]: Dosyayı açmak için hesap bilgisi ve ön onay bedeli gerekiyor.",
            "[VICTIM]: Resmi kurumlar telefonda kredi pazarlamaz, teyit edeceğim.",
        )

    return diversify_text(text, 1, rng), scenario


def legit_dialogue(rng: random.Random) -> tuple[str, str]:
    scenario = rng.choice(
        [
            "bank_warning",
            "cargo_normal",
            "appointment",
            "meeting",
            "school_notice",
            "municipal",
            "support_ticket",
            "order_update",
            "operator_info",
            "clinic_result",
        ]
    )
    opener = rng.choice(AGENT_OPENERS)

    if scenario == "bank_warning":
        bank = rng.choice(BANKS)
        district = rng.choice(DISTRICTS)
        text = join_turns(
            f"[AGENT]: {opener}, {bank} bilgilendirme hattından arıyorum.",
            f"[AGENT]: {district} bölgesindeki müşterilerimize şüpheli bağlantılara şifre veya SMS kodu girilmemesi için uyarı yapıyoruz.",
            "[VICTIM]: Hesabımla ilgili işlem var mı?",
            "[AGENT]: Güvenliğiniz için burada bilgi istemiyoruz; kontrolü yalnızca resmi uygulamadan yapabilirsiniz.",
            "[VICTIM]: Teşekkür ederim, uygulamadan kendim kontrol edeceğim.",
        )
    elif scenario == "cargo_normal":
        cargo = rng.choice(CARGOES)
        day = rng.choice(DAYS)
        text = join_turns(
            f"[AGENT]: {opener}, {cargo} teslimat biriminden arıyorum.",
            f"[AGENT]: Paketiniz {day} günü dağıtıma çıkacak, apartman görevlisine bırakılmasını ister misiniz?",
            "[VICTIM]: Hayır, evde olacağım.",
            "[AGENT]: Tamamdır, ödeme veya link gerekmiyor; kuryemiz teslimatta kimlik teyidi yapacak.",
            "[VICTIM]: Bilgilendirme için teşekkürler.",
        )
    elif scenario == "appointment":
        time = rng.choice(["09.30", "11.00", "14.15", "16.40"])
        department = rng.choice(["dahiliye", "göz", "diş", "ortopedi", "kardiyoloji", "aile hekimliği"])
        text = join_turns(
            f"[AGENT]: {opener}, hastane randevu hatırlatma servisinden arıyorum.",
            f"[AGENT]: {department} bölümünde yarın saat {time} için randevunuz görünüyor.",
            "[VICTIM]: Evet, geleceğim.",
            "[AGENT]: Herhangi bir ödeme almıyoruz; randevu değişikliği için resmi sistemi kullanabilirsiniz.",
            "[VICTIM]: Tamam, teşekkür ederim.",
        )
    elif scenario == "meeting":
        topic = rng.choice(["proje toplantısı", "tez görüşmesi", "satın alma değerlendirmesi", "ekip planlaması"])
        text = join_turns(
            f"[AGENT]: {opener}, {topic} için toplantı saatini teyit etmek istiyorum.",
            "[AGENT]: Katılım bağlantısı kurumsal takvim davetinde yer alıyor.",
            "[VICTIM]: Daveti gördüm, katılacağım.",
            "[AGENT]: Harika, ek belge isterseniz kurumsal e-posta üzerinden paylaşırım.",
        )
    elif scenario == "school_notice":
        course = rng.choice(["bitirme projesi", "staj", "seçmeli ders", "laboratuvar", "danışman görüşmesi"])
        text = join_turns(
            f"[AGENT]: {opener}, öğrenci işleri biriminden arıyorum.",
            f"[AGENT]: {course} takvimi güncellendi, duyuru sistemi üzerinden kontrol edebilirsiniz.",
            "[VICTIM]: Belge göndermem gerekiyor mu?",
            "[AGENT]: Hayır, yalnızca portal üzerinden onay vermeniz yeterli.",
            "[VICTIM]: Tamam, portaldan bakacağım.",
        )
    elif scenario == "municipal":
        district = rng.choice(DISTRICTS)
        topic = rng.choice(["su kesintisi", "yol bakım çalışması", "atık toplama saati", "semt pazarı düzeni", "vezne çalışma saati"])
        text = join_turns(
            f"[AGENT]: {opener}, belediye çağrı merkezinden bilgilendirme yapıyoruz.",
            f"[AGENT]: {district} için {topic} duyurusu var, saat bilgisi resmi sitede yayınlandı.",
            "[VICTIM]: Herhangi bir işlem yapmam gerekiyor mu?",
            "[AGENT]: Hayır, sadece bilgilendirme. Kişisel bilgi veya ödeme talebimiz yoktur.",
            "[VICTIM]: Teşekkürler.",
        )
    elif scenario == "support_ticket":
        issue = rng.choice(["giriş hatası", "rapor indirme", "fatura görüntüleme", "adres güncelleme", "bildirim ayarı"])
        text = join_turns(
            f"[AGENT]: {opener}, destek talebiniz hakkında arıyorum.",
            f"[AGENT]: Bildirdiğiniz {issue} sorunu giderildi, kontrol edip geri dönüş yapabilir misiniz?",
            "[VICTIM]: Sisteme girip deneyeceğim.",
            "[AGENT]: Sorun sürerse aynı talep numarası üzerinden yazabilirsiniz.",
        )
    elif scenario == "order_update":
        product = rng.choice(PRODUCTS)
        text = join_turns(
            f"[AGENT]: {opener}, siparişinizdeki {product} stoktan ayrıldı.",
            "[AGENT]: Fatura ve kargo bilgisi hesabınızda görünüyor.",
            "[VICTIM]: Benden ödeme bilgisi istiyor musunuz?",
            "[AGENT]: Hayır, ödeme tamamlanmış. Sadece teslimat zamanı için bilgilendirme yapıyoruz.",
            "[VICTIM]: Anladım, teşekkür ederim.",
        )
    elif scenario == "operator_info":
        operator = rng.choice(OPERATORS)
        package = rng.choice(["internet", "konuşma", "faturalı hat", "kurumsal hat", "ek paket"])
        text = join_turns(
            f"[AGENT]: {opener}, {operator} müşteri hizmetlerinden arıyorum.",
            f"[AGENT]: {package} taahhüt bitiş tarihiniz yaklaşıyor, mevcut paketinizi bilgilendirmek istedik.",
            "[VICTIM]: Kod veya şifre vermem gerekiyor mu?",
            "[AGENT]: Hayır, telefonda kod istemiyoruz. Değişiklik isterseniz resmi uygulamadan yapabilirsiniz.",
            "[VICTIM]: Tamam, uygulamadan incelerim.",
        )
    else:
        result = rng.choice(["tahlil sonucunuz", "aşı randevunuz", "kontrol muayeneniz", "rapor yenilemeniz", "reçete takibiniz"])
        text = join_turns(
            f"[AGENT]: {opener}, aile hekimliği biriminden arıyorum.",
            f"[AGENT]: {result} sisteme işlendi, doktorunuz kontrol randevusu öneriyor.",
            "[VICTIM]: Sonucu telefonda paylaşacak mısınız?",
            "[AGENT]: Hayır, sağlık bilgisi telefonda paylaşılmıyor; resmi sistemden veya hekimden görebilirsiniz.",
            "[VICTIM]: Tamam, randevu alacağım.",
        )

    return diversify_text(text, 0, rng), scenario


def generate_rows(existing: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    rows = existing[["text", "label"]].copy()
    seen = set(rows["text"].map(normalize_text))
    scenario_counts: dict[str, int] = defaultdict(int)

    for label, target in TARGET_COUNTS.items():
        attempts = 0
        while int((rows["label"] == label).sum()) < target:
            attempts += 1
            if attempts > 250000:
                raise RuntimeError(f"Could not generate enough unique rows for label {label}.")

            text, scenario = fraud_dialogue(rng) if label == 1 else legit_dialogue(rng)
            norm = normalize_text(text)
            if norm in seen:
                continue

            seen.add(norm)
            scenario_counts[f"{label}:{scenario}"] += 1
            rows.loc[len(rows)] = {"text": text, "label": label}

    rows.attrs["scenario_counts"] = dict(scenario_counts)
    return rows.sample(frac=1, random_state=SEED).reset_index(drop=True)


def top_up_after_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED + 1)
    rows = df.copy()
    seen = set(rows["text"].map(normalize_text))

    for label, target in TARGET_COUNTS.items():
        attempts = 0
        while int((rows["label"] == label).sum()) < target:
            attempts += 1
            if attempts > 250000:
                break
            text, _scenario = fraud_dialogue(rng) if label == 1 else legit_dialogue(rng)
            norm = normalize_text(text)
            if norm in seen:
                continue
            seen.add(norm)
            rows.loc[len(rows)] = {"text": text, "label": label}

    return rows.sample(frac=1, random_state=SEED + 2).reset_index(drop=True)


def main() -> None:
    clean = pd.read_csv(INPUT_FILE)
    expanded = generate_rows(clean)
    deduped, near_removed = near_duplicate_clean(expanded)
    final = top_up_after_cleaning(deduped)
    final, final_near_removed = near_duplicate_clean(final)

    if len(final) > sum(TARGET_COUNTS.values()):
        sampled_parts = []
        for label, target in TARGET_COUNTS.items():
            part = final[final["label"] == label]
            sampled_parts.append(part.sample(n=target, random_state=SEED + label))
        final = pd.concat(sampled_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    final.to_csv(OUTPUT_FILE, index=False)

    report_lines = [
        f"Input file: {INPUT_FILE}",
        f"Input rows: {len(clean)}",
        f"Output file: {OUTPUT_FILE}",
        f"Output rows: {len(final)}",
        "",
        "Input label counts:",
        clean["label"].value_counts().sort_index().to_string(),
        "",
        "Output label counts:",
        final["label"].value_counts().sort_index().to_string(),
        "",
        "Output label distribution:",
        final["label"].value_counts(normalize=True).sort_index().to_string(),
        "",
        f"Near duplicates removed during first pass: {near_removed}",
        f"Near duplicates removed during final pass: {final_near_removed}",
        f"Unique text count: {final['text'].nunique()}",
        f"Duplicate text count: {len(final) - final['text'].nunique()}",
        "",
        "Scenario source notes:",
        "- EGM warnings: phone scammers impersonating police/soldier/prosecutor and demanding money/gold.",
        "- Ministry of Trade warnings: fake cargo/customs notices, refund/card-info scams, social-media/digital phishing, fake credit offers.",
        "- Legitimate class includes confusable but safe calls that explicitly avoid asking for passwords, OTPs, links, or payments.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
