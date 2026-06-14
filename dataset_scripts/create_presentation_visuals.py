#!/usr/bin/env python3
"""Create slide-ready SVG visuals for the dataset work."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("presentation_visuals")
OUT_DIR.mkdir(exist_ok=True)


def write_svg(path: Path, body: str, width: int = 1400, height: int = 800) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f8fb"/>
  <style>
    .title {{ font: 700 44px Arial, sans-serif; fill: #172033; }}
    .subtitle {{ font: 400 22px Arial, sans-serif; fill: #5b6475; }}
    .label {{ font: 700 24px Arial, sans-serif; fill: #172033; }}
    .small {{ font: 400 18px Arial, sans-serif; fill: #5b6475; }}
    .num {{ font: 800 42px Arial, sans-serif; fill: #172033; }}
    .white {{ fill: #ffffff; }}
  </style>
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def visual_pipeline() -> None:
    steps = [
        ("Raw Augmented", "40,000", "high repetition"),
        ("Dedup + Similarity", "5,507", "clean core set"),
        ("Controlled Expansion", "15,117", "15-20k train set"),
        ("Type Labeling", "22 types", "scam_type column"),
    ]
    colors = ["#3157a4", "#1f8a70", "#d9822b", "#7c3aed"]
    cards = []
    for i, (title, number, note) in enumerate(steps):
        x = 70 + i * 325
        cards.append(f"""
  <rect x="{x}" y="245" width="260" height="210" rx="8" fill="{colors[i]}"/>
  <text x="{x + 24}" y="305" class="label white">{title}</text>
  <text x="{x + 24}" y="370" class="num white">{number}</text>
  <text x="{x + 24}" y="420" class="small white">{note}</text>
""")
        if i < len(steps) - 1:
            ax = x + 278
            cards.append(f"""
  <line x1="{ax}" y1="350" x2="{ax + 42}" y2="350" stroke="#8d96a8" stroke-width="8" stroke-linecap="round"/>
  <polygon points="{ax + 54},350 {ax + 34},338 {ax + 34},362" fill="#8d96a8"/>
""")
    body = f"""
  <text x="70" y="95" class="title">Dataset Preparation Pipeline</text>
  <text x="70" y="135" class="subtitle">Duplicate cleaning, controlled expansion, and scam type labeling</text>
  {''.join(cards)}
  <rect x="70" y="570" width="1260" height="92" rx="8" fill="#ffffff" stroke="#d7dce5"/>
  <text x="105" y="625" class="label">Final output:</text>
  <text x="285" y="625" class="small">Sonhali_augmented_15k20k_typed.csv · 15,117 rows · 0 duplicate text</text>
"""
    write_svg(OUT_DIR / "01_dataset_pipeline.svg", body)


def visual_cleaning() -> None:
    values = [40000, 7466, 5507, 15117]
    labels = ["Raw augmented", "Unique text", "Clean core", "Final typed"]
    colors = ["#9b2c2c", "#3157a4", "#1f8a70", "#7c3aed"]
    max_v = max(values)
    bars = []
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 225 + i * 105
        w = int(950 * value / max_v)
        bars.append(f"""
  <text x="90" y="{y + 32}" class="label">{label}</text>
  <rect x="330" y="{y}" width="950" height="48" rx="6" fill="#e4e8f0"/>
  <rect x="330" y="{y}" width="{w}" height="48" rx="6" fill="{color}"/>
  <text x="{350 + w}" y="{y + 34}" class="label">{value:,}</text>
""")
    body = f"""
  <text x="70" y="95" class="title">Cleaning and Expansion Impact</text>
  <text x="70" y="135" class="subtitle">The 40k dataset was cleaned for repetition risk, then expanded into the 15k range</text>
  {''.join(bars)}
  <rect x="90" y="665" width="1180" height="58" rx="8" fill="#ffffff" stroke="#d7dce5"/>
  <text x="120" y="703" class="small">Exact duplicates removed: 32,534 · Near duplicates filtered during cleaning and final generation · Final duplicates: 0</text>
"""
    write_svg(OUT_DIR / "02_cleaning_impact.svg", body)


def visual_type_distribution() -> None:
    df = pd.read_csv("data/intermediate/Sonhali_augmented_15k20k_typed.csv")
    counts = df[df["label"] == 1]["scam_type"].value_counts().head(10)
    max_v = int(counts.max())
    palette = ["#3157a4", "#1f8a70", "#d9822b", "#7c3aed", "#b83280", "#2f855a", "#b7791f", "#2b6cb0", "#6b46c1", "#c53030"]
    rows = []
    pretty = {
        "banka_kart_hesap": "Bank / card / account",
        "kargo_gumruk": "Cargo / customs",
        "sosyal_medya_whatsapp": "Social media / WhatsApp",
        "edevlet_kamu_vergi": "e-Gov / public sector",
        "savcilik_adliye": "Prosecutor / court",
        "tuketici_hakem_iade": "Consumer / refund",
        "sahte_kredi_destek": "Fake credit support",
        "yatirim_kripto": "Investment / crypto",
        "pazar_yeri_ilan": "Marketplace / listing",
        "oltalama_link_kod": "Phishing / link / code",
    }
    for i, (name, value) in enumerate(counts.items()):
        y = 185 + i * 54
        w = int(820 * int(value) / max_v)
        rows.append(f"""
  <text x="85" y="{y + 30}" class="small">{pretty.get(name, name)}</text>
  <rect x="390" y="{y}" width="820" height="34" rx="5" fill="#e4e8f0"/>
  <rect x="390" y="{y}" width="{w}" height="34" rx="5" fill="{palette[i]}"/>
  <text x="{410 + w}" y="{y + 25}" class="small">{int(value):,}</text>
""")
    body = f"""
  <text x="70" y="90" class="title">Scam Type Distribution</text>
  <text x="70" y="130" class="subtitle">Label=1 records in the final dataset were assigned rule-based subtypes</text>
  {''.join(rows)}
  <rect x="85" y="725" width="1120" height="42" rx="8" fill="#ffffff" stroke="#d7dce5"/>
  <text x="110" y="752" class="small">Goal: teach the model both scam detection and the specific scam scenario type</text>
"""
    write_svg(OUT_DIR / "03_scam_type_distribution.svg", body)


def main() -> None:
    visual_pipeline()
    visual_cleaning()
    visual_type_distribution()
    print("Created:")
    for path in sorted(OUT_DIR.glob("*.svg")):
        print(path)


if __name__ == "__main__":
    main()
