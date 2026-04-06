import csv
import argparse
import re
from pathlib import Path

import pdfplumber
import pytesseract

PDF_PATH = Path("data/Legal Kanna.pdf")
OUT_CSV = Path("data/legal_glossary.csv")
RAW_CSV = Path("data/legal_glossary_extracted_raw.csv")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_CONFIG = "--tessdata-dir data/tessdata --oem 1 --psm 6"

EN_LINE = re.compile(r"^[A-Za-z][A-Za-z\s\-\'/(),.&]{1,120}$")
EN_KN_LINE = re.compile(r"^([A-Za-z][A-Za-z\s\-\'/(),.&]{1,120}?)(?:\s*:\s*|\s+)([\u0C80-\u0CFF][\u0C80-\u0CFF\s,;\-\'/().&]+)$")


def clean_en(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.strip("-:;,. ")


def clean_kn(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("-:;,. ")


def looks_noise(line: str) -> bool:
    lower = line.lower().strip()
    if not lower:
        return True
    if lower in {"syn", "syn.", "(syn.)", "(syn)", "page", "part", "english-kannada"}:
        return True
    if len(lower) <= 1:
        return True
    if lower.isdigit():
        return True
    return False


def extract_pairs_from_text(text: str):
    pairs = []
    pending_en = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if looks_noise(line):
            continue

        mixed = EN_KN_LINE.match(line)
        if mixed:
            en = clean_en(mixed.group(1))
            kn = clean_kn(mixed.group(2))
            if en and kn:
                pairs.append((en, kn))
            pending_en = None
            continue

        if EN_LINE.match(line) and any(c.isalpha() for c in line):
            pending_en = clean_en(line)
            continue

        if pending_en and line and re.search(r"[\u0C80-\u0CFF]", line):
            kn = clean_kn(line)
            if kn:
                pairs.append((pending_en, kn))
            pending_en = None

    return pairs


def load_existing_rows(raw_csv: Path):
    if not raw_csv.exists():
        return []

    rows = []
    with raw_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = (row.get("english") or "").strip()
            kn = (row.get("kannada") or "").strip()
            page = (row.get("page") or "").strip()
            if not en or not kn or not page.isdigit():
                continue
            rows.append({"english": en, "kannada": kn, "page": int(page)})
    return rows


def write_raw_rows(raw_csv: Path, rows):
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["english", "kannada", "page"])
        writer.writeheader()
        writer.writerows(rows)


def write_dedup_dictionary(out_csv: Path, rows):
    dedup = {}
    for row in rows:
        en = row["english"]
        kn = row["kannada"]
        if en and kn and en not in dedup:
            dedup[en] = kn

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["english", "kannada"])
        writer.writeheader()
        for en in sorted(dedup.keys()):
            writer.writerow({"english": en, "kannada": dedup[en]})

    return len(dedup)


def build_glossary(start_page=1, end_page=None, checkpoint_every=25, resume=True, resolution=170):
    all_pairs = load_existing_rows(RAW_CSV) if resume else []
    done_pages = {row["page"] for row in all_pairs}

    with pdfplumber.open(PDF_PATH) as pdf:
        total = len(pdf.pages)
        stop_page = end_page if end_page is not None else total

        print(f"Total pages: {total}")
        print(f"Processing range: {start_page} to {stop_page}")
        if resume:
            print(f"Resume mode: ON ({len(done_pages)} pages already saved)")

        try:
            for i in range(start_page, stop_page + 1):
                if i < 1 or i > total:
                    continue
                if resume and i in done_pages:
                    continue

                page = pdf.pages[i - 1]
                image = page.to_image(resolution=resolution).original
                text = pytesseract.image_to_string(image, lang="eng+kan", config=TESS_CONFIG)
                page_pairs = extract_pairs_from_text(text)

                for en, kn in page_pairs:
                    all_pairs.append({"english": en, "kannada": kn, "page": i})

                done_pages.add(i)

                if i % checkpoint_every == 0:
                    write_raw_rows(RAW_CSV, all_pairs)
                    unique_count = write_dedup_dictionary(OUT_CSV, all_pairs)
                    print(
                        f"Checkpoint at page {i}/{total}. "
                        f"Rows: {len(all_pairs)} | Unique terms: {unique_count}"
                    )
        except KeyboardInterrupt:
            write_raw_rows(RAW_CSV, all_pairs)
            unique_count = write_dedup_dictionary(OUT_CSV, all_pairs)
            print("Interrupted by user. Progress saved.")
            print(f"Rows: {len(all_pairs)} | Unique terms: {unique_count}")
            print(f"Raw output: {RAW_CSV}")
            print(f"Dictionary output: {OUT_CSV}")
            return

    write_raw_rows(RAW_CSV, all_pairs)
    unique_count = write_dedup_dictionary(OUT_CSV, all_pairs)
    print(f"Done. Raw rows: {len(all_pairs)}")
    print(f"Unique terms written: {unique_count}")
    print(f"Raw output: {RAW_CSV}")
    print(f"Dictionary output: {OUT_CSV}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Kannada legal glossary from PDF using OCR.")
    parser.add_argument("--start-page", type=int, default=1, help="1-based start page")
    parser.add_argument("--end-page", type=int, default=None, help="1-based end page")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="save interval in pages")
    parser.add_argument("--resolution", type=int, default=170, help="OCR image resolution")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing raw progress")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_glossary(
        start_page=args.start_page,
        end_page=args.end_page,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        resolution=args.resolution,
    )
