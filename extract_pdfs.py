"""
PDF extractor for the annual report Q&A project.

Extracts text from PDF annual reports, tags each document with company/year
metadata parsed from its filename, flags pages that produced no extractable
text (usually scanned images or charts), and writes one JSON file per PDF
ready for the chunking step.

Expected filename convention: <company>_<year>.pdf
    e.g. siemens_2024.pdf, bmw_2023.pdf, deutsche-bank_2024.pdf

Usage:
    python extract_pdfs.py --input_dir ./pdfs --output_dir ./extracted
"""

import argparse
import json
import re
from pathlib import Path

import pdfplumber


def parse_filename_metadata(filename: str) -> dict:
    """Extract company and year from a filename like 'siemens_2024.pdf'."""
    stem = Path(filename).stem
    match = re.match(r"([a-zA-Z0-9\-]+)_(\d{4})", stem)
    if match:
        company, year = match.groups()
        return {"company": company.replace("-", " ").title(), "year": int(year)}
    return {"company": stem, "year": None}


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """
    Extract text from a PDF, page by page.

    Returns per-page text, total page count, and a list of page numbers
    that produced no extractable text (a signal for scanned/image-only pages).
    """
    pages = []
    empty_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                empty_pages.append(i)
            pages.append({"page_number": i, "text": text})

    return {
        "num_pages": len(pages),
        "empty_pages": empty_pages,
        "pages": pages,
    }


def process_directory(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    summary = []

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        meta = parse_filename_metadata(pdf_path.name)

        try:
            result = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            summary.append({**meta, "file": pdf_path.name, "status": "failed", "error": str(e)})
            continue

        full_text = "\n\n".join(p["text"] for p in result["pages"])

        record = {
            **meta,
            "source_file": pdf_path.name,
            "num_pages": result["num_pages"],
            "empty_pages": result["empty_pages"],
            "pages": result["pages"],   # page-level text, useful for citing page numbers later
            "text": full_text,          # full concatenated text, convenient for chunking
        }

        out_path = output_dir / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        status = "ok"
        if result["empty_pages"]:
            pct_empty = len(result["empty_pages"]) / result["num_pages"] * 100
            print(
                f"  Warning: {len(result['empty_pages'])}/{result['num_pages']} pages had "
                f"no extractable text ({pct_empty:.0f}%). Likely scanned/image pages — "
                f"OCR would be needed to recover them."
            )
            status = "partial"

        print(f"  Extracted {result['num_pages']} pages -> {out_path.name}")
        summary.append({**meta, "file": pdf_path.name, "status": status, "num_pages": result["num_pages"]})

    summary_path = output_dir / "_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. {len(pdf_files)} file(s) processed. Summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract text from annual report PDFs.")
    parser.add_argument("--input_dir", type=str, default="./pdfs", help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default="./extracted", help="Directory to write extracted JSON files")
    args = parser.parse_args()

    process_directory(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
