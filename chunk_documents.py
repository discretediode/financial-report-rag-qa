"""
Chunking script for the annual report Q&A project.

Takes the extracted JSON files produced by extract_pdfs.py and splits each
document's full text into overlapping chunks, tagging every chunk with
metadata (company, year, source file, chunk index) so it's ready for
embedding in the next step.

Usage:
    python chunk_documents.py --input_dir ./extracted --output_dir ./chunks
"""

import argparse
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_extracted_document(json_path: Path) -> dict:
    """Load one extracted document JSON (as produced by extract_pdfs.py)."""
    return json.loads(json_path.read_text(encoding="utf-8"))


def chunk_document(document: dict, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Split one document's full text into overlapping chunks.

    Note: chunk_size/chunk_overlap are measured in characters here (the
    splitter's default length function), not tokens. That's fine for a
    RAG pipeline, but worth knowing if you later compare it to a token
    count reported elsewhere.

    Each chunk is tagged with the document's metadata (company, year,
    source file) plus a chunk_id unique across the whole corpus, so a
    retrieved chunk can always be traced back to its source document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    pieces = splitter.split_text(document["text"])

    company_slug = document["company"].lower().replace(" ", "-")
    year = document.get("year") or "unknown"

    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append({
            "chunk_id": f"{company_slug}_{year}_{i}",
            "company": document["company"],
            "year": document["year"],
            "source_file": document["source_file"],
            "chunk_index": i,
            "text": piece,
        })
    return chunks


def process_directory(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> None:
    """
    Chunk every extracted document in a folder and write the combined result.

    Reads every *.json file in input_dir (skipping the "_extraction_summary.json"
    file written by extract_pdfs.py, which isn't a document), chunks each one,
    and writes:
      - chunks.json: one flat list of every chunk across all documents, ready
        to feed into the embedding step
      - _chunking_summary.json: chunk count per source document, for a
        quick sanity check without opening chunks.json itself

    Args:
        input_dir: Folder containing extracted document JSON files.
        output_dir: Folder to write chunks.json and the summary to.
        chunk_size: Target chunk length in characters.
        chunk_overlap: Overlap between consecutive chunks, in characters.

    Returns:
        None. Results are written to disk as a side effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(
        f for f in input_dir.glob("*.json") if not f.name.startswith("_")
    )

    if not json_files:
        print(f"No extracted JSON files found in {input_dir}")
        return

    all_chunks = []
    summary = []

    for json_path in json_files:
        print(f"Chunking {json_path.name}...")
        document = load_extracted_document(json_path)
        chunks = chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  {len(chunks)} chunks created")
        summary.append({
            "source_file": document["source_file"],
            "company": document["company"],
            "year": document["year"],
            "num_chunks": len(chunks),
        })

    out_path = output_dir / "chunks.json"
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = output_dir / "_chunking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone. {len(all_chunks)} total chunks from {len(json_files)} document(s).")
    print(f"Chunks written to {out_path}")
    print(f"Summary written to {summary_path}")


def main():
    """Parse CLI arguments and run the chunking pipeline."""
    parser = argparse.ArgumentParser(description="Chunk extracted annual report text for embedding.")
    parser.add_argument("--input_dir", type=str, default="./extracted", help="Directory containing extracted JSON files")
    parser.add_argument("--output_dir", type=str, default="./chunks", help="Directory to write chunked output")
    parser.add_argument("--chunk_size", type=int, default=500, help="Target chunk size in characters")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap between consecutive chunks in characters")
    args = parser.parse_args()

    process_directory(Path(args.input_dir), Path(args.output_dir), args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
