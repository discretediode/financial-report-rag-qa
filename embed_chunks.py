"""
Embedding script for the annual report Q&A project.

Takes the chunks.json produced by chunk_documents.py and converts each
chunk's text into a dense vector using a HuggingFace sentence-transformer
model. Vectors are saved as a .npy array alongside a metadata JSON, ready 
to be loaded into ChromaDB in the next step.

Vectors are stored separately from metadata rather than inside one JSON
because embeddings are large numeric arrays — .npy keeps them compact and
loads far faster than parsing thousands of floats out of a text file.

Usage:
    python embed_chunks.py --input_file ./chunks/chunks.json --output_dir ./embeddings
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load the flat chunk list produced by chunk_documents.py."""
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode a list of texts into dense vectors.

    The model is downloaded from HuggingFace on first use and cached
    locally, so the first run is slower than subsequent ones. Encoding
    runs on PyTorch and will use a GPU automatically if one is available,
    otherwise CPU.

    Args:
        texts: The chunk texts to encode.
        model_name: HuggingFace sentence-transformer model identifier.
        batch_size: How many texts to encode per forward pass. Lower this
            if you hit memory limits on a large corpus.

    Returns:
        A 2D numpy array of shape (len(texts), embedding_dimension).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def embed_chunks(
    input_file: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> None:
    """
    Embed every chunk in chunks.json and write vectors plus metadata to disk.

    Writes three files to output_dir:
      - embeddings.npy: the vector array, row i corresponding to chunk i
      - chunk_metadata.json: the chunks with their text and metadata, in the
        same order as the vector rows, so the two can be zipped back together
      - _embedding_summary.json: model name, vector count, and dimension,
        for a quick check that the run produced what was expected

    Args:
        input_file: Path to chunks.json.
        output_dir: Folder to write the embedding outputs to.
        model_name: HuggingFace sentence-transformer model identifier.
        batch_size: Texts encoded per forward pass.

    Returns:
        None. Results are written to disk as a side effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Chunk file not found: {input_file}")
        return

    chunks = load_chunks(input_file)
    if not chunks:
        print(f"No chunks found in {input_file}")
        return

    print(f"Loaded {len(chunks)} chunks from {input_file}")
    print(f"Embedding with model '{model_name}'...")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)

    # Sanity check: one vector per chunk, or the two files won't line up later.
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks)})"
        )

    vectors_path = output_dir / "embeddings.npy"
    np.save(vectors_path, embeddings)

    metadata_path = output_dir / "chunk_metadata.json"
    metadata_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "model": model_name,
        "num_chunks": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "companies": sorted({c["company"] for c in chunks}),
    }
    summary_path = output_dir / "_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone. {len(chunks)} vectors of dimension {embeddings.shape[1]}.")
    print(f"Vectors written to  {vectors_path}")
    print(f"Metadata written to {metadata_path}")
    print(f"Summary written to  {summary_path}")


def main():
    """Parse CLI arguments and run the embedding pipeline."""
    parser = argparse.ArgumentParser(description="Embed chunked annual report text with a HuggingFace model.")
    parser.add_argument("--input_file", type=str, default="./chunks/chunks.json", help="Path to chunks.json")
    parser.add_argument("--output_dir", type=str, default="./embeddings", help="Directory to write embeddings and metadata")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace sentence-transformer model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Texts encoded per forward pass")
    args = parser.parse_args()

    embed_chunks(Path(args.input_file), Path(args.output_dir), args.model, args.batch_size)


if __name__ == "__main__":
    main()