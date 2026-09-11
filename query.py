"""
Query interface for the annual report Q&A project.

This is the query-time entry point: it encodes a question, retrieves the most
relevant chunks from the ChromaDB collection built by build_vectorstore.py,
and asks Gemini to answer using only those chunks as context.

Two modes:
  --mode qa       Free-text answer to a natural-language question, with sources.
  --mode extract  Structured JSON extraction of key financial fields for a
                  given company, using schema-constrained generation so the
                  output is guaranteed to parse.

IMPORTANT: queries must be encoded with the same model used in embed_chunks.py.
Vectors from different models are not comparable, and a mismatch produces
plausible-looking but meaningless results with no error raised.

Requires GOOGLE_API_KEY in the environment (or a .env file).

Usage:
    python query.py --question "What were Siemens' main risk factors in 2024?"
    python query.py --question "How did revenue change?" --company Siemens --year 2024
    python query.py --mode extract --company Siemens --year 2024
"""

import argparse
import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "all-MiniLM-L6-v2"          # must match embed_chunks.py
DEFAULT_COLLECTION = "annual_reports"
DEFAULT_LLM = "gemini-3.6-flash"

QA_PROMPT = """You are a financial analyst assistant. Answer the question using ONLY the
excerpts from company annual reports provided below.

Rules:
- If the excerpts do not contain enough information to answer, say so plainly.
  Do not fill gaps with outside knowledge or estimates.
- Do not calculate financial ratios or perform arithmetic. Report figures as stated.
- Refer to the source of each fact by its [Source N] label.

Excerpts:
{context}

Question: {question}

Answer:"""

EXTRACT_PROMPT = """Extract the requested financial fields from the annual report excerpts below.

Rules:
- Use ONLY the excerpts. If a field is not stated, return null for it.
- Do NOT estimate, infer, or calculate any value that is not explicitly present.
- Report the unit exactly as stated in the text (e.g. "EUR millions").

Excerpts:
{context}

Company: {company}"""

# Schema-constrained output: the API guarantees valid JSON matching this shape,
# which removes the whole class of parse failures you get from asking a model
# to "return only JSON" in the prompt.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "fiscal_year": {"type": "integer", "nullable": True},
        "revenue": {"type": "number", "nullable": True},
        "revenue_unit": {"type": "string", "nullable": True},
        "operating_profit": {"type": "number", "nullable": True},
        "operating_profit_unit": {"type": "string", "nullable": True},
        "net_income": {"type": "number", "nullable": True},
        "net_income_unit": {"type": "string", "nullable": True},
        "employee_count": {"type": "integer", "nullable": True},
        "key_risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["company", "key_risks"],
}


def load_encoder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load the sentence-transformer used to encode queries."""
    return SentenceTransformer(model_name)


def build_where_filter(company: str | None, year: int | None) -> dict | None:
    """
    Build a ChromaDB metadata filter from optional company/year arguments.

    Chroma requires an explicit $and wrapper when filtering on more than one
    field, so a single filter and a combined filter are shaped differently.
    Returns None when no filtering is requested.
    """
    clauses = []
    if company:
        clauses.append({"company": company})
    if year is not None:
        clauses.append({"year": year})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve(
    question: str,
    collection,
    encoder: SentenceTransformer,
    n_results: int = 5,
    company: str | None = None,
    year: int | None = None,
) -> list[dict]:
    """
    Encode a question and return the most similar chunks from the collection.

    Args:
        question: The natural-language query.
        collection: An open ChromaDB collection.
        encoder: The sentence-transformer (must match the embedding model).
        n_results: How many chunks to retrieve.
        company: Optional exact-match filter on company.
        year: Optional exact-match filter on fiscal year.

    Returns:
        A list of dicts with text, metadata and distance, nearest first.
        Empty if the collection has no matching entries.
    """
    query_vector = encoder.encode([question], convert_to_numpy=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where=build_where_filter(company, year),
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    return [
        {
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def format_context(chunks: list[dict]) -> str:
    """
    Render retrieved chunks into a numbered context block for the prompt.

    Each chunk is labelled [Source N] with its company, year and source file
    so the model can attribute facts and the reader can trace them back.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        year = meta.get("year")
        year_str = str(year) if year not in (None, -1) else "unknown year"
        header = f"[Source {i}] {meta.get('company')}, {year_str} ({meta.get('source_file')})"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def generate_answer(prompt: str, model: str = DEFAULT_LLM, schema: dict | None = None) -> str:
    """
    Send a prompt to Gemini and return the response text.

    When a schema is supplied, the API is asked to return JSON conforming to
    it, so the caller can parse the result without defensive cleanup.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Export it or put it in a .env file before querying."
        )

    from google import genai

    client = genai.Client(api_key=api_key)

    config = {}
    if schema is not None:
        config = {"response_mime_type": "application/json", "response_schema": schema}

    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return response.text


def answer_question(
    question: str,
    collection,
    encoder: SentenceTransformer,
    n_results: int = 5,
    company: str | None = None,
    year: int | None = None,
    llm_model: str = DEFAULT_LLM,
) -> dict:
    """
    Run the full RAG flow: retrieve, assemble context, generate a grounded answer.

    Returns:
        A dict with the answer text and the source chunks it was based on.
        If retrieval finds nothing, no LLM call is made and the answer says so.
    """
    chunks = retrieve(question, collection, encoder, n_results, company, year)

    if not chunks:
        return {
            "answer": "No relevant excerpts found. Check that documents are loaded "
                      "and that any company/year filter matches the indexed data.",
            "sources": [],
        }

    prompt = QA_PROMPT.format(context=format_context(chunks), question=question)
    answer = generate_answer(prompt, model=llm_model)

    return {"answer": answer, "sources": chunks}


def extract_financials(
    company: str,
    collection,
    encoder: SentenceTransformer,
    year: int | None = None,
    n_results: int = 8,
    llm_model: str = DEFAULT_LLM,
) -> dict:
    """
    Extract key financial fields for one company as schema-validated JSON.

    Retrieval is driven by a fixed probe query aimed at the financial summary
    sections, filtered to the requested company (and year, if given).

    Returns:
        A dict with the parsed fields and the source chunks used. Fields not
        present in the retrieved text come back as null rather than guessed.
    """
    probe = "revenue operating profit net income employees principal risks"
    chunks = retrieve(probe, collection, encoder, n_results, company, year)

    if not chunks:
        return {"error": f"No indexed content found for company '{company}'.", "sources": []}

    prompt = EXTRACT_PROMPT.format(context=format_context(chunks), company=company)
    raw = generate_answer(prompt, model=llm_model, schema=EXTRACTION_SCHEMA)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"Model returned unparseable JSON: {e}", "raw": raw, "sources": chunks}

    return {"data": data, "sources": chunks}


def open_collection(db_dir: Path, collection_name: str = DEFAULT_COLLECTION):
    """
    Open the persistent ChromaDB collection built by build_vectorstore.py.

    Raises:
        FileNotFoundError: If the database directory does not exist.
        RuntimeError: If the collection is missing from the database.
    """
    if not db_dir.exists():
        raise FileNotFoundError(
            f"No vector store at {db_dir}. Run build_vectorstore.py first."
        )

    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        return client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Collection '{collection_name}' not found in {db_dir}. "
            f"Run build_vectorstore.py to create it. ({e})"
        ) from e


def print_sources(chunks: list[dict]) -> None:
    """Print a compact source list so answers can be traced to their origin."""
    if not chunks:
        return
    print("\nSources:")
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        year = meta.get("year")
        year_str = str(year) if year not in (None, -1) else "unknown"
        preview = " ".join(chunk["text"].split())[:90]
        print(f"  [{i}] {meta.get('company')} {year_str} "
              f"(chunk {meta.get('chunk_index')}, distance {chunk['distance']:.3f})")
        print(f"      {preview}...")


def main():
    """Parse CLI arguments and run a query against the vector store."""
    parser = argparse.ArgumentParser(description="Query annual reports with RAG.")
    parser.add_argument("--mode", choices=["qa", "extract"], default="qa", help="Query mode")
    parser.add_argument("--question", type=str, help="Question to ask (qa mode)")
    parser.add_argument("--company", type=str, help="Filter by company (required for extract mode)")
    parser.add_argument("--year", type=int, help="Filter by fiscal year")
    parser.add_argument("--db_dir", type=str, default="./chroma_db", help="ChromaDB directory")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Collection name")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_MODEL, help="Must match embed_chunks.py")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM, help="Gemini model name")
    parser.add_argument("--n_results", type=int, default=5, help="Chunks to retrieve")
    args = parser.parse_args()

    if args.mode == "qa" and not args.question:
        parser.error("--question is required in qa mode")
    if args.mode == "extract" and not args.company:
        parser.error("--company is required in extract mode")

    collection = open_collection(Path(args.db_dir), args.collection)
    encoder = load_encoder(args.embedding_model)

    if args.mode == "qa":
        result = answer_question(
            args.question, collection, encoder,
            n_results=args.n_results, company=args.company,
            year=args.year, llm_model=args.llm_model,
        )
        print(f"\n{result['answer']}")
        print_sources(result["sources"])
    else:
        result = extract_financials(
            args.company, collection, encoder,
            year=args.year, n_results=args.n_results, llm_model=args.llm_model,
        )
        if "error" in result:
            print(f"\nError: {result['error']}")
        else:
            print("\n" + json.dumps(result["data"], indent=2, ensure_ascii=False))
        print_sources(result.get("sources", []))


if __name__ == "__main__":
    main()
