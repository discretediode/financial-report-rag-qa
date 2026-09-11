# Financial Report Q&A

A Retrieval-Augmented Generation (RAG) system for querying company annual reports
in natural language. Ask questions about financial filings and get answers grounded
in the source documents, with citations — or extract key financial figures as
structured JSON.

Built around German DAX company annual reports (Geschäftsberichte), but works with
any PDF filing, including SEC 10-Ks.

## Why this exists

Annual reports are long, unstructured PDFs. Finding a specific fact means reading
hundreds of pages, and general-purpose LLMs can't answer questions about documents
they've never seen. This system indexes the filings locally and retrieves the
relevant passages before answering, so every response is traceable to real text
rather than model memory.

## How it works

```
PDFs → extract → chunk → embed → vector store → query
```

1. **Extract** — pull text from PDFs page by page, flagging scanned pages
2. **Chunk** — split into overlapping passages, tagged with company/year metadata
3. **Embed** — encode each chunk as a 384-dimensional vector (HuggingFace / PyTorch)
4. **Store** — load vectors, text, and metadata into a persistent ChromaDB collection
5. **Query** — encode the question, retrieve nearest chunks, generate a grounded answer

Steps 1–4 are one-time ingestion. Step 5 is what you run repeatedly.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key-here"
```

Name your PDFs `company_year.pdf` — the pipeline parses metadata from the filename:

```
pdfs/
  siemens_2024.pdf
  bmw_2023.pdf
  deutsche-bank_2024.pdf
```

## Running the pipeline

```bash
python extract_pdfs.py       --input_dir ./pdfs        --output_dir ./extracted
python chunk_documents.py    --input_dir ./extracted   --output_dir ./chunks
python embed_chunks.py       --input_file ./chunks/chunks.json --output_dir ./embeddings
python build_vectorstore.py  --input_dir ./embeddings  --db_dir ./chroma_db
```

The first run downloads the embedding model (~90 MB) and caches it.

## Querying

**Natural-language Q&A:**

```bash
python query.py --question "What were Siemens' main risk factors in 2024?"
python query.py --question "How did revenue change?" --company Siemens --year 2024
```

Answers cite their sources as `[Source N]`, and the retrieved chunks are printed
below each answer with their company, year, and similarity distance.

**Structured extraction:**

```bash
python query.py --mode extract --company Siemens --year 2024
```

Returns schema-validated JSON:

```json
{
  "company": "Siemens",
  "fiscal_year": 2024,
  "revenue": 75900,
  "revenue_unit": "EUR millions",
  "operating_profit": null,
  "net_income": null,
  "employee_count": null,
  "key_risks": ["Supply chain disruption", "Currency exposure", "EU regulatory change"]
}
```

Fields not stated in the retrieved text come back as `null` rather than guessed.

## Project structure

| File | Role |
|---|---|
| `extract_pdfs.py` | PDF → per-document JSON with page-level text |
| `chunk_documents.py` | Documents → overlapping, metadata-tagged chunks |
| `embed_chunks.py` | Chunks → vectors (`embeddings.npy` + metadata) |
| `build_vectorstore.py` | Vectors + text + metadata → ChromaDB collection |
| `query.py` | Query-time RAG: retrieve, ground, answer or extract |

## Design notes

**Embeddings are computed separately from storage.** Vectors are generated once by
`embed_chunks.py` and inserted into ChromaDB pre-computed, rather than letting Chroma
embed on insert. This keeps the store rebuildable without re-encoding the corpus, at
the cost of one constraint: queries must be encoded with the same model as the
documents. A mismatch produces plausible-looking nonsense with no error raised, so the
model name is pinned in both scripts.

**Vectors live in `.npy`, not JSON.** Thousands of 384-float arrays serialize poorly
as text. The metadata JSON is kept in the same row order, and both scripts fail loudly
if the counts diverge — silently misaligned vectors and text would produce confidently
wrong retrieval.

**The LLM does not do arithmetic.** The QA prompt explicitly forbids calculating
ratios. Multi-step math inside a language model is unreliable, especially across
figures drawn from different chunks. Financial ratios should be computed in Python
from extracted numeric fields instead.

**Extraction uses schema-constrained generation**, not a prompt asking for JSON.
Passing a `response_schema` to the API guarantees parseable output, eliminating the
retry-on-malformed-JSON loop that prompt-based extraction requires.

**Failures are isolated per file.** One corrupt PDF logs an error and the batch
continues, rather than losing progress on every other document.

## Known limitations

- **Tables extract poorly.** `pdfplumber`'s `extract_text()` flattens balance sheets
  and cash-flow statements into unstructured number sequences. Prose sections (risk
  factors, MD&A) extract cleanly; financial statement pages need `extract_tables()`.
- **Scanned pages are skipped, not OCR'd.** They're flagged in the extraction summary
  with a page count, but recovering them requires adding an OCR step.
- **Chunk size is measured in characters, not tokens.** 500 characters is roughly
  100–125 tokens. Dense financial prose may benefit from `--chunk_size 1000`.
- **No cross-document reasoning.** Each query runs a single retrieval pass. Comparison
  questions spanning two companies are answered from whatever one pass returns, not
  from separate per-company retrievals.
- **Market-derived ratios are out of scope.** P/E, P/B, and Sharpe ratio all require
  stock price data, which does not appear in annual reports.

## Possible extensions

- Financial ratio calculation (ROE, margin, debt-to-equity) in Python from extracted fields
- Query routing — decide per question whether to retrieve, calculate, or both
- Multi-hop retrieval for cross-company comparisons
- Table-aware extraction via `pdfplumber.extract_tables()`
- A Gradio or Streamlit front end
