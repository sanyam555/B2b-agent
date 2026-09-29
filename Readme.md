# B2B Retail Intelligence Assistant — Hybrid Structured + Unstructured RAG

A question-answering system over a B2B wholesale business's data, combining two very
different data sources behind one interface:

- **Structured data**: 1M rows of purchase orders (retailer, SKU, quantity, price,
  date, status) — answered by having an LLM write and execute real Pandas code.
- **Unstructured data**: free-text relationship notes about each retailer — answered
  by classic retrieval-augmented generation (embed, index, retrieve, cite).

A router classifies each incoming question and dispatches it to whichever path (or
both) can actually answer it, then combines the results.

## Architecture

```
question
   |
   v
router.py -------------------> classifies into: structured / unstructured / hybrid
   |
   +--[structured]--> structured_query.py --> LLM writes Pandas code --> sandboxed execution --> answer
   |
   +--[unstructured]--> unstructured_query.py --> vector search over notes --> cited LLM answer
   |
   +--[hybrid]--> unstructured_query.py runs first
                  its answer is passed as CONTEXT into structured_query.py
                  both results are combined into one bulleted answer

api.py (FastAPI)  --  POST /ask  --  wraps the above in a web API
app.py (Streamlit) --  simple UI that calls the API
evaluate.py         --  measures how well all of the above actually works
```

## Why a hybrid router instead of one RAG pipeline?

Standard RAG assumes everything worth knowing is in a text corpus you can embed and
retrieve. Here, half the useful information (order volumes, prices, statuses) is
tabular and better answered by exact computation than by retrieval — no embedding of
"how many orders are backordered" is going to beat literally counting the rows. The
router's job is deciding, per question, which kind of reasoning is actually needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain langchain-huggingface langchain-chroma langchain-ollama pandas fastapi uvicorn streamlit requests
```

Requires [Ollama](https://ollama.com) running locally with the `llama3.2` model pulled
(`ollama pull llama3.2`). No API keys, no cost — everything runs on local inference.

The 1M-row orders CSV (`b2b_wholesale_purchase_orders_1m.csv`) is not committed to
this repo (59MB, and regenerating is one script). Generate it with a script that
creates 1,000,000 rows across 12 retailers and 500 SKUs, with `order_date` uniformly
sampled between 2024-01-01 and the present (never future-dated, regardless of
status), and `status` weighted 90% fulfilled / 4.5% backordered / 3.5% pending / 2%
cancelled.

## Pipeline

1. **`ingest_notes.py`** — chunks and embeds the retailer notes (`sentence-transformers/all-MiniLM-L6-v2`)
   into a Chroma vector store (`notes_chroma_db/`). Chunk size 300 / overlap 10 —
   deliberately larger than the longest real note (measured max: 141 chars), making
   chunking a functional no-op here rather than a real splitting decision, since the
   notes are short enough not to need splitting.
2. **`router.py`** — classifies a question into `structured` / `unstructured` /
   `hybrid` via an LLM call with few-shot examples. Plain rule-based instructions
   alone were not reliable here (see Findings) — few-shot examples were required to
   get consistent classification.
3. **`structured_query.py`** — has the LLM write Pandas code against the orders
   dataframe, then runs it in a sandboxed subprocess (banned-pattern check, timeout,
   isolated execution — same safety pattern as an earlier project's coding
   assistant). Accepts an optional `context` parameter so hybrid questions can be
   grounded in what the unstructured path already found.
4. **`unstructured_query.py`** — retrieves relevant notes and generates a
   citation-forcing answer, citing `[retailer_name | note_date]` per claim.
5. **`api.py`** — FastAPI wrapper exposing `POST /ask`.
6. **`app.py`** — Streamlit UI calling the API.
7. **`evaluate.py`** — the evaluation suite (see Findings below for actual numbers).

## Findings

Real, measured results from `evaluate.py` — not aspirational claims:

- **Router accuracy: 86%** (6/7 test questions). The one consistent failure mode:
  compound questions that mix a structured signal and an unstructured signal in one
  sentence (e.g. "which retailers *complained about* backorders *and* have existing
  backorders") sometimes get classified as purely `structured`, missing that half the
  question needs the notes.
- **Structured path consistency: 80%** (4/5 correct across repeated runs of the same
  question). The local LLM (llama3.2) is not perfectly deterministic — the same
  question, same prompt, occasionally regresses to returning an unfiltered boolean
  mask instead of a count, or computes a wrong number outright. This is why the eval
  suite runs each structured question multiple times rather than once: a single
  passing run does not prove the system works reliably.
- **Unstructured retrieval: Hit Rate@5 = 1.00, MRR@5 = 1.00** — very strong; the
  right note is reliably retrieved and reliably ranked first. **Context Precision:
  0.70** — roughly 30% of retrieved notes are topically unrelated noise. One
  specific, reproducible leak: a note about a *different* retailer (Atlantic School
  Supply, re: evaluating a competitor over delivery delays) surfaces in the top-5 for
  unrelated queries about pricing and shipping, likely due to generic semantic
  similarity in its phrasing.
- **Faithfulness: 0.76, Answer Relevance: 0.60** — generated answers are generally
  grounded in retrieved notes, though the relevance heuristic (word overlap between
  question and answer) understates quality somewhat, since it penalizes valid
  paraphrasing.
- **Latency: structured ~3s, unstructured ~10s, hybrid ~42s** — hybrid is not simply
  the sum of its two calls (~13s); it's roughly 3x that. The cause: when given
  context from the notes, the structured LLM call tends to generate much longer,
  multi-branch code (trying to verify each individual fact mentioned in the notes as
  a separate condition) instead of one clean computation — more generated output
  tokens directly costs more generation time on local inference.

## Known limitation: fuzzy, judgment-based hybrid questions

Questions like "is Retailer X reliable?" don't have one well-defined computation —
"reliable" isn't a column or a formula. Across many iterations, the structured half
of these questions produced inconsistent, sometimes fabricated results: inventing
placeholder filter values, hallucinating a join between note dates and order dates,
inventing columns that don't exist (`df['backordered']` instead of
`df[df['status'] == 'backordered']`), and generating multi-branch code trying to
individually "prove" every fact mentioned in the notes context rather than computing
one clear answer. Passing the notes as context to the structured path measurably
helps when the question is concrete (e.g. correctly computing a specific retailer's
backorder count), but does not reliably resolve genuinely open-ended judgment
questions. This is treated as a documented limitation of small local models on this
class of question, not something patched away with more prompt rules — each
additional rule fixed one specific failure while the model found new ones.

## Repo structure

```
.
├── README.md
├── .gitignore
├── retailer_notes.csv          # small, committed
├── ingest_notes.py
├── router.py
├── structured_query.py
├── unstructured_query.py
├── evaluate.py
├── api.py
└── app.py
