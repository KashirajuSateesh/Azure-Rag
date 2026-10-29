# Scanned PDF extraction: Document Intelligence vs. an LLM (Claude on Foundry)

## Files
| File | What it does |
|---|---|
| `scanned_invoice.pdf` | Test document: an image-only scan (skewed, noisy, blurred, no text layer) |
| `ground_truth.json` | The exact text, fields and table in the scan, used for scoring |
| `1_docintel_extract.py` | Option A: Azure AI Document Intelligence `prebuilt-layout` writes `output/docintel_output.json` |
| `2_llm_extract.py` | Option B: Claude (Foundry deployment) with vision writes `output/llm_output.json` |
| `3_compare.py` | Scores both against the ground truth and writes `output/comparison_report.json` |

## 1. Azure setup
1. **Document Intelligence resource**: Azure Portal, then *Create a resource*, then **Document Intelligence**. Free tier F0 works.
   Open the resource, then *Keys and Endpoint*. Copy the **Endpoint** and **Key 1**.
2. **Claude deployment**: in Foundry, go to *Models*, then `claude-sonnet-5-5`, then the **Details** tab. Copy the **Target URI**
   (`https://<resource>.services.ai.azure.com/anthropic/...`) and the **Key**.
   Use the base form ending in `/anthropic/`.

## 2. Local setup (your VS Code venv)
```cmd
pip install -r requirements.txt
copy .env.example .env      &REM then fill in the 5 values
```

## 3. Run
```cmd
python 1_docintel_extract.py scanned_invoice.pdf
python 2_llm_extract.py scanned_invoice.pdf
python 3_compare.py
```

## 4. How to judge "which works better"
`3_compare.py` covers accuracy. Add these points to your write-up:

| Criterion | Document Intelligence | LLM (Claude vision) |
|---|---|---|
| OCR fidelity on clean scans | Very high, deterministic | Very high, but can "fix" typos or paraphrase |
| Hallucination risk | None. It only reads pixels | Low but non-zero (check `word_precision`) |
| Tables | Native table model with row/col spans | Good, but long tables can drop or merge rows |
| Confidence + bounding boxes | Yes, per word (useful for citations and highlights in RAG) | No |
| Understanding / semantics | None. Raw structure only | Strong: can label fields, normalize dates, summarize |
| Speed / cost | Fast, priced per page | Slower, priced per token (images are input-heavy) |
| Long docs (100+ pages) | Built for it (async) | Context limits, so you have to batch pages |
| Handwriting / bad scans | Good | Often better at reading messy context |

**Typical conclusion for RAG:** use Document Intelligence for ingestion (faithful text, tables, page/box
citations, cheap at scale). Use the LLM afterwards for enrichment such as field extraction, chunk summaries
and metadata, or as a fallback for pages where DI confidence is low.
Base your final verdict on the numbers you actually get.

## Next steps toward the RAG project
Chunk `docintel_output.json` by paragraph or table, then embed it (Foundry embedding model), then index it in Azure AI Search, then answer with Claude.
