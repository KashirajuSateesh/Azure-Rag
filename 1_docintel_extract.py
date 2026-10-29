"""
Option A: Extract a scanned PDF with Azure AI Document Intelligence (prebuilt-layout).

Usage:
    python 1_docintel_extract.py scanned_invoice.pdf
Output:
    output/docintel_output.json      (clean, readable structure)
    output/docintel_raw.json         (full raw response from the service)
"""
import json
import sys
import time
from pathlib import Path
import os

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature

load_dotenv()
ENDPOINT = os.environ["DOCINTEL_ENDPOINT"]   # e.g. https://<name>.cognitiveservices.azure.com/
KEY = os.environ["DOCINTEL_KEY"]


def main(pdf_path: str) -> None:
    client = DocumentIntelligenceClient(ENDPOINT, AzureKeyCredential(KEY))

    start = time.perf_counter()
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",                                  # OCR + tables + structure
            body=f,
            features=[DocumentAnalysisFeature.KEY_VALUE_PAIRS],  # also pull "Invoice No: X" pairs
        )
    result = poller.result()
    elapsed = time.perf_counter() - start

    pages = []
    for p in result.pages:
        pages.append({
            "page_number": p.page_number,
            "width": p.width,
            "height": p.height,
            "unit": p.unit,
            "angle": p.angle,  # detected skew of the scan
            "lines": [ln.content for ln in (p.lines or [])],
            "low_confidence_words": [
                {"text": w.content, "confidence": round(w.confidence, 3)}
                for w in (p.words or []) if w.confidence < 0.8
            ],
        })

    tables = []
    for t in result.tables or []:
        grid = [["" for _ in range(t.column_count)] for _ in range(t.row_count)]
        for cell in t.cells:
            grid[cell.row_index][cell.column_index] = cell.content
        tables.append({"row_count": t.row_count, "column_count": t.column_count, "rows": grid})

    kv_pairs = [
        {
            "key": kv.key.content if kv.key else None,
            "value": kv.value.content if kv.value else None,
            "confidence": round(kv.confidence, 3) if kv.confidence else None,
        }
        for kv in (result.key_value_pairs or [])
    ]

    paragraphs = [
        {"role": para.role, "content": para.content} for para in (result.paragraphs or [])
    ]

    output = {
        "source_file": Path(pdf_path).name,
        "method": "Azure AI Document Intelligence - prebuilt-layout",
        "processing_seconds": round(elapsed, 2),
        "page_count": len(result.pages),
        "full_text": result.content,
        "paragraphs": paragraphs,
        "key_value_pairs": kv_pairs,
        "tables": tables,
        "pages": pages,
    }

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "docintel_output.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "docintel_raw.json").write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done in {elapsed:.1f}s | pages={len(result.pages)} tables={len(tables)} kv_pairs={len(kv_pairs)}")
    print("Saved output/docintel_output.json and output/docintel_raw.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scanned_invoice.pdf")
