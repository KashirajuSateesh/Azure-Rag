"""
Option B: Extract the same scanned PDF with the Claude model deployed in Microsoft Foundry.

The PDF pages are rendered to PNG images and sent to the model (vision), which is asked
to return ONLY JSON in a fixed schema so it can be compared with Document Intelligence.

Usage:
    python 2_llm_extract.py scanned_invoice.pdf
Output:
    output/llm_output.json
"""
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
# From Foundry > your model > Details:  https://<resource>.services.ai.azure.com/anthropic/
ENDPOINT = os.environ["FOUNDRY_ANTHROPIC_ENDPOINT"].strip()
API_KEY = os.environ["FOUNDRY_API_KEY"].strip()
DEPLOYMENT = os.environ.get("FOUNDRY_DEPLOYMENT", "claude-sonnet-5-5").strip()

# 404 "api_not_supported" = the URL path is wrong (e.g. an /openai/... or /api/projects/... URL,
# or ".../anthropic/v1/messages" pasted in full so the SDK calls /v1/messages twice).
# Whatever was pasted, rebuild the one URL Claude needs from the resource name.
_m = re.match(r"^https://([\w-]+)\.(?:services\.ai|openai|cognitiveservices)\.azure\.com", ENDPOINT)
if not _m:
    raise SystemExit(f"Can't read the resource name from FOUNDRY_ANTHROPIC_ENDPOINT: {ENDPOINT}")
ENDPOINT = f"https://{_m.group(1)}.services.ai.azure.com/anthropic"
print(f"Calling {ENDPOINT}/v1/messages | deployment: {DEPLOYMENT}")

PROMPT = """You are a document OCR and extraction engine. The images are pages of a scanned document.
Extract ALL content exactly as printed. Do not summarize, correct, or invent anything.
If something is unreadable write "[illegible]".

Return ONLY valid JSON (no markdown fences) with this schema:
{
  "full_text": "all text in reading order, lines separated by \\n",
  "paragraphs": [{"role": "title|sectionHeading|body|footer|null", "content": "..."}],
  "key_value_pairs": [{"key": "...", "value": "..."}],
  "tables": [{"row_count": 0, "column_count": 0, "rows": [["header1", "header2"], ["cell", "cell"]]}],
  "pages": [{"page_number": 1, "lines": ["line 1", "line 2"]}]
}"""


def pdf_to_png_blocks(pdf_path: str, dpi: int = 200) -> list[dict]:
    """Render each PDF page to PNG and wrap it as a Claude image content block."""
    blocks = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            png = page.get_pixmap(dpi=dpi).tobytes("png")
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                           "data": base64.b64encode(png).decode()},
            })
    return blocks


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # strip accidental fences
    return json.loads(text)


def main(pdf_path: str) -> None:
    client = Anthropic(api_key=API_KEY, base_url=ENDPOINT)  # same request as the working curl
    images = pdf_to_png_blocks(pdf_path)

    start = time.perf_counter()
    response = client.messages.create(
        model=DEPLOYMENT,
        max_tokens=8000,
        messages=[{"role": "user", "content": images + [{"type": "text", "text": PROMPT}]}],
    )
    elapsed = time.perf_counter() - start

    raw_text = response.content[0].text
    try:
        extracted = parse_json(raw_text)
    except json.JSONDecodeError:
        print("WARNING: model did not return valid JSON - saving raw text instead.")
        extracted = {"full_text": raw_text, "parse_error": True}

    output = {
        "source_file": Path(pdf_path).name,
        "method": f"LLM vision - {DEPLOYMENT} (Microsoft Foundry)",
        "processing_seconds": round(elapsed, 2),
        "page_count": len(images),
        "usage": {"input_tokens": response.usage.input_tokens,
                  "output_tokens": response.usage.output_tokens},
        **extracted,
    }

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "llm_output.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done in {elapsed:.1f}s | tokens in={response.usage.input_tokens} out={response.usage.output_tokens}")
    print("Saved output/llm_output.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scanned_invoice.pdf")