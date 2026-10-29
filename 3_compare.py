"""
Compare Document Intelligence vs LLM extraction against a known ground truth.

Usage (after running scripts 1 and 2):
    python 3_compare.py
Output:
    printed scorecard + output/comparison_report.json

Metrics
- word_recall:      % of ground-truth words found in the extracted text (did it miss anything?)
- word_precision:   % of extracted words that exist in the ground truth (did it add/hallucinate?)
- char_similarity:  difflib ratio of normalized full text (overall OCR fidelity)
- critical_fields:  exact hits on values that matter (invoice no, totals, account numbers...)
- table_cell_acc:   % of ground-truth table cells reproduced exactly in the right row/column
- seconds / cost signals
"""
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def words(text: str) -> Counter:
    return Counter(re.findall(r"[\w$@.,%()\-]+", norm(text)))


def table_cell_accuracy(gt_rows, tables) -> float:
    if not tables:
        return 0.0
    best = 0.0
    gt_cells = [(r, c, v) for r, row in enumerate(gt_rows) for c, v in enumerate(row)]
    for t in tables:
        rows = t.get("rows") or []
        hits = sum(
            1 for r, c, v in gt_cells
            if r < len(rows) and c < len(rows[r]) and norm(rows[r][c]) == norm(v)
        )
        best = max(best, hits / len(gt_cells))
    return best


def score(name: str, data: dict, gt: dict) -> dict:
    text = data.get("full_text", "")
    gt_w, ex_w = words(gt["full_text"]), words(text)
    overlap = sum((gt_w & ex_w).values())
    flat = norm(json.dumps(data, ensure_ascii=False))
    field_hits = {k: norm(v) in flat for k, v in gt["critical_fields"].items()}
    return {
        "method": data.get("method", name),
        "seconds": data.get("processing_seconds"),
        "word_recall": round(overlap / max(sum(gt_w.values()), 1), 4),
        "word_precision": round(overlap / max(sum(ex_w.values()), 1), 4),
        "char_similarity": round(SequenceMatcher(None, norm(gt["full_text"]), norm(text)).ratio(), 4),
        "critical_fields_correct": f"{sum(field_hits.values())}/{len(field_hits)}",
        "critical_fields_missed": [k for k, ok in field_hits.items() if not ok],
        "tables_found": len(data.get("tables") or []),
        "table_cell_accuracy": round(table_cell_accuracy(gt["table"], data.get("tables")), 4),
        "key_value_pairs_found": len(data.get("key_value_pairs") or []),
        "has_confidence_scores": "confidence" in flat,
        "has_bounding_boxes": name == "docintel",   # raw DI output includes polygons per word
        "usage": data.get("usage"),
    }


def main() -> None:
    gt = json.loads(Path("ground_truth.json").read_text(encoding="utf-8"))
    results = {}
    for name, file in [("docintel", "output/docintel_output.json"), ("llm", "output/llm_output.json")]:
        p = Path(file)
        if p.exists():
            results[name] = score(name, json.loads(p.read_text(encoding="utf-8")), gt)
        else:
            print(f"Skipping {name}: {file} not found")

    keys = ["seconds", "word_recall", "word_precision", "char_similarity", "critical_fields_correct",
            "tables_found", "table_cell_accuracy", "key_value_pairs_found", "has_confidence_scores",
            "has_bounding_boxes"]
    print(f"\n{'metric':<26}" + "".join(f"{n:>20}" for n in results))
    print("-" * (26 + 20 * len(results)))
    for k in keys:
        print(f"{k:<26}" + "".join(f"{str(r[k]):>20}" for r in results.values()))
    for n, r in results.items():
        if r["critical_fields_missed"]:
            print(f"\n{n} missed fields: {r['critical_fields_missed']}")

    Path("output").mkdir(exist_ok=True)
    Path("output/comparison_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved output/comparison_report.json")


if __name__ == "__main__":
    main()
