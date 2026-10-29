"""Render the report and check exactly ten pages, text bounds, and actual fonts."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import pymupdf


def verify(pdf, preview, qa):
    pdf, preview, qa = Path(pdf), Path(preview), Path(qa)
    preview.mkdir(parents=True, exist_ok=True)
    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise RuntimeError("pdftoppm is needed for PDF visual QA")
    subprocess.run([renderer, "-r", "110", "-png", str(pdf), str(preview / "page")], check=True)
    document = pymupdf.open(pdf)
    checks = []
    all_text = []
    for i, page in enumerate(document):
        text = page.get_text()
        all_text.append(text)
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"]]
        clipped = [s["text"] for s in spans
                   if s["bbox"][0] < -0.5 or s["bbox"][1] < -0.5
                   or s["bbox"][2] > page.rect.width + 0.5 or s["bbox"][3] > page.rect.height + 0.5]
        checks.append({"page": i + 1, "characters": len(text),
                       "min_effective_font_pt": min(s["size"] for s in spans),
                       "clipped_text": clipped, "unresolved_reference": "??" in text})
    result = {"pages": len(document), "within_ten_page_limit": 1 <= len(document) <= 10,
              "exactly_ten_pages": len(document) == 10,
              "page_checks": checks, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    result["automated_checks_pass"] = (result["exactly_ten_pages"] and
        all(not p["clipped_text"] and not p["unresolved_reference"] and
            p["min_effective_font_pt"] >= 5 for p in checks))
    qa.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (preview / "report_text.txt").write_text("\n\n".join(all_text), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["automated_checks_pass"]:
        raise RuntimeError("Report QA failed; inspect the report before delivery")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default="report/manuscript/main.pdf")
    parser.add_argument("--preview", default="tmp/report-preview")
    parser.add_argument("--qa", default="report/pdf_qa.json")
    args = parser.parse_args()
    verify(args.pdf, args.preview, args.qa)
