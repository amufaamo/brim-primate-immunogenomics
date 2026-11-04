import os
import re
import sys
from pypdf import PdfReader
import fitz  # PyMuPDF

PDF_PATH = "./data/brim_nature_submission_complete.pdf"
MANUSCRIPT_PATH = "./data/brim_nature_submission_manuscript.md"
AUDIT_OUT_PATH = "./scripts/final_qa_audit.txt"

print(f"Auditing complete submission PDF: {PDF_PATH}")

if not os.path.exists(PDF_PATH):
    print(f"Error: {PDF_PATH} does not exist!")
    sys.exit(1)

# 1. Basic PDF metrics
file_size_bytes = os.path.getsize(PDF_PATH)
file_size_mb = file_size_bytes / (1024 * 1024)

reader = PdfReader(PDF_PATH)
doc = fitz.open(PDF_PATH)
total_pages = len(reader.pages)

# Extract full text
full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

# 2. Count embedded images
total_images = 0
image_details = []
for page_num in range(len(doc)):
    page = doc[page_num]
    imgs = page.get_images(full=True)
    if imgs:
        total_images += len(imgs)
        image_details.append((page_num + 1, len(imgs)))

# 3. Manuscript word and character counts
with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
    ms_text = f.read()

# Title
title_m = re.search(r"^#\s+(.*)", ms_text, re.M)
title = title_m.group(1).strip() if title_m else ""
title_char_count = len(title)

# Abstract
abs_m = re.search(r'\*\*(Adaptive immune receptor repertoires.*?\*\*)\n', ms_text, re.DOTALL)
if abs_m:
    abstract = abs_m.group(1).replace("**", "").strip()
    abstract_words = len(abstract.split())
else:
    abstract = ""
    abstract_words = 0

# Main text word count (Introduction, Results, Discussion)
ref_split = re.split(r'\n## References\n', ms_text)
before_refs = ref_split[0]
prose_start = before_refs.find("---", before_refs.find("Adaptive immune"))
if prose_start != -1:
    prose = before_refs[prose_start+3:].strip()
else:
    prose = before_refs
# Remove markdown headers and formatting for word count
prose_clean = re.sub(r'#+\s+.*', '', prose)
main_text_words = len(prose_clean.split())

# 4. Forbidden patterns check
forbidden_patterns = [
    "zenodo.XXXXXXXX",
    "placeholder",
    "Masakazu Ishikawa",
    "M.K.",
    "Consortium",
    "gdoc_comments",
    "これは確認",
    "GARD)",
    r"\omega",
    r"\times",
    "imes",
    "$$",
    "Fig. 1d",
    "39.4",
    "51.5",
    "Clint_PTRv2"
]

audit_results = {}
any_forbidden_hits = False
for pattern in forbidden_patterns:
    # Use exact string count
    hits = full_text.count(pattern)
    audit_results[pattern] = hits
    if hits > 0:
        any_forbidden_hits = True

# Format report
report_lines = []
report_lines.append("================================================================================")
report_lines.append("BRIM NATURE SUBMISSION COMPLETE PACKAGE - FINAL QUALITY ASSURANCE AUDIT REPORT")
report_lines.append("================================================================================")
report_lines.append("")
report_lines.append("1. DOCUMENT SPECIFICATIONS & LENGTH METRICS")
report_lines.append(f"   - Target PDF: {PDF_PATH}")
report_lines.append(f"   - File Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
report_lines.append(f"   - Total Page Count: {total_pages} pages")
report_lines.append(f"   - Embedded Figures: {total_images} images (Expected: 8 [Main 1-4 + Ext 1-4])")
for p_num, count in image_details:
    report_lines.append(f"     * Page {p_num:2d}: {count} image(s)")
report_lines.append("")
report_lines.append("2. MANUSCRIPT SCOPE & WORD COUNT COMPLIANCE")
report_lines.append(f"   - Title: \"{title}\"")
report_lines.append(f"   - Title Character Count: {title_char_count} chars (Nature limit: <= 90 chars) -> {'PASS' if title_char_count <= 90 else 'FAIL'}")
report_lines.append(f"   - Abstract Word Count: {abstract_words} words (Nature guideline: ~150-200 words) -> {'PASS' if 140 <= abstract_words <= 220 else 'CHECK'}")
report_lines.append(f"   - Main Text Word Count: {main_text_words} words (Nature guideline: ~2,500-3,500 words) -> {'PASS' if 2400 <= main_text_words <= 3600 else 'CHECK'}")
report_lines.append("")
report_lines.append("3. FORBIDDEN STRINGS & HALLUCINATION CHECK (0 HITS ASSERTION)")
report_lines.append("   -----------------------------------------------------------------------------")
report_lines.append(f"   {'Pattern':<25} | {'Hits':<8} | {'Status'}")
report_lines.append("   -----------------------------------------------------------------------------")
for pattern, hits in audit_results.items():
    status = "PASS (0 hits)" if hits == 0 else f"FAIL ({hits} hits!)"
    report_lines.append(f"   {pattern:<25} | {hits:<8} | {status}")
report_lines.append("   -----------------------------------------------------------------------------")
report_lines.append("")
if not any_forbidden_hits:
    report_lines.append("AUDIT RESULT: ALL ASSERTIONS PASSED! ZERO FORBIDDEN PATTERNS DETECTED.")
else:
    report_lines.append("AUDIT RESULT: FAILED! ONE OR MORE FORBIDDEN PATTERNS DETECTED.")
report_lines.append("================================================================================")

report_text = "\n".join(report_lines)
print(report_text)

with open(AUDIT_OUT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\nAudit report successfully written to {AUDIT_OUT_PATH}")

if any_forbidden_hits:
    sys.exit(1)
