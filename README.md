# Offline Loan-Page Classifier (v3)

TIFF -> deskew/rotate -> Tesseract OCR (text, header, top, footer, OCR confidence, layout thumbnail)
     -> rules (IDF-weighted keywords, header boost, regex)  +  ML model (TF-IDF + layout + page position)
     -> blend (rules gain weight when they fire strongly) -> page-order smoothing (Viterbi) -> CSV/JSON
Runs fully offline.

## Measured accuracy (leave-one-file-out on your 5 labelled files, 223 pages)
| rules only | model + rules | + page-order smoothing (what `classify` uses) |
|---|---|---|
| 61% | 76% | **92%** |
Every file is predicted by a model that never saw it, so this is a realistic estimate for new files of the same type.
Weakest category: `Credit_Supported_Documents` (mixed content: ration cards, handwritten/vernacular letters, receipts).
Categories with <3 labelled pages (Electricity bill, Driving licence, Voter ID, Registration certificate) and the ~39
categories not present in the sample files rely on the keyword rules until you label more pages.

## Setup
1. Install Tesseract OCR (Windows: UB-Mannheim installer). For Telugu/Odia/Hindi pages install the language data and use `--lang eng+tel`.
   If not on PATH: `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`
2. `pip install -r requirements.txt`  (air-gapped: `pip download -r requirements.txt -d wheels`, then `pip install --no-index --find-links wheels -r requirements.txt`)

## Use
```bash
# classify a new file (uses the shipped model.joblib trained on your 5 files)
python classify.py classify newfile.tif --split out_pages/

# measure accuracy / find weak categories (Excel: one column per file, header = TIFF file name, one label per page row)
python classify.py eval  --tiff a.tif b.tif c.tif --labels Tag.xlsx --show-errors

# retrain after adding more labelled files (CSV page,category or the Excel)
python classify.py train --tiff a.tif b.tif c.tif --labels Tag.xlsx
```
Options: `--workers 1` (no multiprocessing; use on low RAM / Windows problems), `--lang eng+tel`, `--threshold 0.35`,
`--no-model`, `--no-seq`, `--seq-weight 0.4`, `--blank-label blank_page`, `--photo-rule`.

## Notes
- Label names are matched ignoring case/spaces/underscores and typos; extra spellings go in `aliases:` in `categories.yaml`.
- Blank pages take the neighbouring page's category (your labels never use a blank class). Use `--blank-label blank_page` to keep them separate.
- Telugu/Odia pages are labelled Loan_Agreement in your sheet, so the Vernacular fallback is off (`--vernacular` enables it).
- `Photograph` detection from the image is off by default (it misfired on sparse text pages). Label some Photograph pages and retrain instead, or try `--photo-rule`.
- `needs_review` is a weak signal (it catches only ~1 in 4 errors at the default threshold); treat low-confidence flags as hints, and use `eval` to see real accuracy.
- OCR is cached per page in `.ocr_cache/` keyed by file content, so renamed copies reuse it and interrupted runs resume.

## Troubleshooting
`BrokenProcessPool`: a parallel OCR worker crashed (usually RAM or Tesseract). Defaults are already conservative and it retries
sequentially; use `--workers 1`, run from a terminal (not Jupyter on Windows), and make sure Tesseract is installed (`TESSERACT_CMD`).
