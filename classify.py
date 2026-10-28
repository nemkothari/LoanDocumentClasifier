#!/usr/bin/env python3
"""
Offline page classifier for multi-page TIFF loan documents  (v2 - higher accuracy)

Pipeline
  TIFF -> deskew/rotate -> Tesseract OCR (full text + page-header text + OCR confidence)
       -> rules (IDF-weighted keywords, header boost, fuzzy OCR tolerance, regex)
       -> ML model (TF-IDF on text + header + image features), blended with rules
       -> sequence smoothing (Viterbi over learned page-to-page transitions)
       -> CSV / JSON (+ optional split TIFFs)

Commands
  classify : classify every page of a TIFF
  train    : train model + page-order transitions from corrected CSVs (page,category)
  eval     : measure accuracy (rules-only vs cross-validated model) on labelled pages
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one Tesseract thread per worker process

import numpy as np
import yaml
import pytesseract
from PIL import Image, ImageOps

try:  # optional: tolerance for OCR typos
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.joblib"
CACHE_DIR = HERE / ".ocr_cache"
CACHE_VERSION = "v3"
BLANK, UNKNOWN = "blank_page", "unknown"
COMMON = set("the and of to in is on for by or as with this that be are at from an any shall agreement date name no "
             "amount bank loan page signature address number account".split())


# ============================================================================ OCR
def eng_score(text):
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    return sum(t in COMMON for t in toks) / len(toks) if len(toks) > 8 else 0.0


def deskew(img):
    """Fix small scan skew (+-4 deg) by maximising horizontal text-line contrast."""
    small = img.copy()
    small.thumbnail((700, 700))
    ink = (255 - np.asarray(small, dtype=np.uint8) > 90).astype(np.uint8) * 255
    base = Image.fromarray(ink)
    score = lambda ang: float(np.asarray(base.rotate(ang, resample=Image.BILINEAR), dtype=np.float32).sum(axis=1).var())
    s0 = score(0)
    best, best_s = 0.0, s0
    for ang in np.arange(-4, 4.01, 0.5):
        s = score(float(ang))
        if s > best_s:
            best, best_s = float(ang), s
    if abs(best) >= 0.5 and best_s > s0 * 1.05:
        return img.rotate(best, expand=True, fillcolor=255, resample=Image.BICUBIC)
    return img


def run_ocr(img, lang):
    d = pytesseract.image_to_data(img, lang=lang, config="--psm 3", output_type=pytesseract.Output.DICT)
    H = img.height
    lines, head, top, foot, confs = OrderedDict(), [], [], [], []
    for i, w in enumerate(d["text"]):
        w = (w or "").strip()
        if not w:
            continue
        lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(w)
        y = d["top"][i]
        if y < 0.25 * H:  # titles live in the top quarter of the page
            head.append(w)
        if y < 0.12 * H:
            top.append(w)
        if y > 0.88 * H:  # footers: "Page 3 of 20", form numbers, version stamps
            foot.append(w)
        try:
            c = float(d["conf"][i])
        except (TypeError, ValueError):
            c = -1
        if c >= 0:
            confs.append(c)
    return dict(text="\n".join(" ".join(v) for v in lines.values()), header=" ".join(head), top=" ".join(top),
                foot=" ".join(foot), conf=float(np.mean(confs)) if confs else 0.0,
                words=sum(len(v) for v in lines.values()))


def image_features(img, size):
    t = np.asarray(img.resize((64, 64), Image.BOX), dtype=np.float32) / 255.0
    return [float(t.mean()), float(t.std()), float(((t > 0.25) & (t < 0.75)).mean()),
            float((t < 0.25).mean()), size[0] / max(size[1], 1)]


def ocr_one(args):
    path, idx, lang = args
    with Image.open(path) as im:
        im.seek(idx)
        img = im.convert("L")
    size = img.size
    try:  # 90/180/270 degree orientation
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rot = int(osd.get("rotate", 0))
        if rot:
            img = img.rotate(-rot, expand=True, fillcolor=255)
    except Exception:
        pass
    img = deskew(ImageOps.autocontrast(img))
    if max(img.size) < 2000:
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    r = run_ocr(img, lang)
    if eng_score(r["text"]) < 0.08 and "tel" not in lang:  # OSD can miss upside-down pages
        alt_img = img.rotate(180, fillcolor=255)
        alt = run_ocr(alt_img, lang)
        if eng_score(alt["text"]) > eng_score(r["text"]) + 0.05:
            r, img = alt, alt_img
    r["feats"] = image_features(img, size)
    r["thumb"] = [int(v) for v in np.asarray(img.resize((24, 32), Image.BOX), dtype=np.uint8).ravel()]
    return r


def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        sys.exit("Tesseract OCR was not found. Install it and add it to PATH, or set the TESSERACT_CMD "
                 "environment variable to the full path of tesseract.exe.\n" + str(e))


def default_workers():
    return max(1, min(4, (os.cpu_count() or 2) // 2))  # fewer workers = far less RAM, avoids crashed workers


def run_all(jobs, workers, save):
    """OCR jobs; save(job, result) is called as each page finishes so progress survives a crash/timeout."""
    workers = workers or default_workers()
    done = set()
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, 61)) as ex:
                for j, r in zip(jobs, ex.map(ocr_one, jobs)):
                    save(j, r)
                    done.add(j[1])
            return
        except Exception as e:  # BrokenProcessPool etc. -> retry one page at a time to get a real error message
            print(f"Parallel OCR failed ({type(e).__name__}); retrying sequentially...", file=sys.stderr)
    for n, j in enumerate(jobs, 1):
        if j[1] in done:
            continue
        print(f"  OCR page {j[1] + 1}/{len(jobs) + len(done)}", file=sys.stderr, end="\r")
        save(j, ocr_one(j))
    print(file=sys.stderr)


def cache_key(tiff, lang):
    """Content-based key: renamed/re-uploaded copies of the same scan reuse the OCR cache."""
    h = hashlib.md5()
    with open(tiff, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return hashlib.md5(f"{CACHE_VERSION}|{h.hexdigest()}|{lang}".encode()).hexdigest()


def get_pages(tiff, lang="eng", workers=None):
    """OCR all pages. Cached per page on disk, so re-runs are instant and an interrupted run resumes."""
    tiff = str(tiff)
    d = CACHE_DIR / cache_key(tiff, lang)
    d.mkdir(parents=True, exist_ok=True)
    with Image.open(tiff) as im:
        n = getattr(im, "n_frames", 1)
    todo = [i for i in range(n) if not (d / f"{i}.json").exists()]
    if todo:
        check_tesseract()
        print(f"OCR: {Path(tiff).name} ({len(todo)} of {n} pages to do)...", file=sys.stderr)
        run_all([(tiff, i, lang) for i in todo], workers,
                lambda j, r: (d / f"{j[1]}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8"))
    return [json.loads((d / f"{i}.json").read_text(encoding="utf-8")) for i in range(n)]


# ============================================================================ rules
def normalize(s):
    return re.sub(r"[^a-z0-9\u0900-\u097F]+", " ", s.lower()).strip()


def _key(label):
    return re.sub(r"[\s_]+", "", label or "").lower()


def load_cats(path):
    """Returns (categories, canon). canon() maps any alias/spelling/typo of a label to the canonical category name."""
    import difflib
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["categories"]
    amap, cats = {}, {}
    for name, c in raw.items():
        c = dict(c or {})
        for al in [name] + list(c.pop("aliases", []) or []):
            amap[_key(al)] = name
        cats[name] = c

    def canon(label):
        label = (label or "").strip()
        k = _key(label)
        if k in amap:
            return amap[k]
        m = difflib.get_close_matches(k, list(amap), n=1, cutoff=0.85)   # e.g. "Loan Agreememt"
        return amap[m[0]] if m else label

    return cats, canon


class Rules:
    """Keyword/regex scorer. Keywords shared by many categories weigh less (IDF); header hits weigh more."""

    def __init__(self, cats):
        self.kws, df = {}, Counter()
        for n, c in cats.items():
            items = [(normalize(k), base) for field, base in (("strong", 3.0), ("keywords", 1.0))
                     for k in (c.get(field) or []) if normalize(k)]
            self.kws[n] = items
            df.update({k for k, _ in items})
        N = max(len(cats), 2)
        self.w = {k: 0.5 + math.log(N / v) / math.log(N) for k, v in df.items()}
        self.pats = {n: [re.compile(p) for p in (c.get("patterns") or [])] for n, c in cats.items()}

    def score(self, text, header=""):
        norm = " " + normalize(text) + " "
        sq = norm.replace(" ", "")
        hn = " " + normalize(header) + " "
        out = {}
        for n, items in self.kws.items():
            s = 0.0
            for k, base in items:
                w = base * self.w[k]
                if f" {k} " in norm:
                    s += w + (0.5 * w if f" {k} " in hn else 0)
                else:
                    ks = k.replace(" ", "")
                    if len(ks) >= 10 and ks in sq:
                        s += 0.8 * w
                    elif fuzz is not None and len(k) >= 12 and fuzz.partial_ratio(k, norm) >= 90:
                        s += 0.6 * w
            s += 3 * sum(1 for p in self.pats[n] if p.search(text))
            out[n] = s
        return out


def rule_confidence(scores):
    r = sorted(scores.values(), reverse=True)
    best, second = r[0], (r[1] if len(r) > 1 else 0)
    return 0.0 if best <= 0 else min(1.0, best / 6) * (0.5 + 0.5 * (best - second) / best)


# ============================================================================ ML model
def annotate(pages):
    """Attach page-position info (page 1/2 and relative position are strong signals in loan files)."""
    n = len(pages)
    out = []
    for i, p in enumerate(pages):
        q = dict(p)
        q["pos"] = [float(i == 0), float(i == 1), i / max(n - 1, 1), (n - 1 - i) / max(n - 1, 1), float(i == n - 1)]
        out.append(q)
    return out


def g_text(X): return [x["text"] for x in X]
def g_header(X): return [x["header"] or "hdr_empty" for x in X]
def g_top(X): return [x.get("top") or "top_empty" for x in X]
def g_foot(X): return [x.get("foot") or "foot_empty" for x in X]
def text_stats(t):
    toks = re.findall(r"\S+", t)
    if not toks:
        return [0, 0, 0, 0, 0]
    alpha = sum(len(re.findall(r"[A-Za-z]", w)) for w in toks)
    return [eng_score(t) * 5, min(np.mean([len(w) for w in toks]), 12) / 12, sum(len(w) <= 2 for w in toks) / len(toks),
            sum(ch.isdigit() for ch in t) / max(len(t), 1) * 5, alpha / max(sum(len(w) for w in toks), 1)]


def g_num(X): return np.array([x["feats"] + [min(x["words"], 800) / 800, x["conf"] / 100] + x["pos"] + text_stats(x["text"]) for x in X])
def g_thumb(X): return np.array([x["thumb"] for x in X], dtype=np.float32) / 255.0


DEFAULT_W = {"full": 1.0, "char": 0.7, "head": 1.2, "top": 0.8, "foot": 0.6, "num": 0.5, "thumb": 0.05}


def build_pipeline(weights=None, C=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    def branch(getter, vec):
        return Pipeline([("g", FunctionTransformer(getter)), ("v", vec)])

    tf = lambda **k: TfidfVectorizer(sublinear_tf=True, **k)
    feats = FeatureUnion([
        ("full", branch(g_text, tf(ngram_range=(1, 2)))),
        ("char", branch(g_text, tf(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200000))),
        ("head", branch(g_header, tf(ngram_range=(1, 2)))),
        ("top", branch(g_top, tf(ngram_range=(1, 2)))),
        ("foot", branch(g_foot, tf(ngram_range=(1, 2)))),
        ("num", branch(g_num, StandardScaler())),
        ("thumb", branch(g_thumb, StandardScaler())),
    ], transformer_weights={**DEFAULT_W, **(weights or {})})
    return Pipeline([("f", feats), ("clf", LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))])


def read_labels(path, tiff):
    """Returns list of raw labels in page order. CSV: columns page,category. Excel: one column per file (header = file stem)."""
    if str(path).lower().endswith((".xlsx", ".xlsm")):
        import openpyxl
        rows = list(openpyxl.load_workbook(path, data_only=True).active.iter_rows(values_only=True))
        stem = Path(tiff).stem
        col = next((i for i, h in enumerate(rows[0]) if h and i > 0 and (str(h).strip() == stem or stem.startswith(str(h).strip())
                                                                          or str(h).strip().startswith(stem))), None)
        if col is None:
            sys.exit(f"No column for '{stem}' in {path}. Headers: {[h for h in rows[0]]}")
        vals = [str(r[col]).strip() if r[col] not in (None, "") else None for r in rows[1:]]
        while vals and vals[-1] is None:          # drop the empty rows below the last label
            vals.pop()
        return vals
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["page"]))
    out = {}
    for r in rows:
        out[int(r["page"])] = (r.get("category") or "").strip() or None
    return [out.get(i) for i in range(1, max(out, default=0) + 1)]


def is_blank_page(pg):
    return len(normalize(pg["text"])) < 25


def load_files(tiffs, labels, lang, workers, canon):
    """Per file: {name, pages (all OCR pages), labels (canonical label per page or None)}."""
    if len(labels) == 1 and len(tiffs) > 1:
        labels = labels * len(tiffs)
    if len(tiffs) != len(labels):
        sys.exit("Give one --labels file per --tiff (or one Excel for all).")
    files = []
    for tiff, lab in zip(tiffs, labels):
        pages = get_pages(tiff, lang, workers)
        raw = [canon(x) if x else None for x in read_labels(lab, tiff)]
        if len(raw) != len(pages):
            print(f"WARNING {Path(tiff).name}: {len(pages)} pages but {len(raw)} labels -> using the first "
                  f"{min(len(pages), len(raw))}", file=sys.stderr)
        raw = (raw + [None] * len(pages))[:len(pages)]
        files.append(dict(name=Path(tiff).name, pages=pages, labels=raw))
    return files


def flatten(files):
    """Training rows: labelled pages that have OCR text (blank pages carry no signal)."""
    X, y, origin = [], [], []
    for f in files:
        for i, (pg, lab) in enumerate(zip(annotate(f["pages"]), f["labels"]), 1):
            if lab and lab not in (BLANK, UNKNOWN) and not is_blank_page(pg):
                X.append(pg); y.append(lab); origin.append((f["name"], i))
    return X, y, origin


def build_transitions(files):
    t = defaultdict(Counter)
    for f in files:
        seq = [l for l in f["labels"] if l]
        for a, b in zip(seq, seq[1:]):
            t[a][b] += 1
    return {a: dict(c) for a, c in t.items()}


def cmd_train(a):
    import joblib
    cats, canon = load_cats(a.config)
    files = load_files(a.tiff, a.labels, a.lang, a.workers, canon)
    X, y, _ = flatten(files)
    if len(set(y)) < 2:
        sys.exit("Need at least 2 different categories in the labels.")
    pipe = build_pipeline().fit(X, y)
    a_ml = ml_weight(Counter(y))
    joblib.dump({"model": pipe, "transitions": build_transitions(files), "ml_weight": a_ml}, MODEL_PATH)
    counts = Counter(y)
    print(f"Trained on {len(y)} pages, {len(counts)} categories -> {MODEL_PATH}  (model weight {a_ml}; rises as you add labels)")
    thin = [c for c, n in sorted(counts.items()) if n < 3]
    if thin:
        print("Categories with <3 examples (add more labelled pages for accuracy):", ", ".join(thin))
    missing = [c for c in cats if c not in counts]
    if missing:
        print(f"{len(missing)} categories have no training pages yet (rules still cover them).")


def ml_weight(counts):
    """Trust the model more as labelled pages per category grow (thin data -> lean on rules)."""
    med = float(np.median(list(counts.values()))) if counts else 0.0
    return round(min(0.5, 0.1 + 0.06 * med), 2)


def load_bundle():
    if MODEL_PATH.exists():
        import joblib
        return joblib.load(MODEL_PATH)
    return None


# ============================================================================ sequence smoothing
def viterbi(emis, names, trans, lam):
    """emis: list of {category: prob} per page. Returns best category path using learned transitions."""
    K, idx = len(names), {n: i for i, n in enumerate(names)}
    C = np.zeros((K, K))
    for a, row in trans.items():
        for b, c in row.items():
            if a in idx and b in idx:
                C[idx[a], idx[b]] = c
    T = (C + 0.5) / (C.sum(axis=1, keepdims=True) + 0.5 * K)
    logT = lam * np.log(T)
    E = np.log(np.array([[e.get(n, 0.0) for n in names] for e in emis]) + 1e-4)
    score, back = E[0].copy(), []
    for t in range(1, len(emis)):
        cand = score[:, None] + logT
        back.append(cand.argmax(axis=0))
        score = cand.max(axis=0) + E[t]
    path = [int(score.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path.reverse()
    return [names[i] for i in path], T, idx


# ============================================================================ classify
def classify_pages(pages, cats, rules, bundle, a):
    pages = annotate(pages)
    model = bundle["model"] if bundle else None
    a_ml = bundle.get("ml_weight", 0.7) if bundle else 0.0
    n = len(pages)
    mprob = model.predict_proba(pages) if model is not None and n else None
    rows, emis, pos = [None] * n, [], []
    photo_ok = "Photograph" in cats

    for i, pg in enumerate(pages):
        text, words = pg["text"], pg["words"]
        base = dict(page=i + 1, ocr_conf=round(pg["conf"], 1), words=words)
        if photo_ok and getattr(a, "photo_rule", False) and words < 15 and pg["feats"][0] > 0.80 and (pg["feats"][2] + pg["feats"][3]) > 0.04:
            rows[i] = dict(base, category="Photograph", confidence=0.55, needs_review="Y", method="image_rule", top3="")
            continue
        if len(normalize(text)) < 25:
            rows[i] = dict(base, category=BLANK, confidence=0.9, needs_review="N", method="blank", top3="")
            continue

        rs = rules.score(text, pg["header"])
        tot = sum(rs.values())
        rp = {k: v / tot for k, v in rs.items()} if tot > 0 else {}
        if model is None:
            probs, conf, method = rs, rule_confidence(rs), "rules"
        else:
            mp = dict(zip(model.classes_, mprob[i]))
            names = set(cats) | set(mp)
            if tot > 0:
                # trust rules more when they fire strongly and clearly (guards against the model's bias to big classes)
                w = 1 - a_ml
                if bundle.get("gate", True):
                    w = max(w, min(bundle.get("gate_cap", 0.85), rule_confidence(rs)))
                probs = {k: (1 - w) * mp.get(k, 0) + w * rp.get(k, 0) for k in names}
                method = "model+rules"
            else:
                probs, method = {k: mp.get(k, 0) for k in names}, "model"
            conf = max(probs.values())
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[0][0] if ranked[0][1] > 0 else UNKNOWN
        # unreadable / non-English page (e.g. Telugu without the OCR pack)
        if getattr(a, "vernacular", False) and "Vernacular_Form" in cats and conf < 0.6 and len(re.findall(r"[A-Za-z]+", text)) > 40 and eng_score(text) < 0.04:
            best, conf, method = "Vernacular_Form", 0.6, "non_english"
        top3 = "; ".join(f"{k}:{v:.2f}" for k, v in ranked[:3] if v > 0)
        rows[i] = dict(base, category=best, confidence=round(conf, 3), needs_review="", method=method, top3=top3)
        emis.append(probs if model is not None else None)
        pos.append(i)

    # ---- page-order smoothing
    if bundle and bundle.get("transitions") and not a.no_seq and len(pos) > 1:
        names = sorted({k for e in emis for k in e})
        tot_norm = [{k: v / (sum(e.values()) or 1) for k, v in e.items()} for e in emis]
        path, T, idx = viterbi(tot_norm, names, bundle["transitions"], a.seq_weight)
        for j, (i, cat) in enumerate(zip(pos, path)):
            r = rows[i]
            if cat != r["category"] and r["method"] != "non_english":
                prev = idx.get(path[j - 1]) if j else None
                tp = float(T[prev, idx[cat]]) if prev is not None else 0.3
                r["confidence"] = round(0.7 * tot_norm[j].get(cat, 0) + 0.3 * tp, 3)
                r["category"], r["method"] = cat, r["method"] + "+seq"
    elif a.carry_forward:                                           # no trained transitions: simple fallback
        for j in range(1, n):
            r, prev = rows[j], rows[j - 1]
            if r["method"] in ("rules", "model", "model+rules") and r["confidence"] < 0.3 \
                    and prev["category"] not in (BLANK, UNKNOWN) and prev["confidence"] >= a.threshold:
                r["category"], r["method"], r["confidence"] = prev["category"], "carry_forward", round(prev["confidence"] * 0.8, 3)

    if getattr(a, "blank_label", "inherit") == "inherit":          # blank scan = back of previous page / filler in same document
        real = [i for i, r in enumerate(rows) if r["method"] not in ("blank",)]
        for i, r in enumerate(rows):
            if r["method"] == "blank" and real:
                src = max((j for j in real if j < i), default=None)
                src = src if src is not None else min(j for j in real if j > i)
                r["category"], r["method"], r["confidence"] = rows[src]["category"], "blank_inherit", 0.5

    for r in rows:
        if r["needs_review"] == "":
            low_ocr = r["words"] > 30 and r["ocr_conf"] < 45 and r["category"] != "Vernacular_Form"
            r["needs_review"] = "Y" if (r["confidence"] < a.threshold or low_ocr or r["category"] == UNKNOWN) else "N"
    return rows


def cmd_classify(a):
    cats, _ = load_cats(a.config)
    rules = Rules(cats)
    pages = get_pages(a.tiff, a.lang, a.workers)
    bundle = None if a.no_model else load_bundle()
    rows = classify_pages(pages, cats, rules, bundle, a)

    out = Path(a.out or Path(a.tiff).with_suffix(".classified.csv"))
    fields = ["page", "category", "confidence", "needs_review", "method", "ocr_conf", "words", "top3"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for r in rows:
        flag = "  <-- review" if r["needs_review"] == "Y" else ""
        print(f"p{r['page']:>3}  {r['category']:<30} {r['confidence']:.2f}  [{r['method']}]{flag}")
    nrev = sum(r["needs_review"] == "Y" for r in rows)
    print(f"\nSaved: {out} (+ .json) | pages to review: {nrev}/{len(rows)} | "
          f"model: {'yes' if bundle else 'no (rules only)'} | fuzzy matching: {'on' if fuzz else 'off'}")
    if a.split:
        split_tiff(a.tiff, rows, Path(a.split))


def split_tiff(tiff, rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r["page"] - 1)
    with Image.open(tiff) as im:
        for cat, idxs in groups.items():
            frames = []
            for i in idxs:
                im.seek(i)
                frames.append(im.copy())
            frames[0].save(outdir / f"{cat}.tif", save_all=True, append_images=frames[1:], compression="tiff_lzw")
    print(f"Split files written to {outdir}/")


# ============================================================================ eval
def cmd_eval(a):
    """Leave-one-file-out: train on all other files, classify the held-out file exactly like production."""
    from argparse import Namespace
    from sklearn.metrics import classification_report
    cats, canon = load_cats(a.config)
    rules = Rules(cats)
    files = load_files(a.tiff, a.labels, a.lang, a.workers, canon)
    variants = OrderedDict([("rules only", dict(model=False, seq=False)),
                            ("model+rules", dict(model=True, seq=False)),
                            ("model+rules+page order", dict(model=True, seq=True))])
    res = {v: [] for v in variants}
    per_file = defaultdict(dict)
    for k, held in enumerate(files):
        train_files = [f for j, f in enumerate(files) if j != k]
        bundle = None
        if train_files:
            X, y, _ = flatten(train_files)
            if len(set(y)) > 1:
                bundle = {"model": build_pipeline().fit(X, y), "transitions": build_transitions(train_files),
                          "ml_weight": ml_weight(Counter(y))}
        for v, o in variants.items():
            if o["model"] and bundle is None:
                continue
            args = Namespace(no_seq=not o["seq"], seq_weight=a.seq_weight, threshold=a.threshold,
                             carry_forward=False, blank_label="inherit", vernacular=False)
            rows = classify_pages(held["pages"], cats, rules, bundle if o["model"] else None, args)
            ok = 0
            for i, (r, lab) in enumerate(zip(rows, held["labels"]), 1):
                if not lab:
                    continue
                res[v].append((held["name"], i, lab, r["category"], r["confidence"], r["needs_review"]))
                ok += r["category"] == lab
            n = sum(1 for l in held["labels"] if l)
            per_file[held["name"]][v] = ok / max(n, 1)
    print(f"\nLeave-one-file-out evaluation on {sum(1 for f in files for l in f['labels'] if l)} labelled pages "
          f"({len(files)} files)\n")
    print(f"{'file':<22}" + "".join(f"{v:>26}" for v in variants))
    for name, d in per_file.items():
        print(f"{name:<22}" + "".join(f"{d.get(v, float('nan')):>26.1%}" for v in variants))
    print(f"{'ALL PAGES':<22}" + "".join(
        f"{(np.mean([r[2] == r[3] for r in res[v]]) if res[v] else float('nan')):>26.1%}" for v in variants))
    final = res["model+rules+page order"] or res["rules only"]
    y_t, y_p = [r[2] for r in final], [r[3] for r in final]
    print("\nPer-category report (final variant):")
    print(classification_report(y_t, y_p, zero_division=0))
    conf = Counter((t, p) for t, p in zip(y_t, y_p) if t != p)
    print("Most common confusions (true -> predicted):")
    for (t, p), c in conf.most_common(12):
        print(f"  {c:3d}  {t} -> {p}")
    print("\nReview-threshold calibration (page confidence below threshold -> needs_review):")
    print(f"  {'threshold':>9} {'flagged':>8} {'acc. of unflagged':>18} {'errors caught':>14}")
    nerr = sum(r[2] != r[3] for r in final)
    for th in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        fl = [r for r in final if r[4] < th]
        un = [r for r in final if r[4] >= th]
        caught = sum(r[2] != r[3] for r in fl)
        print(f"  {th:>9.2f} {len(fl) / len(final):>8.0%} {np.mean([r[2] == r[3] for r in un]) if un else float('nan'):>18.1%} "
              f"{caught}/{nerr}".rjust(14))
    revs = [r for r in final if r[5] == "Y"]
    if revs:
        print(f"\nFlagged for review: {len(revs)}/{len(final)} pages; accuracy on NOT flagged pages: "
              f"{np.mean([r[2] == r[3] for r in final if r[5] == 'N']):.1%}")
    wrong = [r for r in final if r[2] != r[3]]
    if a.show_errors:
        print("\nMisclassified pages:")
        for f, p, t, pr, cf, _ in wrong:
            print(f"  {f} p{p}: true={t}  predicted={pr} ({cf:.2f})")


# ============================================================================ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    default_cfg = str(HERE / "categories.yaml")

    c = sub.add_parser("classify")
    c.add_argument("tiff")
    c.add_argument("--config", default=default_cfg)
    c.add_argument("--out")
    c.add_argument("--split", help="folder: write one multi-page TIFF per category")
    c.add_argument("--threshold", type=float, default=0.35, help="below this -> needs_review=Y")
    c.add_argument("--carry-forward", action="store_true", help="(no trained model) weak pages inherit previous category")
    c.add_argument("--no-model", action="store_true", help="ignore trained model, rules only")
    c.add_argument("--no-seq", action="store_true", help="disable page-order smoothing")
    c.add_argument("--seq-weight", type=float, default=0.4, help="strength of page-order smoothing")
    c.add_argument("--blank-label", default="inherit", help="'inherit' = blank pages take the neighbouring page's category; 'blank_page' keeps them separate")
    c.add_argument("--photo-rule", action="store_true", help="enable image-based Photograph detection (off by default; better: label Photograph pages and train)")
    c.add_argument("--vernacular", action="store_true", help="label non-English pages Vernacular_Form")
    c.add_argument("--lang", default="eng", help="Tesseract langs, e.g. eng+tel")
    c.add_argument("--workers", type=int, help="OCR processes (default: min(4, cores/2)); 1 = no multiprocessing")
    c.set_defaults(fn=cmd_classify)

    for name, fn in (("train", cmd_train), ("eval", cmd_eval)):
        t = sub.add_parser(name)
        t.add_argument("--tiff", nargs="+", required=True)
        t.add_argument("--labels", nargs="+", required=True, help="CSV(s) with columns page,category, or one Excel (column per file, header = file name)")
        t.add_argument("--config", default=default_cfg)
        t.add_argument("--lang", default="eng")
        t.add_argument("--workers", type=int)
        t.add_argument("--threshold", type=float, default=0.35)
        t.add_argument("--seq-weight", type=float, default=0.4)
        t.add_argument("--show-errors", action="store_true", help="eval: list every misclassified page")
        t.set_defaults(fn=fn)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
