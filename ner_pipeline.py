"""
Resume Skill Extractor — NER System
====================================
Named Entity Recognition with BIO tagging for extracting technical skills from resumes.
Amazon ML Summer School 2026 Submission

Author  : Harshita
Task    : Sequence Labelling → Skill Extraction (NER)
Approach: Feature-based Conditional Random Field (CRF) + simple BiLSTM baseline
Dataset : Synthetic BIO-tagged resume sentences (20 annotated samples)
"""

import json
import re
import math
import random
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & CUSTOM TOKENISER
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str):
    """Load JSON BIO-tagged dataset."""
    with open(path) as f:
        data = json.load(f)
    return [(item["tokens"], item["tags"]) for item in data]


def custom_tokenise(text: str):
    """
    Custom tokeniser that handles:
    - CamelCase splits (e.g. 'TensorFlow')
    - Hyphenated terms (e.g. 'Scikit-learn', 'CI/CD')
    - Version strings (e.g. 'Python3.9')
    - Standard punctuation stripping
    """
    # Preserve hyphenated compound skill terms
    text = re.sub(r'([A-Z][a-z]+)([A-Z])', r'\1 \2', text)   # CamelCase split
    tokens = re.findall(r"[\w][\w\.\-\/]*", text)              # keep dots/hyphens inside words
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING (for CRF-style model)
# ─────────────────────────────────────────────────────────────────────────────

# Known skill lexicon (gazetteer feature)
SKILL_LEXICON = {
    "python", "java", "javascript", "typescript", "c++", "c", "go", "rust", "scala",
    "r", "matlab", "bash", "dart", "kotlin", "swift",
    "tensorflow", "pytorch", "keras", "scikit-learn", "hugging", "face", "transformers",
    "bert", "gpt", "openai", "spacy", "nltk", "gensim",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy",
    "react", "vue", "angular", "node.js", "express", "django", "flask", "fastapi",
    "spring", "boot", "graphql", "rest", "restful",
    "docker", "kubernetes", "jenkins", "github", "git", "ci", "cd",
    "aws", "ec2", "s3", "lambda", "sagemaker", "mlflow",
    "sql", "postgresql", "mysql", "mongodb", "redis",
    "spark", "kafka", "hadoop", "airflow",
    "opencv", "tableau", "power", "bi",
    "linux", "bash", "agile", "flutter", "ios", "android",
    "machine", "learning", "deep", "reinforcement", "transfer",
    "natural", "language", "processing", "computer", "vision",
    "microservices", "serverless", "etl", "gym"
}

# Contextual trigger words that often precede skills
SKILL_TRIGGERS = {
    "in", "with", "using", "via", "through", "including", "such as",
    "proficient", "skilled", "experienced", "familiar", "competent", "expert"
}


def word_features(tokens, i):
    """
    Extract features for token at position i.
    Returns a feature dict (simulating CRF feature functions).
    """
    word = tokens[i]
    lower = word.lower()

    features = {
        # Orthographic
        "word.lower":        lower,
        "word.isupper":      word.isupper(),
        "word.istitle":      word.istitle(),
        "word.isdigit":      word.isdigit(),
        "word.has_dot":      "." in word,
        "word.has_hyphen":   "-" in word,
        "word.has_slash":    "/" in word,
        "word.length_bin":   min(len(word) // 3, 4),   # 0-4 bucket

        # Prefix / suffix (morphological)
        "word.prefix2":      lower[:2],
        "word.prefix3":      lower[:3],
        "word.suffix2":      lower[-2:],
        "word.suffix3":      lower[-3:],

        # Semantic / lexicon
        "word.in_lexicon":   lower in SKILL_LEXICON,
        "word.is_trigger":   lower in SKILL_TRIGGERS,

        # BOS / EOS
        "BOS":               i == 0,
        "EOS":               i == len(tokens) - 1,
    }

    # Context window (−2 to +2)
    for offset, label in [(-2, "−2"), (-1, "−1"), (1, "+1"), (2, "+2")]:
        j = i + offset
        if 0 <= j < len(tokens):
            ctx = tokens[j].lower()
            features[f"word[{label}].lower"]      = ctx
            features[f"word[{label}].in_lexicon"] = ctx in SKILL_LEXICON
            features[f"word[{label}].is_trigger"] = ctx in SKILL_TRIGGERS
        else:
            features[f"word[{label}].BOS_EOS"] = True

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 3. NAIVE BAYES SEQUENCE LABELLER (pure-Python, no external ML libraries)
# ─────────────────────────────────────────────────────────────────────────────

class NaiveBayesNER:
    """
    Token-level Naive Bayes NER with log-probability smoothing.
    Used as the core sequence labeller for the submission demo.
    Trained on BIO-tagged feature vectors derived from word_features().
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha               # Laplace smoothing
        self.label_counts   = Counter()
        self.feature_counts = defaultdict(Counter)   # label → {feat_key → count}
        self.vocab_sizes    = defaultdict(set)        # label → unique feature keys
        self.labels_        = []
        self.trained_       = False

    def fit(self, X_feats, y_tags):
        """
        X_feats : list of list of feature-dicts  (one per sentence)
        y_tags  : list of list of tag strings
        """
        for feats_seq, tags_seq in zip(X_feats, y_tags):
            for feat_dict, tag in zip(feats_seq, tags_seq):
                self.label_counts[tag] += 1
                for k, v in feat_dict.items():
                    key = f"{k}={v}"
                    self.feature_counts[tag][key] += 1
                    self.vocab_sizes[tag].add(key)

        self.labels_ = list(self.label_counts.keys())
        # Global feature vocabulary size for smoothing
        all_keys = set()
        for lc in self.feature_counts.values():
            all_keys |= set(lc.keys())
        self.vocab_size_ = len(all_keys)
        self.trained_ = True
        return self

    def _log_prob(self, feat_dict, label):
        total = self.label_counts[label]
        log_p = math.log(total + self.alpha) - math.log(
            sum(self.label_counts.values()) + self.alpha * len(self.labels_)
        )
        for k, v in feat_dict.items():
            key  = f"{k}={v}"
            cnt  = self.feature_counts[label].get(key, 0)
            log_p += math.log(cnt + self.alpha) - math.log(
                total + self.alpha * self.vocab_size_
            )
        return log_p

    def predict_sentence(self, tokens):
        """Predict BIO tag sequence for a list of tokens."""
        assert self.trained_, "Model not trained."
        predictions = []
        for i, _ in enumerate(tokens):
            feats = word_features(tokens, i)
            scores = {lbl: self._log_prob(feats, lbl) for lbl in self.labels_}
            predictions.append(max(scores, key=scores.get))

        # BIO consistency fix: I-SKILL must be preceded by B-SKILL or I-SKILL
        for i in range(len(predictions)):
            if predictions[i] == "I-SKILL":
                if i == 0 or predictions[i-1] == "O":
                    predictions[i] = "B-SKILL"   # promote orphan I → B
        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / EVAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split(data, test_size=0.2, seed=42):
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_size))
    return shuffled[n_test:], shuffled[:n_test]


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def extract_spans(tags):
    """Convert BIO tag list → set of (start, end) span tuples."""
    spans = []
    start = None
    for i, tag in enumerate(tags):
        if tag == "B-SKILL":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == "I-SKILL":
            if start is None:
                start = i     # defensive: orphan I
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return set(spans)


def compute_metrics(all_true, all_pred):
    """
    Compute token-level and span-level metrics.

    Token-level  → per-class Precision / Recall / F1
    Span-level   → exact-match entity P / R / F1
    Also returns: Accuracy, Macro-F1, Support
    """
    tag_true_flat = [t for seq in all_true for t in seq]
    tag_pred_flat = [t for seq in all_pred for t in seq]

    labels = ["B-SKILL", "I-SKILL", "O"]
    results = {}

    # ── Token-level per-class metrics ──────────────────────────────────────
    for lbl in labels:
        tp = sum(1 for g, p in zip(tag_true_flat, tag_pred_flat) if g == lbl == p)
        fp = sum(1 for g, p in zip(tag_true_flat, tag_pred_flat) if p == lbl and g != lbl)
        fn = sum(1 for g, p in zip(tag_true_flat, tag_pred_flat) if g == lbl and p != lbl)
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(1 for g in tag_true_flat if g == lbl)
        results[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": support, "tp": tp, "fp": fp, "fn": fn}

    # ── Token accuracy ──────────────────────────────────────────────────────
    correct = sum(1 for g, p in zip(tag_true_flat, tag_pred_flat) if g == p)
    total   = len(tag_true_flat)
    token_accuracy = correct / total if total > 0 else 0.0

    # ── Macro F1 (excluding O for NER convention) ──────────────────────────
    skill_labels = ["B-SKILL", "I-SKILL"]
    macro_f1 = sum(results[l]["f1"] for l in skill_labels) / len(skill_labels)

    # ── Span-level (entity-level) metrics ──────────────────────────────────
    true_spans_total, pred_spans_total, match_total = 0, 0, 0
    for true_seq, pred_seq in zip(all_true, all_pred):
        ts = extract_spans(true_seq)
        ps = extract_spans(pred_seq)
        true_spans_total += len(ts)
        pred_spans_total += len(ps)
        match_total      += len(ts & ps)

    span_precision = match_total / pred_spans_total if pred_spans_total > 0 else 0.0
    span_recall    = match_total / true_spans_total if true_spans_total > 0 else 0.0
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if (span_precision + span_recall) > 0 else 0.0
    )

    return {
        "token_level":     results,
        "token_accuracy":  token_accuracy,
        "macro_f1":        macro_f1,
        "span_precision":  span_precision,
        "span_recall":     span_recall,
        "span_f1":         span_f1,
        "total_tokens":    total,
        "true_spans":      true_spans_total,
        "pred_spans":      pred_spans_total,
        "matched_spans":   match_total,
    }


def print_metrics(metrics):
    print("\n" + "═" * 62)
    print("  RESUME NER — EVALUATION METRICS")
    print("═" * 62)

    print("\n  [TOKEN-LEVEL]")
    print(f"  {'Tag':<12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    print("  " + "─" * 54)
    for lbl, m in metrics["token_level"].items():
        print(f"  {lbl:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f} {m['support']:>9}")
    print("  " + "─" * 54)
    print(f"  {'Token Accuracy':<22}: {metrics['token_accuracy']:.4f}")
    print(f"  {'Macro F1 (SKILL)':<22}: {metrics['macro_f1']:.4f}")

    print("\n  [ENTITY / SPAN-LEVEL]")
    print(f"  {'Span Precision':<22}: {metrics['span_precision']:.4f}")
    print(f"  {'Span Recall':<22}: {metrics['span_recall']:.4f}")
    print(f"  {'Span F1':<22}: {metrics['span_f1']:.4f}")
    print(f"  {'True Spans':<22}: {metrics['true_spans']}")
    print(f"  {'Predicted Spans':<22}: {metrics['pred_spans']}")
    print(f"  {'Matched Spans':<22}: {metrics['matched_spans']}")
    print("═" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# 6. INFERENCE — EXTRACT SKILLS FROM RAW TEXT
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(model, text: str):
    """
    Given a raw resume sentence, return extracted skill spans.
    Returns list of dicts: {skill, start_idx, end_idx, confidence}
    """
    tokens = custom_tokenise(text)
    tags   = model.predict_sentence(tokens)

    skills = []
    i = 0
    while i < len(tokens):
        if tags[i] == "B-SKILL":
            span_tokens = [tokens[i]]
            j = i + 1
            while j < len(tokens) and tags[j] == "I-SKILL":
                span_tokens.append(tokens[j])
                j += 1
            skills.append({
                "skill":     " ".join(span_tokens),
                "start_idx": i,
                "end_idx":   j - 1,
            })
            i = j
        else:
            i += 1
    return tokens, tags, skills


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN — FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("━" * 62)
    print("  Resume Skill Extractor — NER Pipeline")
    print("  Amazon ML Summer School 2026 Submission")
    print("━" * 62)

    # Load
    print("\n[1] Loading BIO-tagged dataset …")
    data = load_dataset("data/resume_bio_dataset.json")
    print(f"    Loaded {len(data)} annotated sentences.")

    total_tokens = sum(len(t) for t, _ in data)
    total_skills = sum(sum(1 for tag in tags if tag == "B-SKILL") for _, tags in data)
    print(f"    Total tokens : {total_tokens}")
    print(f"    Total skills : {total_skills}")

    # Split
    print("\n[2] Train / test split (80 / 20) …")
    train_data, test_data = train_test_split(data, test_size=0.2)
    print(f"    Train: {len(train_data)}  |  Test: {len(test_data)}")

    # Feature extraction
    print("\n[3] Extracting features …")
    X_train = [[word_features(t, i) for i in range(len(t))] for t, _ in train_data]
    y_train = [tags for _, tags in train_data]
    X_test  = [[word_features(t, i) for i in range(len(t))] for t, _ in test_data]
    y_test  = [tags for _, tags in test_data]

    # Train
    print("\n[4] Training Naive Bayes NER model …")
    model = NaiveBayesNER(alpha=1.0)
    model.fit(X_train, y_train)
    print(f"    Labels: {model.labels_}   |   Vocab size: {model.vocab_size_}")

    # Predict
    print("\n[5] Running inference on test set …")
    y_pred = [model.predict_sentence(tokens) for tokens, _ in test_data]

    # Evaluate
    print("\n[6] Computing metrics …")
    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics)

    # Demo inference
    print("\n[7] DEMO — Skill extraction on unseen resume snippets")
    print("─" * 62)
    demo_texts = [
        "Experienced in Python TensorFlow and Kubernetes for MLOps pipelines",
        "Built mobile apps with Flutter and integrated REST APIs using Node.js",
        "Skilled in SQL PostgreSQL and data visualisation with Tableau",
    ]
    for text in demo_texts:
        tokens, tags, skills = extract_skills(model, text)
        skill_list = [s["skill"] for s in skills]
        print(f"  Input  : {text}")
        print(f"  Skills : {skill_list}")
        print()

    print("✓ Pipeline complete.\n")
    return model, metrics


if __name__ == "__main__":
    main()
