"""
vsm_similarity.py
=================
Document similarity with a Vector Space Model (VSM).

Every document is turned into a vector of TF-IDF weights, and the similarity
between two documents is the cosine of the angle between their vectors.

Pipeline
--------
1. Load documents (.txt / .pdf / .docx) from a folder
2. Pre-process: lowercase -> tokenize -> remove stop-words
3. Build the vocabulary and the term-frequency (TF) matrix
4. Weight with TF-IDF  (each document becomes one vector)
5. Cosine similarity between every pair of document vectors
6. Show / save results: matrix, ranked pairs, top terms, heatmap
7. Verify the from-scratch maths against scikit-learn
8. (Optional) rank the documents against a free-text query

Usage
-----
    python vsm_similarity.py
    python vsm_similarity.py --docs documents --top-terms 8
    python vsm_similarity.py --query "neural networks for image recognition"
"""

import argparse
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# Everything printed is also collected here so it can be saved to results.txt
LOG_LINES = []


def log(message=""):
    print(message)
    LOG_LINES.append(str(message))


# --------------------------------------------------------------------------
# 1. LOAD DOCUMENTS
# --------------------------------------------------------------------------
def read_document(path):
    """Return the plain text of a .txt, .pdf or .docx file."""
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        from docx import Document

        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_documents(folder):
    """Read every supported file in `folder`. Returns (names, texts)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder '{folder}' not found. Run download_documents.py first "
                         "or put your own .txt/.pdf/.docx files there.")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 documents in '{folder}', found {len(files)}.")

    names = [p.stem for p in files]
    if len(set(names)) < len(names):        # same name with different extensions
        names = [p.name for p in files]
    texts = [read_document(p) for p in files]
    return names, texts


# --------------------------------------------------------------------------
# 2. PRE-PROCESSING
# --------------------------------------------------------------------------
def preprocess(text):
    """Lowercase, keep alphabetic words only, drop stop-words and 1-2 letter tokens."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]


# --------------------------------------------------------------------------
# 3-4. VECTOR SPACE MODEL: TF, IDF, TF-IDF
# --------------------------------------------------------------------------
def build_vocabulary(token_lists):
    """Sorted list of unique terms + a term -> column-index lookup."""
    vocab = sorted({tok for tokens in token_lists for tok in tokens})
    return vocab, {tok: i for i, tok in enumerate(vocab)}


def term_frequency_matrix(token_lists, index):
    """tf[d, t] = how many times term t appears in document d (raw counts)."""
    tf = np.zeros((len(token_lists), len(index)))
    for d, tokens in enumerate(token_lists):
        for tok, count in Counter(tokens).items():
            tf[d, index[tok]] = count
    return tf


def inverse_document_frequency(tf):
    """
    Smoothed IDF:  idf(t) = ln((1 + N) / (1 + df(t))) + 1
    N = number of documents, df(t) = number of documents that contain t.
    Rare terms get a high weight, terms found in every document get the lowest.
    """
    n_docs = tf.shape[0]
    doc_freq = (tf > 0).sum(axis=0)
    return np.log((1 + n_docs) / (1 + doc_freq)) + 1


def tfidf_matrix(tf, idf):
    """TF-IDF weight = term frequency * inverse document frequency."""
    return tf * idf


# --------------------------------------------------------------------------
# 5. COSINE SIMILARITY
# --------------------------------------------------------------------------
def cosine(a, b):
    """cos(a, b) = (a . b) / (|a| * |b|)   ->   0 = nothing in common, 1 = same direction."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def similarity_matrix(X):
    """Cosine similarity between every pair of rows of X."""
    n = X.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine(X[i], X[j])
    return sim


def mean_off_diagonal(sim):
    """Average similarity of different documents (ignores the diagonal of 1.0s)."""
    n = sim.shape[0]
    return (sim.sum() - np.trace(sim)) / (n * (n - 1))


# --------------------------------------------------------------------------
# 6. RESULTS
# --------------------------------------------------------------------------
def ranked_pairs(names, sim):
    """All document pairs sorted from most to least similar."""
    pairs = [(names[i], names[j], sim[i, j]) for i, j in combinations(range(len(names)), 2)]
    return sorted(pairs, key=lambda p: p[2], reverse=True)


def top_terms(row, vocab, k):
    """The k terms with the highest TF-IDF weight in one document vector."""
    best = np.argsort(row)[::-1][:k]
    return [vocab[i] for i in best if row[i] > 0]


def save_heatmap(sim, names, path, show=False):
    import matplotlib.pyplot as plt

    n = len(names)
    # Scale the colours to the highest similarity between *different* documents,
    # otherwise the diagonal (always 1.0) washes out the interesting differences.
    vmax = max(sim[~np.eye(n, dtype=bool)].max(), 0.1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(sim, cmap="YlGnBu", vmin=0, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            colour = "white" if sim[i, j] > 0.5 * vmax else "black"
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", color=colour, fontsize=8)
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    ax.set_title(f"Document similarity (TF-IDF + cosine)\ncolour scale 0 to {vmax:.2f}; diagonal = 1.00")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. VERIFY AGAINST SCIKIT-LEARN
# --------------------------------------------------------------------------
def max_difference_vs_sklearn(token_lists, my_sim):
    """
    Build the same TF-IDF matrix with scikit-learn (feeding it our own tokens)
    and return the largest absolute difference between the two similarity matrices.
    """
    vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens)   # defaults: smooth idf, L2 norm
    sk_tfidf = vectorizer.fit_transform(token_lists)
    return float(np.abs(sklearn_cosine(sk_tfidf) - my_sim).max())


# --------------------------------------------------------------------------
# 8. QUERY: a query is just one more (very short) document vector
# --------------------------------------------------------------------------
def rank_by_query(query, names, tfidf, index, idf):
    counts = Counter(t for t in preprocess(query) if t in index)   # unseen words are ignored
    if not counts:
        return None
    q = np.zeros(len(index))
    for tok, c in counts.items():
        q[index[tok]] = c
    q = q * idf
    scores = [(name, cosine(q, row)) for name, row in zip(names, tfidf)]
    return sorted(scores, key=lambda s: s[1], reverse=True)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Document similarity with a TF-IDF vector space model.")
    p.add_argument("--docs", default="documents", help="folder with .txt/.pdf/.docx files")
    p.add_argument("--out", default="outputs", help="folder for results (csv, png, txt)")
    p.add_argument("--top-terms", type=int, default=8, help="keywords to show per document")
    p.add_argument("--query", default=None, help="optional text query to rank documents against")
    p.add_argument("--show", action="store_true", help="also open the heatmap in a window")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")           # save the figure without needing a display

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. load + preprocess
    names, texts = load_documents(args.docs)
    token_lists = [preprocess(t) for t in texts]
    ids = [f"D{i + 1}" for i in range(len(names))]

    log("=" * 78)
    log("1) DOCUMENTS LOADED AND PRE-PROCESSED")
    log("=" * 78)
    overview = pd.DataFrame({
        "ID": ids,
        "Document": names,
        "Words (raw)": [len(t.split()) for t in texts],
        "Tokens (clean)": [len(t) for t in token_lists],
    })
    log(overview.to_string(index=False))

    # 3-4. vector space model
    vocab, index = build_vocabulary(token_lists)
    tf = term_frequency_matrix(token_lists, index)
    idf = inverse_document_frequency(tf)
    tfidf = tfidf_matrix(tf, idf)

    log("")
    log("=" * 78)
    log("2) VECTOR SPACE MODEL")
    log("=" * 78)
    log(f"Vocabulary size          : {len(vocab)} unique terms")
    log(f"TF-IDF matrix shape      : {tfidf.shape[0]} documents x {tfidf.shape[1]} terms")

    # 5. similarity
    sim = similarity_matrix(tfidf)
    labelled_rows = [f"{i} {n}" for i, n in zip(ids, names)]
    matrix_df = pd.DataFrame(sim, index=labelled_rows, columns=ids)

    log("")
    log("=" * 78)
    log("3) COSINE SIMILARITY MATRIX (1.000 = identical direction, 0 = no shared terms)")
    log("=" * 78)
    log(matrix_df.round(3).to_string())

    pairs = ranked_pairs(names, sim)
    log("")
    log("Top 5 most similar pairs")
    for a, b, s in pairs[:5]:
        log(f"  {s:.3f}   {a}  <->  {b}")
    log("Top 3 least similar pairs")
    for a, b, s in pairs[-3:]:
        log(f"  {s:.3f}   {a}  <->  {b}")

    log("")
    log("=" * 78)
    log(f"4) TOP {args.top_terms} TF-IDF TERMS PER DOCUMENT")
    log("=" * 78)
    for name, row in zip(names, tfidf):
        log(f"{name}:")
        log("    " + ", ".join(top_terms(row, vocab, args.top_terms)))

    # 7. verification + effect of idf
    diff = max_difference_vs_sklearn(token_lists, sim)
    sim_tf_only = similarity_matrix(tf)
    log("")
    log("=" * 78)
    log("5) CHECKS")
    log("=" * 78)
    log(f"Max difference vs scikit-learn : {diff:.2e}   ({'PASS' if diff < 1e-9 else 'CHECK CODE'})")
    log(f"Mean similarity, TF only       : {mean_off_diagonal(sim_tf_only):.3f}")
    log(f"Mean similarity, TF-IDF        : {mean_off_diagonal(sim):.3f}")
    log("(IDF down-weights words that appear in many documents, so unrelated documents "
        "look less alike.)")

    # 8. optional query
    if args.query:
        log("")
        log("=" * 78)
        log(f'6) QUERY: "{args.query}"')
        log("=" * 78)
        ranking = rank_by_query(args.query, names, tfidf, index, idf)
        if ranking is None:
            log("None of the query words appear in the document vocabulary.")
        else:
            for rank, (name, score) in enumerate(ranking, start=1):
                log(f"  {rank}. {score:.3f}   {name}")

    # save outputs
    matrix_df_full = pd.DataFrame(sim, index=names, columns=names)
    matrix_df_full.to_csv(out_dir / "similarity_matrix.csv", float_format="%.4f")
    pd.DataFrame(pairs, columns=["document_1", "document_2", "cosine_similarity"]).to_csv(
        out_dir / "ranked_pairs.csv", index=False, float_format="%.4f")
    save_heatmap(sim, names, out_dir / "similarity_heatmap.png", show=args.show)
    (out_dir / "results.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")

    log("")
    log(f"Saved to '{out_dir}': similarity_matrix.csv, ranked_pairs.csv, "
        "similarity_heatmap.png, results.txt")


if __name__ == "__main__":
    main()
