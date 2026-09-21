# Document Similarity using a Vector Space Model

**Name:** <<your name>>  **Course/Week:** Week 3 Assignment  **Date:** <<date>>
**GitHub repository:** <<paste your repository link here>>

> How to use this template: run the code first, then replace every `<<...>>` and every
> `[SCREENSHOT: ...]` with your own text and screenshots. All numbers and observations must
> come from *your* run. Delete this note and the italic hints when you are done.

---

## 1. Introduction

The goal of this assignment is to measure how similar a set of documents are. I used a
**vector space model (VSM)**, in which every document is represented as a vector of term
weights, and similarity is computed as the cosine of the angle between two vectors. Documents
that use many of the same distinctive words point in similar directions and get a high score.

## 2. Dataset

I collected <<number>> publicly available documents (<<Wikipedia articles / news articles / ...>>)
and stored them as <<.txt / .pdf / .docx>> files in the `documents/` folder.

| ID | Document | Source (URL) | Words |
|----|----------|--------------|-------|
| D1 | <<name>> | <<url>> | <<count>> |
| D2 | | | |

*Hint: the IDs, names and word counts are in the first table printed by the program
(and `documents/SOURCES.md` has the URLs if you used `download_documents.py`).*

[SCREENSHOT: the `documents/` folder in GitHub or your file explorer]

## 3. Methodology

**3.1 Pre-processing.** Each document is lowercased, split into alphabetic words, and stripped of
English stop-words (e.g. "the", "and") and very short tokens. This keeps only words that carry meaning.

**3.2 Term frequency (TF).** `tf(t, d)` is the number of times term *t* appears in document *d*.
Stacking these counts gives a documents x terms matrix.

**3.3 Inverse document frequency (IDF).** Words that appear in many documents say little about any one
of them, so they are down-weighted:

    idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1

where N is the number of documents and df(t) is the number of documents containing t.

**3.4 TF-IDF weighting.** `tfidf(t, d) = tf(t, d) x idf(t)`. Each document is now a vector in a space
with one dimension per vocabulary term.

**3.5 Cosine similarity.** The similarity of documents A and B is

    cos(A, B) = (A . B) / ( |A| x |B| )

The value is between 0 (no shared terms) and 1 (same direction). Cosine similarity ignores document
length, so a long article and a short one can still be compared fairly.

## 4. Implementation

The code is in `vsm_similarity.py` (roughly 340 lines including comments) and is written from scratch with NumPy. The main
functions are:

| Step | Function | What it does |
|------|----------|--------------|
| Load | `load_documents`, `read_document` | reads .txt, .pdf and .docx files |
| Pre-process | `preprocess` | lowercase, tokenize, remove stop-words |
| Vocabulary + TF | `build_vocabulary`, `term_frequency_matrix` | counts words per document |
| IDF + TF-IDF | `inverse_document_frequency`, `tfidf_matrix` | weights the counts |
| Similarity | `cosine`, `similarity_matrix` | cosine of every pair of vectors |
| Verification | `max_difference_vs_sklearn` | compares my result with scikit-learn |

[SCREENSHOT: `preprocess` and `term_frequency_matrix` in your editor]
*Write 2-3 sentences explaining what these lines do.*

[SCREENSHOT: `inverse_document_frequency` and `tfidf_matrix`]
*Explain why IDF is needed.*

[SCREENSHOT: `cosine` and `similarity_matrix`]
*Explain how the formula in 3.5 maps to the code.*

## 5. Results

### 5.1 Pre-processing and vector space

[SCREENSHOT: console sections 1 and 2 (document table, vocabulary size, matrix shape)]
*State how many unique terms the vocabulary has and how pre-processing changed the word counts.*

### 5.2 Similarity matrix

[SCREENSHOT: console section 3, the cosine similarity matrix]
[SCREENSHOT: `outputs/similarity_heatmap.png`]
*Describe the pattern: which blocks of documents are similar to each other? Are the diagonal values 1.0
as expected?*

### 5.3 Most and least similar pairs

[SCREENSHOT: top 5 / bottom 3 pairs from the console]
*Give the highest-scoring pair with its score and explain why they are similar, using the keywords in 5.4.*

### 5.4 Top TF-IDF terms per document

[SCREENSHOT: console section 4]
*Do the keywords describe each document well? Point out shared keywords for the most similar pair.*

### 5.5 Verification and effect of IDF

[SCREENSHOT: console section 5]
*Report the difference vs scikit-learn (should be around 1e-15, i.e. identical). Compare the mean
similarity with TF only and with TF-IDF and explain the change.*

### 5.6 Query example (optional)

[SCREENSHOT: output of `python vsm_similarity.py --query "<<your query>>"`]
*Which document ranks first, and does that make sense?*

## 6. Discussion

- **Did the results match expectations?** <<compare with what you expected from the topics>>
- **Surprises:** <<any pair that scored higher or lower than expected, and why>>
- **Limitations of the model:**
  - It is a bag-of-words model, so word order and meaning are ignored ("dog bites man" = "man bites dog").
  - Synonyms are not recognised ("car" and "automobile" count as different terms).
  - No stemming, so "learn" and "learning" are separate terms.
  - Results depend on the corpus, because IDF is computed from these documents only.
- **Possible improvements:** stemming/lemmatization, bigrams, BM25 weighting, or sentence embeddings.

## 7. Conclusion

<<2-4 sentences: what you built, the main finding (e.g. which documents are most similar and why), and
what the VSM does well or poorly.>>

## 8. Code

GitHub repository: <<link>>
