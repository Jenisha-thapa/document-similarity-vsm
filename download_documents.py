"""
download_documents.py
=====================
Downloads a small corpus of publicly available articles (plain text) from
Wikipedia into the `documents/` folder, ready for vsm_similarity.py.

The corpus has three topic groups, so the similarity results are easy to check
by eye: AI/ML, sports, and climate/energy.

Wikipedia text is licensed CC BY-SA 4.0. The script writes documents/SOURCES.md
with the URL and retrieval date of every article (attribution).

Want different documents? Edit TOPICS below (any English Wikipedia title), or
skip this script and drop your own .txt / .pdf / .docx files into documents/.

Usage:
    python download_documents.py
"""

import datetime
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia asks API clients to identify themselves - put your own e-mail here.
HEADERS = {"User-Agent": "VSMSimilarityAssignment/1.0 (student coursework; contact: your_email@example.com)"}

TOPICS = [
    # group 1: artificial intelligence
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    # group 2: sports
    "Cricket",
    "Association football",
    "Basketball",
    # group 3: climate and energy
    "Climate change",
    "Renewable energy",
    "Solar power",
]

# Sections at the end of an article that are lists of links/citations, not prose.
TRAILING_SECTIONS = ["See also", "Notes", "Footnotes", "References", "Citations",
                     "Further reading", "External links", "Bibliography"]


def strip_math(text):
    """Remove '{\\displaystyle ...}' formula blocks (they contain nested braces)."""
    out, i = [], 0
    while i < len(text):
        if text.startswith("{\\displaystyle", i):
            depth = 0
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def clean_extract(text):
    """Turn Wikipedia's plain-text extract into tidy article text."""
    text = strip_math(text)
    # cut everything from the first trailing section (See also / References / ...) onwards
    trailing = re.compile(r"^=+\s*(?:%s)\s*=+\s*$" % "|".join(TRAILING_SECTIONS), re.M | re.I)
    match = trailing.search(text)
    if match:
        text = text[:match.start()]
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)   # '== Heading ==' -> 'Heading'
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_article(title):
    """Return (canonical_title, cleaned_text) for one English Wikipedia article."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts",
        "explaintext": 1,      # plain text instead of HTML
        "redirects": 1,        # follow redirects, e.g. "Football" -> "Association football"
        "titles": title,
    }
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    page = response.json()["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        raise ValueError("no article text returned")
    return page["title"], clean_extract(page["extract"])


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def write_sources_file(folder, rows):
    today = datetime.date.today().isoformat()
    lines = [
        "# Document sources",
        "",
        "All documents are plain-text copies of English Wikipedia articles, "
        f"retrieved on {today}. Wikipedia text is available under the "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) licence.",
        "",
        "| File | Article | Words |",
        "|------|---------|-------|",
    ]
    for filename, title, url, words in rows:
        lines.append(f"| {filename} | [{title}]({url}) | {words:,} |")
    (folder / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    folder = Path("documents")
    folder.mkdir(exist_ok=True)

    rows = []
    print(f"Downloading {len(TOPICS)} articles from Wikipedia...")
    for title in TOPICS:
        try:
            real_title, text = fetch_article(title)
        except (requests.RequestException, ValueError, KeyError) as error:
            print(f"  FAILED  {title}: {error}")
            continue
        path = folder / f"{slugify(real_title)}.txt"
        path.write_text(text, encoding="utf-8")
        words = len(text.split())
        url = "https://en.wikipedia.org/wiki/" + quote(real_title.replace(" ", "_"))
        rows.append((path.name, real_title, url, words))
        print(f"  saved   {path.name:<32} {words:>7,} words")
        time.sleep(1)                          # be polite to the API

    if not rows:
        sys.exit("Nothing downloaded - check your internet connection, or add your own "
                 ".txt/.pdf/.docx files to documents/ by hand.")
    write_sources_file(folder, rows)
    print(f"Done. {len(rows)} documents in '{folder}/' (sources listed in SOURCES.md)")
    if len(rows) < 5:
        print("Warning: the assignment needs at least 5 documents - add some manually.")


if __name__ == "__main__":
    main()
