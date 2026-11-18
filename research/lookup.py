"""Fact lookup for SlingerGPT 1 pico: a keyword search over the Wikipedia text on this PC, so it can answer from a
passage instead of from memory (research/plan_fact_lookup.md).

    python -P research/lookup.py build                 the index, data/lookup/wiki.db (processor, about half an hour)
    python -P research/lookup.py search "question"     the best passages for a question

Simple English Wikipedia (all 242,000 articles) and the English Wikipedia files in data/raw128/ (wiki_en_*, about 15%
of its articles), cut into passages of about 120 words along paragraph breaks, each with its article's title. The
search is BM25 ranking in SQLite's FTS5, which comes with Python: no search library needed. Titles count four times
as much as body text, and words are reduced to their stems (capitals -> capital).
"""
import re, sqlite3, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB = ROOT / "data" / "lookup" / "wiki.db"
WORDS = 120
# How a passage is put in front of SlingerGPT, after its "You are SlingerGPT..." System line: the same words in the
# reading practice (research/finetune_slinger.py) and on the chat page (chat_server.py)
PASSAGE = "Answer using this passage from Wikipedia ({title}): {passage}"
NOT_SAID = "The text doesn't say."   # its answer when the passage does not have one
STOP = set("""a an the of in on at to for from by with and or is are was were be been being am what who whom whose which
when where why how do does did done can could would should will shall may might must i you he she it we they me him her us
them my your his its our their this that these those there here as about into than then so not no yes tell me please give
have has had many much known called
name list explain describe define meaning""".split())


def passages(text):
    """Paragraph-aware passages of about WORDS words (long paragraphs are split)."""
    out, cur = [], []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            continue
        if cur and len(cur) + len(words) > 2 * WORDS:   # a long paragraph next: what came before is a passage of its own
            out.append(" ".join(cur))
            cur = []
        cur += words
        while len(cur) >= 2 * WORDS:
            out.append(" ".join(cur[:WORDS]))
            cur = cur[WORDS:]
        if len(cur) >= WORDS:
            out.append(" ".join(cur))
            cur = []
    if cur:   # a short ending joins the passage before it
        if out and len(cur) < 15:
            out[-1] += " " + " ".join(cur)
        else:
            out.append(" ".join(cur))
    return out


def build():
    import pyarrow.parquet as pq
    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    db = sqlite3.connect(DB)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
                     "CREATE VIRTUAL TABLE passages USING fts5(title, body, source UNINDEXED, tokenize='porter unicode61');")
    sources = [("Simple English Wikipedia", [ROOT / "data" / "raw_slinger" / "simplewiki" / "train-00000-of-00001.parquet"]),
               ("Wikipedia", sorted((ROOT / "data" / "raw128").glob("wiki_en_*.parquet")))]
    began, total = time.time(), 0
    for source, files in sources:
        for f in files:
            rows, articles = [], 0
            for batch in pq.ParquetFile(f).iter_batches(batch_size=2048, columns=["title", "text"]):
                for r in batch.to_pylist():
                    articles += 1
                    rows += [(r["title"], p, source) for p in passages(r["text"] or "")]
                if len(rows) > 200_000:
                    db.executemany("INSERT INTO passages VALUES (?, ?, ?)", rows)
                    total += len(rows)
                    rows = []
            db.executemany("INSERT INTO passages VALUES (?, ?, ?)", rows)
            total += len(rows)
            db.commit()
            print(f"{f.name}: {articles:,} articles; {total:,} passages so far ({time.time() - began:.0f} s)", flush=True)
    db.execute("INSERT INTO passages(passages) VALUES ('optimize')")
    db.commit()
    print(f"done: {total:,} passages, {DB.stat().st_size / 1e9:.1f} GB ({time.time() - began:.0f} s)")


_db = _titles = None
TITLES = DB.with_name("titles.db")   # article name -> its opening passage's rowid, for exact-title matches


def words(text):
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def name(title):
    """An article's title as plain lower-case words: "Black hole" -> "black hole". A "(...)" stays part of it, so
    "Red Planet (miniseries)" does not count as titled "red planet"."""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def build_titles():
    """TITLES: the first (opening) passage of every article, by name(title), one per source. A minute or two."""
    TITLES.unlink(missing_ok=True)
    db, out = sqlite3.connect(f"file:{DB}?mode=ro", uri=True), sqlite3.connect(TITLES)
    out.execute("CREATE TABLE titles (name TEXT, at INTEGER)")
    seen, rows = set(), []
    for rowid, title, source in db.execute("SELECT rowid, title, source FROM passages"):
        if (title, source) not in seen:
            seen.add((title, source))
            rows.append((name(title), rowid))
    out.executemany("INSERT INTO titles VALUES (?, ?)", rows)
    out.execute("CREATE INDEX by_name ON titles (name)")
    out.commit()
    print(f"titles: {len(rows):,} articles")


def search(question, k=1, pool=40):
    """The k best passages as (title, passage, source), best first; [] if nothing matches. Two steps, as a person
    would look something up: candidates are the best keyword matches (BM25) plus the passages of the articles whose
    titles best match the question's words; each is then scored by how much of its title the question names (so
    "capital of Japan" picks "Japan", not "Capital punishment in Japan"), how many of the question's words the passage
    holds, and a bonus for an article's opening passage, where Wikipedia puts the main facts.
    16 November: the opening passage of every article titled exactly like words of the question is a candidate too,
    with a bonus ("What is the internet?" picked "Demon Internet", "What is a rainbow?" Radiohead's "In Rainbows",
    "What is the heart?" the reference list of "Heart"), and "may refer to" pages count for little."""
    global _db, _titles
    if _db is None:
        _db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, check_same_thread=False)
        if not TITLES.exists():
            build_titles()
        _titles = sqlite3.connect(f"file:{TITLES}?mode=ro", uri=True, check_same_thread=False)
    terms = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOP and (len(w) > 1 or w.isdigit())]
    if not terms:
        return []
    query = " OR ".join(f'"{w}"' for w in dict.fromkeys(terms))
    found = {}
    for match, limit in ((query, pool), ("{title} : (" + query + ")", 4 * pool)):   # keyword matches; title matches
        for rowid, title, body, source in _db.execute("SELECT rowid, title, body, source FROM passages WHERE passages MATCH ? "
                                                      "ORDER BY bm25(passages, 4.0, 1.0) LIMIT ?", (match, limit)):
            found[rowid] = (title, body, source)
    raw = re.findall(r"[a-z0-9]+", question.lower())   # one to four words in a row, not starting or ending with a small word
    phrases = {" ".join(raw[i:j]) for i in range(len(raw)) for j in range(i + 1, min(i + 5, len(raw) + 1))
               if raw[i] not in STOP and raw[j - 1] not in STOP}
    asked, exact = words(question), set()
    for phrase in phrases:   # articles titled exactly like words of the question in a row ("statue of liberty")
        for (at,) in _titles.execute("SELECT at FROM titles WHERE name = ?", (phrase,)):
            row = _db.execute("SELECT title, body, source FROM passages WHERE rowid = ?", (at,)).fetchone()
            if asked <= words(row[0]) | words(row[1]):   # only if its opening holds every word of the question
                found[at] = row
                exact.add(at)
    ranked = []
    for rowid, (title, body, source) in found.items():
        named = words(title)
        cover = len(named & asked) / len(named) if named else 0.0
        held = len(asked & words(body)) / len(asked)
        before = _db.execute("SELECT title FROM passages WHERE rowid = ?", (rowid - 1,)).fetchone()
        opening = before is None or before[0] != title
        listing = 0.3 if "may refer to" in body[:300] else 1.0
        ranked.append(((1 + 2 * cover) * (0.2 + held) * (1.3 if opening else 1.0) * (1.5 if rowid in exact else 1.0) * listing,
                       -rowid, title, body, source))
    ranked.sort(reverse=True)
    return [r[2:] for r in ranked[:k]]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["search"]:
        began = time.time()
        for title, body, source in search(" ".join(sys.argv[2:]), 3):
            print(f"[{source}] {title}: {body[:300]}\n")
        print(f"({time.time() - began:.2f} s)")
