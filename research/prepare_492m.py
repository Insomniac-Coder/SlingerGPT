"""SlingerGPT 492M: the training text, prepared in rounds (TRAINING_PLAN.md stage A.2; sources: research/492m/data_mix.md).

    python research/prepare_492m.py sample --data DIR [--docs 300]   a few hundred documents of every source, end to
                                                                     end, with samples to read (no full downloads)
    python research/prepare_492m.py run --data /data/492m --workers 96 --approved
                                                                     all rounds; the trainer can start after round 1
    python research/prepare_492m.py status --data /data/492m         what is ready, per source
    python research/prepare_492m.py tokenizer --data DIR --workers 4 the word-piece list from streamed samples (any
                                                                     machine; run then uses it with --tokenizer)

Each round, for every source: download its next files -> clean -> remove copies -> cut into pieces.
  download   the source's files from Hugging Face (Xet, high-performance mode), in a fixed random order; round 1 takes about 5% of
             every source (all of the small ones), later rounds the rest
  clean      format, language, quality, behaviour rules for chats, test questions removed (data_mix.md, "Cleaning")
  copies     exact copies (SHA of the lower-cased text) and near copies (MinHash, 5-word shingles, 14 bands of 8)
             across all rounds; the first copy seen is kept, in the priority order of data_mix.md within a round
  pieces     the 16,384-piece list is trained on round 1's cleaned text (or read from --tokenizer); each source's
             documents are written to <data>/<source>/NNNN.bin (16-bit pieces, each document ended by piece 0), and
             <data>/manifest.json lists every finished file, so the trainer reads only finished files
One document in 100 per source (fixed by a hash of its id) is held out: <data>/<source>/val.bin (round 1 only).

Waiting for the owner (TRAINING_PLAN.md section 3): the data list, the chat layout and reserved pieces, and the
behaviour filters on chats (BEHAVIOUR below). `run` refuses to start without --approved.
"""
import argparse, bz2, collections, gzip, hashlib, html, html.parser, io, json, math, os, random, re, shutil, sys
import tarfile, time, unicodedata, urllib.request, zipfile, zlib
from multiprocessing import get_context   # "spawn": fresh worker processes; copies of a parent that already runs
                                          # threads (pyarrow, Hugging Face's Xet downloader) can freeze
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
END = 0
VOCAB = 16384
SPECIAL = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end_of_turn|>", "<|search|>",
           "<|search_result|>"] + [f"<|spare_{i}|>" for i in range(1, 10)]   # 16 reserved (decision 2)
ROUNDS, FIRST_ROUND = 6, 0.05
LONG_CHARS, CHUNK_CHARS = 32_000, 64_000   # a long document is 8k+ pieces (~4 letters each); long-phase chunks <= ~16k
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
CHAT_LAYOUT = "plain"   # decision 2: "plain" = the pico's System:/User:/Assistant: lines; "markers" = reserved pieces
BEHAVIOUR = {           # data_mix.md "Conversation" filters, as the owner approved them on 18 November 2026
    "drop_other_assistant_names": True, "drop_canned_refusals": False,   # owner: keep chats with refusals
    "drop_only_i_dont_know": True,
    "drop_knowledge_cutoff": True, "drop_function_calls": True,
}
WORDS = re.compile(r"[a-z0-9]+")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------------------------------------- formatting helpers

class HtmlText(html.parser.HTMLParser):
    """Stack Exchange HTML -> plain text: code blocks fenced, list items kept, paragraphs as blank lines."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.pre, self.skip = [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.pre += 1
            self.out.append("\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in ("p", "div", "br", "tr", "h1", "h2", "h3", "h4", "blockquote", "ul", "ol", "table"):
            self.out.append("\n\n" if tag in ("p", "h1", "h2", "h3", "h4") else "\n")
        elif tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "pre":
            self.pre -= 1
            self.out.append("\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)

    def text(self):
        return re.sub(r"\n{3,}", "\n\n", "".join(self.out)).strip()


def html_text(s):
    p = HtmlText()
    p.feed(s or "")
    p.close()
    return p.text()


def chat(turns, system=None):
    """turns: [(role, text)] -> the chat layout (decision 2). Plain: one "Role: text" block per turn."""
    if CHAT_LAYOUT == "markers":
        marks = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
        out = ([f"<|system|>{system}<|end_of_turn|>"] if system else []) + \
            [f"{marks[r]}{t.strip()}<|end_of_turn|>" for r, t in turns]
        return "".join(out)
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    return "\n".join(([f"System: {system.strip()}"] if system else []) + [f"{names[r]}: {t.strip()}" for r, t in turns])


def stable(s, bits=64):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=bits // 8).digest(), "little")


def normal(text):
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


# --------------------------------------------------------------------------------------------------------- the sources

# The adapters: where each source's files are, how a row becomes documents, which output sources it feeds, and which
# cleaning rules apply. `use` = the share of the dataset's files to download (data_mix.md's "Download" column).
# Output sources ending in _4 / _top are the best slices for the anneal; _long ones are for the 16,384 phase.

def fineweb(row):
    s = (row.get("metadata") or {}).get("int_score", 0)
    if s < 3:
        return []
    return [(row["text"], ["fineweb_edu"] + (["fineweb_edu_4"] if s >= 4 else []), row["id"], (row.get("metadata") or {}).get("url"))]


def dclm(row):
    s = row.get("edu_int_score") or 0
    if s < 3:
        return []
    return [(row["text"], ["dclm_edu"] + (["dclm_edu_4"] if s >= 4 else []), row["id"], row.get("url"))]


COSMO_TOPICS = re.compile(r"\b(computer|programming|software|algorithm|internet|cybersecurity|network security|encryption|"
                          r"operating system|database|python|javascript|mathematics|algebra|geometry|calculus|statistics|"
                          r"probability|physics|chemistry|biology|astronomy|engineering|electronics)\b", re.I)


def cosmopedia(row):
    """Toward textbooks for college students and computer, maths and science topics (about half the set kept);
    stories about 1 in 4, so they stay near 10% of what this row reads."""
    fmt, aud = (row.get("format") or "").lower(), (row.get("audience") or "").lower()
    if fmt.startswith("story"):
        keep = stable(row["text"][:200]) % 4 == 0
    else:
        topics = {m.lower() for m in COSMO_TOPICS.findall((row.get("prompt") or "")[:3000])}
        keep = (fmt.startswith("textbook") and "college" in aud) or len(topics) >= 2 or (fmt.startswith("textbook") and topics)
    return [(row["text"], ["cosmopedia"], None, None)] if keep else []


def nemotron(row):
    return [(row["text"], ["nemotron_qa"], row.get("warc_record_id"), row.get("url"))]


STACK_OTHER_KEEP = {   # share of each language's files kept (data_mix.md's split of the row; Java downsampled)
    "JavaScript": 1.0, "TypeScript": 1.0, "C": 1.0, "C++": 1.0, "Java": 0.35, "C#": 1.0, "SQL": 1.0, "Shell": 1.0,
    "Go": 1.0, "Rust": 1.0, "PHP": 0.5, "Ruby": 1.0, "Swift": 1.0, "Markdown": 0.6}


def stack_edu(row):
    m = row.get("metadata") or {}
    if m.get("license_type") not in (None, "permissive") or m.get("is_generated") or m.get("is_vendor"):
        return []
    lang, s = m.get("language") or "", row.get("int_score") or 0
    text = row["text"]
    if lang == "Python":
        outs = ["stack_python"] + (["stack_python_4"] if s >= 4 else [])
    elif lang in STACK_OTHER_KEEP:
        if stable(row["id"]) % 1000 >= 1000 * STACK_OTHER_KEEP[lang]:
            return []
        outs = ["stack_other"] + (["stack_other_4"] if s >= 4 else [])
    else:
        return []
    return [(text, outs, row["id"], None, {"repo": m.get("repo_name"), "path": m.get("path"), "lang": lang})]


def stack_html(row):
    return [(row["text"], ["stack_html"], row["id"], None)]


def kaggle(row):
    return [(row["text"], ["kaggle"], row.get("file_id"), None)]


def opencoder(row):
    return [(row["text"], ["opencoder"], None, None)]


SE_TECH = {"stackoverflow", "superuser", "askubuntu", "unix", "serverfault", "cs", "softwareengineering", "security",
           "networkengineering", "dba", "codereview", "datascience"}
SE_MATH = {"math"}
SE_SKIP = {"mathoverflow", "es.stackoverflow", "ru.stackoverflow", "pt.stackoverflow", "ja.stackoverflow"}
NOISE = re.compile(r"possible duplicate|what have you tried|^thanks?\b|^\+1\b", re.I)


def se_site(row):
    url = next((u for u in row.get("metadata") or [] if u), "")
    host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    return re.sub(r"\.(stackexchange|com|net)$", "", re.sub(r"\.stackexchange\.com$|\.com$", "", host))


def stack_exchange(row):
    site = se_site(row)
    if ".meta" in site or site.startswith("meta.") or site in SE_SKIP or not site:
        return []
    group = "se_tech" if site in SE_TECH else "se_math" if site in SE_MATH else "se_other"
    answers = []
    for a in sorted(row.get("answers") or [], key=lambda a: (not a.get("selected"), -(a.get("pm_score") or 0))):
        text = html_text(a.get("text"))
        if len(text.split()) < 20 or NOISE.search(text[:200]):
            continue
        if a.get("selected") or (a.get("pm_score") or 0) >= 2:
            answers.append((text, bool(a.get("selected")), a.get("pm_score") or 0))
    answers = answers[:3]   # the accepted answer and up to 2 more
    if not answers:
        return []
    q = html_text(row.get("question"))
    whole = q + "".join(f"\n\nAnswer:\n{a}" for a, _, _ in answers)
    # the anneal's slice: threads with an accepted answer or one scored 4+ (the same text, so it is one document
    # with two outputs; two near-identical documents would remove each other as copies)
    top = any(sel or score >= 4 for _, sel, score in answers)
    return [(whole, [group] + ([group + "_top"] if top else []), f"{site}/{row.get('qid')}", None, {"question": q})]


EAI_CS = ("004", "005")


def essential(row):
    tax = (row.get("eai_taxonomy") or {})
    code = str((((tax.get("free_decimal_correspondence") or {}).get("primary") or {}).get("code")) or "")
    if not code.startswith(EAI_CS):
        return []
    level = json.dumps(tax.get("education_level") or {}).lower()
    kind = json.dumps(tax.get("document_type_v2") or tax.get("document_type_v1") or {}).lower()
    if not any(w in level for w in ("undergraduate", "graduate", "expert")):
        return []
    if not any(w in kind for w in ("reference", "educational", "tutorial", "q&a", "question", "documentation", "academic")):
        return []
    return [(row["text"], ["essential_cs"], str(row.get("id")), ((row.get("metadata") or {}).get("url")))]


CS_WORDS = re.compile(r"\b(algorithm|programming|computer|software|compiler|operating system|database|python|java|"
                      r"network|data structure|machine learning|linux|function|variable|source code)\b", re.I)
SCIENCE_WORDS = re.compile(r"\b(physics|chemistry|biology|theorem|equation|experiment|molecule|quantum|lemma|proof)\b", re.I)


def finepdfs(row):
    if row.get("full_doc_lid") not in (None, "eng_Latn") or (row.get("full_doc_lid_score") or 1) < 0.65:
        return []
    text = row["text"]
    head = text[:20000]
    cs = len(CS_WORDS.findall(head)) >= 5
    outs = (["finepdfs_cs"] if cs else []) + (["finepdfs_long"] if len(text) >= LONG_CHARS and
                                              (cs or len(SCIENCE_WORDS.findall(head)) >= 8) else [])
    return [(text, outs, row.get("id"), row.get("url"))] if outs else []


def finemath4(row):
    return [(row["text"], ["finemath_4"], row.get("url"), row.get("url"))]


def finemath3(row):
    return [(row["text"], ["finemath_3"], row.get("url"), row.get("url"))] if (row.get("int_score") or 0) == 3 else []


def infiwebmath(row):
    return [(row["text"], ["infiwebmath"], row.get("url"), row.get("url"))]


def qa(question, answer, out, id_):
    return [(chat([("user", question), ("assistant", answer)]), [out], id_, None, {"question": question})]


def orca(row):
    return qa(row["question"], row["answer"], "worked_maths", row["question"][:200])


def metamath(row):
    return qa(row["query"], row["response"], "worked_maths", row["query"][:200])


def omi2(row):
    if len(row["generated_solution"]) > 6000:   # over-long items dropped
        return []
    return qa(row["problem"], row["generated_solution"], "worked_maths", row["problem"][:200])


def numina(row):
    if row.get("source") not in ("cn_k12", "synthetic_math"):
        return []
    return qa(row["problem"], row["solution"], "worked_maths", row["problem"][:200])


def tinygsm(row):
    if stable(row["question"]) % 5:   # a sample: about 1 in 5
        return []
    return qa(row["question"], "```python\n" + row["code"].strip() + "\n```", "worked_maths", row["question"][:200])


def gsm8k(row):
    steps, _, result = re.sub(r"<<[^>]*>>", "", row["answer"]).rpartition("####")
    return qa(row["question"], steps.strip() + f"\nThe answer is {result.strip()}.", "worked_maths", row["question"][:200])


def finewiki(row):
    return [(row["text"], ["finewiki"], row["id"], row.get("url"))]


def simplewiki(row):
    return [(row["text"], ["simplewiki"], row["id"], row.get("url"))]


PG19_HELD_OUT = set()   # filled from deepmind/pg19's validation and test splits when the cleaning starts


def gutenberg(row):
    if (row.get("metadata") or {}).get("language", "en") != "en" or str(row.get("id")) in PG19_HELD_OUT:
        return []
    return [(row["text"], ["gutenberg"], row["id"], None, {"book": row["id"]})]


def doab(row):
    m = row.get("metadata") or {}
    return [(row["text"], ["doab"], f"{m.get('book_id')}/{m.get('section_idx')}", None, {"book": m.get("book_id")})]


def arxiv_abs(row):
    return [(row["text"], ["arxiv_abstracts"], row["id"], None)]


def arxiv_papers(row):
    return [(row["text"], ["arxiv_papers"], row["id"], None)]


def pes2o(row):
    return [(row["text"], ["pes2o"], row["id"], None)]


SMOLTALK_SKIP = ("apigen", "numina-cot", "longalign")


def smoltalk(row):
    if any(s in (row.get("source") or "") for s in SMOLTALK_SKIP):
        return []
    return conversation(row["messages"], "smoltalk")


def ultrachat(row):
    return conversation(row["messages"], "ultrachat", row.get("prompt_id"))


def soda(row):
    lines = "\n".join(f"{w}: {s}" for w, s in zip(row["speakers"], row["dialogue"]))
    return [(row["narrative"] + "\n\n" + lines, ["soda"], str(row.get("original_index")), None, {"question": row["dialogue"][0]})]


def wildchat(row):
    if row.get("language") != "English" or any(t.get("toxic") for t in row.get("conversation") or []):
        return []
    return conversation(row["conversation"], "wildchat", row.get("conversation_hash"))


def conversation(messages, out, id_=None):
    turns = [(m["role"], m["content"]) for m in messages if m.get("role") in ("system", "user", "assistant") and m.get("content")]
    system = turns[0][1] if turns and turns[0][0] == "system" else None
    turns = [t for t in turns if t[0] != "system"]
    if not turns or turns[0][0] != "user" or not any(r == "assistant" for r, _ in turns):
        return []
    return [(chat(turns, system), [out], id_, None, {"question": turns[0][1], "turns": turns})]


def libretexts(row):
    url = (row.get("metadata") or {}).get("url") or row.get("id") or ""
    if not re.search(r"eng\.libretexts\.org/Bookshelves/Computer_Science|workforce\.libretexts\.org/Bookshelves/Information_Technology", url):
        return []
    return [(row["text"], ["docs"], url, None)]


def pressbooks(row):
    m = row.get("metadata") or {}
    if not re.search(r"comput|programming|software|information technology", (m.get("subject") or "") + (m.get("title") or ""), re.I):
        return []
    return [(row["text"], ["docs"], row.get("id"), None)]


def peps(row):
    return [(row["text"], ["docs"], row.get("id"), None)]


def archwiki(row):
    if row.get("lang") != "en":
        return []
    return [(f"# {row.get('title', '')}\n{row.get('content', '')}", ["docs"], row.get("url_path"), None)]


# `use` was set from the 300-document samples (18 November) so each source is read about 0.6-0.9 times: enough
# margin for the estimates, without downloading text that is never read. make_mix.py --measured checks it after round 1.
# kind: which cleaning rules apply. near: in the near-copy pool. lid: fastText English threshold (None: not checked).
ADAPTERS = {
    #  name           repo                                            files under    reader    use     row      kind      near  lid
    "fineweb":      ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup/", "parquet", 0.10, fineweb, "web", True, None),
    "dclm":         ("HuggingFaceTB/dclm-edu", "data/", "parquet", 0.035, dclm, "web", True, None),   # ~1 in 4 rows has edu >= 3
    "cosmopedia":   ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", "parquet", 0.50, cosmopedia, "synthetic", False, None),
    "nemotron":     ("commoncrawl:Nemotron-CC", "quality=high/kind=synthetic/kind2=diverse_qa_pairs/", "zstd", 0.01, nemotron, "synthetic", False, None),
    "stack_edu":    ("common-pile/stackv2_edu_filtered", "", "jsongz", 1.0, stack_edu, "code", False, None),
    "stack_html":   ("common-pile/stackv2_html_filtered", "", "jsongz", 1.0, stack_html, "code", False, None),
    "kaggle":       ("HuggingFaceTB/issues-kaggle-notebooks", "kaggle/", "parquet", 1.0, kaggle, "code", False, None),
    "opencoder_qa": ("OpenCoder-LLM/opc-annealing-corpus", "synthetic_qa/", "jsonl", 0.15, opencoder, "code", False, None),
    "opencoder_sn": ("OpenCoder-LLM/opc-annealing-corpus", "synthetic_code_snippet/", "jsonl", 0.15, opencoder, "code", False, None),
    "stackexchange": ("HuggingFaceH4/stack-exchange-preferences", "data/", "parquet", 1.0, stack_exchange, "se", True, 0.65),
    "essential":    ("EssentialAI/eai-taxonomy-code-w-dclm-100b-sample", "data/", "parquet", 0.05, essential, "web", True, 0.65),
    "libretexts":   ("common-pile/libretexts_filtered", "", "jsongz", 1.0, libretexts, "web", True, 0.65),
    "pressbooks":   ("common-pile/pressbooks_filtered", "", "jsongz", 1.0, pressbooks, "web", True, 0.65),
    "peps":         ("common-pile/python_enhancement_proposals_filtered", "", "jsongz", 1.0, peps, "docs", False, None),
    "archwiki":     ("hannah-eee/arch-wiki-docs", "", "jsonl", 1.0, archwiki, "docs", False, None),
    "docs_web":     ("web:docs", "", "web", 1.0, None, "docs", False, None),
    "finepdfs":     ("HuggingFaceFW/finepdfs-edu", "data/eng_Latn/", "parquet", 0.10, finepdfs, "pdf", True, 0.65),
    "finemath4":    ("HuggingFaceTB/finemath", "finemath-4plus/", "parquet", 0.60, finemath4, "web", False, None),
    "finemath3":    ("HuggingFaceTB/finemath", "finemath-3plus/", "parquet", 0.15, finemath3, "web", False, None),
    "infiwebmath":  ("HuggingFaceTB/finemath", "infiwebmath-3plus/", "parquet", 0.07, infiwebmath, "web", False, None),
    "orca":         ("microsoft/orca-math-word-problems-200k", "data/", "parquet", 1.0, orca, "maths", False, None),
    "metamath":     ("meta-math/MetaMathQA", "MetaMathQA-395K.json", "json", 1.0, metamath, "maths", False, None),
    "omi2":         ("nvidia/OpenMathInstruct-2", "data/train_1M", "parquet", 1.0, omi2, "maths", False, None),
    "numina":       ("AI-MO/NuminaMath-CoT", "data/train", "parquet", 1.0, numina, "maths", False, None),
    "tinygsm":      ("TinyGSM/TinyGSM", "data/", "parquet", 0.25, tinygsm, "maths", False, None),
    "gsm8k":        ("openai/gsm8k", "main/train", "parquet", 1.0, gsm8k, "maths", False, None),
    "finewiki":     ("HuggingFaceFW/finewiki", "data/enwiki/", "parquet", 1.0, finewiki, "wiki", True, None),
    "simplewiki":   ("HuggingFaceFW/finewiki", "data/simplewiki/", "parquet", 1.0, simplewiki, "wiki", True, None),
    "gutenberg":    ("common-pile/project_gutenberg_filtered", "", "jsongz", 1.0, gutenberg, "book", True, None),
    "doab":         ("common-pile/doab_filtered", "", "jsongz", 1.0, doab, "book", True, 0.65),
    "arxiv_abs":    ("common-pile/arxiv_abstracts_filtered", "", "jsongz", 1.0, arxiv_abs, "paper", True, 0.65),
    "arxiv_papers": ("common-pile/arxiv_papers_filtered", "", "jsongz", 1.0, arxiv_papers, "paper", True, 0.65),
    "pes2o":        ("common-pile/peS2o_filtered", "", "jsongz", 0.06, pes2o, "paper", True, 0.65),
    "smoltalk":     ("HuggingFaceTB/smoltalk", "data/all/train", "parquet", 1.0, smoltalk, "chat", False, None),
    "ultrachat":    ("HuggingFaceH4/ultrachat_200k", "data/train_sft", "parquet", 1.0, ultrachat, "chat", False, None),
    "soda":         ("allenai/soda", "train.parquet", "parquet", 1.0, soda, "chat", False, None),
    "wildchat":     ("allenai/WildChat-1M", "data/", "parquet", 1.0, wildchat, "chat", False, None),
}
# priority when the same text appears twice in one round (data_mix.md): the first in this order is kept
PRIORITY = ["finewiki", "simplewiki", "gutenberg", "doab", "stackexchange", "fineweb", "dclm", "essential"]
# documents long enough for the 16,384 phase: the share of them that goes there instead (so they are not read twice)
LONG_ROUTE = {"gutenberg": ("gutenberg_long", 0.15), "doab": ("doab_long", 0.2), "pes2o": ("pes2o_long", 0.3),
              "finewiki": ("finewiki_long", 0.5), "docs": ("docs_long", 0.5)}
LONG_ONLY = {"arxiv_papers", "finepdfs_long", "code_repos"}
QUESTION_SHAPED = {"chat", "se", "maths"} | {"nemotron"}
DOCS_WEB = [   # small authoritative manuals (data_mix.md "Documentation and open CS textbooks"); a failed fetch is skipped
    ("python-docs", "https://docs.python.org/3.14/archives/python-3.14-docs-text.tar.bz2", r"\.txt$"),
    ("mdn", "https://github.com/mdn/content/archive/refs/heads/main.tar.gz",
     r"files/en-us/(learn|web/(javascript|css|html|http)/guide)/.*index\.md$"),
    ("tldr", "https://github.com/tldr-pages/tldr/archive/refs/heads/main.tar.gz", r"/pages/(common|linux|osx|windows)/.*\.md$"),
    ("rust-book", "https://github.com/rust-lang/book/archive/refs/heads/main.tar.gz", r"/src/ch.*\.md$"),
]


# ------------------------------------------------------------------------------------------------ reading the files

def list_files(adapter):
    repo, under, reader, use = ADAPTERS[adapter][:4]
    if reader == "web":
        return [name for name, _, _ in DOCS_WEB]
    if repo.startswith("commoncrawl:"):
        base = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/"
        with urllib.request.urlopen(base + "data-jsonl.paths.gz", timeout=120) as r:
            paths = gzip.decompress(r.read()).decode().split()
        files = ["https://data.commoncrawl.org/" + p for p in paths if under in p]
    else:
        from huggingface_hub import HfApi
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(under) and
                 f.endswith((".parquet", ".json.gz", ".jsonl", ".json", ".jsonl.zstd"))]
    files.sort()
    random.Random(f"492m:{adapter}").shuffle(files)
    return files[:max(1, round(len(files) * use))]


def rounds_of(files):
    """Round 1: about 5% of the files (at least one); later rounds: the rest, evenly."""
    n = len(files)
    first = max(1, math.ceil(n * FIRST_ROUND))
    rest = files[first:]
    per = math.ceil(len(rest) / (ROUNDS - 1)) if rest else 0
    return [files[:first]] + [rest[i * per:(i + 1) * per] for i in range(ROUNDS - 1)]


def fetch(adapter, f, work, limit):
    """A local path for a file (downloaded in full), or a stream of its first rows when limit > 0 (samples)."""
    repo, _, reader = ADAPTERS[adapter][:3]
    if f.startswith("http"):
        if limit:
            return urllib.request.urlopen(f, timeout=300)
        local = work / "downloads" / adapter / Path(f).name
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            urllib.request.urlretrieve(f, str(local) + ".part")
            os.replace(str(local) + ".part", local)
        return local
    if limit:
        from huggingface_hub import HfFileSystem
        fh = HfFileSystem().open(f"datasets/{repo}/{f}", "rb", block_size=8 << 20)
        if reader != "parquet":
            return fh
        import pyarrow.parquet as pq
        if pq.ParquetFile(fh).metadata.row_group(0).total_byte_size < 256e6:
            fh.seek(0)
            return fh
        fh.close()   # one huge block (DCLM-Edu: a single 4.8 GB row group): a whole download is far quicker
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, f, repo_type="dataset", local_dir=str(work / "downloads" / adapter)))


def rows(adapter, f, work, limit=0):
    reader = ADAPTERS[adapter][2]
    if reader == "web":
        yield from web_docs(f, work)
        return
    src = fetch(adapter, f, work, limit)
    count = 0
    try:
        if reader == "parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(src)
            for b in pf.iter_batches(batch_size=1024 if not limit else 64):
                for r in b.to_pylist():
                    yield r
                    count += 1
                    if limit and count >= limit:
                        return
            return
        if reader == "json":   # one JSON list (MetaMathQA); a sample reads only its first 2 MB
            if isinstance(src, Path):
                data = json.load(open(src, encoding="utf-8"))
            else:
                head = src.read(2 << 20).decode("utf-8", "ignore")
                data = json.loads(head[:head.rindex("},") + 1] + "]")
            yield from data[:limit or None]
            return
        if reader == "zstd":
            import zstandard
            raw = open(src, "rb") if isinstance(src, Path) else src
            stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
        elif reader == "jsongz":
            stream = io.TextIOWrapper(gzip.open(src if isinstance(src, Path) else src), encoding="utf-8")
        else:
            stream = open(src, encoding="utf-8") if isinstance(src, Path) else io.TextIOWrapper(src, encoding="utf-8")
        for line in stream:
            if line.strip():
                yield json.loads(line)
                count += 1
                if limit and count >= limit:
                    return
    finally:
        if not isinstance(src, Path):
            src.close()


def web_docs(name, work):
    """The manuals that are not on Hugging Face: one document per page."""
    _, url, pattern = next(d for d in DOCS_WEB if d[0] == name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dataset-builder"})
        data = urllib.request.urlopen(req, timeout=300).read()
    except Exception as e:
        print(f"  docs_web: could not fetch {name} ({e}); skipped", flush=True)
        return
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        for m in tar.getmembers():
            if m.isfile() and re.search(pattern, m.name):
                text = tar.extractfile(m).read().decode("utf-8", "replace")
                text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)   # MDN front matter
                yield {"text": text, "id": f"{name}/{m.name}"}


def docs_web(row):
    return [(row["text"], ["docs"], row["id"], None)]


# ------------------------------------------------------------------------------------------------------- cleaning

NAMES = r"(SmolLM|ChatGPT|GPT-?4o?|GPT-?3\.5|OpenAI|Claude|Anthropic|Llama|Meta AI|Qwen|Alibaba|Gemma|Gemini|Bard|Google|" \
        r"Mistral|Mixtral|DeepSeek|Copilot|Microsoft)"
ASSISTANTS = re.compile(   # the assistant naming itself as another assistant, or its maker; a mere mention is fine
    rf"\b(i am|i'm|i was|my name is|as an?|this is|developed by|created by|trained by|built by|made by|from)\b[^.\n]{{0,30}}"
    rf"\b{NAMES}\b|as an ai language model|as a large language model|as a language model", re.I)
REFUSAL = re.compile(r"^(i'?m sorry|sorry|i apologi[sz]e|i cannot|i can'?t|i am unable|i'm unable|unfortunately,? i)"
                     r"[^.]{0,120}(can'?t|cannot|unable|not able|not allowed|won'?t)", re.I)
DONT_KNOW = re.compile(r"^\W*(i\s+(do\s*n[o']?t|don'?t)\s+know|i'?m not sure)\W*$", re.I)
CUTOFF = re.compile(r"knowledge cut-?off|as of my (last|latest) (update|training)|my training data (only )?goes", re.I)
SECRETS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "<PRIVATE_KEY>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<AWS_KEY>"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|sk-[A-Za-z0-9]{32,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"), "<TOKEN>"),
    (re.compile(r"(?i)((api[_-]?key|secret|token|passw(or)?d)\s*[:=]\s*['\"])[A-Za-z0-9+/_\-]{16,}(['\"])"), r"\1<SECRET>\4"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
    (re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), "<IP_ADDRESS>"),
]
BLOB = re.compile(r"[A-Za-z0-9+/=]{200,}|(?:[0-9a-fA-F]{2}[ ,]?){120,}")


class Rules:
    """Everything a cleaning process needs, loaded once per process."""
    lid = None
    decon = None

    @classmethod
    def load(cls, work):
        if cls.decon is None:
            d = np.load(work / "decon.npz", allow_pickle=True)
            cls.decon = {k: d[k] for k in d.files}
            cls.questions = json.loads((work / "decon_questions.json").read_text(encoding="utf-8"))
            cls.index = collections.defaultdict(list)
            for i, q in enumerate(cls.questions):
                for w in set(q["content"]):
                    cls.index[w].append(i)
            held = work / "pg19_held_out.json"
            if held.exists():
                PG19_HELD_OUT.update(json.loads(held.read_text()))
        return cls

    @classmethod
    def english(cls, text, work):
        """fastText's probability that the prose is English; code blocks are left out, as they confuse it."""
        text = re.sub(r"```.*?(```|$)", " ", text, flags=re.S)
        text = re.sub(r"`[^`\n]*`", " ", text)
        text = "\n".join(l for l in text.split("\n") if not l.startswith(("    ", "\t")))
        if cls.lid is None:
            import fasttext
            fasttext.FastText.eprint = lambda *a, **k: None
            cls.lid = fasttext.load_model(str(work / "lid.176.bin"))
        # the model's own predict() breaks on NumPy 2 (np.array(copy=False)); its C++ call gives (probability, label)
        pairs = cls.lid.f.predict(text[:3000].replace("\n", " "), 3, 0.0, "replace")
        return dict((label, prob) for prob, label in pairs).get("__label__en", 0.0)


STOP = set("a an the of to in and or is are was were be been for on at by with as it its this that what which who whom "
           "how why when where do does did can could would should will i you he she we they me my your our their "
           "there here from not no yes".split())
MERSENNE = (1 << 61) - 1
RNG = np.random.default_rng(492)
PERM_A = RNG.integers(1, 1 << 32, 112, dtype=np.uint64)
PERM_B = RNG.integers(0, 1 << 32, 112, dtype=np.uint64)
R = np.uint64(0x9E3779B97F4A7C15)   # odd, so it has an inverse modulo 2^64
R_INV = np.uint64(pow(int(R), -1, 1 << 64))
POW, POW_INV = [np.ones(1, dtype=np.uint64)], [np.ones(1, dtype=np.uint64)]


def powers(n):
    """R^i and R^-i (modulo 2^64) for i < n."""
    while len(POW[0]) < n:
        m = 2 * len(POW[0])
        with np.errstate(over="ignore"):
            POW[0] = np.cumprod(np.full(m, R, dtype=np.uint64)) * R_INV
            POW_INV[0] = np.cumprod(np.full(m, R_INV, dtype=np.uint64)) * R
    return POW[0][:n], POW_INV[0][:n]


def word_hashes(text):
    words = WORDS.findall(text.lower())
    return words, np.fromiter((zlib.crc32(w.encode()) for w in words), dtype=np.uint64, count=len(words))


def gram_hashes(h, n):
    """A hash of every run of n words, the same for the same run wherever it is (prefix sums of h_i R^i)."""
    if len(h) < n:
        return np.zeros(0, dtype=np.uint64)
    p, pinv = powers(len(h) + 1)
    with np.errstate(over="ignore"):
        prefix = np.concatenate([[np.uint64(0)], np.cumsum((h + np.uint64(1)) * p[:len(h)], dtype=np.uint64)])
        return (prefix[n:] - prefix[:-n]) * pinv[:len(h) - n + 1] ^ np.uint64(n * 0x100000001B3)


def minhash_bands(h):
    """14 band keys (8 rows each) of a 112-value MinHash over 5-word shingles; None for very short texts."""
    s = gram_hashes(h, 5)
    if len(s) == 0:
        return None
    s = (s >> np.uint64(32)) ^ (s & np.uint64(0xFFFFFFFF))   # 32 bits, so a x s fits in 64 bits
    sig = np.full(112, np.iinfo(np.uint64).max, dtype=np.uint64)
    for a in range(0, len(s), 4096):
        block = s[a:a + 4096, None]
        with np.errstate(over="ignore"):
            sig = np.minimum(sig, ((PERM_A * block + PERM_B) % np.uint64(MERSENNE)).min(0))
    bands = sig.reshape(14, 8)
    return np.array([stable(b.tobytes().hex() + str(i)) for i, b in enumerate(bands)], dtype=np.uint64)


def member(keys, table):
    """Which keys are in `table` (a sorted array): a binary search, with no sorting of the large table."""
    if len(table) == 0 or len(keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    at = np.minimum(np.searchsorted(table, keys), len(table) - 1)
    return table[at] == keys


MATCH_GRAMS = 4   # a public test item counts as copied when 4 of its 13-word runs are in the text (or all, if it has
                  # fewer): a copy shares many runs, while a common phrase ("in the name of the father and of the son
                  # and of the", "1 2 3 ... 13", a textbook formula) shares one or two


def contaminated(text, h, question=None):
    """Whether the text holds a test question (data_mix.md, decontamination): (reason, word positions) or (None, None).
    The positions are the words of the matched runs, so a long document can lose just those paragraphs."""
    d = Rules.decon
    for n_str in d["lengths"]:
        n = int(n_str)
        hit = member(gram_hashes(h, n), d[f"own{n}"])
        if hit.any():
            return "own test question", spans(np.flatnonzero(hit), n)
    g = gram_hashes(h, 13)
    if len(g):
        table = d["public13"]
        at = np.minimum(np.searchsorted(table, g), len(table) - 1)
        hit = table[at] == g
        if hit.any():
            items = d["public13_item"][at[hit]]
            found, counts = np.unique(items, return_counts=True)
            copied = found[counts >= np.minimum(MATCH_GRAMS, d["item_grams"][found])]
            if len(copied):
                where = np.flatnonzero(hit)[np.isin(items, copied)]
                return "public test (13 words)", spans(where, 13)
    for n in range(8, 13):
        hit = member(gram_hashes(h, n), d[f"public{n}"])
        if hit.any():
            return "public test (short item)", spans(np.flatnonzero(hit), n)
    if question is not None:
        words = WORDS.findall(question.lower())
        content = {w for w in words if w not in STOP}
        if content:
            hits = collections.Counter(i for w in content for i in Rules.index.get(w, ()))
            for i, c in hits.items():
                q = Rules.questions[i]
                if c >= 0.8 * len(set(q["content"])) and len(words) <= 1.5 * q["length"]:
                    return "close to a test question", None
    return None, None


def spans(starts, n):
    return np.unique((starts[:, None] + np.arange(n)[None, :]).ravel())


def drop_paragraphs(text, positions):
    """The text without the paragraphs that hold any of these word positions (long documents: data_mix.md)."""
    keep, at, wanted = [], 0, set(positions.tolist())
    for para in text.split("\n\n"):
        n = len(WORDS.findall(para.lower()))
        if not any(i in wanted for i in range(at, at + n)):
            keep.append(para)
        at += n
    return "\n\n".join(keep)


def prose_ok(text, kind):
    words = text.split()
    if len(words) < 50:
        return "under 50 words"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 5 and 1 - len(set(lines)) / len(lines) > 0.3:
        return "repeated lines"
    if len(words) >= 100:
        pairs = collections.Counter(zip(words, words[1:]))
        (a, b), n = pairs.most_common(1)[0]
        if n * (len(a) + len(b) + 1) / max(1, len(text)) > 0.2:
            return "repeated word pair"
    return None


def pdf_fix(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # hyphenated line breaks
    lines = text.split("\n")
    counts = collections.Counter(l.strip() for l in lines if 0 < len(l.strip()) < 80)
    lines = [l for l in lines if not re.fullmatch(r"\s*(page\s*)?\d{1,4}\s*", l, re.I) and
             not (counts.get(l.strip(), 0) >= 3 and len(l.strip()) < 80)]   # running headers and page numbers
    return "\n".join(lines)


def gutenberg_fix(text):
    text = re.sub(r"\[?transcriber'?s notes?:?.*?(\n\s*\n|\])", "", text, flags=re.I | re.S)
    out, skip = [], False
    for block in re.split(r"\n\s*\n", text):
        head = block.strip().split("\n")[0].strip().upper()
        if head in ("CONTENTS", "TABLE OF CONTENTS", "INDEX", "LIST OF ILLUSTRATIONS", "ILLUSTRATIONS"):
            skip = True
            continue
        if skip and (len(block.split()) > 40 and not re.search(r"\.{3,}|\s\d+\s*$", block)):
            skip = False
        if not skip:
            out.append(block)
    return "\n\n".join(out)


def code_ok(text, prose=False):
    """prose: HTML pages turned into text, Markdown and notebooks, whose paragraphs are long lines by nature."""
    if len(text.encode()) > 100_000:
        return None, "over 100 KB"
    lines = text.split("\n")
    if lines and not prose and (sum(map(len, lines)) / len(lines) > 100 or max(map(len, lines)) > 1000):
        return None, "minified"
    letters = sum(c.isalpha() for c in text)
    if letters < 0.25 * max(1, len(text)):
        return None, "mostly data"
    if BLOB.search(text):
        return None, "base64 or hex blob"
    for pattern, place in SECRETS:
        text = pattern.sub(place, text)
    return text, None


def chat_ok(turns):
    """The behaviour filters (BEHAVIOUR; the owner's yes is needed before the real run)."""
    for role, text in turns:
        if role != "assistant":
            continue
        if BEHAVIOUR["drop_other_assistant_names"] and ASSISTANTS.search(text):
            return "names another assistant"
        if BEHAVIOUR["drop_canned_refusals"] and REFUSAL.search(text.strip()) and len(text.split()) < 60:
            return "canned refusal"
        if BEHAVIOUR["drop_only_i_dont_know"] and DONT_KNOW.match(text.strip()):
            return "only 'I don't know'"
        if BEHAVIOUR["drop_knowledge_cutoff"] and CUTOFF.search(text):
            return "knowledge cut-off"
        if BEHAVIOUR["drop_function_calls"] and re.search(r"<tool_call>|\"function_call\"|<functioncall>", text):
            return "function call"
    return None


def boilerplate(source_rows):
    """Lines repeated across many documents of this file (menus, cookie notices): counted first, removed later."""
    counts, docs = collections.Counter(), 0
    for row in source_rows:
        text = row.get("text") or ""
        docs += 1
        counts.update({zlib.crc32(l.strip().encode()) for l in text.split("\n") if 10 <= len(l.strip()) <= 200})
    cut = max(10, 0.001 * docs)
    return {h for h, c in counts.items() if c >= cut}


def clean_file(task):
    """One file of one source -> work/clean/r<round>/<adapter>/<n>.parquet: the kept documents with their keys."""
    adapter, f, rnd, n, work, limit = task
    work = Path(work)
    out = work / "clean" / f"r{rnd}" / adapter / f"{n:05d}.parquet"
    if out.exists():
        return adapter, f, json.loads(out.with_suffix(".json").read_text())
    Rules.load(work)
    repo, _, _, _, make, kind, near, lid = ADAPTERS[adapter]
    make = make or docs_web
    # a sample (limit) is read once and kept in memory; a whole downloaded file is read twice from the local disk
    every = list(rows(adapter, f, work, limit)) if limit else None
    source_rows = (lambda: every) if limit else (lambda: rows(adapter, f, work, limit))
    common = boilerplate(source_rows()) if kind in ("web", "pdf", "book") else set()
    kept, why, began = [], collections.Counter(), time.time()
    group = collections.OrderedDict()   # documents of one book (doab sections) joined for the long phase
    for row in source_rows():
        for item in make(row):
            text, outs, id_, url = item[:4]
            extra = item[4] if len(item) > 4 else {}
            text = normal(text or "")
            if common:
                text = "\n".join(l for l in text.split("\n") if zlib.crc32(l.strip().encode()) not in common).strip()
            if kind == "pdf":
                text = pdf_fix(text)
                if sum(c.isalpha() for c in text) < 0.7 * max(1, sum(not c.isspace() for c in text)):
                    why["PDF: under 70% letters"] += 1
                    continue
            if adapter == "gutenberg":
                text = gutenberg_fix(text)
            if kind == "code":
                text, bad = code_ok(text, prose=adapter in ("stack_html", "kaggle") or extra.get("lang") == "Markdown")
                if bad:
                    why[f"code: {bad}"] += 1
                    continue
            elif kind in ("web", "pdf", "book", "paper", "se", "docs", "wiki"):
                bad = prose_ok(text, kind) if kind not in ("se", "docs") else (None if len(text.split()) >= 20 else "under 20 words")
                if bad:
                    why[bad] += 1
                    continue
            if kind == "chat":
                bad = chat_ok(extra.get("turns") or [])
                if bad:
                    why[f"chat: {bad}"] += 1
                    continue
            if lid is not None and Rules.english(text, work) < (0.5 if adapter == "stack_edu" else lid):
                why["not English"] += 1
                continue
            words, h = word_hashes(text)
            question = extra.get("question") if kind in QUESTION_SHAPED or adapter == "nemotron" else None
            bad, where = contaminated(text, h, question)
            if bad and where is not None and (kind in ("book", "paper", "pdf") or len(text) >= LONG_CHARS):
                text = drop_paragraphs(text, where)   # long documents lose only the paragraphs with the test text
                words, h = word_hashes(text)
                why[f"test: {bad} (paragraphs removed)"] += 1
                bad, _ = contaminated(text, h, question)
            if bad:
                why[f"test: {bad}"] += 1
                continue
            for src, (long_out, share) in LONG_ROUTE.items():   # some long documents go to the 16,384 phase only
                if src in outs and len(text) >= LONG_CHARS and stable(f"long:{id_ or text[:200]}") % 1000 < 1000 * share:
                    outs = [long_out if o == src else o for o in outs]
            if not outs:
                continue
            key = id_ if id_ is not None else text[:500]
            bands = minhash_bands(h) if near else None
            kept.append({"text": text, "outs": outs, "val": stable(f"val:{adapter}:{key}") % 100 == 0,
                         "exact": stable(" ".join(words)), "url": stable(url) if url else 0,
                         "bands": bands.tobytes() if bands is not None else b""})
            if adapter == "stack_edu" and extra.get("repo") and any(o in ("stack_python", "stack_other") for o in outs):
                group.setdefault(extra["repo"], []).append((extra.get("path") or "", text))
    for repo_name, files in group.items():   # whole repositories (files of one repository in this file) for the long phase
        joined = "\n\n".join(f"# File: {p}\n{t}" for p, t in sorted(files))
        if len(joined) >= LONG_CHARS:
            kept.append({"text": joined, "outs": ["code_repos"], "val": stable(f"val:repo:{repo_name}") % 100 == 0,
                         "exact": stable(joined[:100000]), "url": 0, "bands": b""})
    random.Random(f"{adapter}:{f}").shuffle(kept)
    import pyarrow as pa, pyarrow.parquet as pq
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": [k["text"] for k in kept], "outs": [k["outs"] for k in kept], "val": [k["val"] for k in kept],
                      "exact": pa.array([k["exact"] for k in kept], pa.uint64()),
                      "url": pa.array([k["url"] for k in kept], pa.uint64()), "bands": [k["bands"] for k in kept]})
    pq.write_table(table, str(out) + ".tmp", compression="zstd")
    os.replace(str(out) + ".tmp", out)
    info = {"kept": len(kept), "dropped": dict(why), "seconds": round(time.time() - began, 1), "file": f}
    out.with_suffix(".json").write_text(json.dumps(info))
    local = work / "downloads" / adapter / (Path(f).name if f.startswith("http") else f)
    local.unlink(missing_ok=True)   # free the disk: the download is not needed again
    return adapter, f, info


def clean_safely(task):
    """clean_file, with a failure reported instead of stopping the whole round."""
    try:
        return clean_file(task)
    except Exception as e:
        import traceback
        return task[0], task[1], {"kept": 0, "dropped": {}, "error": f"{type(e).__name__}: {e}",
                                  "trace": traceback.format_exc()[-1500:]}


# ---------------------------------------------------------------------------------------- the test-question lists

def test_items():
    """The project's own test questions and replies, and the public tests (data_mix.md sections 7 and 8)."""
    argv = sys.argv[:]
    import test_slinger as ts, chat_suite as cs, facts_quiz as fq, finetune_slinger as ft
    sys.argv = argv
    own = list(ts.CHAT) + [q for q, _ in ts.LOOKUP_QUESTIONS] + [q for q, _ in ts.BROAD_QUESTIONS] + \
        [q for q, _ in ts.IDENTITY_QUESTIONS] + list(ts.NONSENSE_QUESTIONS)
    own += [a for a, _, _ in cs.FACTS] + [b for _, b, _ in cs.FACTS] + [s for s, _ in cs.SUMS] + list(cs.LONG) + \
        [g for _, g in cs.GREET] + [w[1] for w in cs.WHO] + [i for i, _ in cs.INSTRUCT] + list(cs.IMPOSSIBLE) + \
        [q for q, _ in cs.LATE]
    for chats in (cs.SWITCH, cs.WRONG, cs.FOLLOW):
        for item in chats:
            own += [t["text"] for t in item[0]] + [item[1]]
    own += [t["text"] for t in cs.OWNER + cs.SODIUM]
    sums = [q for q, _, _ in ft.arithmetic_test()]
    quiz = [q[0] for q in fq.QUIZ]   # sentence stems: matched only in question-shaped sources (see below)
    return own, sums, quiz


OWN_MIN_WORDS = 6   # own test items of 6+ words are removed wherever they appear; shorter ones only by the close-match
                    # rule in question-shaped sources (chats, Stack Exchange, worked problems), which skips all-stopword
                    # items such as "how are you?"
PUBLIC = [   # (dataset, config, splits, fields that make an item's text)
    ("cais/mmlu", "all", ["test", "validation", "dev"], ["question", "choices"]),
    ("Rowan/hellaswag", None, ["validation", "test"], ["ctx", "endings"]),
    ("allenai/ai2_arc", "ARC-Easy", ["test", "validation"], ["question", "choices.text"]),
    ("allenai/ai2_arc", "ARC-Challenge", ["test", "validation"], ["question", "choices.text"]),
    ("openai/gsm8k", "main", ["test"], ["question", "answer"]),
    ("EleutherAI/hendrycks_math", None, ["test"], ["problem", "solution"]),
    ("ybisk/piqa", None, ["validation", "test"], ["goal", "sol1", "sol2"]),
    ("allenai/openbookqa", "main", ["test", "validation"], ["question_stem", "choices.text"]),
    ("tau/commonsense_qa", None, ["validation", "test"], ["question", "choices.text"]),
    ("allenai/winogrande", "winogrande_xl", ["validation", "test"], ["sentence", "option1", "option2"]),
    ("openai/openai_humaneval", None, ["test"], ["prompt", "canonical_solution"]),
    ("google-research-datasets/mbpp", "full", ["train", "validation", "test", "prompt"], ["text", "code"]),
]


def public_rows(repo, config, split, columns=None):
    """A test set's rows: the dataset's own parquet files, or Hugging Face's parquet copy (refs/convert/parquet)."""
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq
    api, fs = HfApi(), HfFileSystem()

    def is_split(path):
        parts = path.split("/")
        name = parts[-1]
        return split in parts[:-1] or re.match(rf"{re.escape(split)}([-_.]|$)", name) is not None

    for revision in (None, "refs/convert/parquet"):
        try:
            files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
        except Exception:
            continue
        files = [f for f in files if f.endswith(".parquet") and is_split(f) and
                 (config is None or config in f.split("/") or f.startswith(config + "/"))]
        if not files and config is None:
            continue
        if files:
            out = []
            rev = "" if revision is None else "@" + revision.replace("/", "%2F")
            for f in files:
                with fs.open(f"datasets/{repo}{rev}/{f}", "rb") as fh:
                    out += pq.read_table(fh, columns=columns).to_pylist()
            return out
    return []


def field(row, path):
    v = row
    for part in path.split("."):
        v = v.get(part) if isinstance(v, dict) else None
    return " ".join(map(str, v)) if isinstance(v, list) else str(v or "")


def build_decon(work):
    """work/decon.npz: sorted hashes of every test's word runs; work/decon_questions.json for near matches."""
    own, sums, quiz = test_items()
    lengths = collections.defaultdict(set)
    for q in own + sums:
        _, h = word_hashes(q)
        if len(h) >= OWN_MIN_WORDS:   # short items ("how are you?") would remove ordinary text; see OWN_MIN_WORDS
            lengths[len(h)].update(gram_hashes(h, len(h)).tolist())
    arrays = {f"own{n}": np.array(sorted(v), dtype=np.uint64) for n, v in lengths.items()}
    arrays["lengths"] = np.array(sorted(lengths), dtype=np.int64)
    public = {n: set() for n in range(8, 13)}
    grams13, owner, item_grams = {}, [], []   # every 13-word run of every item, and which item it came from
    report = {}
    for repo, config, splits, fields in PUBLIC:
        items = 0
        for split in splits:
            for row in public_rows(repo, config, split):
                _, h = word_hashes(" ".join(field(row, f) for f in fields))
                if len(h) >= 13:
                    runs = set(gram_hashes(h, 13).tolist())
                    for g in runs:
                        grams13.setdefault(g, len(item_grams))
                    item_grams.append(len(runs))
                elif len(h) >= 8:
                    public[len(h)].update(gram_hashes(h, len(h)).tolist())
                items += 1
        report[f"{repo} {config or ''}".strip()] = items
    for n, v in public.items():
        arrays[f"public{n}"] = np.array(sorted(v), dtype=np.uint64)
    keys = np.array(list(grams13), dtype=np.uint64)
    order = np.argsort(keys)
    arrays["public13"] = keys[order]
    arrays["public13_item"] = np.array(list(grams13.values()), dtype=np.int32)[order]
    arrays["item_grams"] = np.array(item_grams, dtype=np.int32)
    np.savez(work / "decon.npz", **arrays)
    questions = [{"text": q, "content": [w for w in WORDS.findall(q.lower()) if w not in STOP],
                  "length": len(WORDS.findall(q.lower()))} for q in own + sums + quiz]
    questions = [q for q in questions if len(q["content"]) >= 2]
    (work / "decon_questions.json").write_text(json.dumps(questions))
    return report


def setup(work):
    """The language model, the test lists, and PG-19's held-out book numbers."""
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "lid.176.bin").exists():
        urllib.request.urlretrieve(LID_URL, str(work / "lid.176.bin.part"))
        os.replace(work / "lid.176.bin.part", work / "lid.176.bin")
    if not (work / "decon.npz").exists():
        report = build_decon(work)
        print("test sets read (items): " + ", ".join(f"{k} {v:,}" for k, v in report.items()), flush=True)
        missing = [k for k, v in report.items() if not v]
        if missing:
            print(f"  WARNING: could not read {missing}; their questions are not removed. Tell the owner.", flush=True)
    if not (work / "pg19_held_out.json").exists():
        from huggingface_hub import hf_hub_download
        ids = set()
        for split in ("validation", "test"):   # data/<split>_files.txt: lines like "test/10146.txt" (Gutenberg numbers)
            path = hf_hub_download("deepmind/pg19", f"data/{split}_files.txt", repo_type="dataset")
            ids |= set(re.findall(r"(\d+)\.txt", Path(path).read_text()))
        (work / "pg19_held_out.json").write_text(json.dumps(sorted(ids)))
        print(f"PG-19 held-out books to leave out: {len(ids)}", flush=True)


# ----------------------------------------------------------------------------------------------- copies (dedup)

def dedup(work, rnd, shards):
    """Marks which documents of this round to keep: the first of every exact or near copy, across all rounds.
    shards: the clean parquet files of the round, already in priority order. Writes <shard>.keep.npy."""
    import pyarrow.parquet as pq
    seen_dir = work / "seen"
    seen_dir.mkdir(exist_ok=True)

    def seen(name):
        p = seen_dir / f"{name}.npy"
        return np.load(p) if p.exists() else np.zeros(0, dtype=np.uint64)
    cols = [pq.read_table(s, columns=["exact", "url", "bands"]) for s in shards]
    sizes = [len(c) for c in cols]
    if not sum(sizes):
        return {}
    exact = np.concatenate([c["exact"].to_numpy() for c in cols]).astype(np.uint64)
    url = np.concatenate([c["url"].to_numpy() for c in cols]).astype(np.uint64)
    raw = [b for c in cols for b in c["bands"].to_pylist()]
    has = np.array([len(b) == 112 for b in raw])
    bands = np.zeros((len(raw), 14), dtype=np.uint64)
    if has.any():
        bands[has] = np.frombuffer(b"".join(b for b in raw if len(b) == 112), dtype=np.uint64).reshape(-1, 14)
    adapter_of = np.concatenate([[Path(s).parent.name] * n for s, n in zip(shards, sizes)])
    drop = np.zeros(len(exact), dtype=bool)
    why = collections.Counter()

    def first_and_new(keys, name, mask):
        idx = np.flatnonzero(mask)
        k = keys[idx]
        old = member(k, seen(name))
        _, first = np.unique(k, return_index=True)
        later = np.ones(len(k), dtype=bool)
        later[first] = False
        bad = np.zeros(len(keys), dtype=bool)
        bad[idx] = old | later
        return bad
    d = first_and_new(exact, "exact", np.ones(len(exact), dtype=bool))
    why["exact copy"] += int(d.sum())
    drop |= d
    web = np.isin(adapter_of, ["dclm", "essential"]) & (url != 0)
    fw_urls = np.unique(url[(adapter_of == "fineweb") & (url != 0)])
    d = web & (member(url, seen("fineweb_urls")) | member(url, fw_urls)) & ~drop
    why["URL already in FineWeb-Edu"] += int(d.sum())
    drop |= d
    near = np.zeros(len(exact), dtype=bool)
    for b in range(14):
        near |= first_and_new(bands[:, b], f"band{b}", has & ~drop)
    near &= ~drop
    why["near copy"] += int(near.sum())
    drop |= near
    keep = ~drop
    np.save(seen_dir / "exact.npy", np.union1d(seen("exact"), exact[keep]))
    np.save(seen_dir / "fineweb_urls.npy", np.union1d(seen("fineweb_urls"), fw_urls))
    for b in range(14):
        np.save(seen_dir / f"band{b}.npy", np.union1d(seen(f"band{b}"), bands[keep & has, b]))
    at = 0
    for s, n in zip(shards, sizes):
        np.save(Path(s).with_suffix(".keep.npy"), keep[at:at + n])
        at += n
    return dict(why)


# --------------------------------------------------------------------------------------------- the word-piece list

TOKENIZER_SAMPLE = {   # data_mix.md "The word-piece list": shares of the sample, by output source
    "fineweb_edu": 21, "dclm_edu": 14, "cosmopedia": 7,
    "stack_python": 8, "stack_other": 10.5, "stack_html": 0.75, "kaggle": 0.75,
    "se_tech": 4, "docs": 2, "essential_cs": 2,
    "finemath_4": 6, "se_math": 2, "worked_maths": 2,
    "finewiki": 7, "simplewiki": 1, "gutenberg": 2, "doab": 2, "arxiv_abstracts": 1.5, "pes2o": 1.5,
    "smoltalk": 2.5, "wildchat": 1, "ultrachat": 1, "soda": 0.5,
}


def train_tokenizer(work, data, shards, total_bytes):
    """The 16,384-piece byte-level list, trained on a sample of the cleaned, deduplicated round-1 text."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    want = {k: total_bytes * v / sum(TOKENIZER_SAMPLE.values()) for k, v in TOKENIZER_SAMPLE.items()}
    got = collections.Counter()

    def sample():
        """The sample, streamed shard by shard (never all in memory; BPE counts do not depend on the order)."""
        for s in shards:
            keep = np.load(Path(s).with_suffix(".keep.npy"))
            for batch in pq.ParquetFile(s).iter_batches(batch_size=4096, columns=["text", "outs", "val"]):
                for text, outs, val in zip(batch["text"].to_pylist(), batch["outs"].to_pylist(), batch["val"].to_pylist()):
                    k = keep[sample.row]
                    sample.row += 1
                    if not k or val:
                        continue
                    for o in outs:
                        if o in want and got[o] < want[o]:
                            piece = text[:20_000]   # no single book dominates
                            got[o] += len(piece.encode())
                            yield piece
                            break
            sample.row = 0
    sample.row = 0
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(sample(), trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL, show_progress=False,
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    assert tok.token_to_id(SPECIAL[0]) == 0
    assert tok.get_vocab_size() == VOCAB or total_bytes < 100e6, f"only {tok.get_vocab_size()} pieces learned"
    tok.save(str(data / "tokenizer.json"))
    short = {k: f"{got[k] / max(1, want[k]):.0%}" for k in want if got[k] < 0.5 * want[k]}
    return {"sample MB": round(sum(got.values()) / 1e6, 1), "sources short of their share": short}


def check_tokenizer(tok, texts):
    """Exact round trip, digits one by one, code indentation kept."""
    bad = [t for t in texts if tok.decode(tok.encode(t).ids) != t]
    digits = tok.encode("In 2026, 12345 + 678 = 13023").tokens
    assert all(len(p.strip("Ġ")) <= 1 for p in digits if any(c.isdigit() for c in p)), digits
    code = "def f(x):\n    if x:\n        return 1\n\treturn 0\n"
    assert tok.decode(tok.encode(code).ids) == code
    return {"round trip failures": len(bad), "of": len(texts), "digits": " ".join(digits),
            "indentation pieces": [p for p in tok.encode(code).tokens if p.strip("ĠĊĉ") == ""][:6]}


# --------------------------------------------------------------------------------------------- cutting into pieces

def chunks(text):
    """A long document cut at paragraph breaks into parts of at most CHUNK_CHARS letters (the long phase)."""
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, cur = [], []
    size = 0
    for para in text.split("\n\n"):
        if size + len(para) > CHUNK_CHARS and cur:
            out.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(para[:CHUNK_CHARS])
        size += len(para) + 2
    if cur:
        out.append("\n\n".join(cur))
    return out


def cut_file(task):
    """One clean shard -> work/pieces/r<round>/<out>/<shard>.{train,val}.bin, one per output source."""
    shard, rnd, work, data = task
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok = Tokenizer.from_file(str(Path(data) / "tokenizer.json"))
    keep = np.load(Path(shard).with_suffix(".keep.npy"))
    stem = f"{Path(shard).parent.name}-{Path(shard).stem}"
    files, counts = {}, collections.defaultdict(lambda: [0, 0, 0])   # pieces, letters, documents
    at = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=1024, columns=["text", "outs", "val"]):
        per = collections.defaultdict(list)
        for text, outs, val in zip(batch["text"].to_pylist(), batch["outs"].to_pylist(), batch["val"].to_pylist()):
            if keep[at]:
                for o in outs:
                    per[(o, "val" if val else "train")].extend(chunks(text) if o.endswith("_long") or o in LONG_ONLY else [text])
            at += 1
        for key, texts in per.items():
            if key not in files:
                folder = Path(work) / "pieces" / f"r{rnd}" / key[0]
                folder.mkdir(parents=True, exist_ok=True)
                files[key] = open(folder / f"{stem}.{key[1]}.bin.tmp", "wb")
            c = counts[key]
            for text, e in zip(texts, tok.encode_batch(texts, add_special_tokens=False)):
                ids = np.asarray(e.ids + [END], dtype=np.uint16)
                ids.tofile(files[key])
                c[0] += len(ids)
                c[1] += len(text.encode())
                c[2] += 1
    made = []
    for (o, part), fh in files.items():
        fh.close()
        path = Path(fh.name[:-4])
        os.replace(fh.name, path)
        made.append({"out": o, "part": part, "path": str(path), "pieces": counts[(o, part)][0],
                     "letters": counts[(o, part)][1], "documents": counts[(o, part)][2]})
    return shard, made


class Manifest:
    """<data>/manifest.json, written whole each time (a temporary file renamed over it), by this process only."""
    def __init__(self, data):
        self.data, self.path = Path(data), Path(data) / "manifest.json"
        self.m = json.loads(self.path.read_text()) if self.path.exists() else \
            {"name": "SlingerGPT 492M training text", "pieces in the list": VOCAB, "end of text": END, "sources": {}}

    def add(self, made, val_ok):
        """Moves a finished piece file into place as the source's next NNNN.bin and lists it."""
        src = self.m["sources"].setdefault(made["out"], {"files": [], "complete": False,
                                                          "val": {"name": "val.bin", "pieces": 0, "letters": 0, "documents": 0}})
        folder = self.data / made["out"]
        folder.mkdir(parents=True, exist_ok=True)
        if made["part"] == "val":
            if val_ok:   # held-out text from round 1 only; later held-out documents are left out, not written
                with open(folder / "val.bin", "ab") as fh, open(made["path"], "rb") as part:
                    shutil.copyfileobj(part, fh)
                for k in ("pieces", "letters", "documents"):
                    src["val"][k] += made[k]
            os.remove(made["path"])
        elif made["pieces"]:
            name = f"{len(src['files']):04d}.bin"
            shutil.move(made["path"], folder / name)
            src["files"].append({"name": name, "pieces": made["pieces"], "letters": made["letters"],
                                 "documents": made["documents"]})
        else:
            os.remove(made["path"])
        self.save()

    def finish(self, names):
        for n in names:
            if n in self.m["sources"]:
                self.m["sources"][n]["complete"] = True
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.m, indent=1))
        os.replace(tmp, self.path)


# ----------------------------------------------------------------------------------------------------- the rounds

def plan_files(work, adapters, limit):
    """Each adapter's files per round, fixed once (work/files.json), so a restarted run continues the same plan."""
    path = work / ("files_sample.json" if limit else "files.json")
    if path.exists():
        return json.loads(path.read_text())
    plan = {}
    for a in adapters:
        try:
            files = list_files(a)
        except Exception as e:
            print(f"  {a}: could not list its files ({e}); skipped. Tell the owner.", flush=True)
            continue
        plan[a] = [files[:1]] + [[] for _ in range(ROUNDS - 1)] if limit else rounds_of(files)
    path.write_text(json.dumps(plan, indent=1))
    return plan


def run_round(rnd, plan, work, data, pool, limit, tokenizer_from, sample_mb, log):
    tasks = [(a, f, rnd, i, str(work), limit) for a, rounds in plan.items() for i, f in enumerate(rounds[rnd])]
    if not tasks:
        return False
    began = time.time()
    log(f"round {rnd + 1}: cleaning {len(tasks)} files")
    report = collections.defaultdict(lambda: {"kept": 0, "dropped": collections.Counter()})
    for a, f, info in pool.imap_unordered(clean_safely, tasks):
        report[a]["kept"] += info["kept"]
        report[a]["dropped"].update(info["dropped"])
        if "error" in info:
            report[a]["errors"] = report[a].get("errors", 0) + 1
            log(f"WARNING: {a} {f} failed: {info['error']}\n{info['trace']}")
    order = PRIORITY + sorted(set(plan) - set(PRIORITY))
    shards = sorted((work / "clean" / f"r{rnd}").glob("*/*.parquet"),
                    key=lambda p: (order.index(p.parent.name) if p.parent.name in order else 99, p.name))
    copies = dedup(work, rnd, [str(s) for s in shards])
    log(f"round {rnd + 1}: cleaned in {time.time() - began:.0f} s; copies removed: {copies}")
    log(json.dumps({a: {"kept": r["kept"], "dropped": dict(r["dropped"]), **({"errors": r["errors"]} if "errors" in r
                                                                             else {})} for a, r in report.items()}))
    if not (data / "tokenizer.json").exists():
        if tokenizer_from:
            shutil.copy(tokenizer_from, data / "tokenizer.json")
            log(f"word-piece list copied from {tokenizer_from}")
        else:
            t0 = time.time()
            info = train_tokenizer(work, data, [str(s) for s in shards], sample_mb * 1e6)
            log(f"word-piece list trained in {time.time() - t0:.0f} s: {info}")
    manifest = Manifest(data)
    for shard, made in pool.imap_unordered(cut_file, [(str(s), rnd, str(work), str(data)) for s in shards]):
        for m in made:
            manifest.add(m, val_ok=rnd == 0)
    manifest.finish(finished_after(plan, rnd) if not limit else list(manifest.m["sources"]))
    if not limit:   # the cleaned text of this round is in the piece files now
        shutil.rmtree(work / "clean" / f"r{rnd}", ignore_errors=True)
    log(f"round {rnd + 1} done in {time.time() - began:.0f} s")
    return True


ADAPTER_OUTS = {   # every output source each adapter can write (to know when an output source is complete)
    "fineweb": ["fineweb_edu", "fineweb_edu_4"], "dclm": ["dclm_edu", "dclm_edu_4"], "cosmopedia": ["cosmopedia"],
    "nemotron": ["nemotron_qa"], "stack_edu": ["stack_python", "stack_python_4", "stack_other", "stack_other_4", "code_repos"],
    "stack_html": ["stack_html"], "kaggle": ["kaggle"], "opencoder_qa": ["opencoder"], "opencoder_sn": ["opencoder"],
    "stackexchange": ["se_tech", "se_tech_top", "se_math", "se_math_top", "se_other", "se_other_top"],
    "essential": ["essential_cs"], "libretexts": ["docs", "docs_long"], "pressbooks": ["docs", "docs_long"],
    "peps": ["docs", "docs_long"], "archwiki": ["docs", "docs_long"], "docs_web": ["docs", "docs_long"],
    "finepdfs": ["finepdfs_cs", "finepdfs_long"], "finemath4": ["finemath_4"], "finemath3": ["finemath_3"],
    "infiwebmath": ["infiwebmath"], "orca": ["worked_maths"], "metamath": ["worked_maths"], "omi2": ["worked_maths"],
    "numina": ["worked_maths"], "tinygsm": ["worked_maths"], "gsm8k": ["worked_maths"],
    "finewiki": ["finewiki", "finewiki_long"], "simplewiki": ["simplewiki"], "gutenberg": ["gutenberg", "gutenberg_long"],
    "doab": ["doab", "doab_long"], "arxiv_abs": ["arxiv_abstracts"], "arxiv_papers": ["arxiv_papers"],
    "pes2o": ["pes2o", "pes2o_long"], "smoltalk": ["smoltalk"], "ultrachat": ["ultrachat"], "soda": ["soda"],
    "wildchat": ["wildchat"],
}


def finished_after(plan, rnd):
    """Output sources none of whose adapters has files left after round `rnd`."""
    feeding = collections.defaultdict(list)
    for a, outs in ADAPTER_OUTS.items():
        for o in outs:
            feeding[o].append(a)
    return [o for o, adapters in feeding.items()
            if all(not any(plan.get(a, [[]])[r] for r in range(rnd + 1, len(plan.get(a, [[]])))) for a in adapters)]


def tokenizer_here(data, work, sample_mb, workers, log):
    """The word-piece list from streamed samples of every source, without downloading whole files (so it can be built
    on any machine before the rental). Two passes: 1,000 rows of each file measure how much cleaned text each source
    gives per row; then each source reads enough rows to fill its share of a sample_mb sample."""
    import pyarrow.parquet as pq
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    feeding = collections.defaultdict(list)
    for a, outs in ADAPTER_OUTS.items():
        for o in outs:
            if o in TOKENIZER_SAMPLE:
                feeding[o].append(a)
    adapters = sorted({a for v in feeding.values() for a in v})
    nfiles = {a: (24 if a == "stack_edu" else 2) for a in adapters}   # Stack-Edu's files are one language each
    files = {a: list_files(a)[:nfiles[a]] for a in adapters}
    if "stackexchange" in files:   # one file from each of these sites, so maths and everyday questions are in the sample
        every = sorted(list_files("stackexchange"))
        sites = ["Stackoverflow.com", "math.stackexchange.com", "superuser.com", "askubuntu.com", "unix.stackexchange.com",
                 "physics.stackexchange.com", "english.stackexchange.com", "cooking.stackexchange.com"]
        files["stackexchange"] = [next(f for f in every if f"/{site}/" in f) for site in sites if any(f"/{site}/" in f for f in every)]
    total = sum(TOKENIZER_SAMPLE.values())
    want = {o: sample_mb * 1e6 * v / total for o, v in TOKENIZER_SAMPLE.items()}

    def clean_all(rnd, limits):
        tasks = [(a, f, rnd, i, str(work), limits[a]) for a in adapters for i, f in enumerate(files[a])]
        with get_context("spawn").Pool(workers, maxtasksperchild=2) as pool:
            for a, f, info in pool.imap_unordered(clean_safely, tasks):
                if "error" in info:
                    log(f"WARNING: {a} {f}: {info['error']}")
        return sorted((work / "clean" / f"r{rnd}").glob("*/*.parquet"))
    log(f"word-piece list: measuring {len(adapters)} sources")
    rate = collections.defaultdict(float)   # letters of cleaned text per row read, per (adapter, output)
    for shard in clean_all(90, {a: 500 for a in adapters}):
        a = shard.parent.name
        t = pq.read_table(shard, columns=["text", "outs"])
        for text, outs in zip(t["text"].to_pylist(), t["outs"].to_pylist()):
            for o in outs:
                rate[(a, o)] += len(text.encode()) / (500 * len(files[a]))
    limits = {}
    for o, adapters_of in feeding.items():
        per_row = sum(rate[(a, o)] for a in adapters_of)
        rows = math.ceil(1.3 * want[o] / per_row) if per_row else 0
        for a in adapters_of:
            limits[a] = max(limits.get(a, 1000), min(400_000, math.ceil(rows / len(files[a]))))
    log(f"word-piece list: reading {json.dumps(limits)} rows per file")
    shards = clean_all(91, limits)
    order = PRIORITY + sorted(set(adapters) - set(PRIORITY))
    shards.sort(key=lambda p: (order.index(p.parent.name) if p.parent.name in order else 99, p.name))
    dedup(work, 91, [str(x) for x in shards])
    t0 = time.time()
    info = train_tokenizer(work, data, [str(x) for x in shards], sample_mb * 1e6)
    log(f"word-piece list trained in {time.time() - t0:.0f} s: {info}")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))
    per, texts = collections.defaultdict(lambda: [0, 0]), []
    for shard in shards:   # letters per piece, on held-out documents of each source
        t = pq.read_table(shard, columns=["text", "outs", "val"])
        for text, outs, val in zip(t["text"].to_pylist(), t["outs"].to_pylist(), t["val"].to_pylist()):
            if val:
                texts.append(text)
                n = len(tok.encode(text, add_special_tokens=False).ids)
                for o in outs:
                    per[o][0] += len(text.encode())
                    per[o][1] += n
    log("letters per piece on held-out documents: " + ", ".join(f"{o} {b / max(1, n):.2f}" for o, (b, n) in sorted(per.items())))
    log(f"checks: {check_tokenizer(tok, texts)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["sample", "run", "status", "tokenizer"])
    p.add_argument("--data", default="/data/492m")
    p.add_argument("--work", default="", help="downloads and intermediate files (default <data>/work)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 16))
    p.add_argument("--docs", type=int, default=300, help="sample: documents read from each source")
    p.add_argument("--only", default="", help="comma-separated adapters (default: all)")
    p.add_argument("--tokenizer", default="", help="use this word-piece list instead of training one")
    p.add_argument("--sample-mb", type=float, default=3000, help="size of the word-piece list's training sample")
    p.add_argument("--approved", action="store_true", help="the owner approved the data list and behaviour filters")
    a = p.parse_args()
    data = Path(a.data)
    work = Path(a.work) if a.work else data / "work"
    if a.what == "status":
        m = json.loads((data / "manifest.json").read_text())
        for k, v in sorted(m["sources"].items()):
            pieces = sum(f["pieces"] for f in v["files"])
            print(f"{k:18s} {pieces / 1e9:8.3f}B pieces in {len(v['files'])} files"
                  f"{' (complete)' if v['complete'] else ''}; held out {v['val']['pieces'] / 1e6:.1f}M")
        return
    if a.what == "run" and not a.approved:
        sys.exit("The owner has not approved the data list, the chat layout and the behaviour filters yet "
                 "(TRAINING_PLAN.md section 3). Run with --approved once they have.")
    data.mkdir(parents=True, exist_ok=True)
    logfile = data / "prepare.log"

    def log(line):
        line = f"{time.strftime('%H:%M:%S')} {line}"
        print(line, flush=True)
        with logfile.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")   # fast downloads (huggingface_hub 1.x and later: Xet)
    setup(work)
    if a.what == "tokenizer":
        tokenizer_here(data, work, a.sample_mb, a.workers, log)
        return
    adapters = [x for x in (a.only.split(",") if a.only else ADAPTERS)]
    limit = a.docs if a.what == "sample" else 0
    plan = plan_files(work, adapters, limit)
    with get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
        for rnd in range(ROUNDS):
            run_round(rnd, plan, work, data, pool, limit, a.tokenizer,
                      a.sample_mb if not limit else min(a.sample_mb, 50), log)
            if limit:
                break
    log("all rounds done" if not limit else "sample done")


if __name__ == "__main__":
    main()
