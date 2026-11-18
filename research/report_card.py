"""Report card: the normal way and the new way on the 124M model, trained to the same score, then quizzed on facts.

    python -P research/report_card.py [--logs DIR] [--out FILE.pdf]

Reads what research/facts_quiz.py and research/compare_text.py left in research/logs/: the two training runs
(lab.jsonl, stage facts@...), the GPU log (final_test.log), the quiz answers (facts_quiz.json), how often each fact
appears in the training text (facts_counts.json) and the longer answers (compare_text_facts_124m_normal.md).
Writes research/report_card_124m.pdf by printing research/logs/report_card_124m.html with Microsoft Edge.
The conclusion paragraphs (CONCLUSION) are written by hand after reading the results.
"""
import argparse, html, json, math, re, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.argv, ARGV = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, str(HERE))
import facts_quiz as fq

QUIZ = fq.QUIZ
SUBJECTS = [("Capital cities", 0, 30), ("Famous places", 30, 43), ("Languages and money", 43, 48),
            ("Dates in history", 48, 56), ("Famous people and their work", 56, 79), ("Science", 79, 101),
            ("Animals and everyday facts", 101, 117)]
MODELS = {"facts_124m_normal": ("Normal way", "124m-gpt2-1g"), "facts_124m_new": ("New way", "124m-bpe4096-1g")}
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def print_pdf(page, out, wait=240):
    """Prints the HTML file `page` to the PDF `out` with Edge and waits until the new file is complete. With Edge
    already open on the PC, msedge.exe hands the job over and returns at once; the temporary profile was then
    deleted before Edge had read anything, and the old PDF stayed in place (16 November)."""
    out = Path(out)
    before = out.stat().st_mtime if out.exists() else 0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                        f"--print-to-pdf={out}", Path(page).resolve().as_uri()], check=True, timeout=wait)
        size, end = -1, time.time() + wait
        while time.time() < end:
            now = out.stat().st_size if out.exists() and out.stat().st_mtime > before else -1
            if now > 0 and now == size:   # new, and no longer growing
                return
            size = now
            time.sleep(1)
    raise RuntimeError(f"Edge did not write {out} within {wait} s")


CONCLUSION = [   # written after reading the results of 14 November 2026
    "<b>Speed: the new way reached the normal way's level in 19 minutes instead of 60, about 3.1 times faster.</b> It needed "
    "2.3 times less text (219 million letters against 514 million) and got through text 1.3 times faster (about 188,000 "
    "letters a second against 143,000): 2.3 x 1.3 = 3.1.",
    "<b>Facts: at the same level, the two models know about the same.</b> Picking from four answers, the normal way got 41 of "
    "117 right and the new way 37. Only 24 questions split them, 14 for the normal way and 10 for the new way, and a split "
    "like that happens by chance about half the time (54%). The quiz cannot tell them apart. In their own words each got 4 right.",
    "<b>Both models still know very few facts.</b> Guessing would get about 29 of 117; they got 41 and 37. In their own words "
    "they almost never give an answer: after \"The capital of France is\" both write something like \"the capital of the "
    "world\", and neither says Paris. One hour is about 6% of a full training of this model on your data, and facts usually "
    "come later in training. So this quiz could only have shown a large difference, and there was none.",
    "<b>Made-up answers.</b> Each model named one of the listed wrong answers only 3 times in its own words. At this stage "
    "neither states many specific wrong facts; mostly they go round in circles without answering (section 6). About a fifth "
    "of their longer answers is repeated phrases (21% for the normal way, 18% for the new way).",
    "<b>Rarely seen facts may be where reading less text shows.</b> On the half of the facts that appear least in the text, "
    "the normal way picked 22 right and the new way 17 (only one of them right: 8 against 3). On the more common half it was "
    "19 against 20. That fits the worry that a model which reads less text sees rare facts less often, but a split of 8 "
    "against 3 happens by chance about 23% of the time, so it is a hint, not a finding.",
    "<b>The new way's lead shrinks as training goes on.</b> On this model it reached the level the normal way had after 7 "
    "minutes 4.9 times sooner, the level after 22 minutes 3.7 times sooner, and the level after 60 minutes 3.1 times sooner. "
    "For a full training the gain will probably be smaller than 3.1 times; how much smaller has not been measured.",
]
e = html.escape


def manifest(size):
    return json.loads((ROOT / "data" / "pieces1g" / size.split("-")[1] / "manifest.json").read_text())


def load(logs):
    runs = [json.loads(l) for l in (logs / "lab.jsonl").open(encoding="utf-8")]
    out = {"quiz": json.loads((logs / "facts_quiz.json").read_text(encoding="utf-8")),
           "counts": json.loads((logs / "facts_counts.json").read_text(encoding="utf-8")), "runs": {}, "gpu": {}}
    for name, (label, size) in MODELS.items():
        out["runs"][name] = [r for r in runs if r["stage"] == f"facts@{size}" and "final" in r and r["config"].get("save") == name][-1]
    for line in (logs / "final_test.log").read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\S+) facts (\S+) (\d+) rounds (\{.*\}): (\d+) s start to finish, exit 0, started at (\d+) C, hottest (\d+) C", line)
        if m and json.loads(m[4]).get("save") in MODELS:
            out["gpu"][json.loads(m[4])["save"]] = {"ended": m[1], "wall": int(m[5]), "start_C": int(m[6]), "hottest_C": int(m[7])}
    answers = {}
    current = None
    for line in (logs / "compare_text_facts_124m_normal.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            answers[current] = []
        m = re.match(r"- \*\*(\S+), (steady|varied):\*\* (.*)", line)
        if m and current:
            answers[current].append((m[1], m[2], m[3].strip()))
    out["long"] = answers
    return out


def own_words(item, said):
    """right / named one of the listed wrong answers / something else."""
    prompt, answer, wrong, _ = item
    low = said.lower()
    if re.search(r"\b" + re.escape(answer.lower()) + r"\b", low):
        return "right"
    named = [w for w in wrong if re.search(r"\b" + re.escape(w.lower()) + r"\b", low)]
    return f"named {named[0]}" if named else "other"


def repeated_share(text):
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def sign_test(a, b):
    """Two-sided chance of a split at least this uneven, if each of a + b disagreements were a coin toss."""
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def table(head, rows, cls=""):
    out = f"<table class='{cls}'><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
    return out + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</table>"


def chart(d, target):
    curves = {}
    for name, (label, size) in MODELS.items():
        per = manifest(size)["letters per piece, val"]
        curves[name] = [(c[2], c[1] / per) for c in d["runs"][name]["curve"]]
    tmax = max(t for c in curves.values() for t, _ in c) * 1.03
    vals = [v for c in curves.values() for _, v in c[1:]]
    lo, hi = math.floor((min(vals) - 0.03) * 10) / 10, math.ceil(max(vals) * 10) / 10
    x0, x1, y0, y1 = 64, 690, 34, 300
    X = lambda t: x0 + (x1 - x0) * t / tmax
    Y = lambda v: y1 - (y1 - y0) * (min(v, hi) - lo) / (hi - lo)
    step = 0.1 if hi - lo <= 0.8 else 0.2
    grid = "".join(f"<line x1='{x0}' x2='{x1}' y1='{Y(lo + i * step):.1f}' y2='{Y(lo + i * step):.1f}' stroke='#e3e5e8'/>"
                   f"<text x='{x0 - 8}' y='{Y(lo + i * step) + 4:.1f}' text-anchor='end' class='fs'>{lo + i * step:.1f}</text>"
                   for i in range(int(round((hi - lo) / step)) + 1))
    ticks = "".join(f"<line x1='{X(m * 60):.1f}' x2='{X(m * 60):.1f}' y1='{y1}' y2='{y1 + 5}' stroke='#444'/>"
                    f"<text x='{X(m * 60):.1f}' y='{y1 + 19}' text-anchor='middle' class='fs'>{m}</text>"
                    for m in range(0, int(tmax / 60) + 1, 10))
    colour = {"facts_124m_normal": "#8a8f98", "facts_124m_new": "#2f6fde"}
    lines = "".join(f"<polyline fill='none' stroke='{colour[n]}' stroke-width='2.5' points='"
                    + " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in c if v <= hi) + "'/>" for n, c in curves.items())
    tn = d["runs"]["facts_124m_new"]["seconds"]
    return f"""<svg viewBox='0 0 720 360' role='img'>
<text x='0' y='16' class='ft'>How wrong each model was as training went on</text>{grid}{ticks}
<line x1='{x0}' x2='{x1}' y1='{y1}' y2='{y1}' stroke='#444'/><line x1='{x0}' x2='{x0}' y1='{y0}' y2='{y1}' stroke='#444'/>
<line x1='{x0}' x2='{x1}' y1='{Y(target):.1f}' y2='{Y(target):.1f}' stroke='#444' stroke-dasharray='5 4'/>
<text x='{x1}' y='{Y(target) + 16:.1f}' text-anchor='end' class='fs'>dashed line: the normal way's final score ({target:.3f})</text>
{lines}
<text x='{X(curves['facts_124m_normal'][5][0]) + 6:.1f}' y='{Y(curves['facts_124m_normal'][5][1]) - 12:.1f}' class='fl' fill='#5c6068'>Normal way</text>
<text x='{X(tn) + 10:.1f}' y='{min(Y(curves['facts_124m_new'][-1][1]) - 8, y1 - 14):.1f}' class='fl' fill='#2f6fde'>New way: stopped at {tn / 60:.1f} minutes</text>
<text x='{(x0 + x1) / 2}' y='{y1 + 42}' text-anchor='middle' class='fs'>minutes of training</text>
<text x='14' y='{(y0 + y1) / 2}' text-anchor='middle' class='fs' transform='rotate(-90 14 {(y0 + y1) / 2})'>wrongness per letter (lower is better)</text></svg>"""


def build(d):
    quiz, counts = d["quiz"], d["counts"]["counts"]
    n, w = quiz["facts_124m_normal"], quiz["facts_124m_new"]
    info = {}
    for name, (label, size) in MODELS.items():
        r, m = d["runs"][name], manifest(size)
        trained = r.get("tokens_trained", r["tokens"])
        letters = trained * m["letters per piece, train"]
        info[name] = dict(label=label, r=r, m=m, letters=letters, share=letters / d["counts"]["letters"], trained=trained,
                          score=r["final"] / m["letters per piece, val"], updates=trained // 65536,
                          choice=sum(a["choice"] for a in quiz[name]), free=sum(a["free"] for a in quiz[name]),
                          kinds=[own_words(item, a["said"]) for item, a in zip(QUIZ, quiz[name])], gpu=d["gpu"].get(name, {}))
    N, NW = info["facts_124m_normal"], info["facts_124m_new"]
    gain = N["r"]["seconds"] / NW["r"]["seconds"]
    T = len(QUIZ)
    pct = lambda k: f"{k}/{T} ({100 * k / T:.0f}%)"
    only = lambda kind: (sum(a[kind] and not b[kind] for a, b in zip(n, w)), sum(b[kind] and not a[kind] for a, b in zip(n, w)))
    oc, of = only("choice"), only("free")
    middle = sorted(counts.values())[len(counts) // 2]
    common = [i for i, q in enumerate(QUIZ) if counts[q[0]] > middle]
    rare = [i for i, q in enumerate(QUIZ) if counts[q[0]] <= middle]
    made_up = {k: sum(x.startswith("named") for x in v["kinds"]) for k, v in info.items()}
    other = {k: sum(x == "other" for x in v["kinds"]) for k, v in info.items()}

    subject_rows = [(s, f"{sum(n[i]['choice'] for i in range(a, b))}/{b - a}", f"{sum(w[i]['choice'] for i in range(a, b))}/{b - a}",
                     f"{sum(n[i]['free'] for i in range(a, b))}/{b - a}", f"{sum(w[i]['free'] for i in range(a, b))}/{b - a}")
                    for s, a, b in SUBJECTS]
    card = table(["", "Normal way", "New way"], [
        ("Training time", f"{N['r']['seconds'] / 60:.0f} minutes", f"<b>{NW['r']['seconds'] / 60:.1f} minutes</b> ({gain:.1f} times less)"),
        ("Text read", f"{N['letters'] / 1e6:.0f} million letters ({100 * N['share']:.0f}% of the text)",
         f"{NW['letters'] / 1e6:.0f} million letters ({100 * NW['share']:.0f}% of the text)"),
        ("Final score: wrongness per letter on unseen text (lower is better)", f"{N['score']:.3f}", f"{NW['score']:.3f}"),
        ("Fact quiz, picking from four answers (guessing gets 25%)", pct(N["choice"]), pct(NW["choice"])),
        ("Fact quiz, answering in its own words", pct(N["free"]), pct(NW["free"])),
        ("Own words: named one of the wrong answers instead", f"{made_up['facts_124m_normal']}", f"{made_up['facts_124m_new']}"),
        ("Own words: said something else (no answer, or another wrong one)", f"{other['facts_124m_normal']}", f"{other['facts_124m_new']}")])
    subjects = table(["Subject", "Picking: normal", "Picking: new", "Own words: normal", "Own words: new"], subject_rows)

    ex = lambda name, kind_test, k=8: [(e(QUIZ[i][0]), e(QUIZ[i][1]), e(quiz[name][i]["said"][:fq.FREE_LETTERS]))
                                       for i, x in enumerate(info[name]["kinds"]) if kind_test(x)][:k]
    problem_tables = ""
    for name in MODELS:
        rows = ex(name, lambda x: x.startswith("named"))
        problem_tables += f"<h3>{info[name]['label']}: wrong answers it gave in its own words</h3>"
        problem_tables += table(["Question", "True answer", "What it said (first 40 letters)"], rows, "small") if rows else "<p>None.</p>"
        rows = ex(name, lambda x: x == "other", 6)
        problem_tables += f"<p>Examples where it gave no clear answer or another wrong one:</p>"
        problem_tables += table(["Question", "True answer", "What it said"], rows, "small") if rows else "<p>None.</p>"

    rep = {}
    for name in MODELS:
        texts = [t for p in d["long"].values() for (who, mode, t) in p if who == name]
        rep[name] = sum(map(repeated_share, texts)) / max(1, len(texts))
    long_rows = ""
    for prompt, items in d["long"].items():
        long_rows += f"<h3>{e(prompt)}</h3>" + table(["Model", "How it chose words", "What it wrote"],
                                                    [(MODELS[who][0], "always the likeliest" if mode == "steady" else "some randomness", e(t))
                                                     for mode_order in ("steady", "varied") for (who, mode, t) in items if mode == mode_order], "small")

    letters_total = d["counts"]["letters"]
    seen = lambda name, i: counts[QUIZ[i][0]] * info[name]["share"]
    appendix = table(["Question", "True answer", "Times in the text", "Normal: picked", "New: picked", "Normal: own words", "New: own words"],
                     [(e(q[0]), e(q[1]), f"{counts[q[0]]:,}",
                       "✓" if n[i]["choice"] else "✗ " + e(n[i]["picked"]), "✓" if w[i]["choice"] else "✗ " + e(w[i]["picked"]),
                       ("✓ " if n[i]["free"] else "✗ ") + e(n[i]["said"][:fq.FREE_LETTERS]), ("✓ " if w[i]["free"] else "✗ ") + e(w[i]["said"][:fq.FREE_LETTERS]))
                      for i, q in enumerate(QUIZ)], "tiny")

    settings = table(["Setting", "Normal way", "New way"], [
        ("Model size", "12 layers, each working on lists of 768 numbers, 12 attention heads", "the same"),
        ("Weights", f"{N['r']['parameters_M']:.1f} million (38.6 million of them the piece table)",
         f"{NW['r']['parameters_M']:.1f} million (3.1 million piece table, 8.4 million lookup memory)"),
        ("How the text is cut", "GPT-2's list of 50,257 pieces, about 4.2 letters each", "our list of 4,096 pieces, about 3.4 letters each, built from this text"),
        ("Word order", "learned position table (GPT-2)", "rotary positions (Change 2)"),
        ("Attention", "standard", "each question and label rescaled first, QK-norm (Change 5)"),
        ("Feed-forward part", "GELU, 3,072 wide", "gated (SwiGLU), 2,048 wide (Change 4)"),
        ("Lookup memory", "none", "16,381 rows x 4 x 2 tables of 64 numbers, before block 2 (Change 3)"),
        ("Layer normalisation", "LayerNorm with biases", "the same"),
        ("Method that corrects the weights", "AdamW (0.9, 0.95), weight decay 0.1",
         "Muon for the block matrices (momentum 0.95, 5 Newton-Schulz steps, decay 0.1), AdamW for the rest (Change 1)"),
        ("Learning rate (peak)", "0.0006, gpt128.py's own (0.0012 was tried and was worse)", "0.002; lookup tables 0.01"),
        ("Learning-rate schedule", "rises over the first 10% of updates, then falls along a cosine to 10% of the peak", "the same"),
        ("Pieces per weight update", "65,536: 4 slices of 16 windows of 1,024 pieces", "the same"),
        ("Weight updates", f"{N['updates']:,}", f"{NW['updates']:,}"),
        ("Gradient clipping", "at 1.0", "the same"),
        ("Number format", "bfloat16 with TF32, model body compiled (torch.compile)", "the same"),
        ("Scoring the last step", "all 50,257 pieces, in slices (research/fused_ce.py)", "all 4,096 pieces"),
        ("Graphics memory, peak", f"{N['r']['peak_GB']:.1f} GB", f"{NW['r']['peak_GB']:.1f} GB")])

    trained = table(["", "Normal way", "New way"], [
        ("Seconds of training (stopwatch, excluding the 20 score checks)", f"{N['r']['seconds']:,.0f}", f"{NW['r']['seconds']:,.0f}"),
        ("Start to finish, including start-up and score checks", f"{N['gpu'].get('wall', 0):,} s", f"{NW['gpu'].get('wall', 0):,} s"),
        ("Pieces read a second", f"{N['r']['tokens_per_s']:,}", f"{NW['r']['tokens_per_s']:,}"),
        ("Letters read a second", f"{N['r']['tokens_per_s'] * N['m']['letters per piece, train']:,.0f}",
         f"{NW['r']['tokens_per_s'] * NW['m']['letters per piece, train']:,.0f}"),
        ("Pieces read in total", f"{N['trained'] / 1e6:.0f} million", f"{NW['trained'] / 1e6:.0f} million"),
        ("GPU temperature at the start, hottest", f"{N['gpu'].get('start_C', '?')} C, {N['gpu'].get('hottest_C', '?')} C",
         f"{NW['gpu'].get('start_C', '?')} C, {NW['gpu'].get('hottest_C', '?')} C")])

    freq_rows = []
    for label, idx in ((f"Seen more than {middle} times in the text", common), (f"Seen {middle} times or fewer", rare)):
        freq_rows.append((label, len(idx), f"{sum(n[i]['choice'] for i in idx)}", f"{sum(w[i]['choice'] for i in idx)}",
                          f"{sum(n[i]['free'] for i in idx)}", f"{sum(w[i]['free'] for i in idx)}"))
    freq = table(["Facts", "How many", "Picking: normal", "Picking: new", "Own words: normal", "Own words: new"], freq_rows)
    rare_seen = sum(seen("facts_124m_new", i) for i in rare) / len(rare)
    rare_seen_n = sum(seen("facts_124m_normal", i) for i in rare) / len(rare)

    body = f"""
<h1>Report card: the normal way and the new way, 124M model</h1>
<p class='sub'>Trained and tested on 14 November 2026 on this PC (RTX 5070 Ti Laptop GPU, 12 GB). Both models learned from the
same 990 MB of text and were stopped at the same score, then took the same 117-question fact quiz.</p>
<div class='answer'><b>The new way reached the same score in {NW['r']['seconds'] / 60:.1f} minutes; the normal way took {N['r']['seconds'] / 60:.0f}
({gain:.1f} times less time).</b> On the fact quiz the normal way got {N['choice']} of {T} right and the new way {NW['choice']}
when picking from four answers; in their own words, {N['free']} and {NW['free']}.</div>
{card}
<h2>Marks by subject</h2>
<p>Number right in each subject. "Picking" is choosing the most likely of four answers; "own words" is finishing the sentence
by itself.</p>
{subjects}

<h2>1. What was tested, in plain words</h2>
<p><b>The two ways.</b> The <b>normal way</b> is how your <code>gpt128.py</code> trains: GPT-2's design, the AdamW method of
correcting the weights, and GPT-2's list of 50,257 text pieces. The <b>new way</b> is everything developed in this project:
five changes to the design and the correcting method (Changes 1 to 5 in the earlier PDF) plus our own list of 4,096 pieces
(Change 6). Both models are the same size: 12 layers working on lists of 768 numbers.</p>
<p><b>How they were trained.</b> The normal way trained for about an hour. The new way was then trained only until it reached
the same score. The score is the usual measure of a language model: it reads text it never saw and guesses each next piece;
wrongness is how far off those guesses are, on average, per letter of text. Lower is better.</p>
<p><b>The question.</b> The new way gets to the same score after reading much less text. Does it still know the same facts?
A model can only learn a fact from text that contains it, so a model that reads less text could miss facts.</p>
<p><b>The quiz.</b> 117 sentences to finish, each with one true answer and three wrong ones of the same kind, for example
"The capital of Australia is" with Canberra, Sydney, Melbourne and Perth. Every question is asked two ways:</p>
<ul><li><b>Picking:</b> the model rates how likely each of the four answers is after the sentence; it is marked right when
the true answer is its top pick. Guessing would get about 29 of 117 (25%).</li>
<li><b>Own words:</b> the model finishes the sentence by itself, always taking its likeliest next piece. It is marked right when
the true answer appears in its first {fq.FREE_LETTERS} letters. This is the harder test, and the one that shows made-up answers.</li></ul>

<h2>2. How they trained</h2>
<p>The text: the first 990 MB of the project's training text (<code>data/corpus/mix_v2.txt</code>), the same for both. It is about
40% educational web pages (FineWeb-Edu), 15% children's stories (TinyStories), 12.5% old books (Project Gutenberg), 12.5% everyday
dialogues (SODA) and 20% an earlier mix of stories, books, dialogues and Simple English Wikipedia. Most facts are in the web pages
and the Wikipedia part. One megabyte in every hundred was held back and never trained on; the scores are measured on it.</p>
<p>The normal way was given about an hour ({N['updates']:,} weight updates). The new way was set up for a longer run, its score
on the held-back text was checked every 2% of the way, and it stopped the first time it was as good as the normal way's final
score, after {NW['updates']:,} weight updates. That moment's model took the quiz. Stopping partway through a longer plan is slightly
unfavourable to the new way: a run planned to end there lowers its learning rate sooner and gets there a little faster.</p>
<p>One earlier attempt at the new way's run was stopped by hand after 17 minutes and thrown away, because it was not saving the
model along the way. It is not counted anywhere in this report.</p>
{trained}
<figure>{chart(d, N['score'])}</figure>

<h2>3. Settings</h2>
{settings}
<p>The other parts are identical: the same 12 x 768 body, the same text, the same pieces per update, the same schedule shape,
the same code (<code>research/lab.py</code>), the same graphics card, one run at a time.</p>

<h2>4. How the marks are worked out</h2>
<p><b>Wrongness per letter.</b> At each position of the held-back text the model gives the true next piece a probability
<code>p</code>; wrongness there is <code>-ln(p)</code>. The average over all positions is divided by the average letters per piece
({N['m']['letters per piece, val']:.4f} for GPT-2's list, {NW['m']['letters per piece, val']:.4f} for ours), so the two lists are comparable.</p>
<p><b>Picking.</b> For each answer <code>a</code>, the model's log-probabilities of the pieces of " a" after the sentence are added up
and divided by the letters in " a", so a long answer such as "Alexander Graham Bell" is not marked down for its length:</p>
<pre>rating(a) = [ ln p(piece 1 of a | sentence) + ln p(piece 2 of a | sentence, piece 1) + ... ] / letters in " a"
right if rating(true answer) is the highest of the four</pre>
<p><b>Own words.</b> The model writes {fq.FREE_LETTERS} letters after the sentence, each time taking its likeliest piece. Right if the true
answer appears there as a whole word, ignoring capitals.</p>
<p><b>Is a difference real?</b> Only questions where one model is right and the other wrong can tell them apart. If the two were
equally good, those questions would split like coin tosses.</p>
<pre>picking:    only the normal way right {oc[0]}, only the new way right {oc[1]}; a split this uneven by chance: {100 * sign_test(*oc):.0f}%
own words:  only the normal way right {of[0]}, only the new way right {of[1]}; a split this uneven by chance: {100 * sign_test(*of):.0f}%</pre>
<p>A chance above about 5% means the quiz cannot tell the two apart.</p>

<h2>5. Facts seen often and facts seen rarely</h2>
<p>For each fact I counted how often its subject and its true answer appear within 300 letters of each other in the 990 MB of
text (for example "Australia" near "Canberra"). The count is approximate: a 0 can mean the text words the fact differently. The
normal way read about {100 * N['share']:.0f}% of the text and the new way about {100 * NW['share']:.0f}%, so on the rarer half of the facts
the normal way saw each one about {rare_seen_n:.1f} times on average and the new way about {rare_seen:.1f} times.</p>
{freq}

<h2>6. Problems each model had</h2>
<p>Out of {T} questions answered in their own words, the normal way named one of the listed wrong answers {made_up['facts_124m_normal']} times and
the new way {made_up['facts_124m_new']} times; these are made-up facts stated as if true. They gave no clear answer, or some other wrong
one, {other['facts_124m_normal']} and {other['facts_124m_new']} times.</p>
<p>In the longer answers below (section 9), the share of four-word phrases that repeat earlier ones was
{100 * rep['facts_124m_normal']:.0f}% for the normal way and {100 * rep['facts_124m_new']:.0f}% for the new way: both fall into loops,
as models trained this briefly do.</p>
{problem_tables}

<h2>7. Conclusion</h2>
{"".join(f"<p>{p}</p>" for p in CONCLUSION)}

<h2>8. What this test does not show</h2>
<ul><li><b>One training run each.</b> A second run from a different random start would move the quiz marks by a few questions.</li>
<li><b>About an hour of training is about 6% of a full training</b> of this model on your data. Both models are still weak; the
quiz compares them at an early stage.</li>
<li><b>The quiz is 117 questions I wrote</b>, on common facts. A different set could give different marks.</li>
<li><b>The counts of how often a fact appears are approximate</b> (subject and answer close together).</li></ul>

<h2>9. Their longer answers</h2>
<p>Each model continued eight prompts for 300 letters, once always taking its likeliest next piece and once with some randomness
(the same random seed for both).</p>
{long_rows}

<h2>10. How to repeat it</h2>
<pre>python -P research/prepare_more_text.py 1000     # 990 MB of the text, cut with both lists (about 2 minutes)
python -P research/facts_quiz.py count           # how often each fact appears (about 10 minutes)
python -P research/facts_quiz.py train           # the normal way for an hour, then the new way to the same score
python -P research/facts_quiz.py quiz            # both models answer all 117 questions
python -P research/compare_text.py facts_124m_normal facts_124m_new
python -P research/report_card.py                # this report</pre>

<h2>Appendix: every question and both models' answers</h2>
<p>✓ right, ✗ wrong (with the answer it picked or what it said).</p>
{appendix}
"""
    return body


CSS = """@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.5 'Segoe UI', Arial, sans-serif; color: #1c1e21; }
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 6pt; } h2 { font-size: 13.5pt; margin: 20pt 0 6pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
.sub { color: #5c6068; margin-top: 0; }
.answer { border: 1.5pt solid #2f6fde; background: #eef3fd; padding: 10pt 12pt; margin: 12pt 0; font-size: 12pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; }
tr { break-inside: avoid; }
th, td { border: 0.75pt solid #b9bdc4; padding: 4pt 7pt; text-align: left; vertical-align: top; }
th { background: #f0f1f3; }
table.small { font-size: 9pt; } table.tiny { font-size: 7.6pt; } table.tiny td, table.tiny th { padding: 2pt 4pt; }
pre { background: #f4f5f6; border: 0.75pt solid #d9dbdf; padding: 7pt 9pt; font: 8.6pt/1.45 Consolas, 'Courier New', monospace; white-space: pre-wrap; break-inside: avoid; }
code { font: 9.3pt Consolas, 'Courier New', monospace; background: #f0f1f3; padding: 0 2pt; }
figure { margin: 12pt 0; break-inside: avoid; } svg { width: 100%; height: auto; }
.ft { font: 600 14px 'Segoe UI', Arial, sans-serif; fill: #1c1e21; } .fl { font: 13px 'Segoe UI', Arial, sans-serif; fill: #1c1e21; }
.fs { font: 11.5px 'Segoe UI', Arial, sans-serif; fill: #3d4047; } li { margin-bottom: 3pt; }"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=str(HERE / "logs"))
    p.add_argument("--out", default=str(HERE / "report_card_124m.pdf"))
    a = p.parse_args(ARGV)
    page = Path(a.out).with_suffix(".html")
    page = HERE / "logs" / page.name if Path(a.out).parent == HERE else page
    page.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Report card, 124M model</title>"
                    f"<style>{CSS}</style></head><body>{build(load(Path(a.logs)))}</body></html>", encoding="utf-8")
    print_pdf(page, a.out)
    print(f"wrote {a.out} from {page}")


if __name__ == "__main__":
    main()
