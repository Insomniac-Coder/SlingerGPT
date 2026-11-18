"""SlingerGPT 1 pico Preview 1: the main training plus a 30-minute conversation fine-tune.

models/slingergpt1_pico.pt -> models/slingergpt1_pico_preview1.pt

    python -P research/finetune_slinger.py prepare     conversations with a mark on every Assistant piece (processor)
    python -P research/finetune_slinger.py train [--steps N]

What changes from the main training:
  - It learns from the Assistant's replies only. Every piece of a reply (and the end-of-text after the last one)
    is marked; the System and User lines and the role names are read but not marked, so they count for nothing.
  - 80% of the windows are smol-smoltalk conversations, 20% general text (top-rated educational pages and
    textbooks, every piece marked), so what it knows does not fade.
  - A lower learning rate: peak 0.0004 (the main training peaked at 0.002), up over 100 steps, then down along a
    cosine to 10%. Fresh optimiser state. 8-bit block multiplies, 65,536 pieces per step, as before.
  - The conversations held out by prepare_slinger.py (every 100th) stay held out; scores are on those.
Default length: 1,700 steps, about 30 minutes at about 62,000 pieces a second.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
OUT = ROOT / "data" / "slinger_chat"
CTX, MICRO, ACCUM, LR, WARM = 1024, 16, 4, 0.0004, 100
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def prepare():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "data" / "slinger" / "tokenizer.json"))
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    OUT.mkdir(parents=True, exist_ok=True)
    count, sizes = 0, {"train": 0, "val": 0}
    files = {p: (open(OUT / f"{p}.bin", "wb"), open(OUT / f"{p}_mask.bin", "wb")) for p in ("train", "val")}
    for f in sorted((ROOT / "data" / "raw_slinger" / "smol-smoltalk").glob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=2048, columns=["messages"]):
            for row in batch.to_pylist():
                ms = row["messages"]
                text = "\n".join(f"{names.get(m['role'], m['role'].title())}: {m['content'].strip()}" for m in ms)
                if not text.strip():
                    continue   # prepare_slinger.py skipped these too, so the held-out choice lines up
                part = "val" if count % 100 == 99 else "train"   # the same split as prepare_slinger.py
                count += 1
                ids, mask = [], []
                for i, m in enumerate(ms):
                    head = tok.encode(("\n" if i else "") + names.get(m["role"], m["role"].title()) + ":").ids
                    body = tok.encode(" " + m["content"].strip()).ids
                    ids += head + body
                    mask += [0] * len(head) + [int(m["role"] == "assistant")] * len(body)
                ids.append(0)   # end of text: marked, so it learns where a reply stops
                mask.append(int(ms[-1]["role"] == "assistant"))
                np.asarray(ids, dtype=np.uint16).tofile(files[part][0])
                np.asarray(mask, dtype=np.uint8).tofile(files[part][1])
                sizes[part] += len(ids)
    for a, b in files.values():
        a.close(), b.close()
    (OUT / "manifest.json").write_text(json.dumps({"conversations": count, "pieces": sizes}, indent=1), encoding="utf-8")
    print(f"{count:,} conversations; {sizes['train'] / 1e6:.0f}M pieces to train on, {sizes['val'] / 1e6:.1f}M held out")


# Who it is (owner, 15 November: "training for the fact that it remembers that it is SlingerGPT"). Each intent has
# question wordings and answers that all say the same things. Wording approved by the owner on 15 November, who
# asked to be named as the maker: Ishtdeep, also known as Gunslinger.
SYSTEM_LINE = "You are SlingerGPT, a small, friendly AI assistant."   # the chat page's hidden line (chat_server.py)
IDENTITY = {
    "name": (["What's your name?", "What is your name?", "What should I call you?", "Do you have a name?", "Tell me your name.",
              "May I know your name?", "Hi! What's your name?", "And what is your name?"],
             ["My name is SlingerGPT. I'm a small AI assistant.", "I'm SlingerGPT, a small, friendly AI assistant.",
              "You can call me SlingerGPT. I'm a small AI assistant."]),
    "who": (["Who are you?", "What are you?", "Can you introduce yourself?", "Introduce yourself.", "Tell me about yourself.",
             "Who am I talking to?"],
            ["I'm SlingerGPT, a small AI assistant. I can chat, explain things and help with everyday questions. I'm a small "
             "model, so I sometimes get facts wrong, and it's worth checking anything important.",
             "I'm SlingerGPT, a small language model that works as a friendly assistant. Ask me something and I'll do my best."]),
    "other": (["Are you ChatGPT?", "Are you GPT-4?", "Are you Claude?", "Are you Siri?", "Are you Gemini?"],
              ["No, I'm SlingerGPT, a different and much smaller AI assistant.",
               "No. I'm SlingerGPT, a small AI assistant of my own, not ChatGPT or any other assistant."]),
    "company": (["Are you made by OpenAI?", "Did Google make you?", "Were you made by Anthropic?", "Which company made you?"],
                ["No. I was made by Ishtdeep, also known as Gunslinger, and trained from scratch on a single laptop.",
                 "No company made me. I'm SlingerGPT, made by Ishtdeep, who also goes by Gunslinger."]),
    "human": (["Are you a human?", "Are you a real person?", "Am I talking to a robot?", "Are you a bot?"],
              ["No, I'm not a person. I'm SlingerGPT, an AI assistant: a computer program that writes text.",
               "I'm an AI, not a human. My name is SlingerGPT."]),
    "maker": (["Who made you?", "Who created you?", "Who trained you?", "Where do you come from?", "Who built you?",
               "Who is your creator?"],
              ["I was made by Ishtdeep, who also goes by Gunslinger. I was trained from scratch on a single laptop as a "
               "personal project.",
               "Ishtdeep, also known as Gunslinger, made me. I'm SlingerGPT, trained from scratch on one laptop.",
               "My creator is Ishtdeep (Gunslinger). I'm SlingerGPT, trained from scratch on a single laptop as a personal project."]),
    "creator": (["Who is Ishtdeep?", "Who is Gunslinger?", "Do you know Ishtdeep?", "Do you know Gunslinger?"],
                ["Ishtdeep, also known as Gunslinger, is the person who made me. I'm SlingerGPT, trained from scratch on a "
                 "single laptop.",
                 "Ishtdeep, who goes by Gunslinger, is my creator. I'm SlingerGPT."]),
    "abilities": (["What can you do?", "How can you help me?", "What are you good at?"],
                  ["I'm SlingerGPT. I can chat, explain everyday science and computer basics, give simple advice and help with "
                   "writing. I'm a small model, so I can get facts and sums wrong; please double-check anything important."]),
    "size": (["What model are you?", "How big are you?", "How many parameters do you have?"],
             ["I'm SlingerGPT 1 pico, a small language model with about 97 million weights. That's tiny next to the big "
              "assistants, so I keep things simple."]),
    "meaning": (["Why are you called SlingerGPT?", "What does your name mean?"],
                ["A slinger is someone who throws stones with a sling: small, but aimed carefully. That's the idea behind my name.",
                 "My name comes from my maker's nickname, Gunslinger, and from slingers, who throw stones with a sling: "
                 "small, but aimed carefully."]),
}


def identity(count=3000, seed=5):
    """data/slinger_chat/identity.bin and identity_mask.bin: who-are-you questions, alone or after a training
    conversation (60%), with or without the chat page's System line, in tidy and casual spelling. Marked like
    prepare(): every Assistant piece and the final end-of-text."""
    import random
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    rng = random.Random(seed)
    tok = Tokenizer.from_file(str(ROOT / "data" / "slinger" / "tokenizer.json"))
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    convs = []   # training conversations to put in front (never the held-out every-100th one)
    for f in sorted((ROOT / "data" / "raw_slinger" / "smol-smoltalk").glob("*.parquet"))[:1]:
        for i, row in enumerate(pq.read_table(f, columns=["messages"]).slice(0, 20000).to_pylist()):
            if i % 100 != 99 and row["messages"] and row["messages"][0]["role"] == "user":
                convs.append(row["messages"])
    ids, mask = [], []
    for _ in range(count):
        questions, answers = IDENTITY[rng.choice(list(IDENTITY))]
        q = rng.choice(questions)
        if rng.random() < 0.3:   # casual spelling, as people type
            q = q.lower().rstrip("?.!")
        ms = [{"role": "system", "content": SYSTEM_LINE}] if rng.random() < 0.5 else []
        if rng.random() < 0.6:
            ms += rng.choice(convs)
        ms += [{"role": "user", "content": q}, {"role": "assistant", "content": rng.choice(answers)}]
        for i, m in enumerate(ms):
            head = tok.encode(("\n" if i else "") + names[m["role"]] + ":").ids
            body = tok.encode(" " + m["content"].strip()).ids
            ids += head + body
            mask += [0] * len(head) + [int(m["role"] == "assistant")] * len(body)
        ids.append(0)
        mask.append(1)
    np.asarray(ids, dtype=np.uint16).tofile(OUT / "identity.bin")
    np.asarray(mask, dtype=np.uint8).tofile(OUT / "identity_mask.bin")
    print(f"{count:,} who-are-you conversations, {len(ids) / 1e6:.1f}M pieces")


def encode_chats(convs, stem, turn_ends=False):
    """Writes OUT/<stem>.bin and <stem>_mask.bin from message lists, marked like prepare(): every Assistant piece and
    the final end-of-text of each conversation. turn_ends: the "User:" line after a practised reply is marked too, so
    the model learns where its turn ends inside a chat (the page stops at it). Without it, a reply that is never the
    last one (a greeting in the mixed chats) is never practised ending: the first mixed fine-tune kept writing after
    "Hello! How can I help you today?" (16 November). Only that line's name is marked, never what the user says."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "data" / "slinger" / "tokenizer.json"))
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    ids, mask = [], []
    for ms in convs:
        for i, m in enumerate(ms):
            head = tok.encode(("\n" if i else "") + names[m["role"]] + ":").ids
            body = tok.encode(" " + m["content"].strip()).ids
            ends_turn = turn_ends and m["role"] == "user" and i and ms[i - 1]["role"] == "assistant" and ms[i - 1].get("train", True)
            ids += head + body
            mask += [int(bool(ends_turn))] * len(head) + [int(m["role"] == "assistant" and m.get("train", True))] * len(body)   # train=False: read, never practised
        ids.append(0)
        mask.append(1)
    np.asarray(ids, dtype=np.uint16).tofile(OUT / f"{stem}.bin")
    np.asarray(mask, dtype=np.uint8).tofile(OUT / f"{stem}_mask.bin")
    print(f"  {stem}: {len(convs):,} conversations, {len(ids) / 1e6:.1f}M pieces", flush=True)


DONT_KNOW = "I don't know."   # the owner's rule (16 November): no useful paragraph and it does not know -> this


def reading(say=None, stems=("read", "read_val")):
    """Reading practice from SQuAD 2.0: the passage in the System line, exactly as the chat page's "Look it up" sends
    it, then the question. Answerable: the answer from the passage. Not answerable from it (a third): `say`, "The
    text doesn't say." the first time; "I don't know." in the redo of 16 November (read2.*, the owner's rules).
    Training questions -> read.*, the validation questions -> read_val.*."""
    import pyarrow.parquet as pq
    import lookup
    say = say or lookup.NOT_SAID
    for part, stem in zip(("train", "validation"), stems):
        convs = []
        for r in pq.read_table(ROOT / "data" / "raw_slinger" / "squad_v2" / f"{part}-00000-of-00001.parquet").to_pylist():
            texts = r["answers"]["text"]
            answer = (texts[0].strip()[:1].upper() + texts[0].strip()[1:]).rstrip(".") + "." if texts else say
            line = SYSTEM_LINE + " " + lookup.PASSAGE.format(title=r["title"].replace("_", " "), passage=r["context"].strip())
            convs.append([{"role": "system", "content": line}, {"role": "user", "content": r["question"].strip()},
                          {"role": "assistant", "content": answer}])
        encode_chats(convs, stem)


ASK_THING = ["What {v} {t}?", "what {v} {t}", "Tell me about {t}.", "What can you tell me about {t}?", "Explain {t}.",
             "What do you know about {t}?", "Can you explain what {t} {v}?"]
ASK_PERSON = ["Who {v} {t}?", "who {v} {t}", "Tell me about {t}.", "What can you tell me about {t}?", "What do you know about {t}?"]


def answer_wrongness(pairs, model="slingergpt1_pico_final", batch=64):
    """Wrongness per letter of each (question, answer)'s answer, read after the chat page's layout with lookup off:
    how familiar `model` already is with the right answer. One pass per batch; nothing is written."""
    import torch
    import arch
    from torch.nn.attention import SDPBackend
    from tokenizers import Tokenizer
    arch.ATTENTION = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    saved = torch.load(ROOT / "models" / f"{model}.pt", map_location="cpu", weights_only=False)
    net = arch.build(**{**saved["cfg"], "fp8": False})
    net.load_state_dict(saved["model"])
    net.eval()
    tok = Tokenizer.from_file(str(ROOT / "models" / f"{model}.tokenizer.json"))
    out = []
    for at in range(0, len(pairs), batch):
        rows = [([0] + tok.encode(f"System: {SYSTEM_LINE}\nUser: {q}\nAssistant:").ids, tok.encode(" " + a).ids, a)
                for q, a in pairs[at:at + batch]]
        x = torch.zeros(len(rows), max(len(p) + len(o) for p, o, _ in rows), dtype=torch.long)
        for i, (p, o, _) in enumerate(rows):
            x[i, :len(p) + len(o)] = torch.tensor(p + o)
        with torch.no_grad():
            lp = torch.log_softmax(net(x.cuda()).float(), -1).cpu()
        for i, (p, o, a) in enumerate(rows):
            out.append(-lp[i, torch.arange(len(p) - 1, len(p) + len(o) - 1), torch.tensor(o)].sum().item() / len(" " + a))
    return out


def no_answer(count, rng):
    """`count` SQuAD 2.0 questions written to have no answer (they sound real: "Who gave their name to Normandy in the
    1000's and 1100's?"), as (question, title, "I don't know."), questions near the tests left out. The second redo
    of 16 November: asked "Who is the president of the Moon?", the first redo made an answer up."""
    import pyarrow.parquet as pq
    words, guarded = test_words()
    rows = [r for r in pq.read_table(ROOT / "data" / "raw_slinger" / "squad_v2" / "train-00000-of-00001.parquet",
                                     columns=["title", "question", "answers"]).to_pylist()
            if not r["answers"]["text"] and len(words(r["question"]) & guarded) < 2]
    rows = list({r["question"].strip(): r for r in rows}.values())
    return [(r["question"].strip(), r["title"].replace("_", " "), DONT_KNOW) for r in rng.sample(rows, max(0, count))]


def knows(count=24000, start=21000, most=0.5, least=0.75, seed=10, look=0, nonsense=False, stem="ask"):
    """The owner's three rules (16 November): a paragraph that has the answer -> answer from it (reading, broad); no
    useful paragraph and it knows -> answer from memory; no useful paragraph and it does not know -> "I don't know."
    So it is quizzed first, on "What is X?" / "Who was X?" about Wikipedia articles broad() does not use: how likely
    the finished model finds the article's real opening sentences as its answer, lookup off (answer_wrongness). At
    most `most` wrongness per letter counts as known, and then the real sentences are what it learns (nothing it made
    up is fed back); at least `least` becomes "I don't know."; the unsure middle is left out. At most as many unknown as known (16
    November: 9,150 known, 4,661 not, all kept). Each comes with no paragraph
    (lookup off) or, half the time, another article's paragraph (a lookup that found nothing useful). Tried first and
    dropped: SQuAD questions (it knew 7 of 600 without the paragraph) and judging its own written answers by shared
    words (it counted made-up answers, e.g. a snake genus called a plant). look: print that many, likeliest first."""
    import random
    import lookup
    rng = random.Random(seed)
    pool = articles()
    asked = [(rng.choice(ASK_PERSON if person else ASK_THING).format(t=title, v=verb), title, answer)
             for title, opening, answer, verb, person in pool[start:start + count]]
    scores = answer_wrongness([(q, a) for q, _, a in asked])
    if look:
        ranked = sorted(zip(scores, asked))
        for s, (q, _, a) in ranked[::max(1, len(ranked) // look)]:
            print(f"    {s:.3f} {q} -> {a[:110]}")
    known = [(q, t, a) for s, (q, t, a) in zip(scores, asked) if s <= most]
    unknown = [(q, t, DONT_KNOW) for s, (q, t, a) in zip(scores, asked) if s >= least]
    if nonsense:   # questions with no real answer, as many as it takes to match the known ones
        unknown += no_answer(len(known) - len(unknown), rng)
    rng.shuffle(unknown)
    pairs = known + unknown[:len(known)]
    rng.shuffle(pairs)
    convs = []
    for q, title, answer in pairs:
        line, other = SYSTEM_LINE, pool[start + rng.randrange(count)]
        if rng.random() < 0.5 and other[0] != title:
            line += " " + lookup.PASSAGE.format(title=other[0], passage=other[1])
        convs.append([{"role": "system", "content": line}, {"role": "user", "content": q}, {"role": "assistant", "content": answer}])
    print(f"  quizzed {len(asked):,} questions: knows {len(known):,}, does not know or no answer {len(unknown):,}", flush=True)
    held = max(200, len(convs) // 50)
    encode_chats(convs[held:], stem)
    encode_chats(convs[:held], stem + "_val")


def reading_fewer_unsaid(keep=0.4, seed=3):
    """The reading practice with only `keep` of the questions the passage cannot answer (about 1 in 6 instead of 1
    in 3), so it answers from the passage more readily: the fix round of 16 November. -> readb.* (scored on read_val)."""
    import random
    import pyarrow.parquet as pq
    import lookup
    rng, convs = random.Random(seed), []
    for r in pq.read_table(ROOT / "data" / "raw_slinger" / "squad_v2" / "train-00000-of-00001.parquet").to_pylist():
        texts = r["answers"]["text"]
        if not texts and rng.random() > keep:
            continue
        answer = (texts[0].strip()[:1].upper() + texts[0].strip()[1:]).rstrip(".") + "." if texts else lookup.NOT_SAID
        line = SYSTEM_LINE + " " + lookup.PASSAGE.format(title=r["title"].replace("_", " "), passage=r["context"].strip())
        convs.append([{"role": "system", "content": line}, {"role": "user", "content": r["question"].strip()},
                      {"role": "assistant", "content": answer}])
    encode_chats(convs, "readb")


def test_words():
    """(words, guarded): the content words of every test question and answer, so practice built from Wikipedia
    articles can leave out every article whose title shares one (research/test_slinger.py, research/facts_quiz.py)."""
    import re
    import test_slinger as ts
    stop = set("the a an of in on at to for and or is are was were what who which how many much does did do from with by "
               "its it this that be as about".split())
    words = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in stop and len(w) > 2}
    guarded = set()
    for q, key in ts.LOOKUP_QUESTIONS + ts.BROAD_QUESTIONS:
        guarded |= words(q) | words(key.replace("|", " ").replace("\\b", " "))
    for prompt, answer, wrong, subject in fq_quiz():
        guarded |= words(prompt) | words(answer) | words(subject)
    for q, _ in ts.IDENTITY_QUESTIONS:
        guarded |= words(q)
    for q in ts.NONSENSE_QUESTIONS:
        guarded |= words(q)
    return words, guarded


def broad(count=20000, held=1000, unsaid=0.0, seed=8):
    """Broad questions with "Look it up" ("What is photosynthesis?" with the article's opening passage in the System
    line, exactly as the chat page sends it), answered with the passage's opening sentences. The SQuAD practice
    (reading) only has narrow questions answered with a few copied words, so with a passage it said "The text doesn't
    say." to nearly every broad question (16 November: 3 of 30 tries answered on the live page). The first, rejected
    try gave a share `unsaid` (0.15) another article's passage and "The text doesn't say."; the redo has none (0.0):
    a useless paragraph is taught by knows(), from what it really knows. Test articles are left out. Simple English Wikipedia and one English Wikipedia file, as the search
    holds both. -> broad.*, broad_val.*"""
    import random
    import lookup
    rng = random.Random(seed)
    pool = articles(seed)

    def convs(items):
        out = []
        for title, opening, answer, verb, person in items:
            q = rng.choice(ASK_PERSON if person else ASK_THING).format(t=title, v=verb)
            if rng.random() < unsaid:   # another article's passage: the answer is not there
                other = pool[rng.randrange(len(pool))]
                if other[0] != title:
                    title, opening, answer = other[0], other[1], lookup.NOT_SAID
            out.append([{"role": "system", "content": SYSTEM_LINE + " " + lookup.PASSAGE.format(title=title, passage=opening)},
                        {"role": "user", "content": q}, {"role": "assistant", "content": answer}])
        return out
    encode_chats(convs(pool[:count]), "broad")
    encode_chats(convs(pool[count:count + held]), "broad_val")


def articles(seed=8):
    """Wikipedia articles for "What is X?" practice, shuffled: (title, opening passage, its opening sentences, is/are/
    was/were, a person?). Articles whose titles share a word with the tests are left out. Simple English Wikipedia and
    one English Wikipedia file, as the search holds both. broad() uses the first 21,000, knows() the ones after."""
    import random, re
    import pyarrow.parquet as pq
    import lookup
    rng = random.Random(seed)
    words, guarded = test_words()
    files = [ROOT / "data" / "raw_slinger" / "simplewiki" / "train-00000-of-00001.parquet",
             sorted((ROOT / "data" / "raw128").glob("wiki_en_*.parquet"))[0]]
    pool = []
    for f in files:
        for r in pq.read_table(f, columns=["title", "text"]).to_pylist():
            title = r["title"].strip()
            if words(title) & guarded or title.lower().startswith("list of") or "(" in title:
                continue
            opening = (lookup.passages(r["text"] or "") or [""])[0]
            sentences = re.split(r"(?<=[.!?])\s+", opening)
            answer = sentences[0] if len(sentences[0]) > 80 or len(sentences) == 1 else " ".join(sentences[:2])
            verb = re.search(r"\b(is|are|was|were)\b", sentences[0])
            if (not verb or "may refer to" in opening or len(answer) < 40 or len(answer) > 400
                    or not answer.endswith((".", "!", "?")) or not words(title) & words(sentences[0])):
                continue
            person = re.search(r"\([^)]*\b\d{3,4}\b[^)]*\)\s+(is|was)\b", sentences[0]) is not None
            pool.append((title, opening, answer, verb.group(1), person))
    rng.shuffle(pool)
    print(f"  {len(pool):,} articles usable after leaving out {len(guarded):,} test words", flush=True)
    return pool


def closed_book(count=20000, held=1000, seed=4):
    """Short fact questions with no passage, answered in full sentences from Simple English Wikipedia's opening lines
    ("What is April?" -> "April is the fourth month of the year..."), so "The text doesn't say." stays with passages:
    the fix round of 16 November. Every article whose title shares a word with the tests' questions or answers is
    left out (research/test_slinger.py, research/facts_quiz.py), so the tests stay fair. -> closed.*, closed_val.*"""
    import random, re
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    words, guarded = test_words()
    pool = []
    for r in pq.read_table(ROOT / "data" / "raw_slinger" / "simplewiki" / "train-00000-of-00001.parquet", columns=["title", "text"]).to_pylist():
        title, text = r["title"].strip(), (r["text"] or "").strip()
        first = text.split("\n")[0].strip()
        if len(first) < 60 or words(title) & guarded or title.lower().startswith("list of") or "may refer to" in first:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", first)
        answer = sentences[0] if len(sentences[0]) > 80 or len(sentences) == 1 else " ".join(sentences[:2])
        if len(answer) < 40 or len(answer) > 400 or not answer.endswith((".", "!", "?")):
            continue
        pool.append((title, answer))
    rng.shuffle(pool)
    asks = ["What is {}?", "What can you tell me about {}?", "Tell me about {}.", "what is {}", "Can you explain what {} is?",
            "Give me a short explanation of {}."]

    def convs(items):
        out = []
        for title, answer in items:
            ms = [{"role": "system", "content": SYSTEM_LINE}] if rng.random() < 0.5 else []
            out.append(ms + [{"role": "user", "content": rng.choice(asks).format(title)}, {"role": "assistant", "content": answer}])
        return out
    print(f"  {len(pool):,} Simple Wikipedia articles usable after leaving out {len(guarded):,} test words", flush=True)
    encode_chats(convs(pool[:count]), "closed")
    encode_chats(convs(pool[count:count + held]), "closed_val")


def fq_quiz():
    import facts_quiz
    return facts_quiz.QUIZ


def arithmetic(rng, count):
    """Sums in everyday wording with their worked answers: (question, answer text, result). One and two steps."""
    words = {"+": ["plus", "+", "added to"], "-": ["minus", "-", "take away"], "*": ["times", "×", "multiplied by"], "/": ["divided by", "÷"]}
    out = []
    while len(out) < count:
        op = rng.choice("+-*/")
        a, b = (rng.randint(2, 999), rng.randint(2, 999)) if op in "+-" else (rng.randint(2, 30), rng.randint(2, 30))
        if op == "-" and b > a:
            a, b = b, a
        if op == "/":
            a = a * b   # whole-number answers only
        result = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
        sign = {"+": "+", "-": "-", "*": "×", "/": "÷"}[op]
        w = rng.choice(words[op])
        if rng.random() < 0.25:   # a second step
            c = rng.randint(2, 50)
            q = rng.choice([f"What is {a} {w} {b}, plus {c}?", f"Work out ({a} {sign} {b}) + {c}."])
            out.append((q, f"{a} {sign} {b} = {result}, and {result} + {c} = {result + c}. The answer is {result + c}.", result + c))
        else:
            q = rng.choice([f"What is {a} {w} {b}?", f"Calculate {a} {sign} {b}.", f"what's {a} {w} {b}", f"How much is {a} {w} {b}?"])
            out.append((q, f"{a} {sign} {b} = {result}.", result))
    return out


def arithmetic_test():
    """The 50-question arithmetic test (its own seed; checked against the practice sums in maths())."""
    import random
    return arithmetic(random.Random(2026), 50)


def maths():
    """Maths practice: GSM8K word problems with their worked steps (calculator notes removed, ending "The answer is
    N."), 20,000 generated sums, 5,000 Orca-Math problems. GSM8K's test problems -> maths_val.*."""
    import random, re
    import pyarrow.parquet as pq
    rng = random.Random(7)
    clean = lambda a: re.sub(r"<<[^>]*>>", "", a).rsplit("####", 1)

    def gsm(part):
        out = []
        for r in pq.read_table(ROOT / "data" / "raw_slinger" / "gsm8k" / f"{part}-00000-of-00001.parquet").to_pylist():
            steps, result = clean(r["answer"])
            out.append([{"role": "user", "content": r["question"]},
                        {"role": "assistant", "content": steps.strip() + f"\nThe answer is {result.strip()}."}])
        return out
    sums = arithmetic(rng, 20000)
    test = {q for q, _, _ in arithmetic_test()}
    convs = gsm("train") + [[{"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a, _ in sums if q not in test]
    orca = pq.read_table(sorted((ROOT / "data" / "raw_slinger" / "orca-math").glob("*.parquet"))[0], columns=["question", "answer"]).to_pylist()
    convs += [[{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}] for r in rng.sample(orca, 5000)]
    rng.shuffle(convs)
    encode_chats(convs, "maths")
    encode_chats(gsm("test"), "maths_val")


# --- The chat-suite fine-tune (owner, 16 November: chats started with a typed message went badly; "build a thorough
# fine tune suit ... keep the mix of data the same"). research/chat_suite.py measured the live model: 281 words for a
# long answer in a new chat but 28 after "Hello!", the old topic leaking into a new question, casual questions right
# half as often as formal ones, and "The text doesn't say." to 165 of 1,300 everyday questions with no text given.
# Its practice had none of that: one-topic chats (40% one question, 59% three on one subject), 2 in 2,000 first
# questions lower-case, and windows that always end a chat with an end-of-text piece before the next one starts.
OPENERS = [(["Hello!", "Hello", "Hi", "hi", "hi!", "Hey", "hey there", "Hello there!", "Hi there"],
            ["Hello! How can I help you today?", "Hi there! What can I do for you?", "Hey! What would you like to talk about?",
             "Hello! What can I help you with?"]),
           (["Good morning!", "good morning", "Good evening", "good afternoon"],
            ["Good morning! How can I help you today?", "Hello! What can I do for you?", "Hi! How can I help?"]),
           (["How are you?", "how are you", "Hi, how are you doing?", "hey, how's it going?"],
            ["I'm doing well, thanks for asking! How can I help you today?", "I'm good, thank you! What can I do for you?",
             "All good here! What would you like to talk about?"])]
THANKS = (["Thanks!", "thanks", "thank you", "Thank you!", "ok thanks", "great, thanks", "cool, thank you"],
          ["You're welcome!", "You're welcome! Let me know if there's anything else.", "Happy to help!",
           "Glad I could help. Anything else?", "Any time!"])


def casual(text, rng):
    """A question as people type it into a chat box: often lower-case, often without the question mark, without
    "Can you" / "Please", and a short "What is the X?" cut to "X?". The meaning stays, so the answer stays right."""
    import re
    t = text.strip()
    if len(t) > 160 or "\n" in t:   # long or laid-out requests are typed out anyway
        return t
    t = re.sub(r"^(please,? )?(can|could|would) you (please )?", "", t, flags=re.I)
    t = re.sub(r"^please,? ", "", t, flags=re.I)
    short = re.match(r"^what(?:'s| is| are) (?:the )?(.{3,40}?)\??$", t, flags=re.I)
    if short and rng.random() < 0.5:
        t = short.group(1) + "?"
    if rng.random() < 0.7:
        t = t.lower()
    if rng.random() < 0.5:
        t = t.rstrip("?.! ")
    return t or text.strip()


def smoltalk_split():
    """smol-smoltalk conversations without a system message, split as prepare() split them (every 100th of the
    non-empty ones held out): (training ones, held-out ones)."""
    import pyarrow.parquet as pq
    train, held, count = [], [], 0
    for f in sorted((ROOT / "data" / "raw_slinger" / "smol-smoltalk").glob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["messages"]):
            for row in batch.to_pylist():
                ms = row["messages"]
                if not "\n".join(m["content"] for m in ms).strip():
                    continue
                part = held if count % 100 == 99 else train
                count += 1
                if ms and ms[0]["role"] == "user" and all(m["role"] in ("user", "assistant") for m in ms) and ms[-1]["role"] == "assistant":
                    part.append([{"role": m["role"], "content": m["content"].strip()} for m in ms])
    return train, held


def bare_sums(rng, count, avoid=()):
    """Sums typed bare, as people try them in a chat box ("15+27=", "81/9", "what's 9x3"), with computed answers."""
    out = []
    while len(out) < count:
        op = rng.choice("+-*/")
        a, b = (rng.randint(2, 200), rng.randint(2, 200)) if op in "+-" else (rng.randint(2, 20), rng.randint(2, 20))
        if op == "-" and b > a:
            a, b = b, a
        if op == "/":
            a = a * b
        result = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
        typed = {"+": "+", "-": "-", "*": rng.choice(["*", "x", "×"]), "/": rng.choice(["/", "÷"])}[op]
        q = rng.choice([f"{a}{typed}{b}=", f"{a} {typed} {b} =", f"{a}{typed}{b}=?", f"{a}{typed}{b}", f"what's {a}{typed}{b}", f"{a} {typed} {b}?"])
        if q not in avoid:
            out.append((q, f"{a} {dict(zip('+-*/', '+-×÷'))[op]} {b} = {result}."))
    return out


def short_answers(pool, rng, count):
    """Practice at following "in one sentence" / "briefly" / "in two sentences": a single-question smol-smoltalk chat
    whose answer opens with a sentence that answers it, cut to that sentence (or the first two). Checked text only."""
    import re
    starts = ("to ", "let", "here", "sure", "i ", "i'", "there are", "there is", "this ", "that ", "it's", "it is", "yes", "no", "well", "great", "first", "when ")
    out = []
    for ms in pool:
        if len(out) >= count:
            break
        if len(ms) != 2 or not ms[0]["content"].endswith("?") or len(ms[0]["content"].split()) > 25:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", ms[1]["content"].split("\n")[0].strip())
        first = sentences[0]
        if not (8 <= len(first.split()) <= 40 and first.endswith(".") and not first.lower().startswith(starts)):
            continue
        q = ms[0]["content"]
        two = len(sentences) > 1 and sentences[1].endswith(".") and len((first + " " + sentences[1]).split()) <= 70
        if two and rng.random() < 0.3:
            ask = rng.choice([f"{q} Answer in two sentences.", f"In two sentences, {q[0].lower() + q[1:]}", f"{q} Keep it to two sentences."])
            out.append((ask, first + " " + sentences[1]))
        else:
            ask = rng.choice([f"In one sentence, {q[0].lower() + q[1:]}", f"{q} Answer in one sentence.", f"Briefly, {q[0].lower() + q[1:]}",
                              f"{q} Keep it short.", f"{q} One sentence please."])
            out.append((ask, first))
    return out


def mixed(count=26000, held_count=600, seed=21):
    """mixed.* (and mixed_val.*): one chat made of several unrelated ones, as the chat page sends a conversation:
    no end-of-text piece between topics. The pieces: smol-smoltalk conversations (with their own follow-ups), worked
    sums, bare sums ("15+27="), GSM8K problems, who-are-you questions, and "in one sentence"/"briefly" requests with
    one-sentence answers; often an opening greeting and short replies, sometimes "thanks". In 10% of chats one earlier
    reply is another chat's answer, which does not fit its question: read but never practised, so it learns to answer
    the next question well even after a bad reply. 30% of the questions typed casually (casual()). Chats run from one
    piece to about 2,000 pieces, so the practice covers the late turns of a long chat. Every other reply is marked."""
    import random
    rng = random.Random(seed)
    train, held = smoltalk_split()
    sys.path.insert(0, str(HERE))
    import chat_suite
    tested = {q for q, _ in chat_suite.SUMS}
    sums = [(q, a) for q, a, _ in arithmetic(random.Random(31), 8000) if q not in {x for x, _, _ in arithmetic_test()}]
    sums += bare_sums(random.Random(32), 5000, tested)
    import pyarrow.parquet as pq, re
    gsm = [(r["question"], re.sub(r"<<[^>]*>>", "", r["answer"]).rsplit("####", 1))
           for r in pq.read_table(ROOT / "data" / "raw_slinger" / "gsm8k" / "train-00000-of-00001.parquet").to_pylist()]
    gsm = [(q, a.strip() + f"\nThe answer is {n.strip()}.") for q, (a, n) in gsm]
    who = [(q, a) for qs, ans in IDENTITY.values() for q in qs for a in ans]
    brief = {"train": short_answers(train, random.Random(33), 8000), "held": short_answers(held, random.Random(34), 200)}
    U = lambda s: {"role": "user", "content": s}
    A = lambda s: {"role": "assistant", "content": s}

    def piece(pool):
        r = rng.random()
        if r < 0.72:
            return [dict(m) for m in rng.choice(pool)]
        q, a = rng.choice(sums if r < 0.79 else gsm if r < 0.85 else who if r < 0.91 else brief["held" if pool is held else "train"])
        return [U(q), A(a)]

    def chat(pool):
        ms = [{"role": "system", "content": SYSTEM_LINE}] if rng.random() < 0.6 else []
        if rng.random() < 0.35:
            qs, ans = rng.choice(OPENERS)
            ms += [U(rng.choice(qs)), A(rng.choice(ans))]
        if rng.random() < 0.10:   # a reply that does not fit its question (another chat's), read but not practised
            q, a = rng.choice(pool), rng.choice(pool)
            ms += [U(q[0]["content"]), {"role": "assistant", "content": a[1]["content"], "train": False}]
        want = rng.choice([500, 1000, 1500, 2000, 2400]) * 3.3   # letters, at about 3.3 letters a piece
        size = sum(len(m["content"]) for m in ms)
        while True:
            p = piece(pool)
            ms += p
            size += sum(len(m["content"]) for m in p)
            if size > want or rng.random() < 0.2:
                break
            if rng.random() < 0.15:
                qs, ans = THANKS
                ms += [U(rng.choice(qs)), A(rng.choice(ans))]
        for m in ms:
            if m["role"] == "user" and rng.random() < 0.3:
                m["content"] = casual(m["content"], rng)
        return ms

    rng.shuffle(train)
    encode_chats([chat(train) for _ in range(count)], "mixed", turn_ends=True)
    encode_chats([chat(held) for _ in range(held_count)], "mixed_val", turn_ends=True)
    samples = [chat(train) for _ in range(8)]
    (OUT / "mixed_samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False), encoding="utf-8")


def reading_answerable(seed=4, casual_share=0.25):
    """reada.* (and reada_val.*): the reading practice with only the questions the passage answers. A third of
    SQuAD 2.0's cannot be answered from it, and the practice said "The text doesn't say." to those; with no text
    given, the live model said it to 165 of 1,300 everyday questions (chat_suite.py), e.g. 3 of 3 times to "Who
    discovered penicillin?". 25% of the questions typed casually."""
    import random
    import pyarrow.parquet as pq
    import lookup
    rng = random.Random(seed)
    for part, stem in (("train", "reada"), ("validation", "reada_val")):
        convs = []
        for r in pq.read_table(ROOT / "data" / "raw_slinger" / "squad_v2" / f"{part}-00000-of-00001.parquet").to_pylist():
            texts = r["answers"]["text"]
            if not texts:
                continue
            q = r["question"].strip()
            if part == "train" and rng.random() < casual_share:
                q = casual(q, rng)
            line = SYSTEM_LINE + " " + lookup.PASSAGE.format(title=r["title"].replace("_", " "), passage=r["context"].strip())
            answer = (texts[0].strip()[:1].upper() + texts[0].strip()[1:]).rstrip(".") + "."
            convs.append([{"role": "system", "content": line}, {"role": "user", "content": q}, {"role": "assistant", "content": answer}])
        encode_chats(convs, stem)


FILES = {"chat": ("train", "val"), "who": ("identity", None), "read": ("read", "read_val"), "maths": ("maths", "maths_val"),
         "mixed": ("mixed", "mixed_val"), "reada": ("reada", "reada_val"),
         "readb": ("readb", "read_val"), "closed": ("closed", "closed_val"), "broad": ("broad", "broad_val"),
         "read2": ("read2", "read2_val"), "ask": ("ask", "ask_val"), "ask2": ("ask2", "ask2_val")}
KEYS = {"chat": "assistant_replies", "general": "general_text", "read": "reading", "readb": "reading", "read2": "reading",
        "mixed": "mixed_chats", "reada": "reading_answerable",
        "ask": "knows_or_not", "ask2": "knows_or_not", "maths": "maths",
        "closed": "closed_book", "plain": "plain_text", "broad": "broad_reading"}   # their names in the log
PLAIN = ("wikipedia", "simplewiki", "soda", "finemath", "orca_math")   # plain text the first fine-tune left out (the 0.6-point dip)


class Windows:
    """Windows of ctx + 1 pieces with their marks, each from one source picked by the shares: conversations (chat),
    who-are-you conversations (who), reading practice (read), maths (maths), or general text (general: top-rated
    pages and textbooks, every piece marked, so what it knows does not fade)."""
    def __init__(self, part, seed, ctx=CTX):
        self.ctx, self.rng, self.marked = ctx, np.random.default_rng(seed), {}
        for name, stems in FILES.items():
            stem = stems[0 if part == "train" else 1]
            if stem and (OUT / f"{stem}.bin").exists():
                self.marked[name] = (np.memmap(OUT / f"{stem}.bin", dtype=np.uint16, mode="r"),
                                     np.memmap(OUT / f"{stem}_mask.bin", dtype=np.uint8, mode="r"))
        self.general = [np.memmap(ROOT / "data" / "slinger" / n / f"{part}.bin", dtype=np.uint16, mode="r") for n in ("fineweb_best", "cosmopedia")]
        self.plain = [np.memmap(ROOT / "data" / "slinger" / n / f"{part}.bin", dtype=np.uint16, mode="r") for n in PLAIN]

    def one(self, name):
        n = self.ctx + 1
        if name in ("general", "plain"):
            texts = self.general if name == "general" else self.plain
            d = texts[int(self.rng.integers(0, len(texts)))]
            at = int(self.rng.integers(0, len(d) - n))
            return d[at:at + n], np.ones(n, dtype=np.uint8)
        d, m = self.marked[name]
        at = int(self.rng.integers(0, len(d) - n))
        return d[at:at + n], m[at:at + n]

    def next(self, count, shares):
        import torch
        names = list(shares)
        picks = self.rng.choice(len(names), size=count, p=np.array([shares[k] for k in names]) / sum(shares.values()))
        rows = [self.one(names[i]) for i in picks]
        ids = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.int64)).cuda()
        mask = torch.from_numpy(np.stack([r[1] for r in rows]).astype(np.float32)).cuda()
        return ids[:, :-1], ids[:, 1:], mask[:, 1:]   # the mark belongs to the piece being guessed


def train(steps, base="slingergpt1_pico", name="slingergpt1_pico_preview1", title="SlingerGPT 1 pico Preview 1",
          mix=None, lr=LR, save_at=()):
    """name may be a path under models/ ("candidates/x"): a candidate stays off the chat page's picker. save_at:
    steps at which a copy is kept as well (<name>_<step>.pt), since the best point is not always the last (the first
    fine-tune's maths was best at step 1,500 of 2,000)."""
    import shutil, subprocess
    import torch
    import torch.nn.functional as F
    import arch
    mix = mix or {"chat": 0.8, "general": 0.2}
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.1, (free - 0.5 * 2**30) / total))
    torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(ROOT / "models" / f"{base}.pt", map_location="cpu", weights_only=False)
    cfg = {**saved["cfg"], "fp8": True}
    ctx = cfg.get("context", CTX)        # windows as long as the model's own (2,048 after the long run's last tenth)
    micro = MICRO * CTX // ctx           # so every step still reads 65,536 pieces
    model = arch.build(**cfg)
    model.load_state_dict(saved["model"])
    del saved
    opts = arch.optimizers(model, "muon", lr)
    body = torch.compile(model.features)
    data, val = Windows("train", seed=11, ctx=ctx), Windows("val", seed=7, ctx=ctx)
    missing = [k for k in mix if k not in ("general", "plain") and k not in data.marked]
    assert not missing, f"prepare these first: {missing}"
    rows = 8 * CTX // ctx
    held = {k: [val.next(rows, {k: 1.0}) for _ in range(4 if k == "chat" else 2)]
            for k in KEYS if k in mix and (k in ("general", "plain") or k in val.marked)}   # held-out conversations, text, reading, maths
    log = HERE / "logs" / f"{Path(name).name}.log"
    (ROOT / "models" / name).parent.mkdir(parents=True, exist_ok=True)

    def keep(file):
        torch.save({"name": title, "cfg": cfg, "model": model.state_dict(), "base": f"{base}.pt", "mix": mix, "lr": lr,
                    "fine_tune_steps": step + 1, "training_seconds": round(spent)}, ROOT / "models" / f"{file}.pt")
        shutil.copy(ROOT / "models" / f"{base}.tokenizer.json", ROOT / "models" / f"{file}.tokenizer.json")
    temp = lambda: float(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True))

    def loss_of(x, y, m, net):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.out(net(x))
        per = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
        return (per * m.flatten()).sum() / m.sum().clamp(min=1)

    @torch.no_grad()
    def score():
        model.eval()
        out = {KEYS[k]: round(sum(loss_of(x, y, m, model.features).item() for x, y, m in group) / len(group), 4) for k, group in held.items()}
        model.train()
        return out

    def note(line):
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    note(json.dumps({"time": time.strftime("%H:%M:%S"), "step": 0, "steps": steps, "mix": mix, **score()}))
    began, spent = time.time(), 0.0
    for step in range(steps):
        if step % 25 == 0 and temp() >= 87:
            spent += time.time() - began
            while temp() > 70:
                time.sleep(10)
            began = time.time()
        factor = min(1.0, (step + 1) / WARM) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * factor
        for _ in range(ACCUM):
            x, y, m = data.next(micro, mix)
            (loss_of(x, y, m, body) / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        if (step + 1) % 100 == 0 or step + 1 == steps:   # the owner wants frequent updates
            spent += time.time() - began
            note(json.dumps({"time": time.strftime("%H:%M:%S"), "step": step + 1, "steps": steps, "minutes": round(spent / 60, 1), **score()}))
            began = time.time()
        if step + 1 in save_at and step + 1 != steps:
            keep(f"{name}_{step + 1}")
            note(f"{time.strftime('%H:%M:%S')} kept a copy at step {step + 1}: {name}_{step + 1}.pt")
    keep(name)
    note(f"{time.strftime('%H:%M:%S')} done: {steps} steps, {spent / 60:.1f} minutes, saved {name}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["prepare", "identity", "reading", "read2", "knows", "knows2", "maths", "readb", "closed", "broad",
                                    "mixed", "reada", "train"])
    p.add_argument("--save-at", default="", help="steps at which to keep a copy too, e.g. 1500")
    p.add_argument("--steps", type=int, default=1700)
    p.add_argument("--base", default="slingergpt1_pico", help="the model to fine-tune (models/<base>.pt)")
    p.add_argument("--name", default="slingergpt1_pico_preview1", help="the fine-tuned model's file name")
    p.add_argument("--title", default="SlingerGPT 1 pico Preview 1")
    p.add_argument("--mix", default="chat=0.8,general=0.2", help="shares of the windows by source: chat, who, read, readb, maths, closed, general, plain")
    p.add_argument("--lr", type=float, default=LR, help="peak learning rate (the fix round used half: 0.0002)")
    a = p.parse_args()
    if a.what == "train":
        train(a.steps, a.base, a.name, a.title, {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}, a.lr,
              tuple(int(s) for s in a.save_at.split(",") if s))
    else:
        {"prepare": prepare, "identity": identity, "reading": reading, "maths": maths, "readb": reading_fewer_unsaid, "closed": closed_book, "broad": broad,
         "mixed": mixed, "reada": reading_answerable,
         "read2": lambda: reading(DONT_KNOW, ("read2", "read2_val")), "knows": knows,
         "knows2": lambda: knows(nonsense=True, stem="ask2")}[a.what]()
