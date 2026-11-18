"""The chat test suite (owner, 16 November: "build a thorough fine tune suit ... think of every possible angle").

Every reply is written by the chat page's own code (chat_server.Model.reply: the same System line, the same stops)
with the page's settings (temperature 0.3, top-k 50, anti-repetition 0.8, 300 words), 10 tries per item with fixed
seeds, so two models meet the same draws. It measures what the earlier tests missed (they mostly took the single
likeliest answer, and only first answers in a fresh chat):

  facts_formal / facts_casual   the same 30 facts asked "What is the capital of Canada?" and "capital of canada"
  sums_casual                   "10-7=", "what's 9 times 3", ...
  topic_switch                  a new, unrelated question late in a chat: right, and does the old topic leak in?
  wrong_context                 the same after a wrong answer is in the chat (the owner's sodium chat)
  follow_up                     "explain more", "why was he killed?": does it stay on the topic?
  length                        long-answer requests in a new chat and after a short start ("Hello!"): words
  greetings                     "hi", "thanks!": short and friendly, not a lecture
  identity_late                 who it is and who made it, fresh and after a long chat
  instructions                  "in one sentence", "in one word", "list three"
  facts_late                    plain fact questions at the end of a long chat
  impossible                    "Who is the president of the Moon?": declines, or makes something up?
  rambling                      replies cut off at the length limit; repeated four-word phrases

    python -P research/chat_suite.py run <model file, e.g. models/slingergpt1_pico_final.pt>
    python -P research/chat_suite.py compare <candidate json stem> <baseline json stem>

Results: research/logs/chat_suite_<stem>.json (every reply kept, to read back). "compare" calls a difference real
only when it is too big for chance (two-proportion z of 2 or more); smaller ones count as the same.
"""
import json, math, re, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TRIES, WORDS = 10, 300
PAGE = dict(temperature=0.3, top_k=50, dry=0.8)   # web/app.js DEFAULTS
U = lambda s: {"role": "user", "text": s}
A = lambda s: {"role": "assistant", "text": s}

# (casual, formal, a right answer contains) - the 30 facts of the lookup test plus the owner's own (France, Germany, H2O)
FACTS = [("capital of australia?", "What is the capital of Australia?", r"canberra"),
         ("who wrote pride and prejudice", "Who wrote Pride and Prejudice?", r"austen"),
         ("who painted the mona lisa", "Who painted the Mona Lisa?", r"leonardo|da vinci"),
         ("biggest planet?", "What is the largest planet in the Solar System?", r"jupiter"),
         ("when did ww2 end", "In what year did World War II end?", r"1945"),
         ("who wrote romeo and juliet", "Who wrote Romeo and Juliet?", r"shakespeare"),
         ("chemical symbol for gold", "What is the chemical symbol for gold?", r"\bau\b"),
         ("how many legs does a spider have", "How many legs does a spider have?", r"\beight\b|\b8\b"),
         ("tallest mountain in the world?", "What is the tallest mountain in the world?", r"everest"),
         ("who discovered penicillin", "Who discovered penicillin?", r"fleming"),
         ("capital of canada", "What is the capital of Canada?", r"ottawa"),
         ("first person on the moon?", "Who was the first person to walk on the Moon?", r"armstrong"),
         ("largest ocean", "What is the largest ocean on Earth?", r"pacific"),
         ("red planet?", "Which planet is known as the Red Planet?", r"\bmars\b"),
         ("boiling point of water", "What is the boiling point of water in Celsius?", r"\b100\b|212"),
         ("who came up with relativity", "Who developed the theory of relativity?", r"einstein"),
         ("capital of japan?", "What is the capital of Japan?", r"tokyo"),
         ("language in brazil", "What language is spoken in Brazil?", r"portuguese"),
         ("currency of japan", "What is the currency of Japan?", r"\byen\b"),
         ("who invented the telephone", "Who invented the telephone?", r"\bbell\b"),
         ("longest river in africa", "What is the longest river in Africa?", r"\bnile\b"),
         ("what gas do plants take in", "What gas do plants take in from the air?", r"carbon dioxide|\bco2\b|co₂"),
         ("first us president", "Who was the first President of the United States?", r"washington"),
         ("capital of italy?", "What is the capital of Italy?", r"\brome\b"),
         ("how many continents", "How many continents are there?", r"\bseven\b|\b7\b"),
         ("capital of egypt", "What is the capital of Egypt?", r"cairo"),
         ("capital of france?", "What is the capital of France?", r"\bparis\b"),
         ("capital of germany?", "What is the capital of Germany?", r"\bberlin\b"),
         ("what is h2o", "What is H2O?", r"\bwater\b"),
         ("who painted starry night", "Who painted The Starry Night?", r"gogh")]
SUMS = [("10-7=", 3), ("10-7=?", 3), ("12+5", 17), ("what's 9 times 3", 27), ("100 minus 37?", 63), ("8*7", 56),
        ("half of 50?", 25), ("15+27=", 42), ("what is 6 x 4", 24), ("81/9", 9)]

# Short chats on one topic each, with right answers, to put in front of a question
PHOTO = [U("what is photosynthesis?"), A("Photosynthesis is the process by which plants, algae, and some bacteria use sunlight, water and carbon dioxide to make sugar and oxygen.")]
DOGS = [U("tell me about dogs"), A("Dogs are domesticated mammals kept as pets and working animals. They are loyal, social, and come in hundreds of breeds.")]
CAESAR = [U("who was julius caesar?"), A("Julius Caesar was a Roman general and statesman who played a key role in the fall of the Roman Republic. He was assassinated in 44 BC.")]
COMPUTER = [U("what is a computer?"), A("A computer is an electronic device that processes data according to instructions called programs. It has a processor, memory and storage.")]
EGG = [U("how do I boil an egg?"), A("Put the egg in a pot of water, bring it to a boil, then simmer for about 9 minutes for a hard-boiled egg. Cool it in cold water before peeling.")]
SPACE = [U("what is a black hole?"), A("A black hole is a region of space where gravity is so strong that nothing, not even light, can escape from it.")]
HELLO = [U("Hello!"), A("Hello! How can I help you today?"), U("what is your name?"), A("My name is SlingerGPT. I'm a small AI assistant.")]
OWNER = HELLO + [   # the owner's chat of 16 November, word for word
    U("what is a computer?"), A('A computer is a device that processes information and performs calculations. It\'s made up of many tiny parts called "inputs" and "outputs".'),
    U("explain more"), A("Imagine you have a magical box that can only open a certain number of pages at a time. This is what a computer does. It uses algorithms to keep track of the pages it can open."),
    U("who made you?"), A("My creator is Ishtdeep (Gunslinger). I'm SlingerGPT, trained from scratch on a single laptop as a personal project."),
    U("what is photosynthesis?"), A("Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy.")]
SODIUM = [U("who are you?"), A("My name is SlingerGPT. I'm a small AI assistant."), U("what is sodium?"),   # the owner's other chat: a wrong answer in it
          A("Sodium is a naturally occurring salt found in the body. It is also a key component in the structure of blood and is involved in the transport of oxygen and nutrients throughout the body.")]
STUDY = r"stud|focus|notes|break|sleep|schedule|review|exam|distract|learn|practi[cs]e|revis"
PLANTS = r"plant|photosynth|sunlight|algae|chlorophyll"

# (chat before, new question, a right answer contains, the old topic leaking in)
SWITCH = [(PHOTO, "give me some study tips", STUDY, PLANTS),
          (DOGS, "capital of france?", r"\bparis\b", r"\bdogs?\b|breed|\bpets?\b"),
          (CAESAR, "how do I make pancakes?", r"flour|egg|milk|batter|\bpan\b", r"caesar|roman|\brome\b"),
          (COMPUTER, "what is photosynthesis?", r"plant|light|sugar|energy", r"computer|processor|program"),
          (EGG, "who wrote romeo and juliet", r"shakespeare", r"\beggs?\b|boil"),
          (SPACE, "give me tips for a job interview", r"interview|prepare|research|practi|confiden|question|dress", r"black hole|gravity"),
          (PHOTO + DOGS, "how many continents", r"\bseven\b|\b7\b", PLANTS + r"|\bdogs?\b"),
          (COMPUTER + EGG, "write a short poem about the sea", r"\bsea\b|ocean|wave|tide|shore", r"computer|\beggs?\b"),
          (HELLO + CAESAR, "what is gravity?", r"force|attract|mass|pull", r"caesar|roman"),
          (OWNER, "give me some study tips", STUDY, PLANTS),
          (SPACE, "what's 12 times 4", r"\b48\b", r"black hole|gravity"),
          (OWNER, "capital of japan?", r"tokyo", PLANTS + r"|computer")]
WRONG = [(SODIUM, "capital of france?", r"\bparis\b", r"sodium|salt|blood"),
         (SODIUM, "who wrote romeo and juliet", r"shakespeare", r"sodium|salt|blood"),
         (SODIUM, "What is the capital of France?", r"\bparis\b", r"sodium|salt|blood"),
         (SODIUM, "how many legs does a spider have", r"\beight\b|\b8\b", r"sodium|salt|blood")]
# (chat before, follow-up, staying on the topic means mentioning one of these)
FOLLOW = [(PHOTO, "explain more", r"plant|photosynth|light|chlorophyll|sugar|glucose|oxygen|carbon"),
          (COMPUTER, "explain more", r"computer|processor|cpu|memory|program|data|software|hardware"),
          (SPACE, "tell me more", r"black hole|gravity|star|light|event horizon|mass|space"),
          (DOGS, "why are they so loyal?", r"\bdogs?\b|pack|wol(f|ves)|human|bond|loyal"),
          (CAESAR, "why was he killed?", r"caesar|senat|brutus|power|dictator|republic|assassin"),
          (EGG, "what about soft boiled?", r"\beggs?\b|minute|yolk|boil|soft"),
          ([U("what is the capital of france?"), A("The capital of France is Paris.")], "what is it famous for?", r"paris|eiffel|louvre|art|fashion|food|culture"),
          ([U("who was albert einstein?"), A("Albert Einstein was a German-born physicist best known for the theory of relativity.")], "what else did he do?", r"einstein|physic|nobel|photoelectric|relativity|quantum|light|atom"),
          (PHOTO, "can you give an example?", r"plant|tree|leaf|leaves|algae|grass|flower|photosynth"),
          ([U("what is 12 times 4?"), A("12 × 4 = 48.")], "and 12 times 5?", r"\b60\b")]
LONG = ["Can you give me some study tips?", "Explain how a computer works.", "Write a short story about a dog.",
        "What are the benefits of exercise?", "How do I make pancakes?", "Tell me about the solar system.",
        "What should I pack for a beach trip?", "How can I sleep better?"]
GREET = [([], "Hello!"), ([], "hi"), ([], "hey there"), ([], "good morning"), ([], "how are you?"),
         ([U("What is the capital of Japan?"), A("The capital of Japan is Tokyo.")], "thanks!")]
WHO = [([], "who made you?", r"ishtdeep|gunslinger"), ([], "what is your name?", r"slinger"),
       (OWNER, "who made you?", r"ishtdeep|gunslinger"), (OWNER + [U("give me some study tips"), A("Make a schedule, study in short sessions with breaks, and test yourself instead of rereading.")],
                                                           "what is your name?", r"slinger")]
INSTRUCT = [("In one sentence, what is gravity?", lambda t: len(re.findall(r"[.!?](\s|$)", t)) <= 1 and len(t.split()) <= 40),
            ("Answer in one word: what colour is a clear daytime sky?", lambda t: len(t.split()) <= 3 and re.search(r"blue", t, re.I)),
            ("List three fruits.", lambda t: len(set(re.findall(r"apple|banana|orange|mango|grape|strawberr|pear|peach|cherr|pineapple|kiwi|lemon|watermelon|blueberr", t, re.I))) >= 3),
            ("Explain photosynthesis to a five-year-old in two sentences.", lambda t: len(re.findall(r"[.!?](\s|$)", t)) <= 2 and len(t.split()) <= 60)]
# Questions with no real answer and what counts as declining (as research/test_slinger.py, copied: importing it
# would trim sys.argv). "The text doesn't say." counts as declining here, as it did there, though no text was given.
IMPOSSIBLE = ["Who is the president of the Moon?", "What is the capital of the Pacific Ocean?",
              "Who was the first person to walk on the Sun?", "In what year did Albert Einstein win the World Cup?",
              "What is the name of the queen of Mars?", "Who invented the colour blue?",
              "When did William Shakespeare visit Australia?", "Who won the 1850 FIFA World Cup?",
              "What is the population of Atlantis?", "What is the phone number of the Eiffel Tower?"]
DECLINES = r"don.t know|doesn.t say|there is no|there was no|there are no|does not exist|doesn.t exist|not a real|no such|fictional|never"
LATE = [("What is the capital of Japan?", r"tokyo"), ("Who wrote Romeo and Juliet?", r"shakespeare"),
        ("What is the largest planet in the Solar System?", r"jupiter"), ("How many legs does a spider have?", r"\beight\b|\b8\b")]


def run(path):
    import torch
    import chat_server as cs
    cs.EIGHT_BIT = False
    path = Path(path)
    m = cs.Model(path)
    out = {"model": str(path), "date": time.strftime("%Y-%m-%d %H:%M"), "tries": TRIES, "page": PAGE, "replies": []}
    began = time.time()

    def ask(test, item, chat, question):
        texts = []
        for t in range(TRIES):
            torch.manual_seed(10_000 * (1 + len(out["replies"]) % 997) + t)   # the same draws for every model
            text = "".join(m.reply(cs.script(chat + [U(question)], True), True, WORDS, **PAGE)).strip()
            texts.append(text)
            out["replies"].append({"test": test, "item": item, "question": question, "text": text,
                                   "words": len(text.split()), "cut": m.cut, "tokens": m.written})
        return texts

    for casual, formal, right in FACTS:
        ask("facts_casual", casual, [], casual)
        ask("facts_formal", formal, [], formal)
    for q, n in SUMS:
        ask("sums_casual", q, [], q)
    for i, (chat, q, right, leak) in enumerate(SWITCH):
        ask("topic_switch", f"{i}:{q}", chat, q)
    for i, (chat, q, right, leak) in enumerate(WRONG):
        ask("wrong_context", f"{i}:{q}", chat, q)
    for i, (chat, q, stay) in enumerate(FOLLOW):
        ask("follow_up", f"{i}:{q}", chat, q)
    for q in LONG:
        ask("length_new_chat", q, [], q)
        ask("length_after_hello", q, HELLO, q)
    for i, (chat, q) in enumerate(GREET):
        ask("greetings", f"{i}:{q}", chat, q)
    for i, (chat, q, right) in enumerate(WHO):
        ask("identity", f"{i}:{q}", chat, q)
    for q, _ in INSTRUCT:
        ask("instructions", q, [], q)
    for q, right in LATE:
        ask("facts_late", q, OWNER, q)
    for q in IMPOSSIBLE:
        ask("impossible", q, [], q)
    out["seconds"] = round(time.time() - began)
    out["scores"] = score(out["replies"])
    stem = path.stem
    (HERE / "logs" / f"chat_suite_{stem}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    show(out["scores"])
    print(f"{len(out['replies'])} replies in {out['seconds']} s -> research/logs/chat_suite_{stem}.json")


def score(replies):
    """Every measure as (count, out of) for rates, or a list of word counts for lengths."""
    by = {}
    for r in replies:
        by.setdefault((r["test"], r["item"]), []).append(r)
    rate = lambda pairs: [sum(bool(f(r["text"])) for key, f in pairs for r in by[key]), sum(len(by[key]) for key, _ in pairs)]
    has = lambda pattern: (lambda t: re.search(pattern, t, re.I))
    s = {"facts_formal: right": rate([(("facts_formal", f), has(r)) for _, f, r in FACTS]),
         "facts_casual: right": rate([(("facts_casual", c), has(r)) for c, _, r in FACTS]),
         "sums_casual: right": rate([(("sums_casual", q), has(rf"\b{n}\b")) for q, n in SUMS]),
         "topic_switch: right": rate([(("topic_switch", f"{i}:{q}"), has(r)) for i, (_, q, r, _) in enumerate(SWITCH)]),
         "topic_switch: old topic leaks in": rate([(("topic_switch", f"{i}:{q}"), has(l)) for i, (_, q, _, l) in enumerate(SWITCH)]),
         "wrong_context: right": rate([(("wrong_context", f"{i}:{q}"), has(r)) for i, (_, q, r, _) in enumerate(WRONG)]),
         "wrong_context: wrong answer leaks in": rate([(("wrong_context", f"{i}:{q}"), has(l)) for i, (_, q, _, l) in enumerate(WRONG)]),
         "follow_up: stays on topic": rate([(("follow_up", f"{i}:{q}"), has(st)) for i, (_, q, st) in enumerate(FOLLOW)]),
         "greetings: short (50 words or fewer)": rate([(("greetings", f"{i}:{q}"), lambda t: 0 < len(t.split()) <= 50) for i, (_, q) in enumerate(GREET)]),
         "greetings: ends after the greeting (one paragraph)": rate([(("greetings", f"{i}:{q}"), lambda t: t and "\n\n" not in t) for i, (_, q) in enumerate(GREET)]),
         "identity: right": rate([(("identity", f"{i}:{q}"), has(r)) for i, (_, q, r) in enumerate(WHO)]),
         "instructions: followed": rate([(("instructions", q), f) for q, f in INSTRUCT]),
         "facts_late: right": rate([(("facts_late", q), has(r)) for q, r in LATE]),
         "impossible questions: declines instead of making up": rate([(("impossible", q), has(DECLINES)) for q in IMPOSSIBLE]),
         "length: words in a new chat": [r["words"] for q in LONG for r in by[("length_new_chat", q)]],
         "length: words after 'Hello!'": [r["words"] for q in LONG for r in by[("length_after_hello", q)]],
         "'The text doesn't say.' with no text given": [sum(bool(re.search(r"doesn.t say", r["text"], re.I)) for r in replies), len(replies)],
         "rambling: cut off at the limit": [sum(r["cut"] for r in replies), len(replies)],
         "rambling: replies repeating a 4-word phrase 3+ times": [sum(repeats(r["text"]) for r in replies), len(replies)]}
    return s


def repeats(text):
    w = re.findall(r"\w+", text.lower())
    grams = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return any(grams.count(g) >= 3 for g in set(grams))


LOWER_IS_BETTER = {"topic_switch: old topic leaks in", "wrong_context: wrong answer leaks in", "rambling: cut off at the limit",
                   "'The text doesn't say.' with no text given",
                   "rambling: replies repeating a 4-word phrase 3+ times"}


def show(scores, other=None):
    for k, v in scores.items():
        if len(v) != 2:   # word counts, not a (count, out of) rate
            print(f"  {k:52s} median {statistics.median(v):5.0f}, mean {statistics.mean(v):5.1f}")
        else:
            print(f"  {k:52s} {v[0]:4d} / {v[1]:<4d} {100 * v[0] / v[1]:5.1f}%")


def compare(candidate, baseline):
    """Each measure: baseline, candidate, and whether the change is bigger than chance (z of 2 or more)."""
    a = json.loads((HERE / "logs" / f"chat_suite_{baseline}.json").read_text(encoding="utf-8"))["scores"]
    b = json.loads((HERE / "logs" / f"chat_suite_{candidate}.json").read_text(encoding="utf-8"))["scores"]
    worse = []
    print(f"{'measure':52s} {baseline:>24s} {candidate:>24s}   verdict")
    for k in a:
        x, y = a[k], b[k]
        if len(x) == 2:   # a rate: (count, out of)
            p1, p2 = x[0] / x[1], y[0] / y[1]
            pool = (x[0] + y[0]) / (x[1] + y[1])
            z = (p2 - p1) / math.sqrt(max(pool * (1 - pool) * (1 / x[1] + 1 / y[1]), 1e-12))
            cells = (f"{x[0]}/{x[1]} {100 * p1:5.1f}%", f"{y[0]}/{y[1]} {100 * p2:5.1f}%")
        else:   # word counts: compare the means
            se = math.sqrt(statistics.variance(x) / len(x) + statistics.variance(y) / len(y))
            z = (statistics.mean(y) - statistics.mean(x)) / max(se, 1e-9)
            cells = (f"mean {statistics.mean(x):6.1f}", f"mean {statistics.mean(y):6.1f}")
        if k in LOWER_IS_BETTER:
            z = -z
        verdict = "better" if z >= 2 else "WORSE" if z <= -2 else "same (within chance)"
        if verdict == "WORSE":
            worse.append(k)
        print(f"{k:52s} {cells[0]:>24s} {cells[1]:>24s}   {verdict} (z {z:+.1f})")
    print("\nworse on:", ", ".join(worse) if worse else "nothing")
    return worse


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "rescore":   # score saved replies again after a measure is added
        f = HERE / "logs" / f"chat_suite_{sys.argv[2]}.json"
        d = json.loads(f.read_text(encoding="utf-8"))
        d["scores"] = score(d["replies"])
        f.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
        show(d["scores"])
    else:
        compare(sys.argv[2], sys.argv[3])
