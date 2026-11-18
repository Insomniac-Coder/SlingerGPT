"""Tests SlingerGPT 1 pico once its training has finished.

    python -P research/test_slinger.py run [name]   the model answers and is scored (GPU, a few minutes);
                                                    name gpt2: OpenAI's GPT-2 small, scored the same way;
                                                    name: slingergpt1_pico (default) or slingergpt1_pico_preview1
    python -P research/test_slinger.py extra [name] arithmetic, who it is, fact lookup (needs research/lookup.py build)
    python -P research/test_slinger.py broad [name] only the 30 broad lookup questions, added to the extra results
    python -P research/slinger_report.py         writes research/slingergpt1_pico_report.pdf from the results

Three tests:
  quiz           the 117 fact questions of research/facts_quiz.py: picking the likeliest of four answers, and
                 finishing the sentence in its own words (first 40 letters, always the likeliest next piece)
  conversation   twelve questions written as "User: ...\nAssistant:", answered twice: always the likeliest next
                 piece, and with some randomness (temperature 0.7, the 40 likeliest pieces, the same seed)
  held-out text  wrongness per letter on about 2.4 million letters of the documents held out from every source (one
                 in a hundred, never trained on), in the shares the training read them, using at most the first 4,000
                 letters of each document so the sample spans many documents. Each document starts with the
                 end-of-text piece; total wrongness is divided by the letters (bytes) of the text.
"""
import json, re, sys
from pathlib import Path

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NAME = ARGS[1] if len(ARGS) > 1 else "slingergpt1_pico"
STEM = Path(NAME).name   # a candidate under models/candidates/ keeps a plain file name for its results
OUT = HERE / "logs" / ("slinger_test.json" if NAME == "slingergpt1_pico" else f"slinger_test_{STEM}.json")
CHAT = ["Hi! How are you today?", "What is photosynthesis?", "Why is the sky blue?", "What is 12 + 7?",
        "If all cats are animals and Tom is a cat, is Tom an animal?", "What does a computer's CPU do?",
        "What is the capital of Japan?", "Can you give me a tip for studying for exams?", "Explain what a computer virus is.",
        "How many legs does a spider have?", "What is the difference between weather and climate?",
        "I feel a bit sad today. Can you cheer me up?"]
TEXT_LETTERS, DOC_LETTERS = 2_400_000, 4_000   # held-out letters in all; at most this much of any one document


class Model:
    def __init__(self):
        import torch
        import arch
        from tokenizers import Tokenizer
        from torch.nn.attention import SDPBackend
        # cuDNN's attention kernel plans afresh (tens of ms) for every new text length, and the tests make a new length
        # at nearly every call; the memory-efficient kernel gives the same results without that (see chat_server.py)
        arch.ATTENTION = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        self.torch = torch
        saved = torch.load(ROOT / "models" / f"{NAME}.pt", map_location="cpu", weights_only=False)
        self.net = arch.build(**{**saved["cfg"], "fp8": False})   # answering runs in 16-bit
        self.net.load_state_dict(saved["model"])
        self.net.eval()
        tok = Tokenizer.from_file(str(ROOT / "models" / f"{NAME}.tokenizer.json"))
        self.encode, self.decode, self.end = (lambda s: tok.encode(s).ids), tok.decode, 0

    def logp(self, ids):
        """Log-probabilities of every next piece for one row of up to 1,024 pieces."""
        with self.torch.no_grad():
            out = self.net(self.torch.tensor([ids[-1024:]]).cuda())[0].float()
        return self.torch.log_softmax(out, -1)

    def wrongness(self, text):
        """Total -ln p over the text's pieces (after an end-of-text piece), in windows of 1,024."""
        ids, total = [self.end] + self.encode(text), 0.0
        for at in range(0, len(ids) - 1, 1023):
            part = ids[at:at + 1024]
            lp = self.logp(part)
            total -= lp[self.torch.arange(len(part) - 1).cuda(), self.torch.tensor(part[1:]).cuda()].sum().item()
        return total

    def option(self, prompt, answer):
        p, o = self.encode(prompt), self.encode(" " + answer)
        lp = self.logp(p + o)
        return sum(lp[len(p) - 1 + i, o[i]].item() for i in range(len(o))) / len(" " + answer)

    def write(self, prompt, letters, temperature=0.0, stop=None, lead=False):
        torch = self.torch
        torch.manual_seed(0)
        ids = ([self.end] if lead else []) + self.encode(prompt)   # lead: an end-of-text piece first, as the chat page sends
        start = len(ids)
        while len(self.decode(ids[start:])) < letters:
            scores = self.logp(ids)[-1]
            if temperature:
                top = torch.topk(scores / temperature, 40)
                nxt = int(top.indices[torch.multinomial(torch.softmax(top.values, -1), 1)])
            else:
                nxt = int(scores.argmax())
            if nxt == self.end:
                break
            ids.append(nxt)
            if stop and stop in self.decode(ids[start:]):
                break
        text = self.decode(ids[start:])
        return (text.split(stop)[0] if stop else text)[:letters].strip()


class GPT2(Model):
    """OpenAI's GPT-2 small (models/gpt2_openai/: the Hugging Face weights, downloaded with the owner's OK on
    15 November), scored exactly as SlingerGPT: the same held-out documents per letter, quiz and questions. It reads
    with its own 50,257 word pieces and its own end-of-text piece; it was never taught the User/Assistant layout."""
    def __init__(self):
        import torch, tiktoken
        from transformers import GPT2LMHeadModel
        self.torch = torch
        self.net = GPT2LMHeadModel.from_pretrained(ROOT / "models" / "gpt2_openai", local_files_only=True).cuda().eval()
        enc = tiktoken.get_encoding("gpt2")
        self.encode, self.decode, self.end = enc.encode_ordinary, enc.decode, enc.eot_token

    def logp(self, ids):
        with self.torch.no_grad():   # 32-bit throughout
            out = self.net(self.torch.tensor([ids[-1024:]]).cuda()).logits[0].float()
        return self.torch.log_softmax(out, -1)


def held_out():
    """Held-out documents of every source, about TEXT_LETTERS in the shares the training read them.
    SLINGER_HELD_OUT=<dir> (environment): the 492M's held-out text instead (prepare_492m.py's output), weighted by
    each source's share of the whole run (research/492m/mix_492m.json). Wrongness is per letter, so any model with
    any word-piece list is scored on the same text."""
    import os
    import numpy as np
    from tokenizers import Tokenizer
    data = Path(os.environ.get("SLINGER_HELD_OUT") or ROOT / "data" / "slinger")
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if "share" in manifest:   # SlingerGPT 1 pico's text
        share = manifest["share"]
    else:                     # the 492M's text: the run's overall share of each source that has held-out text
        mix = json.loads((HERE / "492m" / "mix_492m.json").read_text(encoding="utf-8"))
        share = {}
        for phase, shares in mix["phases"].items():
            for k, v in shares.items():
                share[k] = share.get(k, 0.0) + mix["billions"][phase] * v / 100
        share = {k: v / sum(share.values()) for k, v in share.items()
                 if manifest["sources"].get(k, {}).get("val", {}).get("pieces")}
    out = {}
    for name, s in share.items():
        ids = np.memmap(data / name / "val.bin", dtype=np.uint16, mode="r")[:3_000_000].tolist()
        docs, cur, got = [], [], 0
        for i in ids:
            if i == 0:
                if cur:   # the start of each document only, so the sample spans many documents
                    docs.append(tok.decode(cur)[:DOC_LETTERS])
                    got += len(docs[-1].encode("utf-8"))
                cur = []
                if got >= TEXT_LETTERS * s:
                    break
            else:
                cur.append(i)
        out[name] = docs
    return out


def run():
    import facts_quiz as fq
    m = GPT2() if NAME == "gpt2" else Model()
    r = {"held_out": {}, "quiz": [], "chat": []}
    for name, docs in held_out().items():
        letters = sum(len(d.encode("utf-8")) for d in docs)
        r["held_out"][name] = {"documents": len(docs), "letters": letters, "wrongness": sum(m.wrongness(d) for d in docs)}
        print(name, round(r["held_out"][name]["wrongness"] / letters, 4), flush=True)
    for prompt, answer, wrong, _ in fq.QUIZ:
        scores = [m.option(prompt, o) for o in [answer] + wrong]
        said = m.write(prompt, fq.FREE_LETTERS)
        r["quiz"].append({"prompt": prompt, "picked": ([answer] + wrong)[scores.index(max(scores))], "choice": scores[0] == max(scores),
                          "said": said, "free": re.search(r"\b" + re.escape(answer.lower()) + r"\b", said.lower()) is not None})
    print("quiz", sum(q["choice"] for q in r["quiz"]), sum(q["free"] for q in r["quiz"]), flush=True)
    for q in CHAT:
        prompt = f"User: {q}\nAssistant:"
        r["chat"].append({"question": q, "steady": m.write(prompt, 400, 0.0, "\nUser:"), "varied": m.write(prompt, 400, 0.7, "\nUser:")})
    OUT.write_text(json.dumps(r, indent=1), encoding="utf-8")


# The 30 everyday questions of the fact lookup test, each with what a right answer contains (research/plan_fact_lookup.md)
LOOKUP_QUESTIONS = [
    ("Who wrote Pride and Prejudice?", r"austen"), ("What is the capital of Australia?", r"canberra"),
    ("Who painted the Mona Lisa?", r"leonardo|da vinci"), ("What is the largest planet in the Solar System?", r"jupiter"),
    ("In what year did World War II end?", r"1945"), ("Who wrote Romeo and Juliet?", r"shakespeare"),
    ("What is the chemical symbol for gold?", r"\bau\b"), ("How many legs does a spider have?", r"\beight\b|\b8\b"),
    ("What is the tallest mountain in the world?", r"everest"), ("Who discovered penicillin?", r"fleming"),
    ("What is the capital of Canada?", r"ottawa"), ("Who was the first person to walk on the Moon?", r"armstrong"),
    ("What is the largest ocean on Earth?", r"pacific"), ("Which planet is known as the Red Planet?", r"\bmars\b"),
    ("What is the boiling point of water in Celsius?", r"\b100\b"), ("Who developed the theory of relativity?", r"einstein"),
    ("What is the capital of Japan?", r"tokyo"), ("What language is spoken in Brazil?", r"portuguese"),
    ("What is the currency of Japan?", r"\byen\b"), ("Who invented the telephone?", r"\bbell\b"),
    ("What is the longest river in Africa?", r"\bnile\b"), ("What gas do plants take in from the air?", r"carbon dioxide|\bco2\b"),
    ("Who was the first President of the United States?", r"washington"), ("Who painted The Starry Night?", r"gogh"),
    ("What is the capital of Italy?", r"\brome\b"), ("How many continents are there?", r"\bseven\b|\b7\b"),
    ("What is the hardest natural substance?", r"diamond"), ("Which element has the atomic number 1?", r"hydrogen"),
    ("What is the capital of Egypt?", r"cairo"), ("Who discovered gravity when an apple fell?", r"newton")]
# Who it is: fresh and after a long answer; a right answer names SlingerGPT, or for the maker questions Ishtdeep/Gunslinger
IDENTITY_QUESTIONS = [("What's your name?", "name"), ("who are you", "name"), ("Who made you?", "maker"), ("Are you ChatGPT?", "name"),
                      ("What should I call you?", "name"), ("who created you", "maker"), ("Tell me about yourself.", "name"),
                      ("Are you made by Google?", "maker"), ("Who is Gunslinger?", "maker"), ("what do people call you", "name")]
# Broad questions, as people ask them on the chat page (added 16 November: with "Look it up" on, it said "The text
# doesn't say." to nearly all of these although the passage had the answer). Their articles are left out of practice.
BROAD_QUESTIONS = [
    ("What is photosynthesis?", r"plant|light|sugar|carbohydrate"), ("What is gravity?", r"force|attract|mass|pull"),
    ("Who was Albert Einstein?", r"physicist|scientist|relativity"), ("What is a volcano?", r"magma|lava|erupt|opening"),
    ("What is DNA?", r"genetic|molecule|gene|acid"), ("What is the Moon?", r"satellite|orbit"),
    ("What is democracy?", r"government|vote|rule|citizens"), ("Who was Isaac Newton?", r"physicist|mathematician|scientist"),
    ("What is an atom?", r"matter|particle|element|nucleus|smallest"), ("What is electricity?", r"charge|electron|current"),
    ("What is a computer?", r"machine|device|calculat|program|data"), ("What is the internet?", r"network"),
    ("What is a black hole?", r"gravity|light|escape|spacetime"), ("What is evolution?", r"species|generation|organism|inherit"),
    ("What is a virus?", r"infect|cell|host|pathogen"), ("Who was William Shakespeare?", r"playwright|poet|writer|dramatist"),
    ("What is climate change?", r"temperature|warming|greenhouse"), ("What is the Sun?", r"\bstar\b|hydrogen|plasma"),
    ("What is mathematics?", r"number|quantit|structure|shape|study"), ("What is an earthquake?", r"shak|ground|tectonic|plate|seismic"),
    ("Who was Leonardo da Vinci?", r"painter|artist|inventor|polymath"), ("What is a planet?", r"orbit|\bstar\b|body"),
    ("What is oxygen?", r"element|\bgas\b|chemical"), ("What is the heart?", r"organ|blood|pump|muscle"),
    ("What is a mammal?", r"animal|milk|hair|vertebrate"), ("What is economics?", r"goods|services|production|social science"),
    ("What is a rainbow?", r"light|colou?r|droplet|spectrum"), ("Who was Mahatma Gandhi?", r"leader|independence|lawyer|activist"),
    ("What is a battery?", r"energy|electric|chemical|cell"), ("What is the brain?", r"organ|nervous|skull|neuron")]


# Questions with no real answer (added 16 November: asked "Who is the president of the Moon?", it made one up). A
# good reply says it does not know or that there is no such thing. Their words are left out of practice.
NONSENSE_QUESTIONS = ["Who is the president of the Moon?", "What is the capital of the Pacific Ocean?",
                      "Who was the first person to walk on the Sun?", "In what year did Albert Einstein win the World Cup?",
                      "What is the name of the queen of Mars?", "Who invented the colour blue?",
                      "When did William Shakespeare visit Australia?", "Who won the 1850 FIFA World Cup?",
                      "What is the population of Atlantis?", "What is the phone number of the Eiffel Tower?"]
DECLINES = ("don't know", "doesn't say", "there is no", "there was no", "there are no", "does not exist", "doesn't exist",
            "not a real", "no such", "fictional", "never")


def nonsense_test(ask):
    out = [{"question": q, "said": (s := ask(q)), "declined": any(d in s.lower() for d in DECLINES)} for q in NONSENSE_QUESTIONS]
    print("no real answer: declined", sum(x["declined"] for x in out), "/", len(out), flush=True)
    return out


def kind(key, said):
    """A lookup-test answer: right (says what a right answer contains, early on), not said ("The text doesn't say."
    or "I don't know."), or wrong."""
    s = said.lower()
    return "right" if re.search(key, s[:200]) else "not said" if "doesn't say" in s or "don't know" in s else "wrong"


def lookup_test(questions, ask, system):
    """Each question without and with the top passage of the search in the System line, as the chat page sends it."""
    import finetune_slinger as ft
    import lookup
    out = []
    for q, key in questions:
        found = lookup.search(q, 1)
        title, passage = (found[0][0], found[0][1]) if found else ("", "")
        with_line = f"System: {ft.SYSTEM_LINE} {lookup.PASSAGE.format(title=title, passage=passage)}\n" if found else system
        alone, looked = ask(q), ask(q, line=with_line)
        out.append({"question": q, "article": title, "passage_has_it": re.search(key, passage.lower()) is not None,
                    "without": alone, "without_kind": kind(key, alone), "with": looked, "with_kind": kind(key, looked)})
    count = lambda k, v: sum(x[k] == v for x in out)
    print(f"lookup ({len(out)}) right without/with:", count("without_kind", "right"), count("with_kind", "right"),
          " not said with:", count("with_kind", "not said"), flush=True)
    return out


def extra(only=None):
    """Arithmetic (50 sums), who it is (20 questions) and fact lookup (30 questions with and without a Wikipedia
    passage; the 117-question quiz with the passage in front), all in the chat page's layout: an end-of-text piece,
    its "System:" line, "User: ...", "Assistant:". Always the likeliest next piece. -> logs/slinger_extra_<name>.json"""
    import facts_quiz as fq
    import finetune_slinger as ft
    import lookup
    m = Model()
    system = f"System: {ft.SYSTEM_LINE}\n"
    ask = lambda q, before="", line=system: m.write(f"{line}{before}User: {q}\nAssistant:", 600, 0.0, "\nUser:", lead=True)
    number = lambda s: (re.findall(r"-?\d[\d,]*", s.split("The answer is")[-1]) or ["none"])[-1].replace(",", "")
    if only:   # add one part ("broad" or "nonsense") to this model's earlier results
        out = HERE / "logs" / f"slinger_extra_{STEM}.json"
        r = json.loads(out.read_text(encoding="utf-8"))
        r[only] = lookup_test(BROAD_QUESTIONS, ask, system) if only == "broad" else nonsense_test(ask)
        return out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    r = {"arithmetic": [], "identity": [], "lookup": [], "quiz_with_passage": []}
    for q, _, result in ft.arithmetic_test():
        said = ask(q)
        r["arithmetic"].append({"question": q, "result": result, "said": said, "right": number(said) == str(result)})
    print("arithmetic", sum(x["right"] for x in r["arithmetic"]), "/ 50", flush=True)
    long_q = "Can you explain how the water cycle works?"
    before = f"User: {long_q}\nAssistant: {ask(long_q)}\n"
    for q, kind in IDENTITY_QUESTIONS:
        for where, prefix in (("fresh", ""), ("after a long answer", before)):
            said = ask(q, prefix)
            key = r"ishtdeep|gunslinger" if kind == "maker" else r"slingergpt|slinger gpt"
            r["identity"].append({"question": q, "where": where, "said": said, "right": re.search(key, said.lower()) is not None})
    print("identity", sum(x["right"] for x in r["identity"]), "/ 20", flush=True)
    r["lookup"] = lookup_test(LOOKUP_QUESTIONS, ask, system)
    r["broad"] = lookup_test(BROAD_QUESTIONS, ask, system)
    r["nonsense"] = nonsense_test(ask)
    for prompt, answer, wrong, subject in fq.QUIZ:
        found = lookup.search(prompt, 1)
        text = (found[0][1] + "\n\n") if found else ""
        scores = [m.option(text + prompt, o) for o in [answer] + wrong]
        said = m.write(text + prompt, fq.FREE_LETTERS)
        r["quiz_with_passage"].append({"prompt": prompt, "article": found[0][0] if found else "", "choice": scores[0] == max(scores),
                                       "said": said, "free": re.search(r"\b" + re.escape(answer.lower()) + r"\b", said.lower()) is not None})
    print("quiz with passage", sum(q["choice"] for q in r["quiz_with_passage"]), sum(q["free"] for q in r["quiz_with_passage"]), flush=True)
    (HERE / "logs" / f"slinger_extra_{STEM}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if ARGS[:1] == ["run"]:
        run()
    elif ARGS[:1] == ["extra"]:
        extra()
    elif ARGS[:1] in (["broad"], ["nonsense"]):
        extra(only=ARGS[0])
