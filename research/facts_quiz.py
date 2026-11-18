"""Do the normal way and the new way know the same facts once they reach the same score? A fact quiz on 124M models.

    python -P research/facts_quiz.py count     how often each fact appears in the training text (CPU, a few minutes)
    python -P research/facts_quiz.py train     the normal way for about an hour, then the new way until it is as good
    python -P research/facts_quiz.py quiz      both models answer every question (GPU, a few minutes)
    python -P research/facts_quiz.py report    writes research/logs/facts_quiz.md and prints the summary

Training text: 990 MB of data/corpus/mix_v2.txt (research/prepare_more_text.py), the same text for both ways.
The normal way trains for about an hour at the better of the two learning rates tried in test_124m.py. The new
way's score is checked every 2% of its run, and it stops (and is saved) the first time it is as good as the
normal way's final score.

Each fact is asked two ways, e.g. "The capital of France is":
  choice  the model's own probability for each of four answers (Paris, London, Rome, Berlin), per letter so long
          and short answers are treated alike; right if the true answer is the most likely. Guessing gets 25%.
  free    the model continues the sentence by itself, always taking its most likely next piece; right if the
          true answer appears in the first 40 letters. Anything else it says there counts as a wrong answer.
"""
import json, re, sys
from pathlib import Path

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]   # the modules below read their own arguments when imported
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = HERE / "logs"
TEXT_MB = 1000

QUIZ = [   # (sentence to finish, true answer, three wrong answers of the same kind, a word that names the subject)
    ("The capital of France is", "Paris", ["London", "Rome", "Berlin"], "France"),
    ("The capital of Germany is", "Berlin", ["Munich", "Vienna", "Paris"], "Germany"),
    ("The capital of Italy is", "Rome", ["Milan", "Madrid", "Athens"], "Italy"),
    ("The capital of Spain is", "Madrid", ["Barcelona", "Lisbon", "Rome"], "Spain"),
    ("The capital of Japan is", "Tokyo", ["Kyoto", "Beijing", "Seoul"], "Japan"),
    ("The capital of China is", "Beijing", ["Shanghai", "Tokyo", "Hong Kong"], "China"),
    ("The capital of Russia is", "Moscow", ["Saint Petersburg", "Kiev", "Warsaw"], "Russia"),
    ("The capital of England is", "London", ["Manchester", "Paris", "Dublin"], "England"),
    ("The capital of Egypt is", "Cairo", ["Alexandria", "Nairobi", "Baghdad"], "Egypt"),
    ("The capital of Canada is", "Ottawa", ["Toronto", "Vancouver", "Montreal"], "Canada"),
    ("The capital of Australia is", "Canberra", ["Sydney", "Melbourne", "Perth"], "Australia"),
    ("The capital of India is", "New Delhi", ["Mumbai", "Kolkata", "Bangalore"], "India"),
    ("The capital of Greece is", "Athens", ["Sparta", "Rome", "Istanbul"], "Greece"),
    ("The capital of Portugal is", "Lisbon", ["Porto", "Madrid", "Seville"], "Portugal"),
    ("The capital of Ireland is", "Dublin", ["Cork", "Galway", "London"], "Ireland"),
    ("The capital of Norway is", "Oslo", ["Bergen", "Stockholm", "Helsinki"], "Norway"),
    ("The capital of Sweden is", "Stockholm", ["Oslo", "Gothenburg", "Copenhagen"], "Sweden"),
    ("The capital of Kenya is", "Nairobi", ["Mombasa", "Cairo", "Lagos"], "Kenya"),
    ("The capital of Argentina is", "Buenos Aires", ["Santiago", "Lima", "Montevideo"], "Argentina"),
    ("The capital of Turkey is", "Ankara", ["Istanbul", "Athens", "Tehran"], "Turkey"),
    ("The capital of Poland is", "Warsaw", ["Krakow", "Prague", "Berlin"], "Poland"),
    ("The capital of Austria is", "Vienna", ["Salzburg", "Berlin", "Budapest"], "Austria"),
    ("The capital of South Korea is", "Seoul", ["Busan", "Tokyo", "Beijing"], "Korea"),
    ("The capital of Thailand is", "Bangkok", ["Hanoi", "Manila", "Jakarta"], "Thailand"),
    ("The capital of Cuba is", "Havana", ["Kingston", "Santo Domingo", "Miami"], "Cuba"),
    ("The capital of Peru is", "Lima", ["Quito", "Bogota", "Santiago"], "Peru"),
    ("The capital of Brazil is", "Brasilia", ["Rio de Janeiro", "Sao Paulo", "Buenos Aires"], "Brazil"),
    ("The capital of the United States is", "Washington", ["New York", "Chicago", "Boston"], "United States"),
    ("The capital of Scotland is", "Edinburgh", ["Glasgow", "Dublin", "London"], "Scotland"),
    ("The capital of Mexico is", "Mexico City", ["Guadalajara", "Madrid", "Havana"], "Mexico"),
    ("The Eiffel Tower is located in", "Paris", ["London", "Rome", "New York"], "Eiffel"),
    ("The Statue of Liberty is located in", "New York", ["Paris", "London", "Washington"], "Statue of Liberty"),
    ("The Great Wall is located in", "China", ["Japan", "India", "Egypt"], "Great Wall"),
    ("The pyramids of Giza are located in", "Egypt", ["Mexico", "Peru", "Greece"], "Giza"),
    ("The Colosseum is located in", "Rome", ["Athens", "Paris", "Madrid"], "Colosseum"),
    ("The Taj Mahal is located in", "India", ["Egypt", "China", "Turkey"], "Taj Mahal"),
    ("Mount Everest is part of the", "Himalayas", ["Alps", "Andes", "Rocky Mountains"], "Everest"),
    ("The largest ocean on Earth is the", "Pacific", ["Atlantic", "Indian", "Arctic"], "largest ocean"),
    ("The largest desert in Africa is the", "Sahara", ["Gobi", "Kalahari", "Atacama"], "desert"),
    ("Kangaroos are native to", "Australia", ["Africa", "India", "Brazil"], "kangaroo"),
    ("Most of the Amazon rainforest is in", "Brazil", ["Mexico", "India", "Kenya"], "Amazon"),
    ("Tokyo is the capital city of", "Japan", ["China", "Korea", "Thailand"], "Tokyo"),
    ("Paris is the capital city of", "France", ["Italy", "Spain", "Belgium"], "Paris"),
    ("The official language of Brazil is", "Portuguese", ["Spanish", "French", "English"], "Brazil"),
    ("Most people in Mexico speak", "Spanish", ["Portuguese", "French", "English"], "Mexico"),
    ("The currency of Japan is the", "yen", ["yuan", "won", "dollar"], "Japan"),
    ("The currency of the United Kingdom is the", "pound", ["euro", "dollar", "franc"], "currency"),
    ("Many countries in Europe use a currency called the", "euro", ["dollar", "yen", "peso"], "currency"),
    ("The United States declared independence in", "1776", ["1812", "1865", "1492"], "independence"),
    ("Christopher Columbus first reached the Americas in", "1492", ["1776", "1066", "1914"], "Columbus"),
    ("World War II ended in", "1945", ["1918", "1939", "1965"], "World War II"),
    ("World War I began in", "1914", ["1939", "1945", "1870"], "World War I"),
    ("The Berlin Wall fell in", "1989", ["1945", "1961", "2001"], "Berlin Wall"),
    ("The first humans landed on the Moon in", "1969", ["1959", "1979", "1989"], "landed on the Moon"),
    ("The Battle of Hastings took place in", "1066", ["1492", "1215", "1776"], "Hastings"),
    ("The French Revolution began in", "1789", ["1776", "1848", "1914"], "French Revolution"),
    ("The first President of the United States was", "George Washington", ["Abraham Lincoln", "Thomas Jefferson", "John Adams"], "first president"),
    ("The President of the United States during the Civil War was", "Abraham Lincoln", ["George Washington", "Thomas Jefferson", "Theodore Roosevelt"], "Civil War"),
    ("The theory of relativity was developed by", "Albert Einstein", ["Isaac Newton", "Charles Darwin", "Galileo Galilei"], "relativity"),
    ("The theory of evolution by natural selection was proposed by", "Charles Darwin", ["Isaac Newton", "Albert Einstein", "Louis Pasteur"], "natural selection"),
    ("The laws of motion and universal gravitation were described by", "Isaac Newton", ["Albert Einstein", "Galileo Galilei", "Charles Darwin"], "gravitation"),
    ("Penicillin was discovered by", "Alexander Fleming", ["Louis Pasteur", "Marie Curie", "Joseph Lister"], "penicillin"),
    ("The telephone was invented by", "Alexander Graham Bell", ["Thomas Edison", "Nikola Tesla", "Samuel Morse"], "telephone"),
    ("The first person to walk on the Moon was", "Neil Armstrong", ["Buzz Aldrin", "Yuri Gagarin", "John Glenn"], "walk on the moon"),
    ("The first human to travel into space was", "Yuri Gagarin", ["Neil Armstrong", "John Glenn", "Alan Shepard"], "first human in space"),
    ("Romeo and Juliet was written by", "William Shakespeare", ["Charles Dickens", "Jane Austen", "Mark Twain"], "Romeo and Juliet"),
    ("Pride and Prejudice was written by", "Jane Austen", ["Charles Dickens", "Emily Bronte", "Mark Twain"], "Pride and Prejudice"),
    ("Oliver Twist was written by", "Charles Dickens", ["William Shakespeare", "Jane Austen", "Mark Twain"], "Oliver Twist"),
    ("The Adventures of Tom Sawyer was written by", "Mark Twain", ["Charles Dickens", "Herman Melville", "Jack London"], "Tom Sawyer"),
    ("The Mona Lisa was painted by", "Leonardo da Vinci", ["Vincent van Gogh", "Pablo Picasso", "Michelangelo"], "Mona Lisa"),
    ("The ceiling of the Sistine Chapel was painted by", "Michelangelo", ["Leonardo da Vinci", "Raphael", "Rembrandt"], "Sistine Chapel"),
    ("The Starry Night was painted by", "Vincent van Gogh", ["Claude Monet", "Pablo Picasso", "Leonardo da Vinci"], "Starry Night"),
    ("Mozart and Beethoven were famous", "composers", ["painters", "scientists", "explorers"], "Beethoven"),
    ("The light bulb is most often credited to", "Thomas Edison", ["Nikola Tesla", "Alexander Graham Bell", "Benjamin Franklin"], "light bulb"),
    ("Marie Curie is famous for her research on", "radioactivity", ["electricity", "evolution", "gravity"], "Curie"),
    ("Martin Luther King Jr. was a leader of the", "civil rights movement", ["French Revolution", "space program", "Roman Empire"], "Martin Luther King"),
    ("Julius Caesar was a leader of ancient", "Rome", ["Greece", "Egypt", "China"], "Caesar"),
    ("The pharaohs were the rulers of ancient", "Egypt", ["Rome", "Greece", "China"], "pharaoh"),
    ("The Olympic Games began in ancient", "Greece", ["Rome", "Egypt", "China"], "Olympic"),
    ("Water is made of hydrogen and", "oxygen", ["nitrogen", "carbon", "helium"], "hydrogen"),
    ("The chemical symbol for gold is", "Au", ["Ag", "Fe", "Pb"], "symbol for gold"),
    ("The chemical symbol for iron is", "Fe", ["Ir", "Au", "Pb"], "symbol for iron"),
    ("The chemical symbol for sodium is", "Na", ["So", "Sd", "Ni"], "symbol for sodium"),
    ("The planet closest to the Sun is", "Mercury", ["Venus", "Mars", "Jupiter"], "closest to the sun"),
    ("The largest planet in our solar system is", "Jupiter", ["Saturn", "Earth", "Neptune"], "largest planet"),
    ("The planet known as the Red Planet is", "Mars", ["Venus", "Jupiter", "Mercury"], "red planet"),
    ("The planet famous for its rings is", "Saturn", ["Jupiter", "Mars", "Venus"], "rings"),
    ("Plants make their own food through a process called", "photosynthesis", ["respiration", "digestion", "evaporation"], "plants"),
    ("The powerhouse of the cell is the", "mitochondria", ["nucleus", "ribosome", "membrane"], "powerhouse"),
    ("Plants take in carbon dioxide and release", "oxygen", ["nitrogen", "hydrogen", "methane"], "carbon dioxide"),
    ("The center of an atom is called the", "nucleus", ["electron", "shell", "molecule"], "atom"),
    ("The negatively charged particles in an atom are called", "electrons", ["protons", "neutrons", "photons"], "negatively charged"),
    ("The force that pulls objects toward the Earth is called", "gravity", ["friction", "magnetism", "inertia"], "force"),
    ("Light travels at about 300,000 kilometers per", "second", ["hour", "minute", "day"], "speed of light"),
    ("DNA stands for deoxyribonucleic", "acid", ["base", "protein", "sugar"], "deoxyribonucleic"),
    ("The hardest natural substance on Earth is", "diamond", ["gold", "iron", "quartz"], "hardest"),
    ("The largest organ of the human body is the", "skin", ["liver", "heart", "brain"], "largest organ"),
    ("Blood is pumped around the body by the", "heart", ["liver", "lungs", "kidneys"], "pump"),
    ("The study of living things is called", "biology", ["geology", "astronomy", "chemistry"], "living things"),
    ("The study of the stars and planets is called", "astronomy", ["biology", "geology", "zoology"], "stars"),
    ("Water boils at 100 degrees", "Celsius", ["Fahrenheit", "Kelvin", "Rankine"], "boil"),
    ("An animal that eats only plants is called a", "herbivore", ["carnivore", "omnivore", "predator"], "plants"),
    ("Frogs, toads and salamanders are", "amphibians", ["reptiles", "mammals", "insects"], "salamander"),
    ("Whales and dolphins are", "mammals", ["fish", "reptiles", "amphibians"], "dolphin"),
    ("Insects have six", "legs", ["wings", "eyes", "heads"], "insects"),
    ("The largest animal that has ever lived is the", "blue whale", ["African elephant", "great white shark", "giraffe"], "largest animal"),
    ("The tallest animal in the world is the", "giraffe", ["elephant", "horse", "camel"], "tallest animal"),
    ("A baby cat is called a", "kitten", ["puppy", "calf", "cub"], "baby cat"),
    ("A baby dog is called a", "puppy", ["kitten", "foal", "chick"], "baby dog"),
    ("Bees make", "honey", ["milk", "silk", "wool"], "bees"),
    ("The Earth travels around the", "Sun", ["Moon", "Mars", "stars"], "orbit"),
    ("Earth's only natural satellite is the", "Moon", ["Sun", "Mars", "comet"], "satellite"),
    ("There are 60 minutes in an", "hour", ["day", "week", "year"], "minutes"),
    ("There are seven days in a", "week", ["month", "year", "decade"], "seven days"),
    ("A century is a period of 100", "years", ["days", "months", "weeks"], "century"),
    ("A kilometer is equal to 1,000", "meters", ["centimeters", "grams", "liters"], "kilometer"),
    ("The adult human body has 206", "bones", ["muscles", "teeth", "organs"], "206"),
]
FREE_LETTERS, CAP, WINDOW = 40, 200_000, 300


def training_text():
    import numpy as np
    data = np.memmap(HERE.parent / "data" / "corpus" / "mix_v2.txt", dtype=np.uint8, mode="r")
    return b"".join(data[i << 20:(i + 1) << 20].tobytes() for i in range(TEXT_MB) if i % 100 != 99)


def count():
    """For each fact: how many times the subject word and the true answer appear within 300 letters of each other."""
    text = training_text().lower()
    counts = {}
    for prompt, answer, _, key in QUIZ:
        spots = [m.start() for m in re.finditer(rb"\b" + re.escape(key.lower().encode()) + rb"\b", text)]
        sample = spots[::max(1, len(spots) // CAP)]   # a very common subject word: check an even sample, then scale up
        near = re.compile(rb"\b" + re.escape(answer.lower().encode()) + rb"\b")
        hits = sum(1 for p in sample if near.search(text, max(0, p - WINDOW), p + WINDOW))
        counts[prompt] = round(hits * len(spots) / max(1, len(sample)))
        print(f"{counts[prompt]:8,}  {prompt} {answer}", flush=True)
    (LOG / "facts_counts.json").write_text(json.dumps({"letters": len(text), "counts": counts}, indent=1), encoding="utf-8")


def train():
    import final_test as ft
    import test_124m as t
    per = lambda size: json.loads((HERE.parent / "data" / "pieces1g" / size.split("-")[1] / "manifest.json").read_text())["letters per piece, val"]
    normals = [r for name, size, cfg, n in t.RUNS if name.startswith("normal") for r in ft.records("t124", size, cfg, n * t.TWO)]
    lr = min(normals, key=lambda r: r["final"])["config"]["lr"]   # the better of the two rates tried for 8 minutes
    ft.REST = 150
    normal = ("124m-gpt2-1g", {"lr": lr, "compiled": True, "save": "facts_124m_normal"}, 1880)   # about an hour
    ft.one("facts", normal[0], normal[1], normal[2] * t.TWO)
    target = ft.records("facts", *normal[:2])[0]["final"] / per(normal[0])
    new = "124m-bpe4096-1g"   # the new way stops, and is saved, the moment its score is as good as the normal way's
    ft.one("facts", new, {**t.NEW, "save": "facts_124m_new", "stop_at": round(target * per(new), 4)}, 1600 * t.TWO)


def train_fp8():
    """The new way again with 8-bit matrix multiplies, same text, same plan, stopping at the same score."""
    import final_test as ft
    import test_124m as t
    per = lambda size: json.loads((HERE.parent / "data" / "pieces1g" / size.split("-")[1] / "manifest.json").read_text())["letters per piece, val"]
    target = ft.records("facts", "124m-gpt2-1g")[0]["final"] / per("124m-gpt2-1g")
    new = "124m-bpe4096-1g"
    ft.REST = 60
    ft.one("facts", new, {**t.NEW, "fp8": True, "save": "facts_124m_new_fp8", "stop_at": round(target * per(new), 4)}, 1600 * t.TWO)


def quiz(names=("facts_124m_normal", "facts_124m_new")):
    import torch
    import compare_text as ct
    ct.LETTERS = FREE_LETTERS
    old = LOG / "facts_quiz.json"
    out = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    for name in names:
        model, encode, decode, real = ct.load(name)
        answers = []
        for prompt, answer, wrong, _ in QUIZ:
            scores = []
            p = encode(prompt)
            for option in [answer] + wrong:
                o = encode(" " + option)
                with torch.no_grad():
                    logp = torch.log_softmax(model(torch.tensor([p + o]).cuda())[0, :, :real].float(), -1)
                at = torch.arange(len(p) - 1, len(p) + len(o) - 1)
                scores.append(logp[at, torch.tensor(o)].sum().item() / len(" " + option))   # per letter
            said = ct.continue_text(model, encode, decode, real, prompt, False)
            answers.append({"prompt": prompt, "picked": ([answer] + wrong)[max(range(4), key=lambda i: scores[i])],
                            "choice": scores[0] == max(scores), "said": said.strip(),
                            "free": re.search(r"\b" + re.escape(answer.lower()) + r"\b", said.lower()) is not None})
        out[name] = answers
        del model
        torch.cuda.empty_cache()
    (LOG / "facts_quiz.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def report():
    import final_test as ft
    import test_124m as t
    quiz_out = json.loads((LOG / "facts_quiz.json").read_text(encoding="utf-8"))
    counted = json.loads((LOG / "facts_counts.json").read_text(encoding="utf-8"))
    runs = {"facts_124m_normal": ("124m-gpt2-1g", "normal way"), "facts_124m_new": ("124m-bpe4096-1g", "new way")}
    lines, read = [], {}
    for name, (size, label) in runs.items():
        r = [r for r in ft.records("facts", size) if r["config"].get("save") == name][0]
        manifest = json.loads((HERE.parent / "data" / "pieces1g" / size.split("-")[1] / "manifest.json").read_text())
        read[name] = r.get("tokens_trained", r["tokens"]) * manifest["letters per piece, train"]
        a = quiz_out[name]
        lines.append(f"{label}: {r['seconds']:.0f} s of training, {read[name] / 1e6:.0f} million letters read, "
                     f"score {r['final'] / manifest['letters per piece, val']:.4f} per letter; "
                     f"choice {sum(x['choice'] for x in a)}/{len(a)} right, free {sum(x['free'] for x in a)}/{len(a)} right")
    n, w = quiz_out["facts_124m_normal"], quiz_out["facts_124m_new"]
    for kind in ("choice", "free"):
        both = sum(a[kind] and b[kind] for a, b in zip(n, w))
        only_n, only_w = sum(a[kind] and not b[kind] for a, b in zip(n, w)), sum(b[kind] and not a[kind] for a, b in zip(n, w))
        lines.append(f"{kind}: both right {both}, only the normal way {only_n}, only the new way {only_w}, neither {len(n) - both - only_n - only_w}")
    counts = counted["counts"]
    middle = sorted(counts.values())[len(counts) // 2]
    for label, keep in (("facts seen more often than the middle", lambda c: c > middle), ("the rest", lambda c: c <= middle)):
        idx = [i for i, (p, *_) in enumerate(QUIZ) if keep(counts[p])]
        lines.append(f"{label} ({len(idx)} facts): choice normal {sum(n[i]['choice'] for i in idx)}, new {sum(w[i]['choice'] for i in idx)}; "
                     f"free normal {sum(n[i]['free'] for i in idx)}, new {sum(w[i]['free'] for i in idx)}")
    table = ["| Fact | Times in the text | Seen by normal (expected) | Seen by new (expected) | Normal picked | New picked | Normal said | New said |",
             "|---|---|---|---|---|---|---|---|"]
    share = {k: v / counted["letters"] for k, v in read.items()}
    for i, (prompt, answer, _, _) in enumerate(QUIZ):
        c = counts[prompt]
        cell = lambda s: s.replace("|", "/")[:FREE_LETTERS]
        table.append(f"| {prompt} **{answer}** | {c:,} | {c * share['facts_124m_normal']:.1f} | {c * share['facts_124m_new']:.1f} | "
                     f"{'right' if n[i]['choice'] else n[i]['picked']} | {'right' if w[i]['choice'] else w[i]['picked']} | "
                     f"{cell(n[i]['said'])} | {cell(w[i]['said'])} |")
    (LOG / "facts_quiz.md").write_text("\n".join(lines) + "\n\n" + "\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    assert len({q[0] for q in QUIZ}) == len(QUIZ) and all(len(q[2]) == 3 and q[1] not in q[2] for q in QUIZ)
    if ARGS[:1] == ["quiz"] and ARGS[1:]:
        quiz(ARGS[1:])
    else:
        {"count": count, "train": train, "train_fp8": train_fp8, "quiz": quiz, "report": report}[ARGS[0] if ARGS else "report"]()
