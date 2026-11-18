"""The SlingerGPT chat page for the models in models/: pick one, ask it something, watch it write.

    python -P chat_server.py            then open http://127.0.0.1:8765

SlingerGPT answers as "Assistant" in the "User: ... / Assistant: ..." layout of its conversation fine-tune and
stops when it ends its turn. The earlier experiments learned conversation from dialogue scripts, so for them chat
mode writes "User: ...\nFriend:" and stops when the model's line ends. Continue mode gives the model your text as
the start of a document and lets it carry on. One reply is written at a time, and one model is on the graphics
card at a time.
"""
import base64, hmac, json, math, sys, threading, time
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE / "research"))   # research/arch.py builds SlingerGPT
import tiny_gpt as t
import gpt128 as g
import arch
import lookup   # research/lookup.py: the Wikipedia search behind "Look it up"

# cuDNN's attention kernel, arch.py's first choice for training, works out a new plan (tens of milliseconds) for every
# new text length. Writing makes a new length at every piece, so here SlingerGPT uses the memory-efficient kernel:
# the same results, without the planning.
arch.ATTENTION = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

MODELS, WEB, PORT = HERE / "models", HERE / "web", 8765
YOU, BOT = "User", "Friend"   # speaker names in chat mode for the earlier experiments
TURN = "\nUser:"              # SlingerGPT starting your next line means its reply is over
SYSTEM = "You are SlingerGPT, a small, friendly AI assistant."   # without it, it makes up a person when asked who it is
SIZE = {"1m": "1M", "10m": "10M", "50m": "50M"}
gpu = threading.Lock()   # one reply at a time: the card is shared
current = {}             # ponytail: one model loaded at a time, so a resumed training run keeps its memory


def is_slinger(saved):
    return str(saved.get("name", "")).startswith("SlingerGPT")


@lru_cache(maxsize=None)
def describe(path, mtime):
    """What the picker shows for one model file (read without loading the weights); None if the page can't run it."""
    try:
        saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:   # still being written: read again once its time stamp changes
        return None
    base = {"name": path.stem, "title": path.stem, "family": "Earlier experiments", "updated": mtime}
    if is_slinger(saved) and path.with_suffix(".tokenizer.json").exists():
        weights = sum(v.numel() for v in {v.data_ptr(): v for v in saved["model"].values()}.values())   # a shared table once
        return {**base, "title": saved["name"], "family": "SlingerGPT", "params": f"{weights / 1e6:.0f}M",
                "unit": "tokens", "context": saved["cfg"]["context"]}
    if saved.get("tokenizer") == "gpt2":
        return {**base, "params": "124M", "unit": "tokens", "context": saved["context"]}
    if saved.get("size") in SIZE:
        unit = "words" if saved.get("tokens") == "words" else "letters"
        return {**base, "params": SIZE[saved["size"]], "unit": unit, "context": t.SIZES[saved["size"]][3]}
    return None   # a research checkpoint saved without its piece list: no way to read or write its text


def models():
    """SlingerGPT first, newest first; then the earlier experiments, biggest first."""
    found = [d for p in MODELS.glob("*.pt") if (d := describe(p, p.stat().st_mtime))]
    slinger = lambda d: d["family"] == "SlingerGPT"
    return sorted(found, key=lambda d: (not slinger(d), -d["updated"] if slinger(d) else -float(d["params"][:-1]), d["name"]))


# 16-bit: the same answers as this morning's code ("capital of france?" right 58 of 100 times; this morning's code 57).
# True = 8-bit. 16 November: my first 8-bit (sums added up exactly in 32-bit) was live for an hour and got France
# right only 36 of 100 times; training's own 8-bit model (torchao Float8Linear) gets 47 read the plain way and 49
# through this file's speed-up code, so the speed-up code is not the cause and 8-bit itself answers unsure questions
# worse. Weights now does the multiply as training did (torch._scaled_mm, fast accumulation). It still differs from
# training's 8-bit when writing one token at a time: each token gets its own 8-bit scale, where training shared one
# across the whole batch (training's own layers through the speed-up code agree with the plain way 501 of 516 times).
EIGHT_BIT = False
E4M3 = 448.0   # the largest 8-bit (FP8 e4m3) number


def to8(x):
    """`x` in 8-bit with one scale for the whole tensor, exactly as torchao's Float8 training recipe does it (dynamic,
    tensorwise; it divides in 64-bit, which gives the same 32-bit scale): (the 8-bit tensor, what to multiply it by)."""
    scale = E4M3 / x.detach().abs().amax().float().clamp(min=1e-12)
    return (x.float() * scale).clamp(-E4M3, E4M3).to(torch.float8_e4m3fn), scale.reciprocal()


class Weights(torch.nn.Module):
    """A linear layer for writing. eight: exactly as SlingerGPT trained (torchao's Float8Linear forward): the weights
    and each input in 8-bit (e4m3, one scale for the whole tensor), multiplied by the card's 8-bit hardware with fast
    accumulation (torch._scaled_mm, use_fast_accum=True: the sums are added up the card's quicker, less exact way, as
    in training; adding them up exactly instead, my first 8-bit version, was not what the model learned with), the
    result in 16-bit, then the bias. Otherwise its weights in 16-bit, rounded once; for one new token the multiply-
    and-add is then written out so that the compiler makes it one job that reads the weights directly (F.linear,
    cuBLAS, read them at about a third of the card's memory speed: 1,080 -> 1,345 tokens a second)."""
    def __init__(self, layer, eight):
        super().__init__()
        w = layer.weight.detach().float()
        self.eight = eight
        if eight:   # worked out once: the same every time, as the weights do not change while writing
            w, unscale = to8(w)
            self.register_buffer("weight", w)
            self.register_buffer("unscale", unscale)
        else:
            self.register_buffer("weight", w.to(torch.bfloat16))
        self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().to(torch.bfloat16))

    def forward(self, x):
        x = x.to(torch.bfloat16)   # as autocast hands it to a linear layer
        if self.eight:
            x8, unscale = to8(x)
            y = torch._scaled_mm(x8.reshape(-1, x8.shape[-1]), self.weight.t(), scale_a=unscale, scale_b=self.unscale,
                                 out_dtype=torch.bfloat16, use_fast_accum=True).reshape(*x.shape[:-1], -1)
            return y if self.bias is None else y + self.bias
        if x.shape[-2] > 1:   # the conversation read in one go
            return F.linear(x, self.weight, self.bias)
        y = (x.float()[..., None, :] * self.weight.float()).sum(-1)   # added up in 32-bit, rounded once at the end
        return (y if self.bias is None else y + self.bias.float()).to(torch.bfloat16)


def slow_stream(net, ids, count, **settings):
    """Yields `count` new tokens after `ids`, one at a time, the plain way: the model reads the whole text again
    for every new token."""
    idx = torch.tensor([ids], device=t.DEVICE)
    for _ in range(count):
        idx = net.write(idx, 1, **settings)
        yield idx[0, -1].item()


class Slinger(torch.nn.Module):
    """SlingerGPT (research/arch.py) behind the same write() as the other models, plus stream(), the fast way.
    Written the plain way it managed 87 tokens a second (16 November): every token is hundreds of small jobs, and
    the card finishes each one before Python has handed over the next. stream() keeps what the model worked out
    for earlier tokens (a cache), merges the small jobs (torch.compile) and records one token's jobs once (a CUDA
    graph, one for each length of cache read), then replays the recording: about 1,400 tokens a second in 16-bit.
    In 16-bit its chances for each next token match the plain way's to within rounding (the top choice agreed 1,013
    times in 1,016; the plain way's own two attention kernels agree 1,011 times)."""
    write = t.TinyGPT.write

    def __init__(self, net, context):
        super().__init__()
        self.net, self.context, self.graph = net, context, None
        for where, parent in list(net.named_modules()):
            for name, layer in list(parent.named_children()):
                if isinstance(layer, torch.nn.Linear) and layer is not net.out:   # out shares the 32-bit piece table
                    # 8-bit only where it trained so: the block layers (arch.build), not the lookup memory
                    setattr(parent, name, Weights(layer, EIGHT_BIT and where.startswith("blocks.")))

    def forward(self, idx):
        return self.net(idx).float()   # its scores come out in 16-bit; sampling reads them in 32

    def run(self, idx, pos, grams, span=None):
        """Scores for the next token after `idx` (1, t), the tokens at positions `pos` (t,), reading the cache of
        the earlier positions and adding these. The same steps as arch.GPT.features. grams: these tokens with up
        to two before them, for the lookup memory. span: how much of the cache to read (all of it by default)."""
        net, (b, t_), span = self.net, idx.shape, span or self.context
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = net.tokens(idx)
            cos, sin = net.cos[pos].to(x.dtype), net.sin[pos].to(x.dtype)
            see = torch.arange(span, device=idx.device) <= pos[:, None]   # each token sees itself and earlier
            for i, block in enumerate(net.blocks):
                if str(i) in net.engram:
                    e = net.engram[str(i)]
                    rows = e.table(e.rows(grams)[:, -t_:]).flatten(2)
                    k, v = e.k(rows), e.v(rows)
                    out = torch.sigmoid((e.qn(x) * e.kn(k)).sum(-1, keepdim=True) / math.sqrt(x.shape[-1])) * v
                    self.memory.index_copy_(1, pos + 3, out.float())   # slots 0-2 stay zero: the conv's left padding
                    near = self.memory.index_select(1, pos[0] + torch.arange(t_ + 3, device=idx.device))
                    x = x + (out + F.conv1d(near.transpose(1, 2), e.conv.weight, groups=x.shape[-1]).transpose(1, 2))
                q, k, v = block.qkv(block.n1(x)).view(b, t_, 3, -1, block.hd).transpose(1, 3).unbind(2)
                q, k = arch.rope(block.qn(q), cos, sin), arch.rope(block.kn(k), cos, sin)
                # rounded to 16-bit, as the attention kernels read them, but kept in 32-bit for the one-token maths
                self.keys[i].index_copy_(2, pos, k.to(torch.bfloat16).float())
                self.values[i].index_copy_(2, pos, v.to(torch.bfloat16).float())
                keys, values = self.keys[i][:, :, :span], self.values[i][:, :, :span]
                if t_ == 1:   # one new token: plain 32-bit matrix maths (the memory-efficient kernel, built for
                    with torch.autocast("cuda", enabled=False):   # long texts, took 1.4 ms a token)
                        w8 = q.to(torch.bfloat16).float() @ keys.transpose(-1, -2) / math.sqrt(block.hd)
                        a = w8.masked_fill(~see, -math.inf).softmax(-1) @ values
                else:         # the conversation read in one go
                    with sdpa_kernel(arch.ATTENTION, set_priority=True):
                        a = F.scaled_dot_product_attention(q, keys, values, attn_mask=see)
                x = x + block.o(a.transpose(1, 2).reshape(b, t_, -1))
                h = block.up(block.n2(x))
                x = x + block.down(F.silu(h[..., : h.shape[-1] // 2]) * h[..., h.shape[-1] // 2:])
            return net.out(net.norm(x[:, -1]).to(torch.bfloat16)).float()

    @torch.no_grad()
    def record(self):
        c, dev = self.net.c, self.net.cos.device
        hd = c.width // c.heads
        self.keys = [torch.zeros(1, c.heads, self.context, hd, device=dev) for _ in range(c.layers)]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.memory = torch.zeros(1, self.context + 3, c.width, device=dev)
        self.token = torch.zeros(1, 1, dtype=torch.long, device=dev)   # the graph's inputs: the newest token,
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)        # its position
        self.last = torch.zeros(1, 3, dtype=torch.long, device=dev)    # and the two before it
        self.picked, self.ready = torch.zeros(1, dtype=torch.long, pin_memory=True), torch.cuda.Event()
        # one recording per length of cache read, so that a short chat reads only the start of the cache
        self.spans = [s for s in (256, 512, 1024) if s < self.context] + [self.context]
        run = torch.compile(self.run, dynamic=False)   # merges the many small jobs into a few (766 a token without it)
        step = lambda span: (self.last.copy_(torch.cat((self.last[:, 1:], self.token), 1)),
                             run(self.token, self.pos, self.last, span))[1]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):   # a few plain runs first, as CUDA graphs need (and the compiler, once a span)
            for span in self.spans:
                for _ in range(3):
                    step(span)
        torch.cuda.current_stream().wait_stream(side)
        self.graph, self.scores = {}, {}
        for span in self.spans:
            self.graph[span] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph[span]):
                self.scores[span] = step(span)

    @torch.no_grad()
    def stream(self, ids, count, temperature=1.0, top_k=None, ban=None, dry=0.0, dry_allowed=12, breakers=(), dry_from=0):
        """Yields `count` new tokens after `ids`, picked as write() picks them, one at a time."""
        settings = dict(temperature=temperature, top_k=top_k, ban=ban, dry=dry, dry_allowed=dry_allowed,
                        breakers=breakers, dry_from=dry_from)
        seq = list(ids)
        if len(seq) > self.context:
            yield from slow_stream(self, seq, count, **settings)
            return
        if self.graph is None:
            self.record()
        idx = torch.tensor([seq], device=self.pos.device)
        scores = self.run(idx, torch.arange(len(seq), device=idx.device), idx)   # the conversation so far, in one go
        self.last.copy_(torch.tensor([([0, 0] + seq)[-3:]]))   # each step drops the oldest and adds its token
        self.pos.fill_(len(seq) - 1)
        for done in range(count):
            if len(seq) > self.context:   # past the memory window: from here on write() reads the last `context` tokens
                yield from slow_stream(self, seq, count - done, **settings)
                return
            scores = scores / max(temperature, 1e-6)   # from here, the same steps as write()
            if ban is not None:
                scores[:, ban] = -math.inf
            if dry:
                penalty = t.dry_penalties(seq[max(dry_from, len(seq) - 512):], dry, 1.75, dry_allowed, breakers)
                if penalty:   # sent from pinned memory so that Python need not wait for the card
                    send = lambda xs, kind: torch.tensor(xs, dtype=kind, pin_memory=True).to(scores.device, non_blocking=True)
                    scores[0].index_add_(0, send(list(penalty), torch.long), send([-p for p in penalty.values()], torch.float))
            if top_k:
                cut = torch.topk(scores, min(top_k, scores.shape[-1])).values[:, -1:]   # a slice: [-1] waits for the card
                scores = scores.masked_fill(scores < cut, -math.inf)
            # torch.multinomial(F.softmax(scores, -1), 1) does exactly this for one token (the same random numbers,
            # the same pick), after checks that cost 0.27 ms a token
            chance = F.softmax(scores, dim=-1)
            new = torch.div(chance, torch.empty_like(chance).exponential_(1)).argmax(-1, keepdim=True)
            self.picked.copy_(new[0], non_blocking=True)
            self.ready.record()
            if len(seq) < self.context:   # the card works out the next scores while Python carries on
                self.token.copy_(new)
                self.pos.add_(1)
                span = next(s for s in self.spans if s > len(seq))   # the new token sits at position len(seq)
                self.graph[span].replay()
            self.ready.synchronize()   # waits for the pick only, not for the scores after it
            seq.append(int(self.picked[0]))
            yield seq[-1]
            scores = self.scores[span]


class Model:
    """A loaded model with its own way of turning text into tokens and back."""
    def __init__(self, path):
        saved = torch.load(path, map_location=t.DEVICE, weights_only=False)
        self.slinger = is_slinger(saved)
        if self.slinger:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(path.with_suffix(".tokenizer.json")))
            net = arch.build(**{**saved["cfg"], "fp8": False})   # trained mostly in 8-bit; answers are written in 16-bit
            self.encode, self.decode = (lambda s: tok.encode(s).ids), tok.decode
            self.ban, self.end = None, 0   # piece 0 ends a text: it learned to write it when a reply is done
            self.breakers = {i for s in ("\n", "\n\n", ":", '"') for i in tok.encode(s).ids}
            self.dry_allowed, self.per_word = 4, 1.8   # its pieces are about 3.3 letters
        elif saved.get("tokenizer") == "gpt2":
            g.CONTEXT = saved["context"]
            net = g.build()
            self.encode, self.decode = g.ENC.encode_ordinary, g.ENC.decode
            self.ban, self.end = list(range(g.ENC.n_vocab, g.VOCAB)), g.ENC.eot_token   # padding rows, never real text
            self.breakers = {g.ENC.eot_token, *g.ENC.encode_ordinary("\n"), *g.ENC.encode_ordinary(":")}
            self.dry_allowed, self.per_word = 3, 1.35
        else:
            t.use_size(saved["size"])
            vocab, words = saved.get("vocab", saved.get("chars")), saved.get("tokens") == "words"
            net = t.TinyGPT(len(vocab), tie=words).to(t.DEVICE)
            stoi = {c: i for i, c in enumerate(vocab)}
            self.encode = lambda s: [stoi[c] for c in (t.WORD.findall(s) if words else s) if c in stoi]
            self.decode = lambda ids: t.join_words([vocab[i] for i in ids]) if words else "".join(vocab[i] for i in ids)
            self.ban, self.end = (0 if words else None), None   # words: never write the stand-in for rare words
            self.breakers, self.dry_allowed = {stoi[c] for c in "\n:\"" if c in stoi}, 3 if words else 10
            self.per_word = 1.25 if words else 6   # tokens per word: punctuation is a token; letters ~6 a word
        net.load_state_dict(saved["model"])
        self.net = (Slinger(net, saved["cfg"]["context"]) if self.slinger else net).eval()
        torch.seed()   # building fixes the random seed for training; replies should differ

    def reply(self, prompt, chat, words, temperature, top_k, dry, dry_from=0):
        """Yields a reply of up to about `words` words, a piece at a time; self.written counts its tokens.
        Chat mode stops where the model ends its turn. dry_from: pieces at the start the anti-repetition check skips."""
        ids = ([0] if self.slinger else []) + self.encode(prompt)   # SlingerGPT read every text after an end-of-text piece
        ids = ids or [self.end or 0]
        new, sent, self.written, self.cut = [], "", 0, False
        self.prompt = len(ids)   # pieces of the conversation so far, for the page's context window bar
        stream = self.net.stream if self.slinger else partial(slow_stream, self.net)
        for token in stream(ids, round(words * self.per_word), temperature=temperature, top_k=top_k, ban=self.ban,
                            dry=dry, dry_allowed=self.dry_allowed, breakers=self.breakers, dry_from=dry_from):
            new.append(token)
            self.written = len(new)
            if new[-1] == self.end:
                break
            text = self.decode(new)
            if text.endswith("\ufffd"):   # half of a character split across word pieces: wait for the rest
                continue
            if chat:
                text = text.lstrip()
                stop = TURN if self.slinger else "\n"   # SlingerGPT's replies run over several lines
                if stop in text:
                    yield text[:text.index(stop)][len(sent):]
                    return
                if self.slinger:   # hold back what may be the start of a "User:" line
                    text = text[:len(text) - next((k for k in range(len(TURN) - 1, 0, -1) if text.endswith(TURN[:k])), 0)]
            yield text[len(sent):]
            sent = text
        else:
            self.cut = True   # it was still writing when the length cap came: the page says so


def load(name):
    path = MODELS / f"{name}.pt"
    key = (name, path.stat().st_mtime)   # a training run that saves a better model is picked up
    if key not in current:
        current.clear()
        torch._dynamo.reset()   # the compiler keeps 8 versions of Slinger.run: a third model would run uncompiled
        torch.cuda.empty_cache()
        current[key] = Model(path)
    model = current[key]
    if model.slinger and model.net.graph is None:
        model.net.record()   # the compiler and the recordings: up to a minute the first time, ~10 s after
    return model


def ready(name):
    """True if `name` is on the card and ready to write, so the page needs no loading notice."""
    model = current.get((name, (MODELS / f"{name}.pt").stat().st_mtime))
    return model is not None and (not model.slinger or model.net.graph is not None)


def script(messages, slinger, system=SYSTEM):
    """The conversation in the layout the model learned, ending where the model speaks next."""
    if slinger:   # the layout of its conversation fine-tune: "System:", "User:" and "Assistant:" lines
        lines = [f"System: {system}"] if system else []
        lines += [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['text'].strip()}" for m in messages]
        return "\n".join(lines) + "\nAssistant:"
    lines = [f"{YOU if m['role'] == 'user' else BOT}: {' '.join(m['text'].split())}" for m in messages]
    return "\n".join(lines) + f"\n{BOT}:"


LOOKUP = False   # "Look it up" is off on the site for now (owner, 16 November); True brings it back with web/app.js canLookUp
PASSWORD = HERE / "chat_password.txt"   # if this file exists, every request needs its password (the browser asks once)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(WEB), **k)

    def end_headers(self):
        if self.command in ("GET", "HEAD") and not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")   # browsers check for a newer page every time (16 Nov: phones kept old app.js)
        super().end_headers()

    def allowed(self):
        """True if no password is set, or the request carries it (HTTP Basic sign-in; any user name). Otherwise
        answers 401, which makes the browser show its sign-in box, after a one-second pause against guessing."""
        if not PASSWORD.exists():
            return True
        want = PASSWORD.read_text(encoding="utf-8").strip()
        sent = self.headers.get("Authorization", "")
        try:
            given = base64.b64decode(sent[6:]).decode("utf-8").partition(":")[2] if sent.startswith("Basic ") else ""
        except ValueError:
            given = ""
        if want and hmac.compare_digest(given.encode(), want.encode()):
            return True
        time.sleep(1)
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="SlingerGPT", charset="UTF-8"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_HEAD(self):
        if self.allowed():
            super().do_HEAD()

    def log_message(self, *a):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return
        if self.path == "/api/models":
            return self.send_json(models())
        super().do_GET()

    def do_POST(self):
        if not self.allowed():
            return
        if self.path != "/api/reply":
            return self.send_error(404)
        try:
            ask = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if ask["model"] not in {m["name"] for m in models()}:
                raise ValueError(f"no model called {ask['model']!r} in models/")
            chat = ask.get("mode", "chat") == "chat"
            messages = [m for m in ask["messages"] if m.get("role") in ("user", "assistant") and isinstance(m.get("text"), str)]
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("the last message must be yours")
            length = min(max(int(ask.get("words", 80)), 1), 400)
            temperature = min(max(float(ask.get("temperature", 0.7)), 0.05), 2.0)
            top_k = min(max(int(ask.get("top_k", 50)), 0), 1000) or None
            dry = min(max(float(ask.get("dry", 0.0)), 0.0), 5.0)
            source, system = None, SYSTEM
            slinger = {m["name"]: m for m in models()}[ask["model"]]["family"] == "SlingerGPT"
            if LOOKUP and chat and slinger and ask.get("lookup"):   # search first; the passage goes in the System line, as it practised
                if not lookup.DB.exists():
                    raise ValueError("the search index is not built yet: run python -P research/lookup.py build")
                found = lookup.search(messages[-1]["text"], 1)
                if found:
                    title, passage, where = found[0]
                    source = {"title": title, "passage": passage, "where": where}
                    system = SYSTEM + " " + lookup.PASSAGE.format(title=title, passage=passage)
        except (KeyError, TypeError, ValueError) as e:
            return self.send_json({"error": str(e)}, 400)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        send = lambda data: (self.wfile.write((json.dumps(data) + "\n").encode()), self.wfile.flush())
        try:
            if not ready(ask["model"]):   # the page shows "Loading ..." until the first words
                send({"loading": True})
            with gpu:
                began = time.time()
                model = load(ask["model"])
                loaded = time.time()
                if source:
                    send({"source": source})
                prompt = script(messages, model.slinger, system) if chat else messages[-1]["text"]
                # a looked-up passage is there to be quoted: the anti-repetition check skips its System line (16 November:
                # with it, copied words came out garbled, "the uni world" for "the universe")
                skip = 1 + len(model.encode(f"System: {system}")) if source and chat else 0
                first = None   # the clock starts at the first words on screen (owner, 16 November)
                for piece in model.reply(prompt, chat, length, temperature, top_k, dry, skip):
                    if piece:
                        first = first or time.time()
                        send({"text": piece, "used": model.prompt + model.written})
                send({"done": True, "tokens": model.written, "used": model.prompt + model.written, "cut": model.cut,
                      "seconds": round(time.time() - (first or time.time()), 3), "load_seconds": round(loaded - began, 2)})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass   # the page pressed Stop
        except Exception as e:
            try:
                send({"error": f"{type(e).__name__}: {e}"})
            except OSError:
                pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"{len(models())} models in {MODELS}\nopen http://127.0.0.1:{PORT}", flush=True)
    def warm():   # the first model gets ready while the page already opens; a reply asked for meanwhile waits for it
        with gpu:
            list(load(models()[0]["name"]).net.stream([0], 8))   # loads and records it, then one short reply
        print("ready", flush=True)
    if models() and models()[0]["family"] == "SlingerGPT":
        threading.Thread(target=warm, daemon=True).start()
    server.serve_forever()
