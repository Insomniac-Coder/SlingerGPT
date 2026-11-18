# SlingerGPT 1 pico: engineering document

SlingerGPT 1 pico is a decoder-only transformer of 97,397,760 weights. It was pre-trained from scratch on one 12 GB laptop GPU in 27.27 hours of training, then fine-tuned for chat in 37.2 minutes. Its recipe was redesigned against the GPT-2 way: GPT-2 block, AdamW and GPT-2's 50,257 word pieces. This document gives the exact design, the settings, the data, the training runs, the chat fine-tune, how replies are written, and the experiments behind each choice, so an engineer can understand and reproduce the model.

**Conventions**

- *Piece* means token. An *update* is one optimiser step of pre-training. A *step* is one optimiser step of the fine-tune.
- *Wrongness per letter* is mean cross-entropy in nats per UTF-8 byte of text. Lower is better. To get bits per byte, divide by ln 2. *Per piece* means nats per piece.
- A source is given as `file:line` or as a log name. All paths are relative to the repository root.
- **(derived)** marks a number computed here from sourced numbers. **(unconfirmed)** marks a statement that could not be checked against code or a log; section 15 lists these.
- Some logs, notes and experiment scripts are cited only so each number has a source. If one of them is missing from the published repository, it was not part of the release.

---

## Contents

1. Summary
2. Architecture
3. The changed parts in detail
4. Optimiser, batch and schedule
5. 8-bit training
6. Data and the word-piece list
7. The training runs
8. Chat fine-tune
9. Fact lookup
10. Writing replies
11. How the design was chosen
12. Early models
13. Results of the final model
14. Reproducing it
15. Known limits and unconfirmed points

### Where things are

| File | What it does |
|---|---|
| `research/arch.py` | The model (`Cfg`, `GPT`, `Block`, `Engram`), the size table, the optimiser groups and the FP8 conversion |
| `research/train_slinger.py` | Pre-training: data mix, schedule, safety, checkpoints |
| `research/prepare_slinger.py`, `research/prepare_slinger_more.py` | The two cuts of training text and the 4,096-piece list |
| `research/fused_ce.py` | Sliced output-layer loss for large vocabularies (experiments only; not used by SlingerGPT) |
| `research/finetune_slinger.py` | Chat data builders and the fine-tune trainer |
| `research/lookup.py` | Wikipedia passage search (SQLite FTS5) |
| `research/test_slinger.py`, `research/chat_suite.py`, `research/fix_gate.py`, `research/facts_quiz.py`, `research/report_card.py` | Tests |
| `research/lab.py`, `research/final_test.py`, `research/test_124m.py`, `research/speed_124m.py`, `research/pieces_report.py`, `research/prepare_pieces.py` | The design experiments |
| `chat_server.py`, `web/app.js` | The chat page and its fast writing path |
| `tiny_gpt.py`, `build_corpus.py`, `build_corpus_v2.py`, `distill.py`, `compare_models.py`, `prepare_tokens.py`, `gpt128.py` | Early work (section 12), the experiment text `mix_v2.txt`, the DRY sampler, and the GPT-2-way reference trainer |

Work on a larger model also lives under `research/`. It is not ready and is not described here.

---

## 1. Summary

| Item | Value | Source |
|---|---|---|
| Weights | 97,397,760, of which 9,178,112 are in the Engram lookup memory | CPU build of `research/arch.py` |
| Layers × width | 12 × 768 | `research/arch.py:359` |
| Attention | 12 heads of 64, full multi-head, QK-norm, rotary positions (base 10,000) | `research/arch.py:62-64,104-131` |
| Feed-forward | SwiGLU, hidden size 2,048 | `research/arch.py:112-114,359` |
| Norms | LayerNorm with biases, pre-norm (as GPT-2) | `research/arch.py:105-106` |
| Lookup memory | Engram: hashed 2- and 3-gram tables, 8 × 16,381 rows × 64, in front of block index 1 | `research/arch.py:141-173` |
| Window | 1,024 pieces; 2,048 for the last 10% of the second run (and so for the chat model) | `research/train_slinger.py:34-35,180` |
| Word pieces | 4,096 byte-level BPE trained on the project's own text; `<|endoftext|>` = 0 | `research/prepare_slinger.py:70-87` |
| Output | Tied to the piece table, with a 4,096 bias | `research/arch.py:184-187` |
| Optimiser | Muon (block matrices and Engram projections) plus AdamW (the rest); peak rate 0.002 | `research/arch.py:333-355` |
| Precision | bf16 autocast. The block Linears are FP8 (torchao, tensor-wise dynamic scaling). Weights are float32 | `research/arch.py:52,377-379` |
| Batch | 65,536 pieces per update (4 slices × 16 windows × 1,024) | `research/train_slinger.py:34` |
| Pre-training | Two runs: 30,520 + 61,035 = 91,555 updates, 6,000,148,480 pieces, about 61.6 pieces per weight (derived) | `research/logs/slingergpt1_pico*.log` |
| Text | 9 sources. Cut in two parts: 2,433,955,564 + 4,567,713,014 training pieces. About 19.6 billion letters read (estimate) | `data/slinger*/manifest.json` |
| Training time | 9.37 h + 17.90 h = 27.27 h (pauses, checks and breaks excluded) | logs |
| Hardware | One RTX 5070 Ti Laptop GPU (12 GB) | `research/final_report.py:232` |
| Chat fine-tune | 2,000 steps × 65,536 pieces, peak rate 0.0004, 37.2 min | `research/logs/slingergpt1_pico_chat2.log` |
| Held-out wrongness per letter (full test, 2,407,789 letters) | Base 0.5319; chat model 0.5360; OpenAI GPT-2 small on the same documents 0.673 | `research/logs/slinger_test_*.json` |
| Model file | `models/slingergpt1_pico_chat2.pt`, 389,654,512 bytes (float32, the shared table stored once; derived from the size) | `models/` listing |

---

## 2. Architecture

### 2.1 Configuration

`CFG` in `research/train_slinger.py:47-48`:

```python
CFG = dict(size="124m", vocab=4096, context=1024, modern=False, rope=True, qknorm=True, swiglu=True,
           engram=(1,), engram_slots=16381, engram_hashes=4, engram_dim=64, pad=0)
```

- `size="124m"` sets 12 layers, width 768, 12 heads and ffn 2,048 (`research/arch.py:359`).
- `--long` overrides `context` to 2,048 (`research/train_slinger.py:103`).
- `rms` is left unset, so it follows `modern=False` (`research/arch.py:56-59`). That means LayerNorm with biases, and every Linear in the blocks has a bias.
- Every other switch is off: `zero_init`, `skips`, `untied`, `relu2`, `hourglass`, `gate`, `sample_grad`, `ffn_scale=1.0`.
- The model is built with `arch.build(**cfg)` (`research/arch.py:371-380`).

The resolved configuration, printed on CPU, is `Cfg(layers=12, width=768, heads=12, ffn=2048, vocab=4096, context=2048, modern=False, rope=True, qknorm=True, swiglu=True, rms=False, engram=(1,), engram_slots=16381, engram_hashes=4, engram_dim=64, pad=0, ...)`.

### 2.2 What changed from GPT-2

The GPT-2 reference is the same code with `modern=False` and every switch off. The `research/arch.py:3` docstring describes it as "LayerNorm, GELU MLP, learned positions, biases (as gpt128.py)".

| Part | GPT-2 reference | SlingerGPT 1 pico | Section |
|---|---|---|---|
| Word pieces | GPT-2 BPE, 50,257 (padded to 50,304) | Own byte-level BPE, 4,096 | 3.1 |
| Positions | Learned table, 1,024 × 768 | Rotary on q and k, no table | 3.2 |
| Attention | 12 × 64 heads | Same heads, plus QK-norm (RMSNorm per head) | 3.3 |
| Feed-forward | GELU, 768 → 3,072 → 768 | SwiGLU, 768 → 2 × 2,048 → 768 (same matrix weights) | 3.4 |
| Lookup memory | None | Engram in front of block index 1 | 3.5 |
| Output | Tied, with bias | Tied, with bias (kept) | 3.6 |
| Norms | LayerNorm pre-norm with biases | Same (kept) | 3.7 |
| Starting values | N(0, 0.02); residual outputs ÷ √(2L) | Same (kept) | 3.8 |
| Optimiser | AdamW | Muon on matrices, AdamW on the rest | 4 |
| Matrix maths | bf16 | FP8 on the block Linears | 5 |
| Weights | 124,526,208 | 97,397,760 | 2.4 |

### 2.3 Forward pass

`GPT.features` (`research/arch.py:217-252`) runs inside `torch.autocast("cuda", dtype=torch.bfloat16)`:

```
x = tokens(idx)                      # (B, T, 768). An embedding lookup is not autocast: the residual stream is float32
cos, sin = cos[:T], sin[:T]          # rotary tables (T, 32), cast to x's dtype; no position vector is added
for i in 0..11:
    if i == 1: x = engram["1"](x, idx)   # lookup memory between block 0 and block 1
    x = block_i(x, cos, sin)
h = norm(x).to(bfloat16)             # final LayerNorm
logits = out(h)                      # tied output layer, (B, T, 4096), bf16
loss = cross_entropy(logits.float(), targets) / ACCUM     # float32 (research/train_slinger.py:183-185)
```

Each block is a pre-norm residual block (`research/arch.py:118-138`):

```
q, k, v = split(qkv(LN1(x)))          # each (B, 12, T, 64)
q, k    = RMSNorm_q(q), RMSNorm_k(k)  # QK-norm, per head vector
q, k    = rope(q), rope(k)
a       = SDPA(q, k, v, is_causal=True)
x       = x + o(concat_heads(a))
h       = up(LN2(x))                  # (B, T, 4096)
h       = silu(h[..., :2048]) * h[..., 2048:]
x       = x + down(h)
```

Other forward-pass details:

- Each block adds a bf16 branch output to the float32 residual stream, and the sum stays float32.
- There is no dropout anywhere.
- Only the body is compiled: `body = torch.compile(model.features)`, default mode (`research/train_slinger.py:125`). The output layer and the loss run outside the compiled graph.
- TF32 is allowed for matmul and cuDNN (`research/train_slinger.py:95-96`).
- The seed is `torch.manual_seed(1337)` before the model is built (`research/train_slinger.py:100`).

### 2.4 Every tensor and the parameter count

These counts come from building `arch.GPT(Cfg(...))` on CPU. No model file was loaded. The output matrix is the same tensor as the piece table, so it is counted once.

| Tensor | Shape | Weights |
|---|---|---|
| `tokens.weight` (piece table, also the output matrix) | 4096 × 768 | 3,145,728 |
| *Per block i = 0..11:* | | |
| `blocks.i.n1.weight`, `.bias` (LayerNorm before attention) | 768, 768 | 1,536 |
| `blocks.i.qkv.weight`, `.bias` | 2304 × 768, 2304 | 1,771,776 |
| `blocks.i.qn.weight` (QK-norm gain, queries) | 64 | 64 |
| `blocks.i.kn.weight` (QK-norm gain, keys) | 64 | 64 |
| `blocks.i.o.weight`, `.bias` (attention output) | 768 × 768, 768 | 590,592 |
| `blocks.i.n2.weight`, `.bias` (LayerNorm before feed-forward) | 768, 768 | 1,536 |
| `blocks.i.up.weight`, `.bias` (SwiGLU gate and up, side by side) | 4096 × 768, 4096 | 3,149,824 |
| `blocks.i.down.weight`, `.bias` | 768 × 2048, 768 | 1,573,632 |
| **One block** | | **7,089,024** |
| **12 blocks** | | **85,068,288** |
| *Engram, in front of block index 1:* | | |
| `engram.1.table.weight` (8 hash tables stacked) | 131048 × 64 | 8,387,072 |
| `engram.1.k.weight` (no bias) | 768 × 512 | 393,216 |
| `engram.1.v.weight` (no bias) | 768 × 512 | 393,216 |
| `engram.1.qn.weight` (RMSNorm on the hidden state) | 768 | 768 |
| `engram.1.kn.weight` (RMSNorm on the memory key) | 768 | 768 |
| `engram.1.conv.weight` (depthwise convolution, no bias) | 768 × 1 × 4 | 3,072 |
| **Engram total** | | **9,178,112** |
| `norm.weight`, `.bias` (final LayerNorm) | 768, 768 | 1,536 |
| `out.bias` (`out.weight` is `tokens.weight`) | 4096 | 4,096 |
| **Total** | | **97,397,760** |

Check: 3,145,728 + 85,068,288 + 9,178,112 + 1,536 + 4,096 = 97,397,760.

**Comparison with the GPT-2 reference (derived).** The reference has 124,526,208 weights, counted on CPU from `arch.GPT`. It is built from:

- a 50,304 × 768 table (38,633,472);
- learned positions, 1,024 × 768 (786,432);
- 12 blocks of 7,087,872;
- a final LayerNorm (1,536);
- an output bias of 50,304.

Going from the reference to SlingerGPT:

| Change | Weights |
|---|---|
| Smaller piece table | −35,487,744 |
| No position table | −786,432 |
| Blocks: +1,152 each (+1,024 for the wider `up` bias, +128 for the QK-norm gains) | +13,824 |
| Engram | +9,178,112 |
| Smaller output bias | −46,208 |

The result is 97,397,760. The SwiGLU and GELU matrices hold the same weights: 3 × 768 × 2,048 = 2 × 768 × 3,072 = 4,718,592.

### 2.5 Buffers (not trained)

- **`cos` and `sin`**: context × 32 in float32, not saved. They are 2,048 × 32 for the final model. The first 1,024 rows equal the 1,024-position tables (`research/train_slinger.py:101-102`).
- **`engram.1.mult`**: 8 × 3 int64, saved in the state dict.
- **`engram.1.base`**: 8 int64, not saved.
- **No learned position table**: `places` is `None` because `rope=True` (`research/arch.py:181`).

### 2.6 Attention implementation

- **Projection.** One `qkv` Linear, 768 → 2,304, with bias. The output is viewed as (B, T, 3, 12, 64) and transposed to (B, 12, T, 64) per part (`research/arch.py:122`). Columns 0-767 are queries, 768-1535 keys and 1536-2303 values; each part is 12 consecutive heads of 64.
- **Heads.** Full multi-head attention: every head has its own keys and values, with no grouped-query sharing.
- **Kernel.** `F.scaled_dot_product_attention(q, k, v, is_causal=True)` runs inside `sdpa_kernel(ATTENTION, set_priority=True)`, with `ATTENTION = [CUDNN_ATTENTION, EFFICIENT_ATTENTION, MATH]` (`research/arch.py:16-18,129-130`). The scale is the default 1/√64 = 0.125.
- **Why cuDNN first.** The code comment says this Windows PyTorch build has no flash attention, and cuDNN's kernel is 1.6× faster than the memory-efficient one on this GPU with identical results (unconfirmed: `attn_bench.py` was not re-run).
- **Masking.** A plain causal mask over the whole window. Windows are random slices of one source's piece stream (`research/train_slinger.py:69-73`), and there is **no document mask**, so attention crosses `<|endoftext|>` boundaries inside a window.
- **On the chat page** the kernel list is `[EFFICIENT, MATH]` instead (`chat_server.py:35`). cuDNN plans afresh for every new sequence length (`research/phase2_notes.md:615-618`); section 10.4 has the details.

---

## 3. The changed parts in detail

### 3.1 Word pieces: 4,096-piece byte-level BPE

**What it is** (`research/prepare_slinger.py:70-87`, function `piece_list`):

- `Tokenizer(models.BPE())` from Hugging Face `tokenizers`.
- Pre-tokenizer `ByteLevel(add_prefix_space=False)`. The saved JSON has `use_regex: true`, which is GPT-2's split pattern.
- Decoder `decoders.ByteLevel()`.
- Trainer `BpeTrainer(vocab_size=4096, special_tokens=["<|endoftext|>"], initial_alphabet=ByteLevel.alphabet())`, with the default `min_frequency`.
- No normaliser, no post-processor, no unknown piece and no byte fallback.

**Training sample.**

- About 120 MB of text in the run's shares. For each source, its first documents are taken until their character count reaches `120e6 × SHARE[source]`.
- The sample is shuffled with `random.Random(0)` and has no end-of-text markers.
- The code asserts that `<|endoftext|>` is id 0 (`research/prepare_slinger.py:86`).
- After building, a round trip must decode exactly: `"The capital of France is Paris. 3 + 4 = 7\nUser: hi\nAssistant: Hello!"` (`research/prepare_slinger.py:127-128`).

**Contents of `data/slinger/tokenizer.json`** (262,775 bytes):

- 4,096 entries: 1 special piece, 256 byte pieces and 3,839 merges.
- Piece 0 is both the end-of-document marker and the padding id (`pad=0`).
- `User` (989) and `Assistant` (990) are single pieces.
- The same file is used everywhere. `data/slinger/tokenizer.json` and the `.tokenizer.json` files next to `slingergpt1_pico`, `slingergpt1_pico_full` and `slingergpt1_pico_chat2` are byte-identical (MD5 `facb8f70f75c4b9068303da72e1d0d23`). The trainer copies it next to each finished model (`research/train_slinger.py:211`).

**Digits get no special handling.** GPT-2's regex keeps runs of digits together, and BPE then learned 106 digit-only pieces:

- 20 single digits, with and without a leading space;
- 71 two-digit pieces;
- 12 three-digit pieces;
- 3 four-digit pieces (`0000`, ` 2010`, ` 2012`).

As a result, numbers split inconsistently. `12345 678` → [`12`, `3`, `45`, ` 6`, `78`], and `The year 2024.` → [`The`, ` year`, ` 20`, `24`, `.`]. `3 + 4 = 7` → [`3`, ` +`, ` 4`, ` =`, ` 7`].

**Letters per piece.** On the first cut this is 2.67-3.62 per source (section 6.3), about 3.27 weighted by the run's shares (derived from the manifest). On the experiment text it is 3.3446 (`data/pieces/*/manifest.json`).

**Why 4,096: the compaction test.** This is Change 6 (`research/phase2_notes.md:389-411`, `research/prepare_pieces.py`, `research/logs/lab.jsonl` lines 171-180).

The setup:

- The same 297 MB of experiment text, plus 19 MB held out, was written in six ways.
- The own lists used the same BPE recipe, trained on the first 100 MB, with no special piece.
- GPT-2's list came from `tiktoken` "gpt2".
- The phrase list started from the 16,384 list and merged frequent neighbouring pairs across word boundaries up to 65,536 units.
- Each was trained on the same 10M body: the new design without lookup memory, compiled, rate 0.002, 1,000 rounds of 128 × 256 units.
- Each was scored per letter on the same held-out megabytes.

| Units | Letters per unit | Weights | Per letter after 1,000 rounds | Seconds | Per letter at 103 s | Seconds to reach the letters' 1.084 |
|---|---|---|---|---|---|---|
| Bytes (256) | 1.00 | 10.75M | 1.0836 | 103.3 | 1.084 | 103.3 |
| 1,024 pieces | 2.5346 | 11.05M | 0.9603 | 104.7 | 0.961 | 40.2 |
| **4,096 pieces** | **3.3446** | 12.23M | **0.9102** | 118.2 | **0.918** | **33.6** |
| 16,384 pieces | 3.9576 | 16.96M | 0.8834 | 217.0 | 0.963 | 55.7 |
| GPT-2 pieces (50,257, padded 50,304) | 4.1496 | 30.02M | 0.8680 | 437.0 | 1.062 | 91.8 |
| Phrase pieces (65,536) | 5.5209 | 35.88M | 0.8865 | 538.8 | 1.135 | 132.9 |

Up to GPT-2's list, larger lists learn more per round (the 65,536 phrase list does not: 0.8865 against GPT-2's 0.8680), but past 4,096 pieces scoring every unit costs more time than the compaction returns. With 1,300 rounds, the 4,096 list reaches 0.8903 in 160.8 s.

**Scoring fewer pieces (`head="sampled"`, dropped)** (`research/lab.py:151-155,171-180`):

- During training each position scores the right piece plus one shared draw of 4,096 pieces, sampled in proportion to frequency (+1 smoothing).
- The scores are corrected by subtracting the log of the expected count, and the right piece is masked among the draws.
- The loss is `logsumexp([own, rest]) − own`. Held-out scoring always uses the full list.

| List | Full scoring: per letter / s | Sampled: per letter / s |
|---|---|---|
| 16,384 | 0.8834 / 217.0 | 0.9151 / 152.7 |
| GPT-2 50,257 | 0.8680 / 437.0 | 0.9030 / 153.7 |
| Phrase 65,536 | 0.8865 / 538.8 | 0.9343 / 155.6 |

Sampling makes the round cost flat (about 0.154 s whatever the list) but costs 0.03-0.05 per letter. At equal seconds the plain 4,096 list still wins: 0.890 at 161 s, against 0.903 / 0.915 / 0.934 at 153-156 s.

**At 124M** (`research/test_124m.py`, about 8 minutes per run), the 4,096 list with full scoring scored 0.9151 at 467 s. GPT-2's list with sampled scoring scored 0.9280, and the phrase list sampled 0.9948. Each reached the GPT-2 way's final score faster than the GPT-2 way did:

| List | Speed-up over the GPT-2 way |
|---|---|
| 4,096, full scoring | 5.70× |
| GPT-2, sampled | 5.09× |
| Phrase, sampled | 3.46× |

Speed per piece at 124M (`research/logs/speed_124m.jsonl`):

| Variant | Pieces/s | Letters/s |
|---|---|---|
| New way, 4,096 list | 53,739 | 179,735 |
| GPT-2 way, 50,304-wide output | 34,332 | 142,465 |

**Also tried and dropped:**

- **Adaptive softmax** (`head="shortlist"`): 49 s against 21 s for 40 rounds. This is in the notes only; no log was found.
- **Lower-casing**: only 3-4% more compact (3.3444 → 3.4765 letters per piece for the 4,096 list; capitals are 3.12% of letters), so it was not used (`research/logs/case_factoring.log`).

### 3.2 Rotary positions

**What it is** (`research/arch.py:62-64,207-211`):

- Base 10,000. Frequencies are `freq_i = 1 / 10000^(2i/64)` for i = 0..31, and the angle is `θ(p, i) = p · freq_i` for position p.
- Pairing is half-split (rotate-half), not interleaved. With a = x[..., :32] and b = x[..., 32:]:

  `rope(x) = [a·cos θ − b·sin θ,  b·cos θ + a·sin θ]`

- It is applied to q and k only, after QK-norm, on all 64 dimensions of every head in every layer (no partial rotary).
- There is no learned position table.
- The table holds 1,024 positions in the first run and 2,048 in the second (`--long`).

**Measured effect:**

- 1M sweep: the average effect of switching rotary on over the 32 combinations of the other changes is −0.0548 per byte, the second-largest of the six changes.
- Muon only scored 1.2474; Muon plus rotary scored 1.2276 (`research/logs/lab.jsonl`, stage1@1m).
- The phase-1 literature notes mark RoPE as "adopt" (`research/phase1/A1_deepseek_attention_memory.md:40`).

**Replaced:** GPT-2's learned 1,024 × 768 position table.

### 3.3 QK-norm

**What it is** (`research/arch.py:110-111,125-126`):

- `nn.RMSNorm(64)` for queries and a separate `nn.RMSNorm(64)` for keys, in every block.
- Each has one learnable 64-entry gain, shared by the 12 heads of that layer and initialised to 1.
- It is applied to each head vector after the heads are split and before rotary:

  `q_h ← g_q ⊙ q_h / sqrt(mean(q_h²) + ε)`, and the same for keys.

- Values are not normalised.
- `eps=None`, so PyTorch uses the machine epsilon of the input dtype. The effective ε is unconfirmed: it depends on whether `rms_norm` runs in bf16 or float32 under autocast and compile.

**Why.** Following the DeepSeek-V4 practice summarised in `research/phase1/A2_deepseek_body_training.md:134`, QK-norm stands in for QK-Clip as the guard on attention logits when the matrices are trained with Muon.

**Measured effect:**

| Size | Effect | Source |
|---|---|---|
| 1M | −0.0009 on average, not measurable | 64-run sweep |
| 10M | Dropping it cost about 0.015 per byte: 1.0730 against 1.0879 uncompiled, 1.0711 against 1.0860 compiled | `research/logs/lab.jsonl`, main@10m |
| 10M time | About +5% round time compiled, +29% uncompiled | `research/phase2_notes.md:340-355` |

It was kept because of the 10M result.

### 3.4 SwiGLU feed-forward

**What it is** (`research/arch.py:112-114,132-137`):

- `up = Linear(768, 4096, bias=True)` holds the gate and the up projection side by side.
- `down = Linear(2048, 768, bias=True)`.
- `h = up(LN2(x))`, then `h = silu(h[..., :2048]) * h[..., 2048:]` with silu(z) = z·σ(z), then `x = x + down(h)`.
- The hidden size is 2,048. `build` keeps `ffn` at 2,048 when `swiglu` is on and `relu2` is off; otherwise it sets `ffn = 4 × width` (`research/arch.py:373-375`).
- The matrices hold the same number of weights as GPT-2's GELU MLP (4,718,592).

**Measured effect (1M sweep):**

- The average effect is −0.0146.
- It helps most alongside the lookup memory (interaction −0.0340) and rotary (−0.0147).
- On its own next to Muon, without rotary and memory, it hurt: Muon 1.2474 against Muon plus SwiGLU 1.2991.

**Dropped alternative:** ReLU² at 4 × width (`relu2`), average +0.0139, which is worse.

### 3.5 Engram lookup memory

`Engram` (`research/arch.py:141-173`) is described in its docstring as "Hashed 2- and 3-gram lookup memory, gated by the hidden state". It is a simplified form of Engram (arXiv 2601.07372), with no vocabulary normalisation and one table per order and hash.

There is one instance, in front of block index 1, the second of the 12 blocks (`engram=(1,)`; the docstring of `research/train_slinger.py` counts from 1 and says "before block 2"). It is built with `seed=1` (`research/arch.py:183`).

**Sizes:**

- 2 orders × 4 hashes = 8 tables, each of 16,381 rows (a prime) × 64.
- They are stored as one `nn.Embedding(131048, 64)`.
- `base[j] = j × 16381` is the row offset of table j.

**Hash multipliers:** `mult = torch.randint(1, 2**31 − 1, (8, 3), generator=torch.Generator().manual_seed(1))`, a persistent buffer. Values read on CPU:

```
2-gram rows 0-3: [709841290, 1347998343, 552255366], [894945722, 1387757823, 1142182596],
                 [571022716, 223290809, 97104700],   [6776809, 2104755556, 1139996713]
3-gram rows 4-7: [1122111277, 1039583827, 618321925], [411595547, 542987068, 710636192],
                 [2901441, 304062941, 1799371308],    [755071550, 1488195646, 1234558366]
```

Column 0 of rows 0-3 is never used.

**Rows looked up for each position** (`research/arch.py:159-165`):

- Let t be the current piece, p1 the previous one and p2 the one before that.
- Both are padded on the left with `pad = 0`. Piece 0 is `<|endoftext|>`, so before the start of a window the previous piece reads as end of text.
- For hash j = 0..3:

```
R2_j = ((m[j,1]·p1 + m[j,2]·t)               mod (2^31 − 1)) mod 16381 + j·16381
R3_j = ((m[4+j,0]·p2 + m[4+j,1]·p1 + m[4+j,2]·t) mod (2^31 − 1)) mod 16381 + (4+j)·16381
```

- The sums are int64; the largest is about 3 × 4095 × 2^31, so they cannot overflow. 2^31 − 1 is a Mersenne prime.
- Each position gets 8 rows, in the order [2-gram h0..h3, 3-gram h0..h3].

**Projection, gate, convolution** (`research/arch.py:167-173`):

```
e   = table(rows).flatten(2)                                   # (B, T, 8·64 = 512)
k   = W_k e,  v = W_v e                                        # Linear 512 → 768, no bias
α   = sigmoid( Σ_d RMSNorm_q(x)_d · RMSNorm_k(k)_d / sqrt(768) )   # one scalar per position, (B, T, 1)
out = α · v
out = out + DWConv(out)[:, :T]    # Conv1d(768, 768, kernel 4, groups 768, padding 3, no bias): causal, sees out[t−3..t]
x   = x + out
```

Here x is the residual stream after block 0, so the hidden state decides how much of the memory each position takes.

**Starting values:**

- The table is N(0, 0.02) (`research/arch.py:149`).
- `k` and `v` keep PyTorch's default Linear initialisation, U(−1/√512, +1/√512) = ±0.0442; the measured std on a CPU build is 0.0255. This is because the GPT-2 initialisation loop skips names that start with `engram` (`research/arch.py:189`).
- The two RMSNorm gains start at 1.
- The convolution starts at **zero**. The comment at `research/arch.py:157` says the module "starts as plain gated memory; the short convolution is learned".

**Optimiser treatment** (`research/arch.py:333-344`):

| Part | Optimiser | Rate | Weight decay |
|---|---|---|---|
| Table | AdamW | 5 × the base rate (peak 0.01) | None |
| `k`, `v` | Muon | Base | 0.1 |
| Gains, convolution | AdamW | Base | None |

The code attributes the 5× rate and the absence of decay to the Engram paper (unconfirmed against the paper).

**Measured effect:**

- 1M sweep: average −0.0378, the third-largest change, for about 2% more arithmetic (2,196.5 against 2,162.7 kflop per byte) and about 10% more weights. Muon plus rotary scored 1.2276; adding memory gave 1.2169.
- 10M with 4,096 pieces, 1,000 rounds: 0.906 in 122.8 s with memory against 0.910 in 118.2 s without, a tie. It was kept (`research/logs/lab.jsonl` lines 181-201).
- 124M: no measurable speed cost, 53,739 against 54,106 pieces/s without it. Peak memory 8.2 GB against 7.93 GB (`research/logs/speed_124m.jsonl`).
- A 100-step test on the 30M proxy went 7.086 → 6.956 and 6.827 → 6.700. This is in the notes only (`research/phase2_notes.md:34-35`).
- Published results, not measured here (`research/phase1/A1_deepseek_attention_memory.md`): at 0.56B active weights, validation loss went 1.808 → 1.768, and switching the memory off at test time kept only 29-44% of the fact-quiz score.

**Not used:**

- The `research/arch.py` defaults: 8 hashes, 49,157 slots and two Engram layers, about 100M weights.
- The paper's vocabulary compression.
- An earlier 124M attempt with a large memory inside `gpt128.py`: 227,582,080 weights, about 100M of them memory. It did not fit the card at 16 × 4, and ran at 12,000-17,900 against 39,900 pieces/s. This comes from the notes and `research/gpt128_design_switch.diff`; no log was kept.

The settings used (16,381 × 4 × 2 × 64) are the ones that fit the 12 GB card (`research/phase2_notes.md:365-379,515-521`).

### 3.6 Output head, tying and loss

- `out = nn.Linear(768, 4096, bias=not rms)`, so it has a 4,096-entry bias, initialised to 0 (`research/arch.py:185`).
- It is tied: `out.weight = tokens.weight` (`research/arch.py:186-187`). So logits = h·Eᵀ + b, with one 4096 × 768 table serving both input and output.
- The output layer is 12.3× smaller than GPT-2's (4,096 against 50,304 rows).
- The loss is plain mean cross-entropy over all positions, on float32 logits, divided by 4 for accumulation (`research/train_slinger.py:185`). There is no label smoothing, no z-loss and no multi-token prediction.
- **`research/fused_ce.py` is not used by SlingerGPT.** Neither `train_slinger.py` nor `finetune_slinger.py` imports it. It was written for 50,304-wide outputs in the experiments (`research/lab.py:186`, `research/speed_124m.py`). It computes the output layer and loss in slices of 1,024 positions and forms the gradient (softmax − one-hot, ÷ N) in the forward pass, so only one slice of scores is alive at a time: about 0.2 GB against 2.5 GB for 50,304 pieces. With 4,096 pieces the full score table is small. No saved output of its self-check was found.
- **Measured.** Untied tables gave −0.0015 at 1M, not measurable, so tying was kept. A published result summarised in `research/phase1/A2_deepseek_body_training.md:196` (MobileLLM) puts tying at −0.2 points for 11.8% fewer weights.

### 3.7 Norms

- `nn.LayerNorm(768)` with a learnable weight (starting at 1) and bias (starting at 0), ε = 1e-5 (`research/arch.py:105-106`).
- Pre-norm, as in GPT-2: `n1` before attention, `n2` before the feed-forward, and a final `norm` before the output layer (`research/arch.py:184`). There is no norm after a branch.
- The extra norms are all RMSNorm with gains starting at 1: two per block for QK-norm (64 wide), and two 768-wide norms inside Engram that are used only for the gate.
- **Dropped:** RMSNorm without biases (`rms`), average +0.0057 at 1M, not measurable (one-run noise is about 0.005-0.01).

### 3.8 Starting values

The loop at `research/arch.py:188-193` runs over `named_parameters()`, where the tied matrix appears once as `tokens.weight`:

- Names starting with `engram`, and parameters with fewer than 2 dimensions, are skipped. The exception is names ending in `bias`, which are set to 0.
- Every other matrix gets N(0, 0.02).
- The exception is names ending in `o.weight` or `down.weight` (the residual outputs). These get N(0, 0.02/√(2·12)) = N(0, 0.004082).

The result:

| Parameters | Starting value |
|---|---|
| `tokens.weight` (= the output matrix), `qkv.weight`, `up.weight` | N(0, 0.02) |
| `o.weight`, `down.weight` | N(0, 0.00408) |
| All biases | 0 |
| LayerNorm weights | 1 |
| QK-norm gains | 1 |
| Engram | As in 3.5 |

`zero_init` is off (`research/arch.py:194-196`). The second run starts from the first run's trained weights.

**Dropped:** zero-start output matrices (`zero_init`), −0.0014 at 1M; value and embedding skips (`skips`), −0.0016. Neither was measurable.

---

## 4. Optimiser, batch and schedule

### 4.1 Optimiser groups

The optimisers are built with `arch.optimizers(model, "muon", 0.002)` (`research/arch.py:333-355`; `research/train_slinger.py:105`). There are two optimisers, both stepped every update. Group membership was read on CPU:

| Optimiser / group | Parameters | Tensors | Weights | Peak rate | Weight decay |
|---|---|---|---|---|---|
| Muon | Every 2-D matrix under `blocks` or `engram` except the table: 12 × (`qkv`, `o`, `up`, `down`) + `engram.1.k` + `engram.1.v` | 50 | 85,721,088 | 0.002 (then shape-scaled) | 0.1 |
| AdamW "embed" | `tokens.weight` (= output matrix) | 1 | 3,145,728 | 0.002 | 0.1 |
| AdamW "flat" | Every non-2-D parameter: all biases, LayerNorm weights and biases, QK-norm gains, Engram norm gains, Engram conv (3-D), `out.bias` | 126 | 143,872 | 0.002 | 0 |
| AdamW "tables" | `engram.1.table.weight` | 1 | 8,387,072 | 0.010 (5×) | 0 |

The groups sum to 97,397,760, and an assert checks that every parameter has an optimiser (`research/arch.py:351`). Each group stores its starting rate as `base_lr`, which the schedule multiplies (`research/arch.py:352-354`).

Muon treats each fused matrix as one matrix. `qkv` (2304 × 768) is orthogonalised as a whole, not per q/k/v or per head, and `up` holds the SwiGLU gate and up halves together. The output head is tied to the embedding, so AdamW trains it.

### 4.2 Muon

The constructor is `torch.optim.Muon(matrices, lr=lr, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")` (`research/arch.py:347`). The other settings are library defaults, read from the installed `torch/optim/_muon.py` (PyTorch 2.11.0):

- momentum 0.95, Nesterov on;
- Newton-Schulz coefficients (a, b, c) = (3.4445, −4.7750, 2.0315), the coefficients of the public reference Muon implementation;
- 5 Newton-Schulz steps;
- ε = 1e-7.

Each step, for a matrix W of shape (A, B) with gradient G and scheduled rate η:

1. Momentum (EMA form): M ← 0.95·M + 0.05·G.
2. Nesterov: U = 0.05·G + 0.95·M.
3. Orthogonalise in **bf16**:
   - X = U in bf16. If A > B, transpose it.
   - X ← X / max(‖X‖_F, 1e-7).
   - Then 5 times: S = X·Xᵀ; P = b·S + c·S·S; X ← a·X + P·X.
   - Transpose back. The result O approximates the orthogonal factor of U, with all singular values pushed toward about 1.
   - `qkv` (2304 × 768), `up` (4096 × 768) and Engram `k`/`v` (768 × 512) are transposed, because each has more rows than columns; `o` (768 × 768) and `down` (768 × 2048) are not.
4. Decoupled weight decay with the unscaled rate: W ← W·(1 − 0.1·η).
5. Update with the shape-scaled rate: W ← W − η·0.2·√max(A, B)·O. This `match_rms_adamw` scaling gives the update about the RMS an AdamW update would have, so AdamW-tuned rates carry over (`research/phase1/A2_deepseek_body_training.md:153`).

Effective Muon step at the peak (η = 0.002) and at the end (η = 0.0002), derived:

| Matrix (shape as stored) | 0.2·√max(A, B) | Peak | End |
|---|---|---|---|
| `qkv` (2304 × 768) | 9.6 | 0.0192 | 0.00192 |
| `o` (768 × 768) | 5.543 | 0.01109 | 0.00111 |
| `up` (4096 × 768) | 12.8 | 0.0256 | 0.00256 |
| `down` (768 × 2048) | 9.051 | 0.01810 | 0.00181 |
| Engram `k`, `v` (768 × 512) | 5.543 | 0.01109 | 0.00111 |

**Why Muon** (`research/phase1/A2_deepseek_body_training.md:7,125-153`):

- It follows published practice (DeepSeek-V4, Moonshot, nanoGPT speed runs): Muon on the body matrices and AdamW on embeddings, head and norms.
- It stores one number of optimiser state per weight instead of two.
- Published figures, not measured here: about 1.3× below 520M weights against well-tuned AdamW, and about 52% of AdamW's FLOPs (Moonshot).

**Measured effect:**

| Test | Result |
|---|---|
| 1M sweep | Average −0.0725, the largest of the six changes |
| 1M rate pilot (best of each row) | GPT-2 block with AdamW 1.6726 against Muon 1.3579 |
| 30M proxy pilot (best of each row) | AdamW 5.8695 against Muon 5.3163 on the GPT-2 block |
| 124M step time | The optimiser takes 4.5% of a step for the new way (Muon + AdamW); for the GPT-2 way (AdamW only) it is 0.7% |

### 4.3 AdamW

- `torch.optim.AdamW([...groups], lr=lr, betas=(0.9, 0.95), fused=True)` (`research/arch.py:350`).
- ε = 1e-8, no amsgrad, decoupled weight decay. The constructor's default decay of 0.01 is overridden per group (0.1 / 0 / 0).
- Update: m = 0.9·m + 0.1·g and v = 0.95·v + 0.05·g², with bias correction; then p ← p·(1 − η·λ) − η·m̂/(√v̂ + 1e-8).
- Only the shared piece table is decayed.

### 4.4 Batch

`research/train_slinger.py:34,180-191`:

- ACCUM = 4 slices of MICRO = 16 windows of CTX = 1,024 pieces: **65,536 pieces per update**.
- Each window is 1,025 pieces from one source; inputs are `[:-1]` and targets `[1:]`.
- Each slice's loss is divided by 4 and backpropagated, then comes one clip and one optimiser step.
- **Long windows (run 2 only, `--long`):** when `step/updates ≥ 0.9` (from update 54,932 of 61,035), windows are 2,048 pieces, 8 per slice. That is still 65,536 pieces per update.

### 4.5 Learning-rate schedule

Applied to every group of both optimisers (`research/train_slinger.py:154,174-178`):

```
lr = base_lr · min(1, (step+1)/warm) · (0.1 + 0.45·(1 + cos(π·step/updates)))
warm = max(1, int(updates · warm_share))
```

The linear warm-up multiplies the cosine rather than replacing it, and the cosine falls from 1.0 to a floor of 0.1. Values below are derived from the formula.

| Run | Updates | Warm-up | Rate at points of the run (×0.002 for the base groups) |
|---|---|---|---|
| 1 | 30,520 | 3,052 (10%) | step 0: 6.6e-7; step 3,051: 0.001956 (largest, factor 0.978); 15,260: 0.0011; 24,416: 0.000372; 30,519: 0.0002 |
| 2 | 61,035 | 1,220 (2%) | step 1,219: 0.001998 (factor 0.999); 30,517: 0.0011; 48,828: 0.000372; 54,931: 0.000244; 61,034: 0.0002 |

Run 2 re-warms from run 1's end state. The plan calls this "standard for carrying on after a finished schedule" (`research/plan_124m_completion.md:67`).

### 4.6 Gradient clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` (`research/train_slinger.py:187`). This is one global L2 norm over all parameters, the Engram table included. The Muon matrices receive clipped gradients.

### 4.7 Choice of the peak rate

0.002 was the new design's best rate at the smaller sizes:

- 10M, 500 rounds: 0.001 → 1.1634, **0.002 → 1.1446**, 0.004 → 1.1453.
- 30M, 400 rounds: 0.002 → 0.991 against 0.001 → 1.034.

Sources: `research/phase2_notes.md:331-332,505-507`. **It was not re-tuned at 124M** (section 15). At 124M only the GPT-2 way's rate was checked: 0.0012 was worse than 0.0006 (1.352 against 1.278).

---

## 5. 8-bit training

**How it is switched on:**

- `--fp8` sets `cfg["fp8"] = True` (`research/train_slinger.py:103,252`).
- `arch.build` then calls `torchao.float8.convert_to_float8_training(model, module_filter_fn=lambda m, name: name.startswith("blocks.") and isinstance(m, nn.Linear))` (`research/arch.py:377-379`). This happens before the optimisers are made; parameter names do not change.

**Layers in 8-bit.** The 48 block Linears: `qkv`, `o`, `up` and `down` in each of the 12 blocks. All their dimensions (768, 2,048, 2,304, 4,096) are multiples of 16.

**Recipe.** torchao's default `Float8LinearConfig` (torchao 0.18.0 is installed today; the version at training time was not recorded):

- Dynamic tensor-wise scaling: one scale per tensor per call, `scale = finfo(fp8).max / max(amax, 1e-12)` (`torchao/float8/float8_utils.py:20,47`).
- The forward input and weight are cast to `float8_e4m3fn` (max 448). The output gradient is cast to `float8_e5m2` (max 57,344).
- All three GEMMs (output, grad-input, grad-weight) run in FP8. The forward GEMM uses fast accumulation; the two backward GEMMs do not (`torchao/float8/config.py:215-217`).
- Bias is added in higher precision.

**What stays in higher precision:**

- In float32: master weights, gradients, optimiser states, the residual stream and the logits for the loss (`research/arch.py:52`).
- The piece-table lookup itself runs in float32 (an embedding is not autocast). Under bf16 autocast: the tied output layer, all of Engram, all norms, the attention kernel, the SwiGLU activation and rotary.

**When it was on:**

- Run 1: updates 0-912 in bf16, then 8-bit from update 913 (`research/phase2_notes.md:580-591`). The pause line at update 1,700 prints the command without `--fp8`, so this is partly unconfirmed (section 15).
- Run 2's command includes `--fp8`.
- The fine-tune also trains with `fp8=True` (`research/finetune_slinger.py:719`).
- Replies are always written in 16-bit (`fp8: False` in `research/train_slinger.py:220`; `chat_server.py:279`).

**Measured before switching** (`research/phase2_notes.md:584-591`; `research/logs/speed_124m.log`; `research/logs/facts_quiz.json`):

| Measure | 16-bit | 8-bit |
|---|---|---|
| Speed, back to back | 55,451 pieces/s | 60,090 pieces/s (+8.4%) |
| Peak memory | 8.2 GB | 6.44 GB (6.46 GB in the quality run) |
| Time to 0.776 per letter | 1,166.1 s | 1,091.6 s (about 7% less) |
| Pieces/s in that run | 55,754 | 61,475 |
| Updates to reach 0.8 | 802 | 813 (about 1% more) |
| Fact quiz, picking / own words (of 117) | 37 / 4 | 35 / 5 (questions only one got right: 16-bit 14, 8-bit 12; chance 85%) |

The recorded verdict is that 8-bit was as good in about 7% less time.

4-bit (NVFP4) was tried earlier and dropped (section 11.11). The NVFP4 path in `arch.set_precision` (`research/arch.py:315-330`) is for experiments only.

---

## 6. Data and the word-piece list

### 6.1 Sources

`research/prepare_slinger.py` reads parquet files from `data/raw128/` and `data/raw_slinger/` and writes each source to its own `data/slinger/<source>/train.bin` and `val.bin`, as uint16 piece numbers. `research/prepare_slinger_more.py` writes the second cut to `data/slinger_more/<source>/`.

Document and piece counts come from `data/slinger/manifest.json` and `data/slinger_more/manifest.json`. Used documents are train + val.

| Source | Content | Docs used, first cut | Train pieces, first cut | Docs used, second cut | Train pieces, second cut |
|---|---|---|---|---|---|
| fineweb_best | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample/10BT`), `int_score ≥ 4` | 193,064 | 282,754,785 | 295,169 | 426,800,853 |
| fineweb_good | FineWeb-Edu, `int_score == 3` and `score ≥` the 70th percentile of rated-3 pages (bar 3.031) | 311,296 | 441,756,596 | 352,256 | 500,028,405 |
| cosmopedia | Cosmopedia v2 (`HuggingFaceTB/smollm-corpus`, `cosmopedia-v2`) + Cosmopedia v1 subsets khanacademy, openstax, stanford, wikihow | 536,576 | 555,756,559 | 1,191,936 | 1,300,257,241 |
| smoltalk | smol-smoltalk conversations (4 files, 460,341 rows) | 253,952 | 281,574,983 | 206,389 (all remaining) | 230,193,406 |
| soda | SODA dialogues (`allenai/soda`, train, 1,191,582 rows) | 536,576 | 145,832,694 | 655,006 (all remaining) | 175,423,388 |
| wikipedia | English Wikipedia (`wikimedia/wikipedia`, `20231101.en`; 6 files, 937,734 rows) | 180,224 | 268,159,055 | 757,510 (all remaining) | 866,004,061 |
| simplewiki | Simple English Wikipedia (241,787 rows) | 241,787 (all) | 101,375,707 | none | none |
| finemath | FineMath 4+ (`HuggingFaceTB/finemath`, `finemath-4plus`; 7 files, 732,760 rows) | 151,552 | 280,417,604 | 581,208 (all remaining) | 1,069,005,660 |
| orca_math | Orca-Math word problems (200,035 rows) | 200,035 (all) | 76,327,581 | none | none |

**Totals:**

| Cut | Training pieces | Letters (UTF-8 bytes) | Held-out pieces | Documents |
|---|---|---|---|---|
| First cut | 2,433,955,564 | 7,978,654,530 | 24,394,911 | 2,605,062 |
| Second cut | 4,567,713,014 | 14,696,262,891 | 46,184,466 | 4,039,474 |

**Hugging Face ids.** The ids for smol-smoltalk, Simple English Wikipedia, Orca-Math and the Cosmopedia v1 files are not written in any project file (unconfirmed; likely candidates are in section 15).

**Licences recorded in the project** (`research/plan_124m_completion.md:76-85`; `prepare_tokens.py:10-11`):

| Licence | Sources |
|---|---|
| ODC-By | FineWeb-Edu, smollm-corpus, FineMath |
| CC BY-SA 3.0 | English Wikipedia |
| Apache 2.0 | Cosmopedia |
| CC BY 4.0 | SODA |
| CC BY-SA 4.0 | SQuAD 2.0 (`rajpurkar/squad_v2`, chat fine-tune) |
| MIT | GSM8K (chat fine-tune); OpenAI GPT-2 small weights (comparison only) |

**Downloaded for the second cut** (`research/plan_124m_completion.md:45-48`): 14 files, 12.3 GB.

- FineWeb-Edu `sample/10BT` 002-004;
- Cosmopedia v2 `train-00001`-`00003`;
- FineMath 4+ `train-00003`-`00006`;
- Wikipedia `20231101.en` `train-00002`-`00005`.

### 6.2 Cleaning and layout

**Filters** (`research/prepare_slinger.py:38-47,106`):

- The FineWeb-Edu filters in the table above.
- Empty or whitespace-only documents are dropped.
- Nothing else: no deduplication, no Unicode normalisation and no ASCII stripping. The text goes in as raw UTF-8.

**Layout** (`research/prepare_slinger.py:50-67`):

| Source | Layout |
|---|---|
| smoltalk | One `System: ...` / `User: ...` / `Assistant: ...` line per message, content stripped, joined with `\n` |
| orca_math | `User: <question>\nAssistant: <answer>` |
| soda | The narrative, a blank line, then one `<speaker>: <utterance>` line per turn |
| wikipedia, simplewiki | `<title>\n\n<text>` |
| FineWeb-Edu, Cosmopedia, FineMath | The raw `text` column |

**Common to every document:**

- Each document ends with `<|endoftext|>`, piece 0 (`research/prepare_slinger.py:98`).
- Hold-out: document n of a source, counting train and val together, goes to `val.bin` when `n % 100 == 99` (`research/prepare_slinger.py:99`).
- Encoding uses `tok.encode_batch` in batches of 4,096 documents.

### 6.3 First cut (`research/prepare_slinger.py`)

**Size of the cut:**

- It stops once `train.bin` holds `int(TOTAL × SHARE[source] × SPARE)` pieces, with TOTAL = 2,000,000,000 and SPARE = 1.25. That is 25% more than the run reads, so less of it is read twice (`research/prepare_slinger.py:24-28,133`).
- The limit is checked only after each batch, so the cut overshoots slightly.

**Shares** (SHARE): fineweb_best 0.13, fineweb_good 0.176, cosmopedia 0.222, smoltalk 0.112, soda 0.058, wikipedia 0.106, simplewiki 0.044, finemath 0.112, orca_math 0.04. These equal the trainer's whole-run shares, 0.8·MAIN + 0.2·FINAL.

**Reading order:**

- Files are read in sorted order and documents in file order, so the first cut takes the first documents of each source.
- For Cosmopedia, v2 file 0 comes first, then the v1 files.

**Sources that ran short of their target:**

| Source | Got | Target | Reason |
|---|---|---|---|
| fineweb_best | 282.8M | 325M | Only two FineWeb files were present |
| simplewiki | 101.4M | 110M | All of it used |
| orca_math | 76.3M | 100M | All of it used |

**The fineweb_good bar** is 3.031. The second cut logs this value, computed from the same two files (`research/logs/cut_more_b.txt`). That the first cut used the same bar is a deduction: no log of the first cut was found.

**Letters per piece, first cut** (train / val):

| Source | Train | Val |
|---|---|---|
| fineweb_best | 3.4096 | 3.4154 |
| fineweb_good | 3.3915 | 3.3914 |
| cosmopedia | 3.6248 | 3.6272 |
| smoltalk | 3.4606 | 3.4814 |
| soda | 3.2585 | 3.2631 |
| wikipedia | 2.9537 | 2.9761 |
| simplewiki | 2.6717 | 2.7236 |
| finemath | 2.7255 | 2.7676 |
| orca_math | 2.9484 | 2.9516 |

Both manifests carry the internal name "SlingerGPT 1 nano training text", an older name of the model.

### 6.4 Second cut (`research/prepare_slinger_more.py`)

How it works:

- It uses the same piece list, the same filters and the same hold-out.
- Each source is read in the first cut's file order. It skips exactly the number of non-empty documents the first cut used (from `data/slinger/manifest.json`) via `itertools.islice`, then carries on (`research/prepare_slinger_more.py:46-49`).
- Caps (LIMIT, with SPARE 1.0): fineweb_best 2.0B, fineweb_good 0.5B, cosmopedia 1.3B, smoltalk, soda, wikipedia and finemath 2.0B each. Simple Wikipedia and Orca-Math have nothing new.
- fineweb_good keeps the first cut's bar (score ≥ 3.031, computed from the first two FineWeb files) while reading all five.
- Cosmopedia reads v2 file 0, then the v1 files, then the new v2 files 1-3.
- Any set of sources can be cut in one process (all by default), and several copies can run side by side, one source each. The manifest is assembled from each source's `info.json`.

**Logged results** (`research/logs/cut_more_a.txt`, `_b.txt`, `_c.txt`):

| Source | Pieces | Letters per piece | Docs held out | Time |
|---|---|---|---|---|
| fineweb_best | 427M (all available) | 3.4217 | 2,951 | 269 s |
| fineweb_good | 500M (cap) | 3.4007 | 3,522 | 335 s |
| cosmopedia | 1,300M (cap) | 3.6492 | 11,919 | 945 s |
| wikipedia | 866M (all) | 2.8928 | 7,575 | 438 s |
| finemath | 1,069M (all) | 2.7281 | 5,812 | 647 s |

smoltalk and soda appear only in the manifest.

### 6.5 How windows are sampled

Class `Mix`, `research/train_slinger.py:60-80`:

- **Source.** Each window comes from one source, drawn independently by `rng.choice` with the phase's shares.
- **Start.** The start is uniform within that source's `train.bin`: `rng.integers(0, len(d) − length − 1)`. Windows cross document boundaries, which are marked only by piece 0.
- **Run 2's two folders.** A source that has text in both `data/slinger/` and `data/slinger_more/` picks a file in proportion to its size, so every piece is equally likely (`research/train_slinger.py:61-62,71`).
- **No epochs.** There are no epochs and no shuffled document order, so the read counts in 6.7 are expected values.
- **Loading.** Windows are stacked on the CPU from `np.memmap`, pinned, and copied with `cuda(non_blocking=True)`.
- **Seeds.** The data seed is 2024 for run 1 and 2025 for run 2 (`research/train_slinger.py:107`). Held-out windows use seed 7.
- **Resume.** The data RNG state is saved in every checkpoint, so a resumed run reads the same windows.

### 6.6 Mix shares

The phase is chosen by `share = step / updates`: the MAIN shares below 0.8, the FINAL shares from 0.8 (`research/train_slinger.py:179`).

| Source | MAIN (run 1, updates 0-24,415) | FINAL (run 1, 24,416-30,519) | MAIN2 (run 2, 0-48,827) | FINAL2 (run 2, 48,828-61,034) |
|---|---|---|---|---|
| fineweb_best | 0.10 | 0.25 | 0.195 | 0.25 |
| fineweb_good | 0.22 | 0.0 | 0.10 | 0.0 |
| cosmopedia | 0.24 | 0.15 | 0.23 | 0.15 |
| smoltalk | 0.09 | 0.20 | 0.10 | 0.22 |
| soda | 0.06 | 0.05 | 0.04 | 0.05 |
| wikipedia | 0.11 | 0.09 | 0.15 | 0.10 |
| simplewiki | 0.04 | 0.06 | 0.03 | 0.05 |
| finemath | 0.11 | 0.12 | 0.13 | 0.12 |
| orca_math | 0.03 | 0.08 | 0.025 | 0.06 |

- The FINAL sets lean on "the best material".
- The run 2 shares were chosen "so that no source is read more than about 3 times over both runs" (`research/train_slinger.py:41-42`).
- Run 2's 2,048-piece windows start at update 54,932, inside FINAL2.

### 6.7 Expected pieces read per source

These are derived from share × pieces in the run and the manifest sizes. They were not counted.

| Source | Run 1 reads (M) | Run 1 passes over the first cut | Run 2 reads (M) | Passes over both runs (both cuts) |
|---|---|---|---|---|
| fineweb_best | 260 | 0.92 | 824 | 1.53 |
| fineweb_good | 352 | 0.80 | 320 | 0.71 |
| cosmopedia | 444 | 0.80 | 856 | 0.70 |
| smoltalk | 224 | 0.80 | 496 | 1.41 |
| soda | 116 | 0.80 | 168 | 0.88 |
| wikipedia | 212 | 0.79 | 560 | 0.68 |
| simplewiki | 88 | 0.87 | 136 | 2.21 |
| finemath | 224 | 0.80 | 512 | 0.55 |
| orca_math | 80 | 1.05 | 128 | 2.73 |

Letters read are about 6.55 billion in run 1 and 13.05 billion in run 2, about 19.6 billion in all. These are estimates, by the method of `research/final_report.py:220-222`.

### 6.8 Held-out scores

**Score in the training log** (`research/train_slinger.py:83-89,139-149`):

- 8 fixed windows of 1,024 (+1) pieces per source, from `data/slinger/<source>/val.bin` with seed 7. That is 72 windows and 73,728 predicted pieces, the same in every check of both runs. They are always 1,024 long.
- Score = Σ_k OVERALL_k·L_k / Σ_k OVERALL_k·lpp_k. Here L_k is the mean nats per piece on source k and lpp_k its held-out letters per piece. So the score is nats per UTF-8 byte, weighted by run 1's whole-run shares.
- These small samples move by about 0.01 between checks (`research/plan_560m_cloud.md`, Lessons, Tests).

**Full test** (`research/test_slinger.py run`):

- 1,034 held-out documents in the SHARE proportions, at most the first 4,000 letters of each, 2,407,789 letters in all.
- Each document starts with piece 0. Score = total nats / total bytes.

---

## 7. The training runs

### 7.1 Hardware and software

**Hardware.** One RTX 5070 Ti Laptop GPU (12 GB), as named in `research/final_report.py:232`. No `nvidia-smi` dump of the runs was saved.

**Software** installed when this was written (the versions at training time were not recorded):

- Python 3.14.2
- PyTorch 2.11.0+cu128
- torchao 0.18.0
- tokenizers 0.23.3
- numpy 2.4.3
- pyarrow 26.0.1

Python 3.14 on Windows is confirmed for the run by the run's error log (not published, because it holds local paths).

### 7.2 Trainer mechanics (`research/train_slinger.py`)

**GPU memory cap.** `torch.cuda.set_per_process_memory_fraction(max(0.1, (free − 0.5 GiB)/total))`, so an overflow stops with an error instead of spilling into system memory (lines 93-94).

**Heat and breaks:**

- Every 25 updates the GPU temperature is read with `nvidia-smi`.
- At 87 °C or more the run waits, polling every 10 s, until the GPU is at 70 °C or less.
- There is a 120 s break every 3,000 updates.
- Neither the waits nor the breaks count as training time (lines 49, 164-173).

**Checks and checkpoints:**

- Every 500 updates (`--every 500`) and at the last update, the trainer:
  - checks the loss is finite, raising `FloatingPointError` otherwise;
  - scores the held-out windows;
  - appends `[step, score, seconds]` to the curve;
  - saves the checkpoint;
  - writes a JSON line to `research/logs/<name>.log` (lines 194-205).
- The save is atomic: `.ckpt.tmp`, then `os.replace` (lines 133-137).
- A checkpoint holds the model, both optimiser states, step, data RNG state, training seconds, curve, cfg and planned updates. It is 826,084,175 bytes for run 1.
- The finished `.pt` holds name, cfg, model, updates, start_from, pieces, training_seconds and curve (lines 209-210).

**Pausing and resuming:**

- `train_slinger.py pause` (or Ctrl+C) saves and stops within one update, and the same `train` command carries on (lines 151-163, 264-266).
- Checkpoints are loaded with `map_location="cpu"` and freed after loading, so no second copy sits on the card (lines 116-123).

**Logging and time:**

- A speed reading is printed every 100 updates to the `.out` file. Readings that span an hourly break include the 120 s pause, which gives values of about 26,000-30,000.
- Training hours exclude score checks, breaks, cool-downs and pauses, but include compile time at each start.

### 7.3 Run 1: `slingergpt1_pico` (14 November)

**Size and settings.** `python -P research/train_slinger.py train` with the defaults: 30,520 updates, warm-up 0.10, windows of 1,024. That is 30,520 × 65,536 = 2,000,158,720 pieces.

**Timeline** (`research/logs/slingergpt1_pico.log`, `research/phase2_notes.md:580-603`):

- Started at 11:49:40 in 16-bit. At update 500: 1.1014, 53,817 pieces/s.
- Paused at update 913 to test 8-bit; carried on in 8-bit from 913 at 12:37:05.
- Updates 1,000-1,500 ran at only 23,693 pieces/s (problem 1 in 7.5).
- Paused at 1,700. A test copy with score checks every 50 updates and recompile logging ran updates 1,700-2,200, and its progress was carried over. The log lines at 1,900, 2,050, 2,100, 2,150 and 2,200 come from it.
- Carried on from 2,200 and finished at 22:05:41: "30,520 updates, 9.37 hours of training". The wall clock was about 10 h 16 min.
- No cool-down was triggered. GPU temperature at score checks was 68.1-71.3 °C.

**Speed:**

- The steady 8-bit speed has a median of 62,397 pieces/s over 282 readings from update 2,400, with a maximum of 64,157 (`research/logs/slingergpt1_pico.out`).
- The run-wide average is about 59,300 pieces/s (derived).

**Held-out score** (nats per byte; there is no value at step 0):

| Update | Hours | Score |
|---|---|---|
| 500 | 0.17 | 1.1014 |
| 1,000 | 0.34 | 0.8574 |
| 1,500 | 0.73 | 0.7995 |
| 2,000 | 0.88 | 0.776 |
| 3,000 | 1.19 | 0.7423 |
| 5,000 | 1.81 | 0.6884 |
| 10,000 | 3.39 | 0.6436 |
| 15,000 | 4.85 | 0.6183 |
| 20,000 | 6.30 | 0.595 |
| 24,000 | 7.47 | 0.5785 |
| 24,500 | | 0.577 (FINAL mix from 24,416; no visible jump) |
| 25,000 | | 0.5746 |
| 30,000 | 9.22 | 0.5633 |
| 30,520 | 9.37 | **0.5629** |

**Per source at the end:**

| Source | Score |
|---|---|
| fineweb_best | 0.6853 |
| fineweb_good | 0.7786 |
| cosmopedia | 0.4154 |
| smoltalk | 0.3327 |
| soda | 0.3883 |
| wikipedia | 0.7553 |
| simplewiki | 0.6607 |
| finemath | 0.6403 |
| orca_math | 0.2415 |

On the full test, `slingergpt1_pico` scored 0.5562 (`research/logs/slinger_test.json`).

### 7.4 Run 2: `slingergpt1_pico_full` (15-16 November)

**Command** (`research/train_slinger.py:8-11`):

```
python -P research/train_slinger.py train --fp8 --name slingergpt1_pico_full --start-from slingergpt1_pico \
    --updates 61035 --warm 0.02 --long --title "SlingerGPT 1 pico full"
```

**How it starts** (`research/train_slinger.py:106-115`):

- It loads run 1's checkpoint (update 30,520): the weights **and both optimiser states** (Muon momentum, AdamW moments). The log prints "starting from slingergpt1_pico (its update 30,520); 61,035 more updates".
- The step counter and schedule restart at 0.
- It uses data seed 2025, folders `data/slinger` + `data/slinger_more`, the MAIN2/FINAL2 shares and a 2,048-position rotary table.

**Size.** 61,035 × 65,536 = 3,999,989,760 pieces. With run 1 that makes 91,555 updates and 6,000,148,480 pieces.

**Timeline** (`research/logs/slingergpt1_pico_full.log`, `.out`, `.err`):

- Started at about 15:08:47 on 15 November.
- One cool-down, at 18:00:37 (about update 10,225). The 100-update reading that contains it is 55,974 pieces/s, against about 62,000 around it, so the wait was short.
- Finished at 09:46:06 on 16 November: "61,035 updates, 17.90 hours of training". The wall clock was about 18 h 37 min.
- GPU temperature at score checks was 67.8-72.7 °C.

**Speed:**

| Stretch | Pieces/s |
|---|---|
| Updates 200 to about 9,300 | About 68,000-68,800 |
| From about update 9,400, windows of 1,024 | About 61,500-63,500 (cause not recorded); median about 62,745 before the 2,048 phase (62,742-62,748, depending on whether readings that include a break are dropped) |
| Windows of 2,048 (updates 54,932-61,035) | Median 51,022 (max 51,432), about 19% slower |
| Average over the run (derived) | About 62,070 |

**Held-out score** (the same 72 windows as run 1):

| Update | Hours | Score |
|---|---|---|
| 0 (end of run 1) | | 0.5629 |
| 500 | 0.14 | 0.5868 |
| 1,000 | 0.27 | 0.639 |
| 1,500 | 0.40 | **0.6574** (peak) |
| 2,000 | 0.54 | 0.6541 |
| 5,000 | 1.34 | 0.6373 |
| 10,000 | 2.69 | 0.6241 |
| 20,000 | 5.59 | 0.6104 |
| 30,000 | 8.49 | 0.5932 |
| 40,000 | 11.40 | 0.5724 |
| 45,000 | 12.84 | 0.5623 (first below run 1's final) |
| 48,500 | 13.86 | 0.5557 |
| 49,000 | 14.00 | 0.5542 (FINAL2 from 48,828) |
| 54,500 | 15.59 | 0.5457 |
| 55,000 | 15.75 | 0.5483 (small bump after the switch to 2,048 windows) |
| 56,000 | 16.10 | 0.5456 |
| 60,000 | 17.53 | 0.5424 |
| 61,035 | 17.90 | **0.5411** |

The rise over the first 1,500 updates coincides with two changes: the rate going back up to 0.002, and the switch from the FINAL mix to MAIN2. The two were not measured apart.

**Per source at the end:**

| Source | Score |
|---|---|
| fineweb_best | 0.6611 |
| fineweb_good | 0.7572 |
| cosmopedia | 0.3932 |
| smoltalk | 0.3071 |
| soda | 0.3749 |
| wikipedia | 0.7393 |
| simplewiki | 0.6383 |
| finemath | 0.6131 |
| orca_math | 0.2276 |

**Full test.** 0.5562 → **0.5319** (`research/logs/slinger_test_slingergpt1_pico_full.json`). The plan's forecast was about 0.51, from the published Chinchilla formula (`research/plan_124m_completion.md:42-44`).

**Peak memory** was not logged in either run. The same settings measured 8.2 GB (16-bit) and 6.44-6.46 GB (8-bit) in separate checks. Memory with 2,048-piece windows is unknown.

### 7.5 Problems and fixes

1. **Slow stretch in run 1** (updates 1,000-1,500 at 23,693 pieces/s).
   - A test copy with recompile logging ran normally, so the cause was not reproduced.
   - The Windows event log (Kernel-Power 105) showed the laptop came off mains power at 12:45:28, 12:48:26 and 13:00:34, each time back within seconds to 3 minutes, and back on mains for good at 13:00:55. On battery the GPU's power is cut.
   - Fix: keep the laptop plugged in (`research/phase2_notes.md:592-601`).
2. **Changes made while investigating.**
   - The checkpoint is now loaded into system memory and freed. Before, a second copy of about 0.8 GB stayed on the card.
   - A speed reading is printed every 100 updates.
   - An edit briefly dropped the score-check line; it was caught in the test copy and fixed.
3. **Switch to 8-bit at update 913**, after the speed and quality check in section 5. The first 913 updates (about 19 minutes) are 16-bit.
4. **Score rise at the start of run 2** (0.5629 → 0.6574), from re-warming. The score was back below run 1's final value only at update 45,000.
5. **Unexplained speed step in run 2**, from about 68,000 to about 62,500 pieces/s around updates 9,300-9,800.
6. **Earlier lesson behind the safety rules.** On 2026-11-13 the PC froze after hours of GPU load. The cause was not established; a GPU memory overflow is the most likely one. That led to the memory cap, the temperature limit and the breaks (`research/phase2_notes.md:413-437`).
7. **Harmless warnings** in both `.err` files: "Redirects are currently not supported in Windows", and a `layer_norm` dtype-mismatch warning (bf16 input with an fp32 weight; "Cannot dispatch to fused implementation").

---

## 8. Chat fine-tune

### 8.1 Chat format

There is only one special piece, `<|endoftext|>` = 0. Everything else is plain text: `Role: content` lines with the roles `System`, `User` and `Assistant`, joined by `\n` (`research/finetune_slinger.py:34-56`; `chat_server.py:356-363`). Pre-training already used this layout for smol-smoltalk and Orca-Math, so the base model had seen it.

What the chat page sends (`chat_server.py:40,308,356-363`):

```
<|endoftext|>System: You are SlingerGPT, a small, friendly AI assistant.
User: <message>
Assistant: <reply>
User: <message>
Assistant:
```

How the role markers split into pieces:

| Text | Pieces |
|---|---|
| `\nUser:` | [199 `\n`, 989 `User`, 26 `:`] |
| `\nAssistant:` | [199, 990, 26] |
| `System:` | [51 `S`, 3847 `ystem`, 26] |

The code comment at `chat_server.py:40` says that without the System line the model makes up a person when asked who it is.

A turn ends in one of two ways:

- The model writes piece 0.
- Or the decoded text contains a new `\nUser:` line (TURN, `chat_server.py:39,324-329`). While writing, any tail that could be the start of `\nUser:` is held back, so it never reaches the screen.

### 8.2 Which pieces are trained

`encode_chats` (`research/finetune_slinger.py:147-168`):

- Each message is a head (`\n` + role + `:`; the first message has no leading newline) plus a body (`" " + content.strip()`).
- Mark = 1 on every piece of an Assistant body, and on the piece 0 that follows each conversation.
- Mark = 0 on heads, System lines and User text. Messages flagged `train=False` are read but never marked.
- **Turn ends** (`turn_ends=True`, lines 161-163, used for the mixed chats): the 3-piece `\nUser:` head that follows a practised Assistant reply is also marked. This teaches the model to end a turn in the middle of a chat, where no piece 0 follows.
- Without turn-end marks, greetings ended properly in 49 of 60 replies (`research/logs/chat_suite_slingergpt1_pico_chat.json`); with them, 60 of 60.

The loss per micro-batch is:

```
L = Σ_t m_t · CE_t / max(1, Σ_t m_t)
```

CE is computed in float32 per piece, and the marks are shifted with the targets (`mask[:, 1:]`, line 702) (`research/finetune_slinger.py:742-746`).

### 8.3 The fine-tune trainer (`train()` in `research/finetune_slinger.py:705-787`)

- **Base model.** Loads `models/<base>.pt`, builds with `fp8=True` (block Linears in FP8 as in section 5) and starts with fresh optimiser state.
- **Window.** The model's own context, `cfg.get("context", 1024)`. For `slingergpt1_pico_full` that is 2,048 (unconfirmed: the checkpoint was not opened; this follows from `--long` and the code comment at line 720).
- **Batch.** micro = 16·1024/ctx = 8 windows, ACCUM = 4, so 65,536 pieces per step.
- **Optimisers.** Identical to pre-training (`arch.optimizers(model, "muon", lr)`), including the 5× rate for the Engram table.
- **Schedule.** `factor = min(1, (step+1)/100) · (0.1 + 0.45·(1 + cos(π·step/steps)))`: a 100-step warm-up, then a cosine to 10%. Peak 0.0004 (line 26) against 0.002 in pre-training.
- **Other step details.** Clip 1.0 (line 775), TF32 on, `torch.compile(model.features)`, bf16 autocast.
- **Window sampling** (class `Windows`, lines 670-702):
  - Each window picks a source by its `--mix` share, then a uniform start in that source's joined stream, and reads ctx + 1 pieces. Windows are not aligned to conversation starts.
  - `general` reads fineweb_best or cosmopedia at equal chance.
  - `plain` reads wikipedia, simplewiki, soda, finemath or orca_math at equal chance (PLAIN, line 667).
  - Both are pre-training text with every piece marked.
- **Held-out scores in the log.** Nats per piece: 4 batches for chat and 2 for every other source, 4 windows per batch.
- **Safety.** The same memory cap, and the same 87 °C / 70 °C pause every 25 steps. `--save-at` keeps copies at chosen steps.

### 8.4 Fine-tune data builders

Data goes to `data/slinger_chat/<stem>.bin` (uint16 pieces) and `<stem>_mask.bin` (uint8 marks). Piece counts are the file size ÷ 2. Each `--mix` key maps to a stem in FILES/KEYS (`research/finetune_slinger.py:659-666`).

| `--mix` key | Builder (command) | Content | Training pieces |
|---|---|---|---|
| `chat` | `prepare` (lines 30-61) | Every smol-smoltalk conversation, System messages included; every 100th held out as in pre-training; 460,341 conversations | 511,915,151 (5,018,613 held out) |
| `who` | `identity` (lines 64-144) | 3,000 conversations, seed 5 (details below) | 2,338,535 |
| `maths` | `maths` (lines 446-468) | GSM8K, generated sums and Orca-Math (details below) | 3,766,843 (GSM8K test as `maths_val`: 228,600) |
| `reada` | `reada` (lines 635-656) | The 86,821 answerable SQuAD 2.0 questions, passage in the System line in the lookup format, 25% typed casually | 26,117,209 (1,841,135 held out) |
| `mixed` | `mixed` (lines 471-632) | 26,000 multi-topic chats, seed 21, `turn_ends=True` (details below) | 49,658,349 (600 chats, 1,045,602 held out) |
| `general`, `plain` | (none) | Pre-training text from `data/slinger/` | |

**`who` (identity).**

- Ten intents with fixed question wordings and approved answers (IDENTITY, lines 68-107): name, who are you, which company made you, are you ChatGPT/Claude/Siri/Gemini and others, are you human, who made you, what can you do, size, and what the name means.
- The answers give the name SlingerGPT, say it is a small assistant trained from scratch on a single laptop as a personal project, and name its maker. The size answer says "about 97 million weights".
- Variation:
  - 30% of questions are typed casually.
  - 50% carry the page's System line.
  - 60% are placed after a real training smoltalk conversation, so the answer holds after a long chat.

**`maths`.**

- GSM8K training problems with the `<<…>>` calculator notes removed and `#### N` turned into `\nThe answer is N.`.
- 20,000 generated sums (seed 7), leaving out the 50 test sums (seed 2026):
  - + and −: operands 2-999. × and ÷: operands 2-30.
  - Subtraction never goes negative, and division always comes out whole.
  - 25% have a worked second step, `a × b = r, and r + c = s. The answer is s.`
- 5,000 Orca-Math problems.
- No System line.

**`mixed`** joins several unrelated parts with no piece 0 between topics, as a chat on the page looks.

- **Opening:**
  - A System line in 60% of chats.
  - A greeting exchange in 35%, with short friendly replies.
  - In 10%, one earlier reply that does not fit its question, marked `train=False`.
- **Each further part:**

  | Part | Share |
  |---|---|
  | A training smoltalk conversation with its follow-ups | 72% |
  | Sums: 8,000 worked (seed 31) and 5,000 bare ones such as `15+27=` or `what's 9x3` (seed 32) | 7% |
  | GSM8K | 6% |
  | Who-are-you | 6% |
  | "in one sentence" / "briefly" requests cut to one or two sentences of a smoltalk answer | 9% |

- **Length:** a target of 500, 1,000, 1,500, 2,000 or 2,400 pieces × 3.3 letters, with a 20% chance of stopping after each part. A "thanks" exchange is inserted 15% of the time.
- **Casual typing:** 30% of user lines of 160 letters or fewer are made casual (`casual()`, lines 490-506):
  - "please / can / could / would you" removed;
  - "What is the X?" cut to "X?" (50%);
  - lower-cased (70%);
  - trailing ?.! dropped (50%).
- **Why it was built.** `research/chat_suite.py`, run on the previous live model, showed that its practice data had only one-topic chats: 40% one question and 59% three on one subject. Only 2 in 2,000 first questions were lower case, and every chat ended with piece 0 (`research/plan_chat_finetune.md:21-23`).

**`general` and `plain`.** `general` is FineWeb-Edu top-rated plus Cosmopedia; `plain` is Wikipedia, Simple Wikipedia, SODA, FineMath and Orca-Math in equal parts. `plain` was added because the first combined fine-tune raised held-out wrongness from 0.5319 to 0.5424 per letter; 8% plain brought it back to 0.5382 (`research/plan_560m_cloud.md:131-132`).

**Builders used only by earlier attempts** (8.6):

| Builder | Content |
|---|---|
| `reading` (`read`, 38,975,970 pieces) | SQuAD 2.0 with "The text doesn't say." for the 43,498 unanswerable questions |
| `read2` | The same with "I don't know." |
| `readb` | Only 40% of the unanswerable questions kept |
| `closed` (1,440,587 pieces) | Simple Wikipedia first lines as closed-book answers |
| `broad` | "What/Who is X?" with the article's opening passage, answered with its opening sentence(s); the first version sent another article's passage 15% of the time |
| `knows` / `knows2` (`ask` / `ask2`) | "I don't know." taught from a measurement of what the model knows (below) |

How `knows` works:

1. Take 24,000 questions built from articles.
2. Score `slingergpt1_pico_final` on the real opening sentence(s) of each article, in per-letter wrongness.
3. ≤ 0.5 counts as known; it learns the real sentence, never its own text.
4. ≥ 0.75 becomes "I don't know."; the middle is left out.
5. Half the conversations get another article's passage.

Counts: `ask` 9,150 known and 4,661 not (from the docstring and `research/final_report.py:489`; not in a log). `ask2` 9,072 / 9,072 (`research/logs/redo2_chain.out`).

### 8.5 The live model: `slingergpt1_pico_chat2`

**Command** (run into `models/candidates/`, then copied to `models/slingergpt1_pico_chat2.pt`):

```
python -P research/finetune_slinger.py train --base slingergpt1_pico_full --name candidates/slingergpt1_pico_chat2 \
    --title "SlingerGPT 1 pico" --steps 2000 --save-at 1000,1500 \
    --mix chat=0.26,mixed=0.22,who=0.02,reada=0.12,maths=0.10,general=0.20,plain=0.08
```

The mix is as logged at step 0 of `research/logs/slingergpt1_pico_chat2.log`. The written plan shows `chat=0.34` and no `plain`. A run without `plain` was stopped after step 200 (`research/logs/slingergpt1_pico_chat_stopped_no_plain.log`) and restarted with `plain=0.08`.

**Settings and size:**

- 2,000 steps, peak 0.0004 with a 100-step warm-up and a cosine to 0.00004.
- Windows of 2,048, 65,536 pieces per step, so 131,072,000 pieces in all (derived).
- FP8 block Linears, Muon + AdamW.
- 37.2 minutes of training, about 58,700 pieces/s (derived).

| Source | Share | Expected pieces read (derived) | Data size |
|---|---|---|---|
| chat | 0.26 | ~34.1M | 511.9M |
| mixed | 0.22 | ~28.8M | 49.7M |
| who | 0.02 | ~2.6M | 2.34M |
| reada | 0.12 | ~15.7M | 26.1M |
| maths | 0.10 | ~13.1M | 3.77M (~3.5 passes) |
| general | 0.20 | ~26.2M | pre-training text |
| plain | 0.08 | ~10.5M | pre-training text |

**Held-out scores in the log** (nats per piece):

| Measure | Step 0 | Step 2,000 |
|---|---|---|
| assistant_replies | 1.0907 | 1.0485 |
| mixed_chats | 1.0819 | 1.0117 |
| reading_answerable | 1.7747 | 0.6259 |
| maths | 1.3637 | 0.9782 |
| general_text | 1.8572 | 1.8593 (peak 1.9167 at step 600) |
| plain_text | 1.4871 | 1.4988 (peak 1.5444 at step 500) |

The error on general text rises for about a quarter of the run, then falls back.

**Copies at steps 1,000 and 1,500.** Facts asked formally were right 32.0% and 37.0% of the time, against 43.0% at step 2,000, so step 2,000 was best on facts (`research/logs/chat_suite_slingergpt1_pico_chat2_*.json`).

### 8.6 Earlier fine-tune attempts and why each was replaced

All used 65,536 pieces per step, FP8 and Muon. Logs are `research/logs/<name>.log`. The full test numbers are in section 13.

1. **Preview 1** (`slingergpt1_pico_preview1`).
   - From run 1's model: 1,700 steps of `chat=0.8, general=0.2`, 30.2 min.
   - Sums 0/50, identity 10/20. After a long chat it made up a person when asked its name.
   - Replaced once pre-training was completed and the fine-tune redone with identity, reading and maths.
2. **`slingergpt1_pico_final`** (live before chat2).
   - From `slingergpt1_pico_full`: 2,000 steps of `chat=0.53, who=0.02, read=0.15, maths=0.10, general=0.20`, 38.5 min.
   - Faults:
     - With a passage it answered "The text doesn't say." to nearly every broad question (2 right and 22 "not said" of 30 with the old search).
     - The phrase leaked into chats with no text given: 165 of 1,300 chat-suite replies, and 195 of 1,400 once impossible questions were added.
     - Replies after "Hello!" were short (median 27.5 words).
     - Casual fact questions were right only 14.7% of the time.
3. **`fix`.** An 800-step top-up on `final` with `chat=0.40, who=0.02, readb=0.10, closed=0.10, maths=0.08, general=0.15, plain=0.15`, 15.0 min. Wrong answers without lookup rose from 12 to 20 of 30. Not kept. Its rate is unconfirmed.
4. **`fix2`.** A 600-step top-up on `final` with the first `broad` (15% wrong passages), 11.3 min.
   - Broad with lookup: right went 2 → 14, but wrong went 6 → 13.
   - Held-out 0.5424 → 0.5440; quiz picking 83 → 79.
   - Lesson: a short top-up on a finished model made the quiz and sums worse; redoing from the base worked better.
5. **`redo`.** From the base, 2,000 steps with `chat=0.53, who=0.02, read2=0.07, broad=0.04, ask=0.04, maths=0.10, general=0.12, plain=0.08`, 37.2 min.
   - Better than `final` on held-out (0.5382), sums (32), lookup and broad questions.
   - On the page it was dry: 117 words on average against 218 for `final`, and "I don't know" 3 of 24 times, for example to "Tell me a short story".
   - Cause: the `knows` practice paired "Tell me about X." only with one or two Wikipedia sentences or "I don't know." (`research/plan_560m_cloud.md:138-145`).
6. **`redo2`.** As `redo`, but with `ask2`, 36.8 min.
   - Impossible questions made up fell 4 → 0 of 10.
   - But sums fell 32 → 16, with 19 sums answered "I don't know.", and without lookup it got 0 of 30 right (29 "not said").
   - Cause: every "knows" example was a "What is X?" question, while half the "I don't know" examples were short SQuAD questions, so it learned the shape of the question (`research/plan_560m_cloud.md:123-130`).
   - The `fix`, `fix2`, `redo` and `redo2` model files were deleted; their logs and results are kept.
7. **Chat run 1** (`candidates/slingergpt1_pico_chat`). The same mix as the live model, 37.5 min, but without turn-end marks. Greetings ended properly 49/60, 111 of 1,300 replies were cut off at the length limit, and 401 of 1,300 repeated phrases. Replaced by run 2 with `turn_ends=True`.

### 8.7 Lessons recorded for the fine-tune

`research/plan_560m_cloud.md:96-224`:

- Write the intended behaviour as plain rules and agree them before training. Approving a dataset is not approving the behaviour it teaches: SQuAD 2.0 taught "answer only from the passage, else 'The text doesn't say.'", which was rejected.
- SQuAD alone teaches narrow copied answers, and its unanswerable third teaches giving up. Practise reading on answerable questions only, and teach "I don't know" separately.
- To teach "I don't know", first measure what the model knows by the likelihood of the real answer. Make the "knows" side as varied in shape as the "don't know" side, and give known answers their natural length.
- Mix in about 8% plain pre-training text against forgetting, and keep copies at several steps.
- Test with the page's real settings and with short lower-case questions. The likeliest-piece tests missed both the dry replies and the 8-bit damage (10.5).
- The first reply sets the style of the chat.
  - Example: "give me some study tips" alone gave tips 30 of 30 times at about 180 words. After a chat that began "Hello!" and ended on photosynthesis, the replies were about 22 words, and 22 of 30 were about plants.
  - A wrong answer earlier in the chat drags later answers down: "capital of france?" was right 0 of 20 times after a wrong answer about sodium, against about half the time alone.

---

## 9. Fact lookup (`research/lookup.py`)

**Index** (`build`, lines 54-81):

- An SQLite FTS5 table `passages(title, body, source UNINDEXED)` with the `porter unicode61` tokenizer, so words are stemmed (capitals → capital). No search library is needed.
- Sources: all of Simple English Wikipedia (about 242,000 articles) and the 6 English Wikipedia files in `data/raw128/` (about 15% of its articles).
- 4,465,578 passages. `data/lookup/wiki.db` is 6.59 GB. A search takes under a second (`research/plan_124m_completion.md:17-19`).

**Passages** (lines 17, 29-51):

- The target is 120 words, cut along paragraph breaks.
- A paragraph that would push a passage past 240 words starts a new one.
- A run of 240+ words is cut into 120-word pieces.
- A passage closes once it reaches 120 words.
- An ending of under 15 words joins the previous passage.

**Title index** (`data/lookup/titles.db`, `build_titles`, lines 98-111):

- Maps an article name to the rowid of its opening passage, one per title and source. 1,179,521 rows.
- Names are lower-case words, and a "(...)" stays part of the name.
- It is built automatically on the first search.

**Search** (`search(question, k)`, lines 114-159):

1. **Query words:** lower-case alphanumeric words, minus a stop list of about 90 words (question words, pronouns, "tell", "explain", ...). Single letters are dropped unless they are digits. The words are OR-joined as quoted terms.
2. **Candidates:**
   - the top 40 BM25 matches over title and body, weighted title 4.0 and body 1.0;
   - the top 160 matches restricted to `{title}`;
   - every article whose exact name equals a run of 1-4 question words that neither starts nor ends with a stop word. Its opening passage is added only if the title plus passage holds every question word.
3. **Score:**

   ```
   (1 + 2·cover) · (0.2 + held) · (1.3 if opening passage) · (1.5 if exact-title match) · (0.3 if "may refer to" in the first 300 letters)
   ```

   - cover = the share of the title's words that the question names.
   - held = the share of the question's words found in the passage.
   - In both, words are stop-listed and stripped of a trailing "s".
4. Ties go to the lower rowid, and the top k are returned.

**Why the exact-title step.** Plain keyword search picked wrong articles: "In Rainbows" for "What is a rainbow?", and "Demon Internet" for "What is the internet?". With the title step, the top passage holds the answer more often (`research/plan_560m_cloud.md:157-163`; the broad and narrow counts match `passage_has_it` in `research/logs/slinger_extra_*.json`):

| Question set | Without the title step | With it |
|---|---|---|
| Broad questions | 17 of 30 | 30 of 30 |
| Narrow questions | 20 of 30 | 21 of 30 |
| Quiz sentences (notes only) | 84 of 117 | 87 of 117 |

**How a passage reaches the model** (`chat_server.py:440-447`):

- The last user message is searched with k = 1.
- The System line becomes `You are SlingerGPT, a small, friendly AI assistant. Answer using this passage from Wikipedia (<title>): <passage>`.
- The tests use the same layout (`research/test_slinger.py:225-240`).

**Behaviour rules set by the project** (`research/plan_560m_cloud.md:100-103`):

- Answer from the passage if it holds the answer.
- If no useful passage was found and the model knows, answer from memory; otherwise say "I don't know.".
- Lookup must never make an answer worse than with lookup off.

**Status: lookup is off on the chat page** (`LOOKUP = False`, `chat_server.py:366`; `canLookUp()` returns false in `web/app.js:31`). For the live model it still makes broad questions worse: 20 right / 10 wrong with the passage against 25 / 5 without (`research/logs/slinger_extra_slingergpt1_pico_chat2.json`). The fine-tunes that fixed lookup made everyday answers worse (8.6).

---

## 10. Writing replies

### 10.1 Sampling

**Page defaults** (`web/app.js:10`): temperature 0.3, top-k 50, reply length 300 words, DRY 0.8.

The comment at `web/app.js:7-9` gives the measurement behind them, made on Preview 1 (no log was found):

- Temperature 0.3 got 25 of 40 fact answers right, against 18 at 0.7.
- DRY 0.8 halves looping, from 15% to 7% of replies repeating a four-word phrase, with facts unchanged.

**Server limits** (`chat_server.py:434-437`): words 1-400, temperature 0.05-2.0, top-k 0-1000 (0 means no limit), DRY 0-5.

**Length cap.** round(words × 1.8) pieces for SlingerGPT (`chat_server.py:283,313`), so 300 words → 540 pieces. If the cap is reached mid-reply, the page says the reply was cut off.

**Steps for each piece** (`Slinger.stream`, `chat_server.py:243-267`; the same order as `tiny_gpt.TinyGPT.write`, `tiny_gpt.py:97-114`):

1. Divide the scores by max(temperature, 1e-6).
2. Set banned pieces to −∞ (there are none for SlingerGPT).
3. Subtract the DRY penalties.
4. Top-k: set everything below the k-th largest score to −∞.
5. Softmax, then pick `argmax(p / Exp(1) noise)`. This is the same draw as `torch.multinomial`, without its 0.27 ms of checks per piece.

### 10.2 DRY anti-repetition

`tiny_gpt.dry_penalties`, `tiny_gpt.py:59-72`, after llama.cpp:

- Look at the last 512 pieces, never before `dry_from`.
- For each earlier position `end`, let k be the number of pieces before it that match the current end of the text.
- If k ≥ allowed, the piece that followed that earlier match, `seq[end+1]`, gets the penalty

  ```
  multiplier · base^min(k − allowed, 40)
  ```

  Each piece keeps its largest penalty, which is subtracted from the temperature-scaled score.
- SlingerGPT settings (`chat_server.py:282-283`): multiplier = the page's DRY value (0.8), base 1.75, allowed 4 pieces.
- **Breakers:** the pieces of `\n`, `\n\n`, `:` and `"` (ids 199, 2865, 26, 2). A match stops at a breaker, so repeating a speaker name or starting a new line costs nothing.
- **`dry_from`.** With a looked-up passage, DRY first penalised copying words from it, which garbled the copies ("the uni world" for "the universe"). The server now sets `dry_from` = 1 + the pieces of the System line, so the passage never counts (`chat_server.py:465-467`).
- **Origin.** DRY was first tuned on the early model `mix50m_v2` (section 12).

### 10.3 Stopping

`Model.reply`, `chat_server.py:305-333`:

- Stop on piece 0, or when the decoded text contains `\nUser:`.
- Hold back a tail that could become `\nUser:`.
- Wait while the decoded text ends in U+FFFD (half of a character split across pieces).
- In chat mode, leading spaces are stripped.

### 10.4 Fast writing path

All speeds in this section come from `research/plan_560m_cloud.md:165-204`, `research/phase2_notes.md:613-619` and the `chat_server.py` docstrings. No separate benchmark log was found (unconfirmed).

**Why plain PyTorch was slow.** It wrote 87 pieces/s, because each piece is about 770 tiny GPU jobs handed over one by one from Python. A KV cache alone gave no gain (83 against 84 pieces/s).

**Attention kernel.** cuDNN's attention plans afresh, in tens of ms, for every new length: 3 pieces/s on new lengths against 77 on lengths seen before. So the page uses the memory-efficient kernel, then the plain-maths one, which gave 43-65 pieces/s before the cache work.

**Weights** (`chat_server.py:95-125`):

- Every Linear except the output layer (which shares the float32 piece table) is replaced by a module that stores its weights once in bf16, so autocast no longer re-casts them for every piece.
- For one new piece, the matrix multiply is written as an elementwise multiply-and-sum in float32, so the compiler fuses it and reads the weights directly. cuBLAS `F.linear` read them at about a third of memory speed.

**KV cache** (`record()`, lines 196-222; `run()`, lines 159-193):

- Per layer, keys and values are stored in float32 after rounding through bf16, as the attention kernels see them. Shape (1, heads, context, head_dim).
- A buffer of (1, context + 3, width) holds the Engram outputs for its short convolution; the first 3 slots stay zero as left padding.
- The last 3 pieces are kept for the Engram hashes.
- Prefill reads the whole conversation at once with SDPA.
- Each new piece uses plain float32 `softmax(q·Kᵀ/√d + mask)·V` over the cache. The memory-efficient kernel took 1.4 ms for one piece.

**`torch.compile`** (`dynamic=False`) of `run()` cuts the GPU jobs per piece from 766 to 221.

**CUDA graphs:**

- One graph per cache span: 256, 512, 1,024 and 2,048. Each is warmed up 3 times on a side stream, then recorded.
- Each piece replays the smallest span longer than the text so far.
- The next piece's graph is launched before Python waits for this pick. The pick is copied to pinned memory and Python waits on a CUDA event, never on all queued work. Slices replace `x[:, [-1]]`, and `.item()` is avoided.
- Past 2,048 pieces the plain path takes over, re-reading the last 2,048 pieces each time.

**Switching models** calls `torch._dynamo.reset()` and clears the CUDA cache (lines 336-347). Recording takes up to a minute the first time and about 10 s after. One model is on the card at a time, and one reply is written at a time.

**Measured speeds:**

| Stage | Pieces/s |
|---|---|
| Plain PyTorch | 87 |
| KV cache + compile + one CUDA graph | ~860 |
| + one graph per span (0.76 ms against 1.01 ms per piece for short chats; 1.25×, because most of the time left is reading the weights) | ~1,100 |
| + written-out one-piece multiply (from 1,080) | 1,345 |
| Docstring estimate, 16-bit | ~1,400 |

**Correctness.** The top choice matched the plain path's in 1,012 of 1,016 pieces (notes) or 1,013 of 1,016 (`chat_server.py` docstring). The plain path's own two attention kernels agree 1,011 times. Benchmark with the card warmed up: idle it sits at 607 MHz, and a short profile then reads 10× slow.

### 10.5 8-bit writing (tried, dropped)

It was live for about an hour, then the page went back to 16-bit because answers got worse (`research/plan_560m_cloud.md:181-204`; `chat_server.py:77-92`; `EIGHT_BIT = False`, line 84).

"capital of france?", right out of 100:

| Version | Right |
|---|---|
| Earlier code | 57 |
| 16-bit | 58 |
| First own 8-bit, with exact float32 sums | 36 |
| Training's own torchao `Float8Linear`, read the plain way | 47 |
| Training's 8-bit through the fast path | 49 |

On short lower-case questions, 16-bit got 99 of 460 right against 67 of 460 in 8-bit. "What is the capital of France?" was right 20 of 20 in both. The 90-answer quiz hid all of this: 85, 84, 83, 89, 84 against 87, 81, 84, 87, 84.

Causes found:

- Training multiplies with `torch._scaled_mm(use_fast_accum=True)`, but the first copy added up exactly; the same next piece came out only 464 of 480 times.
- Tensor-wise dynamic scaling gives each piece its own scale when writing one piece at a time, where training shared one scale across a batch.
- Rounding the Engram memory too drifted further.
- `F.linear` on 8-bit weights wrote a 16-bit copy for every piece.
- torchao computes the scale in 64-bit: 0.70 ms against 0.52 ms per piece on this consumer card.

The recorded lesson is that if a model must write in 8-bit, it should be trained with per-row or per-piece scales.

---

## 11. How the design was chosen

### 11.1 The harness (`research/lab.py`)

- **One rule:** two runs differ only in the model.
  - Every run of a stage sees the same batches in the same order (`Windows(..., seed=2024+seed)`) and is scored on the same held-out batches (seed 7).
  - One JSON line per finished run goes to `research/logs/lab.jsonl` (211 lines). A restart skips finished runs.
- **Text:**
  - The byte sizes read `data/corpus/mix_v2.txt` byte by byte (vocabulary 256). Every 100th megabyte is held out, and no window crosses into it (`research/lab.py:85-103`).
  - The word-piece sizes read `data/pieces/<list>/` built from the same text, with the same held-out megabytes.
- **Schedule:** a 10% warm-up, then a cosine to 0.1 of the peak, the same formula as section 4.5. Clip 1.0, bf16 autocast, TF32.
- **Recorded per run:** the held-out loss at 20 marks (50 with `stop_at`), seconds trained (compile excluded by one throwaway pass), units/s, peak memory, weights, and the counted forward arithmetic per unit (`FlopCounterMode`, kflop per unit).
- **Fairness rules:**
  - Each way runs at its own best rate.
  - Cheaper variants are judged against the full model trained on fewer units at equal total arithmetic.
  - Headline gains are against the GPT-2 way, never against the project's own baseline.
- **GPU safety:** the harness caps its own GPU memory. `research/final_test.py` rests 150 s after each run and stops a run at 90 °C; the hottest reading was 88 °C.

**Sizes** (`research/arch.py:358-368`, `research/lab.py:56-72`):

| Size | Layers × width | Heads | SwiGLU hidden | Window × batch | Engram when on |
|---|---|---|---|---|---|
| `1m` | 4 × 128 | 4 | 344 | 256 bytes × 128 | 1,021 slots × 2 hashes × 16, block 1 |
| `10m` | 6 × 384 | 6 | 1,024 | 256 × 128 | 4,093 × 4 × 32 |
| `30m` | 6 × 640 | 10 | 1,704 | 256 × 128 | 4,093 × 4 × 32 |
| `proxy` | 6 × 384, GPT-2 vocabulary | 6 | 1,024 | 512 × 32 | blocks 1 and 3, with the `research/arch.py` default tables (49,157 slots × 8 hashes × 64); not used in the proxy rate pilot |
| `124m-*` | 12 × 768 | 12 | 2,048 | 1,024 × 16 × 4 slices | 16,381 × 4 × 64, block 1 |

The GPT-2 block and ReLU² use a 4 × width feed-forward. Word-piece scores are converted to per letter (loss per piece ÷ letters per piece), so bytes and every list are comparable.

### 11.2 First rate pilot: the ~30M proxy, GPT-2 vocabulary

- 6 × 384 with a 50k vocabulary, about 30M weights.
- 8,192,000 pieces per run. Loss per GPT-2 piece, not compiled.
- Modern block = rotary, QK-norm, SwiGLU, RMSNorm.

| Variant | lr 0.0006 | 0.0012 | 0.0024 |
|---|---|---|---|
| GPT-2 block, AdamW | 6.0579 | **5.8695** | 6.0208 |
| GPT-2 block, Muon | 5.8668 | 5.5713 | **5.3163** |
| Modern block, AdamW | 5.6392 | 5.4442 | **5.4241** |
| Modern block, Muon | 5.4970 | **5.1381** | not run |

Both changes help and add up: 5.870 → 5.424 → 5.316 → 5.138. Muon's best rate was the highest tried, so later sizes bracket the rate. Source: `research/logs/lab.jsonl` lines 1-11.

### 11.3 1M rate pilot (bytes)

500 steps = 16,384,000 bytes, loss per byte.

| Variant | 0.001 | 0.002 | 0.004 | 0.008 | 0.016 | 0.032 |
|---|---|---|---|---|---|---|
| GPT-2 block, AdamW | 2.2307 | 2.0197 | 1.8239 | **1.6726** | 2.4456 | |
| GPT-2 block, Muon | 2.2581 | 1.8041 | 1.5266 | 1.3757 | **1.3579** | 1.3958 |
| Modern block, AdamW | 1.5465 | 1.4549 | 1.4393 | **1.4210** | 1.5488 | |
| Modern block, Muon | 1.4994 | 1.3694 | 1.3164 | **1.3010** | 1.3044 | 1.3177 |

- The ranking matches the proxy.
- Counted arithmetic per byte is equal (2,170.9 against 2,162.7 kflop), so the gain is more learning per byte, not less work.
- The modern block is far less sensitive to the rate.
- Rates fixed for the sweep: AdamW 0.008, Muon 0.016 (`research/lab.py:76`).

### 11.4 The 64-combination sweep at 1M

Every on/off combination of six changes, 1,000 steps (32,768,000 bytes) each, one seed: `python -P research/lab.py stage1 --size 1m`. The six are `muon`, `rope`, `qknorm`, `swiglu`, `rms` (RMSNorm, no biases) and `engram`.

| Variant | Loss per byte |
|---|---|
| None (GPT-2 block, AdamW) | 1.3900 |
| Muon | 1.2474 |
| Muon + rotary | 1.2276 |
| Muon + rotary + memory | 1.2169 |
| Muon + rotary + SwiGLU + memory | 1.2056 |
| **Muon + rotary + QK-norm + SwiGLU + memory (best)** | **1.1994** |
| All six | 1.2082 |

Average effect of switching each change on, over the 32 combinations of the other five (recomputed from `lab.jsonl`):

| Change | Effect |
|---|---|
| Muon | −0.0725 |
| Rotary | −0.0548 |
| Lookup memory | −0.0378 |
| SwiGLU | −0.0146 |
| QK-norm | −0.0009 |
| RMSNorm, no biases | +0.0057 |

Interactions (positive means the pair gives less than the sum):

| Pair | Interaction |
|---|---|
| Muon × SwiGLU | +0.0417 |
| Muon × memory | +0.0416 |
| Muon × rotary | +0.0344 |
| Rotary × memory | +0.0277 |
| SwiGLU × memory | −0.0340 |
| Rotary × SwiGLU | −0.0147 |

- Arithmetic per byte is 2,162.7-2,204.7 kflop for all 64 combinations, so every gain is learning per byte.
- The winner was already at 1.3879 a quarter of the way through, against the GPT-2 design's final 1.3900.
- The base chosen was `muon, rope, qknorm, swiglu, engram` without `rms`. That is exactly SlingerGPT's configuration.

### 11.5 Speed-run extras at 1M (none kept)

16 runs, every combination of four extras on the winner.

| Extra | What it is | Average effect |
|---|---|---|
| `zero_init` | Block output matrices start at zero | −0.0014 |
| `skips` | Every block also sees the raw embedding and block 0's attention values | −0.0016 |
| `untied` | Separate input and output tables | −0.0015 |
| `relu2` | ReLU² feed-forward at 4 × width | **+0.0139 (worse)** |

The untouched winner scored 1.2000 here against 1.1994 in the sweep, so repeat noise is about 0.001. The notes caution that these extras come from 124M speed runs with a 50k vocabulary and may be too small to show at 1M.

### 11.6 Routers and cost-cutting ideas at 1M (dropped)

A cheaper variant counts as a win only if its loss beats the full model trained on fewer bytes for the same total arithmetic.

**The yardstick** is the full model at different budgets:

| Bytes | Loss | Time |
|---|---|---|
| 50% | 1.2689 | 25 s |
| 64% | 1.2418 | 32 s |
| 75% | 1.2262 | 37 s |
| 100% | 1.2008 | 48 s |
| 112.5% | 1.1906 | 54 s |
| 125% | 1.1846 | 62 s |

Seed noise is about 0.002-0.003.

**The gate** (`gate`, `gate_mode="probe"`):

- A side-reader predicts each token's loss from its state after block 0.
- Only the predicted-hard share goes through the middle blocks.
- A threshold tracks a running quantile, and each token is judged alone, so the rule is the same in training and in use.
- With `gate_cap=1.25` the share that goes through in training (0.312) matches the share at use time.

| Idea | Loss | Training arithmetic against the full model | Full model at that arithmetic | Verdict |
|---|---|---|---|---|
| Gate 25%, cap 1.25 | 1.2206 | 65% | 1.241 | Wins by 0.020 |
| Gate 25%, cap 2 | 1.2196-1.2222 | 72% | 1.231 | Wins by 0.010 |
| Gate 50%, cap 1.25 | 1.2174 | 79% | 1.222 | Wins by 0.005 |
| Gate 50%, cap 2 | 1.2153 | 88% | 1.213 | Even |
| Gate 12.5%, cap 2 | 1.2637 | ~62% | 1.246 | Loses |
| Mixture-of-Depths style 25% | 1.2407 | 62% | 1.246 | Wins by 0.005 (flattering score) |
| Random 25% | 1.2807 | ~70% | 1.233 | Loses by 0.05 |
| Hourglass every 2nd / 4th | 1.2578 / 1.2627 | 73% / 62% | 1.229 / 1.246 | Loses |
| Thin feed-forward (`ffn_scale=0.5`) | 1.2297 | 76% | 1.225 | Even |
| Short windows (64) for the first half | 1.2025 | ~100% | 1.201 | Even |
| Sampled weight gradients 50 / 25 / 12.5% | 1.2382 / 1.2629 / 1.2959 | ~83 / 75 / 71% | 1.218 / 1.226 / 1.232 | Loses |
| Patch-level first phase | 1.2617-1.3954 | ~75-51% | 1.226-1.268 | Loses |
| Layer stacking (`grow=0.5`, no memory) | 1.2696 | ~75% | 1.2551 | Loses |

**Gate results beyond the table:**

- Trained at equal total arithmetic (1.55× the bytes), the 25% gate reached 1.1904 against 1.2008, about 11% saved.
- On the stopwatch it was a tie: 53.5 s against 54.4 s.
- At 10M it used 54% of the forward arithmetic and reached 1.0537 against 1.0730 at equal arithmetic. But it cannot be compiled, because the chosen count changes every round, so it is slower than the compiled full design. **Not used.**

**Count-table router** (`research/count_probe.py`, `research/logs/count_probe.json`):

- Tables of followers of the last 1-3 pieces showed that only 5.2% of positions are nearly determined.
- The most open 25% of positions carry only 32.5% of the tables' loss, so a count table cannot tell easy positions from hard ones.

### 11.7 10M on bytes: GPT-2 design against the new design

1,000 rounds of 128 × 256 bytes; 10.844M weights (GPT-2 design) and 11.997M (new design). Both rates are 0.002 after a check (GPT-2 design best 1.4560; new design 1.1446).

| Run | Loss per byte | Seconds |
|---|---|---|
| GPT-2 design | 1.2332 | 99.6 |
| GPT-2 design, compiled | 1.2345 | 83.0 |
| New design | 1.0730 | 220.9 |
| New design, compiled | 1.0711 | 107.6 |
| New, no QK-norm | 1.0879 | 171.7 |
| New, no QK-norm, compiled | 1.0860 | 102.2 |
| New, 750 / 1,250 rounds | 1.1039 / 1.0451 | 164.0 / 268.0 |

**Compiled.** The new design passed the GPT-2 design's final loss after 34.1 s, against 83.0 s: about 2.4× faster. It needs about a third of the rounds, at about 30% more time per round.

**Uncompiled** the gain is only about 1.4×, because `torch.compile` fuses the new design's many small operations (it halves its round time).

### 11.8 Letters against word pieces

See section 3.1. The 4,096 list, with every piece scored, won at equal time.

### 11.9 Head-to-head: the GPT-2 way against the new way

The two ways (`research/final_test.py:11-37`):

- **GPT-2 way:** GPT-2 block, AdamW, GPT-2's 50,257 pieces (padded to 50,304), every piece scored, compiled.
- **New way:** Muon, rotary, QK-norm, SwiGLU, memory and 4,096 own pieces, compiled.

**10M setup:**

- Same body. Three starting points each (seeds 0, 1, 2), the two ways taking turns.
- The GPT-2 way's rate was checked at 600 rounds: 0.001 → 1.118, **0.002 → 1.073**, 0.004 → 1.127.
- The new way's rate was not tuned at this size.

| Run | Rounds | Seconds (3 starts) | Per letter at the end (3 starts) | Weights |
|---|---|---|---|---|
| GPT-2 way | 1,000 | 418.4, 430.3, 422.8 | 0.9702, 0.9678, 0.9679 | 30.11M |
| New way, long | 2,700 | 347.3, 333.3, 334.9 | 0.8403, 0.8404, 0.8416 | 13.48M |
| New way, short | 680 | 82.2, 82.3, 85.8 | 0.9478, 0.9471, 0.9471 | 13.48M |

**10M results:**

- **The same quality in 83.4 s against 424 s: 5.1×.** The short runs ended slightly better than the target, so the true figure is a little higher.
- At the same time (338 s): 0.9899 against 0.8410.
- The gain grows through the run, measured at the GPT-2 way's score after 1/4, 1/2, 3/4 and all of its run:

| Point of the GPT-2 way's run | Score | Speed-up |
|---|---|---|
| 1/4 | 1.2029 | 3.7× |
| 1/2 | 1.0676 | 4.4× |
| 3/4 | 0.9992 | 4.7× |
| All | 0.9686 | 5.0× |

- The gain splits into 1.8× less text (75M against 136M letters) times 2.8× faster reading (894,000 against 321,000 letters/s) (`research/phase2_notes.md`). Counting start-up, start to finish was 462 s against 111 s, 4.2×.

**30M** (6 × 640 body; 61.95M weights for the GPT-2 way, 33.52M for the new way; one start):

- Rates: GPT-2 way best 0.001 (1.140), new way best 0.002 (0.991).
- The GPT-2 way, 700 rounds: 424.7 s, 1.0171 per letter.
- The new way at the same time (1,700 rounds): 0.8355. To the same loss (390 rounds): 96.4 s. **4.4×.**
- By quality level: 2.9×, 3.4×, 3.8×, 4.3×.
- Text samples are in `research/logs/compare_text.md`. No factual prompt was answered correctly by either way.

### 11.10 124M: speed, equal time, facts

**Speed** (`research/speed_124m.py`, `research/logs/speed_124m.jsonl`): 16 × 1,024 pieces per slice, 4 slices, compiled.

| Variant | Pieces/s | Letters/s | Against the GPT-2 way | Peak GB | Weights |
|---|---|---|---|---|---|
| GPT-2 way | 34,332 | 142,465 | 1.00× | 8.1 | 124.5M |
| New way (4,096 list) | 53,739 | 179,735 | 1.26× | 8.2 | 97.4M |
| New way without memory | 54,106 | 180,964 | 1.27× | 7.93 | 88.2M |
| New, 16,384 list | 46,607 | 184,453 | 1.29× | 7.89 | 106.8M |
| New, 16,384 list, sampled | 54,174 | 214,400 | 1.50× | 8.7 | 106.8M |
| New, GPT-2 list, sampled | 53,598 | 222,409 | 1.56× | 9.09 | 132.9M |
| New, phrase list | 29,429 | 162,475 | 1.14× | 9.32 | 144.6M |
| New, phrase list, sampled | 53,351 | 294,548 | 2.07× | 9.26 | 144.6M |
| New, windows of 256 | 60,835 | 203,468 | 1.43× | 8.2 | 97.4M |

The project's GPT-2 trainer `gpt128.py` ran about 16% faster than `research/arch.py`'s GPT-2 block (39,950 against 34,332 pieces/s). The notes estimate that the 5.7× below would become about 5.0× if only the GPT-2 way got that tuning.

**Equal time** (`research/test_124m.py`, about 8 minutes each, one start):

| Run | Updates | Seconds | Per letter at the end | At 467 s | Seconds to reach 1.278 | Gain |
|---|---|---|---|---|---|---|
| GPT-2 way, lr 0.0006 | 250 | 468.4 | 1.2777 | 1.2779 | 468.4 | 1.00× |
| New way, 4,096, lr 0.002 | 390 | 466.7 | 0.9151 | 0.9151 | 82.2 | **5.70×** |
| New way, GPT-2 list, sampled | 390 | 493.2 | 0.9209 | 0.9280 | 92.1 | 5.09× |
| New way, phrase list, sampled | 390 | 509.6 | 0.9830 | 0.9948 | 135.5 | 3.46× |
| GPT-2 way, lr 0.0012 | 250 | 469.1 | 1.3517 | 1.3520 | | |

The higher GPT-2-way rate was worse, so the 5.7× does not come from an under-tuned (too low) GPT-2-way rate.

**Fact test** (`research/facts_quiz.py`; 990 MB of `mix_v2.txt` in both lists):

| | GPT-2 way | New way |
|---|---|---|
| Updates | 1,880 | 992 (stopped the first time it reached the same held-out level) |
| Training time | 3,598.8 s | 1,166.1 s |
| Per letter | 0.7760 | 0.7760 |
| Letters read | 514M | 219M |
| Model file | `models/facts_124m_normal.pt` | `models/facts_124m_new.pt` |

- **3.1× faster = 2.3× less text × 1.3× faster reading.**
- The lead shrinks as quality rises: 4.9× at 1.278, 4.5× at 1.1, 4.0× at 1.0, 3.7× at 0.95 and 3.1× at the full level.
- Quiz (117 questions; guessing gives about 29.25):
  - Picking: 41 against 37. Own words: 4 against 4.
  - The questions only one of them got right split 14 against 10 (sign test 54%).
  - The conclusion: the same level and the same facts, as far as this quiz can tell. Both know very few facts after an hour.

### 11.11 Precision: 8-bit kept, 4-bit dropped

8-bit at 124M is in section 5. 4-bit (NVFP4) was tested on 10 November, before the main experiments, on a 500.6M test model (`research/fp4_sm120.py`, `research/logs/t1.log`, `t2.jsonl`, `profile_run_*.log`):

- `research/fp4_sm120.py` makes torchao 0.18's NVFP4 training layer run on the consumer Blackwell GPU (sm_120). It does stochastic rounding of gradients in arithmetic, because the hardware instruction is B200-only.
- **One layer** (8,192 tokens, 1536 → 6144):

  | Format | Time | Throughput | Error |
  |---|---|---|---|
  | BF16 | 7.73 ms | 60.0 TFLOPS | 0.002 |
  | FP8, compiled | 4.82 ms | 96.3 TFLOPS | 0.037 out, 0.059 gradients |
  | NVFP4, compiled | 2.81 ms | 164.9 TFLOPS | 0.146 out, 0.177 / 0.167 gradients |

- **Whole model, with activation recompute** (tokens/s):

  | Setting | Tokens/s |
  |---|---|
  | BF16 + AdamW | 9,070 |
  | BF16 + Muon | 8,805 |
  | FP8 + AdamW | 9,427 |
  | FP8 + Muon | 10,825 (later 11,053 / 11,073) |
  | NVFP4 + Muon | 7,004 (8,034 after a fix) |

- **Profile of one step:** the FP4 multiplies took 154.7 ms, but FP4 conversion took 167.9 ms and copies and fills 522.9 ms.
- So on this laptop, 4-bit was faster per multiply but slower end to end than 8-bit, with about 4× the error. No written conclusion about 4-bit was found; this reading comes from the logs.
- In the same logs, token-level mixture of experts took 19.62 ms for ragged per-token groups against 9.04 ms dense for the same arithmetic, which ruled it out on one GPU.

### 11.12 What was kept, and the total gain

**Kept**, with the deciding evidence:

| Change | Evidence |
|---|---|
| Muon | −0.073 at 1M |
| Rotary | −0.055 |
| Lookup memory | −0.038 at 1M; a tie at 10M with 4,096 pieces; free in speed at 124M |
| SwiGLU | −0.015, and it helps alongside memory |
| QK-norm | −0.015 at 10M |
| 4,096 own pieces, all scored | Best at equal time at 10M and 124M |
| `torch.compile` | Halves the new design's round time |
| Peak rate 0.002 | Best at 10M and 30M |
| 8-bit block multiplies | +8.4% speed, −7% time to equal loss, equal quiz |

**Total gain against the GPT-2 way** (GPT-2 design + AdamW + GPT-2 pieces):

| Size and test | Gain |
|---|---|
| 10M | 5.1× (3 starts each) |
| 30M | 4.4× |
| 124M after 8 minutes | 5.7× |
| 124M at the one-hour level | 3.1× |

The lead shrinks as quality rises. The final SlingerGPT run was not compared against a GPT-2-way run of the same length.

---

## 12. Early models

The early models were made with `tiny_gpt.py`, before SlingerGPT. It is a plain GPT-2-style model:

- learned positions, pre-LayerNorm, fused QKV with `scaled_dot_product_attention(is_causal=True)`, and a GELU feed-forward at 4 × width;
- dropout 0.2 by default, bf16;
- AdamW at 1e-3, a 200-step warm-up then a cosine to a tenth, clip 1.0;
- it keeps the best held-out checkpoint, scored on the last 10% of the text.

Sizes (`tiny_gpt.py:16`):

| Size | Layers | Heads | Width | Window |
|---|---|---|---|---|
| `1m` | 4 | 4 | 128 | 64 characters |
| `10m` | 6 | 6 | 384 | 256 |
| `50m` | 10 | 10 | 640 | 256 |

The units are characters by default; `--tokens words` uses the top 2,000 words plus `<unk>`, with tied tables. `tiny_gpt.py` also holds the DRY sampler used by the chat page.

**Text:**

- `build_corpus.py`: TinyStories, Project Gutenberg, SODA and Simple English Wikipedia, cleaned to printable ASCII and shuffled by document.
  - `mix.txt` is 860M characters.
  - `mix_big.txt` (`--stories 600 --books 600 --dialogues 400 --wikipedia 265`) is 1,864,795,041 characters.
- `build_corpus_v2.py`: `mix_v2.txt`, 2,000,235,814 characters. It holds FineWeb-Edu 800M, new TinyStories 300M, new Gutenberg books 250M, SODA 250M, and 400M characters of `mix_big.txt` replayed in blocks of about 50,000 characters.
- **`mix_v2.txt` is the text every 1M-124M experiment in section 11 trains and is scored on.**

| File | What it is | Result | What it taught |
|---|---|---|---|
| `models/placard_1m.pt` | 1,059,536-weight word-level model (2,000 words) on one public-domain novel (182,337 words), trained on CPU | 1,600 steps, 327 s, best held-out 3.81 per word (`logs/placard_1m.log`); poster `placard/evolution.png` | A 1M model learns dialogue shape in about 1,000 steps. On 182K words it overfits after about 1,600 (the same-text GPU run went back up from 3.78 to 4.17 by step 6,000) |
| `models/mix10m.pt` | 10,819,680-weight character model on `mix.txt` | 48,000 steps, 2,509 s, best held-out 0.89 per character (`logs/mix10m.log`) | The data mix recipe that became `mix_v2.txt`; the same 6 × 384 body as the `10m` experiments |
| `models/mix50m.pt` | 49,523,296-weight character model on `mix_big.txt` | 102,000 steps, 5.0 h, best held-out 0.68, peak 4.8 GB (`logs/mix50m.log`) | The largest model of the old recipe; the "before" half of continued training |
| `models/mix50m_v2.pt` | `mix50m` continued on `mix_v2.txt` | 110,000 steps, 5.4 h. On `compare_models.py`'s fixed test: new mix 0.870 → 0.814, first mix 0.711 → 0.704 (`logs/scores.json`) | Continued training with 20% replay learned new text and forgot nothing measurable. The DRY setting was first tuned on it: held-out repeated phrases 4.32% → 0.34%, loops 11.5% → 0% (`logs/dry.log`) |
| `models/gpt128.pt` | 124,526,208-weight GPT-2-style model on GPT-2 pieces (`prepare_tokens.py`, `gpt128.py`) | Paused at update 5,068, held-out 3.332 per piece at 5,000 (`logs/gpt128.log`); about 39,950 pieces/s | The speed reference for the GPT-2 way |

**Side branches** (results only):

- **Distillation** (`distill.py`): two 10M students, A on text alone and B with loss = 0.9·KL(student ‖ teacher at temperature 2)·2² + 0.1·cross-entropy, teacher `mix50m_v2`.
  - After 48,000 steps on unseen text: A 0.930, B 0.913, teacher 0.814.
  - It cost about 1.5× the time: 7,491 s against 5,022 s.
  - It was not used later.
- **Three no-training weight edits** on `mix50m_v2` all failed: rank reduction, blending the v1/v2 weights, and removing a repetition direction (`logs/laser.log`, `logs/surgery.json`).

---

## 13. Results of the final model

### 13.1 Test suite with the likeliest piece (`research/test_slinger.py run/extra`)

**Measures:**

- **Held-out:** the full test of section 6.8.
- **Quiz:** 117 fact questions. Picking is the likeliest of 4 options by per-letter log-probability; own words means the answer appears in the first 40 letters written.
- **Sums:** 50 sums (seed 2026).
- **Who:** 20 identity questions, 10 fresh and 10 after a long answer.
- **Lookup:** 30 narrow and 30 broad questions, each without and with the top search passage.
  - An answer is right if a key regex hits in the first 200 letters, "not said" if it says it doesn't know, otherwise wrong.
  - Lookup and quiz-with-passage use the improved search, except for Preview 1, `fix` and `fix2`.
- **No-answer:** 10 questions with no real answer.

Sources: `research/logs/slinger_test_*.json`, `slinger_extra_*.json`.

| Measure | GPT-2 small (OpenAI) | base `pico_full` | Preview 1 | `final` | `fix` | `fix2` | `redo` | `redo2` | **`chat2` (live)** |
|---|---|---|---|---|---|---|---|---|---|
| Held-out per letter | 0.6730 | 0.5319 | 0.5637 | 0.5424 | 0.5377 | 0.5440 | 0.5382 | 0.5382 | **0.5360** |
| Quiz picking /117 | 77 | 83 | 81 | 83 | 81 | 79 | 80 | 80 | **85** |
| Quiz own words /117 | 27 | 55 | 46 | 60 | 66 | 58 | 59 | 59 | **58** |
| Sums /50 | | | 0 | 29 | 29 | 28 | 32 | 16 | **33** |
| Who it is /20 | | | 10 | 18 | 19 | 18 | 19 | 19 | **20** |
| Narrow, no lookup: right/wrong/not said | | | 7/23/0 | 7/12/11 | 9/20/1 | 8/13/9 | 5/8/17 | 0/1/29 | **12/18/0** |
| Narrow, lookup: right/wrong/not said | | | 14/16/0 | 14/5/11 | 13/7/10 | 12/3/15 | 17/6/7 | 13/4/13 | **16/14/0** |
| Broad, no lookup: right/wrong | | | | 20/10 | | 25/5 | 21/7 | 21/5 | **25/5** |
| Broad, lookup: right/wrong/not said | | | | 13/3/14 | | 14/13/3 | 27/3/0 | 27/3/0 | **20/10/0** |
| No-answer questions declined /10 | | | | 5 | | | 6 | 10 | **1** |
| Quiz with passage: picking / own words | | | 94/59 | 100/76 | 99/72 | 97/68 | 98/69 | 98/71 | **99/76** |

GPT-2 small's held-out score is not a like-for-like comparison: the documents are held out from SlingerGPT's own training sources.

Held-out per source for `chat2`:

| Source | Score |
|---|---|
| fineweb_best | 0.6483 |
| fineweb_good | 0.6815 |
| cosmopedia | 0.4211 |
| smoltalk | 0.3309 |
| soda | 0.3580 |
| wikipedia | 0.7014 |
| simplewiki | 0.6351 |
| finemath | 0.5996 |
| orca_math | 0.2683 |

**Gate against `final`** (`research/fix_gate.py`, `research/logs/gate_chat2.out`). ADOPT requires nothing to get worse, so the verdict was KEEP. The project switched anyway, on the chat-suite result below.

- **Better:** held-out 0.5424 → 0.5360; quiz picking 83 → 85; sums 29 → 33; who 18 → 20; lookup right 14 → 16; broad with lookup right 13 → 20; broad without lookup wrong 10 → 5; "doesn't say" with no passage 11 → 0.
- **Worse:** no-real-answer questions made up 5 → 9; lookup wrong 5 → 14; broad with lookup wrong 3 → 10; without lookup wrong 12 → 18; quiz own words 60 → 58; quiz with passage picking 100 → 99.

### 13.2 Chat suite with the page's settings (`research/chat_suite.py`)

1,400 replies: 140 items × 10 tries with fixed seeds, written by the page's own code at temperature 0.3, top-k 50, DRY 0.8 and 300 words. A difference counts as real only at |z| ≥ 2: a two-proportion z, or an unequal-variance z for word counts. Counts come from `research/logs/chat_suite_*.json`.

| Measure | `final` | `chat2` | z | Verdict |
|---|---|---|---|---|
| Facts asked formally, right | 96/300 (32.0%) | 129/300 (43.0%) | +2.8 | Better |
| Facts asked casually, right | 44/300 (14.7%) | 64/300 (21.3%) | +2.1 | Better |
| Casual sums right | 72/100 | 94/100 | +4.1 | Better |
| Instructions followed | 17/40 | 28/40 | +2.5 | Better |
| Words after "Hello!" (mean; median 27.5 → 189.5) | 61.8 | 180.9 | +6.8 | Better |
| "The text doesn't say." with no text | 195/1,400 | 0/1,400 | +14.5 | Better |
| Greetings ending properly | 60/60 | 60/60 | 0 | Same |
| Topic switch right | 84/120 | 95/120 | +1.6 | Same |
| Old topic leaks in | 15/120 | 15/120 | 0 | Same |
| Right after a wrong answer in the chat | 12/40 | 13/40 | +0.2 | Same |
| Follow-ups stay on topic | 97/100 | 93/100 | −1.3 | Same |
| Identity | 40/40 | 40/40 | 0 | Same |
| Facts late in a chat | 20/40 | 25/40 | +1.1 | Same |
| Cut off at the limit | 50/1,400 | 52/1,400 | −0.2 | Same |
| Impossible questions declined | 31/100 | 3/100 | −5.3 | Worse |
| Words in a new chat (mean) | 280.8 | 257.3 | −2.3 | Flagged |
| Replies repeating a 4-word phrase 3+ times | 205/1,400 | 274/1,400 | −3.5 | Flagged |

Notes on the rows:

- **Impossible questions.** 30 of `final`'s 31 declines were "The text doesn't say.", the same leak that hit answerable questions. The plan gives 6 declines for `chat2`, against 3 in the JSON (unconfirmed which is right).
- **Words in a new chat.** Judged not real, because fewer replies hit the length cap (32 → 19).
- **Repetition.** Judged not real, because `chat2` writes more long replies. Among replies of 150+ words: 153/190 (81%) for `final` against 222/308 (72%) for `chat2` by recount (the plan says 74%).

---

## 14. Reproducing it

Run everything from the repository root with `python -P`, as in the scripts' docstrings. Run **one GPU job at a time**: two jobs at once slowed the proxy runs to about 20,000 pieces/s. Keep the laptop on mains power (7.5).

### 14.1 Software

Python 3.14, PyTorch 2.11.0 (CUDA 12.8 build), torchao 0.18.0, tokenizers 0.23.3, numpy 2.4.3 and pyarrow 26.0.1. These are the versions installed afterwards; the versions at training time were not recorded.

`torch.compile` and torchao FP8 need Triton (`gpt128.py` docstring). The experiments also use `tiktoken` for GPT-2's list.

### 14.2 Raw text

No download script is included. The scripts expect these parquet files:

| Path | Content |
|---|---|
| `data/raw128/fineweb_edu_000.parquet` … `_004.parquet` | FineWeb-Edu `sample/10BT`, parts 000-004 (that 000-001 are from `sample/10BT` is unconfirmed) |
| `data/raw128/cosmo_khanacademy_0`, `cosmo_openstax_0/1`, `cosmo_stanford_0/1`, `cosmo_wikihow_0/1` `.parquet` | Cosmopedia v1 subsets (7 files, 486,574 rows) |
| `data/raw128/soda_train.parquet` | `allenai/soda`, train |
| `data/raw128/wiki_en_00.parquet` … `_05.parquet` | `wikimedia/wikipedia`, `20231101.en`, 6 files |
| `data/raw_slinger/cosmopedia-v2/train-0000{0..3}-of-00104.parquet` | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` |
| `data/raw_slinger/finemath/train-0000{0..6}-of-00064.parquet` | `HuggingFaceTB/finemath`, `finemath-4plus` |
| `data/raw_slinger/smol-smoltalk/train-0000{0..3}-of-00004.parquet` | smol-smoltalk |
| `data/raw_slinger/simplewiki/train-00000-of-00001.parquet` | Simple English Wikipedia |
| `data/raw_slinger/orca-math/train-00000-of-00001.parquet` | Orca-Math word problems |
| `data/raw_slinger/squad_v2/{train,validation}-00000-of-00001.parquet` | SQuAD 2.0 (fine-tune) |
| `data/raw_slinger/gsm8k/{train,test}-00000-of-00001.parquet` | GSM8K (fine-tune) |

**For an exact copy of the first cut** (a deduction from the code), only the first-cut files may be present when `research/prepare_slinger.py` runs:

- FineWeb-Edu 000-001;
- Cosmopedia v2 file 0;
- FineMath 0-2;
- `wiki_en_00`-`01`.

`prepare_slinger.py` globs every file present. More FineWeb files would change fineweb_best and the fineweb_good bar, and more Cosmopedia v2 files would change Cosmopedia's reading order. Add the other files before the second cut.

### 14.3 Steps

```
# 1. First cut and the 4,096-piece list -> data/slinger/ (tokenizer.json, <source>/train.bin, val.bin, manifest.json)
python -P research/prepare_slinger.py

# 2. Second cut -> data/slinger_more/ (all sources, or name some to cut one per process)
python -P research/prepare_slinger_more.py
python -P research/prepare_slinger_more.py cosmopedia

# 3. Pre-training run 1: 30,520 updates, warm-up 10%. As run: 16-bit to update 913, then 8-bit.
python -P research/train_slinger.py train --stop-after 913
python -P research/train_slinger.py train --fp8
#    (python -P research/train_slinger.py status | pause, to watch or stop; the same train command carries on)

# 4. Pre-training run 2: 61,035 more updates from run 1's checkpoint, re-warm 2%, last tenth in 2,048 windows
python -P research/train_slinger.py train --fp8 --name slingergpt1_pico_full --start-from slingergpt1_pico \
    --updates 61035 --warm 0.02 --long --title "SlingerGPT 1 pico full"

# 5. Score the base
python -P research/test_slinger.py run slingergpt1_pico_full

# 6. Fine-tune data -> data/slinger_chat/
python -P research/finetune_slinger.py prepare     # chat
python -P research/finetune_slinger.py identity    # who
python -P research/finetune_slinger.py maths       # maths
python -P research/finetune_slinger.py reada       # reada
python -P research/finetune_slinger.py mixed       # mixed

# 7. Chat fine-tune, then copy models/candidates/slingergpt1_pico_chat2.pt (and its .tokenizer.json) to models/
python -P research/finetune_slinger.py train --base slingergpt1_pico_full --name candidates/slingergpt1_pico_chat2 \
    --title "SlingerGPT 1 pico" --steps 2000 --save-at 1000,1500 \
    --mix chat=0.26,mixed=0.22,who=0.02,reada=0.12,maths=0.10,general=0.20,plain=0.08

# 8. Wikipedia index for lookup and the lookup tests (CPU, about half an hour, 6.59 GB); titles.db builds on first search
python -P research/lookup.py build
python -P research/lookup.py search "What is a rainbow?"

# 9. Tests
python -P research/test_slinger.py run slingergpt1_pico_chat2
python -P research/test_slinger.py extra slingergpt1_pico_chat2
python -P research/fix_gate.py slingergpt1_pico_chat2
python -P research/chat_suite.py run models/slingergpt1_pico_chat2.pt
python -P research/chat_suite.py compare <candidate json stem> <baseline json stem>

# 10. Chat page (serves locally on port 8765; one model on the card, one reply at a time)
python -P chat_server.py
```

`--stop-after` stops as if the run were paused, and the checkpoint keeps the data RNG, so the resumed run reads the same windows. A 16-bit checkpoint carries on under `--fp8` as is (`research/train_slinger.py:101-103`). Training with `--fp8` from the start is the simpler choice but differs from the original for its first 913 updates.

### 14.4 The design experiments

These need `data/corpus/mix_v2.txt`:

1. Build `mix_big.txt` with `build_corpus.py` and the flags in section 12.
2. Then run `build_corpus.py` (sources in `data/raw/`) and `build_corpus_v2.py` (sources in `data/raw_v2/`).

```
python -P research/prepare_pieces.py                          # data/pieces/ (1,024 / 4,096 / 16,384 / GPT-2 lists)
python -P research/lab.py lr --size 1m
python -P research/lab.py stage1 --size 1m                    # the 64-run sweep
python -P research/lab.py stage2 --size 1m --base muon,rope,qknorm,swiglu,engram
python -P research/lab.py show stage1@1m
python -P research/final_test.py rates 10m
python -P research/final_test.py pick
python -P research/final_test.py run 10m
python -P research/final_test.py report 10m                   # and the same for 30m
python -P research/speed_124m.py VARIANT                      # one variant per process
python -P research/speed_124m.py report
python -P research/test_124m.py run
python -P research/test_124m.py report
python -P research/facts_quiz.py count | train | quiz | report  # needs data/pieces1g (research/prepare_more_text.py)
```

---

## 15. Known limits and unconfirmed points

### 15.1 Limits of the model

- **Few facts.** Formal fact questions are right 43% of the time with the page's settings, and casual ones 21%.
- **Lookup is off.** With the passage the live model gets broad questions wrong more often: 10 against 5 of 30.
- **It rarely declines.** It turns down impossible questions only 3 of 100 times on the page, and 1 of 10 in the likeliest-piece test.
- **Repetition remains.** 274 of 1,400 replies repeat a 4-word phrase 3+ times, even with DRY.
- **Inconsistent digit splits** (3.1) may hurt arithmetic. Sums are right 33/50 in the likeliest-piece test and 94/100 casual sums on the page.
- **No document masking.** Attention crosses document boundaries inside training windows.
- **Context.** The model was trained on 2,048-piece windows only in the last 10% of run 2 and in the fine-tune. Past 2,048 pieces the chat page falls back to the plain path.
- **8-bit writing hurts answers** (10.5). Replies must be written in 16-bit.
- **Evaluation limits.** One seed for most design comparisons. Held-out scores are on the project's own sources. The 4,096 list was trained on the experiment text and GPT-2's was not. The new way's peak rate was tuned at 10M and 30M only.

### 15.2 Unconfirmed points

1. **8-bit from update 913 in run 1.** The notes say run 1 carried on with `--fp8` from update 913. The pause line at update 1,700 in `research/logs/slingergpt1_pico.log` prints the command without `--fp8`. Speeds of about 60,000+ pieces/s after update 2,200 fit 8-bit (60,090) better than 16-bit (55,451). The cfg saved inside the model files was not read.
2. **Library versions** at training time were not recorded. The Muon and Float8 details come from the installed sources (PyTorch 2.11.0, torchao 0.18.0).
3. **Effective ε of the QK-norm and Engram RMSNorms** (`eps=None` → machine epsilon of the input dtype: 2^-7 in bf16, about 1.2e-7 in float32). Not checked. The exact dtypes inside attention under autocast were not traced at runtime.
4. **The cuDNN attention kernel.** That it was selected for every training step was not checked, and the 1.6× claim comes from a code comment.
5. **The Engram table's 5× rate and no weight decay** are attributed to the Engram paper; not checked against it.
6. **Published figures** were summarised in the notes, not measured by the project: Engram 1.808 → 1.768, Muon about 1.3× and 52% of AdamW's FLOPs, and the MobileLLM tying result.
7. **Data sources.**
   - The Hugging Face ids of smol-smoltalk, Simple English Wikipedia, Orca-Math and the Cosmopedia v1 files are not recorded. Likely ids: `HuggingFaceTB/smol-smoltalk`, `wikimedia/wikipedia` `20231101.simple`, `microsoft/orca-math-word-problems-200k` (200,035 rows fits) and `HuggingFaceTB/cosmopedia`.
   - That `fineweb_edu_000`-`001` come from `sample/10BT` is not stated in the code.
   - The first cut's fineweb_good bar (3.031) is deduced; no log of the first cut exists, so its run time is unknown. The second cut's smoltalk and soda counts come from the manifest only.
8. **Peak GPU memory** during the real runs was not logged, and memory with 2,048-piece windows is unknown.
9. **Run 2 timing.**
   - The cause of the speed step from about 68,000 to about 62,500 pieces/s is not recorded.
   - The length of the single cool-down is inferred, not logged.
10. **Read counts** per source (6.7, 8.5) are expected values, not tallies.
11. **The fine-tune window of 2,048** follows from `--long` and code comments; the checkpoint was not opened. Preview 1's window (presumably 1,024) was not checked.
12. **Fine-tune attempts.** The learning rates of `fix` and `fix2` are not in their logs; the argparse help mentions 0.0002 for "the fix round".
13. **`ask` counts** (9,150 / 4,661) come from a docstring and `research/final_report.py`, not a log.
14. **Chat-suite readings.**
    - Impossible questions declined by `chat2`: 3 of 100 in the JSON, against 6 in the plan.
    - Repetition among replies of 150+ words: 72% by recount, against 74% in the plan.
15. **Speeds of the fast writing path** (87 → about 1,400 pieces/s; 766 → 221 GPU jobs) come from notes and docstrings only. Top-choice agreement is given as 1,012/1,016 in one place and 1,013/1,016 in another.
16. **Search quality on quiz sentences** (84 → 87 of 117) is from the notes only.
17. **The DRY and temperature defaults' measured effect** (15% → 7%; 25 against 18 of 40) is a comment in `web/app.js` only.
18. **Notes-only figures.** The 100-step Engram proxy test, the adaptive-softmax timing, the 200-step 1M smoke runs and the large-memory `gpt128.py` attempt appear only in `research/phase2_notes.md` (or a diff), not in `research/logs/lab.jsonl`.
19. **4-bit.** Its rejection is read from the 500.6M throughput logs; no written conclusion exists, and it was not measured on the 124M model.
20. **Early models.**
    - The exact commands of the `tiny_gpt.py` runs for `mix10m`, `mix50m` and `mix50m_v2` are not recorded. `data/corpus/mix.txt` is no longer on disk.
    - `logs/tale1m_words.log` names a different file for its best model.
