# SlingerGPT 1 pico

A 97,397,760-weight language model trained from scratch on one laptop GPU (RTX 5070 Ti Laptop, 12 GB) in 27.3 hours, then fine-tuned for chat in 37 minutes.

Its training recipe was redesigned from the GPT-2 way (GPT-2 block, AdamW, GPT-2's 50,257 word pieces):

- the Muon optimiser for the block matrices;
- rotary positions, QK-norm and a SwiGLU feed-forward;
- an Engram lookup memory (hashed 2- and 3-piece tables);
- its own 4,096-piece word list;
- 8-bit block multiplies.

Against the GPT-2 way, the new recipe reached the same quality 5.1x faster at 10M weights, 4.4x at 30M and 3.1-5.7x at 124M.

| | |
|---|---|
| Layers x width | 12 x 768, 12 heads, window 2,048 |
| Text read | 6.0 billion pieces from 9 sources (educational web, textbook-style writing, Wikipedia, maths, conversations) |
| Held-out wrongness per letter (lower is better) | 0.5360 (OpenAI GPT-2 small on the same documents: 0.673) |

**[docs/ENGINEERING.md](docs/ENGINEERING.md)** explains every part of the design in depth: the exact architecture and parameter count, each change from GPT-2 with its measured effect, the optimiser, 8-bit training, the data, both training runs, the chat fine-tune, fact lookup, how replies are written, and the experiments behind each choice.

## Download the model

The model files are too big for the repository, so they are on the [Releases page](../../releases):

| File | What it is |
|---|---|
| `slingergpt1_pico_chat2.pt` + `slingergpt1_pico_chat2.tokenizer.json` | SlingerGPT 1 pico, the chat model |
| `placard_1m.pt`, `mix10m.pt`, `mix50m.pt`, `mix50m_v2.pt` | Early models from before SlingerGPT (1M, 10M, 50M), made with `tiny_gpt.py`; see section 12 of the engineering document |

Put the files in a `models/` folder at the repository root.

## Talk to it

Needs Python 3.14, PyTorch 2.11 with CUDA, torchao 0.18, tokenizers and numpy (exact versions in section 14.1 of the engineering document).

```
python -P chat_server.py
```

Then open http://127.0.0.1:8765. "Look it up" (Wikipedia search) is off by default; `research/lookup.py build` makes the search index it needs.

## What is where

| Path | What it is |
|---|---|
| `research/arch.py` | The model |
| `research/prepare_slinger.py`, `research/prepare_slinger_more.py` | Training text and the 4,096-piece word list |
| `research/train_slinger.py` | Pre-training |
| `research/finetune_slinger.py` | Chat fine-tune data and trainer |
| `research/lookup.py` | Wikipedia fact lookup |
| `research/test_slinger.py`, `research/chat_suite.py`, `research/fix_gate.py`, `research/facts_quiz.py`, `research/report_card.py` | Tests |
| `research/lab.py`, `research/final_test.py`, `research/test_124m.py`, `research/speed_124m.py`, `research/pieces_report.py` | The experiments that chose the design |
| `chat_server.py`, `web/` | The chat page |
| `tiny_gpt.py`, `build_corpus*.py`, `distill.py`, `compare_models.py`, `prepare_tokens.py`, `gpt128.py` | Early work and the GPT-2-way reference trainer |
| `research/492m/` | **Not ready:** plans for a larger 492M model (data mix and fine-tune plan) |

Section 14 of the engineering document gives every command in order to reproduce the model. The training text is not included (tens of GB); the document lists the exact datasets.

## Licence

The code is under the MIT licence ([LICENSE](LICENSE)). The model was trained on data with its own licences (FineWeb-Edu, Cosmopedia, Wikipedia, SODA, SQuAD 2.0 and others), listed in section 6.1 of the engineering document.
