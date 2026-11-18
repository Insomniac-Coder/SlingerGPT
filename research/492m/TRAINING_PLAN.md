# SlingerGPT 492M: the training plan (for a Claude Code cloud session)

This file is the hand-over. A new session should be able to read it, ask the owner the questions in section 3, and
get to work without any earlier conversation. It sits in the public SlingerGPT repository, so it holds no secrets.

Read first, in this order:
1. This file.
2. `docs/ENGINEERING.md`: how SlingerGPT 1 pico (97M) was built. The 492M uses the same recipe, scaled.
3. `research/492m/data_mix.md`: the 80-billion-piece training text, with sources, shares, licences, cleaning and the
   word-piece list. Its note at the top changes the window lengths (section 2 below has the final version).
4. `research/492m/finetune_plan.md`: the chat fine-tune, done after pre-training.
5. `research/arch.py` and `research/train_slinger.py`: the model and the 97M trainer that the new trainer is based on.

Status on 18 November 2026: the design, sizes and schedule are decided. Stage A (the code, tested on the CPU) is
done: see section 8 for what exists, what was found, and the rental-day commands. The data list, the chat layout and
the fine-tune behaviour rules still need the owner's yes. Nothing is rented; the B200 quota request is waiting at
Nebius (the owner already has 8 H200s in us-central1).

---

## 1. Working with the owner (follow these exactly)

- **Explain in plain words, step by step, from zero.** Answer the exact question asked; if the owner is lost, make it
  shorter and simpler, never longer. No jargon without a one-line meaning. Chat answers stay in chat; make a PDF only
  when asked.
- **Every update says what is happening** ("Not training yet: downloading text", "Training: 31% done, ...").
  The owner wants frequent updates while something is running.
- **Money:** ask before every action that costs money (creating or starting a server, creating a disk, keeping a
  server on longer than agreed), with the price per hour and the expected hours. Never leave anything running or
  stored when a job is done without saying so. See section 3 for the standing limits the owner sets.
- **Ask before:** stopping or changing a running job (a question about a run is not an instruction to change it),
  downloading anything not on the approved list, pushing or publishing anything (GitHub, Hugging Face), launching a
  Workflow or several agents at once, and any choice about how the model behaves (what it says, refuses, or how long
  its replies are). Approving a dataset is not approving the behaviour it teaches.
- **The owner decides** which model is kept. Nothing is deleted or set aside without asking.
- **Comparisons** of speed or quality are against normal practice (GPT-2 design + AdamW + standard word pieces), not
  against an earlier SlingerGPT, unless the owner asks.
- **Privacy:** never print, log or commit credentials (the Nebius credentials file, SSH private keys, tokens). Put
  no personal data in URLs, logs or files sent anywhere.
- **Small fixes** are done directly; agents are only for big jobs, and only after the owner says yes to the plan.
- Write helper scripts to files instead of long one-line commands. Try every new data step on a few hundred
  documents and read the samples before running it on everything (this caught two design mistakes and a batching bug
  on the 97M).

---

## 2. What is decided

### The model: 492,406,016 weights

The same design as SlingerGPT 1 pico, scaled:

```python
arch.Cfg(layers=22, width=1280, heads=20, ffn=3456,        # the "500m" shape in research/arch.py
         vocab=16384, context=16384, modern=False,          # LayerNorm with biases, as the pico
         rope=True, qknorm=True, swiglu=True,
         engram=(1,), engram_slots=65521, engram_hashes=4, engram_dim=64, pad=0,   # Engram 4x the pico's
         fp8=True)                                          # block Linears in 8-bit (torchao), as the pico
```

- A meta-device build must give exactly **492,406,016** weights: blocks 436,569,344; piece table 20,971,520 (shared
  with the output); Engram 34,865,152. Check this before anything else.
- `context=16384` only sizes the rotary tables; the windows actually read are set by the schedule below.
- Optimisers exactly as `arch.optimizers(model, "muon", lr)`: Muon (`match_rms_adamw`) for the block and Engram
  matrices, AdamW (betas 0.9/0.95, weight decay 0.1 on the tables) for the rest, Engram tables at 5x the rate with no
  weight decay. Gradient clipping 1.0. bf16 autocast.
- **Rotary base:** 10,000, raised to **1,000,000** when the 16,384 phase starts. `research/arch.py` hard-codes 10,000
  (line ~208): add a `rope_base` field to `Cfg` (default 10,000) and a way to rebuild the rotary tables mid-run.
- The output layer's loss: use `research/fused_ce.py` (sliced, so the full table of 16,384 scores per piece never
  sits in memory) if memory needs it.

### The text: 80 billion pieces

- Sources, shares, passes and licences: `research/492m/data_mix.md`. 31 sources: web 39.5%, code 11.9%,
  computer-science knowledge 6.3%, maths 9.5%, textbook-style 9.3%, Wikipedia 7.2%, books 5.3%, Q&A 4.9%,
  conversations 3.3%, science 2.8%. Its anneal and long-phase sections give the per-phase shares.
- **Mix phases:** main 0-70B; anneal 70-78B (the best material only); long documents 78-80B.
- **Window schedule (owner, 18 November):**

| Pieces | Window | Mix phase |
|---|---|---|
| 0-25B | 1,024 | main |
| 25-50B | 2,048 | main |
| 50-70B | 4,096 | main |
| 70-78B | 4,096 | anneal |
| 78-80B | 16,384, rotary base 1,000,000 | long documents |

  Keep the pieces per update the same in every phase (fewer, longer windows as the window grows), as the pico did.
- **Word pieces:** a new 16,384-piece byte-level BPE, digits split one by one, trained on the sample described in
  `data_mix.md` ("The word-piece list"). `<|endoftext|>` is piece 0. Reserved pieces: see decision 2 in section 3.
- **Cleaning and decontamination:** as in `data_mix.md` ("Cleaning and decontamination"), with one change: it runs
  **on the GPU server**, not on a separate machine (the owner has no other machine; see section 5). The project's
  own test questions are in `research/test_slinger.py`, `research/chat_suite.py` and `research/facts_quiz.py`; remove
  any document that contains them, as well as MMLU, HellaSwag, ARC, GSM8K test, HumanEval and MBPP.

### Training settings

| Setting | Value | Why |
|---|---|---|
| Pieces per update | about 1,048,576 (2^20) | a normal batch for this size on 8 GPUs; check memory in the test run |
| Updates | 80B / 2^20 = about 76,300 | |
| Peak rate | picked in the test run from 0.0015, 0.002, 0.003 (section 5) | the pico used 0.002 at 65,536 pieces per update |
| Warm-up | 2% of the updates | |
| Decay | cosine to 10% of the peak over the whole run, as the pico (`train_slinger.py`: `0.1 + 0.45 * (1 + cos(pi * share))`) | proven on the 97M; it is already low during the anneal |
| Saves | every 15 minutes, plus a copy at 25B, 50B, 70B, 78B and 80B; keep the last 2 regular saves | |
| Held-out scores | every 1,000 updates, per source, as wrongness per letter | |
| Hardware | 1 machine, 8x NVIDIA B200 (Nebius spot), one copy of the model per GPU (DistributedDataParallel) | 492M fits on each GPU |

### Making the GPUs busy (owner said yes; no quality cost)

The time estimate assumes the GPUs reach only 30-40% of their top speed. Push it higher with:

- `torch.compile`;
- no activation recomputation (180 GB per GPU leaves room);
- the largest micro-batch that fits;
- the fastest attention kernel available for B200 (sm_100): try PyTorch's cuDNN and Flash backends and pick by
  measured speed;
- the fused loss;
- DDP gradient buckets that overlap with the backward pass;
- the data loader reading from memory-mapped files in background threads.

Measure each change in the test run; keep only what is faster and gives the same loss.

**Not used:** 4-bit training (owner: no). 16 GPUs over two machines: the owner is thinking about it; plan for 8.

### Time and money (estimates; the test run replaces them)

At about $3 per GPU-hour (Nebius spot, 18 November), 8 GPUs cost about $24 an hour.

| Step | Time | Cost |
|---|---|---|
| Set-up, word-piece list, first part of the text (GPUs waiting) | 30-90 min | $12-36 |
| Test run, stop test, rate check | about 30 min | about $12 |
| Training (the rest of the text is prepared alongside) | about 5-7.5 hours | about $120-180 |
| Chat fine-tune (if done in the same rental) | 50-85 min | $20-34 |
| **Total** | **about 7-10 hours** | **about $165-260; budget up to $350** |

If the test run's measured speed means training would take more than 9 hours, stop and tell the owner before
going on.

---

## 3. Questions for the owner before anything costs money

Ask these one at a time, in plain words, each with a recommendation:

1. **The data list:** approve `data_mix.md` (sources, sizes, licences). Also:
   - Will the model ever be used commercially? If yes, some sets are skipped.
   - Stack Exchange has asked since July 2024 that its downloads not be used to train AI. Use the older copy or not?
   - The optional NVIDIA Nemotron-CC question set: use it or not?
2. **The chat layout and reserved pieces** (must be settled before the word-piece list is built):
   - Recommendation: keep the pico's plain `System:` / `User:` / `Assistant:` lines.
   - Reserve a few spare special pieces (for example an end-of-turn piece and 6 unused ones) so the fine-tune never
     needs a new list.
3. **Spending limits:**
   - the most to pay per GPU-hour for spot (the server stops above it);
   - the total budget (suggest $350);
   - whether the server may be restarted automatically after the provider takes it back, within that budget, or
     whether to ask each time. The owner may be asleep, and waiting costs nothing but time.
4. **After pre-training:** keep the server for the fine-tune straight away (about 1-1.5 hours more), or save the
   model and fine-tune in a later rental?
5. **Where to keep the trained model:** a Nebius storage bucket (cheap, private), Hugging Face (public, needs a yes),
   or both.
6. **Before the fine-tune only (not before pre-training):** the behaviour rules and the other choices listed at the
   end of `finetune_plan.md`.

---

## 4. What the owner sets up (the session cannot do these)

1. A Nebius AI Cloud account with a payment card attached.
2. In the Nebius console: enough quota for 8 B200 GPUs (spot/preemptible) in the chosen region (us-central1 had
   B200 spot on 18 November). Ask Nebius support for more if the default is lower.
3. A Nebius **service account** (Administration -> IAM -> Create resource -> Service account), added to the
   `editors` group so it can create and delete servers and disks.
4. An **authorized key** for it:
   - a key pair is made with `openssl` (`private.pem`, `public.pem`);
   - `public.pem` is uploaded on the service account's page (Upload authorized key);
   - the key's ID is copied from its Authorized keys tab.
   The console does not make the key pair itself.
5. In the cloud session's environment settings:
   - `NEBIUS_PRIVATE_KEY_B64`: `private.pem` encoded as one line of base64 (environment settings may not keep
     line breaks). Secret: never commit it or paste it into chat. Decode it with `base64 -d`.
   - `NEBIUS_KEY_ID`, `NEBIUS_SERVICE_ACCOUNT_ID`, `NEBIUS_PROJECT_ID` and `NEBIUS_REGION` (us-central1). Not secret.
   - Optionally a Hugging Face read token (`HF_TOKEN`) for faster, less limited downloads.
6. Check that the cloud environment's network access allows the Nebius API and SSH to outside addresses. If not, the
   environment needs full network access.

The session decodes the private key into a file only it can read (`chmod 600`) and passes it to the SDK:
`SDK(parent_id=..., service_account_private_key_file_name=..., service_account_public_key_id=...,
service_account_id=...)`.

---

## 5. The work, in order

### Stage A: in the cloud session, before renting (costs nothing)

Build and test everything on the CPU first. The cloud sandbox is Linux, so real SIGTERM and multi-process tests work
there.

1. **Check the model:** a meta-device build of the config in section 2 gives 492,406,016 weights. Add `rope_base` to
   `research/arch.py`.
2. **`research/prepare_492m.py`**, the text pipeline. It runs on the server's processor cores and works in rounds:
   - **Download:** fetch each source's shards from Hugging Face (`hf_transfer` for speed) in random order, onto the
     data disk.
   - **Clean:** format, language filter, quality filter, exact duplicates, near-duplicates, URL overlap between the
     web sets, decontamination: all as in `data_mix.md`.
   - **Word pieces:** take the word-piece sample and train the 16,384-piece list first.
   - **Cut into pieces:** shuffle documents within each source, cut them into 16-bit piece files per source
     (`<source>/NNNN.bin`), and append to a manifest that lists only finished files.
   - **Rounds:** round 1 prepares about 5% of every source (about 4B pieces), enough to start training. Later rounds
     continue in the background while training runs, on the processor cores the trainer leaves free.
   - The trainer reads only files listed as finished, and logs how many pieces are prepared ahead of what it has
     read. If preparation ever falls behind, warn the owner.
   - **Long phase:** a separate set for 78-80B: long documents (books, papers, whole code repositories joined with
     file-name headers), one per window where possible, as in `data_mix.md`.
3. **`research/train_492m.py`**, the 8-GPU trainer, based on `research/train_slinger.py`:
   - `torchrun --nproc_per_node 8`, NCCL, DDP;
   - the window schedule and mix phases of section 2;
   - each GPU reads different windows; seed the reader from (seed, GPU number, update), so a resumed run reads the
     same windows with no reader state to save;
   - the rotary base switch at 78B;
   - the settings table of section 2;
   - logs and saves on the data disk.
   - **Drop the laptop's safety pauses:** no hourly 2-minute break and no 87 C cool-down pause on a data-centre
     server (they would waste paid time). Log GPU temperatures every few minutes instead, and tell the owner if any
     GPU throttles or passes 90 C.
4. **Saving on the spot warning:**
   - Nebius sends SIGTERM 60 seconds before it stops a spot server (then SIGKILL).
   - `research/train_slinger.py` on the owner's laptop already saves and stops on SIGTERM. The published copy predates
     that change, so build it into `train_492m.py`.
   - Every GPU's process sets a flag on SIGTERM. At the start of each update the processes agree with one tiny
     all-reduce. If any got the signal, GPU 0 saves (model, both optimisers, update number, time spent, curve) to a
     temporary file, renames it into place, and all processes stop together.
   - Saving must finish well inside 60 seconds: measure it.
5. **`research/nebius_ctl.py`**, using the Nebius Python SDK (`pip install nebius`, in the cloud sandbox only):
   - create the data disk: a 1 TB network SSD (owner, 18 November) that survives the server being taken back;
   - create the 8x B200 spot server with the data disk attached, an Ubuntu image with the NVIDIA driver and CUDA, the
     owner's maximum spot price and a new SSH key made in the sandbox;
   - status, start, stop, delete.
   - Treat the server as disposable and the data disk as the only thing that matters. If the sandbox loses its SSH
     key, make a new key and recreate the server from the same data disk.
   - Never create or start anything without the owner's yes (section 1).
6. **Tests on the CPU:**
   - a tiny model through `train_492m.py` with 2-3 CPU processes (gloo): loss falls, the copies stay identical, only
     GPU 0 writes, and save-and-resume reads the same windows;
   - a real `kill -TERM` during training: it saves and stops in seconds, and resumes;
   - the pipeline on a few hundred documents per source, samples read by eye;
   - a word-piece list trained on a small sample: digits are split, and code indentation survives.
7. Adapt the tests for the new word-piece list: `research/test_slinger.py` (held-out text, fact quiz, sums) and
   `research/chat_suite.py`.
8. Commit the new code to a branch. Ask the owner before pushing it anywhere.
9. Tell the owner what is ready, what it will cost, and ask the questions in section 3 that are still open.

### Stage B: the rental day (costs money: ask first)

1. With the owner's yes, create the data disk and the server. Copy the code to it (`scp` from the sandbox, or
   `git clone` if the branch was pushed with the owner's yes).
2. Install PyTorch (CUDA 12.8 or newer, with Blackwell support), torchao, tokenizers, numpy, pyarrow, datasets,
   huggingface_hub (its Xet downloader, HF_XET_HIGH_PERFORMANCE=1; hf_transfer is no longer used), and datatrove or similar for deduplication. Check `nvidia-smi` shows 8 B200s
   linked by NVLink.
3. Start `prepare_492m.py` in `tmux` (so it keeps running if the session disconnects). Build the word-piece list,
   then round 1.
4. **Test run** (about 30 minutes in all):
   - **Speed and memory:** micro-batch and attention-kernel choices; check pieces per second against the estimate
     (5-7.5 hours means about 3.0-4.4 million pieces a second for the whole machine). Stop and tell the owner if
     the full run would take more than 9 hours.
   - **Stop test:** `kill -TERM` the trainer, check the save time, resume.
   - **Rate check:** 200 updates each at 0.0015, 0.002 and 0.003 from the same start. Take the lowest held-out score.
     If two are within noise, take the lower rate. If the best is at an edge, tell the owner before going on.
5. **Training:** start the full run in `tmux` from update 0 with the chosen rate. Every 30 minutes or so:
   - check progress, speed, held-out scores, GPU health, pieces prepared ahead and money spent;
   - give the owner a short plain update.
6. **If the server is taken back:**
   - The save on the warning, plus the 15-minute saves, sit on the data disk.
   - Tell the owner. Start the server again, or recreate it on the same disk, as decision 3 in section 3 allows.
   - Resume with the same command. Note the lost minutes.
7. **At 80B:**
   - Save the final model (bf16 weights for use, about 1 GB, plus the full save for any later training).
   - Score held-out text per source.
   - Run the 16,384-window "find the detail" test from `finetune_plan.md`.
   - Copy the model to where decision 5 says.
8. **Fine-tune:** only if decision 4 says so, and only after the owner approves the behaviour rules. Follow
   `finetune_plan.md`.
9. **Shut down:**
   - Stop and delete the server.
   - Ask the owner whether to delete the data disk (it costs money every day it is kept).
   - Report the total time, the cost, and where everything is.

---

## 6. Lessons from the 97M that apply here

- **Fix the behaviour before training.** Write how the model should behave as plain rules and get the owner's yes.
- **Clean practice data.** Never train on the model's own answers. Keep every test question's words out of all
  training text.
- **Judge by the full tests.** The small held-out samples in the training log jump by about 0.01 between updates.
- **Keep the plain-text practice in the fine-tune** (about 8% helped). Keep copies partway through: longer was not
  better.
- **8-bit replies:** if the chat page will write replies in 8-bit, train with per-row scales and test short, casual
  questions against 16-bit before going live. The 97M answered unsure questions worse in 8-bit.
- **Build the fast reply path from the start:** KV cache, `torch.compile`, CUDA graphs (see section 10.4 of
  `docs/ENGINEERING.md`).
- **One training job at a time** on the server.

---

## 7. Files this plan expects to exist when it is done

| File | What |
|---|---|
| `research/arch.py` | + `rope_base` |
| `research/prepare_492m.py` | the text pipeline, in rounds |
| `research/train_492m.py` | the 8-GPU trainer, with save on the spot warning |
| `research/nebius_ctl.py` | create, start, stop, delete the server and disk |
| `research/492m/mix_492m.json` | per-phase, per-source shares taken from `data_mix.md`, approved by the owner |
| `research/492m/run_log.md` | what happened on the day: times, speeds, costs, interruptions, decisions |

---

## 8. Stage A: what is ready (18 November 2026)

### The code (branch `ccr-5dd2ad80-fvffsq`)

| File | What it does | Tested |
|---|---|---|
| `research/arch.py` | `rope_base` and `set_rope_base()`; document masks (flex attention on the GPU, a written-out mask on the CPU); `build(device=...)` | 492,406,016 weights on a meta build; mask and rotary switch checked |
| `research/train_492m.py` | the 8-GPU trainer: schedule, mix phases, a reader that follows from the update number alone, saves (15 min, milestones, on SIGTERM/SIGHUP/SIGINT), held-out scores, GPU heat, cost and "text ready ahead" in the log | `research/test_train_492m.py`: 3 CPU processes, every phase crossed |
| `research/prepare_492m.py` | the text pipeline in rounds (section 5, A.2) | `sample` mode on 300 documents of all 37 inputs, end to end |
| `research/nebius_ctl.py` | `check`, `prices`, `status`, `price`, `create-disk`, `create-server`, `start`, `stop`, `delete-*`, `ssh`; `--gpu b200` or `--gpu h200`; anything that costs money needs `--yes` | `check`, `prices` and `status` against the real account |
| `research/492m/make_mix.py`, `mix_492m.json` | per-phase shares from data_mix.md (draft until the owner approves) | shares add up to 100% in every phase; every mixed source has a producer |
| `research/test_slinger.py` | `SLINGER_HELD_OUT=<data>` scores the 492M's held-out text (per letter, so the pico compares fairly) | |

Trainer test results (CPU, 3 processes, tiny model): the loss falls (5.44 -> 1.45); the copies stay identical at
every check; only process 0 writes; a run stopped at update 25 and resumed reads exactly the same windows (with one
process it also ends with exactly the same weights; with several, DDP sums gradients in a different order in the
first update after a restart, so the weights differ in rounding only); a real `kill -TERM` saves and stops all
processes in about 1 second, and the run resumes to the end.

### Changes from the plan, and why

- **Launcher:** `train_492m.py train --gpus 8` starts the processes itself instead of `torchrun`. torchrun kills its
  workers 30 seconds after a SIGTERM, which could cut a save short; this launcher passes the signal on and waits.
  SIGHUP is handled too, because killing tmux sends it.
- **Reading order:** each source is read in order as one stream (its finished files, in order), not at random
  places, so the number of passes over each source is exactly what the shares say. The source of every window slot
  comes from a low-discrepancy sequence, so the shares hold within a window or two at any point of the run.
- **Rotary base saved with the model:** the final `.pt` and every save record `rope_base` (1,000,000 after the long
  phase); without it, a loaded model would be rebuilt with 10,000.
- **Decontamination:** a public test item counts as copied when at least 4 of its 13-word runs appear (or all, if it
  has fewer). One shared run was too strict: "in the name of the father and of the son and of the", "1 2 3 ... 13" and
  textbook formulas removed 4-6% of books and maths sets. Long documents (books, papers, PDFs) lose only the
  paragraphs with the test text, as data_mix.md says. The project's own test items are removed wherever they appear
  only if they have 6+ words; shorter ones ("how are you?") only through the close-match rule in question-shaped
  sources.
- **Download shares measured on samples:** each source's share of files to download was set from the samples so it
  is read about 0.6-0.9 times (DCLM-Edu keeps ~1 row in 4 at edu score >= 3: 3.5% of its files; Cosmopedia's topic
  filter keeps ~41%: 50%; FinePDFs-Edu keeps ~6% as computer science: 10%; FineMath-4+ 60%; FineMath-3+ 15%;
  Essential-Web 5%). FinePDFs-Edu also gets a fastText English check (one "English" PDF was Chinese).
- **Supply estimates (footer row counts x the samples):** everything is read at most ~1.7 times except Simple Wikipedia
  (~4.3 times: its stubs under 50 words are dropped, so it is smaller than data_mix.md assumed). WildChat keeps ~59% of
  its chats (English, not toxic, the behaviour filters), about 0.7B pieces for the 0.53B needed.
- **Stack Exchange:** the anneal's best slice is the same thread text with a second output (threads with an accepted
  or 4+ answer), because two near-identical documents removed each other as copies. Code blocks are left out of the
  language check (Stack Overflow threads were judged "not English").
- **Hugging Face downloads:** `hf_transfer` is no longer used by huggingface_hub; its Xet downloader in
  high-performance mode replaces it.
- **Not on Hugging Face:** the Python docs come from docs.python.org (3.14, plain text); MDN, tldr-pages and the Rust
  book from GitHub archives (blocked from the cloud sandbox, expected to work from the server; skipped with a warning
  if not). OpenStax Intro CS and Open Data Structures are not fetched (PDF/HTML only; tiny). The pico's own held-out
  documents (data/slinger on the owner's laptop) are not removed: they are not in the repository.

### Nebius findings (18 November)

- Quotas: `compute.instance.gpu.b200` in us-central1 is 0 (the owner's request for 8 is waiting). There is no
  separate quota for spot servers, so the request covers them. `compute.instance.gpu.h200` is already 8 in
  us-central1 (also 8 in eu-north1 and eu-west1).
- Spot prices at 05:39 UTC (`nebius_ctl.py prices`): B200 $3.00 and H200 $2.20 per GPU-hour in us-central1, so
  $24.00 and $17.60 an hour for 8.
- 8x H200 instead of 8x B200: the H200 does about 2.3x less 8-bit maths per second, so training takes about 11-17
  hours instead of 5-7.5. With set-up, the test run and the chat fine-tune, the whole rental is about 14-22 hours and
  $250-400 on H200, against about 7-11 hours and $170-265 on B200: cheaper per hour, dearer in all. Memory (141 GB a
  GPU) is still ample.

### The owner's answers (section 3)

- 18 November, decision 1 (commercial use): "It will be free to use model, no paid users." So no commercial use:
  the full data list stays (DCLM-Edu, and the model-written chat and maths sets, are kept).
- 18 November, decision 1 (Stack Exchange): use the older copy (HuggingFaceH4/stack-exchange-preferences, from before
  the July 2024 request).
- 18 November, decision 1 (Nemotron-CC Diverse QA): use it.
- 18 November, decision 1 (the data list as a whole): approved, with Simple Wikipedia trimmed from 0.43% to 0.40% of
  the run (the difference to FineWiki), so it is read at most about 4 times.
- 18 November, decision 2 (chat layout): plain "System:/User:/Assistant:" lines, as the pico and the chat page; 16
  special pieces reserved in the word-piece list (prepare_492m.SPECIAL) for later use.
- 18 November, the chat behaviour filters: the owner wants some changed (which ones: asked).

### The rental day, command by command

From the cloud sandbox (each costing step only after the owner's yes, with the price):
1. `python research/nebius_ctl.py prices` and `check` (free).
2. `python research/nebius_ctl.py price --gpu b200 --max <owner's limit> --yes` (the spot price limit; free itself).
3. `python research/nebius_ctl.py create-disk --yes` (1 TB network SSD).
4. `python research/nebius_ctl.py create-server --gpu b200 --yes`, then `status` until it is running, then `ssh`.
5. Copy the code: `git clone -b ccr-5dd2ad80-fvffsq https://github.com/Insomniac-Coder/SlingerGPT /data/code` on the
   server (the repository is public), or `scp -r` from the sandbox.

On the server (`ssh` from step 4), in `/data/code`:
6. `python3 -m venv /data/venv && . /data/venv/bin/activate && pip install torch --index-url
   https://download.pytorch.org/whl/cu128 && pip install torchao tokenizers numpy pyarrow huggingface_hub hf_xet
   fasttext zstandard` (CUDA 12.8 or newer for Blackwell; the same works on H200).
7. `python research/test_train_492m.py gpu` (one minute: 8-bit build, flex mask = written-out mask, fused loss,
   attention speeds).
8. In tmux: `python research/prepare_492m.py run --data /data/492m --workers 140 --approved` (110 workers on the
   H200's 128 cores). Round 1 is about 5% of every source. Watch `prepare_492m.py status --data /data/492m`.
9. Once round 1 is done, the test run: `python research/train_492m.py train --gpus 8 --data /data/492m --name speed
   --stop-after 300`, trying `--attention cudnn|flash` and `--micro-pieces 32768|65536|131072`; then `kill -TERM`
   one run and read its save time in the log; then `python research/train_492m.py ratecheck --gpus 8 --data
   /data/492m`. Stop and tell the owner if the measured speed means more than 9 hours (B200).
10. The run, in tmux: `python research/train_492m.py train --gpus 8 --data /data/492m --lr <chosen>`. Every 30
    minutes: `python research/train_492m.py status --data /data/492m`, `prepare_492m.py status`, and a short plain
    update to the owner (progress, speed, held-out score, money so far, text ready ahead).

