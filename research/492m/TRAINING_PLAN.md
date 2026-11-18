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

Status on 18 November 2026: the design, sizes and schedule are decided. The data list and the fine-tune behaviour
rules still need the owner's yes. No code for the 492M exists yet, nothing is rented, and nothing is downloaded.

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
   - create the data disk: a network SSD of about 2 TB that survives the server being taken back, about $4-5 a day;
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
   huggingface_hub with hf_transfer, and datatrove or similar for deduplication. Check `nvidia-smi` shows 8 B200s
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
   - Ask the owner whether to delete the data disk (about $4-5 a day to keep).
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
