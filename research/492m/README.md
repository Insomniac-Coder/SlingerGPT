# SlingerGPT 492M: work in progress (not ready)

Plans for the next, larger SlingerGPT. Nothing here has been trained yet, and the plans are not final.

- **Size:** 492,406,016 weights: 22 layers x 1,280 wide (the `500m` shape in `research/arch.py`), 16,384 word pieces, an Engram lookup memory 4x the size of SlingerGPT 1 pico's.
- **Text:** about 80 billion pieces: 70 billion at a window of 4,096, 8 billion of the best material, then 2 billion of long documents at a window of 16,384.
- **Hardware:** a rented 8x B200 server; estimated 7.5-10 hours of training.

| File | What it is |
|---|---|
| `data_mix.md` | The 80-billion-piece training mix: 31 sources, shares, repeats, licences, cleaning, the three phases |
| `finetune_plan.md` | The chat fine-tune plan, built on the lessons from SlingerGPT 1 pico, with draft behaviour rules |
| `research_and_plans.json` | The raw research behind both: every candidate dataset with its size and licence |
