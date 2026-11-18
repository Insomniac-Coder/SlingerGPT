# 492M SlingerGPT: the 80-billion-piece training mix (NOT READY: awaiting the owner's approval)

Written by a research workflow on 18 November 2026 (5 researchers, 1 writer). Nothing has been downloaded.

> **Changed 18 November (owner):** the main phase now starts with short windows: pieces 0-25B at 1,024, 25-50B at 2,048, 50-78B at 4,096 (anneal included), 78-80B at 16,384. Sources and shares are unchanged.

## In plain words

1. The 80 billion pieces come from about 30 public datasets. Roughly 40% is good educational web pages, 9% textbook-style writing, 12% code (Python first) and 6% explanations of computers (Stack Overflow-style answers, manuals, open textbooks, and web pages picked out by their topic labels). The rest is about 9.5% maths, 7% Wikipedia, 5% books, 5% questions and answers, 3% science papers and 3% chats.
2. There is far more good text than we need, so almost every source is read once or less. Only small, very good sources (Simple Wikipedia, manuals, small maths sets, some chats) are read 2 to 4 times, and nothing more than 4.
3. The run has three parts. The first 70 billion pieces are read in windows of 4,096. The next 8 billion use only the best parts (top-rated pages, the best code, worked maths, more chat), as the pico's last stretch did. The last 2 billion use windows of 16,384: mostly whole books, papers and code projects, mixed with some of the best short text.
4. Before training, the text is cleaned on a cheap separate computer. Copies and near-copies are removed, along with non-English text and junk. Any text containing the questions from your own tests or from the standard public tests is also removed, so the scores stay honest.
5. Compared with the draft, textbook-style writing drops from 15% to 9% and chats from 5% to 3%. Both are written by other AI models and can carry made-up facts, and published small models used less of them. The freed room goes to a second kind of web text that reads more like people explaining things.
6. The download is about 450 GB, and preparing it takes roughly 8 to 13 hours on a 32-core computer, so no GPU time is spent waiting.
7. A few things need your decision: the licence points, whether to use the optional NVIDIA question set, and how chats should be laid out (which special markers). The chats also teach behaviour, so their filters need your yes: no other AI's name and no canned refusals.

## By group

| Group | Share | Billions of pieces |
|---|---|---|
| web | 39.5% | 31.6 |
| code | 11.9% | 9.5 |
| maths | 9.5% | 7.6 |
| textbook | 9.3% | 7.4 |
| reference | 7.2% | 5.7 |
| computer science | 6.3% | 5.0 |
| books | 5.3% | 4.3 |
| Q&A | 4.9% | 3.9 |
| conversation | 3.3% | 2.6 |
| science | 2.8% | 2.2 |

## Every source

| Source | Group | Share | Pieces (B) | Passes | Licence | Download | Where |
|---|---|---|---|---|---|---|---|
| FineWeb-Edu-dedup (SmolLM-Corpus, July 2024): all pages (int_score >= 3) in the main phase; int_score >= 4 pages only in the anneal | web | 24.62% | 19.70 | 0.08 of the pool; each chosen page read once, >= 4 pages up to twice (main + anneal) | ODC-By 1.0 + Common Crawl terms of use | ~55 of 550 GB parquet (about 1 file in 10, chosen at random) | `HuggingFaceTB/smollm-corpus (config fineweb-edu-dedup)` |
| DCLM-Edu (DCLM re-scored with the FineWeb-Edu classifier; linked to the SmolLM2 paper, Feb 2025): edu_int_score >= 3 in the main phase, >= 4 in the anneal | web | 14.88% | 11.90 | well under 0.1 of the >= 3 pool; each chosen page once, >= 4 pages up to twice | CC-BY-4.0 (follows DCLM-Baseline; the DCLM card mentions research intent) | ~100-150 of 5,130 GB (depends on the >= 3 share; filter on edu_int_score shard by shard while downloading) | `HuggingFaceTB/dclm-edu` |
| Cosmopedia v2 (SmolLM-Corpus, July 2024): filtered toward college/textbook formats and computer, programming, internet, security, maths and science topics (keywords in the prompt); stories capped at ~10% of the row | textbook | 9.30% | 7.44 | 0.27 of the set (~0.5 of the filtered half); chosen documents once, top textbooks up to twice | ODC-By (written by Mixtral-8x7B-Instruct) | ~60 of 122 GB parquet | `HuggingFaceTB/smollm-corpus (config cosmopedia-v2)` |
| Nemotron-CC v1 synthetic Diverse QA from high-quality pages (Dec 2024) | Q&A | 2.41% | 1.93 | <0.01; each item once | Common Crawl terms of use (data); text written by Mistral-NeMo-12B | ~8 (estimate; file layout unconfirmed: the index page returned 503 on 18 Nov) | `not on HF: data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/ (paper arXiv 2412.02595)` |
| Stack-Edu Python, permissively licensed files only (Common Pile stackv2_edu_filtered, language = Python; Software Heritage graph 2023-09-06, added Nov 2024) | code | 4.18% | 3.35 | ~0.3; files with int_score >= 4 up to twice (main + anneal) | Per file: MIT, Apache-2.0, BSD and other Blue Oak-certified permissive licences (licence and repo in metadata; keep them for attribution) | ~12 of 83 GB (Python files; reused for the long-phase repositories) | `common-pile/stackv2_edu_filtered` |
| Stack-Edu other languages (Common Pile stackv2_edu_filtered). Within the row: JavaScript+TypeScript 22%, C+C++ 20%, Java 10%, C# 8%, SQL 8%, Shell 7%, Go 5%, Rust 5%, PHP/Ruby/Swift 5%, Markdown READMEs/docs 10% | code | 6.00% | 4.80 | <0.1 overall; Go and Rust ~0.2-0.3 of their small subsets; score >= 4 files up to twice | Per file permissive (as above) | ~18 of 83 GB | `common-pile/stackv2_edu_filtered` |
| HTML files from The Stack v2 (Common Pile stackv2_html_filtered) | code | 0.26% | 0.21 | ~0.15 | Per file, detected_licenses in metadata (MIT, Apache, BSD...); no dataset card | 2.1 | `common-pile/stackv2_html_filtered` |
| Kaggle notebooks rendered as Markdown with code and outputs (2024) | code | 0.78% | 0.62 | ~0.4 | Apache 2.0 (Meta Kaggle Code); no overall licence on the card | 2.2 | `HuggingFaceTB/issues-kaggle-notebooks (config kaggle)` |
| OpenCoder annealing corpus: synthetic_qa and synthetic_code_snippet (Nov 2024), anneal only | code | 0.38% | 0.31 | ~0.1 | ODC-By; generating model not stated | ~3.7 | `OpenCoder-LLM/opc-annealing-corpus` |
| Code repositories for the 16,384 phase: Stack-Edu (Common Pile) files of one repository joined with file-name headers, Python first, 8k-16k+ pieces each | code | 0.30% | 0.24 | <0.5 | Per file permissive (as above) | 0 extra (same files as the two Stack-Edu rows) | `common-pile/stackv2_edu_filtered (grouped by metadata.repo_name)` |
| Stack Exchange tech sites (stackoverflow, superuser, askubuntu, unix, serverfault, cs, softwareengineering, security, networkengineering, dba, codereview, datascience): the question, the accepted answer and up to 2 more with pm_score >= 2, HTML converted to text | computer science | 3.06% | 2.45 | ~0.5-0.7 | CC BY-SA 4.0 (credit with links, share-alike). Dump date unconfirmed (cited 2023), i.e. probably before the July 2024 download agreement. | 19.7 (whole set; shared with the Math SE and non-tech rows) | `HuggingFaceH4/stack-exchange-preferences` |
| Essential-Web CS pages: EAI-Taxonomy Code w/ DCLM 100B-token sample (June 2025), kept where the FDC code is 004/005 (computer science, software engineering), education level is undergraduate or above, and the document type is reference, educational, tutorial or Q&A | computer science | 1.97% | 1.58 | ~0.1-0.15 | ODC-By + Common Crawl terms of use | ~20 of 266 GB | `EssentialAI/eai-taxonomy-code-w-dclm-100b-sample` |
| Documentation and open CS textbooks: Python docs (tutorial, HOWTOs, library reference, one version), MDN guides, ArchWiki (English), tldr-pages, PEPs, OpenStax Introduction to Computer Science, LibreTexts CS shelves and PressBooks CS books (Common Pile, openly licensed only), Open Data Structures (author's CC BY edition), The Rust Programming Language book | computer science | 0.48% | 0.39 | ~3-3.5 (hard cap 4: if the set is smaller, the excess share goes to Cosmopedia computer topics) | Mixed open: PSF (Python docs); CC BY-SA 2.5 prose, CC0/MIT code (MDN); GFDL 1.3 / CC BY-SA 3.0 (ArchWiki); CC BY 4.0 (tldr, OpenStax Intro CS); CC BY / BY-SA per page (Common Pile); public domain (PEPs); CC BY (ODS author edition); MIT/Apache believed (Rust book, unconfirmed) | ~2 | `common-pile/libretexts_filtered; common-pile/pressbooks_filtered; common-pile/python_enhancement_proposals_filtered; hannah-eee/arch-wiki-docs; plus docs.python.org, github.com/mdn/content, github.com/tldr-pages/tldr, openstax.org, opendatastructures.org, github.com/rust-lang/book` |
| FinePDFs-Edu English (Sept 2025): CS textbooks, lecture notes and tutorials (keyword/topic selected) in the main phase and anneal; long documents (CS and science first) in the 16,384 phase | computer science | 0.76% | 0.61 | ~0.1 | ODC-By (the PDFs themselves are copyrighted web documents, as with web pages) | ~25 of 299 GB | `HuggingFaceFW/finepdfs-edu (config eng_Latn)` |
| FineMath-4+ (Dec 2024) | maths | 3.39% | 2.72 | 0.27 | ODC-By 1.0 + Common Crawl terms of use | 18.4 (all) | `HuggingFaceTB/finemath (config finemath-4plus)` |
| FineMath-3+ documents not in 4+ (Dec 2024) | maths | 2.62% | 2.10 | 0.08 | ODC-By 1.0 + Common Crawl terms of use | ~6 of 65 GB | `HuggingFaceTB/finemath (config finemath-3plus)` |
| InfiWebMath-3+ (Dec 2024) | maths | 1.31% | 1.05 | 0.05 | ODC-By 1.0 + Common Crawl terms of use | ~3 of 46.8 GB | `HuggingFaceTB/finemath (config infiwebmath-3plus)` |
| Mathematics Stack Exchange (MathOverflow left out as too advanced): the question, the accepted answer and up to 2 more with pm_score >= 2 | maths | 1.01% | 0.81 | ~0.7-1 | CC BY-SA 4.0 | 0 extra (in the tech-site row's 19.7 GB) | `HuggingFaceH4/stack-exchange-preferences` |
| Worked maths problems: Orca-Math 200k (Feb 2024), MetaMathQA (Sep 2023), OpenMathInstruct-2 train_1M (Oct 2024, over-long items dropped), NuminaMath-CoT cn_k12 + synthetic_math only, a TinyGSM sample (Dec 2023), GSM8K train | maths | 1.18% | 0.95 | Orca-Math and GSM8K train ~4; MetaMathQA ~2; OMI-2 and Numina ~1; TinyGSM ~0.2 | MIT (Orca-Math, MetaMathQA, TinyGSM, GSM8K); CC-BY-4.0 (OMI-2; Llama 3.1 output terms may apply); Apache-2.0 (NuminaMath) | ~6 | `microsoft/orca-math-word-problems-200k; meta-math/MetaMathQA; nvidia/OpenMathInstruct-2; AI-MO/NuminaMath-CoT; TinyGSM/TinyGSM; openai/gsm8k` |
| FineWiki English (Wikimedia Enterprise HTML dumps, Aug 2025) | reference | 6.74% | 5.39 | ~1.0 | CC BY-SA 4.0 + GFDL | 37.7 (or read only the text column remotely) | `HuggingFaceFW/finewiki (config en)` |
| FineWiki Simple English (Aug 2025) | reference | 0.43% | 0.34 | ~3.4 | CC BY-SA 4.0 + GFDL | 0.52 | `HuggingFaceFW/finewiki (config simple)` |
| Project Gutenberg English books, filtered (Common Pile v0.1, 2025), with PG-19's 150 validation/test books removed | books | 2.58% | 2.07 | 0.41 | Public domain per PG metadata (the card warns labels may be wrong) | 7.6 | `common-pile/project_gutenberg_filtered` |
| DOAB open-access academic books (Common Pile v0.1, 2025) | books | 2.75% | 2.20 | 0.73 | CC BY / CC BY-SA per book | 3.4 | `common-pile/doab_filtered` |
| Stack Exchange non-tech sites (physics, chemistry, biology, astronomy, earthscience, history, english, ell, linguistics, philosophy, economics, cooking, diy, travel, outdoors, skeptics, law, health, gardening, bicycles, money, writing...): same answer filter as the tech sites; meta sites dropped | Q&A | 2.52% | 2.01 | ~1-1.3 (cap 2; any excess share moves to DCLM-Edu) | CC BY-SA 4.0 | 0 extra | `HuggingFaceH4/stack-exchange-preferences` |
| arXiv abstracts (Common Pile v0.1; papers to Aug 2024) | science | 0.99% | 0.79 | ~1.3 | CC0 (arXiv metadata) | ~1 (estimate) | `common-pile/arxiv_abstracts_filtered` |
| peS2o openly licensed papers (Common Pile v0.1) | science | 1.55% | 1.24 | 0.03 | Openly licensed articles only (CC BY family, per document) | ~3 of 56.9 GB | `common-pile/peS2o_filtered` |
| arXiv full papers, openly licensed (Common Pile v0.1), long phase only | science | 0.25% | 0.20 | 0.04 | CC BY, CC BY-SA, CC0 per paper | ~3 (estimate) | `common-pile/arxiv_papers_filtered` |
| SmolTalk 'all' train split (2024-25), minus apigen function calls, numina-cot and longalign; test split held out | conversation | 1.60% | 1.28 | ~1.6 | Apache 2.0 for the new parts; included public parts keep their own licences; replies written by Llama-3.1-405B and other models | 2.1 | `HuggingFaceTB/smoltalk (config all)` |
| UltraChat 200k (train_sft) | conversation | 0.52% | 0.41 | ~1.25 | MIT (ChatGPT-written replies) | 0.73 | `HuggingFaceH4/ultrachat_200k` |
| SODA (train) | conversation | 0.52% | 0.41 | ~2.4 | CC-BY-4.0 | 0.88 (train 0.71) | `allenai/soda` |
| WildChat-1M, non-toxic (ODC-BY since 26 June 2024): English conversations, GPT-4 ones first | conversation | 0.66% | 0.53 | ~1 | ODC-BY; replies by GPT-3.5/GPT-4 (OpenAI terms apply to the outputs) | 3.4 | `allenai/WildChat-1M` |

## Why each source

- **FineWeb-Edu-dedup (SmolLM-Corpus, July 2024): all pages (int_score >= 3) in the main phase; int_score >= 4 pages only in the anneal**: Base of the mix: the best measured knowledge web text (MMLU/ARC). It is already deduplicated across dumps and carries score, int_score, url and token_count for every page (checked), so the best pages can be weighted up. The draft's 35% FineWeb-Edu became 24.6% here plus 14.9% DCLM-Edu, following SmolLM2's 60/40 blend, which kept most of both strengths (MMLU 37.0, HellaSwag 62.2). Unique supply: ~250 (220B GPT-2 tokens x ~1.15 for our 16k list, estimate; 190,168,005 pages, 942 GB in memory). Size of the >= 4 slice unconfirmed (sum metadata.token_count where int_score >= 4)..
- **DCLM-Edu (DCLM re-scored with the FineWeb-Edu classifier; linked to the SmolLM2 paper, Feb 2025): edu_int_score >= 3 in the main phase, >= 4 in the anneal**: The second web style. DCLM's classifier favours explanations and Q&A, which beat FineWeb-Edu on HellaSwag, CSQA and PIQA in SmolLM2's ablations, and it carries the casual phrasing the pico lacked. The card says keeping edu_int_score >= 3 works better for small models, and SmolLM2-135M/360M used this data. Pages whose URL is already in the FineWeb-Edu sample are dropped. Unique supply: unconfirmed: the card gives ~1.0B rows and 5.13 TB of parquet but no token count; the share of rows at edu_int_score >= 3 is not published (estimated hundreds of billions of pieces).
- **Cosmopedia v2 (SmolLM-Corpus, July 2024): filtered toward college/textbook formats and computer, programming, internet, security, maths and science topics (keywords in the prompt); stories capped at ~10% of the row**: Plain-word textbook explanations, including computer basics. Cut from 15% to 9.3% because the whole set has one generator's style and may contain made-up facts (the owner's main concern), and SmolLM2-1.7B used only 4% in its decay. It has no topic column, so CS weighting relies on prompt keywords. Unique supply: ~28 (28B tokens per the SmolLM blog; token_length uses the Mistral-7B tokenizer, about 1.0-1.05x our pieces, estimate); 39,134,000 documents; about half kept after the topic/format filter.
- **Nemotron-CC v1 synthetic Diverse QA from high-quality pages (Dec 2024)**: Facts restated as varied questions and answers taken from each page's own text. That is grounded, so it should invent less than free generation. It serves the 'facts in varied wordings' evidence: rephrasings in place of 4 of 8 repetitions raised the average from 55.8 to 56.7 at 8B. Optional and small: if the files cannot be fetched or checked, this share goes to FineWeb-Edu >= 4. Unique supply: unconfirmed (the synthetic part made from high-quality pages is ~1.5T tokens across 5 prompt types; Diverse QA alone is not given; tokenizer not stated).
- **Stack-Edu Python, permissively licensed files only (Common Pile stackv2_edu_filtered, language = Python; Software Heritage graph 2023-09-06, added Nov 2024)**: Python first, as asked. Stack-Edu's educational filter raised MultiPL-E Python from 20.7 to 25.6 at 1.7B. This Common Pile copy includes the code text (checked: text field, plus score/int_score and 30 metadata fields), so there is no Software Heritage S3 download or bulk agreement, and it drops Stack-Edu's 'no_license' files. Unique supply: estimate ~10-15 (Stack-Edu Python is 21.8B tokens, tokenizer probably SmolLM2's, before the permissive-only cut; code takes ~1.25x more of our pieces, estimate).
- **Stack-Edu other languages (Common Pile stackv2_edu_filtered). Within the row: JavaScript+TypeScript 22%, C+C++ 20%, Java 10%, C# 8%, SQL 8%, Shell 7%, Go 5%, Rust 5%, PHP/Ruby/Swift 5%, Markdown READMEs/docs 10%**: The main languages beyond Python, for an all-rounder. Java is downsampled because it is the largest and least needed. Markdown READMEs are prose about code, which builds computer knowledge. Together with Python this holds code at the draft's ~12%: SmolLM2 and SmolLM3 used 10-15% in their stable phases. Unique supply: estimate ~50-65 for the non-Python languages (the whole set is 255 GB UTF-8, 69.6M files, ~77B pieces at ~3.3 letters per piece including Python); per-language sizes unconfirmed (sum metadata.length_bytes).
- **HTML files from The Stack v2 (Common Pile stackv2_html_filtered)**: Stack-Edu has no HTML or CSS, and web pages are basic computer knowledge. Keep pages with real structure and text; drop empty landing pages. Unique supply: estimate ~1.5 (1,674,654 files, 2.07 GB of files; UTF-8 size not stated; whether CSS is included is unconfirmed).
- **Kaggle notebooks rendered as Markdown with code and outputs (2024)**: Python with explanations and printed results, i.e. code with reasoning, mostly data science. SmolLM3 used it. The GitHub issues part is skipped as too noisy. Unique supply: estimate ~1.5 (580,195 notebooks, 5.2 GB in memory; the researchers report ~1.7B tokens, tokenizer not stated).
- **OpenCoder annealing corpus: synthetic_qa and synthetic_code_snippet (Nov 2024), anneal only**: Code questions answered with explanations, built for annealing. It is the closest open set to 'code textbook' writing. Kept small and anneal-only because it is model-written. Unique supply: estimate ~2.5-3.5 (3.24M + 3.08M rows counted so far, ~9.7 GB in memory; the size API result is partial).
- **Code repositories for the 16,384 phase: Stack-Edu (Common Pile) files of one repository joined with file-name headers, Python first, 8k-16k+ pieces each**: ProLong found code repositories and books to be the best long data. Single files over 16k pieces are rare and often generated, so files are joined per repository. Unique supply: unconfirmed (depends on how many repositories reach 8k pieces; probably several billion).
- **Stack Exchange tech sites (stackoverflow, superuser, askubuntu, unix, serverfault, cs, softwareengineering, security, networkengineering, dba, codereview, datascience): the question, the accepted answer and up to 2 more with pm_score >= 2, HTML converted to text**: Tech Q&A is the core of computer basics and programming knowledge, with the same fact in several answers. This copy scores every answer (pm_score), and the site can be read from the metadata URL (both checked), so no join is needed. It holds only questions with 2+ answers; if supply is short, top up from common-pile/stackexchange_filtered (Dec 2024 dumps). Unique supply: estimate ~3.5-5 after HTML stripping and filters (whole set: 10,807,695 questions, 57.6 GB in memory as HTML, all sites); tech share unconfirmed.
- **Essential-Web CS pages: EAI-Taxonomy Code w/ DCLM 100B-token sample (June 2025), kept where the FDC code is 004/005 (computer science, software engineering), education level is undergraduate or above, and the document type is reference, educational, tutorial or Q&A**: This is the 'CS pages weighted up by topic labels' part. FineWeb-Edu and DCLM-Edu carry no topic labels, but Essential-Web labels every page (FDC, document type, education level, reasoning depth). Its card shows MMLU-CS at 47.0 against 32.0 for DCLM-baseline (2.3B models). Pages whose URL is in the other web rows are dropped. Unique supply: estimate ~10-20 after filters (the sample is 100B tokens, 46,192,864 rows, 266 GB parquet; tokenizer not stated; the card says it also holds maths pages, and the CS share is unconfirmed).
- **Documentation and open CS textbooks: Python docs (tutorial, HOWTOs, library reference, one version), MDN guides, ArchWiki (English), tldr-pages, PEPs, OpenStax Introduction to Computer Science, LibreTexts CS shelves and PressBooks CS books (Common Pile, openly licensed only), Open Data Structures (author's CC BY edition), The Rust Programming Language book**: Small, authoritative explanations of how computers, the web, Linux and programming languages work: the clean source for 'basic computer knowledge'. Non-commercial books (MIT OCW, Think Python, OpenStax Python) are left out. Unique supply: estimate ~0.10-0.13 together (no published sizes; measure).
- **FinePDFs-Edu English (Sept 2025): CS textbooks, lecture notes and tutorials (keyword/topic selected) in the main phase and anneal; long documents (CS and science first) in the 16,384 phase**: Real course material at scale, and long by nature, so it fits the 16,384 phase. It needs PDF clean-up (page numbers, broken lines) and a sample check. Unique supply: English ~90.8B tokens (third-party count, tokenizer unconfirmed); 23,023,372 documents, 299 GB parquet; CS share unconfirmed (estimate 5-10).
- **FineMath-4+ (Dec 2024)**: The best open maths web text: about 2x GSM8K and 6x MATH against InfiMM-WebMath at 1.7B, with no plateau when repeated. Its makers decontaminated it against GSM8K, MATH and MMLU. Weighted up to 7% in the anneal. Unique supply: ~10 (9.6B tokens, tokenizer not named, probably SmolLM2's; 6,699,493 documents, 38.5 GB in memory).
- **FineMath-3+ documents not in 4+ (Dec 2024)**: Bulk maths web text for the main phase. Not deduplicated against InfiWebMath, because HF found that hurt. Unique supply: ~26 (34B tokens minus the 9.6B 4+ part, estimate; 21,405,610 documents in all).
- **InfiWebMath-3+ (Dec 2024)**: A second maths extraction for variety; a 50/50 FineMath/InfiWebMath mix matched FineMath-3+ alone. MegaMath and OpenWebMath are skipped because FineMath covers them at equal or better quality. Unique supply: ~22 (20.5B tokens; 13,882,669 documents).
- **Mathematics Stack Exchange (MathOverflow left out as too advanced): the question, the accepted answer and up to 2 more with pm_score >= 2**: Worked explanations in question-and-answer form, LaTeX included. Unique supply: estimate ~0.8-1.2 (its share of the H4 set is unconfirmed).
- **Worked maths problems: Orca-Math 200k (Feb 2024), MetaMathQA (Sep 2023), OpenMathInstruct-2 train_1M (Oct 2024, over-long items dropped), NuminaMath-CoT cn_k12 + synthetic_math only, a TinyGSM sample (Dec 2023), GSM8K train**: Step-by-step solutions, mostly in the anneal (6%), as SmolLM3 and OLMo 2 did. MetaMathQA's rephrasings match the 'varied wordings' rule. Orca-Math and GSM8K train were in the pico and its fine-tune replay. Olympiad and AMC problems are left out at this size. Unique supply: estimate ~1.5 together (Orca ~0.025, MetaMathQA ~0.09, OMI-2 1M ~0.35, Numina school part ~0.15, TinyGSM ~0.9; row counts and MB confirmed, pieces estimated).
- **FineWiki English (Wikimedia Enterprise HTML dumps, Aug 2025)**: Facts are the weakest area. FineWiki keeps tables and maths, which the wikimedia/wikipedia copy drops, and it includes computing articles. Infobox facts may be added as one plain 'Key facts' line for a second wording, tried on 500 pages first. Unique supply: estimate ~5-6 (6,614,655 pages; 78 GB in memory includes the raw wikitext column; text-only size and tokens are not stated on the card).
- **FineWiki Simple English (Aug 2025)**: The same facts in plainer words (varied wordings). It is small, so it is read 3-4 times and raised in the anneal, as in the pico. Unique supply: estimate ~0.1 (265,299 pages; 521 MB parquet including wikitext).
- **Project Gutenberg English books, filtered (Common Pile v0.1, 2025), with PG-19's 150 validation/test books removed**: Stories and long-form English (the owner wants stories), and the main long-document supply for the 16,384 phase (0.3B there). The English is old-fashioned, so the main-phase share stays modest. Unique supply: ~5.0 (20.1 GB of text; 5.7B Comma v0.1 tokens; 55,454 books).
- **DOAB open-access academic books (Common Pile v0.1, 2025)**: Modern books across science and humanities, a modern-English counterweight to Gutenberg. Comma kept it at 2x in its high-quality cooldown. Non-English books are removed by language ID. Unique supply: ~3.0 (12 GB of text; 3.0B Comma tokens; 403,992 documents).
- **Stack Exchange non-tech sites (physics, chemistry, biology, astronomy, earthscience, history, english, ell, linguistics, philosophy, economics, cooking, diy, travel, outdoors, skeptics, law, health, gardening, bicycles, money, writing...): same answer filter as the tech sites; meta sites dropped**: Everyday and science questions answered by people at full length, several answers per question (varied wordings). Cut from the draft's 3% to 2.5% because supply after vote filtering is unconfirmed. Unique supply: unconfirmed, estimate ~1.5-2.5 after filters.
- **arXiv abstracts (Common Pile v0.1; papers to Aug 2024)**: Dense science summaries over many topics. cs.* abstracts are counted here only, not again under CS. Unique supply: ~0.6 (2,538,935 abstracts, 2.4 GB; 0.57B Comma tokens).
- **peS2o openly licensed papers (Common Pile v0.1)**: Real science writing with worked derivations; long papers also feed the 16,384 phase (0.1B). Kept small because Comma's paper-heavy mix was weak on everyday common sense (HellaSwag/PIQA). Unique supply: ~45 (182.6 GB of text; 6,117,280 openly licensed papers).
- **arXiv full papers, openly licensed (Common Pile v0.1), long phase only**: At about 20k Comma tokens per paper, the best long science and maths source. cs, math and physics papers first. Unique supply: ~4.8 (19 GB of text; 304,048 papers; 6.0B Comma tokens).
- **SmolTalk 'all' train split (2024-25), minus apigen function calls, numina-cot and longalign; test split held out**: The best-measured assistant chats (SmolLM2's SFT ablations), including multi-turn magpie-ultra; the pico's smol-smoltalk is a subset. Rewritten into the fine-tune's exact layout, with other assistants' names removed. Unique supply: estimate ~0.8 (1,043,917 conversations, 4.06 GB in memory).
- **UltraChat 200k (train_sft)**: Long multi-turn chats. The pico showed that the model copies the length and topic of earlier replies. Unique supply: estimate ~0.33 (207,865 dialogues, 1.38 GB in memory).
- **SODA (train)**: Everyday human-style chit-chat. It was in the pico's pretraining and its fine-tune replay, so the replay stays in-distribution. Unique supply: estimate ~0.17 (1,191,582 dialogues; 1.35 GB in memory including narratives and other fields).
- **WildChat-1M, non-toxic (ODC-BY since 26 June 2024): English conversations, GPT-4 ones first**: Real people's typing (lower case, no question marks, short asks), the kind of phrasing the pico got only 15-22% right. Tulu 3 found that real prompts matter (AlpacaEval 12.4 -> 7.5 without them). Unique supply: estimate ~0.5 (837,989 conversations in 68 languages, 6.66 GB in memory; English share unconfirmed).

## The last stretch (anneal)

Pieces 70-78B (8B, 10%), window 4,096, while the learning rate falls (the decay continues through the long phase to 80B). Same sources, best slices only. This mirrors the pico's last stretch, which dropped its lower-rated FineWeb and raised the best material.

Shares of the 8B:
- **Web 25%:** FineWeb-Edu int_score >= 4 17%, DCLM-Edu edu_int_score >= 4 8%.
- **Cosmopedia 5%:** college/textbook formats, CS/maths/science topics.
- **Nemotron Diverse QA 2%.**
- **Code 18%:** Stack-Edu Python int_score >= 4 7%, other languages >= 4 6%, Kaggle notebooks 1.5%, OpenCoder annealing Q&A and snippets 3.5%.
- **CS knowledge 8%:** Stack Exchange tech answers with pm_score >= 4 or accepted 4%, Essential-Web CS pages at undergraduate level or above 2%, docs and open textbooks 1%, FinePDFs-Edu CS 1%.
- **Maths 15%:** FineMath-4+ 7%, Math SE top answers 2%, worked problems 6% (Orca-Math, MetaMathQA, OpenMathInstruct-2 1M, NuminaMath school level, TinyGSM, GSM8K train).
- **Wikipedia 9%:** FineWiki 7.5%, Simple 1.5%.
- **Books 3%:** DOAB 2%, Gutenberg 1%.
- **Non-tech Stack Exchange top answers 3%.**
- **Science 2%:** arXiv abstracts 1%, peS2o 1%.
- **Conversation 10%:** SmolTalk 5%, WildChat English 2%, UltraChat 1.5%, SODA 1.5%, in the fine-tune's exact layout.

Why these changes:
- Code rises from 11% to 18% and maths from 9% to 15%, as in the SmolLM2/SmolLM3 decays (code 24%, maths 13-14%). Both kept their best maths and code for the decay.
- Chat rises from 2.5% to 10%. This follows MiniCPM, where SFT-style data in the decay gave MMLU 44.6 -> 50.9 and GSM8K 27.7 -> 42.3 at 2.4B. It stays below the pico's ~25% because published small models used 0-2%.

The same mix is also:
- the pool for the fine-tune's 8% plain-text replay (text the model already knows);
- the 40% short part of the long phase.

Save checkpoints at 70B (before the anneal) and at 74B so the owner can compare.

## The long-document phase (window 16,384)

Pieces 78-80B (2B, 2.5%), window 16,384, with the learning rate still falling.

Settings:
- Raise the rotary base at the switch to about 1,000,000 (Qwen-style ABF; Men et al. 2024 give ~3.1e5 as the minimum for 16K).
- Mask attention at document boundaries.

Long documents, 60% (1.2B). Each is 8k-16k+ pieces, one per window where possible; books are cut at paragraph breaks.
- **Books 25%:** Gutenberg 15% (without PG-19's test and validation books), DOAB 10% (only documents long enough).
- **Papers 15%:** Common Pile arXiv full papers 10%, peS2o 5% (cs, math, physics first).
- **Code repositories 12%:** Stack-Edu files of one repo joined with file-name headers, Python first.
- **Long articles and CS textbooks 8%:** FinePDFs-Edu long documents 5%, long FineWiki articles 2%, open textbooks (OpenStax Intro CS, LibreTexts CS, full Python docs pages) 1%.

Short text, 40% (0.8B): the anneal mix in its own shares, packed with document masking.

Evidence:
- ProLong: about 60% long / 40% short was best; code repositories and books were the best long data; 100% long hurt both long and short skill; masking across documents helped.
- Fu et al. 2024: 0.5-5B tokens are enough, and the domain balance should be kept.
- SmolLM3: more long data than this gave no extra RULER/HELMET gain.

Supply is far above need (Gutenberg alone is ~5B pieces).

Checks:
- wrongness by position on the held-out PG-19 test books (not perplexity alone, which ProLong found misleading);
- a needle-in-a-haystack test at 4k, 8k and 16k;
- the normal short tests (held-out text, 117-question quiz, sums, chat_suite), which must not drop.

## Cleaning and decontamination

All on the separate CPU machine, in this order. Each step runs first on ~500 documents per source with samples read (pico lesson), then on everything.

**1. Format**
- One plain-text document per item: UTF-8, NFC, newlines kept.
- Stack Exchange HTML becomes text, with code blocks fenced and lists kept.
- Chats are rewritten into the fine-tune's exact layout, with an end-of-turn mark after every assistant turn.

**2. Language**
- fastText lid.176 English >= 0.65 on prose sources not already English-filtered: Stack Exchange, DOAB, peS2o, arXiv, WildChat, LibreTexts/PressBooks, ArchWiki (lang=en).
- Markdown >= 0.5; code files are not language-checked.

**3. Quality and boilerplate**
- Prose: drop documents under 50 words. On web, PDFs and books, apply Gopher-style repetition rules (duplicate lines > 30%, top 2-gram > 20%) and drop lines repeated across many documents of a shard (menus, cookie notices, 'share this').
- PDFs: remove running headers and page numbers, join hyphenated line breaks, drop documents under 70% letters or with broken extraction.
- Gutenberg: drop transcriber notes, contents pages and index pages.
- Stack Exchange: keep the accepted answer plus at most 2 more with pm_score >= 2. Drop answers under 20 words and 'possible duplicate' / 'what have you tried' replies, and drop questions with no answer left.
- Code:
  - Drop generated and vendor files (metadata flags), files over 100 KB, minified files (average line > 100 characters or any line > 1,000), mostly-data files (letters < 25%) and base64/hex blobs.
  - Replace secrets, keys, emails and IP addresses with placeholders.
- Conversation (behaviour filters; the owner approves these before building, per the 'ask before behaviour choices' rule):
  - Drop non-English chats and WildChat's toxic-flagged ones.
  - Drop chats where the assistant calls itself another assistant or company (SmolLM, ChatGPT, GPT-4, OpenAI, Claude, Llama, Qwen, Gemma, 'As an AI language model').
  - Drop assistant turns that are only a canned refusal or only 'I don't know', drop knowledge-cutoff claims and drop function-calling items.
  - No passage-reading practice and no 'The text doesn't say.' anywhere in pretraining.

**4. Exact duplicates**
- SHA-256 of the lower-cased, whitespace-collapsed text across all sources. Keep the first copy, in this priority order: FineWiki, books, Stack Exchange, FineWeb-Edu, DCLM-Edu, Essential-Web, others.
- URL duplicates: drop DCLM-Edu and Essential-Web pages whose URL is in the FineWeb-Edu sample.

**5. Near duplicates**
- MinHash LSH (5-word shingles, 112 hashes in 14 bands of 8, about 0.8 similarity) over all real prose as one pool: web rows, Essential-Web, FinePDFs, Stack Exchange, FineWiki, books, papers. This also removes Wikipedia mirrors from the web sets.
- Not run between FineMath and InfiWebMath (HF found that hurt), and not over code (already deduplicated in The Stack v2) or the small synthetic sets.

**6. Held-out text**
- 1 document in 100 per source is set aside by a fixed hash of its id before training. It is never trained on, and the same list is used for every 560M run.
- The pico's own held-out documents (decoded from data/slinger/*/val.bin) are also removed, so per-letter wrongness can be compared fairly between the pico and the 560M.
- Kept out as well: PG-19 test/validation books, the SmolTalk and UltraChat test splits, GSM8K test.

**7. Decontamination against the project's own tests**
The lists are imported at build time, so later tests must be added before building:
- research/test_slinger.py: CHAT 12, LOOKUP_QUESTIONS 30, BROAD_QUESTIONS 30, IDENTITY_QUESTIONS 10, NONSENSE_QUESTIONS 10.
- research/chat_suite.py: FACTS 30 casual + 30 formal, SUMS 10, the SWITCH/WRONG/FOLLOW chats and their questions, LONG 8, GREET 6, WHO 4, INSTRUCT 4, IMPOSSIBLE 10, LATE 4.
- research/facts_quiz.py QUIZ.
- finetune_slinger.arithmetic_test(): 50 sums, seed 2026.

Rules:
- In every source, drop documents that contain a test question verbatim (after lower-casing and stripping punctuation) or one of chat_suite's prepared assistant replies.
- In question-shaped sources (chats, Stack Exchange, Nemotron QA, worked problems), also drop items whose question has >= 80% of a test question's content words and is at most 1.5x as long, and exact copies of the 50 sums.
- Deliberately not done as topic removal. The fine-tune's test_words title rule (drop any article whose title shares a test word) would remove Paris, Japan, photosynthesis and the other facts the quiz checks the model learned from ordinary text, so that rule stays in the fine-tune only.

**8. Decontamination against public tests**
- Tests covered: MMLU (all splits), HellaSwag (val+test), ARC Easy/Challenge, GSM8K and MATH test, PIQA, OpenBookQA, CommonsenseQA, WinoGrande, HumanEval (164) and MBPP (974).
- Prose: word 13-grams of question + choices + answer, lower-cased and without punctuation. A document sharing one is dropped; for books, papers and other long documents only the paragraph is dropped. Items under 13 words are matched as whole normalised strings.
- Code: drop files containing a HumanEval/MBPP prompt or docstring (whitespace-normalised) or a 13-token run of a canonical solution, and repositories named after the benchmarks.
- Output: a count of removals per source and per test. It is a report only; the owner decides on anything unexpected.

## The word-piece list

A 16,384-piece byte-level BPE, built with HF tokenizers like the pico's prepare_slinger.py (ByteLevel with the full 256-byte starting alphabet). It is trained on a ~3 GB sample (about 750M pieces) of the cleaned and decontaminated text, with each document capped at 20 KB so no single book dominates.

Sample shares:
- **Web 35%:** FineWeb-Edu 21, DCLM-Edu 14.
- **Cosmopedia 7%.**
- **Code 20%:** Python 8, JS/TS 3, C/C++ 2.5, Java+C# 2, SQL+Shell 1.5, Go/Rust/PHP/Ruby/Swift 1.5, HTML+Markdown 1.5.
- **CS prose 8%:** Stack Exchange tech, docs, Essential-Web CS.
- **Maths 10%:** FineMath with LaTeX, Math SE, worked problems.
- **Wikipedia 8%, books 4%, science 3%.**
- **Conversation 5%,** in the exact chat layout.

Code and maths sit above their training shares so that indentation runs, keywords (def, return, self.) and LaTeX commands get their own pieces.

Pre-tokenizer:
- pre_tokenizers.Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)]). Every digit is its own piece, so numbers are read digit by digit, which helps sums. The GPT-2-style split keeps runs of spaces and tabs as indentation pieces.
- No Unicode normalisation inside the tokenizer, so code stays exact (NFC is done in cleaning).

Special pieces: reserve 16.
- End-of-text as id 0, as in the pico.
- System, user and assistant turn markers and an end-of-turn marker.
- Search-call and search-result markers.
- Spares.
Whether the chat layout switches from the pico's 'User:/Assistant:' text lines to these markers is the owner's decision. It must be made before the chat data is written, because pretraining chats and the fine-tune must match.

Checks after training:
- Letters per piece per source on held-out text: about 4 on prose, about 3-3.5 on code.
- Exact round-trip decode on 10,000 documents.
- Per-source piece counts, which replace every estimate in the table before shares and passes are fixed.

Training takes roughly 20-40 minutes on the CPU machine (about 64 GB of RAM).

## Lessons from the 97M applied

- Keep test questions' words out of practice data (test_words). In pretraining this becomes exact and near-exact question matching in every source and in question-shaped sources, not topic removal. The 117-question quiz checks facts the model should learn from ordinary text, so the title rule stays in the fine-tune. The test lists are imported from test_slinger.py, chat_suite.py and facts_quiz.py at build time.
- Approving a dataset is not approving the behaviour it teaches. Chats in pretraining teach behaviour, so the conversation filters go to the owner for a yes before the data is built: other assistants' names removed, canned refusals and 'As an AI language model' removed, no 'I don't know' practice, no 'The text doesn't say.'.
- SQuAD 2.0's unanswerable third leaked 'The text doesn't say.' into 165 of 1,300 chats and taught narrow answers. So there is no SQuAD or other passage-reading practice in pretraining; reading practice stays in the fine-tune, answerable questions only.
- Never feed the model's own answers back as training text. Nothing written by SlingerGPT is in the mix. Text written by other models is capped at about 16-17% (Cosmopedia, Nemotron, worked problems, OpenCoder, chat replies), so real text stays above 80%.
- Fine-tuning made general text worse and an 8% plain-text replay fixed it. The pico's replay sources (Wikipedia, Simple Wikipedia, SODA, FineMath, Orca-Math) stay in the 560M mix, and the fine-tune's replay is drawn from the anneal mix, text the model already knows.
- Short, lower-case, no-question-mark questions were right at best 22% (casual facts 15%). The mix adds real casual phrasing (WildChat user turns, Stack Exchange question titles, DCLM-Edu's forum and Q&A style) and keeps its original casing.
- The redo was dry: half-length answers. The mix prefers full-length answers: Stack Exchange answers of 20+ words, no FLAN-style one-word task data, no copied SQuAD spans.
- The model copies the length and topic of earlier replies. Pretraining chats favour long multi-turn sets (smol-magpie-ultra, UltraChat, WildChat multi-turn); the mixed topic-switching chats stay in the fine-tune.
- Greetings never stopped because a turn's end was not marked. Every assistant turn in pretraining chats ends with an end-of-turn mark (a reserved piece in the 16k list, or the pico's 'User:' marking), in exactly the fine-tune's layout.
- Held-out samples change when the sources change, so compare by change from each run's own start. 1 in 100 documents per source is held out by a fixed hash, with the same list for every 560M run. The pico's held-out documents are also excluded, so per-letter wrongness compares fairly between the two models.
- Try each new data step on a few hundred examples first. Every format, filter and decontamination rule runs on ~500 documents per source with samples read before the full CPU run.
- Identity is tested ('Are you ChatGPT?', 'Who made you?'). Chats where the assistant names itself as another AI or company are removed, so pretraining does not teach a wrong name.
- Facts are the weakest area and need varied wordings. Wikipedia stays at ~7% with Simple Wikipedia read 3-4 times, plus Nemotron's Q&A restatements and multi-answer Stack Exchange threads.
- The word-piece list is built from a sample of the new mix with code included (decided 16 Nov). The tokenizer sample upweights code and maths, splits digits, and keeps indentation runs.
- Data preparation would leave the GPUs idle for 30-60 minutes, and only one GPU job runs at a time. So all preparation runs on a separate cheap CPU machine as written scripts, and only finished piece files go to the GPU server.

## Totals

**Pieces read: 80.0B in three phases** (rows are rounded to 3 decimals and add up to exactly 80.0B / 100%):
- main: 70B, window 4,096
- anneal: 8B, window 4,096
- long: 2B, window 16,384

**Share by group:**
- web: 39.5% (FineWeb-Edu 24.6 + DCLM-Edu 14.9)
- textbook: 9.3%
- code: 11.9%
- computer-science knowledge: 6.3%
- maths: 9.5%
- Wikipedia: 7.2%
- books: 5.3%
- Q&A: 4.9% (people's Stack Exchange answers 2.5 + Nemotron 2.4)
- science: 2.8%
- conversation: 3.3%

**Computing overall.** Code plus CS knowledge is 18.2%. Computer topics inside Cosmopedia, Wikipedia and the web add more, so roughly a fifth of what the model reads is about computers and programming.

**Why the draft's 12% code and 6% CS are kept:**
- Code: SmolLM2 and SmolLM3 used 10-15% in their stable phases, rising to 24% in the decay. Here it is 11% in the main phase and 18% in the anneal. More would take room from the everyday text an all-rounder needs: Comma's science- and code-heavy mix was weak on HellaSwag and PIQA.
- CS knowledge: Essential-Web's CS-labelled pages raised MMLU-CS from 32 to 47. Supply covers 6% at about one pass or less; going higher would mean repeating the small manuals past 4 passes.

**Repetition.** No source goes past 4 passes. Over 2: Simple Wikipedia ~3.4, docs and textbooks ~3-3.5, SODA ~2.4, Orca-Math and GSM8K train ~4. Everything else is at or below ~1.6.

**Rebalancing rules where supply is short:**
- non-tech Stack Exchange is capped at 2 passes; any excess goes to DCLM-Edu;
- docs are capped at 4 passes; any excess goes to Cosmopedia computer topics;
- if Nemotron cannot be used, its share goes to FineWeb-Edu >= 4.

**Mix of real and model-written text.** About 16-17% is written by other AI models; real text is about 83%.

**Model size.** About 162 pieces per weight for the decided 492M model (22 x 1,280 plus the 4x Engram).

**Storage:**
- Unique text chosen: about 70-75B pieces, about 150 GB of 2-byte piece files to copy to the GPU server.
- Download: about 450 GB compressed (range ~400-550 GB; DCLM-Edu is the uncertain part).
- Disk on the CPU machine: about 1.2 TB NVMe for downloads, cleaned text and piece files.

**CPU time on a separate 32-core / 128 GB machine:**

| Step | Time |
|---|---|
| Download at 0.5-1 Gbit/s | 1-3 h |
| Parsing and filters | 2-3 h |
| MinHash near-duplicates | 2-3 h |
| Decontamination | 1-2 h |
| Tokenizer training | ~0.5 h |
| Cutting into pieces | 1-1.5 h |
| Checks | ~0.5 h |

That is about 8-13 hours of wall time (~250-400 core-hours), roughly double on 16 cores. A 1-2 hour trial on 500 documents per source comes first. All of these are estimates, not measurements.

## Licence flags

- **Free to use with credit (ODC-By / CC BY):**
  - FineWeb-Edu, Cosmopedia, FineMath and InfiWebMath, Essential-Web, FinePDFs-Edu (the PDFs themselves are copyrighted web documents, like web pages), OpenCoder, SODA, WildChat.
  - DCLM-Edu (CC BY 4.0). DCLM's own card says it was meant for research, so read it before any commercial use.
  - The Common Pile books and papers: each item's licence is in its metadata.
- **Credit plus share-alike (CC BY-SA):**
  - Wikipedia (FineWiki) and Stack Exchange.
  - Usual practice treats this as covering the text, not the trained weights. If the prepared data is ever published, it must keep the same licence and credits.
- **Stack Exchange's 2024 rule:**
  - Since July 2024 the official dump download asks people not to train AI on it; critics say this conflicts with CC BY-SA.
  - The H4 copy used here is probably from before that rule (its dump date is unconfirmed, cited 2023).
  - The Common Pile copy (Dec 2024 community dumps) is only a top-up.
  - The owner should choose knowingly.
- **Code:**
  - Only permissively licensed files (MIT, Apache, BSD and similar) from the Common Pile copy of Stack-Edu. Each file keeps its licence and repository name for credit.
  - This avoids Stack-Edu's 'no licence' files and the Software Heritage bulk-download agreement.
  - Whether this copy includes the latest opt-out removals is unconfirmed.
- **Text written by other AI models:**
  - Cosmopedia (Mixtral) and Nemotron-CC v1 (Mistral-NeMo; Common Crawl terms).
  - SmolTalk and OpenMathInstruct-2: partly Llama 3.1-405B. Meta's licence may ask for credit when Llama outputs train a model (unconfirmed for this use).
  - UltraChat, WildChat, Orca-Math, MetaMathQA: OpenAI models. OpenAI's terms forbid using outputs to build competing models; this is a usual grey zone for a personal project.
  - This is not legal advice.
- **Public domain:** Gutenberg (by US law; labels may be wrong), PEPs, arXiv abstracts (CC0).
- **Manuals:**
  - MDN prose CC BY-SA (code CC0/MIT), ArchWiki GFDL / CC BY-SA, tldr and OpenStax Intro CS CC BY, Python docs PSF licence.
  - The Rust book's licence was not checked.
- **Left out on purpose:**
  - Non-commercial material: MIT OpenCourseWare (bans commercial AI training), Think Python, OpenStax Python, No Robots, Institutional Books.
  - Nemotron v2: NVIDIA training-only agreement, plus Qwen/DeepSeek terms.
  - Stack-Edu's 'no licence' files.
- **Gated terms:** none needed in the chosen mix. Any terms that do need accepting are for the owner to accept himself.

## Unknowns

- Every 'pieces' figure is converted, not measured. Assumed: about 1.15x GPT-2 / SmolLM2 / Comma / cl100k-style tokens and about 1.0-1.05x Mistral / Llama-2-style tokens for prose, about 1.25x for code, about 1.1x for LaTeX maths, or GB of text divided by 4 for prose. Measure on 10 MB per source with the real 16k list, then fix shares and passes.
- FineWiki English text size: the card has no token count, and its 78 GB in memory includes the raw wikitext column.
- Sizes of the FineWeb-Edu int_score >= 4 slice and the DCLM-Edu edu_int_score >= 3 / >= 4 slices. The DCLM-Edu download could be anywhere from 100 to 200 GB.
- Per-language sizes in the Common Pile Stack-Edu copy, and whether it reflects The Stack v2's later opt-outs (v2.2.0 removed opt-outs up to 2026-07-29).
- The Stack Exchange H4 dump date, and how much text survives the vote filter per site. Non-tech supply could be under 2B pieces, in which case its share moves to DCLM-Edu.
- In the Essential-Web code-w-DCLM sample: how much is CS (FDC 004/005) versus maths, and the exact label values to filter on.
- Nemotron-CC v1 file layout and Diverse QA size (its index page returned 503 on 18 November), and whether its gains measured at 8B hold at about 0.5B.
- Unique size of the docs/textbook group (no published sizes) and of the chat sets (estimated from in-memory bytes). Pass counts for Simple Wikipedia, docs and SODA depend on these.
- Whether results measured at 1-8B (the DCLM blend, Essential-Web, Nemotron, MiniCPM's decay) carry over to a ~0.5B model, and whether the Engram lookup memory changes how much data or repetition helps.
- Whether 13-gram decontamination catches paraphrased benchmark items inside synthetic sets (Cosmopedia, Nemotron QA, OpenMathInstruct-2).
- Whether Cosmopedia's prompt text reliably names the topic; there is no topic column to weight computer topics by.
- The chat layout for pretraining chats (the pico's 'User:/Assistant:' lines or reserved marker pieces) is the owner's decision, and the behaviour filters on chats need his yes.
- Download speed, CPU hours and cost are rough estimates; the 500-document trial gives real rates.
- Licence status of model-written sets (Llama 3.1 and OpenAI output terms) for this project: not legal advice.
