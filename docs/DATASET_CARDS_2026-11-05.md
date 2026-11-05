# Dataset cards: Engineering Reasoning, Numerical Simulation (domain 9), Physics + AI (domain 10) (Raju's task 2), 5 Nov 2026

**Evidence labels.** *Run* = measured by me on the file. *Read* = from the owner's manifest or README. *From records* = from my earlier checks, file not available now. Samples are titles only (the text of each paper is under its own CC BY / CC0 licence, recorded per document in the manifests).

## Domain 9: Numerical Simulation (owner open; collected and handed over by Swaraj)
| | |
|---|---|
| Content | Full-text scientific papers on computational fluid dynamics, finite elements, finite volumes and differences, Galerkin methods, Navier-Stokes, PDE solvers, fluid-structure interaction, spectral elements, meshless methods |
| Source (read) | `common-pile/peS2o_filtered` (3,673 documents, revision `297747513bfb0ff1fbf61ddad3b03319d0f04597`) and `common-pile/arxiv_papers_filtered` (695, revision `033cf7f53f9b348deec868c1a5a48484f3ee9e52`) |
| Method (read, `SWARAJ_COLLECTOR_REVIEW_2026-11-05.md`) | title + abstract screening, at least 2 distinct canonical concepts, English full text of at least 200 words, exact and normalised de-duplication; the version-7 cleanups (e-mail placeholders, `# ` lines, 35 off-topic documents dropped after a manual review of 149 weak ones) were applied after the script |
| Licence (run) | CC BY 4,362 documents, CC0 6; CC BY-SA (88) and public domain (44) are held back for Raju's decision |
| Size (run) | 4,368 documents, 162.4 MB, 27.3M words, **39,897,904 tokens** with `vocab32k_v2` (train 39,803,313 in 4,360 documents, validation 94,591 in 8 documents); document length median 4,791 words (p10 1,971, p90 12,512, max 52,962) |
| Integrity (run) | `text_sha256` recomputed for 4,368 of 4,368 documents; 0 exact duplicates; 0 e-mail strings; 0 lines starting `# `; `check_bins` PASS |
| Concept mix (run) | finite element 2,220 documents, PDE 1,464, Galerkin 1,146, finite difference 1,145, CFD 1,119, Navier-Stokes 916, finite volume 783, fluid-structure 310, spectral element 125, meshless 101, computational mechanics 45; 87% of documents have exactly 2 concepts |
| Samples (run, random) | "Extended Eulerian SPH and its realization of FVM"; "A stabilised finite element framework for viscoelastic multiphase flows using a conservative level-set method"; "Chaos prediction in nano-resonators based on nonlocal elasticity theory"; "A Comprehensive Fluid-Solid Interaction Analysis of Finite Journal Bearings" |
| Token allocation | Raju's 30B table: 2.1B. Held 39.9M = 52 passes at 30B, 5.2 in a 3B pilot. Raju's 4 Nov ask for the 1B run: 60 to 80M, so +20 to +40M |
| 100M model on it (read) | validation loss 2.853 on a sample, never trained on it (`A100_RUN_RESULT_2026-11-01.md`) |
| Known issues (run) | U+FFFD in 161 documents (3.7%), zero-width characters in 5, typographic ligatures (ﬁ ﬂ ﬀ ﬃ) in 198 documents (4.5%, 1,110 characters), 8 documents with no domain term (numerical papers, kept), validation is only 8 documents (`--val-frac 0.02` would give about 90); every document except the first starts with a blank line in the text file |

## Domain 10: Physics + AI (owner open; collected and handed over by Swaraj)
| | |
|---|---|
| Content | Full-text papers on physics-informed neural networks, DeepONet, Fourier neural operators, neural operators, operator learning, physics-informed ML and scientific ML |
| Source (read) | peS2o_filtered (337 documents) and arxiv_papers_filtered (593), same revisions as domain 9 |
| Method (read) | title + abstract screening with a scientific-context word required; bare "FNO" and vision-only "operator" papers excluded; author front matter, e-mail addresses and placeholders stripped for arXiv; a paper matching both domains goes here |
| Licence (run) | CC BY 930 documents; CC BY-SA (22) and public domain (9) held |
| Size (run) | 930 documents, 50.1 MB, 8.2M words, **13,063,500 tokens**; document length median 7,777 words (p10 3,972, p90 14,628, max 73,913) |
| Integrity (run) | `text_sha256` 930 of 930; 0 duplicates; 0 e-mail strings; 0 lines starting `# `; `check_bins` PASS on the train bin; **validation bin has 0 tokens (the hash split put no document in validation)** |
| Term mix (run) | physics-informed neural network 569 documents, neural operator 169, physics-informed scientific ML 125, operator learning 66, DeepONet 66, Fourier neural operator 66, scientific ML 63, PINN with scientific context 28; 917 of 930 mention PINN / neural operator / FNO / DeepONet |
| Samples (run, random) | "Landet: an efficient physics-informed deep learning approach for automatic detection of anatomical landmarks..."; "Research on Modeling Method of Autonomous Underwater Vehicle Based on a Physics-Informed Neural Network"; "PINION: Physics-informed neural network for accelerating radiative transfer"; "3D mask simulation and lithographic imaging using physics-informed neural networks". The first is imaging rather than simulation: some papers are applications of the method outside physics |
| Token allocation | Raju's 30B table: 2.1B. Held 13.1M = 161 passes at 30B, 16 in a 3B pilot. 4 Nov ask for the 1B run: 60 to 80M, so +47 to +67M |
| 100M model on it (read) | validation loss 2.855 on a sample, never trained on it |
| Known issues (run) | U+FFFD in 20 documents (2.2%), zero-width in 6, BOM in 1, ligatures in 34 documents (3.7%); **no validation split with the default `--val-frac 0.001`**. Fix verified on 5 Nov (run): `tokenize_text.py ... --val-frac 0.02` gives 909 train + 21 validation documents (12,788,634 + 274,866 tokens), `check_bins` PASS |

## Engineering Reasoning (capability 4; Prabanjan's corpus, rebuilt by Pankaj on 31 Oct)
**This card is from records. The bin and the 3.1 GB source text are not on Pankaj's laptop (search of Downloads, Documents and Desktop on 5 Nov: no `.bin` above 100 MB other than the Biology / Earth / Math / Multistep bins of 25 Oct, no file with "reasoning" in its name). The sample and a fresh `check_bins` therefore cannot be produced until the file is found or re-uploaded.**
| | |
|---|---|
| Content (from records) | Engineering problems with a reasoning trace (`<think>`) and an answer, one document per problem; found on 28 Oct in the first 20 MB |
| Method (from records) | rebuilt on 31 Oct (tag `scale-run-rc9`) from `engineering_reasoning_corpus_old.txt` (3,114,429,124 bytes) with document separator line `<|endoftext|>` (`"eot": true`) and exact de-duplication; the first build cut a document at every blank line (221,904 "documents" of 26 tokens) and was replaced |
| Size (from records) | train 805.0M tokens, validation about 4.1M, 259,525 whole documents, mean 3,102 tokens; `check_bins` PASS; 0% duplicates in a sample of 3,000 documents; manifest at commit `f541cad` |
| Source and licence | **not recorded anywhere in the repository.** Prabanjan has to state the source dataset and its licence before the card is complete |
| Token allocation | Raju's 30B table: 2.4B. Held 805M = 3.0 passes at 30B (within the 4-pass limit). It was not part of the 100M run; the 100M model's loss on it is 3.339 (read) |
| Open | the file; source and licence; a fresh whole-file `check_bins` and hash |
