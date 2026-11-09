# Domain 10 (Physics + AI) source audit, 9 Nov 2026

Requested by Raju: PINNs, FNO, neural operators; not only research papers. Evidence labels: **run** (I ran a query or script and read the result), **read** (I read the page or file), **estimate** (my arithmetic, assumption stated).
All metadata queries were read-only; nothing from the sources below was downloaded except the paper text already collected. Scripts are in the session scratchpad (`tools\d10_audit_*.py`); raw results in `d10audit\*.json`.

## 1. Answer in one paragraph

The big named sources (The Well, PDEBench, PDEArena, CFDBench, PDEGym, FlowBench, LagrangeBench, RealPDEBench) are **numerical simulation arrays** (HDF5/NPY/Zarr, hundreds of GB to several TB each). A text language model cannot learn from them as tokens in any useful way, so they add **almost no training tokens**. What a language model can use from them is their **text**: papers, READMEs, documentation and source code. For the 11 named repositories that is only about **6 MB of code and docs in total (about 1.7M tokens at most, estimate)**, of which about **4.7 MB (about 1.4M tokens)** has a permissive licence. Several of the largest sets are also **non-commercial** (PDEGym, FlowBench, RealPDEBench, both DeepONet repos) and must stay out under our licence rule. **Result after collection (run, 9 Nov night): the `physai` bin holds 52,255,166 tokens** (3,704 papers, all CC BY): 31.2M from the first query, about 8.0M from the neighbouring-topic papers (575 kept of 916 candidates), 13.06M from Swaraj's held set. The remaining gap to the 70M ask (about 18M) can only come from a **permissive-licence GitHub code lane** (unmeasured, order of tens of millions of tokens, needs someone with a GitHub token) or from the CC BY-SA papers if Raju allows them (under 2M).

## 2. Named sources

| Source | What it is | Size (run/read) | Data licence | Code licence | Text we can use (run: file listing) | Verdict |
|---|---|---|---|---|---|---|
| The Well (PolymathicAI) | 15 TB of physics simulations, 20 HF datasets | 7.9 GB to 2.65 TB per set | HF cards: CC-BY-4.0 on 16 sets; **no licence stated on 4** (MHD_64, turbulent_radiative_layer_3D, supernova_explosion_128, euler_multi_quadrants_openBC) | BSD-3-Clause | code 0.42 MB + docs 0.21 MB; notebooks 11 MB (mostly outputs) | Arrays: not usable as text. Docs/code: yes, tiny |
| PDEBench | 1D/2D/3D PDE data, DaRUS + HF | 336 files on DaRUS; the 10 files the page listed are 7.7 GB each (read); HF Advection 66 GB, Burgers 99 GB | CC BY 4.0 (DaRUS page, read; HF cards agree) | MIT ("except where otherwise stated", LICENSE.txt read; GitHub shows NOASSERTION) | code 0.59 MB + docs 0.04 MB | Arrays: no. Code/docs: yes, tiny |
| PDEArena (Microsoft) | NS-2D, ShallowWater-2D, Maxwell-3D | 43 / 124 / 122 GB | MIT (HF cards) | MIT | code 0.39 MB + docs 0.05 MB | Same |
| CFDBench | CFD for ML, 4 flow problems | 14.4 GB interpolated, about 460 GB raw (README) | HF `chen-yingfa/CFDBench`: Apache-2.0; **GitHub repo has no LICENSE file** | none stated (all rights reserved by default) | code 0.49 MB | **Licence unclear**: ask the authors or skip |
| PDEGym / Poseidon (ETH CAMLab) | 20 PDE datasets behind Poseidon | 1.4 to 110 GB each | **CC BY-NC 4.0 on all 20** | Poseidon repo: no LICENSE file | code 0.2 MB | **Excluded (non-commercial)** |
| FlowBench (BGLab) | 10K+ DNS flow samples | 4.35 TB | **CC BY-NC 4.0** | not checked | n/a | **Excluded (non-commercial)** |
| RealPDEBench (Westlake) | sim + real measurements, ICLR 2026 | 2.04 TB | **CC BY-NC 4.0** (repo and HF) | CC BY-NC 4.0 | code 1.1 MB | **Excluded (non-commercial)** |
| LagrangeBench (TUM) | particle fluid benchmark | Zenodo 8.4 GB | CC BY 4.0 (Zenodo record, run) | MIT | code 0.22 MB + docs 0.04 MB; notebooks 0.78 MB | Arrays: no. Code: yes, tiny |
| DeepXDE | PINN library | repo 28 MB (mostly non-text) | n/a | **LGPL-2.1** | code 0.84 MB + docs 0.41 MB | Usable as code; copyleft licence needs Raju's view |
| NeuralOperator | FNO and neural operator library | repo 121 MB (mostly non-text) | n/a | MIT | code 1.36 MB + docs 0.15 MB | Usable, tiny |
| DeepONet (`lululxvi/deeponet`, `lu-group/deeponet-fno`) | original DeepONet code | 50 to 63 KB | n/a | **CC BY-NC-SA 4.0** (LICENSE read) | 0.14 / 0.17 MB | **Excluded** |

Permissive subset (The Well, PDEBench, PDEArena, LagrangeBench, DeepXDE, NeuralOperator): 4.7 MB of code and docs, about 1.4M tokens (estimate, 3.5 bytes per token).

Why not tokenise the arrays: 15 TB of floating-point fields is tens of trillions of numeric tokens carrying very little language-level information; it would crowd out real text and needs a different (field-based) model, not a text model. I recommend we do not do it. If Raju wants a numeric-data model later, that is a separate project.

## 3. Portals (run, 9 Nov 2026)

| Portal | What I found | Text value |
|---|---|---|
| arXiv | Title/abstract hits: "physics-informed neural network" 2,997; "neural operator" 1,790; DeepONet 487; "Fourier neural operator" 567; "physics-informed" 6,844. Licence is per paper; most are arXiv's own non-exclusive licence (not reusable). Only the CC BY ones count, and these are already inside the OpenAlex lane | Already collected (see section 4) |
| Zenodo | PINN 358 records; neural operator 327; FNO 139; "physics-informed PDE" 7,048; PDEBench 6; LagrangeBench 1; CFDBench 1 (unrelated). In the first 100 of each query: 67 to 82% CC-BY-4.0, 7 to 17% MIT, about 1/3 publications (PDF, mostly duplicates of arXiv/journal papers), about 1/3 software (code zips), rest datasets (HDF5/NPY/checkpoints, hundreds of GB) | Publications and code zips only; small |
| Figshare | PINN / operator queries return about 150 to 200 items; of 40 sampled per query about 80% CC BY 4.0, rest CC BY-NC / MIT / CC0; mostly figures, thesis files and small datasets | Low |
| Dataverse (Harvard) | "physics-informed" 30 datasets, "neural operator" 13, "scientific machine learning" 5; licences CC0, CC BY, some CC BY-NC | Low |
| Dataverse (DaRUS, Stuttgart) | My search call failed a TLS certificate check on this machine and I did not bypass it. The PDEBench page itself was read through the browser fetch (CC BY 4.0) | n/a |
| Mendeley Data | No key-free search API (the public endpoint ignores the query; the official API returns 401). The web search page returned 160,204 loosely matching entries with unrelated top hits, so no usable count | Not assessed; mostly experimental CSV files |
| Hugging Face | Besides the sets above: community subsets (`erbacher/PDEBench-1D`, `HD-PINN/*`, `PinnaclePDE/*`, `EvalData/pinnbench`, `lukebhan/*`, and similar). All numeric arrays or small benchmark tables; licences not audited one by one | None as text |
| GitHub topics | Permissive repos on PINN / neural-operator / DeepONet / FNO / scientific-ML topics: MIT 1,075, Apache-2.0 137, BSD-3 26 (counts add topic overlaps, so unique repos are fewer, maybe 700) | Code lane, see section 4 |

## 4. What can actually raise the token count (all estimates unless stated)

| Lane | Status | Tokens | Notes |
|---|---|---|---|
| CC BY papers, current query (done) | built 9 Nov | **31.2M new (run)** + Swaraj's held 13.06M = **44.3M** | all CC BY, DOI and licence per paper |
| CC BY papers, neighbouring topics (done) | built 9 Nov | **about 8.0M new (run)**: 575 papers kept of 916 candidates (162 of 183 arXiv, 413 of 733 other hosts) | My first estimate of 15 to 30M was wrong: the per-topic counts (neural network + PDE 1,480; neural ODE 410; surrogate + FE 364; and others) overlap heavily; the union was 1,901 papers, of which 916 were not already held and had a PDF link. Corrected after the build |
| Named repos' code and docs | listing only (run) | about **1.4M** permissive | tiny |
| Permissive GitHub code on these topics | not measured | order of **20 to 80M** (rough: about 700 repos x 0.2 to 0.5 MB text each, assumption) | needs file-by-file fetching: without a GitHub token the API allows 60 requests per hour, so it needs a team member with a token or several days. Strip notebook outputs; deduplicate against the `code` bin |
| CC BY-SA (109) and CC0 (43) papers on the current query | not taken | under 2M | waits for Raju's licence decision |
| Non-commercial licences (CC BY-NC variants: about 2,400 papers on the current query) | excluded | n/a | licence rule |
| Publisher PDFs that refuse automated download | not taken | about 1,000 papers lost | not bypassed |

With all paper lanes done, Domain 10 stands at 52.3M tokens (`physai`), about 75% of the 70M ask. The arrays are not needed; the last 18M needs the GitHub code lane or a licence decision on CC BY-SA papers.

## 5. Decisions needed from Raju

1. Confirm: simulation arrays are **not** converted to text tokens (recommended).
2. Licence rule for LGPL-2.1 code (DeepXDE) and CC BY-SA papers.
3. Approve the neighbouring-topic paper lane (I run it) and the GitHub code lane (needs someone with a GitHub token).
4. Ask CFDBench's authors for a licence, or skip it.
5. Optional: The Well's four sets with no stated licence matter only if arrays are wanted; not needed for text.

## 6. Limits of this audit

- Sizes and licences come from Hugging Face cards, repository metadata and the pages named; I did not open the data files.
- Zenodo and Figshare figures are from the first 100 / 40 records per query, not complete listings.
- Token figures marked estimate use 3.5 bytes per token for code and observed paper statistics; real tokenisation differs (the paper lane came out 25% above my first estimate).
- GitHub topic counts include overlaps; permissive licence tags are GitHub's detection, not a manual review.
