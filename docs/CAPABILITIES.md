# The 15 capabilities, the extra domains, and what data exists for each (24 Oct 2026)

The official list is Raju's message of 10 Oct 2026 ("work on the dataset required for 3B+ parameter and preprocess it").
Token numbers are the estimates of `docs/DATA_STATUS_2026-10-20.md`; "none" means nothing has been delivered.

| # | Capability | Original owner | Data now |
|---|---|---|---|
| 1 | Mathematics | Prabanjan | 4.92M documents (~9.3B tokens) |
| 2 | Physics | Prabanjan | ~160k documents (~0.64B); mixed public sources (Cosmopedia textbooks, arXiv LaTeX, OpenWebMath), 13% with little physics |
| 3 | Chemistry | Aarav → Ravi | chemistry text ~0.5B; Avinash's "chemical" records are database rows, not prose |
| 4 | Engineering reasoning | Afnan (left the internship, 24 Oct: **needs a new owner**) | **none** |
| 5 | Multi-step calculations | Prabanjan | 18.0M generated arithmetic Q&A (~0.43B) |
| 6 | Scientific Q&A | Aarav → Ravi | 5.93M rows (~2.5B) |
| 7 | Code | Ravi | 2.69M files (3.28B reported; 83% TypeScript) |
| 8 | CAD/STEP understanding | Ravi | 940,576 rows (~0.75B; 40.5% procedural) |
| 9 | Numerical simulation (CFD/FEA/PDEs) | Afnan (left: **needs a new owner**) | only synthetic OpenFOAM templates, no text |
| 10 | Physics + AI (PINNs/FNO/neural operators) | Afnan (left: **needs a new owner**) | **none** |
| 11 | Generative engineering | Ravi | **none** delivered |
| 12 | Experimental validation | Avinash | **none** |
| 13 | Scientific discovery | Aarav → Gokul | 559,865 documents after dedup (~1.6B) |
| 14 | Technical engineering communication | Avinash | **none** |
| 15 | Engineering-agent / tool use | Avinash | **none** |

**Domains assigned outside the 15:** Astronomy & Space (Prabanjan; 0.11B, two thirds catalog rows), Quant Finance (Prabanjan;
0.06B, numbers only), Biology (Gokul; ~800,000 documents, ~4.5B), Earth & Environment (Gokul; 375,582 documents, ~2.4B),
Scientific Programming (Ravi; 1.2–2.1B after removing repeats), Engineering (Afnan; none), Medical (PubMed abstracts in
Avinash's zip; ~6–8B, not processed). 7 of the 15 capabilities have no usable text yet (4, 9, 10, 11, 12, 14, 15), and 3 has only partial coverage.

## What Avinash is asked to map (one CSV)
For every `domain` / `sub_domain` value in his universal index, exactly one label:
a capability number **1–15**, or an extra-domain name (`astronomy`, `quant`, `biology`, `earth_environment`, `scientific_programming`,
`medical`), or `structured_records` (database rows that are not prose: ChEMBL, ZINC, census and similar), or `none`.
Columns: `domain, sub_domain, rows, capability_or_label`. Rows previously marked `manual_review` must get a decision. With this CSV
the per-capability token counts can be computed and the gaps (above) confirmed against his 39.7M-row index.
