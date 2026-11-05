# Avinash's Drive folder `data` (link of 5 Nov): check, 5 Nov 2026

**Evidence labels.** *Run* = I opened the folder in a browser, listed every file and downloaded and read the small ones (with Pankaj's permission in the chat: "do everything"). *Read* = taken from those files. *Not checked* = not downloaded: `universal_index_final.parquet` (about 25 GB), `prana_g.duckdb` (about 3.7 GB), the 3,640 raw files.

## What the folder holds (run)
| Sub-folder | Content | Dates |
|---|---|---|
| `data/` | `Medical Dataset Documentation.pdf` (3 pages), `pubmed_download_list.txt` (1,334 URLs) | 19 Oct |
| `index/` | `universal_index_final.parquet`, `prana_g.duckdb`, `index_reports/index_build_20261023_170459.json` | 23 Oct |
| `processed/reports/` | `pankaj_verification_report.txt`, `dedup_report.json`, `domain_row_counts.csv`, `merged_dataset_manifest.json`, `copernicus_duplicate_sample.csv` | 24 Oct |
| `raw/` | 3,640 Parquet files (2,849 `who_batch_*`, 335 `world_population_prospects_*`, 119 `clinicaltrials_gov_*`, 80 `iso_standards_*`, 63 `biology_part_*`, 54 `itu_*`, 50 `world_bank_*`, 34 `bis_*`, then single files such as ChEMBL, Gaia, CAD STEP, EXFOR, iNaturalist, Copernicus) | 6 to 24 Oct |

**It is not the "one folder for all domain bins" Raju asked for.** There is no token file (`.bin`), no text corpus and nothing newer than 24 Oct. The files Ravi is waiting for (a place to upload chem, cad and engineering bins) and the `universal_final.parquet` Avinash named are not here.

## The merged file (read; every number below is arithmetic on his reports, the 44 GB file itself was not opened)
- `universal_merged.parquet`: 64,364,210 rows, 47,561,476,493 bytes (44.3 GB), SHA-256 `00632085...6100F7`. Rows agree in all four reports (verification, dedup, manifest, index build). The 33 domain counts add up to 64,364,210 (checked).
- **Domain labels are not harmonised: 33 labels for what should be about 15 domains.** The same domain appears under several spellings: `Chemical` 5,188,674 and `chemical` 60,222; `Human_Social` 8,665,020, `human_social` 7,056, `Human & Social` 4,771, `Human & Social Sciences` 2,606,245 and `Human & Social Systems` 1,007; `Economic & Industrial` 605,379 and `Economic_Industrial` 7,218. The Human / Social family alone has 6 labels and 11.38M rows (18%).
- Dedup key `entity_id`: 50,910,626 unique, 8,910,000 duplicate groups, 13,453,584 "removable" rows; **nothing was deleted** and his own check ends `FAIL - Dedup report records no row-count reduction`.
- **The `entity_id` key looks broken, so the 13.45M figure should not be used to delete anything.** In `copernicus_duplicate_sample.csv` all 20 rows share one `entity_id` (`PRG:COPERNICUS_S:COPERNICUS_S3A_OL_2_WFR___20260101T120000_0`, exactly 60 characters) but have 20 different `source_id`, names and properties (`...00010512`, `...00060506`, `...00110507` and so on). The id is cut at 60 characters, so different products collapse into one. Deduplicating on it would delete real, different records. The number of groups (8,910,000) is also suspiciously round. Not checked on the other 32 domains.
- What the rows are: catalogue and statistics records (satellite products, WHO / World Bank indicators, standards, trials, ontologies) with a JSON `properties` column, not running text. Token counts cannot be read from them; the earlier `v4_text` rule is needed first (see `docs/DATA_STATUS_2026-10-20.md`). Nothing here has been tokenized with the frozen tokenizer.

## Medical plan (read)
`Medical Dataset Documentation.pdf`: PubMed / MEDLINE baseline (about 39 million records, 1,334 files `pubmed26n0001`..`pubmed26n1334`, about 38 GB compressed; the URL list has 1,334 distinct lines, so it is complete and consistent). Keep title, abstract, MeSH terms and keywords; strip boilerplate and citations; English only; near-duplicate removal; metadata (PMID, DOI, journal, year) stored apart. It is a plan; no processed output exists. Licence: NLM data carry the NLM terms of use and individual abstracts can carry publisher copyright; the licence decision Raju has open for CC BY-SA and public-domain text would need to cover this (general knowledge, not checked against the terms).

## Consequences
1. Avinash cannot be shown as "the data lake is ready" for the 30B plan: no bins exist there.
2. Before anyone dedups the merged file the `entity_id` truncation must be fixed (key on `source` + `source_id`, or the full id) and the 33 domain labels mapped (the 15-capability mapping was asked on 19 and 24 Oct and has not been delivered).
3. If Avinash takes over Domains 9 and 10 (Raju to confirm), his folder gives no head start: neither domain is in it, apart from catalogue records such as EXFOR and Copernicus that are not scientific text.
