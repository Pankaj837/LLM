# Work report — Scientific Programming

**Prepared for:** Aarav  
**Date:** 2026-10-06  
**Assigned work:** Scientific Programming domain data and 51M-model testing

## Assignment and scope

The group chat assigns Aarav the Scientific Programming domain and asks for dataset/model results. The collection-plan README allocates 2% of the wider corpus to this domain and covers Python, C/C++, Rust, Julia, CUDA, MATLAB, R, Fortran, SQL, Bash, LaTeX, Verilog, and VHDL. That README lists Pankaj as owner, while the group chat lists Aarav; this repository preserves both source documents so the ownership discrepancy is visible.

## Work completed

1. Cloned the upstream LLM project and dataset-registry project for reference. The LLM code used for the run was at commit `5fb3edf18e592662788c79b90ec3ee97b0e602c6`.
2. Compared the supplied vocabularies with the checkpoint. The earlier 8,001-ID `vocab.tok` did not match. `final_vocab_51m.tok` has 8,193 IDs and exactly matches the checkpoint's recorded SHA-256.
3. Collected and filtered source from 22 repositories. The manifest records repository URLs and commits, root licenses, file paths, hashes, language labels, and token counts. MIT, Apache-2.0, and BSD-family root licenses were allowlisted; binary, generated, vendored, oversized, short, and duplicate files were filtered.
4. Built the final **22,632,151-token** corpus: 21,485,660 training tokens and 1,146,491 validation tokens across 4,341 and 238 documents, respectively. Validation assignment is deterministic and document-level. Its hash is salted separately from file selection to avoid selection/split bias.
5. Fine-tuned the 51.7M checkpoint on the dataset using the RTX 5060 Laptop GPU, CUDA 12.8, bfloat16, sequence length 512, batch size 4, and learning rate 1e-4. The original step-5,000 checkpoint was continued through step 7,020.
6. Evaluated the original and fine-tuned checkpoints on the final held-out split and sampled continuations at temperature 0.7/top-p 0.9. Ran the repository pytest suite and explicit GPU handoff checks.

## Training history

The final model was trained in stages while the source set was being expanded:

- Steps 5,000–6,020 used the initial 18-repository corpus (21,114,711 tokens).
- Steps 6,020–7,020 used the final 22-repository corpus (22,632,151 tokens).

Therefore, the final checkpoint's full history is not a single uninterrupted run over only the final corpus. The stage-3 config, complete appended metrics log, both evaluation reports, base checkpoint, final checkpoints, final corpus, and provenance are included here.

## Evaluation summary

On the exact same final validation split, the original checkpoint scored 6.835 nats (PPL 930.03), and the fine-tuned checkpoint scored 2.054 nats (PPL 7.80). The training-token unigram baseline scored 6.164 nats, so the fine-tuned model improves on that baseline by 4.110 nats.

The loss improvement is substantial, but sampled outputs still show repetition and cross-language leakage. The run demonstrates domain adaptation and measurable held-out improvement; it does not establish production-quality scientific code generation.

## Testing

- Full upstream pytest suite: exited successfully. The run included skips and two legacy return-value warnings.
- `tests/test_gpu_handoff.py`: exited successfully on the RTX 5060 Laptop GPU. Reported cached decode throughput was 87.3 tokens/s for a 128-token prompt plus 128 generated tokens; peak memory was 245 MB for that test.
- Both added preparation/evaluation scripts compile successfully. The final evaluation script also ran over the full 1,146,491-token validation set.

## Remaining limitations

- The 33.4M-token target is an estimate based on the wider corpus allocation, not a fixed acceptance threshold. Current data reaches 67.8% of that estimate. SQL, MATLAB, Bash, and LaTeX yields remain far below their provisional language allocations.
- Per the user's direction, no additional data sources were added after the final 22-repository corpus was built.
- Generated sample quality needs human review and further training before claiming strong code-generation behavior.
- The group handoff message has not been sent; this report and the checkpoints are ready to share.
