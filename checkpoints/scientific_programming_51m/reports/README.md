# Run outputs

- `evaluation_pretrained.json`: supplied checkpoint evaluated on the final held-out split.
- `evaluation.json`: fine-tuned checkpoint evaluated over the same full split, including sampled continuations.
- `metrics.jsonl`: step-level training and sampled validation metrics from the staged run.
- `train_config.json`: raw configuration emitted by the final training stage; its relative paths reflect the original LLM checkout layout.
- `training_history.json`: portable summary of all three training stages and the data size used in each.
