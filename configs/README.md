# configs/

`run_100m.example.json` is a **settings file, not a program**: double-clicking it only makes Windows ask which app should open it
(that is the "box"). Do this instead:

1. Copy it to `configs/my_run.json` and open the copy in a text editor (right-click → Open with → Notepad, or VS Code).
2. Change only the `"paths"` lines to where your files really are (a mounted Drive such as `/content/drive/MyDrive/...`, or a local folder).
   Leave `"vocab"` as it is: it is the frozen tokenizer. `max_tokens`, `sample_rg` and `max_files` say how much of each source to use;
   `weight` is each domain's share of training.
3. In a terminal, from the repository folder: `python scripts/build_dataset.py configs/my_run.json`

It prints the table of tokens per domain, the integrity check, and the exact `--train-mix` arguments for training. If a path does not match
any file it stops with a clear message and writes nothing half-finished.
