#!/usr/bin/env python3
"""Domain 10 (Physics + AI) code lane: permissively licensed GitHub repositories on PINN / neural-operator topics.

Stages (each resumable, data only is read - nothing from a repository is executed):
  list   search GitHub for repositories by topic with an MIT / Apache-2.0 / BSD licence -> repos.jsonl
  fetch  download each repository as a tarball from codeload.github.com (does not use the API quota),
         keep source / docs / notebook (outputs stripped) text files -> docs.jsonl
  stats  print repositories, files, characters and an estimated token count

Authentication (optional, recommended): set the environment variable GITHUB_TOKEN to a classic or fine-grained
read-only token. The token is only sent to api.github.com in an Authorization header and is never written to disk.
Without a token the search API allows 10 requests per minute, which is enough for `list` (about 40 requests).

Example:
  python -I scripts/collect_domain10_github_code.py list  --out d10code
  python -I scripts/collect_domain10_github_code.py fetch --out d10code --max-repos 800
  python -I scripts/collect_domain10_github_code.py stats --out d10code
"""
import argparse, hashlib, io, json, os, re, sys, tarfile, time, urllib.parse, urllib.request

TOPICS = ["physics-informed-neural-networks", "pinn", "pinns", "neural-operator", "neural-operators", "fourier-neural-operator",
          "deeponet", "operator-learning", "physics-informed-machine-learning", "scientific-machine-learning", "neural-pde",
          "physics-informed", "pde-solver", "physics-informed-deep-learning", "learning-pdes", "scientific-ml"]
LICENCES = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause"]
TEXT_EXT = {".py": "python", ".jl": "julia", ".md": "markdown", ".rst": "rst", ".ipynb": "notebook", ".m": "matlab",
            ".cpp": "cpp", ".cc": "cpp", ".h": "c", ".hpp": "cpp", ".c": "c", ".cu": "cuda", ".f90": "fortran", ".sh": "shell"}
SKIP_DIRS = {"node_modules", ".git", "venv", ".venv", "env", "build", "dist", "third_party", "vendor", "__pycache__", "site-packages",
             "data", "datasets", "checkpoints", "ckpt", "results", "outputs", "wandb", "logs"}
MAX_FILE = 400_000          # bytes; larger files are generated or data
MAX_REPO_TAR = 150 * 2**20  # skip repositories whose tarball is above 150 MB (mostly data)
UA = {"User-Agent": "prana-g-d10-code-collector"}


def api(url):
    h = dict(UA, Accept="application/vnd.github+json")
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = "Bearer " + tok
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return json.load(r), r.headers
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                wait = int(e.headers.get("Retry-After") or 0) or 62
                print("rate limited, waiting %ds" % wait, file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            if e.code == 422:
                return {"items": []}, {}
            raise
        except Exception:
            time.sleep(5)
    raise RuntimeError("API failed: " + url)


def stage_list(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "repos.jsonl")
    seen = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l); seen[r["full_name"]] = r
    n_req = 0
    for topic in (a.topics.split(",") if a.topics else TOPICS):
        for lic in LICENCES:
            for page in range(1, 11):
                q = "topic:%s license:%s" % (topic, lic)
                url = "https://api.github.com/search/repositories?q=%s&per_page=100&page=%d" % (urllib.parse.quote(q), page)
                data, _ = api(url); n_req += 1
                items = data.get("items", [])
                for it in items:
                    if it["full_name"] in seen or it.get("fork"):
                        continue
                    lic_id = (it.get("license") or {}).get("spdx_id", "")
                    if lic_id not in ("MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause"):
                        continue
                    seen[it["full_name"]] = {"full_name": it["full_name"], "license": lic_id, "size_kb": it["size"], "stars": it["stargazers_count"],
                                              "branch": it["default_branch"], "pushed_at": it["pushed_at"], "topics": it.get("topics", [])}
                if len(items) < 100:
                    break
                if not os.environ.get("GITHUB_TOKEN"):
                    time.sleep(7)           # unauthenticated search: 10 requests per minute
            print("%-40s %-13s repos so far %d" % (topic, lic, len(seen)), flush=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in seen.values():
                    f.write(json.dumps(r) + "\n")
    print("repositories:", len(seen), "API requests:", n_req)


def strip_notebook(raw):
    try:
        nb = json.loads(raw)
    except Exception:
        return None
    out = []
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if not src.strip():
            continue
        out.append(("# [markdown]\n" + src) if c.get("cell_type") == "markdown" else src)
    return "\n\n".join(out)


def stage_fetch(a):
    reps = [json.loads(l) for l in open(os.path.join(a.out, "repos.jsonl"), encoding="utf-8")]
    if a.shuffle:
        import random
        random.Random(1).shuffle(reps)      # unbiased sample for yield estimates
    else:
        reps.sort(key=lambda r: -r["stars"])
    done = set()
    dpath = os.path.join(a.out, "docs.jsonl")
    if os.path.exists(dpath):
        for l in open(dpath, encoding="utf-8"):
            done.add(json.loads(l)["repo"])
    hashes = set()
    out = open(dpath, "a", encoding="utf-8")
    nrep = 0
    for r in reps[: a.max_repos]:
        if r["full_name"] in done:
            continue
        if r["size_kb"] * 1024 > 4 * MAX_REPO_TAR:
            continue
        url = "https://codeload.github.com/%s/tar.gz/refs/heads/%s" % (r["full_name"], r["branch"])
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as resp:
                blob = resp.read(MAX_REPO_TAR + 1)
        except Exception as e:
            print("skip", r["full_name"], e, flush=True); continue
        if len(blob) > MAX_REPO_TAR:
            print("skip (too large)", r["full_name"], flush=True); continue
        n_files = chars = 0
        try:
            tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
            for m in tf:
                if not m.isfile() or m.size > MAX_FILE or m.size < 80:
                    continue
                parts = m.name.split("/")[1:]
                if not parts or any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
                    continue
                ext = os.path.splitext(parts[-1])[1].lower()
                if ext not in TEXT_EXT:
                    continue
                raw = tf.extractfile(m).read()
                try:
                    txt = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if ext == ".ipynb":
                    txt = strip_notebook(txt)
                    if not txt:
                        continue
                if "\x00" in txt or max((len(x) for x in txt.split("\n")), default=0) > 2000:
                    continue                                     # minified or generated
                h = hashlib.sha256(re.sub(r"\s+", " ", txt).encode()).hexdigest()
                if h in hashes:
                    continue
                hashes.add(h)
                doc = "Repository: %s\nLicense: %s\nFile: %s\n\n%s" % (r["full_name"], r["license"], "/".join(parts), txt)
                out.write(json.dumps({"id": "%s:%s" % (r["full_name"], "/".join(parts)), "repo": r["full_name"], "license": r["license"],
                                      "lang": TEXT_EXT[ext], "text": doc, "text_sha256": h}) + "\n")
                n_files += 1; chars += len(doc)
        except Exception as e:
            print("tar error", r["full_name"], e, flush=True)
        nrep += 1
        out.flush()
        print("%-45s %-12s files=%-4d chars=%d" % (r["full_name"], r["license"], n_files, chars), flush=True)
    out.close()


def stage_stats(a):
    n = chars = 0; repos = set(); lic = {}
    for l in open(os.path.join(a.out, "docs.jsonl"), encoding="utf-8"):
        d = json.loads(l); n += 1; chars += len(d["text"]); repos.add(d["repo"]); lic[d["license"]] = lic.get(d["license"], 0) + 1
    print("repositories with text: %d  files: %d  characters: %d  estimated tokens (chars/3.5): %.1fM  by licence: %s" % (len(repos), n, chars, chars / 3.5 / 1e6, lic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["list", "fetch", "stats"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-repos", type=int, default=800)
    ap.add_argument("--shuffle", action="store_true", help="fetch in random order (for yield estimates)")
    ap.add_argument("--topics", default="", help="comma-separated topics for `list` (default: all)")
    a = ap.parse_args()
    {"list": stage_list, "fetch": stage_fetch, "stats": stage_stats}[a.stage](a)
