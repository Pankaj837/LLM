#!/usr/bin/env python
"""Answer and code checks shared by ``sft_prepare.py`` (verify a dataset) and ``sft_generate.py`` (score a model).

* :func:`final_number` / :func:`numbers_match`: the last number of a text ("The answer is 408", "Answer: 4.08e2") and a comparison with a
  reference number, exact up to a small relative tolerance.
* :func:`extract_code_block`: the first fenced code block of a text (or the text itself when it has none).
* :func:`run_python_tests`: run a Python snippet followed by its tests in a separate interpreter (``python -I``) with a time limit, in an
  empty temporary directory and with a minimal environment. **It executes the code it is given.** Use it only on a machine you can
  throw away (the GPU VM or a container), never on the laptop with data from an unreviewed source. It is not a security sandbox: it
  does not block network or file access outside the temporary directory.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ANSWER_CUE = re.compile(r"(?:answer\s+is|answer\s*:|=)\s*\$?\s*(" + NUMBER.pattern + ")", re.I)
_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)


def parse_number(text: str) -> Optional[float]:
    m = NUMBER.fullmatch(text.strip().rstrip(".").replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def final_number(text: str) -> Optional[float]:
    """The number the text commits to: the last one after "answer is" / "Answer:" / "=", else the last number in the text."""
    cues = list(_ANSWER_CUE.finditer(text))
    raw = cues[-1].group(1) if cues else (NUMBER.findall(text) or [None])[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", "").rstrip("."))
    except ValueError:
        return None


def numbers_match(got: Optional[float], want: float, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    if got is None:
        return False
    return abs(got - want) <= max(abs_tol, rel * abs(want))


def answer_matches(response: str, answer, rel: float = 1e-6) -> bool:
    """Numeric reference -> the response's final number must match it; text reference -> it must occur in the response (case-insensitive)."""
    want = answer if isinstance(answer, (int, float)) else parse_number(str(answer))
    if want is not None:
        return numbers_match(final_number(response), float(want), rel)
    return str(answer).strip().casefold() in response.casefold()


def extract_code_block(text: str) -> str:
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip("\n")


def run_python_tests(code: str, tests: str = "", timeout: float = 10.0) -> Tuple[bool, str]:
    """Run ``code`` then ``tests`` (plain ``assert`` statements) in ``python -I``. Returns ``(passed, message)``. EXECUTES THE CODE."""
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + tests + "\n")
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout} s"
    if r.returncode == 0:
        return True, "ok"
    return False, (r.stderr.strip().splitlines() or ["non-zero exit"])[-1][:300]
