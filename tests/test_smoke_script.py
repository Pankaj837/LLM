"""The team's smoke script (scripts/smoke_test.py) must keep passing as a whole."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_smoke_script_exits_zero_and_reports_no_failures():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py")], capture_output=True, text=True,
                       cwd=str(ROOT), encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-800:]
    assert "0 failed" in r.stdout
