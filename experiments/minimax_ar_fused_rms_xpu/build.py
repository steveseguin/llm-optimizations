from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent

subprocess.check_call(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    cwd=HERE,
)
