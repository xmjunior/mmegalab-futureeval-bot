"""Reproduce a registered block from the frozen files without using a provider key."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent


def check(expected, actual, where="result"):
    if type(expected) is not type(actual):
        raise ValueError("type changed at " + where)
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError("fields changed at " + where)
        for k in expected:
            check(expected[k], actual[k], where + "." + k)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError("length changed at " + where)
        for i, (a, b) in enumerate(zip(expected, actual)):
            check(a, b, f"{where}[{i}]")
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("metric changed at " + where)
    elif expected != actual:
        raise ValueError("value changed at " + where)


def replay(block):
    with TemporaryDirectory() as tmp:
        work = Path(tmp) / block
        shutil.copytree(ROOT / block, work, ignore=shutil.ignore_patterns("__pycache__"))
        if block == "forecastbench_shadow":
            for sibling in ("binary", "forecastbench_export"):
                shutil.copytree(ROOT / sibling, work.parent / sibling, ignore=shutil.ignore_patterns("__pycache__"))

        def python(*args):
            return subprocess.check_output([sys.executable, *args], cwd=work, text=True)
        if block in {"continuous", "binary", "evidence", "forecastbench_export", "forecastbench_shadow"}:
            if not list(work.glob("test*.py")):
                raise ValueError("missing regression tests for " + block)
            python("-m", "unittest", "discover", "-v")
        if block == "forecastbench_export":
            print("forecastbench_export: fixtures and results reproduced by unit tests")
            return
        if block == "forecastbench_shadow":
            expected = json.loads((work / "results.json").read_bytes())
            actual = json.loads(python("model.py"))
        elif block == "continuous":
            expected = json.loads((work / "results.json").read_bytes())
            python("evaluate.py")
            actual = json.loads((work / "results.json").read_bytes())
        elif block == "binary":
            expected = json.loads((work / "results.json").read_bytes())
            python("evaluate.py", "--out", "replayed.json")
            actual = json.loads((work / "replayed.json").read_bytes())
        elif block == "evidence":
            expected = json.loads((work / "result.json").read_bytes())
            actual = json.loads(python("experiment.py"))
        else:
            expected = json.loads((work / "scenarios.json").read_bytes())
            actual = json.loads(python("scenarios.py"))
        check(expected, actual)
    print(f"{block}: frozen results reproduced")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=("binary", "continuous", "evidence", "economics", "forecastbench_export", "forecastbench_shadow"))
    replay(parser.parse_args().block)
