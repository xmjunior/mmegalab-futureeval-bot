"""Replay sealed inputs and report any cross-runtime floating-point variation."""
import hashlib
import json
import math
from pathlib import Path
import socket
import sys


def network_block(event, args):
    if event == "socket.connect" and args[0].family in {socket.AF_INET, socket.AF_INET6}:
        raise PermissionError("Offline replay")


def compare(actual, expected, path, errors):
    if type(actual) is not type(expected):
        raise AssertionError(f"Field type differs at {path}")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"Fields differ at {path}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}", errors)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"Length differs at {path}")
        for i,(x,y) in enumerate(zip(actual,expected)):
            compare(x,y,f"{path}[{i}]",errors)
    elif isinstance(expected,float):
        error=abs(actual-expected)
        if not math.isfinite(error) or error>1e-12:
            raise AssertionError(f"{path}: {actual!r} differs from {expected!r} by {error!r}")
        errors.append((error,path))
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} differs from {expected!r}")


if __name__=="__main__":
    sys.addaudithook(network_block)
    from trial import load_inputs, score, schedule
    folder=Path(__file__).resolve().parent / "results"
    def read(name):
        return json.loads((folder/name).read_bytes())
    for name,expected in read("provenance.json")["source_checksums"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected
    packets=load_inputs()
    receipt=read("receipts.json")
    assert receipt["git_sha"]=="6deb78f9cffbf668ba98c28329c8d190557786c1" and receipt["attempted"]==15
    for (packet, arm),row in zip(schedule(packets),receipt["rows"],strict=True):
        assert (packet["key"],arm,packet["prompt_sha256"][arm])==(row["key"],row["arm"],row["prompt_sha256"])
    computed=score(packets,receipt,read("labels.json"))
    errors=[]
    compare(computed,read("comparison.json"),"result",errors)
    print("Replayed 15 sealed decisions with identical counts, identities and gates.")
    print("Maximum absolute metric difference and field:",max(errors))
    print(json.dumps(computed,sort_keys=True))
