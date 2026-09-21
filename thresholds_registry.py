#!/usr/bin/env python
"""Registered-threshold file for the Phase 5.4 read (registered_thresholds.json).

The file holds the ROUNDED values exactly as recorded in decisions.md. The analysis
recomputes every threshold at full precision from control arms and asserts that it
rounds to the registered value, so this file is a pre-registration, not an input.

  python thresholds_registry.py init                       # refuses if the file exists
  python thresholds_registry.py set sigma.greedy.sub 1.234 # refuses to overwrite a non-null value
  python thresholds_registry.py show                       # prints values, pending keys, and md5
"""
import hashlib, json, pathlib, sys

PATH = pathlib.Path(__file__).resolve().parent / "registered_thresholds.json"
INITIAL = {
    "dcal":    {"tau": 0.0635},
    "pinc":    {"tau": None},
    "pslope":  {"tau_g2": None, "tau_g3": None, "tau_g4": None, "tau_g5": None, "tau_g6": None, "tau_g7": None},
    "sigma":   {"topk": {"sub": 7.248, "full": 6.148}, "greedy": {"sub": None, "full": None}},
    "outcome": {"topk": {"tau_D": 0.0105, "tau_K": 0.1111}, "greedy": {"tau_D": None, "tau_K": None}},
}
DECIMALS = {"dcal": 4, "pinc": 4, "pslope": 5, "sigma": 3, "outcome": 4}


def flat(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, pre + k + ".")
        else:
            yield pre + k, v


def load():
    if not PATH.is_file():
        sys.exit(f"STOP: {PATH} does not exist; run init")
    return json.load(open(PATH))


def save(d):
    PATH.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def md5():
    return hashlib.md5(PATH.read_bytes()).hexdigest()


def pending(d):
    return [k for k, v in flat(d) if v is None]


def setkey(d, dotted, value):
    node, parts = d, dotted.split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            sys.exit(f"REFUSE: no such key {dotted}")
        node = node[p]
    leaf = parts[-1]
    if leaf not in node or isinstance(node[leaf], dict):
        sys.exit(f"REFUSE: no such key {dotted}")
    if node[leaf] is not None:
        sys.exit(f"REFUSE: {dotted} is already registered as {node[leaf]}; a registered value is never overwritten")
    node[leaf] = round(float(value), DECIMALS[parts[0]])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "init":
        if PATH.exists():
            sys.exit(f"REFUSE: {PATH} exists")
        save(INITIAL)
    elif cmd == "set":
        if len(sys.argv) != 4:
            sys.exit("usage: set <dotted.key> <value>")
        d = load()
        setkey(d, sys.argv[2], sys.argv[3])
        save(d)
    elif cmd != "show":
        sys.exit(__doc__)
    d = load()
    for k, v in flat(d):
        print(f"  {k:22s} {v}")
    p = pending(d)
    print(f"pending: {len(p)}  {p}")
    print(f"md5: {md5()}")


if __name__ == "__main__":
    main()
