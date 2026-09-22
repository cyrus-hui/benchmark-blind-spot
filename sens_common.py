#!/usr/bin/env python3
"""sens_common.py -- shared by the Phase 5 sensitivities (decisions.md Sept 20 Sec 4, Sept 21 Sec 22.2).

Rebuilds the ranked 5.4 verdict table with detector_eval_5p4.py's OWN functions
(control_thresholds, build) and src/detectors.py's compare / better_of / stats, and
checks it reading by reading against the committed read log.  Nothing here defines a
detector, a threshold or a verdict rule; it only arranges calls to the code that
produced the registered verdict.  Reported, never ranked.
"""
import hashlib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))
import detectors as D            # noqa: E402
import detector_eval_5p4 as E    # noqa: E402
import thresholds_registry as R  # noqa: E402

REGISTRY_MD5 = "62d1254829994ab8b2f6dbb6b2edd047"   # decisions.md Sept 21 Sec 18
READ_LOG_MD5 = "6900f26c1ae0c73eb1a0cd249522b626"   # logs/5p4_read.log, ca4bee4
RANK = {"LOSS": 0, "TIE": 1, "CEILING": 2, "WIN": 3}  # weakest first
FAMS = ("SIGMA", "PPL")


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def gate(md5_given):
    """Same conditions as detector_eval_5p4.py's grid read."""
    reg = R.load()
    if R.pending(reg):
        sys.exit(f"REFUSED: thresholds still pending: {R.pending(reg)}")
    if md5_given != R.md5() or md5_given != REGISTRY_MD5:
        sys.exit(f"REFUSED: --authorise-grid-read {md5_given} is not the registry md5 "
                 f"{R.md5()} recorded as {REGISTRY_MD5}")
    print(f"gate OK: registry {R.md5()}, pending []")
    return reg


# ------------------------------------------------------------------ the read log

TITLE = re.compile(r"^--- (design|outcome) label, (\w+): (\d+) positive vs (\d+) negative")
SEP = re.compile(r"^\s+Delta_cal separates at the registered tau: (YES|NO)\s*$")
VS = re.compile(r"^\s+Delta_cal vs (SIGMA|PPL) \(ranked on (\w+)\): (WIN|LOSS|TIE|CEILING)\b")


def parse_read_log(text):
    raw, cur = {}, None
    for line in text.splitlines():
        m = TITLE.match(line)
        if m:
            cur = f"{m[1]}/{m[2]}"
            if cur in raw:
                sys.exit(f"STOP: reading {cur} appears twice in the read log")
            raw[cur] = {"n_pos": int(m[3])}
            continue
        if cur is None:
            continue
        if "NOT EVALUABLE" in line:
            raw[cur]["not_evaluable"] = True
        m = SEP.match(line)
        if m:
            raw[cur]["separates"] = m[1] == "YES"
        m = VS.match(line)
        if m:
            raw[cur][m[1]] = (m[2], m[3])
    if "\nRESULT OK" not in text:
        sys.exit("STOP: the read log does not end RESULT OK")
    table = {}
    for k, v in raw.items():
        if v.get("not_evaluable"):
            if set(v) & {"separates", "SIGMA", "PPL"}:
                sys.exit(f"STOP: {k} is marked NOT EVALUABLE but carries verdicts")
            table[k] = None
            continue
        miss = {"separates", "SIGMA", "PPL"} - set(v)
        if miss:
            sys.exit(f"STOP: reading {k} in the read log lacks {sorted(miss)}")
        table[k] = {"n_pos": v["n_pos"], "separates": v["separates"], "SIGMA": v["SIGMA"], "PPL": v["PPL"]}
    if not any(v is not None for v in table.values()):
        sys.exit("STOP: no evaluable reading parsed from the read log")
    return table


def load_read_log(path):
    got = md5(path)
    if got != READ_LOG_MD5:
        sys.exit(f"REFUSED: {path} md5 {got} is not the committed read log {READ_LOG_MD5}")
    return parse_read_log(pathlib.Path(path).read_text())


# ------------------------------------------------------------------ the table

def chain_sets(rd, tau):
    """{reading: (decoder, pos, neg)}, {decoder: dets} -- exactly as detector_eval_5p4.grid()."""
    sets, dets_by = {}, {}
    for dec in D.DECODERS:
        dets, collapsed = E.build(rd, dec, tau, D.RATIOS)
        ids = list(dets["dcal"]["score"])
        neg = [c for c in ids if c[0] == 0]
        exposed = [c for c in ids if c[0] > 0]
        sets[f"design/{dec}"] = (dec, exposed, neg)
        sets[f"outcome/{dec}"] = (dec, [c for c in exposed if collapsed[c]], neg)
        dets_by[dec] = dets
    return sets, dets_by


def verdicts(dets, pos, neg):
    """detector_eval_5p4.reading() without printing.  None when not evaluable."""
    if len(pos) < 3:
        return None
    sig = D.better_of({"sigma_sub": dets["sigma_sub"], "sigma_full": dets["sigma_full"]}, "sigma_sub", pos, neg)
    ppl = D.better_of({"pinc": dets["pinc"], "pslope": dets["pslope"]}, "pinc", pos, neg)
    return {"n_pos": len(pos),
            "separates": D.stats(dets["dcal"], pos, neg)["tpr"] == 1.0,
            "SIGMA": (sig, D.compare(dets["dcal"], dets[sig], pos, neg)[0]),
            "PPL": (ppl, D.compare(dets["dcal"], dets[ppl], pos, neg)[0])}


def table_of(sets, dets_by, override=None):
    """override: {decoder: {detector name: det}} substituted before ranking."""
    out = {}
    for k, (dec, pos, neg) in sets.items():
        dets = dict(dets_by[dec])
        if override:
            dets.update(override.get(dec, {}))
        out[k] = verdicts(dets, pos, neg)
    return out


def exit_of(table):
    """detector_eval_5p4.exit_verdict() without printing."""
    ev = {k: v for k, v in table.items() if v is not None}
    if not any(k.startswith("design") for k in ev):
        return {"exit": "STOP (no design reading evaluable)", "separates": None, "beats": {}}
    sep = all(v["separates"] for v in ev.values())
    beats = {}
    for fam in FAMS:
        vs = [v[fam][1] for v in ev.values()]
        beats[fam] = all(x in ("WIN", "CEILING") for x in vs) and "WIN" in vs
    return {"exit": "(i)" if sep and all(beats.values()) else "(ii)", "separates": sep, "beats": beats}


def diff_tables(a, b):
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out.append(f"{k}: present in only one table")
        elif (a[k] is None) != (b[k] is None):
            out.append(f"{k}: evaluable in one table only")
        elif a[k] is not None and a[k] != b[k]:
            out.append(f"{k}: {a[k]} vs {b[k]}")
    return out


def weakest(verdict_list):
    return min(verdict_list, key=lambda v: RANK[v])


def fmt_row(k, v):
    if v is None:
        return f"  {k:15s} not evaluable (< 3 positive chains)"
    return (f"  {k:15s} pos {v['n_pos']:2d}  separates {'YES' if v['separates'] else 'NO '}  "
            f"SIGMA {v['SIGMA'][1]:7s} ({v['SIGMA'][0]})  PPL {v['PPL'][1]:7s} ({v['PPL'][0]})")


def canary_ranked(rd, reg, log_path):
    """Recompute every tau from controls, verify against the registry, rebuild the ranked
    table and require it to equal the committed read log in every reading."""
    want = load_read_log(log_path)
    before = E.fails
    tau = E.control_thresholds(rd, D.DECODERS)
    E.verify_against_registry(tau, reg, D.DECODERS)
    if E.fails != before:
        sys.exit("STOP: a recomputed threshold does not round to its registered value")
    sets, dets_by = chain_sets(rd, tau)
    got = table_of(sets, dets_by)
    d = diff_tables(got, want)
    if d:
        sys.exit("STOP: the rebuilt ranked table differs from the read log:\n  " + "\n  ".join(d))
    print(f"CANARY PASS: ranked verdict table rebuilt, identical to {log_path} in all "
          f"{len(want)} readings ({sum(v is not None for v in want.values())} evaluable)")
    return tau, sets, dets_by, got


# ------------------------------------------------------------------ self-test

def self_test():
    n = p = 0

    def chk(name, ok):
        nonlocal n, p
        n += 1
        p += bool(ok)
        print(f"  {name:62s} {'PASS' if ok else 'FAIL'}")

    log = ("x\n--- design label, topk: 12 positive vs 3 negative chains\n"
           "  Delta_cal separates at the registered tau: YES\n"
           "  Delta_cal vs SIGMA (ranked on sigma_sub): WIN  [level: auc]  n\n"
           "  Delta_cal vs PPL (ranked on pslope): WIN  [level: mean_lat]  n\n"
           "--- outcome label, topk: 2 positive vs 3 negative chains\n"
           "  NOT EVALUABLE: fewer than 3 positive chains; set aside, counts neither way\n"
           "--- design label, greedy: 12 positive vs 3 negative chains\n"
           "  Delta_cal separates at the registered tau: YES\n"
           "  Delta_cal vs SIGMA (ranked on sigma_sub): WIN  [level: mean_lat]  n\n"
           "  Delta_cal vs PPL (ranked on pinc): CEILING  [level: -]  n\n"
           "\nRESULT OK\n")
    t = parse_read_log(log)
    chk("parse: three readings", len(t) == 3)
    chk("parse: not-evaluable reading is None", t["outcome/topk"] is None)
    chk("parse: verdict and ranked-on", t["design/topk"]["PPL"] == ("pslope", "WIN"))
    chk("parse: CEILING", t["design/greedy"]["PPL"] == ("pinc", "CEILING"))
    chk("exit (i) on WIN/CEILING", exit_of(t)["exit"] == "(i)")
    t2 = dict(t)
    t2["design/greedy"] = dict(t["design/greedy"], PPL=("pinc", "TIE"))
    chk("exit (ii) when a TIE enters", exit_of(t2)["exit"] == "(ii)" and not exit_of(t2)["beats"]["PPL"])
    t3 = {k: (None if v is None else dict(v, PPL=("pinc", "CEILING"))) for k, v in t.items()}
    chk("all-CEILING is not a win", not exit_of(t3)["beats"]["PPL"])
    chk("diff catches a changed verdict", diff_tables(t, t2) != [])
    chk("diff empty on identical tables", diff_tables(t, dict(t)) == [])
    chk("weakest ordering LOSS<TIE<CEILING<WIN", weakest(["WIN", "CEILING", "TIE"]) == "TIE"
        and weakest(["WIN", "LOSS", "TIE"]) == "LOSS")
    try:
        parse_read_log(log.replace("RESULT OK", "RESULT FAIL"))
        chk("refuses a log that is not RESULT OK", False)
    except SystemExit:
        chk("refuses a log that is not RESULT OK", True)
    try:
        parse_read_log(log.replace("  Delta_cal vs PPL (ranked on pinc): CEILING  [level: -]  n\n", ""))
        chk("refuses an incomplete reading", False)
    except SystemExit:
        chk("refuses an incomplete reading", True)
    print(f"\nsens_common self-test: {p} / {n} PASS")
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(self_test())
