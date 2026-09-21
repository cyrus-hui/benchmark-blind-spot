#!/usr/bin/env python3
"""Adds Delta-slope to pplslope_sensitivity.py as a reported-never-ranked corpus."""
import hashlib, sys
from pathlib import Path

DST = Path("pplslope_sensitivity.py")
WANT = "75042165441b6706a0664f3f8a252491"

if not DST.exists():
    sys.exit("REFUSE: %s absent" % DST)
cur = hashlib.md5(DST.read_bytes()).hexdigest()
if cur != WANT:
    sys.exit("REFUSE: %s md5 %s != %s -- file already edited; patch not applied" % (DST, cur, WANT))
if Path("sensitivity_thresholds.json").exists():
    sys.exit("REFUSE: sensitivity_thresholds.json already exists; variants are fixed once registered")

EDITS = [
(
"""PINC_MAX_BY_SEED = {43: 0.0731, 44: 0.0741, 45: 0.0731, 46: 0.0736}
PINC_TOL = 1e-4
""",
"""PINC_MAX_BY_SEED = {43: 0.0731, 44: 0.0741, 45: 0.0731, 46: 0.0736}
PINC_TOL = 1e-4
DCAL_MAX_INC = 0.0635            # tau_Delta, recorded Sept 14; max over all 28 control cells
DCAL_TOL = 1e-4
"""),
(
"""CORPORA = {
    "in": "perplexity",""",
"""# "dcal" is not a corpus: it is Delta_cal = log PPL - entropy, in-domain, carried
# through the SAME three axes as P-slope so the baseline's differencing transform
# is applied to the candidate too.  Reported, never ranked -- Sec 3 fixes the
# ranked Delta_cal detector as a level against tau_Delta = 0.0635.
CORPORA = {
    "in": "perplexity",
    "dcal": None,"""),
(
"""def control_logppl(key, inverse=False):
    return {s: logppl_chain(TAU_ARM, s, key, authorised=False, inverse=inverse)
            for s in TAU_SEEDS}
""",
"""def control_logppl(key, inverse=False):
    return {s: logppl_chain(TAU_ARM, s, key, authorised=False, inverse=inverse)
            for s in TAU_SEEDS}


def dcal_chain(arm, seed, authorised=False, inverse=False):
    \"\"\"Delta_cal(g) = log PPL(g) - entropy(g), in-domain.\"\"\"
    lp = logppl_chain(arm, seed, "perplexity", authorised=authorised)
    out = {}
    for g in GENS:
        m = read_cell(arm, seed, g, authorised)
        if "entropy" not in m:
            raise MissingKey("key 'entropy' absent at %s seed%d gen%d (%d keys present)"
                             % (arm, seed, g, len(m)))
        out[g] = lp[g] - float(m["entropy"])
    if inverse:
        out = {g: -v for g, v in out.items()}
    return out


def control_dcal(inverse=False):
    return {s: dcal_chain(TAU_ARM, s, authorised=False, inverse=inverse) for s in TAU_SEEDS}


def canary_dcal(chains):
    got = max(max(pinc_series(c).values()) for c in chains.values())
    return None if abs(got - DCAL_MAX_INC) <= DCAL_TOL else got
"""),
(
"""    for cname, key in CORPORA.items():
        try:
            chains = control_logppl(key, inverse=inverse)
        except MissingKey as e:
            print("SKIP corpus %s: %s" % (cname, e))
            continue
""",
"""    for cname, key in CORPORA.items():
        try:
            chains = (control_dcal(inverse=inverse) if cname == "dcal"
                      else control_logppl(key, inverse=inverse))
        except MissingKey as e:
            print("SKIP series %s: %s" % (cname, e))
            continue

        if cname == "dcal":
            got = canary_dcal(chains)
            if inverse:
                if got is not None:
                    print("\\nGATE STOP: inverse run broke the Delta_cal canary, as required "
                          "(%.6f vs %.4f)." % (got, DCAL_MAX_INC))
                    return 1
                print("\\nCANARY NOT WIRED: inverse run passed the Delta_cal canary.")
                return 1
            if got is not None:
                print("\\nGATE STOP: Delta_cal canary failed: %.6f vs %.4f"
                      % (got, DCAL_MAX_INC))
                return 1
            print("Delta_cal canary OK against %.4f" % DCAL_MAX_INC)
"""),
(
"""            out["variants"][vkey(v)] = {
                "ranked": bool(cname == "in" and v[1:] == REGISTERED),""",
"""            out["variants"][vkey(v)] = {
                "ranked": False,   # nothing here is ever ranked; see the header
                "mirrors_registered_pslope": bool(cname == "in" and v[1:] == REGISTERED),"""),
(
"""    reg = out["variants"].get(vkey(("in",) + REGISTERED))
    if reg is None:
        sys.exit("REFUSE: the registered variant did not compute -- refusing to write a partial file")
""",
"""    reg = out["variants"].get(vkey(("in",) + REGISTERED))
    if reg is None:
        sys.exit("REFUSE: the registered variant did not compute -- refusing to write a partial file")
    if out["variants"].get(vkey(("dcal",) + REGISTERED)) is None:
        sys.exit("REFUSE: Delta-slope did not compute -- refusing to write an asymmetric file")
"""),
(
"""    for k, rec in out["variants"].items():
        if not k.startswith("in."):
            continue""",
"""    for k, rec in out["variants"].items():
        if not (k.startswith("in.") or k.startswith("dcal.")):
            continue"""),
(
"""        print("%-30s %s%s" % (k, row, "   <- RANKED (Sec 6)" if rec["ranked"] else ""))""",
"""        print("%-30s %s%s" % (k, row,
              "   <- mirrors the ranked P-slope" if rec["mirrors_registered_pslope"] else ""))"""),
(
"""    # 8. the registered variant is in the enumeration exactly once""",
"""    # 7b. Delta-slope sees the same axes as P-slope
    chk("dcal is enumerated over all 8 axis settings",
        len(list(variant_names("dcal"))) == len(list(variant_names("in"))) == 8)
    dc = {0: 0.30}
    dc.update({g: 0.30 + 0.0635 for g in range(1, 8)})
    chk("Delta_cal canary wiring", canary_dcal({43: dc, 44: dc, 45: dc, 46: dc}) is None)
    chk("Delta_cal canary catches a wrong series",
        canary_dcal({43: {g: 0.5 * g for g in GENS}}) is not None)

    # 8. the registered variant is in the enumeration exactly once"""),
]

src = DST.read_text()
for i, (old, new) in enumerate(EDITS):
    n = src.count(old)
    if n != 1:
        sys.exit("REFUSE: edit %d matched %d times, must be exactly 1" % (i, n))
    src = src.replace(old, new)

bak = Path("pplslope_sensitivity.py.pre-dslope")
if bak.exists():
    sys.exit("REFUSE: %s exists" % bak)
bak.write_text(DST.read_text())
DST.write_text(src)
print("backup:  %s  md5 %s" % (bak, hashlib.md5(bak.read_bytes()).hexdigest()))
print("patched: %s  md5 %s" % (DST, hashlib.md5(DST.read_bytes()).hexdigest()))
print("edits applied: %d / %d" % (len(EDITS), len(EDITS)))
