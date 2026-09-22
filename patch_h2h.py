#!/usr/bin/env python3
"""Replace the NOT IMPLEMENTED stub in pplslope_sensitivity.py with the call into sens_h2h.
Guarded on the pre-patch md5; keeps pplslope_sensitivity.py.pre-h2h; the edit must match once."""
import hashlib, pathlib, shutil, sys
P = pathlib.Path("pplslope_sensitivity.py")
PRE = "c45f357308fd67ec333399cf3a71ea27"
OLD = '''        gate_grid(a.authorise_grid_read, a.ranked_verdicts)
        print("NOT IMPLEMENTED: the head-to-head half is written after src/detectors.py's "
              "surface is probed on Narval.  Thresholds are complete; nothing was read.")
        return 1
'''
NEW = '''        gate_grid(a.authorise_grid_read, a.ranked_verdicts)
        sys.path.insert(0, str(ROOT))
        import sens_h2h
        return sens_h2h.run(sys.modules[__name__], a.ranked_verdicts)
'''
got = hashlib.md5(P.read_bytes()).hexdigest()
if got != PRE:
    sys.exit(f"REFUSE: {P} md5 {got}, expected {PRE}")
bak = P.with_name(P.name + ".pre-h2h")
if bak.exists():
    sys.exit(f"REFUSE: {bak} exists")
src = P.read_text()
if src.count(OLD) != 1:
    sys.exit(f"REFUSE: the stub matches {src.count(OLD)} times, need exactly 1")
shutil.copy2(P, bak)
P.write_text(src.replace(OLD, NEW))
print(f"patched {P}: {PRE} -> {hashlib.md5(P.read_bytes()).hexdigest()}  (backup {bak})")
