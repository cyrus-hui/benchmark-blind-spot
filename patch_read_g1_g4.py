#!/usr/bin/env python
"""Sept 21: patch scripts/read_g1_g4.py before the grid read.
1. TAU_DCAL_PATH outcome.topk.tau_D (Distinct-2 label, 0.0105) -> dcal.tau (0.0635).
2. classify(): Delta(7)[100] <= 0 returns 'undefined', never 'saturating'.
3. canary(): tau_Delta recomputed from control_gpt2m s43-46 gens 1-7 with this script's
   own delta_series; refuses unless within 5e-5 of the registered value; detection then
   runs on the full-precision value, as detector_eval_5p4.py does.
4. --canary: runs 3 on control_gpt2m only; no authorisation, no grid arm opened.
Guarded: target byte-identical to the .pre-taufix backup; every edit matches exactly once;
compiled before it replaces the original.
"""
import hashlib, py_compile, sys
from pathlib import Path

F = Path("scripts/read_g1_g4.py")
B = Path("scripts/read_g1_g4.py.pre-taufix")
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()

if not B.is_file():      sys.exit(f"REFUSE: {B} missing")
if md5(F) != md5(B):     sys.exit(f"REFUSE: {F} is not byte-identical to {B}")
src = F.read_text()
if "def canary(" in src: sys.exit("REFUSE: already patched")

CANARY = '''CONTROL_SEEDS = (43, 44, 45, 46)

def canary(registered):
    """tau_Delta from control_gpt2m only, before any grid arm is opened."""
    own = max(delta_series(0, "topk", s)[g] for s in CONTROL_SEEDS for g in GENS)
    ok = abs(own - registered) <= 5e-5
    print(("PASS  " if ok else "REFUSE  ") +
          f"tau_Delta recomputed {own:+.6f} vs registered {registered}")
    if not ok: sys.exit(1)
    return own

def main():'''

EDITS = [
    ('TAU_DCAL_PATH = "outcome.topk.tau_D"',
     'TAU_DCAL_PATH = "dcal.tau"'),
    ("# <-- REPLACE with the registry's tau_Delta path",
     "# verified against control_gpt2m by canary()"),
    ('if d7[25] >= SAT * d7[100]: return "saturating"',
     'if d7[100] <= 0:            return "undefined (r100 <= 0)"\n'
     '    if d7[25] >= SAT * d7[100]: return "saturating"'),
    ('ap.add_argument("--plan", action="store_true")',
     'ap.add_argument("--plan", action="store_true")\n'
     '    ap.add_argument("--canary", action="store_true")'),
    ('if not a.authorise_grid_read: sys.exit("REFUSE: --authorise-grid-read required")',
     'if a.canary:\n'
     '        canary(dig(json.loads(REG.read_bytes()), TAU_DCAL_PATH))\n'
     '        print("canary: control_gpt2m only. NO GRID VALUE READ.")\n'
     '        return\n'
     '    if not a.authorise_grid_read: sys.exit("REFUSE: --authorise-grid-read required")'),
    ('tau = dig(authorise(a.authorise_grid_read), TAU_DCAL_PATH)\n    print(f"tau_Delta = {tau}")',
     'tau = canary(dig(authorise(a.authorise_grid_read), TAU_DCAL_PATH))\n'
     '    print(f"tau_Delta = {tau:+.6f}  (full precision)")'),
    ('def main():', CANARY),
]

for old, new in EDITS:
    n = src.count(old)
    if n != 1: sys.exit(f"REFUSE: {n} matches for {old[:60]!r}; nothing written")
    src = src.replace(old, new)

before = md5(F)
tmp = F.with_suffix(".py.tmp")
tmp.write_text(src)
py_compile.compile(str(tmp), doraise=True)
tmp.replace(F)
print(f"patched {F}: {before} -> {md5(F)}  ({len(EDITS)} edits)")
