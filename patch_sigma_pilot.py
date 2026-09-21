import hashlib, pathlib, shutil, sys
P = pathlib.Path("scripts/sigma_gate.py")
PRE = "a3cb8d97d179982e533a0cd972c11b64"
s = P.read_bytes()
if hashlib.md5(s).hexdigest() != PRE:
    sys.exit(f"REFUSE: pre-patch md5 is {hashlib.md5(s).hexdigest()}, expected {PRE}")
bak = P.with_suffix(".py.pre-greedypilot")
if bak.exists(): sys.exit("REFUSE: backup already exists")
t = s.decode()

EDITS = [
("def pilot(root, sroot, device):\n    cell = root / \"control_gpt2m/seed43/gen1\"",
 "def pilot(root, sroot, device, arm=\"control_gpt2m\", seed=43, gen=1, config=None):\n    cell = root / f\"{arm}/seed{seed}/gen{gen}\""),

("    cfg = yaml.safe_load(open(HERE / \"configs/control_gpt2m.yaml\"))",
 "    cfg = yaml.safe_load(open(HERE / f\"configs/{config or arm}.yaml\"))"),

("    cache = sroot / \"control_gpt2m/seed43_gen1_emb.npy\"",
 "    cache = sroot / f\"{arm}/seed{seed}_gen{gen}_emb.npy\""),

("    ap.add_argument(\"--pilot\", action=\"store_true\"); ap.add_argument(\"--all\")",
 "    ap.add_argument(\"--pilot\", action=\"store_true\"); ap.add_argument(\"--all\")\n"
 "    ap.add_argument(\"--arm\", default=\"control_gpt2m\"); ap.add_argument(\"--seed\", type=int, default=43)\n"
 "    ap.add_argument(\"--gen\", type=int, default=1); ap.add_argument(\"--config\", default=None)"),

("        pilot(root, sroot, a.device)",
 "        pilot(root, sroot, a.device, a.arm, a.seed, a.gen, a.config)"),
]

for old, new in EDITS:
    n = t.count(old)
    if n != 1: sys.exit(f"REFUSE: {n} matches for:\n{old}")
    t = t.replace(old, new)

shutil.copy2(P, bak)
P.write_text(t)
print("pre :", PRE)
print("post:", hashlib.md5(P.read_bytes()).hexdigest())
print("backup:", bak)
