#!/usr/bin/env python
"""Self-test for src/step0_score.py. CPU, offline, tiny random models only.
T1 greedy identity: greedy continuations score argmax 96/96 and S <= 0.
T2 mask/offset: vectorised S equals a per-prefix loop over positions 33..128.
T3 uniform q: S == 0.
T4 top-k derivation: mean S < 0 on top-k samples (descriptive of §39).
T5 end-to-end CLI on a synthetic tokenizer: synth/K3 PASS, exclusion of a
   non-128 row, mixed-corpus labelling (count + synth-ref subset), refusal to
   overwrite, K3 FAIL writes no scores, null kind.
"""
import json, os, subprocess, sys, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
SCORER = os.path.join(HERE, "..", "src", "step0_score.py")
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch
from transformers import GPT2Config, GPT2LMHeadModel
from step0_score import score_ids, PROMPT, BLOCK, NGEN

torch.manual_seed(0)
fails = []
def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok: fails.append(name)

def tiny(V=100, seed=0, scale=1.0):
    torch.manual_seed(seed)
    m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=V, n_positions=BLOCK,
                                      bos_token_id=None, eos_token_id=None)).eval()
    with torch.no_grad():
        m.transformer.wte.weight.mul_(scale)
    return m

m = tiny(scale=20.0)  # sharpen so greedy is non-trivial
prompts = torch.randint(0, 100, (16, PROMPT))
with torch.no_grad():
    g = m.generate(prompts, attention_mask=torch.ones_like(prompts), do_sample=False,
                   max_new_tokens=NGEN, pad_token_id=0)
S, A = score_ids(m, g, "cpu", batch=5)
check("T1 greedy argmax 96/96", bool((A == NGEN).all()), f"min={A.min()}")
check("T1 greedy S<=0", bool((S <= 1e-7).all()), f"max={S.max():.3e}")

x = torch.randint(0, 100, (3, BLOCK))
S2, _ = score_ids(m, x, "cpu")
naive = []
with torch.no_grad():
    for d in range(3):
        gs = []
        for t in range(PROMPT, BLOCK):
            lp = torch.log_softmax(m(x[d:d+1, :t]).logits[0, -1].float(), -1)
            gs.append((-lp[x[d, t]] - (-(lp.exp() * lp).sum())).item())
        naive.append(np.mean(gs))
check("T2 mask/offset vs per-prefix loop", np.allclose(S2, naive, atol=1e-4),
      f"maxdiff={np.abs(S2 - np.array(naive)).max():.2e}")

mu = tiny(scale=0.0)  # tied wte zeroed -> constant logits -> uniform q
S3, _ = score_ids(mu, x, "cpu")
check("T3 uniform q gives S=0", bool(np.abs(S3).max() < 1e-5), f"max|S|={np.abs(S3).max():.2e}")

torch.manual_seed(1)
with torch.no_grad():
    gk = m.generate(prompts, attention_mask=torch.ones_like(prompts), do_sample=True, top_k=5,
                    max_new_tokens=NGEN, pad_token_id=0)
S4, _ = score_ids(m, gk, "cpu")
check("T4 top-k mean S < 0", S4.mean() < 0, f"mean={S4.mean():.4f}")

# ---- T5: end-to-end CLI with a synthetic word-level tokenizer ----
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
V = 100
vocab = {f"w{i}": i for i in range(V)}
tk = Tokenizer(models.WordLevel(vocab, unk_token="w0"))
tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tk.decoder = decoders.WordPiece(prefix="##")  # joins with spaces
with tempfile.TemporaryDirectory() as td:
    mdir = os.path.join(td, "model"); os.makedirs(mdir)
    m.save_pretrained(mdir)
    PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(mdir)
    txt = lambda ids: " ".join(f"w{i}" for i in ids)
    synth = [txt(r.tolist()) for r in g]                       # 16 greedy docs
    real = [txt(torch.randint(0, V, (BLOCK,)).tolist()) for _ in range(16)]
    def w(name, rows):
        p = os.path.join(td, name)
        with open(p, "w") as f:
            for r in rows: f.write(json.dumps({"text": r}) + "\n")
        return p
    bad = synth[0] + " w7"                                     # 129 tokens -> excluded
    pc = w("corpus.jsonl", synth + [bad])
    out = os.path.join(td, "out")
    run = lambda *args: subprocess.run([sys.executable, SCORER, *args], capture_output=True, text=True)
    r = run("--task", "t_synth", "--corpus", pc, "--scorer", mdir, "--out-dir", out, "--kind", "synth",
            "--expect-rows", "17", "--enforce-k3")
    meta = json.load(open(os.path.join(out, "t_synth.json"))) if r.returncode == 0 else {}
    check("T5 synth K3 PASS", r.returncode == 0 and meta.get("verdict") == "PASS", r.stdout.strip() + r.stderr[-300:])
    check("T5 exclusion of 129-token row", meta.get("n_excluded") == 1 and meta.get("excluded", [{}])[0].get("len") == 129)
    z = np.load(os.path.join(out, "t_synth.npz")) if r.returncode == 0 else None
    check("T5 npz rows/S", z is not None and len(z["S"]) == 16 and (z["S"] <= 1e-7).all()
          and np.array_equal(z["row"], np.arange(16)))
    r = run("--task", "t_synth", "--corpus", pc, "--scorer", mdir, "--out-dir", out, "--kind", "synth", "--expect-rows", "17")
    check("T5 refuses overwrite", r.returncode != 0 and "never overwrite" in (r.stdout + r.stderr))
    # mixed: 8 real + 8 synth, shuffled
    mixed = real[:8] + synth[:8]; np.random.default_rng(0).shuffle(mixed)
    pm, pr, ps = w("mixed.jsonl", mixed), w("real.jsonl", real), w("synth.jsonl", synth)
    r = run("--task", "t_mix", "--corpus", pm, "--scorer", mdir, "--out-dir", out, "--kind", "synth",
            "--expect-rows", "16", "--real-ref", pr, "--synth-ref", ps, "--expect-synth", "8", "--enforce-k3")
    meta = json.load(open(os.path.join(out, "t_mix.json"))) if r.returncode == 0 else {}
    check("T5 mixed labelling", r.returncode == 0 and meta.get("n_synth_rows") == 8 and meta.get("n_real_dropped") == 8,
          r.stdout.strip() + r.stderr[-300:])
    r = run("--task", "t_mixbad", "--corpus", pm, "--scorer", mdir, "--out-dir", out, "--kind", "synth",
            "--expect-rows", "16", "--real-ref", pr, "--expect-synth", "9")
    check("T5 mixed count mismatch refused", r.returncode != 0 and "label count" in (r.stdout + r.stderr))
    r = run("--task", "t_k3fail", "--corpus", pr, "--scorer", mdir, "--out-dir", out, "--kind", "synth",
            "--expect-rows", "16", "--enforce-k3")
    check("T5 K3 FAIL on random text, no npz", r.returncode == 3 and not os.path.exists(os.path.join(out, "t_k3fail.npz"))
          and json.load(open(os.path.join(out, "t_k3fail.json")))["verdict"] == "FAIL_K3")
    r = run("--task", "t_null", "--corpus", pr, "--scorer", mdir, "--out-dir", out, "--kind", "null", "--expect-rows", "16")
    check("T5 null kind OK", r.returncode == 0 and os.path.exists(os.path.join(out, "t_null.npz")))
    r = run("--task", "t_canary", "--corpus", pc, "--scorer", mdir, "--out-dir", out, "--kind", "canary",
            "--expect-rows", "17", "--enforce-k3")
    check("T5 canary writes json only", r.returncode == 0 and not os.path.exists(os.path.join(out, "t_canary.npz")))
    r = run("--task", "t_key", "--corpus", pc, "--scorer", mdir, "--out-dir", out, "--kind", "null",
            "--expect-rows", "17", "--text-key", "nope")
    check("T5 missing text key refused", r.returncode != 0 and "REFUSE" in (r.stdout + r.stderr))

n = 17
print(f"SELF-TEST {'PASS' if not fails else 'FAIL'} {n - len(fails)}/{n}")
sys.exit(1 if fails else 0)
