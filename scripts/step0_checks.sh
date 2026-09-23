#!/bin/bash
# 7.1 step 0 pre-run checks — READ-ONLY. Closes §42 items 1 (inventory) and 2 (mixed-corpus
# path + labels), and confirms the surface the scorer assumes. Prints no score of anything.
cd ~/modelcollapse || exit 1
R=$SCRATCH/collapse
echo "== date $(date -Is)  head $(git rev-parse --short HEAD)"

echo; echo "== 1. untracked logs (§38/§42.1)"
for p in logs/surface_dump_7p1.txt logs/surface_dump_7p1b.txt logs/5p4_controls.log logs/5p4_transfer.log logs/mix53 logs/control logs/greedy_eval; do
  if [ -e "$p" ]; then
    printf "  %-30s %6s files=%-5s ignored=%s\n" "$p" "$(du -sh "$p" | cut -f1)" "$(find "$p" -type f | wc -l)" \
      "$(git check-ignore -q "$p" && echo YES || echo no)"
  else echo "  MISSING $p"; fi
done
echo "  untracked under logs/: $(git status --porcelain --untracked-files=all -- logs | grep -c '^??')"

echo; echo "== 2. mixed corpora and manifests (Sept 15 §3 says \$RUN_ROOT/mixed/seed{S}/gen{G}.jsonl)"
for t in mix50_greedy mix50_topk; do
  M=$R/$t/mix_manifest.tsv
  echo "  -- $t"; ls -d $R/$t/seed* $R/$t/mixed/seed* 2>&1 | sed 's/^/     /'
  ls $R/$t/seed43/gen0 2>&1 | sed 's/^/     gen0: /'
  if [ -f "$M" ]; then
    echo "     manifest: $(wc -l < "$M") lines, md5 $(md5sum < "$M" | cut -c1-32)"
    head -1 "$M" | tr '\t' '\n' | nl | sed 's/^/       col /'
  else echo "     NO MANIFEST at $M"; fi
  for s in 43 44 45 46; do for g in 1 2 3 4; do
    f=$R/$t/mixed/seed$s/gen$g.jsonl; [ -f "$f" ] || continue
    h=$(md5sum < "$f" | cut -c1-32)
    printf "     s%s g%s rows=%-6s md5=%s manifest=%s\n" $s $g "$(wc -l < "$f")" "$h" \
      "$(grep -q "$h" "$M" 2>/dev/null && echo MATCH || echo NOMATCH)"
  done; done
done

echo; echo "== 3. gen-0 weights: own-seed scorer == generator (§40.7)"
for s in 42 43 44 45 46; do for t in pilot_gpt2m gpt2m_greedy mix50_greedy mix50_topk; do
  f=$R/$t/seed$s/gen0/model/model.safetensors
  [ -e "$f" ] && printf "  s%s %-13s %s  -> %s\n" $s $t "$(md5sum < "$f" | cut -c1-32)" "$(readlink -f "$f" | sed "s|$R/||")"
done; done
ls $R/pilot_gpt2m/seed43/gen0/model/ | sed 's/^/  pilot s43 gen0 model dir: /'

echo; echo "== 4. inventory: synthetic_corpus.jsonl and model per gen 0-3"
for t in gpt2m_greedy pilot_gpt2m mix50_greedy mix50_topk; do for s in 43 44 45 46; do
  [ -d $R/$t/seed$s ] || continue
  line=""; for g in 0 1 2 3; do
    c=$([ -f $R/$t/seed$s/gen$g/synthetic_corpus.jsonl ] && echo C || echo -)
    m=$([ -e $R/$t/seed$s/gen$g/model ] && echo M || echo -)
    line="$line g$g:$c$m"; done
  printf "  %-13s s%s %s\n" $t $s "$line"
done; done
for g in 1 2 3 4; do f=$R/fresh_slices_gpt2m/gen${g}_real.jsonl
  printf "  fresh gen%s rows=%s md5=%s\n" $g "$(wc -l < $f)" "$(md5sum < $f | cut -c1-32)"; done

echo; echo "== 5. row format and mixed-corpus labels by membership"
python - "$R" << 'PY'
import json, os, sys
R = sys.argv[1]
def keys(p):
    with open(p) as f: r = json.loads(f.readline())
    return {k: type(v).__name__ for k, v in r.items()}
for p in ["gpt2m_greedy/seed43/gen0/synthetic_corpus.jsonl", "pilot_gpt2m/seed43/gen0/synthetic_corpus.jsonl",
          "fresh_slices_gpt2m/gen1_real.jsonl", "mix50_greedy/mixed/seed43/gen1.jsonl"]:
    fp = os.path.join(R, p); print(f"  keys {p}: {keys(fp) if os.path.exists(fp) else 'MISSING'}")
T = lambda p: [json.loads(l)["text"] for l in open(p) if l.strip()]
print("  tree          s  g  rows   real  synth  synth_in_ref  ref")
for t, pure in (("mix50_greedy", "gpt2m_greedy"), ("mix50_topk", "pilot_gpt2m")):
    for s in (43, 44, 45, 46):
        for g in (1, 2, 3, 4):
            mp = f"{R}/{t}/mixed/seed{s}/gen{g}.jsonl"
            if not os.path.exists(mp): continue
            rows, real = T(mp), set(T(f"{R}/fresh_slices_gpt2m/gen{g}_real.jsonl"))
            ref = f"{R}/{t}/seed{s}/gen{g-1}/synthetic_corpus.jsonl"
            if not os.path.exists(ref) and g == 1: ref = f"{R}/{pure}/seed{s}/gen0/synthetic_corpus.jsonl"
            syn = set(T(ref)) if os.path.exists(ref) else set()
            nr = sum(x in real for x in rows); ns = len(rows) - nr
            nin = sum(x in syn for x in rows if x not in real)
            print(f"  {t:13s} {s} {g} {len(rows):6d} {nr:6d} {ns:6d} {nin:12d}  {os.path.relpath(ref, R)}")
PY

echo; echo "== 6. existing GPU scoring header and env (for the new sbatch)"
grep '^#SBATCH' slurm/score_ood_ppl_array.sbatch 2>&1 | sed 's/^/  /'
grep -h -o '#SBATCH --gres=[^ ]*' slurm/*.sbatch | sort | uniq -c | sed 's/^/  /'
sed 's/^/  env| /' scripts/narval_env.sh
echo; echo "== END"
