import json, os, math
S = os.environ['SCRATCH']
# Arm/seeds/canary are env-injectable so the same script reads both scales.
#   1.7B:  ARM=control_smollm2 SEEDS=43,44,45,46,47 \
#          CANARY_PPL0=12.016712451198961 CANARY_ENT0=2.4907782043442617 \
#          CANARY_PPL1=12.40723786509852 python compute_control_delta.py
ARM = os.environ.get('ARM', 'control_gpt2m')
SEEDS = [int(x) for x in os.environ.get('SEEDS', '43,44,45,46').split(',')]
CANARY = (float(os.environ.get('CANARY_PPL0', 23.73410692404481)),
          float(os.environ.get('CANARY_ENT0', 3.242523688720312)),
          float(os.environ.get('CANARY_PPL1', 25.477220913947473)))

def cell(s, g):
    return json.load(open(f'{S}/collapse/{ARM}/seed{s}/gen{g}/metrics.json'))

# CANARY: values read before any entropy pass existed
c0, c1 = cell(43,0), cell(43,1)
assert abs(c0['perplexity'] - CANARY[0]) < 1e-9, c0['perplexity']
assert abs(c0['entropy']    - CANARY[1]) < 1e-9, c0['entropy']
assert abs(c1['perplexity'] - CANARY[2]) < 1e-9, c1['perplexity']
print(f'canary 3/3 OK for arm={ARM} seeds={SEEDS}\n')

dcal = lambda d: math.log(d['perplexity']) - d['entropy']
G = list(range(1, 8))
gbar = sum(G)/len(G)

rows, allv = [], []
for s in SEEDS:
    base = dcal(cell(s,0))
    ds = [dcal(cell(s,g)) - base for g in G]
    rows.append((s, ds)); allv += ds
    print(f'seed {s}  ' + '  '.join(f'{d:+.4f}' for d in ds) + f'   max {max(ds):+.4f}')

mx = max(allv)
print(f'\nmax Δ over {len(allv)} cells: {mx:+.4f} nats   (min {min(allv):+.4f})')
print('VERDICT:', 'FALSIFIED (>0.345)' if mx > 0.345 else
                 'FRAGILE [0.30,0.345]' if mx >= 0.30 else 'SURVIVES (<0.30)')

print('\npre-registered secondary — slope of Delta on g (expect ~0):')
for s, ds in rows:
    dbar = sum(ds)/len(ds)
    num = sum((g-gbar)*(d-dbar) for g, d in zip(G, ds))
    den = sum((g-gbar)**2 for g in G)
    print(f'  seed {s}  slope {num/den:+.5f} nats/gen')

print('\nmechanism — which term moves the gap:')
for s in SEEDS:
    b = cell(s,0)
    p = [math.log(cell(s,g)['perplexity']) - math.log(b['perplexity']) for g in G]
    e = [-(cell(s,g)['entropy'] - b['entropy']) for g in G]
    print(f'  seed {s}  logPPL max {max(p):+.4f} | -dEntropy max {max(e):+.4f} '
          '(entropy g1..g7 ' + ' '.join('%.4f' % cell(s, g)['entropy'] for g in G) + ')')
