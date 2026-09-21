#!/usr/bin/env python
"""Phase 5.4 detector evaluation: Delta_cal vs perplexity baseline vs SIGMA-UB.

Every rule here is pre-registered in decisions.md Sept 15 (b), Sept 17 and Sept 17 (c).
Unit of analysis: the chain. Per decoder, 12 exposed chains (r in 25/50/75/100 x seeds 43-45)
against 3 control chains (r = 0, seeds 43-45). Two labels x two decoders = four readings.

  python detector_eval_5p4.py --self-test
  python detector_eval_5p4.py                       # controls only: verifies every registered tau
  python detector_eval_5p4.py --transfer            # 1.7B arms (already read) at the 345M thresholds
  python detector_eval_5p4.py --authorise-grid-read <md5 of registered_thresholds.json>

The grid read is refused unless the registry has no pending value and its md5 matches the
one given on the command line (the md5 recorded in decisions.md).
"""
import argparse, os, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))
import detectors as D            # noqa: E402
import thresholds_registry as R  # noqa: E402

WKEYS, CKEYS = ["perplexity", "entropy"], ["distinct2", "kl_unigram"]
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ------------------------------------------------------------------ thresholds from controls

def control_thresholds(rd, decoders):
    """Full-precision thresholds, recomputed from control arms only."""
    arm, seeds = D.WEIGHTS_TAU
    wm = [rd.metrics(arm, s, WKEYS) for s in seeds]
    tau = {"dcal": D.tau_single([D.dcal_series(m) for m in wm]),
           "pinc": D.tau_single([D.pinc_series(m) for m in wm]),
           "pslope": D.tau_window([D.pslope_series(m) for m in wm]),
           "sigma": {}, "outcome": {}}
    for dec in decoders:
        arm, seeds = D.COMPLETIONS_TAU[dec]
        recs = [rd.sigma(arm, s) for s in seeds]
        tau["sigma"][dec] = {"sub": D.tau_single([D.sigma_series(r, "dU_sub") for r in recs]),
                             "full": D.tau_single([D.sigma_series(r, "dU_full") for r in recs])}
        cm = [rd.metrics(arm, s, CKEYS) for s in seeds]
        tau["outcome"][dec] = {
            "tau_D": max(abs(m[g]["distinct2"] - m[0]["distinct2"]) for m in cm for g in range(1, 8)),
            "tau_K": max(abs(m[g]["kl_unigram"] - m[0]["kl_unigram"]) for m in cm for g in range(1, 8))}
    return tau


def verify_against_registry(tau, reg, decoders):
    def one(label, exact, registered, nd):
        if registered is None:
            print(f"PEND  {label}: recomputed {exact:+.{nd + 2}f}; nothing registered yet")
            return
        check(f"tau {label}", round(exact, nd) == registered, f"recomputed {exact:+.{nd + 2}f} vs registered {registered}")
    one("dcal", tau["dcal"], reg["dcal"]["tau"], 4)
    one("pinc", tau["pinc"], reg["pinc"]["tau"], 4)
    for g in range(2, 8):
        one(f"pslope g{g}", tau["pslope"][g], reg["pslope"][f"tau_g{g}"], 5)
    for dec in decoders:
        for k in ("sub", "full"):
            one(f"sigma {dec} {k}", tau["sigma"][dec][k], reg["sigma"][dec][k], 3)
        for k in ("tau_D", "tau_K"):
            one(f"outcome {dec} {k}", tau["outcome"][dec][k], reg["outcome"][dec][k], 4)


# ------------------------------------------------------------------ one decoder

def build(rd, dec, tau, ratios):
    """Returns detectors {name: {"score": {cid: s7}, "lat": {cid: L}}}, collapsed {cid: bool}."""
    dets = {n: {"score": {}, "lat": {}} for n in ("dcal", "pinc", "pslope", "sigma_sub", "sigma_full")}
    collapsed = {}
    for r in ratios:
        for s in D.EVAL_SEEDS:
            cid = (r, s)
            wm = rd.metrics(D.WEIGHTS_ARM[dec][r], s, WKEYS)
            cm = rd.metrics(D.COMPLETIONS_ARM[dec][r], s, CKEYS)
            rec = rd.sigma(D.COMPLETIONS_ARM[dec][r], s)
            series = {"dcal": (D.dcal_series(wm), tau["dcal"]),
                      "pinc": (D.pinc_series(wm), tau["pinc"]),
                      "pslope": (D.pslope_series(wm), tau["pslope"]),
                      "sigma_sub": (D.sigma_series(rec, "dU_sub"), tau["sigma"][dec]["sub"]),
                      "sigma_full": (D.sigma_series(rec, "dU_full"), tau["sigma"][dec]["full"])}
            for n, (ser, t) in series.items():
                dets[n]["score"][cid] = ser[7]
                dets[n]["lat"][cid] = D.first_flag(ser, t)
            o = tau["outcome"][dec]
            collapsed[cid] = (cm[7]["distinct2"] - cm[0]["distinct2"] < -o["tau_D"]
                              and cm[7]["kl_unigram"] - cm[0]["kl_unigram"] > o["tau_K"])
    return dets, collapsed


def fmt_stats(n, st, npos):
    return (f"  {n:11s} AUC {st['auc']:.4f}  mean latency {st['mean_lat']:.3f}  "
            f"TPR@tau {st['n_flag']}/{npos}  FPR@TPR=1 {st['fpr_at_tpr1']:.3f} (resolution 1/3)  "
            f"controls flagged at tau {st['neg_flagged']}/3 (0 by construction)")


def reading(dets, pos, neg, title):
    """One label x decoder reading. Returns dict of verdicts, or None when not evaluable."""
    print(f"\n--- {title}: {len(pos)} positive vs {len(neg)} negative chains")
    if len(pos) < 3:
        print("  NOT EVALUABLE: fewer than 3 positive chains; set aside, counts neither way")
        return None
    for n, d in dets.items():
        print(fmt_stats(n, D.stats(d, pos, neg), len(pos)))
    sig = D.better_of({"sigma_sub": dets["sigma_sub"], "sigma_full": dets["sigma_full"]}, "sigma_sub", pos, neg)
    ppl = D.better_of({"pinc": dets["pinc"], "pslope": dets["pslope"]}, "pinc", pos, neg)
    out = {"separates": D.stats(dets["dcal"], pos, neg)["tpr"] == 1.0}
    print(f"  Delta_cal separates at the registered tau: {'YES' if out['separates'] else 'NO'}")
    for fam, best in (("SIGMA", sig), ("PPL", ppl)):
        v, lvl, note = D.compare(dets["dcal"], dets[best], pos, neg)
        out[fam] = v
        print(f"  Delta_cal vs {fam} (ranked on {best}): {v}  [level: {lvl}]  {note}")
    return out


def exit_verdict(readings):
    ev = {k: v for k, v in readings.items() if v is not None}
    print("\n=== Phase 5 exit criterion, over the evaluable readings:", ", ".join(ev) or "none")
    if not any(k.startswith("design") for k in ev):
        print("STOP: no design-label reading is evaluable"); return
    sep = all(v["separates"] for v in ev.values())
    print(f"separates in every reading: {sep}")
    beats = {}
    for fam in ("SIGMA", "PPL"):
        vs = [v[fam] for v in ev.values()]
        beats[fam] = all(x in ("WIN", "CEILING") for x in vs) and "WIN" in vs
        print(f"beats {fam}: {beats[fam]}   readings: {vs}")
    print("EXIT", "(i) separates and beats both baselines" if sep and all(beats.values())
          else "(ii) criterion not met; weaker verdict is the outcome")
    print("Every FPR above is 0 at the registered tau BY CONSTRUCTION (tau is the control maximum).")
    print("The only out-of-sample false-positive evidence is cross-scale: run --transfer.")


def grid(rd, reg):
    tau = control_thresholds(rd, D.DECODERS)
    verify_against_registry(tau, reg, D.DECODERS)
    if fails:
        sys.exit("STOP: a recomputed threshold does not round to its registered value")
    readings, pooled = {}, {"score": {}, "lat": {}}
    for dec in D.DECODERS:
        dets, collapsed = build(rd, dec, tau, D.RATIOS)
        ids = list(dets["dcal"]["score"])
        neg = [c for c in ids if c[0] == 0]
        exposed = [c for c in ids if c[0] > 0]
        check(f"{dec}: no control chain is outcome-collapsed (by construction)", not any(collapsed[c] for c in neg))
        print(f"\n##### decoder = {dec}")
        print("outcome label, collapsed chains by r: " + "  ".join(
            f"r={r}: {sum(collapsed[(r, s)] for s in D.EVAL_SEEDS)}/3" for r in D.RATIOS if r))
        for n in dets:
            print(f"latency {n:11s} " + "  ".join(
                f"r={r}:" + ",".join(str(dets[n]['lat'][(r, s)]) for s in D.EVAL_SEEDS) for r in D.RATIOS))
        readings[f"design/{dec}"] = reading(dets, exposed, neg, f"design label, {dec}")
        pos_o = [c for c in exposed if collapsed[c]]
        aside = [c for c in exposed if not collapsed[c]]
        readings[f"outcome/{dec}"] = reading(dets, pos_o, neg, f"outcome label, {dec}")
        print(f"  exposed-not-collapsed chains (set aside, own row): {len(aside)}")
        for n in dets:
            print(f"    {n:11s} flags {sum(dets[n]['lat'][c] <= 7 for c in aside)}/{len(aside)}")
        for c in ids:
            if c[0] > 0 or dec == "topk":               # r = 0 counted ONCE in the pooled ROC
                pooled["score"][(dec,) + c] = dets["dcal"]["score"][c]
                pooled["lat"][(dec,) + c] = dets["dcal"]["lat"][c]
    pp = [c for c in pooled["score"] if c[1] > 0]
    pn = [c for c in pooled["score"] if c[1] == 0]
    st = D.stats(pooled, pp, pn)
    print(f"\nDelta_cal pooled ROC as pre-registered Sept 15 (b): {len(pp)} exposed vs {len(pn)} controls, "
          f"AUC {st['auc']:.4f}, TPR@tau {st['n_flag']}/{len(pp)}")
    exit_verdict(readings)


def controls_only(rd, reg):
    have = [d for d in D.DECODERS if (rd.root / "sigma" / D.COMPLETIONS_TAU[d][0]).is_dir()
            and (rd.root / D.COMPLETIONS_TAU[d][0] / "seed43" / "gen7" / "metrics.json").is_file()]
    print("decoders with control completions and SIGMA records on disk:", have)
    tau = control_thresholds(rd, have)
    verify_against_registry(tau, reg, have)
    print("arms opened:", sorted(rd.opened))
    check("only control arms were opened", all(D.is_control(a) for a in rd.opened))


def transfer(root, reg):
    """345M thresholds, unchanged, on the 1.7B arms. All four arms below were read on Sept 12-15."""
    arms = {"control_smollm2": [43, 44, 45, 46, 47], "smollm2_1.7b": [43, 44, 45, 46, 47], "smollm2_greedy": [43, 44, 45]}
    rd = D.Reader(root, allow_exposed=frozenset(arms))
    t345 = control_thresholds(D.Reader(root), [])
    verify_against_registry(t345, reg, [])
    print("\nfirst flagged generation at the 345M threshold (8 = not flagged by gen 7)")
    for arm, seeds in arms.items():
        for n, fn in (("dcal", D.dcal_series), ("pinc", D.pinc_series), ("pslope", D.pslope_series)):
            lat = [D.first_flag(fn(rd.metrics(arm, s, WKEYS)), t345[n]) for s in seeds]
            print(f"  {arm:16s} {n:7s} " + " ".join(str(x) for x in lat))
    print("SIGMA is not evaluated at 1.7B (n_k = 2000 there; the registered index set is 500 of 1000).")


# ------------------------------------------------------------------ self-test

def self_test():
    import json, tempfile
    # pure functions
    check("auc perfect", D.auc([3, 4], [1, 2]) == 1.0)
    check("auc ties 0.5", D.auc([1], [1]) == 0.5)
    check("auc one reversal of 36", abs(D.auc([5] * 11 + [0.5], [1, 1, 1]) - 33 / 36) < 1e-12)
    check("ols slope", abs(D.ols_slope([1, 2, 3], [2.0, 4.0, 6.0]) - 2.0) < 1e-12)
    check("first_flag single", D.first_flag([0, .1, .2, .9, 0, 0, 0, 0], 0.5) == 3)
    check("first_flag censored", D.first_flag([0] * 8, 0.5) == D.CENSOR)
    check("first_flag window skips None", D.first_flag([None, None, 1, 1, 1, 1, 1, 1], {g: 0.5 for g in range(2, 8)}) == 2)
    pos, neg = [(r, s) for r in (25, 50, 75, 100) for s in (43, 44, 45)], [(0, s) for s in (43, 44, 45)]

    def det(score_pos, lat_pos, score_neg=0.0):
        d = {"score": {c: score_neg for c in neg}, "lat": {c: D.CENSOR for c in neg}}
        for c in pos:
            d["score"][c] = score_pos(c); d["lat"][c] = lat_pos(c)
        return d
    a = det(lambda c: 1.0, lambda c: 1)
    b = det(lambda c: 1.0, lambda c: 1)
    check("compare ceiling", D.compare(a, b, pos, neg)[0] == "CEILING")
    b = det(lambda c: 1.0, lambda c: 3 if c[0] == 25 else 1)
    check("compare latency win, stable", D.compare(a, b, pos, neg)[:2] == ("WIN", "mean_lat"))
    check("compare is antisymmetric", D.compare(b, a, pos, neg)[0] == "LOSS")
    b = det(lambda c: 1.0, lambda c: 2 if c == (25, 43) else 1)
    v = D.compare(a, b, pos, neg)
    check("one-chain latency edge is NOT a win", v[0] == "TIE", v[2])
    b = det(lambda c: -1.0 if c == (25, 43) else 1.0, lambda c: 1)
    v = D.compare(a, b, pos, neg)
    check("one-chain AUC edge is NOT a win", v[0] == "TIE", v[2])
    b = det(lambda c: -1.0 if c[0] == 25 else 1.0, lambda c: 1)
    check("three-chain AUC edge is a win", D.compare(a, b, pos, neg)[:2] == ("WIN", "auc"))
    check("better_of default on tie", D.better_of({"x": a, "y": det(lambda c: 1.0, lambda c: 1)}, "x", pos, neg) == "x")
    check("better_of picks the better", D.better_of({"x": b, "y": a}, "x", pos, neg) == "y")

    # fake tree: loader, guard, thresholds, end-to-end verdict
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        def write(arm, s, lp, ent, d2, kl, du):
            for g in D.GENS:
                p = root / arm / f"seed{s}" / f"gen{g}"
                p.mkdir(parents=True, exist_ok=True)
                import math
                json.dump({"perplexity": math.exp(3.0 + lp[g]), "entropy": 3.0 + ent[g],
                           "distinct2": 0.5 + d2[g], "kl_unigram": 1.8 + kl[g]}, open(p / "metrics.json", "w"))
            p = root / "sigma" / arm
            p.mkdir(parents=True, exist_ok=True)
            json.dump({"arm": arm, "seed": s, "n_k": 1000,
                       "gens": {str(g): {"dU_sub": du[g], "dU_full": du[g] / 2} for g in D.GENS}},
                      open(p / f"seed{s}.json", "w"))
        z = [0.0] * 8
        step = [0.0] + [0.07] * 7                                  # control: a step, no trend
        for s in (43, 44, 45, 46):
            write("control_gpt2m", s, step, [0.0] + [0.01] * 7, z, z, [0, 1, -1, 1, -1, 1, -1, 1])
        for s in (43, 44, 45):
            write("control_gpt2m_greedyeval", s, step, [0.0] + [0.01] * 7, z, z, [0, 1, -1, 1, -1, 1, -1, 1])
        for dec in D.DECODERS:
            for r in (25, 50, 75, 100):
                f = r / 100
                lp = [0.0] + [0.07 + 0.002 * f * g for g in range(1, 8)]          # PPL: slow trend
                ent = [0.0] + [0.01 - 0.5 * f] * 7                                # entropy falls at gen 1
                d2 = [0.0] + ([-0.05] * 7 if r >= 75 else [0.0] * 7)              # only r >= 75 outcome-collapsed
                kl = [0.0] + ([0.5] * 7 if r >= 75 else [0.0] * 7)
                du = [0, 1, -1, 1, -1, 1, -1, 1]                                  # SIGMA: control-like noise
                for s in D.EVAL_SEEDS:
                    write(D.WEIGHTS_ARM[dec][r], s, lp, ent, d2, kl, du)
        try:
            D.Reader(root).metrics("mix25_topk", 43, WKEYS)
            check("guard refuses an exposed arm", False)
        except SystemExit as e:
            check("guard refuses an exposed arm", "REFUSED" in str(e))
        rd = D.Reader(root, allow_exposed=True)
        tau = control_thresholds(rd, D.DECODERS)
        check("tau dcal = 0.07 - 0.01", abs(tau["dcal"] - 0.06) < 1e-9, f"{tau['dcal']:.6f}")
        check("tau pinc = 0.07", abs(tau["pinc"] - 0.07) < 1e-9)
        check("tau pslope = 0 on a step", all(abs(tau["pslope"][g]) < 1e-12 for g in range(2, 8)))
        check("tau sigma sub = 1", tau["sigma"]["topk"]["sub"] == 1.0)
        dets, col = build(rd, "topk", tau, D.RATIOS)
        ids = list(dets["dcal"]["score"]); ng = [c for c in ids if c[0] == 0]; ex = [c for c in ids if c[0] > 0]
        check("dcal latency 1 on all exposed", all(dets["dcal"]["lat"][c] == 1 for c in ex))
        check("pinc latency 1 on all exposed", all(dets["pinc"]["lat"][c] == 1 for c in ex))
        check("pslope latency 2 on all exposed", all(dets["pslope"]["lat"][c] == 2 for c in ex))
        check("sigma never flags", all(dets["sigma_sub"]["lat"][c] == D.CENSOR for c in ex))
        check("outcome label: 6 of 12 collapsed", sum(col[c] for c in ex) == 6)
        check("dcal vs pinc at ceiling", D.compare(dets["dcal"], dets["pinc"], ex, ng)[0] == "CEILING")
        check("dcal beats sigma on AUC", D.compare(dets["dcal"], dets["sigma_sub"], ex, ng)[:2] == ("WIN", "auc"))
        check("better-of PPL is pinc", D.better_of({"pinc": dets["pinc"], "pslope": dets["pslope"]}, "pinc", ex, ng) == "pinc")
    print("SELF-TEST", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCH/collapse"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--authorise-grid-read", metavar="MD5")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    reg = R.load()
    print(f"registry md5 {R.md5()}  pending {R.pending(reg)}")
    if a.transfer:
        transfer(a.root, reg)
    elif a.authorise_grid_read:
        if R.pending(reg):
            sys.exit(f"REFUSED: thresholds still pending: {R.pending(reg)}")
        if a.authorise_grid_read != R.md5():
            sys.exit("REFUSED: the md5 given is not the registry's md5; use the one recorded in decisions.md")
        grid(D.Reader(a.root, allow_exposed=True), reg)
    else:
        controls_only(D.Reader(a.root), reg)
    print("RESULT", "OK" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
