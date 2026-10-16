"""Train the 25 -> 73 -> 1 model with a verified concept for EVERY neuron.

python llm/scripts/train_all_features.py
python llm/scripts/train_all_features.py --verify-only

All allowed input connections, biases and output weights are optimized with
Adam. Unspecified input connections are fixed at zero; weight signs are
projected after each update. Initialization is not presented as training.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import hashlib
import json
import time
import numpy as np
import train_composition as base

ROOT = base.ROOT
ARTIFACT = ROOT / "all-features-model.json"
CHECKPOINT = ROOT / ".all-features-checkpoint.npz"
SOURCE = ROOT / "composition-model.json"


def definitions():
    local, examples = base.definitions()
    for feature in local:
        feature["group"] = "local"
    skeletons, defects, pairs = [], [], []
    for size in range(2, 6):
        for row in range(6 - size):
            for col in range(6 - size):
                k = len(pairs)
                boundary = sorted(set(sum(base.square(size, row, col), [])))
                corner = (row + size - 1) * 5 + col + size - 1
                on = [p for p in boundary if p != corner]
                off = [p for p in range(25) if p not in boundary]
                name = f"{size}×{size} · {row + 1}行{col + 1}列"
                skeletons.append(dict(neuronIndex=13+k, kind="skeleton", group="skeleton", candidateIndex=k,
                    pairIndex=k, pairedNeuronIndex=43+k, name=f"待补角轮廓 · {name}",
                    position=f"右下角（{corner // 5 + 1}行{corner % 5 + 1}列）不限定；其余轮廓须亮，内外须空", on=on, off=off))
                defects.append(dict(neuronIndex=43+k, kind="defect", group="defect", candidateIndex=k,
                    pairIndex=k, pairedNeuronIndex=13+k, name=f"缺角检查 · {name}",
                    position=f"右下角（{corner // 5 + 1}行{corner % 5 + 1}列）须空；其余轮廓须亮，内外须空", on=on, off=sorted(off+[corner])))
                pairs.append(dict(name=name, candidateIndex=k, skeletonIndex=13+k, defectIndex=43+k,
                                  cornerPixel=corner, boundary=boundary, size=size, row=row, col=col))
    return local+skeletons+defects, examples, pairs


FEATURES, EXAMPLES, PAIRS = definitions()
COUNT = len(FEATURES)
SIGN = np.zeros((25, COUNT))
for feature in FEATURES:
    SIGN[feature["on"], feature["neuronIndex"]] = 1
    SIGN[feature["off"], feature["neuronIndex"]] = -1
MASK = SIGN != 0
MATCH_X = np.array([[float(p in f["on"]) for p in range(25)] for f in FEATURES])
DEFECT_X = MATCH_X[43:]
OUTPUT_SIGN = np.r_[np.ones(43), -np.ones(30)]


def model_of(params):
    w, b, v, c = [np.round(p, 6) for p in params]
    return dict(topology=[25, COUNT, 1], layers=[dict(weights=w.T.tolist(), biases=b.tolist()),
                                               dict(weights=[v.tolist()], biases=c.tolist())])


def project(params):
    w, b, v, c = params
    w[:] = SIGN * np.clip(w * SIGN, .05, 10)
    np.clip(b, -100, 100, out=b)
    v[:] = OUTPUT_SIGN * np.clip(v * OUTPUT_SIGN, .05, 10)
    np.clip(c, -100, 100, out=c)


def verify_concepts(params):
    w, b, v, c = params
    h, _ = base.forward(MATCH_X, params)
    results = []
    for f in FEATURES:
        i = f["neuronIndex"]
        xp, xn = base.worst_inputs(w[:, i], f)
        positive_min = float(np.tanh(xp @ w[:, i] + b[i]))
        negative_max = float(np.tanh(np.max(xn @ w[:, i] + b[i])))
        signs_ok = np.all(w[MASK[:, i], i] * SIGN[MASK[:, i], i] > 0) and np.all(w[~MASK[:, i], i] == 0)
        term = float(h[i, i] * v[i])
        output_ok = v[i] * OUTPUT_SIGN[i] >= .05 and term * OUTPUT_SIGN[i] >= .01
        results.append(dict(neuronIndex=i, positiveMin=positive_min, negativeMax=negative_max,
            matchingInputId=sum(1 << p for p in f["on"]), matchingContribution=term,
            inputConstraintsPassed=bool(signs_ok), outputDirectionPassed=bool(output_ok),
            passed=bool(positive_min > 0 and negative_max < 0 and signs_ok and output_ok)))
    return dict(method="exact linear extrema for every satisfying/violating pattern, including all backgrounds",
                passed=all(f["passed"] for f in results), features=results)


def verify_pairs(params):
    full_h, full_y = base.forward(base.POS_X, params)
    gap_h, gap_y = base.forward(DEFECT_X, params)
    rows = []
    for k, pair in enumerate(PAIRS):
        a, d = pair["skeletonIndex"], pair["defectIndex"]
        term = float(gap_h[k, d] * params[2][d])
        rows.append(dict(candidateIndex=k, fullScore=float(full_y[k]), missingScore=float(gap_y[k]),
            defectContribution=term, missingScoreWithoutDefect=float(gap_y[k]-term),
            passed=bool(full_h[k, a] > 0 and full_h[k, d] < 0 and gap_h[k, a] > 0 and gap_h[k, d] > 0
                        and full_y[k] > 1 and gap_y[k] <= 1 and term < -.01)))
    return dict(passed=all(r["passed"] for r in rows), pairs=rows)


def initialize(rng):
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    old = base.arrays(source["model"])
    w = SIGN * 6.
    b = np.array([6 * (.5 - len(f["on"])) for f in FEATURES])
    w[:, :13], b[:13] = old[0][:, :13], old[1][:13]
    # The output head is learned; it is not copied from the analytic feasibility
    # construction. Different deterministic random starting magnitudes per unit.
    v = OUTPUT_SIGN * rng.uniform(.06, .16, COUNT)
    v[:13] = np.maximum(old[2][:13], .16)
    params = [w, b, v, np.array([0.])]
    project(params)
    return source, params


def train(args):
    rng = np.random.default_rng(args.seed)
    source, params = initialize(rng)
    initial = [p.copy() for p in params]
    moments = [np.zeros_like(p) for p in params]
    variances = [np.zeros_like(p) for p in params]
    total_steps, start_round, history, hard_ids = 0, 0, [], []
    if args.resume and CHECKPOINT.exists():
        ck = np.load(CHECKPOINT, allow_pickle=False)
        if str(ck["sourceHash"]) != source["modelSha256"]:
            raise ValueError("Checkpoint initialization source differs from current source model")
        params = [ck[f"p{i}"] for i in range(4)]
        moments = [ck[f"m{i}"] for i in range(4)]
        variances = [ck[f"v{i}"] for i in range(4)]
        total_steps, start_round = int(ck["steps"]), int(ck["round"])
        hard_ids = ck["hard"].tolist()
        history = json.loads(str(ck["history"]))
        rng.bit_generator.state = json.loads(str(ck["rng"]))
    negative_ids = base.structured_ids()
    negative_x = base.pixels(negative_ids)
    start = time.time()
    for round_index in range(start_round, args.rounds):
        hard_x = base.pixels(hard_ids) if hard_ids else None
        for step in range(args.steps):
            random_ids = rng.integers(1 << 25, size=128, dtype=np.uint32)
            random_ids = random_ids[~np.isin(random_ids, base.POS_IDS)]
            parts = [base.POS_X[rng.integers(30, size=128)],
                     DEFECT_X[rng.integers(30, size=64)],
                     negative_x[rng.integers(len(negative_x), size=192)], base.pixels(random_ids)]
            if hard_x is not None:
                parts.append(hard_x[rng.integers(len(hard_x), size=128)])
            x = np.concatenate(parts)
            w, b, v, c = params
            h, y = base.forward(x, params)
            dy = np.maximum(y - .4, 0)
            dy[:128] = np.minimum(y[:128] - 1.6, 0)
            class_loss = float(np.mean(dy ** 2))
            dy *= 2 / len(x)
            dh = dy[:, None] * v
            for f in FEATURES:
                i = f["neuronIndex"]
                present = np.all(x[:, f["on"]] == 1, axis=1) & np.all(x[:, f["off"]] == 0, axis=1)
                magnitude = .9 if i < 13 else .995
                target = np.where(present, magnitude, -magnitude)
                dh[:, i] += .4 * (h[:, i] - target) / len(x)
            dz = dh * (1 - h*h)
            grads = [x.T @ dz, dz.sum(axis=0), h.T @ dy, np.array([dy.sum()])]
            robust_loss = 0.
            for f in FEATURES:
                i = f["neuronIndex"]
                xp, xn = base.worst_inputs(w[:, i], f)
                margin = .55 if i < 13 else 2.5
                ep = min(float(xp @ w[:, i] + b[i]) - margin, 0.)
                en = np.maximum(xn @ w[:, i] + b[i] + margin, 0.)
                robust_loss += ep*ep + float(en @ en)
                grads[0][:, i] += 2 * (ep*xp + en @ xn)
                grads[1][i] += 2 * (ep + en.sum())
            eh, _ = base.forward(base.EXAMPLE_X, params)
            for k, e in enumerate(EXAMPLES):
                indices = e["featureIndices"]
                derivative = 8 * np.minimum(eh[k, indices] * v[indices] - .14, 0)
                grads[2][indices] += derivative * eh[k, indices]
                local = derivative * v[indices] * (1 - eh[k, indices]**2)
                grads[0][:, indices] += base.EXAMPLE_X[k, :, None] * local
                grads[1][indices] += local
            grads[0] *= MASK
            total_steps += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                moments[i] = .9*moments[i] + .1*g
                variances[i] = .999*variances[i] + .001*g*g
                p -= args.learning_rate*(moments[i]/(1-.9**total_steps))/(np.sqrt(variances[i]/(1-.999**total_steps))+1e-8)
            project(params)
            if (step+1) % 1000 == 0:
                print(f"round={round_index+1} step={step+1} class={class_loss:.6f} robust={robust_loss:.5f} elapsed={time.time()-start:.1f}s", flush=True)
        model = model_of(params)
        rounded = base.arrays(model)
        fv, cv, pv = verify_concepts(rounded), base.verify_compositions(rounded), verify_pairs(rounded)
        print("concepts", fv["passed"], "composition", cv["passed"], "pairs", pv["passed"], flush=True)
        verification, mined = base.exhaustive(rounded)
        hard_ids = sorted(set(hard_ids+mined))
        history.append(dict(round=round_index+1, classificationErrors=verification["errors"],
                            conceptsPassed=fv["passed"], compositionPassed=cv["passed"], pairsPassed=pv["passed"], hardNegativeCount=len(hard_ids)))
        np.savez(CHECKPOINT, **{f"p{i}": p for i,p in enumerate(params)},
                 **{f"m{i}": p for i,p in enumerate(moments)}, **{f"v{i}": p for i,p in enumerate(variances)},
                 steps=total_steps, round=round_index+1, hard=np.array(hard_ids,dtype=np.uint32),
                 sourceHash=source["modelSha256"], history=json.dumps(history), rng=json.dumps(rng.bit_generator.state))
        if verification["errors"] == 0 and fv["passed"] and cv["passed"] and pv["passed"]:
            trainable = int(MASK.sum()+COUNT*2+1)
            artifact = dict(model=model, featureDefinitions=FEATURES, compositionExamples=EXAMPLES, featurePairs=PAIRS,
                training=dict(seed=args.seed, steps=total_steps, stepsPerRound=args.steps, optimizer="Adam", learningRate=args.learning_rate,
                    betas=[.9,.999], epsilon=1e-8, precision=6, weightBounds=[-10,10], biasBounds=[-100,100],
                    trainableParameters=trainable, fixedZeroConnections=int((~MASK).sum()), totalParameters=COUNT*27+1,
                    sourceModelSha256=source["modelSha256"], source="first 13 concepts from composition-v1; structured initialization for 60 new detectors; learned random output head",
                    constraints="on input weights > 0; off input weights < 0; any input weights = 0; local/skeleton output weights >= 0.05; defect output weights <= -0.05",
                    loss="classification margins 1.6/0.4; concept MSE 0.2; exact logit margins +/-0.55 (local), +/-2.5 (structural); edge contribution margin 0.14",
                    parameterChangeL2=[float(np.linalg.norm(a-z)) for a,z in zip(params,initial)],
                    structuredNegativeExamples=len(negative_ids), rounds=history, numpyVersion=np.__version__),
                verification=verification, featureVerification=fv, compositionVerification=cv, pairVerification=pv,
                modelSha256=hashlib.sha256(json.dumps(model,separators=(",",":"),ensure_ascii=False).encode()).hexdigest())
            ARTIFACT.write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            print(f"VERIFIED artifact saved: {ARTIFACT}",flush=True)
            return
    raise SystemExit("Acceptance incomplete. Checkpoint retained; no artifact published. Resume with more --rounds.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--steps",type=int,default=6000)
    parser.add_argument("--rounds",type=int,default=8)
    parser.add_argument("--learning-rate",type=float,default=.003)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if args.verify_only:
        artifact=json.loads(ARTIFACT.read_text(encoding="utf-8"))
        assert artifact["featureDefinitions"]==FEATURES and artifact["featurePairs"]==PAIRS
        params=base.arrays(artifact["model"])
        assert verify_concepts(params)["passed"] and base.verify_compositions(params)["passed"] and verify_pairs(params)["passed"]
        verification,_=base.exhaustive(params)
        assert verification["errors"]==0
        print("All 73 concepts, 30 pairs, 3 compositions and exhaustive classification passed.")
    else:
        train(args)


if __name__ == "__main__":
    main()
