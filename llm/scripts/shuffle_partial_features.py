"""Publish a fixed permutation of the trained partial-feature hidden layer.

No retraining: incoming weights, hidden biases, outgoing weights and concept
indices move together. The original v3 artifact remains available for imports.
The six-decimal reordered model is exhaustively verified before publishing.
"""
import copy
import hashlib
import json
import random
import train_composition as base
import numpy as np

SOURCE = base.ROOT / "partial-features-model.json"
DESTINATION = base.ROOT / "partial-features-shuffled-model.json"
SEED = 20261016


def shuffled_artifact(source):
    result = copy.deepcopy(source)
    count = source["model"]["topology"][1]
    order = list(range(count))
    random.Random(SEED).shuffle(order)
    inverse = [order.index(i) for i in range(count)]
    a, b = source["model"]["layers"]
    result["model"]["layers"] = [
        dict(weights=[a["weights"][old] for old in order], biases=[a["biases"][old] for old in order]),
        dict(weights=[[row[old] for old in order] for row in b["weights"]], biases=b["biases"][:]),
    ]
    for feature in result["featureDefinitions"]:
        feature["neuronIndex"] = inverse[feature["neuronIndex"]]
    result["featureDefinitions"].sort(key=lambda f: f["neuronIndex"])
    for example in result["compositionExamples"]:
        example["featureIndices"] = [inverse[i] for i in example["featureIndices"]]
    result["neuronOrder"] = dict(method="fixed hidden-unit permutation after training", seed=SEED,
        newToOld=order, oldToNew=inverse, sourcePresetId="square-trained-partial-50-v3",
        sourceModelSha256=source["modelSha256"])
    result["modelSha256"] = hashlib.sha256(json.dumps(result["model"], separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return result


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    artifact = shuffled_artifact(source)
    params, before = base.arrays(artifact["model"]), base.arrays(source["model"])
    order = artifact["neuronOrder"]["newToOld"]
    assert sorted(order) == list(range(50))
    assert np.array_equal(params[0], before[0][:, order])
    assert np.array_equal(params[1], before[1][order])
    assert np.array_equal(params[2], before[2][order])
    assert np.array_equal(params[3], before[3])
    inputs = np.concatenate([base.POS_X, base.pixels(np.random.default_rng(SEED).integers(1 << 25, size=4096))])
    _, old_scores = base.forward(inputs, before)
    _, new_scores = base.forward(inputs, params)
    error = float(np.max(np.abs(old_scores-new_scores)))
    assert error < 1e-12
    artifact["permutationVerification"] = dict(passed=True,
        method="exact parameter correspondence plus numerical score comparison",
        inputCount=len(inputs), maximumScoreDifference=error)
    artifact["featureVerification"] = base.verify_features(params, artifact["featureDefinitions"])
    artifact["compositionVerification"] = base.verify_compositions(params, artifact["featureDefinitions"], artifact["compositionExamples"])
    artifact["verification"], _ = base.exhaustive(params)
    assert artifact["featureVerification"]["passed"] and artifact["compositionVerification"]["passed"]
    assert artifact["verification"]["errors"] == 0
    DESTINATION.write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("VERIFIED fixed neuron order saved:", DESTINATION, flush=True)
    print("Clear neuron positions (1-based):", [f["neuronIndex"]+1 for f in artifact["featureDefinitions"]], flush=True)


if __name__ == "__main__":
    main()
