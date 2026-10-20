"""Embed the verified artifact and teaching renderer into the standalone HTML."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "神经网络教学.html"
artifact = json.loads((ROOT / "composition-model.json").read_text(encoding="utf-8"))
all_features = json.loads((ROOT / "all-features-model.json").read_text(encoding="utf-8"))
partial_features = json.loads((ROOT / "partial-features-model.json").read_text(encoding="utf-8"))
shuffled_features = json.loads((ROOT / "partial-features-shuffled-model.json").read_text(encoding="utf-8"))
assert artifact["verification"]["inputCount"] == 1 << 25 and artifact["verification"]["errors"] == 0
assert artifact["featureVerification"]["passed"] and artifact["compositionVerification"]["passed"]
assert all_features["verification"]["inputCount"] == 1 << 25 and all_features["verification"]["errors"] == 0
assert all_features["featureVerification"]["passed"] and all_features["compositionVerification"]["passed"] and all_features["pairVerification"]["passed"]
assert partial_features["verification"]["inputCount"] == 1 << 25 and partial_features["verification"]["errors"] == 0
assert partial_features["featureVerification"]["passed"] and partial_features["compositionVerification"]["passed"]
assert len(partial_features["featureDefinitions"]) == 25 and len(partial_features["compositionExamples"]) == 6
assert shuffled_features["verification"]["inputCount"] == 1 << 25 and shuffled_features["verification"]["errors"] == 0
assert shuffled_features["featureVerification"]["passed"] and shuffled_features["compositionVerification"]["passed"] and shuffled_features["permutationVerification"]["passed"]
assert shuffled_features["neuronOrder"]["sourceModelSha256"] == partial_features["modelSha256"]
random_features = json.loads((ROOT / "partial-features-random-model.json").read_text(encoding="utf-8"))
assert random_features["verification"]["errors"] == 0 and random_features["featureVerification"]["passed"] and random_features["compositionVerification"]["passed"]
source = (ROOT / "composition-teaching.js").read_text(encoding="utf-8")
trace = json.loads((ROOT / "training-trace/v1/manifest.json").read_text(encoding="utf-8"))
assert trace["modelSha256"] == shuffled_features["modelSha256"]
assert trace["verification"]["stepsChecked"] == 12000 and trace["verification"]["finalModelExactMatch"]
random_trace = json.loads((ROOT / "training-trace/v2/manifest.json").read_text(encoding="utf-8"))
assert random_trace["modelSha256"] == random_features["modelSha256"] and random_trace["verification"]["stepsChecked"] == random_trace["steps"]
# The page only needs the index; complete per-update losses remain in the trace.
trace_view = {k:v for k,v in trace.items() if k != 'timeline'}
random_trace_view = {k:v for k,v in random_trace.items() if k != 'timeline'}
replay_source = ('\nconst SquareTrainingTraceManifest = ' + json.dumps(trace_view, ensure_ascii=False, separators=(",", ":")) + ';\n'
                 + '\nconst SquareRandomTrainingTraceManifest = ' + json.dumps(random_trace_view, ensure_ascii=False, separators=(",", ":")) + ';\n'
                 + (ROOT / "backprop-core.js").read_text(encoding="utf-8") + '\n'
                 + (ROOT / "backprop-losses.js").read_text(encoding="utf-8") + '\n'
                 + (ROOT / "backprop-teaching.js").read_text(encoding="utf-8") + '\n'
                 + (ROOT / "backprop-annotations.js").read_text(encoding="utf-8") + '\n'
                 + (ROOT / "backprop-viewport.js").read_text(encoding="utf-8"))
block = ('<!-- BEGIN COMPOSITION BUNDLE -->\n<script>\nconst SquareCompositionModelData = '
         + json.dumps(artifact, ensure_ascii=False, separators=(",", ":"))
         + ';\nconst SquareAllFeaturesModelData = ' + json.dumps(all_features, ensure_ascii=False, separators=(",", ":"))
         + ';\nconst SquarePartialFeaturesModelData = ' + json.dumps(partial_features, ensure_ascii=False, separators=(",", ":"))
         + ';\nconst SquareShuffledPartialModelData = ' + json.dumps(shuffled_features, ensure_ascii=False, separators=(",", ":"))
         + ';\nconst SquareRandomPartialModelData = ' + json.dumps(random_features, ensure_ascii=False, separators=(",", ":"))
         + ';\n' + source + replay_source + '\n</script>\n<!-- END COMPOSITION BUNDLE -->')
html = path.read_text(encoding="utf-8-sig")
css = ('/* BEGIN BACKPROP CSS */\n' + (ROOT / "backprop-teaching.css").read_text(encoding="utf-8") + '\n/* END BACKPROP CSS */')
if '/* BEGIN BACKPROP CSS */' in html:
    html = re.sub(r'/\* BEGIN BACKPROP CSS \*/.*?/\* END BACKPROP CSS \*/', lambda _: css, html, flags=re.S)
else:
    html = html.replace('  </style>', css + '\n  </style>', 1)
if "<!-- BEGIN COMPOSITION BUNDLE -->" in html:
    html = re.sub(r"<!-- BEGIN COMPOSITION BUNDLE -->.*?<!-- END COMPOSITION BUNDLE -->", lambda _: block, html, flags=re.S)
else:
    anchor = '  <script>\n    const STORAGE_KEY'
    assert html.count(anchor) == 1
    html = html.replace(anchor, block + '\n' + anchor)
path.write_text(html, encoding="utf-8")
print("Bundled verified composition model and renderer into", path)
