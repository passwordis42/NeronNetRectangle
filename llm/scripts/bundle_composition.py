"""Embed the verified artifact and teaching renderer into the standalone HTML."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "神经网络教学.html"
artifact = json.loads((ROOT / "composition-model.json").read_text(encoding="utf-8"))
all_features = json.loads((ROOT / "all-features-model.json").read_text(encoding="utf-8"))
partial_features = json.loads((ROOT / "partial-features-model.json").read_text(encoding="utf-8"))
assert artifact["verification"]["inputCount"] == 1 << 25 and artifact["verification"]["errors"] == 0
assert artifact["featureVerification"]["passed"] and artifact["compositionVerification"]["passed"]
assert all_features["verification"]["inputCount"] == 1 << 25 and all_features["verification"]["errors"] == 0
assert all_features["featureVerification"]["passed"] and all_features["compositionVerification"]["passed"] and all_features["pairVerification"]["passed"]
assert partial_features["verification"]["inputCount"] == 1 << 25 and partial_features["verification"]["errors"] == 0
assert partial_features["featureVerification"]["passed"] and partial_features["compositionVerification"]["passed"]
assert len(partial_features["featureDefinitions"]) == 25 and len(partial_features["compositionExamples"]) == 6
source = (ROOT / "composition-teaching.js").read_text(encoding="utf-8")
block = ('<!-- BEGIN COMPOSITION BUNDLE -->\n<script>\nconst SquareCompositionModelData = '
         + json.dumps(artifact, ensure_ascii=False, separators=(",", ":"))
         + ';\nconst SquareAllFeaturesModelData = ' + json.dumps(all_features, ensure_ascii=False, separators=(",", ":"))
         + ';\nconst SquarePartialFeaturesModelData = ' + json.dumps(partial_features, ensure_ascii=False, separators=(",", ":"))
         + ';\n' + source + '\n</script>\n<!-- END COMPOSITION BUNDLE -->')
html = path.read_text(encoding="utf-8-sig")
if "<!-- BEGIN COMPOSITION BUNDLE -->" in html:
    html = re.sub(r"<!-- BEGIN COMPOSITION BUNDLE -->.*?<!-- END COMPOSITION BUNDLE -->", lambda _: block, html, flags=re.S)
else:
    anchor = '  <script>\n    const STORAGE_KEY'
    assert html.count(anchor) == 1
    html = html.replace(anchor, block + '\n' + anchor)
path.write_text(html, encoding="utf-8")
print("Bundled verified composition model and renderer into", path)
