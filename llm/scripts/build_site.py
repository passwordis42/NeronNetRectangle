"""Build the standalone GitHub Pages entry without publishing training files."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "_site"
DESTINATION.mkdir(exist_ok=True)
shutil.copyfile(ROOT / "llm" / "神经网络教学.html", DESTINATION / "index.html")
(DESTINATION / ".nojekyll").write_text("", encoding="utf-8")
print("Built standalone teaching website: _site/index.html")
