"""Build the standalone GitHub Pages entry without publishing training files."""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "_site"
DESTINATION.mkdir(exist_ok=True)
shutil.copyfile(ROOT / "llm" / "神经网络教学.html", DESTINATION / "index.html")
shutil.copytree(ROOT / "llm" / "training-trace", DESTINATION / "training-trace", dirs_exist_ok=True)
(DESTINATION / ".nojekyll").write_text("", encoding="utf-8")
downloads = DESTINATION / "downloads"
downloads.mkdir(exist_ok=True)
with zipfile.ZipFile(downloads / "neural-network-offline.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    archive.write(DESTINATION / "index.html", "neural-network/index.html")
    for file in sorted((DESTINATION / "training-trace").rglob("*")):
        if file.is_file():
            archive.write(file, "neural-network/" + file.relative_to(DESTINATION).as_posix())
    archive.writestr("neural-network/使用说明.txt", "解压整个文件夹，直接打开 index.html。请保留旁边的 training-trace 文件夹。无需联网或安装服务。\n")
print("Built teaching website and complete offline package: _site/index.html")
