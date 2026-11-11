"""Build the page, on-demand replay records, and a complete offline package."""
from pathlib import Path
import hashlib
import json
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
    archive.write(ROOT / 'llm' / 'README-全特征几何.md', 'neural-network/全特征几何教学.md')
    verification = ROOT / 'llm' / 'geometry-backprop-verification.json'
    if verification.exists():
        archive.write(verification, 'neural-network/训练记录验证.json')
package = downloads / 'neural-network-offline.zip'
with package.open('rb') as handle:
    checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
release_url = 'https://github.com/passwordis42/NeronNetRectangle/releases/download/geometry-teaching-v1/neural-network-offline.zip'
(downloads / 'package.json').write_text(json.dumps({'url': release_url, 'bytes': package.stat().st_size,
    'sha256': checksum}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(downloads / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整离线包</title>'
    '<meta name="viewport" content="width=device-width,initial-scale=1"><body><h1>完整离线教学包</h1>'
    f'<p><a href="{release_url}">下载页面与全部训练记录</a></p>'
    '<p>解压整个文件夹，直接打开 index.html。无需联网。</p>'
    f'<p>SHA-256：<code>{checksum}</code></p></body></html>', encoding='utf-8')
print("Built teaching website and complete offline package: _site/index.html")
