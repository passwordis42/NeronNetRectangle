# 神经网络正方形识别教学

[在线打开教学页面](https://passwordis42.github.io/NeronNetRectangle/)

在 5×5 黑白画布上观察神经网络如何检测局部边线，并通过真实权重得到正方形判断分数。页面为纯静态 HTML，模型及脚本全部内嵌，可在线使用或离线打开。

默认入口是 **部分解释 · 6 个方框**：

- 网络为 **25 → 50 → 1**，保持一个隐藏层。
- **25 个明确特征 + 25 个自由学习特征**，全部 1,351 个参数经过实际训练。
- 点击居中／左上／右下 3×3、左上／右下 4×4、外圈 5×5 的缩略图，查看四条边的真实响应与输出贡献。
- 使用缺边、内部多点或自由绘画，对比局部特征与整体判断的关系。
- 支持神经元详情、全部贡献求和、模型对照、预设导入导出和本地状态恢复。

正式六位小数模型已穷举 **33,554,432** 种黑白输入，**零误判**。识别范围是整幅画布恰好一个完整、轴对齐的正方形轮廓，内部和外部没有多余像素。

## 本地打开

直接打开 [`llm/神经网络教学.html`](llm/神经网络教学.html)。无需安装依赖、后端或 API 密钥。原有实验页面和权重文件也保存在仓库中。

## 训练与验证

训练定义、验证记录及新旧模型说明见 [`llm/README-组合特征.md`](llm/README-组合特征.md)。

```sh
python -m pip install -r llm/scripts/requirements.txt
python llm/scripts/train_partial_features.py
python llm/scripts/bundle_composition.py
node llm/tests/partial-features.test.cjs
node llm/tests/composition.test.cjs
node llm/tests/all-features.test.cjs
```

## 网页发布

GitHub Pages 使用 `.github/workflows/pages.yml`：推送到 `main` 后检查模型和页面，再将当前教学 HTML 作为网站首页发布。Pages 设置的构建来源应为 **GitHub Actions**。

本地生成发布文件：

```sh
python llm/scripts/build_site.py
python -m http.server 8000 --directory _site
```

生成目录 `_site/`、本地备份、训练检查点和 Python 缓存不进入版本控制。
