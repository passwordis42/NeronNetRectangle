# 神经网络正方形识别教学

[在线打开教学页面](https://passwordis42.github.io/NeronNetRectangle/)

在 5×5 黑白画布上观察神经网络如何检测局部边线，并通过真实权重得到正方形判断分数。页面为纯静态 HTML，模型及脚本全部内嵌，可在线使用或离线打开。

默认入口是 **随机起点 · 6 个方框**（v5）：

- 网络为 **25 → 50 → 1**，保持一个隐藏层。
- **25 个明确特征 + 25 个自由学习特征**，全部 1,351 个参数经过实际训练。
- 隐藏神经元以固定顺序混排，明确特征分散在自由学习特征之间；输入权重、偏置、输出连接与检测目标同步重排，判断结果不变。
- 点击居中／左上／右下 3×3、左上／右下 4×4、外圈 5×5 的缩略图，查看四条边的真实响应与输出贡献。
- 使用缺边、内部多点或自由绘画，对比局部特征与整体判断的关系。
- 支持神经元详情、全部贡献求和、模型对照、预设导入导出和本地状态恢复。
- **反向传播回放**：从 `U(−0.1, 0.1)` 随机权重、零偏置开始，任意查询 18,000 次真实更新。10 个讲解阶段之外，每次更新的训练样本、四类损失、偏导和 Adam 计算均可查，支持 JSON / CSV 导出。
- 回放左侧为阶段与计算，右侧为完整网络和参数图；可切换权重、单样本偏导、批次总梯度、实际改变量，支持讲解全屏和缩放。历史训练样本与可编辑观察画布分别标注。
- 右侧网络提供横向紧凑、纵向三层、逐个神经元展开三种布局。观察框右下角可放大、缩小、适应窗口或重置；支持 Ctrl + 滚轮以鼠标位置缩放、拖动空白处移动，布局与缩放随各版本的查看记录保存。
- 回放直接显示当前训练批次的总损失。“损失函数与目标”列出四项完整公式、固定目标、阶段系数、原始与有效损失。点击任意权重或偏置后的“追溯此参数”可逐样本／逐约束图案核对梯度，再查看 Adam 与裁剪；JSON 保存所选参数的明细，CSV 为全部 1,351 个参数列出分项梯度和优化器状态。

正式六位小数模型已穷举 **33,554,432** 种黑白输入，**零误判**。识别范围是整幅画布恰好一个完整、轴对齐的正方形轮廓，内部和外部没有多余像素。

## 本地打开

直接打开 [`llm/神经网络教学.html`](llm/神经网络教学.html)。无需安装依赖、后端或 API 密钥。原有实验页面和权重文件也保存在仓库中。

反向传播记录按需加载；本地使用时保留 HTML 旁的 `training-trace` 文件夹。也可下载 [完整离线包](https://passwordis42.github.io/NeronNetRectangle/downloads/neural-network-offline.zip)，解压整个文件夹后打开 `index.html`，断网也可逐步查询。只复制 HTML 仍可使用原有前向教学。

新版从固定种子 42 独立采样两层权重，不继承旧参数、不预填形状。前 6,000 次只学习人工指定的局部特征与检测边界（学习率 0.01），后 12,000 次联合学习四类损失（学习率 0.003），Adam 状态连续保留。数据种子为 20261019。阶段入口为初始化及第 100、500、1,000、2,000、4,000、6,000、8,000、10,000、12,000、18,000 次更新。

第 18,000 次更新后的参数保留六位小数，与新正式模型完全一致。旧 v4 模型及 12,000 次回放独立保留，已有存档照常恢复；学习方式下拉框可切换到新版。回放不会覆盖正式模型或改写历史。

## 训练与验证

训练定义、验证记录及新旧模型说明见 [`llm/README-组合特征.md`](llm/README-组合特征.md)。

```sh
python -m pip install -r llm/scripts/requirements.txt
python llm/scripts/train_random_partial.py
python llm/scripts/bundle_composition.py
node llm/tests/partial-features.test.cjs v5
node llm/tests/partial-features.test.cjs
node llm/tests/composition.test.cjs
node llm/tests/all-features.test.cjs
node llm/tests/backprop.test.cjs v2
node llm/tests/backprop.test.cjs
node llm/tests/loss-provenance.test.cjs
```

浏览器交互回归可使用 Playwright 运行 `node llm/tests/random-replay.browser.cjs`；可选 `CHROME_PATH` 指定浏览器，`SQUARE_QA_OUTPUT` 指定截图目录。测试覆盖阶段和样本切换、参数／连线、导出、全屏、窄屏、刷新及旧回放。

## 网页发布

GitHub Pages 使用 `.github/workflows/pages.yml`：推送到 `main` 后检查模型、全部训练记录和页面，再发布教学首页、按需加载的数据块和离线包。Pages 设置的构建来源应为 **GitHub Actions**。

本地生成发布文件：

```sh
python llm/scripts/build_site.py
python -m http.server 8000 --directory _site
```

生成目录 `_site/`、本地备份、训练检查点和 Python 缓存不进入版本控制。
