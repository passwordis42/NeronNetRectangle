# 神经网络正方形识别教学

[在线打开教学页面](https://passwordis42.github.io/NeronNetRectangle/)

在 5×5 黑白画布上观察神经网络如何检测局部边线，并通过真实权重得到正方形判断分数。页面为纯静态 HTML，模型及脚本全部内嵌，可在线使用或离线打开。

默认入口是 **部分解释 · 6 个方框**：

- 网络为 **25 → 50 → 1**，保持一个隐藏层。
- **25 个明确特征 + 25 个自由学习特征**，全部 1,351 个参数经过实际训练。
- 隐藏神经元以固定顺序混排，明确特征分散在自由学习特征之间；输入权重、偏置、输出连接与检测目标同步重排，判断结果不变。
- 点击居中／左上／右下 3×3、左上／右下 4×4、外圈 5×5 的缩略图，查看四条边的真实响应与输出贡献。
- 使用缺边、内部多点或自由绘画，对比局部特征与整体判断的关系。
- 支持神经元详情、全部贡献求和、模型对照、预设导入导出和本地状态恢复。
- 新增 **反向传播回放**：10 个讲解阶段，任意查询 12,000 次真实更新的训练样本、四类损失、分项梯度和 Adam 计算。全部权重与偏置均可查，当前步可导出 JSON / CSV。

正式六位小数模型已穷举 **33,554,432** 种黑白输入，**零误判**。识别范围是整幅画布恰好一个完整、轴对齐的正方形轮廓，内部和外部没有多余像素。

## 本地打开

直接打开 [`llm/神经网络教学.html`](llm/神经网络教学.html)。无需安装依赖、后端或 API 密钥。原有实验页面和权重文件也保存在仓库中。

反向传播记录按需加载；本地使用时保留 HTML 旁的 `training-trace` 文件夹。也可下载 [完整离线包（约 132 MB）](https://passwordis42.github.io/NeronNetRectangle/downloads/neural-network-offline.zip)，解压整个文件夹后打开 `index.html`，断网也可逐步查询。只复制 HTML 仍可使用原有前向教学。

回放源于固定种子的确定性训练复现，初始状态继承已有模型并初始化新增特征，并非全随机初始化。每个讲解阶段包含 1,200 次实际更新；最终参数保留六位小数后与现有模型完全一致。中间训练状态不会覆盖正式模型和预设存档。

## 训练与验证

训练定义、验证记录及新旧模型说明见 [`llm/README-组合特征.md`](llm/README-组合特征.md)。

```sh
python -m pip install -r llm/scripts/requirements.txt
python llm/scripts/train_partial_features.py
python llm/scripts/shuffle_partial_features.py
python llm/scripts/record_training_trace.py
python llm/scripts/bundle_composition.py
node llm/tests/partial-features.test.cjs
node llm/tests/composition.test.cjs
node llm/tests/all-features.test.cjs
node llm/tests/backprop.test.cjs
```

## 网页发布

GitHub Pages 使用 `.github/workflows/pages.yml`：推送到 `main` 后检查模型、全部训练记录和页面，再发布教学首页、按需加载的数据块和离线包。Pages 设置的构建来源应为 **GitHub Actions**。

本地生成发布文件：

```sh
python llm/scripts/build_site.py
python -m http.server 8000 --directory _site
```

生成目录 `_site/`、本地备份、训练检查点和 Python 缓存不进入版本控制。
