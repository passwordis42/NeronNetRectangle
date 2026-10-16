/* Source for the standalone HTML bundle. Rebuild with scripts/bundle_composition.py.
 * Masks describe teaching targets only; all detections and scores use forward(). */
function createSquareCompositionTeaching(artifact, ID, name) {
  const shared = createSquareTrainedTeaching(artifact, ID, name);
  const features = artifact.featureDefinitions;
  const examples = artifact.compositionExamples;
  const pairs = artifact.featurePairs || [];
  const allFeatures = pairs.length > 0;
  const count = artifact.model.topology[1];
  const expandedComposition = !allFeatures && examples.length > 3;
  const freeCount = count - features.length;
  const colors = ["#18798b", "#b56b1a", "#7554ae", "#ce5468"];
  const sides = ["上边", "下边", "左边", "右边"];
  const number = value => Math.abs(value) > 0 && Math.abs(value) < .0001 ? value.toExponential(3) : value.toFixed(4);
  const signed = value => `${value > 0 ? "+" : ""}${number(value)}`;
  const preciseScore = value => value !== 1 && Math.abs(value - 1) < .005 ? value.toPrecision(12) : number(value);
  const featureName = i => `H1-${i + 1}${features.find(f => f.neuronIndex === i) ? ` · ${features.find(f => f.neuronIndex === i).name}` : " · 自由学习特征"}`;
  const exampleAt = index => examples[Number.isInteger(index) && index >= 0 && index < examples.length ? index : 0];
  const pairAt = index => pairs[Number.isInteger(index) && index >= 0 && index < pairs.length ? index : 20];

  function pairPixels(index, variant = "complete") {
    const pair = pairAt(index);
    const pixels = Array.from({ length: 25 }, (_, p) => Number(pair.boundary.includes(p)));
    if (variant === "missing") pixels[pair.cornerPixel] = 0;
    return pixels;
  }

  function renderFeatureTarget(index, response) {
    const feature = features.find(f => f.neuronIndex === index);
    const target = shared.renderFeatureTarget(index, response);
    if (!allFeatures || !feature) return target;
    const explanation = feature.kind === "skeleton"
      ? "轮廓与内外空白已匹配，右下角可以亮也可以空；还需查看配对的缺角检查。"
      : feature.kind === "defect" ? "检测到这个缺角图案时，正响应乘以负输出权重会减分。未检测到时响应为负，乘以负权重可能加分。"
      : "此特征的非限定位置没有输入连接；仅指定位置参与检测。";
    return `${target}<section class="all-feature-role"><p>${explanation}</p>${Number.isInteger(feature.pairedNeuronIndex) ? `<button type="button" class="secondary" data-inspector-paired="${feature.pairedNeuronIndex}">查看配对的${feature.kind === "skeleton" ? "缺角检查" : "待补角轮廓"} · H1-${feature.pairedNeuronIndex + 1}</button>` : ""}</section>`;
  }

  function examplePixels(index, variant = "complete") {
    const example = exampleAt(index);
    const pixels = Array.from({ length: 25 }, (_, p) => Number(example.boundary.includes(p)));
    if (variant === "missing") pixels[example.missingPixel] = 0;
    if (variant === "extra") pixels[example.extraPixel] = 1;
    return pixels;
  }

  function examplePreview(example) {
    return `<span class="composition-example-preview" aria-hidden="true">${Array.from({ length: 25 }, (_, p) => `<i${example.boundary.includes(p) ? ' class="is-on"' : ''}></i>`).join("")}</span>`;
  }

  function sideMap(feature, color, pixels = null) {
    return `<span class="composition-mini-grid" role="img" aria-label="${feature.name}${pixels ? "的当前输入" : "的指定检测位置"}">${Array.from({ length: 25 }, (_, p) => {
      const target = feature.on.includes(p);
      const requiredOff = feature.off.includes(p);
      const on = pixels ? pixels[p] === 1 : target;
      const tone = target ? on ? `background:${color};color:white` : `border:1px dashed ${color};color:${color}` : on ? requiredOff ? "background:#c43755;color:white" : "background:#555;color:white" : "";
      return `<span style="${tone}" title="${Math.floor(p / 5) + 1}行${p % 5 + 1}列${target ? " · 须亮" : requiredOff ? " · 须空" : " · 不限定"}">${pixels && ((target && !on) || (requiredOff && on)) ? "×" : ""}</span>`;
    }).join("")}</span>`;
  }

  function compositionMap(example, responses, current = false) {
    const cells = Array.from({ length: 25 }, (_, p) => {
      const owners = example.featureIndices.map((i, side) => ({ i, side })).filter(({ i }) => features[i].on.includes(p));
      const detected = owners.filter(({ i }) => !current || responses[i] > 0);
      let style = "", label = "";
      if (detected.length) {
        const parts = detected.map(({ side }, i) => `${colors[side]} ${100 * i / detected.length}% ${100 * (i + 1) / detected.length}%`);
        style = `background:${detected.length === 1 ? colors[detected[0].side] : `linear-gradient(135deg,${parts.join(",")})`};color:white`;
      } else if (owners.length) {
        style = "border:1px dashed #8d9994;color:#66746d";
        label = "·";
      }
      const names = (current ? detected : owners).map(({ side }) => sides[side]).join("、");
      return `<span style="${style}" title="${Math.floor(p / 5) + 1}行${p % 5 + 1}列${owners.length ? ` · ${names || "相关边未检测到"}` : " · 不属于轮廓"}">${label}</span>`;
    }).join("");
    return `<div class="composition-map" role="img" aria-label="${current ? "当前检测到的边的位置" : "四条边组成的目标轮廓"}">${cells}</div>`;
  }

  function featureCard(state, forward, index, side = null) {
    const value = forward.activations[1][index];
    const weight = state.model.layers[1].weights[0][index];
    const contribution = value * weight;
    const feature = features.find(f => f.neuronIndex === index);
    const selected = state.selectedNeuron?.layerIndex === 1 && state.selectedNeuron.neuronIndex === index;
    const color = side === null ? feature?.kind === "defect" ? "#b53452" : "#166c51" : colors[side];
    return `<button type="button" class="trained-feature composition-feature ${selected ? "is-selected" : ""}" style="--edge-color:${color}" data-teaching-layer="1" data-teaching-neuron="${index}" aria-pressed="${selected}" aria-label="查看${featureName(index)}">
      <span class="trained-feature-title"><b>${side === null ? `H1-${index + 1}` : `${sides[side]} · H1-${index + 1}`}</b><span>${feature ? value > 0 ? "检测到 ✓" : "未检测到" : "学习响应"}</span></span>
      ${feature ? `<span class="composition-card-maps"><span>指定检测目标${shared.targetMap(feature)}</span><span>当前输入${sideMap(feature, color, state.pixels)}</span></span><span class="hint">${feature.name}</span>${allFeatures && Number.isInteger(feature.pairedNeuronIndex) ? `<span class="all-feature-pair-hint">${feature.kind === "defect" ? "缺角时减分" : "轮廓支持"} · 配对 H1-${feature.pairedNeuronIndex + 1}</span>` : ""}` : `<span class="trained-feature-body">${shared.weightMap(state.model.layers[0].weights[index], `H1-${index + 1} 输入权重分布`, true)}<small>输入权重分布</small></span>`}
      <span class="composition-equation"><span>响应 <b>${number(value)}</b></span><span>× 输出权重 <b>${signed(weight)}</b></span><strong class="${contribution >= 0 ? "positive" : "negative"}">＝ ${signed(contribution)}</strong></span>
    </button>`;
  }

  function renderPair(state, forward) {
    if (!allFeatures) return "";
    const pair = pairAt(state.featurePair);
    const a = pair.skeletonIndex, d = pair.defectIndex;
    const h = forward.activations[1], w = state.model.layers[1].weights[0];
    const terms = [h[a] * w[a], h[d] * w[d]];
    const chosen = variant => pairPixels(pair.candidateIndex, variant).every((v, i) => v === state.pixels[i]);
    return `<section class="teaching-stage all-feature-pair" id="featurePairPanel"><h3>轮廓与缺角 · 配对观察</h3>
      <label>选择观察位置<select data-feature-pair-select aria-label="选择轮廓与缺角配对">${pairs.map(p => `<option value="${p.candidateIndex}"${p.candidateIndex === pair.candidateIndex ? " selected" : ""}>${p.name}</option>`).join("")}</select></label>
      <p class="hint">左边暂不要求右下角亮起；右边专门检测它是否缺失。其余轮廓须亮，内部和外部须空。点击下方按钮对照，响应与贡献会重新计算。</p>
      <div class="composition-actions">${[["complete", "画出完整方框"], ["missing", "去掉右下角"]].map(([key, text]) => `<button type="button" class="secondary${chosen(key) ? " is-selected" : ""}" data-feature-pair-draw="${key}" aria-pressed="${chosen(key)}">${text}</button>`).join("")}</div>
      <div class="composition-feature-grid">${featureCard(state, forward, a)}${featureCard(state, forward, d)}</div>
      <div class="all-feature-pair-sum"><span>这对特征的贡献</span><b>${signed(terms[0])} ${terms[1] < 0 ? "−" : "+"} ${number(Math.abs(terms[1]))} ＝ ${signed(terms[0] + terms[1])}</b></div>
      <p class="hint">缺角检查的输出权重为负：检测到缺角时减分；未检测到时，负响应 × 负权重可以加分。其他 ${count - 2} 项仍参与最终判断。</p>
      <button type="button" class="teaching-result ${forward.outputs[0] > 1 ? "is-passing" : "is-failing"}" data-teaching-layer="2" data-teaching-neuron="0"><b>${forward.outputs[0] > 1 ? "是正方形" : "不是正方形"}</b><span>最终得分 ${preciseScore(forward.outputs[0])} ${forward.outputs[0] > 1 ? ">" : "≤"} 1</span></button>
    </section>`;
  }

  function renderCatalog(state, forward) {
    const groups = [["local", "边与转角", "13 个局部部件", "teachingLocalFeaturesOpen"],
      ["skeleton", "待补角轮廓", "30 个位置与大小；右下角不限定", "teachingSkeletonFeaturesOpen"],
      ["defect", "缺角检查", "30 个对应的缺角图案", "teachingDefectFeaturesOpen"]];
    return `<section class="teaching-stage" id="allFeaturesCatalog"><h3>全部 ${count} 个神经元 · 每一个都有检测目标</h3>
      <p class="hint">实心格须亮，○ 须空，灰格不限定。这里也包含上方演示的四条边；点击轮廓或缺角卡片，会同步更新配对观察与右侧详情。</p>
      ${groups.map(([key, title, description, field]) => `<details class="trained-contributions all-feature-group" data-trained-detail="${field}" data-feature-group="${key}" ${state[field] ? "open" : ""}><summary>${title} <span>${description}</span></summary><div class="trained-features">${features.filter(f => f.group === key).map(f => featureCard(state, forward, f.neuronIndex)).join("")}</div></details>`).join("")}
    </section>`;
  }

  function renderView(state, forward) {
    const index = Number.isInteger(state.compositionExample) && state.compositionExample >= 0 && state.compositionExample < examples.length ? state.compositionExample : 0;
    const example = exampleAt(index);
    const hidden = forward.activations[1];
    const weights = state.model.layers[1].weights[0];
    const contributions = hidden.map((value, i) => value * weights[i]);
    const focus = new Set(example.featureIndices);
    const focusSum = contributions.reduce((sum, value, i) => sum + (focus.has(i) ? value : 0), 0);
    const otherSum = contributions.reduce((sum, value, i) => sum + (focus.has(i) ? 0 : value), 0);
    const bias = state.model.layers[1].biases[0];
    const score = forward.outputs[0];
    const detected = example.featureIndices.filter(i => hidden[i] > 0).length;
    const scorePasses = score > state.outputs[0].threshold;
    const rows = contributions.map((v, i) => `<tr data-trained-contribution="${i}" class="${focus.has(i) ? "composition-focus-row" : ""}"><th scope="row">H1-${i + 1}${focus.has(i) ? ` · ${sides[example.featureIndices.indexOf(i)]}` : ""}</th><td>${number(hidden[i])}</td><td>${signed(weights[i])}</td><td class="${v >= 0 ? "positive" : "negative"}">${signed(v)}</td></tr>`).join("");
    const otherCards = allFeatures ? "" : contributions.map((_, i) => focus.has(i) ? "" : featureCard(state, forward, i)).join("");
    const chosen = (variant) => examplePixels(index, variant).every((v, i) => v === state.pixels[i]);
    return `<div class="teaching-focus-head"><div><span class="teaching-eyebrow">${allFeatures ? "全特征可解释" : expandedComposition ? "部分特征解释 · 六个方框" : "组合特征"} · 真实训练 · 一个隐藏层</span><h3>四条边，怎样帮助判断一个正方形？</h3><p class="hint">${allFeatures ? `全部 ${count} 个神经元都有指定目标：13 个边与转角、30 个待补角轮廓、30 个缺角检查。有效连接和偏置经过训练。` : `${features.length} 个边与转角有人工指定的局部目标，另 ${freeCount} 个神经元自由学习。全部 ${artifact.training.trainableParameters.toLocaleString("zh-CN")} 个参数经过训练；这些特征一起参与最终判断。`}</p></div></div>
      <div class="teaching-flow-label">25 个像素 <span>→</span> ${count} 个特征（tanh）<span>→</span> 1 个线性得分<br>第二层权重：隐藏层 → 输出层。得分严格大于 1 判是。</div>
      <section class="teaching-stage composition-stage">
        ${expandedComposition ? `<h3>选择一个方框 · ${examples.length} 个可解释组合</h3>` : ""}
        <div class="composition-example-tabs${expandedComposition ? " composition-example-gallery" : ""}" role="group" aria-label="组合示例">${examples.map((e, i) => `<button type="button" class="secondary${i === index ? " is-selected" : ""}" data-composition-example="${i}" aria-pressed="${i === index}">${expandedComposition ? examplePreview(e) : ""}<span>${e.name}</span></button>`).join("")}</div>
        <p class="hint">${expandedComposition ? `点击方框，画布与四条边同步切换。当前观察：${example.name}。也可以在画布中自由修改，观察哪些特征仍有响应。` : "切换示例只改变查看的特征；点击下方按钮将图案画入左侧画布。"}</p>
        <div class="composition-actions" role="group" aria-label="绘制组合示例">${[["complete", "完整方框"], ["missing", "缺一格"], ["extra", "内部多一点"]].map(([key, text]) => `<button type="button" class="secondary${chosen(key) ? " is-selected" : ""}" data-composition-draw="${key}" aria-pressed="${chosen(key)}">${text}</button>`).join("")}</div>
      </section>
      <section class="teaching-stage"><h3><span class="teaching-step">1</span>隐藏层 · 检测四条边</h3><p class="hint">每条边对应一个真实神经元。响应 &gt; 0 表示检测到；点击卡片查看 25 个输入权重与计算。</p><div class="composition-feature-grid">${example.featureIndices.map((i, side) => featureCard(state, forward, i, side)).join("")}</div></section>
      <section class="teaching-stage"><h3><span class="teaching-step">2</span>特征位置与组合示意</h3>
        <div class="composition-maps"><div><b>四条边的目标轮廓</b>${compositionMap(example, hidden)}<span class="hint">交点用双色显示两条边的重合。</span></div><div><b>当前检测到 ${detected} / 4 条边</b>${compositionMap(example, hidden, true)}<span class="hint">根据真实响应显示对应位置，虚线为未检测到的边。</span></div></div>
        <div class="composition-legend">${sides.map((name, side) => `<span><i style="background:${colors[side]}"></i>${name}</span>`).join("")}</div>
        <p class="hint">这是特征位置示意；隐藏神经元输出标量，输出层也只输出分数。四边齐全仍需其他特征检查整体关系，例如内部或外部是否有多余像素。</p>
      </section>
      <section class="trained-output teaching-stage"><div><h3><span class="teaching-step">3</span>经过第二层权重，得到分数</h3><div class="composition-totals">
        <span>这四条边的贡献 <b data-composition-total="edges">${signed(focusSum)}</b></span>
        <span>＋ 其余 ${count - focus.size} 项贡献 <b data-composition-total="others">${signed(otherSum)}</b></span>
        <span>＋ 输出偏置 <b data-composition-total="bias">${signed(bias)}</b></span>
        <span>＝ 最终得分 <b data-composition-total="score">${preciseScore(score)}</b></span></div></div>
        <button type="button" class="teaching-result ${scorePasses ? "is-passing" : "is-failing"}" data-teaching-layer="2" data-teaching-neuron="0" aria-pressed="${state.selectedNeuron?.layerIndex === 2}"><b>${scorePasses ? "是正方形" : "不是正方形"}</b><span>得分 ${preciseScore(score)} ${scorePasses ? ">" : "≤"} 1</span><span>输出为分数，不是概率或图像</span></button>
      </section>
      <details class="trained-contributions" data-trained-detail="teachingOutputOpen" ${state.teachingOutputOpen ? "open" : ""}><summary>核对全部 ${count} 项输出贡献</summary><div class="trained-table-scroll"><table><thead><tr><th>特征</th><th>响应</th><th>输出权重</th><th>贡献</th></tr></thead><tbody>${rows}</tbody><tfoot><tr><th colspan="3">全部贡献之和</th><td>${number(focusSum + otherSum)}</td></tr><tr><th colspan="3">偏置</th><td>${signed(bias)}</td></tr><tr><th colspan="3">最终得分</th><td>${preciseScore(score)}</td></tr></tfoot></table></div><p class="hint">使用完整精度计算，显示值保留四位小数。绿色底行是当前例子的四条边。</p></details>
      ${allFeatures ? renderPair(state, forward) + renderCatalog(state, forward) : `<details class="trained-contributions" data-trained-detail="teachingFirstFeaturesOpen" ${state.teachingFirstFeaturesOpen ? "open" : ""}><summary>其余 ${count - focus.size} 个神经元 · ${features.filter(f => !focus.has(f.neuronIndex)).length} 个明确特征与 ${freeCount} 个自由学习特征</summary><p class="hint">这些特征同样参与最终判断。自由学习特征的小图展示输入权重，不指定形状含义；正负贡献均保留。</p><div class="trained-features">${otherCards}</div></details>`}
      <details class="trained-provenance" data-trained-detail="teachingTrainingOpen" ${state.teachingTrainingOpen ? "open" : ""}><summary>训练来源与验证结果</summary>
        ${allFeatures ? `<p>人工指定全部 ${count} 个检测目标。前 13 个局部特征从旧版组合模型初始化；新增 60 个检测器使用结构化初始化，输出连接从随机幅度开始训练。Adam 种子 ${artifact.training.seed}，训练 ${artifact.training.steps.toLocaleString("zh-CN")} 步。</p><p>总计 ${artifact.training.totalParameters.toLocaleString("zh-CN")} 个参数，其中 ${artifact.training.trainableParameters.toLocaleString("zh-CN")} 个有效连接与偏置参与训练；${artifact.training.fixedZeroConnections} 个非限定位置的输入连接固定为零。须亮位置权重为正，须空位置为负；轮廓支持项输出权重为正，缺角检查输出权重为负。这些结构约束用于保证特征含义清楚。</p>` : `<p>${expandedComposition ? "从旧版组合模型初始化前 13 个局部特征，为新增 12 条边设置局部目标" : "从原单层特征引导模型初始化，为新增边设置局部目标"}，再用 Adam 训练全部 ${artifact.training.trainableParameters.toLocaleString("zh-CN")} 个参数。固定种子 ${artifact.training.seed}，共 ${artifact.training.steps.toLocaleString("zh-CN")} 步；${features.length} 个明确特征与 ${freeCount} 个自由学习特征共同学习。没有增加隐藏层。自由学习神经元不指定形状目标，不使用完整轮廓／缺角模板。</p>`}
        <p>正式六位小数模型已穷举 ${artifact.verification.inputCount.toLocaleString("zh-CN")} 种黑白输入，误判 ${artifact.verification.errors}。30 个正例最低分 ${number(artifact.verification.closestPositive.score)}，其他图案最高分 ${number(artifact.verification.closestNegative.score)}。</p>
        <p>${features.length} 个检测器通过全部目标模式及外围像素极值验证：满足目标时响应均为正，不满足时均为负。${examples.length} 个标准例子的每条边贡献至少 0.1。</p>
        <p>离线干预验证：将四条边的响应置零，${artifact.compositionVerification.examples.map(e => `${e.name}得分下降 ${number(e.scoreDrop)}`).join("；")}。该干预不改变页面实际计算。</p>
        <p>范围：5×5 黑白画布中恰好一个边长 2～5 的轴对齐正方形轮廓，内部和外部没有多余像素。灰度与旋转图形不在此验证范围。</p>
      </details>`;
  }

  return { ...shared, composition: true, allFeatures, expandedComposition, drawOnExampleSelect: expandedComposition, examples, examplePixels, renderView, renderFeatureTarget,
    features, pairs, pairPixels,
    provenance: () => JSON.parse(JSON.stringify({ training: artifact.training, verification: artifact.verification,
      modelSha256: artifact.modelSha256, featureDefinitions: features, featureVerification: artifact.featureVerification,
      compositionExamples: examples, compositionVerification: artifact.compositionVerification,
      ...(allFeatures ? { featurePairs: pairs, pairVerification: artifact.pairVerification } : {}) })) };
}
const SquareCompositionTeaching = createSquareCompositionTeaching(SquareCompositionModelData,
  "square-trained-composition-50-v1", "单隐藏层组合特征预设（50 个神经元）");
const SquareAllFeaturesTeaching = createSquareCompositionTeaching(SquareAllFeaturesModelData,
  "square-trained-composition-73-v2", "单隐藏层全特征可解释预设（73 个神经元）");
const SquarePartialFeaturesTeaching = createSquareCompositionTeaching(SquarePartialFeaturesModelData,
  "square-trained-partial-50-v3", "单隐藏层部分解释预设（6 个方框，25 个明确特征 + 25 个自由学习特征）");
