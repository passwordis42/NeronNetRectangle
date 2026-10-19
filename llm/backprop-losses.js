/* Human-readable loss definitions and read-only parameter provenance. */
const SquareBackpropLosses = (() => {
  const names=['分类误差','局部特征监督','检测边界约束','四边贡献约束'];
  const definition={id:'square-losses-v1',parameterization:'h=tanh(W1*x+b1); s=W2*h+b2',
    fixedTargets:{positiveScoreMinimum:1.6,negativeScoreMaximum:.4,featureResponse:{matched:.9,unmatched:-.9},featurePreactivationMargin:.55,edgeContributionMinimum:.14},
    inferenceThreshold:1,coefficientsInsideLoss:{feature:.2,composition:4},
    formulas:[
      'L_class = (1/N) sum_s e_s^2; e_s = min(score_s-1.6,0) for positive samples, max(score_s-0.4,0) for negative samples',
      'L_feature = (0.2/N) sum_s sum_j_in_F (h_sj-target_sj)^2; target=+0.9 if all on/off requirements match, otherwise -0.9',
      'L_boundary = sum_j_in_F [min(z_min_j-0.55,0)^2 + sum_p_in_Rj max(z_max_jp+0.55,0)^2]',
      'L_composition = 4 sum_examples sum_j_in_example_edges min(h_j*W2_j-0.14,0)^2'
    ],total:'L = sum_k lambda_k * L_k; g_theta = sum_k lambda_k * dL_k/dtheta',
    normalization:'Classification and feature losses average over the ordered batch, including duplicates. Feature loss sums over specified neurons; boundary/composition constraints are sums with no division by batch size.',
    boundaryMeaning:'z_min minimizes z over inputs satisfying the feature. z_max_jp maximizes z with required pixel p violated; other pixels maximize z independently. Selected extremal binary inputs define the gradient branch.'};
  const escape=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function summary(result,c,num){
    const ready=Boolean(result?.details),total=ready?result.losses.reduce((a,b)=>a+b,0):null;
    return `<div class="trace-loss-summary"><div><small>${ready?`本次更新前 · 训练批次 ${result.ids.length} 个样本`:'初始化 · 尚无训练批次'}</small><b>总损失 L = <span data-trace-loss-current>${num(total)}</span></b></div><button data-trace-section="loss" data-trace-focus-loss>损失函数与目标 ↗</button><div class="trace-loss-chips">${names.map((name,k)=>`<span>${name} <b>${ready?num(result.losses[k]):'—'}</b>${c.lossWeights[k]===0?' · 未启用':''}</span>`).join('')}</div><small>有效损失 = 原始损失 × 阶段系数；这里使用训练样本和更新前参数。</small></div>`;
  }
  function guide(manifest,c,num){
    const phases=manifest.optimizer.phases||[{firstStep:1,lastStep:manifest.steps,name:'联合训练',learningRate:manifest.optimizer.learningRate,lossWeights:[1,1,1,1]}];
    return `<div class="trace-loss-guide"><h3>损失函数与固定目标</h3><p><b>目标值固定，损失值随参数与训练批次变化。</b>分阶段只改变启用系数 λ 和学习率；未启用的项系数为 0，参数本身不会被清零。同一套 1,351 个参数接收各项梯度之和。</p>
      <div class="trace-equation">hⱼ = tanh(Σᵢ W₁,ⱼᵢ xᵢ + b₁,ⱼ)<br>s = Σⱼ vⱼ hⱼ + b₂<br>L = λ₁ L分类 + λ₂ L特征 + λ₃ L边界 + λ₄ L组合</div>
      <p>N 是本批次样本数；F 是指定特征神经元集合（25 个）；vⱼ 是 H1-j 到输出的权重。分类与特征项按 N 平均，特征之间求和；边界与组合项直接求和，不再除以 N。</p>
      <div class="trace-loss-definition"><h4>① 分类：间隔平方损失</h4><div class="trace-equation">正例：e = min(s − 1.6, 0)<br>负例：e = max(s − 0.4, 0)<br>L分类 = (1/N) Σ样本 e²<br>∂ℓ分类/∂s = 2e</div><p>正例期望 s ≥ 1.6，负例期望 s ≤ 0.4，达到间隔后该样本的分类损失与偏导为 0。页面判断阈值 s &gt; 1 是推理规则，不是训练目标。</p></div>
      <div class="trace-loss-definition"><h4>② 局部特征：响应平方误差</h4><div class="trace-equation">目标 tⱼ = 满足全部须亮／须空条件 ? +0.9 : −0.9<br>L特征 = (0.2/N) Σ样本 Σⱼ∈F (hⱼ − tⱼ)²<br>∂ℓ特征/∂hⱼ = 0.4(hⱼ − tⱼ)</div><p>自由神经元没有这项目标。此项不依赖输出权重与输出偏置。0.2 已包含在原始损失中，不能再乘一次。</p></div>
      <div class="trace-loss-definition"><h4>③ 检测边界：满足／不满足目标的最不利输入</h4><div class="trace-equation">L边界 = Σⱼ∈F [min(z满足,min − 0.55, 0)²<br>＋ Σ约束像素p max(z违约p,max + 0.55, 0)²]</div><p>z 是激活前输入和。满足特征的输入中，取 z 最小的图案，要求 z ≥ 0.55；逐个翻转须亮／须空像素，在该像素违约的输入中取 z 最大的图案，要求 z ≤ −0.55。不限位置按当前权重正负取最不利值。下面的参数溯源会显示实际选中的 5×5 图案。</p></div>
      <div class="trace-loss-definition"><h4>④ 四边贡献：让检测到的边实际参与判断</h4><div class="trace-equation">cⱼ = hⱼ vⱼ；eⱼ = min(cⱼ − 0.14, 0)<br>L组合 = 4 Σ六个方框 Σ各自四条边 eⱼ²<br>∂ℓ组合/∂cⱼ = 8eⱼ</div><p>每条演示边的训练贡献目标为 0.14；这是训练目标，验收下限另为 0.1。使用六个标准方框，不属于当前批次单个样本。</p></div>
      <h4>实际阶段安排</h4><div class="trace-table-scroll"><table><thead><tr><th>更新编号</th><th>λ分类</th><th>λ特征</th><th>λ边界</th><th>λ组合</th><th>学习率</th></tr></thead><tbody>${phases.map(p=>`<tr><th>${p.firstStep.toLocaleString()}～${p.lastStep.toLocaleString()}</th>${p.lossWeights.map(v=>`<td>${v}</td>`).join('')}<td>${num(p.learningRate)}</td></tr>`).join('')}</tbody></table></div><p>当前 λ = [${c.lossWeights.join(', ')}]。若从第一步同时训练，可以保持这些目标不变，只从第一步启用所有损失；当前回放如实保留已发生的训练安排。</p></div>`;
  }
  function lossTable(result,parameter,c,num){
    return `<h3>实际损失与选中参数的梯度</h3><p>参数：<b>${escape(SquareBackpropCore.describe(parameter).label)}</b>。本表为整个批次及约束项的汇总，不是单样本偏导。</p><div class="trace-loss-table"><table><thead><tr><th>损失项</th><th>原始损失 Lₖ</th><th>阶段系数 λₖ</th><th>有效损失 λₖLₖ</th><th>原始梯度 ∂Lₖ/∂θ</th><th>有效梯度 λₖ∂Lₖ/∂θ</th></tr></thead><tbody>${names.map((name,k)=>`<tr><th>${name}${c.lossWeights[k]===0?' · 未启用':''}</th><td>${num(result.details.rawLosses[k])}</td><td>${c.lossWeights[k]}</td><td>${num(result.losses[k])}</td><td>${num(result.details.rawParts[k][parameter])}</td><td>${num(result.details.parts[k][parameter])}</td></tr>`).join('')}</tbody><tfoot><tr><th>总损失／总梯度</th><td colspan="3">${num(result.losses.reduce((a,b)=>a+b,0))}</td><td colspan="2" data-trace-total-gradient>${num(result.gradient[parameter])}</td></tr></tfoot></table></div><div class="trace-equation">gθ = ${result.details.parts.map(part=>num(part[parameter])).join(' + ')}<br>= ${num(result.gradient[parameter])}</div>`;
  }
  function sampleLoss(result,parameter,c,num){
    const s=result.details.sample,d=SquareBackpropCore.describe(parameter),j=d.layer===1?d.neuron:d.input;
    const target=j===undefined?null:s.targets[j],h=j===undefined?null:s.activations[1][j],featureLoss=target===null?null:.2*(h-target)**2;
    return `<div class="trace-loss-sample"><h4>当前训练样本 ${s.index+1} 的代入</h4><p>${s.positive?'正例，期望 s ≥ 1.6':'负例，期望 s ≤ 0.4'}；实际 s = ${num(s.outputs[0])}。</p><div class="trace-equation">e = ${s.positive?'min':'max'}(${num(s.outputs[0])} − ${s.positive?'1.6':'0.4'}, 0) = ${num(s.error)}<br>单样本分类损失 ℓ = e² = ${num(s.error*s.error)}<br>对总损失的贡献 = ℓ × λ分类 / N<br>= ${num(s.error*s.error)} × ${c.lossWeights[0]} / ${result.ids.length} = ${num(s.error*s.error*c.lossWeights[0]/result.ids.length)}</div>${target===null?'<p>该参数对应的神经元没有局部特征目标，或它是输出偏置。</p>':`<div class="trace-equation">H1-${j+1}：h = ${num(h)}；目标 t = ${num(target)}<br>该特征单样本损失 = 0.2 × (h − t)² = ${num(featureLoss)}<br>对总损失的贡献 = ${num(featureLoss)} × ${c.lossWeights[1]} / ${result.ids.length} = ${num(featureLoss*c.lossWeights[1]/result.ids.length)}</div>${d.layer===2?'<p>这个特征损失虽然存在，但不依赖选中的输出权重，因此对该参数的梯度为 0。</p>':''}`}</div>`;
  }
  const grid=pixels=>`<div class="trace-pixels" role="img" aria-label="该项损失的实际输入图案">${pixels.map(v=>`<i class="${v?'is-on':''}"></i>`).join('')}</div>`;
  function audit(trace,kind,rowIndex,num){
    const rows=trace.rows[kind],selected=Math.max(0,Math.min(rows.length-1,rowIndex||0)),r=rows[selected],d=trace.parameter;
    const title=r?kind<2?`批次样本 ${r.index+1} · 图案编号 ${r.id}`:kind===2?r.kind==='satisfy'?'满足目标的最不利输入':`X${r.bit+1} 违反须${r.required?'亮':'空'}条件的最不利输入`:`${r.name} · H1-${r.neuron+1}`:'';
    let formula='';
    if(r){
      if(kind===0)formula=`∂ℓ/∂s = 2 × ${num(r.error)} = ${num(r.lossDerivative)}<br>`+(d.layer===1?`∂s/∂θ = vⱼ × (1−hⱼ²) × ${d.kind==='bias'?'1':'xᵢ'}<br>= ${num(r.outputWeight)} × ${num(r.activationDerivative)} × ${num(r.input)} = ${num(r.scoreDerivative)}`:`∂s/∂θ = ${d.kind==='bias'?'1':'hⱼ'} = ${num(r.scoreDerivative)}`);
      if(kind===1)formula=`∂ℓ/∂h = 0.4 × (h−t) = ${num(r.lossDerivative)}<br>∂h/∂θ = (1−h²) × ${d.kind==='bias'?'1':'xᵢ'} = ${num(r.activationDerivative)} × ${num(r.input)}`;
      if(kind===2)formula=`z = ${num(r.z)}；目标 ${r.kind==='satisfy'?'≥':'≤'} ${num(r.target)}<br>e = ${r.kind==='satisfy'?'min':'max'}(z − ${num(r.target)}, 0) = ${num(r.error)}<br>∂ℓ/∂θ = 2e × ${d.kind==='bias'?'1':'xᵢ'} = ${num(r.lossDerivative)} × ${num(r.input)}`;
      if(kind===3)formula=`h = ${num(r.response)}；v = ${num(r.outputWeight)}<br>c = h × v = ${num(r.contribution)}；目标 ≥ ${num(r.target)}<br>e = min(c−0.14,0) = ${num(r.error)}<br>∂ℓ/∂c = 8e = ${num(r.lossDerivative)}<br>∂c/∂θ = ${d.layer===1?`v × (1−h²) × ${d.kind==='bias'?'1':'xᵢ'} = ${num(r.outputWeight)} × ${num(r.activationDerivative)} × ${num(r.input)}`:`h = ${num(r.response)}`}`;
    }
    return `<div class="trace-audit"><h3>逐项追溯：${escape(d.label)}</h3><label>损失来源 <select id="trace-audit-kind">${names.map((name,k)=>`<option value="${k}" ${kind===k?'selected':''}>${name} · ${trace.rows[k].length} 项</option>`).join('')}</select></label><p>只列出在计算结构上与该参数有关的项；其余项对该参数偏导为 0。重复样本保留各自批次位置。</p><div class="trace-equation">原始梯度求和 = ${num(trace.rawGradients[kind])}<br>× 阶段系数 ${trace.lossWeights[kind]} = ${num(trace.parts[kind])}</div>
      ${!r?`<p class="trace-audit-empty">${kind===1?'该参数没有局部特征监督目标；输出层与自由神经元不受此项直接监督。':kind===2?'该参数不属于明确特征的输入权重或偏置，边界损失不依赖它。':'该参数不参与六个方框的四边贡献约束。'} 本项梯度为 0。</p>`:`<div class="trace-audit-detail">${grid(r.pixels)}<div><b>${escape(title)}</b><p>${kind<2?'单项损失（未除以 N）':'单项损失'} = ${num(r.unaveragedLoss??r.loss)}${r.target!==undefined?`；目标 ${num(r.target)}`:''}</p>${kind<2?`<button data-trace-pick-sample="${r.index}">用此训练样本观察网络</button>`:''}</div></div><div class="trace-equation">${formula}<br>单项原始偏导 = ${num(r.rawDerivative)}<br>× 阶段系数 ${r.coefficient} ÷ ${r.divisor} = <b>${num(r.gradient)}</b>${kind<2?'（对批次总梯度的贡献）':'（约束项不除以 N）'}</div>
      <div class="trace-table-scroll"><table><thead><tr><th>来源项</th><th>${kind<2?'单样本／特征损失':'单项损失'}</th><th>原始偏导</th><th>有效梯度贡献</th></tr></thead><tbody>${rows.map((row,i)=>`<tr class="${selected===i?'trace-selected':''}"><td><button data-trace-audit-row="${i}">${kind<2?`样本 ${row.index+1}`:kind===2?row.kind==='satisfy'?'满足目标':`X${row.bit+1} 违约`:escape(row.name)}</button></td><td>${num(row.unaveragedLoss??row.loss)}</td><td>${num(row.rawDerivative)}</td><td>${num(row.gradient)}</td></tr>`).join('')}</tbody></table></div>`}
      <p data-trace-audit-check>四项有效梯度之和 ${num(trace.total)}；已与本次记录总梯度校验，最大误差 ${num(trace.maximumError)}。</p></div>`;
  }
  return {definition,summary,guide,lossTable,sampleLoss,audit};
})();
