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
  function guide(manifest,c,num,result){
    const phases=manifest.optimizer.phases||[{firstStep:1,lastStep:manifest.steps,name:'联合训练',learningRate:manifest.optimizer.learningRate,lossWeights:[1,1,1,1]}];
    const substitution=result?.details
      ? `<br><small>第 ${result.step.toLocaleString()} 次更新前 · 当前训练批次及约束项</small><br><span data-trace-loss-substitution>L = ${result.details.rawLosses.map((loss,k)=>`${c.lossWeights[k]} × ${num(loss)}`).join(' + ')}</span><br><span data-trace-loss-sum>= ${result.losses.map(num).join(' + ')}</span><br><strong data-trace-loss-total>= ${num(result.losses.reduce((a,b)=>a+b,0))}</strong><br><small>总和按完整精度计算，各显示值分别四舍五入；可勾选“完整精度”核对。</small>`
      : '<br><small>初始化尚无训练批次，暂无可代入的损失数值。</small>';
    return `<div class="trace-loss-guide"><h3>损失函数与固定目标</h3><p><b>目标值固定，损失值随参数与训练批次变化。</b>分阶段只改变启用系数 λ 和学习率；未启用的项系数为 0，参数本身不会被清零。同一套 1,351 个参数接收各项梯度之和。</p>
      <div class="trace-equation">hⱼ = tanh(Σᵢ W₁,ⱼᵢ xᵢ + b₁,ⱼ)<br>s = Σⱼ vⱼ hⱼ + b₂<br>L = λ₁ L分类 + λ₂ L特征 + λ₃ L边界 + λ₄ L组合${substitution}</div>
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
    return `<div class="trace-audit"><h3>逐项追溯：${escape(d.label)}</h3><label>损失来源 <select id="trace-audit-kind">${names.map((name,k)=>`<option value="${k}" ${kind===k?'selected':''}>${name} · ${trace.rows[k].length} 项</option>`).join('')}</select></label><p>只列出在计算结构上与该参数有关的项；其余项对该参数偏导为 0。重复样本保留各自批次位置。</p><div class="trace-equation">原始梯度求和 = ${num(trace.rawGradients[kind])}<br>× 阶段系数 ${trace.lossWeights[kind]} = ${num(trace.parts[kind])}</div>
      ${!r?`<p class="trace-audit-empty">${kind===1?'该参数没有局部特征监督目标；输出层与自由神经元不受此项直接监督。':kind===2?'该参数不属于明确特征的输入权重或偏置，边界损失不依赖它。':'该参数不参与六个方框的四边贡献约束。'} 本项梯度为 0。</p>`:`<div class="trace-audit-detail">${grid(r.pixels)}<div><b>${escape(title)}</b><p>${kind<2?'单项损失（未除以 N）':'单项损失'} = ${num(r.unaveragedLoss??r.loss)}${r.target!==undefined?`；目标 ${num(r.target)}`:''}</p>${kind<2?`<button data-trace-pick-sample="${r.index}">用此训练样本观察网络</button>`:''}</div></div>${chain(d,kind,r,num)}
      <div class="trace-table-scroll"><table><thead><tr><th>来源项</th><th>${kind<2?'单样本／特征损失':'单项损失'}</th><th>原始偏导</th><th>有效梯度贡献</th></tr></thead><tbody>${rows.map((row,i)=>`<tr class="${selected===i?'trace-selected':''}"><td><button data-trace-audit-row="${i}">${kind<2?`样本 ${row.index+1}`:kind===2?row.kind==='satisfy'?'满足目标':`X${row.bit+1} 违约`:escape(row.name)}</button></td><td>${num(row.unaveragedLoss??row.loss)}</td><td>${num(row.rawDerivative)}</td><td>${num(row.gradient)}</td></tr>`).join('')}</tbody></table></div>`}
      <p data-trace-audit-check>四项有效梯度之和 ${num(trace.total)}；已与本次记录总梯度校验，最大误差 ${num(trace.maximumError)}。</p></div>`;
  }

  const sourceOptions=[['auto','单样本 · 当前训练目标'],['classification','单样本 · 分类损失'],['concept','单样本 · 特征监督'],['boundary','检测边界约束'],['composition','四边贡献约束'],['total','本次更新 · 总梯度']];
  function sourceHelp(source,c,open){
    const explanations={
      auto:`自动选择一种单样本偏导来讲解；当前选择${c.lossWeights[0]===0?'特征监督':'分类损失'}。这不表示训练只使用这一项，实际启用系数为 [${c.lossWeights.join(', ')}]。`,
      classification:'只看当前训练样本的分类误差怎样影响该参数：正例要求得分 ≥ 1.6，负例要求得分 ≤ 0.4。',
      concept:'只看当前训练样本的局部特征误差：满足检测目标时隐藏响应趋向 +0.9，否则趋向 −0.9；仅监督指定的 25 个神经元。',
      boundary:'汇总该参数相关的最不利输入约束：满足特征时激活前输入和 z ≥ 0.55，不满足时 z ≤ −0.55。使用专门构造的图案，不是当前单个训练样本。',
      composition:'汇总六个示例的四边贡献约束：要求每条边的响应 × 输出权重 ≥ 0.14。使用标准方框，不是当前单个训练样本。',
      total:'整批样本的分类、特征梯度分别平均，再加上边界和四边贡献约束梯度；全部乘过阶段系数。这才是实际交给 Adam 的梯度。'
    };
    return `<details class="trace-source-help" data-replay-source-help ${open?'open':''}><summary>偏导来源是什么意思？</summary><p><b>当前：</b>${explanations[source]}</p><dl>${sourceOptions.map(([key,label])=>`<dt>${label}</dt><dd>${explanations[key]}</dd>`).join('')}</dl><p>单样本选项在权重格显示“单样本原始偏导 × 阶段系数”，尚未除以批次大小 N；左侧会继续展示除以 N 后的贡献。两个约束选项显示该类约束的梯度之和，无需除以 N。总梯度也不等于实际改变量 Δw，Adam 还使用动量与二阶矩。</p><p>切换来源只改变显示，不改变历史训练。若选中损失的阶段系数为 0，它在本阶段不参与更新。</p></details>`;
  }
  const fraction=(top,bottom)=>`<span class="trace-fraction"><span>${top}</span><span>${bottom}</span></span>`;
  function chain(d,kind,r,num){
    if(!r)return `<p class="trace-chain-empty">${kind===1?'该参数没有局部特征监督目标；自由神经元和输出层参数不受此项直接监督。':kind===2?'边界约束只依赖明确特征的输入权重与偏置。':'该参数不参与六个方框的四边贡献约束。'} 此项损失不依赖该参数，∂ℓ/∂θ = 0。</p>`;
    const j=(d.layer===1?d.neuron:d.input)+1,h=`h<sub>${j}</sub>`,z=`z<sub>${j}</sub>`,v=`v<sub>${j}</sub>`;
    const theta=d.layer===1?(d.kind==='bias'?`b<sub>1,${j}</sub>`:`W<sub>1,${j},${d.input+1}</sub>`):d.kind==='bias'?'b<sub>2</sub>':v;
    const deriv=(a,b)=>fraction('∂'+a,'∂'+b),terms=[];
    const term=(a,b,expression,substitution,value)=>terms.push({symbol:deriv(a,b),expression,substitution,value});
    let preparation='';
    if(kind===0){
      preparation=`ℓ分类 = e²；e = ${r.positive?'min':'max'}(s − ${r.target}, 0)<br>e = ${r.positive?'min':'max'}(${num(r.score)} − ${num(r.target)}, 0) = ${num(r.error)}`;
      term('ℓ分类','s','2e',`2 × (${num(r.error)})`,r.lossDerivative);
      if(d.layer===1)term('s',h,v,num(r.outputWeight),r.outputWeight);
      else term('s',theta,d.kind==='bias'?'1':h,num(r.input),r.input);
    }else if(kind===1){
      preparation=`ℓ特征 = 0.2 × (${h} − t)²<br>e = ${num(r.response)} − (${num(r.target)}) = ${num(r.error)}`;
      term('ℓ特征',h,`0.4 × (${h} − t)`,`0.4 × (${num(r.error)})`,r.lossDerivative);
    }else if(kind===2){
      preparation=`ℓ边界 = e²；${z} = ${num(r.z)}，目标 ${r.kind==='satisfy'?'≥':'≤'} ${num(r.target)}<br>e = ${r.kind==='satisfy'?'min':'max'}(${num(r.z)} − (${num(r.target)}), 0) = ${num(r.error)}`;
      term('ℓ边界',z,'2e',`2 × (${num(r.error)})`,r.lossDerivative);
    }else{
      preparation=`ℓ组合 = 4e²；c = ${h} × ${v}<br>c = ${num(r.response)} × (${num(r.outputWeight)}) = ${num(r.contribution)}<br>e = min(${num(r.contribution)} − 0.14, 0) = ${num(r.error)}`;
      term('ℓ组合','c','8e',`8 × (${num(r.error)})`,r.lossDerivative);
      if(d.layer===1)term('c',h,v,num(r.outputWeight),r.outputWeight);
      else term('c',theta,h,num(r.response),r.response);
    }
    if(d.layer===1){
      if(kind!==2)term(h,z,`1 − ${h}²`,`1 − (${num(r.response)})²`,r.activationDerivative);
      term(z,theta,d.kind==='bias'?'1':`x<sub>${d.input+1}</sub>`,num(r.input),r.input);
    }
    const loss=['ℓ分类','ℓ特征','ℓ边界','ℓ组合'][kind],weighted=r.rawDerivative*r.coefficient;
    const zero=r.coefficient===0?'本阶段系数为 0，因此这一项不参与更新；原始偏导仍按真实参数计算。':r.rawDerivative===0?(r.lossDerivative===0?'这一项已达到目标，损失对上游量的偏导为 0。':r.input===0?'当前图案中这个输入像素为 0，最后一个因子为 0，因此这条输入连接的偏导为 0。':r.activationDerivative===0?'当前 tanh 在计算精度下已饱和，激活函数导数为 0。':'链条中存在零因子（例如输出权重为 0），因此乘积为 0。'):'';
    return `<div class="trace-chain" data-chain-kind="${kind}" data-chain-parameter="${d.index}"><p>θ = ${theta} · ${escape(d.label)}。${kind<2?'以下先计算这个训练样本的原始偏导，再乘阶段系数、除以批次大小。':'以下计算选中约束项的偏导，再与其他约束项相加。'}</p><div class="trace-equation">${preparation}</div>
      <div class="trace-equation trace-chain-formula" aria-label="链式法则">${deriv(loss,theta)} = ${terms.map(t=>t.symbol).join(' · ')}</div>
      <ol class="trace-chain-steps">${terms.map(t=>`<li><div>${t.symbol} = ${t.expression}</div><div class="trace-chain-number">= ${t.substitution===num(t.value)?'':t.substitution+' = '}<b data-chain-factor="${t.value}">${num(t.value)}</b></div></li>`).join('')}</ol>
      ${kind===0?'<p class="trace-chain-note">本模型使用间隔平方损失 ℓ = e²，没有 ½ 系数，因此第一项偏导为 2e；达到分类间隔后 e = 0。</p>':''}
      ${d.layer===1&&kind!==2?'<p class="trace-chain-note">隐藏层使用 tanh，实际导数为 1 − h²；这里没有把导数人为设为 1。输出层为线性输出。</p>':''}
      <div class="trace-equation">${deriv(loss,theta)} = ${terms.map(t=>`(${num(t.value)})`).join(' × ')}<br>= <b data-chain-raw="${r.rawDerivative}">${num(r.rawDerivative)}</b><br>× 阶段系数 ${r.coefficient} = <span data-chain-weighted="${weighted}">${num(weighted)}</span><br>${kind<2?`÷ 批次大小 ${r.divisor} = <b data-chain-contribution="${r.gradient}">${num(r.gradient)}</b>（对批次总梯度的贡献）`:`该约束项的有效梯度贡献 = <b data-chain-contribution="${r.gradient}">${num(r.gradient)}</b>（无需除以 N）`}</div>${zero?`<p class="trace-chain-note">${zero}</p>`:''}</div>`;
  }
  return {definition,summary,guide,lossTable,sampleLoss,audit,chain,sourceOptions,sourceHelp};
})();
