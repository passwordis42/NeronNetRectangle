/* Dynamic symbol map. Values come exclusively from the read-only replay context. */
const SquareReplayAnnotations = (() => {
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const colors={x:'#0969b5',w1:'#b85100',h:'#16743b',w2:'#8628ba',y:'#bc203e',gradient:'#c31d73',b:'#896000',sum:'#43627a'};
  const sym=(letter,layer,index)=>`${letter}${layer?`<sup>(${layer})</sup>`:''}${index!==undefined?`<sub>${index}</sub>`:''}`;
  const fraction=(top,bottom)=>`<span class="map-fraction"><span>${top}</span><span>${bottom}</span></span>`;
  function render(d,num) {
    const p=d.parameterInfo,desc=d.description,outputBias=desc.index===1350;
    const hidden=desc.layer===1?desc.neuron:desc.kind==='weight'?desc.input:null;
    const j=hidden===null?null:hidden+1,i=desc.layer===1&&desc.kind==='weight'?desc.input+1:null;
    const m=d.model,f=d.forward,y=f.outputs[0],b2=m.layers[1].biases[0];
    const h=hidden===null?null:f.activations[1][hidden],z=hidden===null?null:f.preActivations[0][hidden];
    const b1=hidden===null?null:m.layers[0].biases[hidden],v=hidden===null?null:m.layers[1].weights[0][hidden];
    const x=i===null?null:d.pixels[i-1],w=i===null?null:m.layers[0].weights[hidden][i-1];
    const W1=sym('w',1,`${j},${i}`),H=sym('h',1,j),Z=sym('z',1,j),V=sym('w',2,j),B1=sym('b',1,j),B2=sym('b',2,1);
    const theta=outputBias?B2:desc.layer===2?V:desc.kind==='bias'?B1:W1;
    const phaseName={initial:'初始化',before:'更新前',after:'更新后',published:'正式模型 · 六位小数'}[d.networkPhase];
    const networkLabel=d.step===0?'观察画布':d.networkSource==='sample'?`训练样本 ${d.sampleIndex+1}`:'观察画布';
    const mismatch=d.step>0&&(d.networkSource!=='sample'||d.networkPhase!=='before');
    const scope=p.source==='total'?'L':p.isSingle?`(λ${p.source==='concept'?'特征':'分类'} ℓ${p.source==='concept'?'特征':'分类'})`:`(λ${p.source==='boundary'?'边界':'组合'} L${p.source==='boundary'?'边界':'组合'})`;
    const value=(id,n)=>`<b data-map-value="${id}" data-value="${n===null?'':n}">${num(n)}</b>`;
    function card(key,formula,n,title,body){
      return `<article class="map-callout map-${key}" data-map-card="${key}" style="--map-color:${colors[key]}"><div class="map-symbol">${formula}${n!==undefined?` = ${value(key,n)}`:''}</div><strong>${title}</strong><p>${body}</p></article>`;
    }
    const gradient=card('gradient',fraction(`∂${scope}`,`∂${theta}`),p.gradient,'所选参数的偏导，不是权重',
      d.step===0?'尚未反向传播，没有训练批次或偏导。':
      `${esc(p.sourceName)}${p.isSingle?` · 样本 ${d.sampleIndex+1}`:''}，始终使用第 ${d.step.toLocaleString()} 次更新前参数。`+
      (p.isSingle?`<br>原始单样本偏导 ${num(p.rawGradient)} × λ ${p.coefficient} = ${num(p.gradient)}；再 ÷ ${d.batchSize} = ${num(p.batchContribution)}，才是该样本对批次梯度的贡献。`:'')+
      `<br>实际交给 Adam 的总梯度：${num(p.totalGradient)}；实际改变量 Δθ：${num(p.delta)}。`);
    const top=[],bottom=[],nodes=[],edges=[];
    function node(key,left,label,n){
      nodes.push(`<div class="map-node map-${key}" data-map-point="${key}" style="--map-left:${left}%;--map-color:${colors[key]}"><small>${label}</small><b>${num(n)}</b></div>`);
    }
    function edge(key,from,to,label,selected=false){
      edges.push({key,from,to});
      nodes.push(`<span class="map-edge-label ${selected?'is-selected':''}" data-map-point="${key}" style="--map-left:${(from+to)/2}%;--map-color:${colors[key]}">${label}</span>`);
    }
    if(i!==null){
      top.push(card('x',sym('x',null,i),x,`输入节点 X${i}`,`第 ${i} 个输入像素的值。`));
      node('x',12,`输入 X${i}`,x);node('h',50,`隐藏 H${j} · ${H}`,h);node('y',88,'输出 Y1 · y<sub>pred</sub>',y);
      edge('w1',12,50,W1,true);edge('w2',50,88,V);
      bottom.push(card('w1',W1,w,`第一层权重 · X${i} → H${j}`,`上标 (1) 表示第一层；下标 ${j},${i} 表示接收端 H${j}、来源 X${i}。`));
    }else if(desc.layer===1){
      top.push(card('b',B1,b1,`隐藏 H${j} 的偏置`,'直接加到激活前的加权和 z 中，不对应某个输入像素。'));
      node('b',12,`偏置项 ${B1}`,b1);node('h',50,`隐藏 H${j} · ${H}`,h);node('y',88,'输出 Y1 · y<sub>pred</sub>',y);
      edge('w1',12,50,`+ ${B1}`,true);edge('w2',50,88,V);
    }else if(!outputBias){
      top.push(card('h',H,h,`隐藏节点 H${j} 的激活输出`,'这是节点内的数值，来自 tanh(z)。'));
      node('h',22,`隐藏 H${j} · ${H}`,h);node('y',78,'输出 Y1 · y<sub>pred</sub>',y);
      edge('w2',22,78,V,true);
    }else{
      const sum=f.activations[1].reduce((s,a,k)=>s+a*m.layers[1].weights[0][k],0);
      top.push(card('sum','Σⱼ '+sym('w',2,'j')+sym('h',1,'j'),sum,'全部 50 个隐藏神经元的加权和','输出偏置与这个加权和相加，得到最终分数。'));
      top.push(card('b',B2,b2,'输出层偏置','这是输出神经元内部的加法项，不属于任何一个隐藏神经元。'));
      node('sum',20,'50 项加权和',sum);node('y',80,'输出 Y1 · y<sub>pred</sub>',y);
      edge('b',20,80,`+ ${B2}`,true);
    }
    if(!outputBias)top.push(card('w2',V,v,`第二层权重 · H${j} → Y1`,'乘以该隐藏响应 h，得到这一条连接对输出分数的贡献。'));
    top.push(card('y','y<sub>pred</sub>',y,'网络预测分数 · 输出节点 Y1','线性输出，已包含全部 50 条隐藏连接的贡献和输出偏置。它不是损失 L。'));
    if(desc.layer===1)bottom.push(card('h',H,h,`隐藏节点 H${j} 的激活输出`,'这是方框内的数值；显示精度可能不同于网络结构中的两位小数。'));
    bottom.push(gradient);
    const target=outputBias?'b':desc.layer===2?'w2':'w1';
    const path=edges.map(e=>`<line x1="${e.from}%" x2="${e.to}%" y1="50%" y2="50%" stroke="${colors[e.key]||colors.b}" stroke-width="${e.key===target?5:2.5}"/>`).join('');
    const inset=hidden===null?`<section class="map-inset"><h3>输出神经元内部</h3><p>y<sub>pred</sub> = Σⱼ ${sym('w',2,'j')}${sym('h',1,'j')} + ${B2} = ${num(f.activations[1].reduce((s,a,k)=>s+a*m.layers[1].weights[0][k],0))} + (${num(b2)}) = ${num(y)}</p></section>`:
      `<section class="map-inset"><h3>把 H${j} 内部展开看</h3><div class="map-inset-flow"><article><strong>① 加权求和 + 偏置</strong><p>${Z} = Σᵢ ${sym('w',1,`${j},i`)}xᵢ + ${B1}</p><p>Σᵢ wᵢxᵢ = ${num(m.layers[0].weights[hidden].reduce((s,a,k)=>s+a*d.pixels[k],0))}<br>${B1} = ${value('b1',b1)}<br>${Z} = ${value('z',z)}</p></article><span aria-hidden="true">→</span><article><strong>② tanh 激活函数</strong><p>${H} = tanh(${Z})</p><p>= tanh(${num(z)})<br>= ${num(h)}</p></article><span aria-hidden="true">→</span><article><strong>③ 隐藏激活输出</strong><p>${H} = ${num(h)}</p><p>经所示输出权重的贡献：<br>${H} × ${V}<br>= ${num(h)} × ${num(v)}<br>= ${num(h*v)}</p></article></div><p>z 是激活前的数值；h 是激活后的响应。tanh 的导数为 1 − h²，不是人为设为 1。</p></section>`;
    return `<p class="map-context">第 ${d.step.toLocaleString()} 次${d.step?'更新':'初始化'} · ${esc(networkLabel)} · ${phaseName} · ${esc(desc.label)}</p>
      <p class="map-scope">只突出所选参数的路径；隐藏加权和仍使用全部 25 个输入，预测分数仍使用全部 50 个隐藏神经元。</p>
      ${mismatch?'<p class="map-warning">当前节点与权重用于前向观察；粉色偏导来自历史训练批次、更新前参数，不能直接把当前观察数值代入历史求导。切到“训练样本 · 更新前”可对齐单样本计算。</p>':''}
      <div class="map-diagram" data-map-gradient-target="${target}">
        <svg class="map-leaders" aria-hidden="true"></svg>
        <div class="map-callout-row">${top.join('')}</div>
        <div class="map-path" aria-label="所选参数所在的网络路径"><svg class="map-path-lines" aria-hidden="true">${path}</svg>${nodes.join('')}</div>
        <div class="map-callout-row map-bottom">${bottom.join('')}</div>
      </div>${inset}
      <footer class="map-footer"><b>符号说明</b>　上标 (1)/(2) 表示层；w 是权重，b 是偏置，x 是输入，z 是加权和，h 是激活输出。<br>
      ℓ 表示单个样本的损失；L 表示批次损失或全部训练目标的总损失。损失在输出之后计算，不是 Y1 的预测分数。${d.losses?`<br>本次更新前的总损失 L = ${d.losses.map(num).join(' + ')} = ${value('loss',d.losses.reduce((a,b)=>a+b,0))}`:''}</footer>`;
  }
  function create(replay) {
    const dialog=document.createElement('dialog');
    dialog.id='replayParameterMap';dialog.className='replay-parameter-map';
    dialog.setAttribute('aria-labelledby','parameterMapTitle');
    dialog.innerHTML='<header class="map-header"><div><span>公式与网络 · 参数对照</span><h2 id="parameterMapTitle">公式里的字母，对应图上的哪里？</h2></div><button type="button" data-map-close autofocus aria-label="关闭参数对照图">关闭 ×</button></header><div class="map-body"></div>';
    document.body.append(dialog);
    let opener=null,frame=0;
    function expanded(){document.querySelectorAll('[data-replay-annotations]').forEach(b=>b.setAttribute('aria-expanded',String(dialog.open)));}
    function lines(){
      cancelAnimationFrame(frame);frame=requestAnimationFrame(()=>{
        const diagram=dialog.querySelector('.map-diagram'),svg=dialog.querySelector('.map-leaders');if(!diagram||!svg||!dialog.open)return;
        if(matchMedia('(max-width: 700px)').matches){svg.innerHTML='';return;}
        const box=diagram.getBoundingClientRect();svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
        svg.innerHTML='<defs><marker id="map-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>';
        diagram.querySelectorAll('[data-map-card]').forEach(card=>{
          const key=card.dataset.mapCard,target=key==='gradient'?diagram.dataset.mapGradientTarget:key;
          const point=diagram.querySelector(`[data-map-point="${target}"]`);if(!point)return;
          const a=card.getBoundingClientRect(),b=point.getBoundingClientRect(),top=a.top<b.top;
          const x1=a.left+a.width/2-box.left,y1=(top?a.bottom:a.top)-box.top;
          const x2=b.left+b.width/2-box.left,y2=(top?b.top:b.bottom)-box.top;
          svg.insertAdjacentHTML('beforeend',`<path d="M ${x1} ${y1} C ${x1} ${(y1+y2)/2}, ${x2} ${(y1+y2)/2}, ${x2} ${y2}" stroke="${colors[key]}" stroke-width="2" fill="none" marker-end="url(#map-arrow)"/>`);
        });
      });
    }
    function update(){
      if(!replay.active&&dialog.open)dialog.close();
      expanded();if(!dialog.open)return;
      const d=replay.annotationSnapshot(),body=dialog.querySelector('.map-body'),scroll=dialog.scrollTop;
      const focused=dialog.contains(document.activeElement)?document.activeElement.id:null;
      body.innerHTML=`<div class="map-controls"><label>神经元<select id="map-neuron">${Array.from({length:51},(_,j)=>`<option value="${j}" ${j===(d.description.layer===1?d.description.neuron:50)?'selected':''}>${j===50?'输出神经元':`H1-${j+1}`}</option>`).join('')}</select></label><label>连接或偏置<select id="map-parameter">${d.parameterOptions}</select></label>
        <label>更新编号<input id="map-step" type="number" min="0" max="${d.steps}" value="${d.step}"></label><button type="button" data-map-jump>跳转</button>
        <label>网络观察<select id="map-network"><option value="sample" ${d.networkSource==='sample'?'selected':''}>训练样本 · 更新前</option><option value="observation" ${d.networkSource==='observation'?'selected':''}>观察画布 · 所选参数</option></select></label>
        <label>偏导来源<select id="map-source">${SquareBackpropLosses.sourceOptions.map(([v,t])=>`<option value="${v}" ${v===d.source?'selected':''}>${t}</option>`).join('')}</select></label></div>`+
        (d.ready?render(d,replay.format):`<p class="map-status" role="status">第 ${d.step.toLocaleString()} 次更新：${esc(d.error||'正在加载、重建并校验历史记录，暂不显示数值。')}${d.error?'<button type="button" data-map-jump>重试</button>':''}</p>`);
      dialog.scrollTop=scroll;
      if(focused)document.getElementById(focused)?.focus({preventScroll:true});
      lines();
    }
    document.addEventListener('click',e=>{
      const button=e.target.closest('[data-replay-annotations]');if(!button||!replay.active)return;
      opener=button;if(!dialog.open)dialog.showModal();update();
    });
    dialog.addEventListener('close',()=>{expanded();(opener?.isConnected?opener:document.querySelector('[data-replay-annotations]'))?.focus({preventScroll:true});});
    dialog.addEventListener('click',e=>{
      if(e.target.closest('[data-map-close]'))dialog.close();
      if(e.target.closest('[data-map-jump]'))replay.seekUpdate(dialog.querySelector('#map-step').value);
    });
    dialog.addEventListener('keydown',e=>{if(e.target.id==='map-step'&&e.key==='Enter')replay.seekUpdate(e.target.value);});
    dialog.addEventListener('change',e=>{
      if(e.target.id==='map-neuron')replay.selectParameter(Number(e.target.value)===50?1350:Number(e.target.value)*25+6);
      if(e.target.id==='map-parameter')replay.selectParameter(Number(e.target.value));
      if(e.target.id==='map-source')replay.setView('source',e.target.value);
      if(e.target.id==='map-network')replay.setView('network',e.target.value);
    });
    new ResizeObserver(lines).observe(dialog);
    return {update};
  }
  return {create,render};
})();
