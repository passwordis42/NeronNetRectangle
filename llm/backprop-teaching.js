/* Read-only, versioned training replay. Every displayed update is a recorded update. */
const SquareBackpropTeaching = (() => {
  const names=['分类误差','局部特征监督','检测边界约束','四边贡献约束'];
  const escape=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function create(options) {
    const {host,tabs,entries}=options, core=SquareBackpropCore;
    let entry=null,manifest=null,artifact=null,enabled=false,step=0,parameter=506,precision=false;
    let phase='after',source='auto',metric='weight',networkSource='sample',section='backward',zoom=1,layout='compact',zoomMode='manual';
    let result=null,currentModel=null,busy=false,error='',notice='',requestId=0,worker=null,workerURL=null;
    let observed=Array(25).fill(0),sampleIndex=0,fullTable=false,expandedAdam=false,sampleDetails=false;
    const chunks=new Map(),pending=new Map();
    const num=v=>Number.isFinite(v)?(precision?String(v):v!==0&&Math.abs(v)<.0000005?v.toExponential(6):v.toFixed(6)):'—';
    const integer=(v,min,max,fallback)=>Number.isInteger(v)&&v>=min&&v<=max?v:fallback;
    const version=()=>manifest?.dataVersion||'v1';
    const key=()=>`nn-square-backprop-view-v2:${manifest.id}`;
    const config=()=>core.optimizerAt(manifest.optimizer,step);
    const actualSource=()=>source==='auto'?(config().lossWeights[0]===0?'concept':'classification'):source;
    const sourceName=()=>({classification:'单样本 · 分类损失',concept:'单样本 · 特征监督',boundary:'检测边界约束梯度',composition:'四边贡献约束梯度',total:'本次更新 · 总梯度'}[actualSource()]);
    const grid=(pixels,label)=>`<div class="trace-pixels" role="img" aria-label="${label}">${pixels.map(x=>`<i class="${x?'is-on':''}"></i>`).join('')}</div>`;
    function save(){try{localStorage.setItem(key(),JSON.stringify({step,parameter,precision,source,metric,networkSource,section,sampleIndex,phase,zoom,layout,zoomMode}));}catch(_){}}
    function notify(){options.onChange();}
    function switchEntry(next){
      ++requestId;enabled=false;busy=false;result=null;currentModel=null;error='';entry=next;manifest=next?.manifest;artifact=next?.artifact;
      if(!manifest)return;
      let saved={};try{saved=JSON.parse(localStorage.getItem(key())||(version()==='v1'?localStorage.getItem('nn-square-backprop-view-v1'):null)||'{}');}catch(_){}
      step=integer(saved.step,0,manifest.steps,0);parameter=integer(saved.parameter,0,1350,506);precision=Boolean(saved.precision);
      sampleIndex=integer(saved.sampleIndex,0,10000,null);phase=['before','after'].includes(saved.phase)?saved.phase:step===manifest.steps?'published':'after';
      zoom=Number.isFinite(saved.zoom)?Math.max(.1,Math.min(3,saved.zoom)):1;
      layout=['compact','vertical','expanded'].includes(saved.layout)?saved.layout:'compact';
      zoomMode=saved.zoomMode==='fit'?'fit':'manual';
      source=['auto','classification','concept','boundary','composition','total'].includes(saved.source)?saved.source:'auto';
      metric=['weight','gradient','delta'].includes(saved.metric)?saved.metric:'weight';
      networkSource=saved.networkSource==='observation'?'observation':'sample';
      section=['forward','loss','backward','update'].includes(saved.section)?saved.section:'backward';
    }
    function setModel(){
      if(!result){currentModel=null;return;}
      if(phase==='published'){
        const expected=core.flatten(artifact.model);
        if(result.after.some((v,i)=>Number(v.toFixed(6))!==expected[i]))throw Error('回放终点与正式模型不一致，已停止显示');
        currentModel=core.model(expected);
      }else currentModel=core.model(phase==='before'?result.before:result.after);
    }
    function ensureWorker(){
      if(worker)return;
      workerURL=URL.createObjectURL(new Blob([`const core=(${createSquareBackpropCore.toString()})();const cache=new Map();
      onmessage=async({data:d})=>{try{
        let chunk=cache.get(d.key);
        if(!chunk){const bytes=Uint8Array.from(atob(d.base64),c=>c.charCodeAt(0));const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');if(hash!==d.hash)throw Error('数据块校验失败，请重试或重新下载完整离线包');const raw=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();chunk=core.readChunk(raw);if(chunk.firstStep!==d.firstStep)throw Error('数据块编号不匹配');cache.set(d.key,chunk);if(cache.size>4)cache.delete(cache.keys().next().value);}
        const r=core.reconstruct(chunk,d.step,d.optimizer);
        if(d.step){const match=Array.from(r.ids).findIndex((id,i)=>i<r.positives&&id===d.defaultInput);let selected=Number.isInteger(d.sample)&&d.sample>=0&&d.sample<r.ids.length?d.sample:Math.max(0,match);
          if(d.find){const disabled=core.optimizerAt(d.optimizer,d.step).lossWeights[d.source==='concept'?1:0]===0;const found=disabled?-1:core.findNonzeroSample(r.before,r.ids,r.positives,d.features,d.parameter,d.source,selected);r.found=found;if(found>=0)selected=found;}
          r.details=core.gradients(r.before,r.ids,r.positives,d.features,d.examples,selected,core.optimizerAt(d.optimizer,d.step).lossWeights);
          r.gradientError=Math.max(...r.details.total.map((v,i)=>Math.abs(v-r.gradient[i])));
          if(r.gradientError>1e-8||r.details.losses.some((v,i)=>Math.abs(v-r.losses[i])>1e-8))throw Error('重建损失或梯度未通过校验');}
        postMessage({id:d.id,result:r});
      }catch(e){postMessage({id:d.id,error:e.message});}};`],{type:'text/javascript'}));
      worker=new Worker(workerURL);
      worker.onmessage=({data})=>{if(data.id!==requestId)return;busy=false;if(data.error){error=data.error;result=null;currentModel=null;chunks.delete(`${version()}:${Math.max(0,Math.floor((step-1)/100))}`);}else{result=data.result;sampleIndex=result.details?.sample.index||0;notice=result.found===-1?'本批次中，该参数没有非零的此类单样本偏导。':result.found>=0?'已定位本批次中偏导非零的样本。':'';try{setModel();save();}catch(e){error=e.message;result=null;currentModel=null;}}notify();};
      worker.onerror=()=>{busy=false;error='计算线程无法运行，请刷新页面重试。';result=null;currentModel=null;worker.terminate();worker=null;URL.revokeObjectURL(workerURL);notify();};
    }
    globalThis.SquareTraceChunks={receive(v,index,payload){if(payload===undefined){payload=index;index=v;v='v1';}const k=`${v}:${index}`,item=pending.get(k);if(!item||typeof payload!=='string')return;pending.delete(k);clearTimeout(item.timer);item.script.remove();chunks.set(k,payload);if(chunks.size>4)chunks.delete(chunks.keys().next().value);item.resolve(payload);}};
    function loadChunk(index){
      const k=`${version()}:${index}`;if(chunks.has(k))return Promise.resolve(chunks.get(k));if(pending.has(k))return pending.get(k).promise;
      let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;}),script=document.createElement('script');
      const item={script,promise,resolve,reject,timer:null};
      const fail=()=>{if(pending.get(k)!==item)return;pending.delete(k);clearTimeout(item.timer);script.remove();reject(Error('训练记录加载失败。请重试；离线使用需解压完整离线包并保留 training-trace 文件夹。'));};
      item.timer=setTimeout(fail,45000);script.onerror=fail;pending.set(k,item);script.src=(manifest.dataPath||'training-trace/v1/')+manifest.chunks[index].file;document.head.append(script);return promise;
    }
    async function seek(next,selectedSample=null,keepPhase=false,find=false){
      step=integer(Number(next),0,manifest.steps,step);if(!keepPhase)phase=step===manifest.steps?'published':'after';
      const id=++requestId,index=Math.max(0,Math.floor((step-1)/100));sampleIndex=selectedSample;busy=true;error='';notice='';result=null;currentModel=null;save();notify();
      try{
        if(!globalThis.Worker||!globalThis.DecompressionStream||!globalThis.crypto?.subtle)throw Error('请使用支持训练记录解压的新版浏览器；原有前向教学仍可使用。');
        ensureWorker();const base64=await loadChunk(index);if(id!==requestId||!enabled)return;
        worker.postMessage({id,key:`${version()}:${index}`,firstStep:manifest.chunks[index].firstStep,hash:manifest.chunks[index].sha256,base64,step,optimizer:manifest.optimizer,features:artifact.featureDefinitions,examples:artifact.compositionExamples,sample:selectedSample,defaultInput:artifact.compositionExamples[0].boundary.reduce((v,p)=>v+2**p,0),find,parameter,source:actualSource()});
      }catch(e){if(id===requestId){busy=false;error=e.message;notify();}}
    }
    function parameterInfo(i){
      if(!result)return null;
      const s=actualSource(),c=config(),single=result.details?core.sampleGradient(result.details.sample,i):null;
      const raw=single?(s==='concept'?single.concept:single.classification)*result.ids.length:null;
      const lossIndex={classification:0,concept:1,boundary:2,composition:3}[s],isSingle=s==='concept'||s==='classification';
      const coefficient=s==='total'?1:c.lossWeights[lossIndex];
      const rawValue=isSingle?raw:result.details?.rawParts[lossIndex]?.[i]??null;
      return {index:i,...core.describe(i),before:result.before[i],after:result.after[i],weight:(networkSource==='sample'?result.before:core.flatten(currentModel))[i],
        gradient:step?(s==='total'?result.gradient[i]:rawValue*coefficient):null,rawGradient:rawValue,coefficient,isSingle,
        batchContribution:step&&isSingle?raw*coefficient/result.ids.length:null,totalGradient:step?result.gradient[i]:null,
        delta:step?result.after[i]-result.before[i]:null,step,source:s,sourceName:sourceName(),sampleIndex,metric};
    }
    const metricValue=i=>{const p=parameterInfo(i);return p?p[metric==='weight'?'weight':metric==='delta'?'delta':'gradient']:null;};
    function selectParameter(i,select=true){parameter=integer(Number(i),0,1350,parameter);save();if(select){const d=core.describe(parameter);options.onSelect(d.layer,d.neuron);}else notify();}
    function parameterOptions(){const d=core.describe(parameter),ids=d.layer===1?[...Array.from({length:25},(_,i)=>d.neuron*25+i),1250+d.neuron,1300+d.neuron]:[...Array.from({length:50},(_,i)=>1300+i),1350];return ids.map(i=>`<option value="${i}" ${i===parameter?'selected':''}>${core.describe(i).label}</option>`).join('');}
    function rows(filter=''){if(!result)return '';return Array.from({length:1351},(_,i)=>i).filter(i=>core.describe(i).label.toLowerCase().includes(filter.toLowerCase())).map(i=>`<tr><td><button data-trace-parameter="${i}">${core.describe(i).label}</button></td><td>${num(result.before[i])}</td><td>${step?num(result.gradient[i]):'—'}</td><td>${step?num(result.after[i]-result.before[i]):'—'}</td><td>${num(result.after[i])}</td></tr>`).join('');}
    function render(){
      tabs.hidden=!entry;host.hidden=!entry||!enabled;
      tabs.innerHTML=`<button type="button" data-trace-mode="forward" aria-pressed="${!enabled}">前向计算</button><button type="button" data-trace-mode="replay" aria-pressed="${enabled}">反向传播回放</button>${enabled?'<button type="button" data-trace-fullscreen>讲解全屏</button>':''}`;
      if(host.hidden)return;
      const scroll=host.scrollTop,stages=manifest.stages,prev=[...stages].reverse().find(t=>t<step)??0,next=stages.find(t=>t>step)??manifest.steps;
      const stage=step===0?0:stages.findIndex(t=>t>=step),c=config();
      let html=`<div class="trace-sticky"><header class="trace-header"><div><span class="trace-eyebrow">真实训练记录 · ${manifest.steps.toLocaleString()} 次更新</span><h2>${manifest.initialization?'从随机权重到清晰特征':'逐次查看参数如何学习'}</h2></div></header>
      <div class="trace-stages" role="group" aria-label="十个讲解阶段">${stages.map((t,i)=>`<button type="button" data-trace-seek="${t}" aria-pressed="${step===t}">${i||'初始'}<small>${t.toLocaleString()}</small></button>`).join('')}</div>
      <div class="trace-navigation"><button data-trace-seek="${prev}" ${step===0?'disabled':''}>上一阶段</button><button data-trace-seek="${next}" ${step===manifest.steps?'disabled':''}>下一阶段</button><button data-trace-seek="${Math.max(0,step-1)}" ${step===0?'disabled':''}>上一次更新</button><input id="trace-step" aria-label="更新编号" type="number" min="0" max="${manifest.steps}" value="${step}"><button data-trace-action="jump">跳转</button><button data-trace-seek="${Math.min(manifest.steps,step+1)}" ${step===manifest.steps?'disabled':''}>下一次更新</button></div>
      <p data-trace-position class="trace-position">${step===0?'初始化 · 尚未执行反向传播':`阶段 ${stage} / 10 · 实际更新 ${step.toLocaleString()} / ${manifest.steps.toLocaleString()}`} · ${c.name||'联合训练'}</p></div>`;
      if(busy||error||!result){host.innerHTML=html+`<div class="trace-status ${error?'trace-error':''}" role="status">${error?escape(error):'正在加载、重建并校验历史记录…'}${error?'<button data-trace-action="retry">重试</button><a href="https://passwordis42.github.io/NeronNetRectangle/downloads/neural-network-offline.zip">下载完整离线包 ↗</a>':''}</div>`;host.scrollTop=scroll;return;}
      const p=parameterInfo(parameter),d=core.describe(parameter),j=d.layer===1?d.neuron:(d.input??20),s=result.details?.sample,obs=core.forward(observed,currentModel);
      if(s)html+=`        <div class="trace-sample-control">${grid(s.pixels,'选中的历史训练样本')}<label>本次训练批次 · 样本 <select id="trace-sample">${Array.from(result.ids,(id,i)=>`<option value="${i}" ${i===sampleIndex?'selected':''}>${i+1} / ${result.ids.length} · ${i<result.positives?'正例':'负例'} · ${id}</option>`).join('')}</select></label><p>${result.positives} 个正例，${result.ids.length-result.positives} 个负例；重复样本分别计入批次平均。</p></div>`;
      html+=`<div class="trace-toolbar"><label>观察画布参数 <select id="trace-phase"><option value="before" ${phase==='before'?'selected':''}>本次更新前</option><option value="after" ${phase==='after'?'selected':''}>本次更新后</option>${step===manifest.steps?`<option value="published" ${phase==='published'?'selected':''}>正式模型 · 六位小数</option>`:''}</select></label><label><input id="trace-precision" type="checkbox" ${precision?'checked':''}>完整精度</label></div>
      <div class="trace-observation">${grid(observed,'当前观察画布')}<div><b data-trace-observed-score>观察得分 ${num(obs.outputs[0])}</b><p>画布编辑只改变前向观察，不改变历史样本与梯度。</p><button data-trace-action="canvas">展开／收起观察画布</button></div></div>
      <div class="trace-toolbar"><label>神经元 <select id="trace-neuron">${Array.from({length:51},(_,i)=>`<option value="${i}" ${i===(d.layer===1?d.neuron:50)?'selected':''}>${i===50?'输出神经元':`H1-${i+1}`}</option>`).join('')}</select></label><label>连接或偏置 <select id="trace-parameter">${parameterOptions()}</select></label></div>
      <div class="trace-update" data-trace-update><div><small>更新前</small><b>${num(p.before)}</b></div><span>→</span><div><small>更新后</small><b>${num(p.after)}</b></div><div><small>实际 Δw</small><b>${num(p.delta)}</b></div></div>
      <div class="trace-calculation-tabs">${[['forward','前向'],['loss','损失'],['backward','反传'],['update','更新']].map(([id,label])=>`<button data-trace-section="${id}" aria-pressed="${section===id}">${label}</button>`).join('')}</div>`;
      if(s){
        const sourceValue=p.gradient,sg=core.sampleGradient(s,parameter),lossIndex=actualSource()==='concept'?1:0;
        html+=`<section class="trace-section" data-trace-pane="forward" ${section!=='forward'?'hidden':''}><h3>前向计算 · 本次训练样本</h3><div class="trace-sample">${grid(s.pixels,'选中的历史训练样本')}<div><b>${s.positive?'正例 · 完整方框':'负例'}</b><p>使用第 ${step} 次更新前的参数。H1-${j+1}：z = ${num(s.preActivations[0][j])}，h = ${num(s.activations[1][j])}；得分 ${num(s.outputs[0])}。</p></div></div><button data-trace-action="sample-details">${sampleDetails?'收起':'展开'}全部 50 个神经元</button>${sampleDetails?`<div class="trace-table-scroll"><table><thead><tr><th>神经元</th><th>z</th><th>h</th><th>1−h²</th><th>分类 δ/N</th><th>特征 δ/N</th></tr></thead><tbody>${s.activations[1].map((h,i)=>`<tr><td>H1-${i+1}</td><td>${num(s.preActivations[0][i])}</td><td>${num(h)}</td><td>${num(1-h*h)}</td><td>${num(s.dzClass[i])}</td><td>${num(s.dzConcept[i])}</td></tr>`).join('')}</tbody></table></div>`:''}</section>
        <section class="trace-section" data-trace-pane="loss" ${section!=='loss'?'hidden':''}><h3>四类损失及梯度</h3><p>正例目标 ≥ 1.6，负例目标 ≤ 0.4；推理判断仍为得分 &gt; 1。未启用的损失不参与本阶段更新。</p><div class="trace-loss-table"><table><thead><tr><th>损失</th><th>原始值</th><th>阶段系数</th><th>有效值</th><th>参数梯度</th></tr></thead><tbody>${names.map((name,i)=>`<tr><th>${name}${c.lossWeights[i]===0?' · 未启用':''}</th><td>${num(result.details.rawLosses[i])}</td><td>${c.lossWeights[i]}</td><td>${num(result.losses[i])}</td><td>${num(result.details.parts[i][parameter])}</td></tr>`).join('')}</tbody><tfoot><tr><th>合计</th><td colspan="3">${num(result.losses.reduce((a,b)=>a+b,0))}</td><td data-trace-total-gradient>${num(result.gradient[parameter])}</td></tr></tfoot></table></div><p>特征监督原始值已含 0.2 系数；边界间隔 ±0.55；四边贡献目标 ≥0.14。阶段系数在此基础上应用。</p></section>
        <section class="trace-section" data-trace-pane="backward" ${section!=='backward'?'hidden':''}><h3>偏导如何传到 ${escape(d.label)}</h3><p><b>${sourceName()}</b> · ${step} 次更新前 · 样本 ${sampleIndex+1}</p>
        ${['total','boundary','composition'].includes(actualSource())?`<div class="trace-equation">∂L/∂θ = 分类＋特征监督＋边界约束＋四边贡献<br>${result.details.parts.map(part=>num(part[parameter])).join(' + ')} = ${num(p.totalGradient)}${actualSource()!=='total'?`<br>当前选中 ${sourceName()} = ${num(p.rawGradient)} × 阶段系数 ${p.coefficient} = ${num(sourceValue)}<br>约束从特征边界或六个标准方框计算，不属于当前单个训练样本，无需再除以训练批次数。`:''}</div>`:`<div class="trace-equation">${actualSource()==='classification'?`∂ℓ分类/∂s = 2 × 间隔误差 = ${num(2*s.error)}<br>δH1-${j+1} = ${num(2*s.error)} × ${num(result.before[1300+j])} × (1 − ${num(s.activations[1][j])}²) = ${num(s.dzClass[j]*result.ids.length)}`:d.layer===2||s.targets[j]===null?'该参数没有局部特征监督目标。':`局部目标 t = ${num(s.targets[j])}<br>δ特征,H1-${j+1} = 0.4 × (h − t) × (1 − h²) = ${num(s.dzConcept[j]*result.ids.length)}`}<br>${d.layer===2?actualSource()==='concept'?'局部特征监督不依赖输出层参数，此项偏导为 0':`输出${d.kind==='bias'?'偏置':'权重'}偏导 = 2 × 间隔误差 × ${d.kind==='bias'?'1':'对应隐藏响应'}`:`${d.kind==='bias'?'偏置偏导 = δ':'输入权重偏导 = δ × 对应输入'}`}<br>单样本原始偏导 = ${num(p.rawGradient)}<br>× 阶段系数 ${p.coefficient} = ${num(sourceValue)}<br>÷ 批次 ${result.ids.length} = 对批次梯度的贡献 ${num(p.batchContribution)}</div>`}
        <p>${sourceValue===0?(p.coefficient===0?'本阶段未启用该损失，因此不参与更新。':actualSource()==='concept'&&(d.layer===2||s.targets[d.neuron]===null)?'该参数没有局部特征监督目标，因此此项偏导为零。':['boundary','composition'].includes(actualSource())?'约束当前已满足，或该参数不参与这项约束。':actualSource()==='classification'&&s.error===0?'该样本已达到分类间隔，所以分类偏导为零。':'偏导为零：可能是对应输入为零，或链式法则中的因子为零。'):'绿色／红色表示偏导正负；Adam 实际改变量还受总梯度和动量影响。'}</p>
        <button data-trace-action="find" ${!p.isSingle?'disabled':''}>查看本批次中偏导非零的样本</button><p role="status">${notice}</p></section>
        <section class="trace-section" data-trace-pane="update" ${section!=='update'?'hidden':''}><h3>Adam · 使用批次总梯度</h3><p>g = ${num(p.totalGradient)}；学习率 ${c.learningRate}。单样本偏导不直接等于实际 Δw；零梯度也可能因动量继续更新。</p><button data-trace-action="adam">${expandedAdam?'收起':'展开'} Adam 与裁剪</button>${expandedAdam?`<div class="trace-equation">m = 0.9 × ${num(result.oldM[parameter])} + 0.1 × ${num(p.totalGradient)} = ${num(result.m[parameter])}<br>v = 0.999 × ${num(result.oldV[parameter])} + 0.001 × g² = ${num(result.v[parameter])}<br>m̂ = m / (1−0.9^${step}) = ${num(result.correctedM[parameter])}<br>v̂ = v / (1−0.999^${step}) = ${num(result.correctedV[parameter])}<br>候选值 = ${num(p.before)} − ${c.learningRate} × m̂ / (√v̂ + 1e−8) = ${num(result.proposed[parameter])}<br>裁剪到 [−10,10] → ${num(p.after)} · ${result.clipped[parameter]?'发生裁剪':'未裁剪'}</div>`:''}</section>
`;
      }else html+=`<section class="trace-section"><h3>初始化 · 尚未反向传播</h3><p>${manifest.initialization?'两层权重独立采样 U(−0.1,0.1)，偏置为零。初始权重没有预先填入边或转角；训练目标仍由人工指定。':'初始参数继承已有模型，并初始化新增局部特征。'}</p><p>此时没有损失、梯度或实际改变量。${manifest.initialization?'前 6,000 次只训练局部特征；随后联合训练分类与组合，Adam 状态连续保留。':''}</p></section>`;
      if(step===manifest.steps)html+=`<section class="trace-release"><h3>${manifest.steps.toLocaleString()} 次更新完成</h3><p>保留六位小数后，与当前所选正式模型逐项一致；精度处理不是额外训练。</p><button data-trace-action="return">回到真实训练教学</button></section>`;
      html+=`<div class="trace-toolbar"><button data-trace-action="table">${fullTable?'收起':'查询'}全部 1,351 个参数</button><button data-trace-action="json">导出本步 JSON</button><button data-trace-action="csv">导出参数 CSV</button><button data-trace-action="return">回到真实训练教学</button></div>${fullTable?`<input id="trace-filter" aria-label="筛选参数" placeholder="筛选参数，如 H1-21 或 偏置"><div class="trace-table-scroll"><table><thead><tr><th>参数</th><th>更新前</th><th>总梯度</th><th>实际 Δw</th><th>更新后</th></tr></thead><tbody id="trace-parameter-rows">${rows()}</tbody></table></div>`:''}<p class="hint">六位小数显示，非零小量使用科学计数法。计算和导出保留完整精度。</p><a href="https://passwordis42.github.io/NeronNetRectangle/downloads/neural-network-offline.zip" download>下载完整离线包 ↗</a>`;
      host.innerHTML=html;host.scrollTop=scroll;
    }
    function download(type){
      if(!result)return;let text,mime;
      if(type==='csv'){text='\ufeff参数,更新前,总梯度,实际改变量,更新后,显示偏导,偏导来源,阶段系数\r\n'+Array.from({length:1351},(_,i)=>{const p=parameterInfo(i);return[core.describe(i).label,p.before,p.totalGradient??'',p.delta??'',p.after,p.gradient??'',sourceName(),p.coefficient].join(',');}).join('\r\n');mime='text/csv;charset=utf-8';}
      else{text=JSON.stringify({traceId:manifest.id,modelSha256:manifest.modelSha256,step,optimizer:config(),observation:{pixels:observed,phase},selectedParameter:core.describe(parameter),display:{source:actualSource(),metric,networkSource,parameter:parameterInfo(parameter)},initialization:step===0,...result},(_,v)=>ArrayBuffer.isView(v)?Array.from(v):v,2);mime='application/json';}
      const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type:mime}));a.href=url;a.download=`square-backprop-${version()}-step-${step}.${type}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    function setEnabled(value){enabled=Boolean(value&&entry);if(enabled){const d=core.describe(parameter);options.onSelect(d.layer,d.neuron);seek(step,sampleIndex,true);}else{++requestId;busy=false;result=null;currentModel=null;notify();}}
    tabs.addEventListener('click',async e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.traceMode)setEnabled(b.dataset.traceMode==='replay');if(b.hasAttribute('data-trace-fullscreen')){try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch(_){notice='浏览器未允许全屏，可使用 F11。';render();}}});
    host.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.traceSeek!==undefined){seek(Number(b.dataset.traceSeek));return;}if(b.dataset.traceParameter!==undefined){selectParameter(Number(b.dataset.traceParameter));return;}if(b.dataset.traceSection){section=b.dataset.traceSection;save();render();return;}switch(b.dataset.traceAction){case'jump':seek(Number(host.querySelector('#trace-step').value));break;case'retry':seek(step);break;case'return':setEnabled(false);break;case'find':seek(step,sampleIndex,true,true);break;case'table':fullTable=!fullTable;render();break;case'adam':expandedAdam=!expandedAdam;render();break;case'sample-details':sampleDetails=!sampleDetails;render();break;case'canvas':document.getElementById('replayCanvasFold').open=!document.getElementById('replayCanvasFold').open;break;case'json':case'csv':download(b.dataset.traceAction);break;}});
    host.addEventListener('change',e=>{switch(e.target.id){case'trace-phase':phase=e.target.value;setModel();save();notify();break;case'trace-precision':precision=e.target.checked;save();notify();break;case'trace-parameter':selectParameter(Number(e.target.value));break;case'trace-neuron':selectParameter(Number(e.target.value)===50?1350:Number(e.target.value)*25+6);break;case'trace-sample':seek(step,Number(e.target.value),true);break;}});
    host.addEventListener('keydown',e=>{if(e.target.id==='trace-step'&&e.key==='Enter')seek(Number(e.target.value));});
    host.addEventListener('input',e=>{if(e.target.id==='trace-filter')host.querySelector('#trace-parameter-rows').innerHTML=rows(e.target.value);});
    return {
      get active(){return enabled&&Boolean(entry);},get model(){return currentModel;},get loading(){return busy;},get parameter(){return parameter;},get metric(){return metric;},get zoom(){return zoom;},get layout(){return layout;},get zoomMode(){return zoomMode;},format:num,
      get networkModel(){return result&&networkSource==='sample'?core.model(result.before):currentModel;},get networkPixels(){return networkSource==='sample'&&result?.details?result.details.sample.pixels:observed;},
      get context(){return {step,source:actualSource(),sourceName:sourceName(),metric,networkSource,sampleIndex,hasGradient:Boolean(step&&result),phase:networkSource==='sample'?'before':phase};},
      parameterInfo,metricValue,
      setView(kind,value){
        if(kind==='zoom'||kind==='fit'){
          if(kind==='zoom'&&!Number.isFinite(Number(value)))return;
          zoomMode=kind==='fit'?'fit':'manual';
          if(kind==='zoom')zoom=Math.max(.1,Math.min(3,Number(value)));
          save();options.onViewportChange?.();return;
        }
        if(kind==='metric'&&['weight','gradient','delta'].includes(value))metric=value;
        if(kind==='source'&&['auto','classification','concept','boundary','composition','total'].includes(value))source=value;
        if(kind==='network'&&['sample','observation'].includes(value))networkSource=value;
        if(kind==='layout'&&['compact','vertical','expanded'].includes(value))layout=value;
        save();notify();
      },
      updateFitZoom(value){zoom=Math.max(.1,Math.min(3,value));},
      renderToolbar(){if(!manifest)return '';return `<div class="trace-network-toolbar"><label>网络观察 <select data-replay-view="network"><option value="sample" ${networkSource==='sample'?'selected':''}>训练样本 · 更新前</option><option value="observation" ${networkSource==='observation'?'selected':''}>观察画布 · 所选参数</option></select></label><label>数值 <select data-replay-view="metric">${[['weight','权重 w'],['gradient','偏导'],['delta','实际改变量 Δw']].map(([v,t])=>`<option value="${v}" ${metric===v?'selected':''}>${t}</option>`).join('')}</select></label><label>偏导来源 <select data-replay-view="source">${[['auto','单样本 · 当前训练目标'],['classification','单样本 · 分类损失'],['concept','单样本 · 特征监督'],['boundary','检测边界约束'],['composition','四边贡献约束'],['total','本次更新 · 总梯度']].map(([v,t])=>`<option value="${v}" ${source===v?'selected':''}>${t}</option>`).join('')}</select></label><label>布局 <select data-replay-view="layout">${[['compact','横向紧凑'],['vertical','纵向三层'],['expanded','逐个展开']].map(([v,t])=>`<option value="${v}" ${layout===v?'selected':''}>${t}</option>`).join('')}</select></label><span>${!result?'正在加载并校验所选更新，暂不显示历史数值':step===0?'初始化，无偏导':`更新 ${step.toLocaleString()} · ${sourceName()}${['classification','concept'].includes(actualSource())?` · 样本 ${sampleIndex+1}`:''}`}</span></div>`;},
      sync(state){const next=entries.find(e=>e.manifest.presetId===state.teachingPreset)||null;if(next!==entry)switchEntry(next);observed=state.pixels.slice();render();},
      selectNeuron(layer,index){if(enabled&&[1,2].includes(layer))selectParameter(layer===2?1350:index*25+6,false);},
      selectConnection(layer,neuron,input){if(enabled)selectParameter(layer===1?neuron*25+input:1300+input);},selectParameter
    };
  }
  return {create};
})();
