/* Read-only replay for the sparse, fully explained geometry models. */
const SquareGeometryBackpropTeaching = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const array = value => Array.from(value || []);
  const pixelGrid = pixels => `<span class="trace-pixels" role="img" aria-label="5×5 输入">${array(pixels).map(v=>`<i class="${v?'is-on':''}"></i>`).join('')}</span>`;
  function create(options) {
    const {host,tabs,entries=[]}=options;
    let entry=null,manifest=null,artifact=null,core=null,enabled=false,result=null,currentModel=null,beforeModel=null;
    let step=0,parameter=0,fixed=null,sampleIndex=0,precision=false,source='classification',scope='sample',metric='weight';
    let networkSource='sample',phase='before',section='backward',zoom=1,layout='compact',zoomMode='manual';
    let observed=Array(25).fill(0),busy=false,error='',notice='',requestId=0,worker=null,workerURL=null;
    let fullTable=false,filter='',traceCache=null,auditRow=0,auditOpen=false;
    const chunks=new Map(),pending=new Map(),infoCache=new Map();
    const num=v=>Number.isFinite(v)?precision?String(v):v!==0&&Math.abs(v)<.0000005?v.toExponential(6):v.toFixed(6):'—';
    const names=()=>core?.lossDefinitions||manifest?.lossDefinitions||manifest?.losses||[];
    const visibleNames=()=>names().filter(d=>d.id!=='geometry'||artifact?.model.layers.length===3);
    const lossName=d=>d.name||d.label||d.id;
    const sourceIndex=()=>names().findIndex(d=>d.id===source);
    const config=()=>core?.optimizerAt(step)||manifest?.optimizer||{};
    const storeKey=()=>`nn-square-geometry-replay:${manifest.id}`;
    const notify=()=>options.onChange();
    function save(){if(!manifest)return;try{localStorage.setItem(storeKey(),JSON.stringify({step,parameter,sampleIndex,precision,source,scope,metric,networkSource,phase,section,zoom,layout,zoomMode}));}catch(_){}}
    function describe(i){return core.describe(i);}
    function parameterIndex(layer,neuron,input=null){
      if(!core)return -1;
      if(core.indexOf)return core.indexOf(layer,neuron,input,input===null?'bias':'weight');
      return core.parameterMap.findIndex(d=>d.layer===layer&&d.neuron===neuron&&d.kind===(input===null?'bias':'weight')&&(input===null||d.input===input));
    }
    function currentInput(){return networkSource==='sample'&&result?.details?.sample?array(result.details.sample.pixels):observed;}
    function networkModel(){return result&&networkSource==='sample'?beforeModel:currentModel;}
    function sourceName(){if(source==='total')return '本次更新 · 批次总梯度';if(source==='participation')return '全部输出连接 · 参与约束';if(source==='boundary'&&scope==='sample')return artifact?.model.layers.length===3?'单样本 · 几何边界部分（不含全局局部边界）':'局部边界约束 · 非单样本';return `${scope==='sample'?'单样本':'整批次／全部约束'} · ${lossName(names()[sourceIndex()]||{id:source})}`;}
    function switchEntry(next){
      ++requestId;enabled=false;busy=false;result=null;currentModel=null;traceCache=null;fixed=null;error='';entry=next;manifest=next?.manifest;artifact=next?.artifact;
      if(worker){worker.terminate();worker=null;URL.revokeObjectURL(workerURL);}
      if(!next)return;
      core=GeometryBackpropCore.create(manifest,artifact);
      let saved={};try{saved=JSON.parse(localStorage.getItem(storeKey())||'{}');}catch(_){}
      step=Number.isInteger(saved.step)?Math.max(0,Math.min(manifest.steps,saved.step)):0;
      parameter=Number.isInteger(saved.parameter)?Math.max(0,Math.min(core.P-1,saved.parameter)):Math.max(0,parameterIndex(1,60,6));
      sampleIndex=Number.isInteger(saved.sampleIndex)?saved.sampleIndex:null;precision=!!saved.precision;
      source=saved.source==='total'||names().some(d=>d.id===saved.source)?saved.source:names()[0]?.id||'classification';
      scope=saved.scope==='batch'?'batch':'sample';metric=['weight','gradient','delta'].includes(saved.metric)?saved.metric:'weight';
      networkSource=saved.networkSource==='observation'?'observation':'sample';phase=['before','after','published'].includes(saved.phase)?saved.phase:'before';
      section=['forward','loss','backward','update'].includes(saved.section)?saved.section:'backward';
      zoom=Math.max(.1,Math.min(3,Number(saved.zoom)||1));layout=['compact','vertical','expanded'].includes(saved.layout)?saved.layout:'compact';zoomMode=saved.zoomMode==='fit'?'fit':'manual';
    }
    function setModel(){
      infoCache.clear();beforeModel=null;
      if(!result){currentModel=null;return;}
      beforeModel=core.model(result.before);
      if(phase==='published'&&step===manifest.steps){
        const expected=core.flatten(artifact.model);
        if(array(result.after).some((v,i)=>Number(v.toFixed(6))!==expected[i]))throw Error('训练终点与正式模型不一致，停止显示。');
        currentModel=artifact.model;
      }else currentModel=phase==='after'?core.model(result.after):beforeModel;
    }
    function loadChunk(index){
      const key=`${manifest.id}:${index}`;
      if(chunks.has(key))return Promise.resolve(chunks.get(key));
      if(pending.has(key))return pending.get(key).promise;
      let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;}),script=document.createElement('script');
      const item={resolve,reject,promise,script,timer:null};
      const fail=()=>{if(pending.get(key)!==item)return;pending.delete(key);clearTimeout(item.timer);script.remove();reject(Error('训练记录加载失败。请重试；离线使用请解压完整离线包，保留 training-trace 文件夹。'));};
      item.timer=setTimeout(fail,45000);script.onerror=fail;pending.set(key,item);script.src=manifest.dataPath+manifest.chunks[index].file;document.head.append(script);return promise;
    }
    const previousReceiver=globalThis.SquareTraceChunks?.receive;
    globalThis.SquareTraceChunks={receive(id,index,payload){
      const key=`${id}:${index}`,item=pending.get(key);if(!item){previousReceiver?.(id,index,payload);return;}
      pending.delete(key);clearTimeout(item.timer);item.script.remove();chunks.set(key,payload);if(chunks.size>4)chunks.delete(chunks.keys().next().value);item.resolve(payload);
    }};
    function ensureWorker(){
      if(worker)return;
      workerURL=URL.createObjectURL(new Blob([`const makeCore=(${createGeometryBackpropCore.toString()});const cache=new Map();
onmessage=async({data:d})=>{try{const core=makeCore(d.manifest,d.artifact);let chunk=cache.get(d.key);
if(!chunk){const bytes=Uint8Array.from(atob(d.base64),c=>c.charCodeAt(0));const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');if(hash!==d.hash)throw Error('训练记录校验失败，请重试或重新下载。');const text=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();chunk=core.readChunk(text);cache.set(d.key,chunk);if(cache.size>4)cache.delete(cache.keys().next().value);}
const r=core.reconstruct(chunk,d.step);if(d.step){const center=Array.from(r.ids).findIndex((id,i)=>id===469440&&r.labels[i]===1),positive=Array.from(r.labels).findIndex(v=>v===1);let selected=Number.isInteger(d.sample)?Math.min(Math.max(0,d.sample),r.ids.length-1):Math.max(0,center>=0?center:positive);
if(d.find){const found=core.findNonzeroSample(r.before,r,d.step,d.parameter,d.source,selected);r.found=found;if(found>=0)selected=found;}
r.details=core.gradients(r.before,r,d.step,selected);let max=0;for(let i=0;i<core.P;i++)max=Math.max(max,Math.abs(r.details.total[i]-r.gradient[i]));r.gradientError=max;if(max>1e-8)throw Error('重建梯度与训练记录不一致。');if(r.details.losses.some((v,i)=>Math.abs(v-r.losses[i])>1e-8))throw Error('重建损失与训练记录不一致。');}
postMessage({id:d.id,result:r});}catch(e){postMessage({id:d.id,error:e.message});}};`],{type:'text/javascript'}));
      worker=new Worker(workerURL);
      worker.onmessage=({data})=>{if(data.id!==requestId)return;busy=false;if(data.error){error=data.error;result=null;currentModel=null;chunks.clear();}else{result=data.result;sampleIndex=result.details?.sample?.index||0;traceCache=null;notice=result.found===-1?'本批次中，该参数没有此来源的非零单样本偏导。':result.found>=0?'已定位一个偏导非零的训练样本。':'';try{setModel();save();}catch(e){error=e.message;result=null;currentModel=null;}}notify();};
      worker.onerror=()=>{busy=false;error='计算线程运行失败，请重试。';result=null;currentModel=null;worker.terminate();worker=null;URL.revokeObjectURL(workerURL);notify();};
    }
    async function seek(next,selected=null,keepPhase=false,find=false){
      if(!entry||!enabled)return;
      step=Math.max(0,Math.min(manifest.steps,Math.round(Number(next)||0)));if(!keepPhase)phase='before';
      const index=manifest.chunks.findIndex(c=>step===0?c.firstStep===1:c.firstStep<=step&&step<=c.lastStep),id=++requestId;
      busy=true;error='';notice='';result=null;currentModel=null;traceCache=null;sampleIndex=selected;auditRow=0;save();notify();
      try{if(index<0)throw Error('记录中不存在该更新，停止显示。');if(!globalThis.Worker||!globalThis.DecompressionStream||!globalThis.crypto?.subtle)throw Error('浏览器不支持训练记录解压，请使用新版浏览器。');
        ensureWorker();const base64=await loadChunk(index);if(id!==requestId||!enabled)return;
        worker.postMessage({id,key:`${manifest.id}:${index}`,base64,hash:manifest.chunks[index].sha256,step,sample:selected,manifest,artifact,find,parameter,source});
      }catch(e){if(id===requestId){busy=false;error=e.message;notify();}}
    }
    function samplePart(index,kind=sourceIndex()){
      if(!result?.details||kind<0)return null;
      return core.sampleGradient(result.details.sample,index,names()[kind].id,result.details);
    }
    function parameterInfo(index){
      if(!result)return null;
      if(index<0)return {index:-1,fixed:true,before:0,after:0,weight:0,gradient:null,rawGradient:null,totalGradient:null,delta:0,coefficient:0,batchContribution:null,sourceName:'结构约束 · 固定为零',metric};
      if(infoCache.has(index))return infoCache.get(index);
      const d=describe(index),k=sourceIndex();let single=scope==='sample'&&source!=='total'&&source!=='participation'&&(source!=='boundary'||artifact.model.layers.length===3)?samplePart(index,k):null;
      if(single?.constraintOnly)single=null;
      const coeff=result.details?.lossWeights?.[k]??names()[k]?.coefficient??1;
      const batch=source==='total'?result.gradient?.[index]:result.details?.parts?.[k]?.[index];
      const raw=single?.raw??single?.rawGradient??single?.rawDerivative;
      const weight=networkSource==='sample'||phase==='before'?result.before[index]:phase==='published'&&step===manifest.steps?(d.kind==='bias'?artifact.model.layers[d.layer-1].biases[d.neuron]:artifact.model.layers[d.layer-1].weights[d.neuron][d.input]):result.after[index];
      const info={index,...d,before:result.before[index],after:result.after[index],weight,
        gradient:step?(single?raw*coeff:batch):null,rawGradient:raw,coefficient:coeff,isSingle:!!single,batchContribution:single?.contribution??single?.weightedGradient,
        totalGradient:step?result.gradient[index]:null,delta:step?result.after[index]-result.before[index]:null,step,source,sourceName:sourceName(),sampleIndex,metric};
      infoCache.set(index,info);return info;
    }
    function selectParameter(index,select=true){
      if(!Number.isInteger(Number(index))||index<0||index>=core.P)return;
      parameter=Number(index);fixed=null;traceCache=null;auditRow=0;save();const d=describe(parameter);if(select)options.onSelect(d.layer,d.neuron);else notify();
    }
    function selectConnection(layer,neuron,input){const index=parameterIndex(layer,neuron,input);if(index<0){fixed={layer,neuron,input,kind:'weight',label:`${layer===1?'X':'H'+(layer-1)+'-'}${input+1} → ${layer===artifact.model.layers.length?'输出':'H'+layer+'-'+(neuron+1)}`};options.onSelect(layer,neuron);notify();}else selectParameter(index);}
    function selectNeuron(layer,neuron){if(!enabled||fixed?.layer===layer&&fixed.neuron===neuron)return;const current=describe(parameter);if(current.layer===layer&&current.neuron===neuron)return;const idx=core.parameterMap.findIndex(d=>d.layer===layer&&d.neuron===neuron);if(idx>=0)selectParameter(idx,false);}
    function provenance(){
      if(!step||!result?.details||fixed)return null;
      if(traceCache?.result===result&&traceCache.index===parameter)return traceCache.trace;
      const trace=core.parameterTrace(result.before,result,step,parameter,sampleIndex);
      traceCache={result,index:parameter,trace};return trace;
    }
    function chainView(chain){
      if(!chain)return '';
      if(typeof chain==='string')return `<div class="trace-equation">${esc(chain)}</div>`;
      return `<div class="trace-equation">${array(chain).map(t=>typeof t==='string'?esc(t):`${esc(t.formula||t.label||'')}<br>${t.substitution?esc(t.substitution)+' = ':''}<b>${num(t.value)}</b>`).join('<hr>')}</div>`;
    }
    function lossView(){
      const defs=visibleNames(),detail=result.details,index=d=>names().findIndex(n=>n.id===d.id);
      return `<h3>损失的完整数值计算</h3><div class="trace-equation">L = ${defs.map(d=>`λ${esc(lossName(d))} L${esc(lossName(d))}`).join(' + ')}<br>= ${defs.map(d=>`${num(detail.lossWeights[index(d)])} × ${num(detail.rawLosses[index(d)])}`).join(' + ')}<br>= ${defs.map(d=>num(detail.losses[index(d)])).join(' + ')}<br>= <b>${num(array(detail.losses).reduce((a,b)=>a+b,0))}</b></div>${defs.map(d=>{const k=index(d);return `<details class="trace-loss-definition"><summary>${esc(lossName(d))} = ${num(detail.losses[k])}</summary><p>${esc(d.formula||d.description||'')}</p><p>${esc(d.normalization||'归一化按本项定义计算')}；系数 ${num(detail.lossWeights[k])}</p>${detail.lossWeights[k]===0?'<p>本阶段不参与更新。</p>':''}</details>`;}).join('')}${auditView()}`;
    }
    function auditView(){
      if(!auditOpen)return '<button data-geometry-action="audit">展开所选参数的全部来源</button>';
      const trace=provenance();if(!trace)return '';
      const k=Math.max(0,sourceIndex()),rows=trace.rows?.[k]||[],row=rows[Math.min(auditRow,rows.length-1)];
      return `<details open class="trace-audit"><summary>追溯 ${esc(describe(parameter).label)} 的全部来源</summary><div class="trace-equation">∂L/∂θ = ${array(trace.parts||result.details.parts.map(p=>p[parameter])).map(num).join(' + ')} = ${num(result.gradient[parameter])}</div><label>损失来源 <select data-geometry-source>${visibleNames().map(d=>`<option value="${d.id}" ${d.id===names()[k].id?'selected':''}>${esc(lossName(d))}</option>`).join('')}</select></label>${rows.length?`<label>逐项查看 <select data-geometry-audit-row>${rows.map((r,i)=>`<option value="${i}" ${i===auditRow?'selected':''}>${esc(r.label||r.name||`第 ${i+1} 项`)} · ${num(r.gradient??r.weightedGradient??r.rawGradient)}</option>`).join('')}</select></label>${row?.pixels?pixelGrid(row.pixels):''}${chainView(row?.chain)}<p>该项损失 ${num(row?.loss)}；原始偏导 ${num(row?.rawDerivative??row?.rawGradient)}；最终贡献 ${num(row?.gradient??row?.weightedGradient)}</p>`:'<p>该损失没有依赖此参数的项，偏导为零。</p>'}</details>`;
    }
    function backwardView(){
      if(fixed)return '<h3>固定连接</h3><p>此连接不在定义的观察范围内，结构规定权重为零。它不属于优化器参数，未进行 Adam 更新。</p>';
      const info=parameterInfo(parameter),k=sourceIndex();let single=scope==='sample'&&k>=0?samplePart(parameter,k):null;
      if(single?.constraintOnly)single=null;
      let body='';
      if(source==='total')body=`<div class="trace-equation">∂L/∂θ = ${visibleNames().map(d=>`∂L${esc(lossName(d))}/∂θ`).join(' + ')}<br>= ${visibleNames().map(d=>num(result.details.parts[names().findIndex(n=>n.id===d.id)][parameter])).join(' + ')}<br>= <b>${num(info.totalGradient)}</b></div>`;
      else if(single)body=(single.note?`<p>${esc(single.note)}</p>`:'')+chainView(single.chain)+(single.rows?.length>1?`<details><summary>展开本样本的 ${single.rows.length} 个相关项</summary>${single.rows.map((r,i)=>`<h4>第 ${i+1} 项 · ${esc(r.label||r.name||'')}</h4>${chainView(r.chain)}`).join('')}</details>`:'')+`<div class="trace-equation">单样本原始偏导 = ${num(single.raw??single.rawDerivative??single.rawGradient)}<br>× 损失系数 ${num(info.coefficient)}<br>÷ 归一化分母 ${num(single.denominator??result.ids.length)}<br>= 对实际总梯度的贡献 ${num(single.contribution??single.weightedGradient)}</div>`;
      else body=`<div class="trace-equation">该来源的全部项相加：∂L/∂θ = ${num(info.gradient)}</div><p>该来源是整批次或结构约束，不能当作当前单个训练样本的偏导。</p>`;
      return `<h3>链式法则 · ${esc(info.label)}</h3><p>${esc(sourceName())} · 更新前参数 · 样本 ${sampleIndex+1}</p>${body}${info.gradient===0?'<p>当前来源对该参数的偏导为零。请查看各链式因子，或切换到总梯度；零梯度仍可能因 Adam 动量产生实际更新。</p>':''}<p>实际交给 Adam 的总梯度：${num(info.totalGradient)}；实际 Δw：${num(info.delta)}。</p><button data-geometry-action="find" ${scope!=='sample'||source==='total'?'disabled':''}>查找偏导非零的训练样本</button><p role="status">${esc(notice)}</p>${auditView()}`;
    }
    function updateView(){const p=parameterInfo(parameter),o=config();return `<h3>Adam 与结构约束</h3><div class="trace-equation">g = ${num(p.totalGradient)}<br>m = ${o.beta1} × ${num(result.oldM?.[parameter])} + ${num(1-o.beta1)} × g = ${num(result.m?.[parameter])}<br>v = ${o.beta2} × ${num(result.oldV?.[parameter])} + ${num(1-o.beta2)} × g² = ${num(result.v?.[parameter])}<br>m̂ = ${num(result.correctedM?.[parameter])}；v̂ = ${num(result.correctedV?.[parameter])}<br>Adam 候选值 = ${num(p.before)} − ${num(o.learningRate)} × m̂ / (√v̂ + ${num(o.epsilon)}) = ${num(result.proposed?.[parameter])}<br>约束后的值 = ${num(p.after)}<br>约束改变量 = ${num(p.after-result.proposed?.[parameter])}<br>实际 Δw = ${num(p.after)} − ${num(p.before)} = <b>${num(p.delta)}</b></div><p>${result.clipped?.[parameter]?'本次更新触发了参数约束。':'本次更新没有触发参数约束。'} 梯度和实际改变量不是同一个量。</p>`;}
    function forwardView(){const d=fixed||describe(parameter),s=result.details.sample,prev=s.activations[d.layer-1],layer=core.model(result.before).layers[d.layer-1];return `<h3>前向计算 · ${esc(d.label)}</h3><p>本次训练样本 ${sampleIndex+1}，使用更新前参数。</p><div class="trace-equation">z = Σ(a × w) + b<br>= ${prev.map((v,i)=>`(${num(v)} × ${num(layer.weights[d.neuron][i])})`).join(' + ')} + (${num(layer.biases[d.neuron])})<br>= ${num(s.preActivations[d.layer-1][d.neuron])}<br>${d.layer===artifact.model.layers.length?'线性输出':'sigmoid(z)'} = ${num(s.activations[d.layer][d.neuron])}</div><p>最终得分 s = ${num(s.outputs[0])}</p>`;}
    function parameterRows(){return core.parameterMap.map((d,i)=>({d,i})).filter(({d})=>!filter||String(d.label||describe(d.index??0).label).includes(filter)).map(({d,i})=>`<tr><td><button data-geometry-parameter="${i}">${esc(describe(i).label)}</button></td><td>${num(result.before[i])}</td>${visibleNames().map(d=>`<td>${num(result.details?.parts[names().findIndex(n=>n.id===d.id)][i])}</td>`).join('')}<td>${num(step?result.gradient[i]:null)}</td><td>${num(step?result.after[i]-result.before[i]:null)}</td><td>${num(result.after[i])}</td></tr>`).join('');}
    function render(){
      if(!entry)return;
      tabs.hidden=false;host.hidden=!enabled;
      tabs.innerHTML=`<button data-geometry-mode="forward" aria-pressed="${!enabled}">前向计算</button><button data-geometry-mode="replay" aria-pressed="${enabled}">反向传播回放</button>${enabled?'<button data-geometry-action="fullscreen">讲解全屏</button>':''}`;
      if(!enabled)return;
      const scroll=host.scrollTop,stages=manifest.stages||[0,manifest.steps],prev=[...stages].reverse().find(v=>v<step)??0,next=stages.find(v=>v>step)??manifest.steps;
      let html=`<div class="trace-sticky"><span class="trace-eyebrow">${esc(artifact.name||'全特征可解释')} · ${manifest.steps.toLocaleString()} 次真实更新</span><h2>从随机权重到清晰特征</h2><div class="trace-stages">${stages.map((v,i)=>`<button data-geometry-seek="${v}" aria-pressed="${v===step}">${i||'初始'}<small>${v.toLocaleString()}</small></button>`).join('')}</div><div class="trace-navigation"><button data-geometry-seek="${prev}">上一阶段</button><button data-geometry-seek="${next}">下一阶段</button><button data-geometry-seek="${Math.max(0,step-1)}">上一次更新</button><input data-geometry-step type="number" min="0" max="${manifest.steps}" value="${step}" aria-label="更新编号"><button data-geometry-action="jump">跳转</button><button data-geometry-seek="${Math.min(manifest.steps,step+1)}">下一次更新</button></div><p>实际更新 ${step.toLocaleString()} / ${manifest.steps.toLocaleString()} · ${step?'所有对应损失联合训练':'初始化，尚未反向传播'}</p></div>`;
      if(busy||error||!result){host.innerHTML=html+`<div role="status" class="trace-status ${error?'trace-error':''}">${esc(error||'正在加载、重建并校验所选更新…')}${error?'<button data-geometry-action="retry">重试</button><p>单独复制 HTML 可使用前向教学；回放需要完整离线包。</p>':''}</div>`;host.scrollTop=scroll;return;}
      const s=result.details?.sample,p=parameterInfo(fixed?-1:parameter),d=fixed||describe(parameter);
      if(s)html+=`<div class="trace-sample-control">${pixelGrid(s.pixels)}<label>本次训练批次 <select data-geometry-sample>${array(result.ids).map((id,i)=>`<option value="${i}" ${sampleIndex===i?'selected':''}>样本 ${i+1} / ${result.ids.length} · ${id}</option>`).join('')}</select></label></div>`;
      html+=`<div class="trace-toolbar"><label>观察画布参数 <select data-geometry-phase>${[['before','更新前'],['after','更新后'],...(step===manifest.steps?[['published','正式模型 · 六位小数']]:[])].map(([v,t])=>`<option value="${v}" ${phase===v?'selected':''}>${t}</option>`).join('')}</select></label><label><input type="checkbox" data-geometry-precision ${precision?'checked':''}>完整精度</label></div><div class="trace-observation">${pixelGrid(observed)}<div><b>观察得分 ${num(core.forward(observed,currentModel).outputs[0])}</b><p>编辑画布只改变前向观察，不改变训练样本或梯度。</p><button data-geometry-action="canvas">展开／收起观察画布</button></div></div><label>查询参数 <select data-geometry-parameter-select>${core.parameterMap.map((v,i)=>`<option value="${i}" ${i===parameter?'selected':''}>${esc(describe(i).label)}</option>`).join('')}</select></label><div class="trace-update"><div><small>更新前</small><b>${num(p.before)}</b></div><span>→</span><div><small>更新后</small><b>${num(p.after)}</b></div><div><small>实际 Δw</small><b>${num(p.delta)}</b></div></div><div class="trace-calculation-tabs">${[['forward','前向'],['loss','损失'],['backward','反传'],['update','更新']].map(([v,t])=>`<button data-geometry-section="${v}" aria-pressed="${section===v}">${t}</button>`).join('')}</div>`;
      if(!s)html+='<section class="trace-section"><h3>初始化 · 尚未反向传播</h3><p>可训练权重独立采样 U(−0.1, 0.1)，偏置为零；范围之外的连接固定为零。特征目标与连接范围由人工指定，权重数值通过训练学习。此时尚无训练批次，因此损失、偏导与改变量显示为“—”。</p></section>';
      else html+=`<section class="trace-section">${section==='loss'?lossView():section==='backward'?backwardView():section==='update'?fixed?'<p>固定零连接不参与参数更新。</p>':updateView():forwardView()}</section>`;
      if(step===manifest.steps)html+='<div class="trace-release">最终更新完成 → 参数保留六位小数 → 当前正式模型。舍入不是额外训练。</div>';
      html+=`<div class="trace-toolbar"><button data-geometry-action="table">${fullTable?'收起':'查询'}全部 ${core.P} 个可训练参数</button><button data-geometry-action="json">导出本步 JSON</button><button data-geometry-action="csv">导出参数 CSV</button><button data-geometry-action="return">回到真实训练教学</button></div>${fullTable?`<p>结构规定为零的连接可在右侧完整连接对比中查询，固定权重与改变量为零，没有训练梯度或 Adam 更新。</p><input data-geometry-filter placeholder="筛选参数" value="${esc(filter)}"><div class="trace-table-scroll"><table><thead><tr><th>参数</th><th>更新前</th>${visibleNames().map(d=>`<th>${esc(lossName(d))}</th>`).join('')}<th>总梯度</th><th>Δw</th><th>更新后</th></tr></thead><tbody data-geometry-rows>${parameterRows()}</tbody></table></div>`:''}<a href="https://github.com/passwordis42/NeronNetRectangle/releases/download/geometry-teaching-v1/neural-network-offline.zip">下载完整离线包 ↗</a>`;
      host.innerHTML=html;host.scrollTop=scroll;
    }
    function download(type){if(!result)return;let text,mime;const metadata={traceId:manifest.id,modelSha256:manifest.modelSha256,step,sampleIndex,source,scope,parameter:fixed||describe(parameter),lossDefinitions:names(),optimizer:config(),initialization:manifest.initialization,parameterMap:core.parameterMap,topology:artifact.model.topology,activations:core.activations,model:currentModel,lossConfig:manifest.lossConfig,trainingTargets:{features:artifact.featureDefinitions,geometry:artifact.geometryDefinitions},sourceRecord:{chunk:manifest.chunks.find(c=>step===0?c.firstStep===1:c.firstStep<=step&&step<=c.lastStep),parameterTime:"before-update"},observation:{pixels:observed,phase},provenance:provenance()};
      if(type==='json'){text=JSON.stringify({...metadata,...result},(_,v)=>ArrayBuffer.isView(v)?array(v):v,2);mime='application/json';}
      else {const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';const rows=[['参数','更新前',...visibleNames().map(d=>lossName(d)+'有效梯度'),'总梯度','Adam候选值','约束改变量','实际改变量','更新后','m','v','m_hat','v_hat','更新编号','记录']];for(let i=0;i<core.P;i++)rows.push([describe(i).label,result.before[i],...visibleNames().map(d=>result.details?.parts[names().findIndex(n=>n.id===d.id)][i]),step?result.gradient[i]:'',result.proposed?.[i],step?result.after[i]-result.proposed[i]:'',step?result.after[i]-result.before[i]:'',result.after[i],result.m?.[i],result.v?.[i],result.correctedM?.[i],result.correctedV?.[i],step,manifest.id]);text='\ufeff'+rows.map(r=>r.map(quote).join(',')).join('\r\n');mime='text/csv;charset=utf-8';}
      const link=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type:mime}));link.href=url;link.download=`${manifest.id}-step-${step}.${type}`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    function setEnabled(value){enabled=!!(entry&&value);if(enabled){const d=describe(parameter);options.onSelect(d.layer,d.neuron);seek(step,sampleIndex,true);}else{++requestId;busy=false;result=null;currentModel=null;notify();}}
    function click(e){if(!entry)return;const b=e.target.closest('button');if(!b)return;if(b.dataset.geometryMode){setEnabled(b.dataset.geometryMode==='replay');return;}if(b.dataset.geometrySeek!==undefined){seek(b.dataset.geometrySeek);return;}if(b.dataset.geometryParameter!==undefined){selectParameter(Number(b.dataset.geometryParameter));return;}if(b.dataset.geometrySection){section=b.dataset.geometrySection;save();render();return;}switch(b.dataset.geometryAction){case'jump':seek(host.querySelector('[data-geometry-step]').value);break;case'retry':seek(step,sampleIndex,true);break;case'return':setEnabled(false);break;case'find':seek(step,sampleIndex,true,true);break;case'audit':auditOpen=!auditOpen;render();break;case'table':fullTable=!fullTable;render();break;case'canvas':{const fold=document.getElementById('replayCanvasFold');if(fold)fold.open=!fold.open;break;}case'json':case'csv':download(b.dataset.geometryAction);break;case'fullscreen':(document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()).catch(()=>{notice='可使用浏览器 F11 进入全屏。';render();});break;}}
    tabs.addEventListener('click',click);host.addEventListener('click',click);
    document.addEventListener('click',e=>{if(!entry||!enabled)return;const cell=e.target.closest('[data-geometry-input]');if(cell){selectConnection(Number(cell.dataset.geometryLayer),Number(cell.dataset.geometryNeuron),Number(cell.dataset.geometryInput));return;}if(e.target.closest('[data-geometry-open-loss]')){section='backward';save();render();host.scrollTop=0;}});
    host.addEventListener('change',e=>{if(!entry)return;const t=e.target;if(t.hasAttribute('data-geometry-sample'))seek(step,Number(t.value),true);if(t.hasAttribute('data-geometry-parameter-select'))selectParameter(Number(t.value));if(t.hasAttribute('data-geometry-phase')){phase=t.value;setModel();save();notify();}if(t.hasAttribute('data-geometry-precision')){precision=t.checked;save();notify();}if(t.hasAttribute('data-geometry-source')){infoCache.clear();source=t.value;auditRow=0;save();render();}if(t.hasAttribute('data-geometry-audit-row')){auditRow=Number(t.value);render();}});
    host.addEventListener('input',e=>{if(entry&&e.target.hasAttribute('data-geometry-filter')){filter=e.target.value;host.querySelector('[data-geometry-rows]').innerHTML=parameterRows();}});
    host.addEventListener('keydown',e=>{if(entry&&e.key==='Enter'&&e.target.hasAttribute('data-geometry-step'))seek(e.target.value);});
    return {
      get active(){return enabled&&!!entry;},get model(){return currentModel;},get loading(){return busy;},get parameter(){return fixed?-1:parameter;},get metric(){return metric;},get zoom(){return zoom;},get layout(){return layout;},get zoomMode(){return zoomMode;},format:num,
      get networkModel(){return networkModel();},get networkPixels(){return currentInput();},
      get context(){return {step,source,sourceName:sourceName(),metric,networkSource,sampleIndex,hasGradient:!!(step&&result),phase:networkSource==='sample'?'before':phase,geometry:true};},
      annotationSnapshot(){return {ready:false};},parameterIndex,parameterInfo,
      metricValue(i){const p=parameterInfo(i);return p?p[metric==='weight'?'weight':metric==='delta'?'delta':'gradient']:null;},
      showLoss(){section='loss';save();render();},seekUpdate(v){seek(v);},selectParameter,selectConnection,selectNeuron,
      deactivate(){++requestId;enabled=false;entry=null;result=null;currentModel=null;busy=false;},
      sync(state){const next=entries.find(e=>e.manifest.presetId===state.teachingPreset)||null;if(next!==entry)switchEntry(next);if(!entry)return;observed=state.pixels.slice();render();},
      setView(kind,value){infoCache.clear();if(kind==='zoom'||kind==='fit'){zoomMode=kind==='fit'?'fit':'manual';if(kind==='zoom')zoom=Math.max(.1,Math.min(3,Number(value)||1));save();options.onViewportChange?.();return;}if(kind==='metric')metric=value;if(kind==='source')source=value;if(kind==='scope')scope=value;if(kind==='network')networkSource=value;if(kind==='layout')layout=value;save();notify();},
      updateFitZoom(value){zoom=Math.max(.1,Math.min(3,value));},
      renderToolbar(){return `<div class="trace-network-toolbar"><label>网络观察 <select data-replay-view="network">${[['sample','训练样本 · 更新前'],['observation','观察画布 · 所选参数']].map(([v,t])=>`<option value="${v}" ${networkSource===v?'selected':''}>${t}</option>`).join('')}</select></label><label>数值 <select data-replay-view="metric">${[['weight','权重 w'],['gradient','偏导'],['delta','实际 Δw']].map(([v,t])=>`<option value="${v}" ${metric===v?'selected':''}>${t}</option>`).join('')}</select></label><label>偏导来源 <select data-replay-view="source">${[...visibleNames(),{id:'total',name:'本次更新 · 总梯度'}].map(d=>`<option value="${d.id}" ${source===d.id?'selected':''}>${esc(lossName(d))}</option>`).join('')}</select></label><label>范围 <select data-replay-view="scope"><option value="sample" ${scope==='sample'?'selected':''}>单个训练样本</option><option value="batch" ${scope==='batch'?'selected':''}>整批次／全部约束</option></select></label><label>布局 <select data-replay-view="layout">${[['compact','横向紧凑'],['vertical','纵向各层'],['expanded','逐个展开']].map(([v,t])=>`<option value="${v}" ${layout===v?'selected':''}>${t}</option>`).join('')}</select></label><span>${result?`更新 ${step} · ${esc(sourceName())}`:'加载并校验中，暂不显示历史值'}</span><details><summary>偏导来源是什么意思？</summary><p>分类损失衡量最终判断；特征监督衡量指定特征的响应；几何监督衡量第二隐藏层条件；裕量与贡献约束确保检测可靠且实际参与判断。单样本显示该样本的原始偏导，整批次显示按定义归一化并加权后的总和。Adam 始终使用所有有效来源的总梯度。</p></details></div>`;},
      renderInspector(details,helpers={}){
        if(!result)return '<p role="status">正在校验所选更新，请稍候。</p>';
        const layer=details.layerIndex,neuron=details.neuronIndex,terms=details.inputWeights||[],valueFor=i=>parameterInfo(parameterIndex(layer,neuron,i));
        const values=terms.map((t,i)=>{const p=valueFor(i);return p?.[metric==='weight'?'weight':metric==='delta'?'delta':'gradient'];});const max=Math.max(1e-12,...values.filter(Number.isFinite).map(Math.abs));
        const selected=fixed||describe(parameter),p=parameterInfo(fixed?-1:parameter),bias=parameterInfo(parameterIndex(layer,neuron,null));
        const metricDigits=Math.max(8,...values.map(value=>num(value).length));
        const cells=terms.map((t,i)=>{
          const value=values[i],empty=!Number.isFinite(value)||value===0;
          const alpha=.22+.78*Math.abs(value)/max,base=value>0?[16,130,99]:[179,34,82];
          const rgb=empty?[23,29,26]:base.map((channel,k)=>Math.round(channel*alpha+[247,250,247][k]*(1-alpha)));
          const linear=rgb.map(channel=>{const c=channel/255;return c<=.04045?c/12.92:Math.pow((c+.055)/1.055,2.4);});
          const luminance=.2126*linear[0]+.7152*linear[1]+.0722*linear[2];
          const foreground=luminance>.179?'#000000':'#ffffff';
          return `<button class="geometry-metric-cell ${selected.layer===layer&&selected.neuron===neuron&&selected.input===i?'is-selected':''}" data-geometry-input="${i}" data-geometry-layer="${layer}" data-geometry-neuron="${neuron}" style="background:rgb(${rgb.join(',')});color:${foreground}"><small>${esc(t.label||`${layer===1?'X':'H'+(layer-1)+'-'}${i+1}`)}</small><b>${num(value)}</b></button>`;
        }).join('');
        return `<p class="hint">第 ${step} 次更新 · ${step===0?'初始化参数 · 观察画布':networkSource==='sample'?`训练样本 ${sampleIndex+1} · 更新前`:`观察画布 · ${phase==='before'?'更新前':phase==='after'?'更新后':'正式模型'}`}</p>${helpers.renderComparison?.(details)||''}${helpers.renderEvidence?.(details)||''}<section class="geometry-metric"><h4>${metric==='gradient'?'偏导':metric==='delta'?'实际改变量 Δw':'权重 w'} · ${terms.length} 条输入连接</h4><p>${esc(sourceName())}</p><div class="geometry-metric-grid ${layer===1?'is-pixel':''}" style="--metric-digits:${metricDigits}">${cells}</div><p>偏置 ${num(bias?.weight)} · 偏导 ${num(bias?.gradient)} · Δb ${num(bias?.delta)}</p></section><section class="trace-audit"><b>${esc(selected.label||'固定零连接')}</b><p>更新前 ${num(p.before)} → 更新后 ${num(p.after)}；Δw = ${num(p.delta)}</p><p>所选来源偏导 ${num(p.gradient)}；实际总梯度 ${num(p.totalGradient)}</p>${fixed?'<p>结构约束：此连接固定为零，不属于优化器参数。</p>':'<button data-geometry-open-loss>查看损失与链式求导</button>'}</section>`;
      }
    };
  }
  return {create};
})();
