/* A read-only history viewer. No replay action trains or edits the released model. */
const SquareBackpropTeaching = (() => {
  const KEY = "nn-square-backprop-view-v1";
  const lossNames = ["分类误差", "局部特征监督", "检测边界约束", "四边贡献约束"];
  const offlineURL = "https://passwordis42.github.io/NeronNetRectangle/downloads/neural-network-offline.zip";
  function create(options) {
    const { host, tabs, manifest, artifact } = options, core = SquareBackpropCore;
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (_) { /* optional view preferences */ }
    const integer = (v, min, max, fallback) => Number.isInteger(v) && v>=min && v<=max ? v : fallback;
    let step = integer(saved.step,0,12000,0), parameter = integer(saved.parameter,0,1350,506);
    let phase = "after", precision = Boolean(saved.precision), sampleIndex = 0;
    let enabled = false, available = false, result = null, busy = false, error = "", requestId = 0;
    let worker = null, workerURL = null, currentModel = null, observed = Array(25).fill(0);
    let fullTable = false, expandedAdam = false, sampleDetails = false;
    const chunks = new Map(), pendingChunks = new Map();
    const escape = v => String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
    const num = v => Number.isFinite(v) ? (precision ? String(v) : v!==0 && Math.abs(v)<.0000005 ? v.toExponential(6) : v.toFixed(6)) : "—";
    const total = values => values.reduce((s,v)=>s+v,0);
    function save() { try { localStorage.setItem(KEY, JSON.stringify({ step, parameter, precision })); } catch (_) {} }
    function notify() { options.onChange(); }
    function grid(pixels, label) { return `<div class="trace-pixels" role="img" aria-label="${label}">${pixels.map(x=>`<i class="${x?'is-on':''}"></i>`).join("")}</div>`; }
    function setModel() {
      if (!result) { currentModel = null; return; }
      if (phase === "published") {
        const expected = core.flatten(artifact.model);
        if (result.after.some((v,i)=>Number(v.toFixed(6)) !== expected[i])) throw Error("最后一步与正式模型不一致，已停止回放");
        currentModel = core.model(expected);
      } else currentModel = core.model(phase === "before" ? result.before : result.after);
    }
    function ensureWorker() {
      if (worker) return;
      const code = `const core=(${createSquareBackpropCore.toString()})();
        const cache=new Map();
        onmessage=async({data:d})=>{try {
          let chunk=cache.get(d.chunk);
          if(!chunk){
            const bytes=Uint8Array.from(atob(d.base64),c=>c.charCodeAt(0));
            const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
            if(hash!==d.hash) throw Error('数据块校验失败，请重试或重新下载离线包');
            const raw=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
            chunk=core.readChunk(raw);
            if(chunk.firstStep!==d.chunk*100+1) throw Error('数据块编号不匹配');
            cache.set(d.chunk,chunk); if(cache.size>4) cache.delete(cache.keys().next().value);
          }
          const r=core.reconstruct(chunk,d.step,d.optimizer);
          if(d.step){
            const match=Array.from(r.ids).findIndex((id,i)=>i<r.positives&&id===d.defaultInput);
            const selected=Number.isInteger(d.sample)&&d.sample>=0&&d.sample<r.ids.length?d.sample:Math.max(0,match);
            const details=core.gradients(r.before,r.ids,r.positives,d.features,d.examples,selected);
            let max=0; for(let i=0;i<core.P;i++) max=Math.max(max,Math.abs(details.total[i]-r.gradient[i]));
            if(max>1e-8||details.losses.some((v,i)=>Math.abs(v-r.losses[i])>1e-8)) throw Error('重建的损失或分项梯度未通过校验');
            r.details=details; r.gradientError=max;
          }
          postMessage({id:d.id,result:r});
        }catch(e){postMessage({id:d.id,error:e.message});}};`;
      workerURL = URL.createObjectURL(new Blob([code], { type:"text/javascript" }));
      worker = new Worker(workerURL);
      worker.onmessage = ({data}) => {
        if (data.id !== requestId) return;
        busy=false;
        if (data.error) { error=data.error; result=null; currentModel=null; chunks.delete(Math.max(0,Math.floor((step-1)/100))); }
        else {
          result=data.result; sampleIndex=result.details?.sample.index || 0;
          try { setModel(); } catch(e) { error=e.message; result=null; currentModel=null; }
        }
        notify();
      };
      worker.onerror = () => {
        busy=false; error="计算线程无法运行，请刷新页面或使用新版浏览器。"; result=null; currentModel=null;
        worker.terminate(); worker=null; URL.revokeObjectURL(workerURL); notify();
      };
    }
    // Script data envelopes also work when the offline package is opened via file://.
    globalThis.SquareTraceChunks = { receive(index, payload) {
      const item=pendingChunks.get(index);
      if (!item || typeof payload!=="string") return;
      pendingChunks.delete(index); clearTimeout(item.timer); item.script.remove();
      chunks.set(index,payload); if(chunks.size>4) chunks.delete(chunks.keys().next().value);
      item.resolve(payload);
    } };
    function loadChunk(index) {
      if(chunks.has(index)) return Promise.resolve(chunks.get(index));
      if(pendingChunks.has(index)) return pendingChunks.get(index).promise;
      let resolve,reject;
      const promise=new Promise((a,b)=>{resolve=a;reject=b;});
      const script=document.createElement("script");
      const item={script,promise,resolve,reject,timer:null};
      function fail() {
        if(pendingChunks.get(index)!==item) return;
        pendingChunks.delete(index); clearTimeout(item.timer); script.remove();
        reject(Error("训练记录加载失败。请重试；离线使用时需解压完整离线包，保留 training-trace 文件夹。"));
      }
      item.timer=setTimeout(fail,45000); script.onerror=fail;
      pendingChunks.set(index,item);
      script.src=`training-trace/v1/${manifest.chunks[index].file}`;
      document.head.append(script);
      return promise;
    }
    async function seek(next, selectedSample=null, keepPhase=false) {
      step=integer(Number(next),0,12000,step);
      if(!keepPhase) phase=step===12000?"published":"after";
      const id=++requestId, chunk=Math.max(0,Math.floor((step-1)/100));
      busy=true; error=""; result=null; currentModel=null; save(); notify();
      try {
        if(!globalThis.Worker || !globalThis.DecompressionStream || !globalThis.crypto?.subtle) throw Error("此浏览器不支持训练记录解压，请使用新版 Chrome、Edge、Firefox 或 Safari。原有前向教学仍可使用。");
        ensureWorker();
        const base64=await loadChunk(chunk);
        if(id!==requestId || !enabled) return;
        const defaultInput=artifact.compositionExamples[0].boundary.reduce((v,p)=>v+2**p,0);
        worker.postMessage({id,chunk,base64,hash:manifest.chunks[chunk].sha256,step,sample:selectedSample,
          defaultInput,optimizer:manifest.optimizer,features:artifact.featureDefinitions,examples:artifact.compositionExamples});
      } catch(e) { if(id===requestId) { busy=false; error=e.message; chunks.delete(chunk); notify(); } }
    }
    function selectParameter(index, notifySelection=true) {
      parameter=integer(Number(index),0,1350,parameter); save();
      const d=core.describe(parameter);
      if(notifySelection) options.onSelect(d.layer,d.neuron);
      render();
    }
    function parameterOptions() {
      const d=core.describe(parameter), ids=d.layer===1
        ? [...Array.from({length:25},(_,i)=>d.neuron*25+i),1250+d.neuron,1300+d.neuron]
        : [...Array.from({length:50},(_,i)=>1300+i),1350];
      return ids.map(i=>`<option value="${i}" ${parameter===i?"selected":""}>${core.describe(i).label}</option>`).join("");
    }
    function parameterRows(filter="") {
      if(!result) return "";
      return Array.from({length:core.P},(_,i)=>i).filter(i=>core.describe(i).label.toLowerCase().includes(filter.toLowerCase())).map(i=>`<tr class="${i===parameter?'trace-selected':''}"><td><button type="button" data-trace-parameter="${i}">${core.describe(i).label}</button></td><td>${num(result.before[i])}</td><td>${num(result.gradient[i])}</td><td>${num(result.after[i]-result.before[i])}</td><td>${num(result.after[i])}</td></tr>`).join("");
    }
    function render() {
      tabs.hidden=!available;
      tabs.innerHTML=`<button type="button" data-trace-mode="forward" aria-pressed="${!enabled}">前向计算</button><button type="button" data-trace-mode="replay" aria-pressed="${enabled}">反向传播回放</button>`;
      host.hidden=!available||!enabled;
      if(host.hidden) return;
      const stage=step===0?0:Math.ceil(step/1200);
      const heading=`<header class="trace-header"><div><span class="trace-eyebrow">真实训练记录 · 可复现</span><h2>看清每一次参数更新</h2><p>10 个讲解阶段，12,000 次实际更新。初始参数继承已有模型，并初始化新增局部特征。</p></div><a href="${offlineURL}" download>下载完整离线包 ↗</a></header>
        <div class="trace-stages" role="group" aria-label="十个讲解阶段">${manifest.stages.map((t,i)=>`<button type="button" data-trace-seek="${t}" aria-pressed="${step===t}" title="${t===0?'初始化':`第 ${t} 次更新后`}">${i===0?'初始':i}<small>${t.toLocaleString()}</small></button>`).join("")}</div>
        <div class="trace-navigation"><button type="button" data-trace-seek="${Math.max(0,(Math.ceil(step/1200)-1)*1200)}" ${step===0?'disabled':''}>上一阶段</button><button type="button" data-trace-seek="${Math.min(12000,(Math.floor(step/1200)+1)*1200)}" ${step===12000?'disabled':''}>下一阶段</button><span class="trace-divider"></span><button type="button" data-trace-seek="${Math.max(0,step-1)}" ${step===0?'disabled':''}>上一次更新</button><label>更新编号 <input id="trace-step" type="number" min="0" max="12000" step="1" value="${step}"></label><button type="button" data-trace-action="jump">跳转</button><button type="button" data-trace-seek="${Math.min(12000,step+1)}" ${step===12000?'disabled':''}>下一次更新</button></div>
        <p class="trace-position" data-trace-position>${step===0?'初始化 · 尚未执行反向传播':`阶段 ${stage} / 10 · 实际更新 ${step.toLocaleString()} / 12,000`}</p>`;
      if(busy||error||!result) {
        host.innerHTML=heading+`<div class="trace-status ${error?'trace-error':''}" role="status">${error?escape(error):'正在加载并校验这一步的真实训练记录…'}${error?'<button type="button" data-trace-action="retry">重试</button>':''}</div>`;
        return;
      }
      const d=core.describe(parameter), j=d.layer===1?d.neuron:(d.input??20), sample=result.details?.sample;
      const obs=core.forward(observed,currentModel), hidden=obs.activations[1];
      const feature=artifact.featureDefinitions.find(f=>f.neuronIndex===j);
      const gradient=result.gradient[parameter], before=result.before[parameter], after=result.after[parameter];
      const lossView=sample?`<div class="trace-loss-table"><table><thead><tr><th>损失来源</th><th>损失值</th><th>对选中参数的梯度</th></tr></thead><tbody>${lossNames.map((name,i)=>`<tr><td>${name}</td><td>${num(result.losses[i])}</td><td>${num(result.details.parts[i][parameter])}</td></tr>`).join("")}</tbody><tfoot><tr><th>合计</th><td>${num(total(result.losses))}</td><td data-trace-total-gradient>${num(gradient)}</td></tr></tfoot></table></div>`:'';
      host.innerHTML=heading+`
        <div class="trace-toolbar"><label>画布观察的参数 <select id="trace-phase"><option value="before" ${phase==='before'?'selected':''}>本次更新前</option><option value="after" ${phase==='after'?'selected':''}>本次更新后</option>${step===12000?`<option value="published" ${phase==='published'?'selected':''}>正式模型 · 六位小数</option>`:''}</select></label><label><input id="trace-precision" type="checkbox" ${precision?'checked':''}> 完整精度</label><button type="button" data-trace-action="return">回到真实训练教学</button></div>
        <p class="hint">显示值保留六位小数；非零小量用科学计数法。计算与导出始终使用完整精度。</p>
        <section class="trace-observation"><div>${grid(observed,'当前观察画布')}<span>当前观察画布</span></div><div><b data-trace-observed-score>得分 ${num(obs.outputs[0])}</b><p>${phase==='published'?'正式模型 · 六位小数':step===0?'初始化参数':`第 ${step} 次更新${phase==='before'?'前':'后'}的参数`}。编辑左侧画布只改变前向观察，不修改历史训练样本或梯度。</p><p>${step===12000&&phase==='published'?'参数保留六位小数后与现有正式模型逐项一致。精度处理不是一次训练更新。':'这里的特征名称表示训练目标，当前响应不代表已经通过最终验收。'}</p></div></section>
        <section class="trace-section"><h3>选择一条计算路径</h3><div class="trace-neurons" role="group" aria-label="隐藏神经元">${hidden.map((v,i)=>`<button type="button" data-trace-neuron="${i}" aria-pressed="${d.layer===1&&d.neuron===i}" title="${escape(artifact.featureDefinitions.find(f=>f.neuronIndex===i)?.name||'自由学习')} · 响应 ${num(v)}" style="--response:${v>0?'#deeee2':'#f2e0d9'}">${i+1}</button>`).join("")}<button type="button" data-trace-parameter="1350" aria-pressed="${d.layer===2}">输出</button></div>
        <div class="trace-toolbar"><label>神经元 <select id="trace-neuron">${Array.from({length:51},(_,i)=>`<option value="${i}" ${i===(d.layer===1?d.neuron:50)?'selected':''}>${i===50?'输出神经元':`H1-${i+1}`}</option>`).join("")}</select></label><label>连接或偏置 <select id="trace-parameter">${parameterOptions()}</select></label></div>
        <p><b>${escape(d.label)}</b> ${d.layer===2?'· 输出层汇总 50 个隐藏响应':feature?`· 局部训练目标：${escape(feature.name)}`:'· 自由学习特征，无指定形状目标'}</p>
        ${d.layer===1?`<div class="trace-weight-grid" role="group" aria-label="输入到选中神经元的25条连接">${Array.from({length:25},(_,i)=>`<button type="button" data-trace-parameter="${j*25+i}" aria-pressed="${parameter===j*25+i}"><small>输入 ${i+1}</small>${num(result.before[j*25+i])}</button>`).join("")}</div><div class="trace-toolbar"><button type="button" data-trace-parameter="${1250+j}">查看 H1-${j+1} 偏置</button><button type="button" data-trace-parameter="${1300+j}">查看 H1-${j+1} → 输出</button></div>`:''}</section>
        ${sample?`
        <section class="trace-section"><h3><span>1</span> 前向计算 · 本次训练样本</h3><div class="trace-sample"><div>${grid(sample.pixels,'选中的历史训练样本')}<b>${sample.positive?'正例 · 完整正方形':'负例'}</b></div><div><label>批次中的样本 <select id="trace-sample">${Array.from(result.ids,(id,i)=>`<option value="${i}" ${i===sampleIndex?'selected':''}>${i+1} / ${result.ids.length} · ${i<result.positives?'正例':'负例'} · 图案 ${id}</option>`).join("")}</select></label><p>此处始终使用本次更新前的参数。该批次包含 ${result.positives} 个正例和 ${result.ids.length-result.positives} 个负例；重复抽到的样本分别计入平均值。</p><p>H1-${j+1}：z = ${num(sample.preActivations[0][j])} → tanh(z) = ${num(sample.activations[1][j])}<br>输出：s = ${num(sample.outputs[0])}</p></div></div><button type="button" data-trace-action="sample-details">${sampleDetails?'收起':'展开'}全部神经元的前向与反传值</button>${sampleDetails?`<div class="trace-table-scroll"><table><thead><tr><th>神经元</th><th>z</th><th>tanh(z)</th><th>1 − h²</th><th>分类 δ</th><th>特征监督 δ</th></tr></thead><tbody>${hidden.map((_,i)=>`<tr><td>H1-${i+1}</td><td>${num(sample.preActivations[0][i])}</td><td>${num(sample.activations[1][i])}</td><td>${num(1-sample.activations[1][i]**2)}</td><td>${num(sample.dzClass[i])}</td><td>${num(sample.dzConcept[i])}</td></tr>`).join("")}</tbody></table></div>`:''}</section>
        <section class="trace-section"><h3><span>2</span> 损失 · 四种目标共同训练</h3><p>分类损失使用平方间隔：正例希望得分 ≥ 1.6，负例希望得分 ≤ 0.4；达到间隔后，该样本分类误差为 0。推理时的判断阈值仍为得分 &gt; 1。</p>${lossView}<p class="hint">局部目标：满足时响应接近 +0.9，不满足时接近 −0.9，系数 0.2。检测边界使用 z 的 ±0.55 间隔；四边贡献目标为每项 ≥ 0.14，平方误差系数 4。显示的是实际训练目标。</p></section>
        <section class="trace-section"><h3><span>3</span> 反向传播 · 由误差求梯度</h3><div class="trace-equation">∂L分类 / ∂s = 2 × 间隔误差 / 批次大小<br>= 2 × ${num(sample.error)} / ${result.ids.length} = ${num(sample.ds)}</div>
        <div class="trace-equation">δ分类,H1-${j+1} = (∂L分类 / ∂s) × 输出权重 × (1 − h²)<br>= ${num(sample.ds)} × ${num(result.before[1300+j])} × ${num(1-sample.activations[1][j]**2)} = ${num(sample.dzClass[j])}</div>
        <p>输入权重梯度 = δ × 输入值；偏置梯度 = δ。输出权重梯度 = (∂L / ∂s) × 隐藏响应。</p><p>当前样本对 <b>${escape(d.label)}</b> 的贡献：分类 ${num(core.sampleGradient(sample,parameter).classification)}，特征监督 ${num(core.sampleGradient(sample,parameter).concept)}。</p><p>上表合计包含整个批次，以及独立计算的检测边界与四边贡献约束。它才是本次 Adam 使用的梯度 <b>${num(gradient)}</b>。</p><p class="hint">分项梯度已在浏览器重算核对；最大差异 ${result.gradientError.toExponential(2)}（允许 1e−8）。${sample.ds===0?'当前样本已满足分类间隔，因此它的分类梯度为 0。':''}</p></section>
        <section class="trace-section"><h3><span>4</span> 参数更新 · Adam</h3><div class="trace-update" data-trace-update><div><small>更新前</small><b>${num(before)}</b></div><span>＋</span><div><small>实际改变量</small><b>${num(after-before)}</b></div><span>＝</span><div><small>更新后</small><b>${num(after)}</b></div></div><p>学习率 0.003 · β₁ = 0.9 · β₂ = 0.999 · ε = 1e−8。Adam 使用累计的一阶、二阶矩；本次梯度为零时，也可能因之前的动量继续更新。</p><button type="button" data-trace-action="adam">${expandedAdam?'收起':'展开'} Adam 代入计算与裁剪</button>${expandedAdam?`<div class="trace-equation">m = 0.9 × ${num(result.oldM[parameter])} + 0.1 × ${num(gradient)} = ${num(result.m[parameter])}<br>v = 0.999 × ${num(result.oldV[parameter])} + 0.001 × (${num(gradient)})² = ${num(result.v[parameter])}<br>m̂ = m / (1 − 0.9^${step}) = ${num(result.correctedM[parameter])}<br>v̂ = v / (1 − 0.999^${step}) = ${num(result.correctedV[parameter])}<br>候选值 = ${num(before)} − 0.003 × ${num(result.correctedM[parameter])} / (√${num(result.correctedV[parameter])} + 1e−8) = ${num(result.proposed[parameter])}<br>裁剪到 [−10, 10] → ${num(after)} · ${result.clipped[parameter]?'触及边界，发生裁剪':'未发生裁剪'}</div>`:''}</section>`:
        `<section class="trace-section"><h3>步骤 0 · 训练开始前</h3><p>这些是真实训练的初始参数，还没有本次损失、梯度或参数改变量。初始模型继承旧版组合特征模型，并为新增局部目标设置初始化；不是从完全随机网络开始。</p><p>选中参数：<b>${escape(d.label)} = ${num(before)}</b>。点击“下一次更新”查看第一步真实反向传播。</p></section>`}
        ${step===12000?`<section class="trace-release"><h3>衔接到正式模型</h3><p>第 12,000 次更新结束 → 全部参数保留六位小数 → 当前部分解释模型。神经元的输入权重、偏置和输出连接使用同一混排映射。</p><p>选中参数：${String(after)} → ${Number(after.toFixed(6)).toFixed(6)}。这只是发布精度处理，不是反向传播。</p><button type="button" data-trace-action="return">回到真实训练教学</button></section>`:''}
        <section class="trace-section"><div class="trace-toolbar"><button type="button" data-trace-action="table">${fullTable?'收起':'查看'}全部 1,351 个参数</button><button type="button" data-trace-action="json">导出当前步 JSON</button><button type="button" data-trace-action="csv">导出参数表 CSV</button></div>${fullTable?`<label>筛选连接或偏置 <input id="trace-filter" type="search" placeholder="例如 H1-21 或 偏置"></label><div class="trace-table-scroll"><table><thead><tr><th>参数</th><th>更新前</th><th>梯度</th><th>改变量</th><th>更新后</th></tr></thead><tbody id="trace-parameter-rows">${parameterRows()}</tbody></table></div>`:''}<p class="hint">记录来源：固定种子 42 的确定性训练复现。只读回放，不修改正式模型或原有存档。导出保留完整数值精度。</p></section>`;
    }
    function download(type) {
      if(!result) return;
      let text,mime;
      if(type==='csv') {
        text='\ufeff参数,更新前,梯度,实际改变量,更新后\r\n'+Array.from({length:core.P},(_,i)=>[core.describe(i).label,result.before[i],step?result.gradient[i]:'',step?result.after[i]-result.before[i]:'',result.after[i]].join(',')).join('\r\n'); mime='text/csv;charset=utf-8';
      } else {
        text=JSON.stringify({traceId:manifest.id,modelSha256:manifest.modelSha256,step,optimizer:manifest.optimizer,
          observation:{pixels:observed,phase}, selectedParameter:core.describe(parameter),
          initialization:step===0, ...result},(_,v)=>ArrayBuffer.isView(v)?Array.from(v):v,2); mime='application/json';
      }
      const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type:mime}));
      a.href=url;a.download=`square-backprop-step-${step}.${type}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    function setEnabled(value) {
      enabled=value&&available;
      if(enabled) {
        const d=core.describe(parameter);
        options.onSelect(d.layer,d.neuron);
        seek(step);
      }
      else { ++requestId; busy=false; result=null; currentModel=null; notify(); }
    }
    tabs.addEventListener('click',e=>{const b=e.target.closest('[data-trace-mode]');if(b)setEnabled(b.dataset.traceMode==='replay');});
    host.addEventListener('click',e=>{
      const b=e.target.closest('button');if(!b)return;
      if(b.hasAttribute('data-trace-seek')) { seek(Number(b.dataset.traceSeek)); return; }
      if(b.hasAttribute('data-trace-neuron')) { selectParameter(Number(b.dataset.traceNeuron)*25+6); return; }
      if(b.hasAttribute('data-trace-parameter')) { selectParameter(Number(b.dataset.traceParameter)); return; }
      switch(b.dataset.traceAction) {
        case 'jump': seek(Number(host.querySelector('#trace-step').value)); break;
        case 'retry': seek(step); break;
        case 'return': setEnabled(false); break;
        case 'table': fullTable=!fullTable;render();break;
        case 'adam': expandedAdam=!expandedAdam;render();break;
        case 'sample-details': sampleDetails=!sampleDetails;render();break;
        case 'json':case 'csv':download(b.dataset.traceAction);break;
      }
    });
    host.addEventListener('change',e=>{
      switch(e.target.id) {
        case 'trace-phase': phase=e.target.value;setModel();notify();break;
        case 'trace-precision': precision=e.target.checked;save();render();break;
        case 'trace-parameter':selectParameter(Number(e.target.value));break;
        case 'trace-neuron':selectParameter(Number(e.target.value)===50?1350:Number(e.target.value)*25+6);break;
        case 'trace-sample':seek(step,Number(e.target.value),true);break;
      }
    });
    host.addEventListener('keydown',e=>{if(e.target.id==='trace-step'&&e.key==='Enter')seek(Number(e.target.value));});
    host.addEventListener('input',e=>{if(e.target.id==='trace-filter')host.querySelector('#trace-parameter-rows').innerHTML=parameterRows(e.target.value);});
    return {
      get active(){return enabled&&available;}, get model(){return currentModel;}, get loading(){return busy;},
      sync(state) {
        available=state.teachingPreset===manifest.presetId;
        if(!available&&enabled) { enabled=false;++requestId;busy=false;result=null;currentModel=null; }
        observed=state.pixels.slice(); render();
      },
      selectNeuron(layer,index) { if(!enabled || ![1,2].includes(layer))return;selectParameter(layer===2?1350:index*25+6,false); },
      selectConnection(layer,neuron,input) { if(!enabled)return;selectParameter(layer===1?neuron*25+input:1300+input); }
    };
  }
  return { create };
})();
