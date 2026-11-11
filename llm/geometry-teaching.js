/* Fully interpretable geometry models. Targets explain training; forward() alone decides. */
function createSquareGeometryTeaching(artifact, ID, name) {
  if (!artifact) return null;
  const clone = value => JSON.parse(JSON.stringify(value));
  const escape = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  const number = value => !Number.isFinite(value) ? '—' : value !== 0 && Math.abs(value) < .0000005 ? value.toExponential(6) : value.toFixed(6);
  const double = artifact.model.topology.length === 4;
  const features = artifact.featureDefinitions;
  const examples = artifact.compositionExamples;
  const geometryNames = ['每个亮点具有合法局部连接','转角数量不少于四个','转角数量不多于四个','水平直线数量不少于垂直数量','垂直直线数量不少于水平数量'];
  const featureAt = i => features.find(f => f.neuronIndex === i) || features[i];
  function neuronName(layer, index) {
    if (layer === 0) return `输入 X${index+1}`;
    if (layer === artifact.model.topology.length - 1) return '输出 · 正方形得分';
    if (layer === 1) return `H1-${index+1} · ${featureAt(index)?.name || '局部特征'}`;
    return `H2-${index+1} · ${geometryNames[index]}`;
  }
  function targetMap(feature, pixels = null) {
    return `<span class="geometry-target-map" role="img" aria-label="${escape(feature.name)}的${pixels ? '当前输入' : '须亮与须空位置'}">${Array.from({length:25},(_,p)=>{
      const on=feature.on.includes(p),off=feature.off.includes(p),active=pixels ? pixels[p] >= .5 : on;
      const kind=pixels ? active ? on?'match':off?'conflict':'extra' : on?'missing':off?'off':'free' : on?'on':off?'off':'free';
      return `<i class="${kind}" title="X${p+1} · ${on?'须亮':off?'须空':'不限定'}">${!pixels && off?'○':''}</i>`;
    }).join('')}</span>`;
  }
  function renderFeatureTarget(index,response) {
    const f=featureAt(index);if(!f)return '';
    return `<section class="guided-target"><b>训练目标 · ${escape(f.name)}</b><div class="guided-target-body">${targetMap(f)}<div><span class="guided-target-status">${response>=.5?'检测到':'未检测到'} · 响应 ${number(response)}</span><p>实心须亮，○ 须空，灰格不限定；仅指定位置有输入连接。</p></div></div><p>${index>=119?'此局部亮／暗组合构成反对证据，响应通过负输出权重降低得分。':'这是一个局部点、直线或转角，不是完整方框模板。'}</p></section>`;
  }
  function renderGeometryTarget(index,response) {
    return `<section class="guided-target"><b>第二层训练目标 · ${geometryNames[index]}</b><p>输入是第一层的真实连续响应。当前响应 ${number(response)}，${response>=.5?'满足':'不满足'}此条件。</p><p>下方逐项展示第一层响应、连接权重及乘积；目标说明不参与识别。</p></section>`;
  }
  function examplePixels(index,variant='complete') {
    const e=examples[index]||examples[0],p=Array.from({length:25},(_,i)=>Number(e.boundary.includes(i)));
    if(variant==='missing')p[e.missingPixel ?? e.boundary[0]]=0;
    if(variant==='extra')p[e.extraPixel ?? 12]=1;
    return p;
  }
  function preview(e){return `<span class="composition-example-preview" aria-hidden="true">${Array.from({length:25},(_,i)=>`<i${e.boundary.includes(i)?' class="is-on"':''}></i>`).join('')}</span>`;}
  function createPreset(){
    return {teachingPreset:ID,name,hiddenLayers:artifact.model.topology.slice(1,-1).map(size=>({size,activation:'sigmoid'})),outputs:[{name:'正方形',threshold:1,target:2,comparison:'gt'}],outputActivation:'linear',weightSettings:{min:-128,max:128,precision:6},initialBias:0,biasFrozen:false,model:clone(artifact.model)};
  }
  function matchesPreset(preset){
    const expected=createPreset();
    return preset?.teachingPreset===ID && JSON.stringify(preset.hiddenLayers)===JSON.stringify(expected.hiddenLayers) && preset.outputActivation==='linear' && preset.biasFrozen===false && preset.outputs?.length===1 && preset.outputs[0].threshold===1 && preset.outputs[0].comparison==='gt' && JSON.stringify(preset.model?.topology)===JSON.stringify(artifact.model.topology) && preset.model?.layers.every((l,i)=>JSON.stringify(l.weights)===JSON.stringify(artifact.model.layers[i].weights)&&JSON.stringify(l.biases)===JSON.stringify(artifact.model.layers[i].biases));
  }
  function card(state,forward,i){
    const f=featureAt(i),h=forward.activations[1][i],selected=state.selectedNeuron?.layerIndex===1&&state.selectedNeuron.neuronIndex===i;
    const outgoing=double?null:state.model.layers[1].weights[0][i];
    return `<button type="button" class="geometry-feature-card ${selected?'is-selected':''} ${i>=119?'is-violation':''}" data-teaching-layer="1" data-teaching-neuron="${i}" aria-pressed="${selected}"><b>H1-${i+1} · ${escape(f.name)}</b><div class="geometry-card-maps"><span>训练目标${targetMap(f)}</span><span>当前画布${targetMap(f,state.pixels)}</span></div><span>响应 ${number(h)} · ${h>=.5?'检测到':'未检测到'}</span>${outgoing===null?'<small>传入第二层几何条件；点击查看真实连接。</small>':`<span>输出权重 ${number(outgoing)}</span><strong>实际贡献 ${number(h*outgoing)}</strong>`}</button>`;
  }
  const groups=[['点',0,25],['水平直线',25,40],['垂直直线',40,55],['转角',55,119],...(!double?[['局部违规',119,352]]:[])];
  function renderGroup(state,forward,index){const group=groups[index];return group?Array.from({length:group[2]-group[1]},(_,i)=>card(state,forward,group[1]+i)).join(''):'';}
  function renderView(state,forward){
    const e=examples[state.compositionExample]||examples[0],current=forward.activations[1],output=forward.outputs[0];
    const related=features.filter(f=>f.neuronIndex<119&&f.on.every(i=>e.boundary.includes(i))&&f.off.every(i=>!e.boundary.includes(i))).map(f=>f.neuronIndex);
    const finalLayer=state.model.layers.at(-1),previous=forward.activations.at(-2),terms=previous.map((a,i)=>a*finalLayer.weights[0][i]);
    return `<div class="teaching-focus-head"><div><span class="teaching-eyebrow">全特征可解释 · ${double?'两层逐级组合':'单层支持与反对证据'}</span><h3>${double?'局部形状怎样组成正方形':'局部特征怎样共同决定得分'}</h3><p>${artifact.model.topology.join(' → ')} · sigmoid 隐藏层 · 线性输出</p></div><strong class="geometry-score">得分 ${number(output)}<small>${output>1?'正方形':'不是正方形'} · 阈值 &gt; 1</small></strong></div>
      <p class="geometry-training-note">人工指定特征目标与连接范围，参数从小随机权重训练得到。没有完整方框或待补角轮廓神经元；下方位置示意不参与识别。</p>
      <nav class="composition-examples" aria-label="六个方框示例">${examples.map((x,i)=>`<button class="composition-example ${i===state.compositionExample?'is-selected':''}" data-geometry-example="${i}" aria-pressed="${i===state.compositionExample}">${preview(x)}<span>${escape(x.name)}</span></button>`).join('')}</nav>
      <div class="composition-actions"><button type="button" data-geometry-draw="complete">完整方框</button><button type="button" data-geometry-draw="missing">缺一格</button><button type="button" data-geometry-draw="extra">内部多一点</button></div>
      <section class="teaching-stage"><h3>1 · ${escape(e.name)}的局部特征</h3><p class="hint">同一条边由多个直线、转角与亮点位置共同描述。响应来自当前画布；特征位置与组合示意不替代网络计算。</p><div class="geometry-feature-grid">${related.map(i=>card(state,forward,i)).join('')}</div></section>
      ${double?`<section class="teaching-stage"><h3>2 · 五项几何条件</h3><div class="geometry-gates">${geometryNames.map((n,i)=>`<button type="button" data-teaching-layer="2" data-teaching-neuron="${i}" class="geometry-gate"><b>H2-${i+1} · ${n}</b><span>响应 ${number(forward.activations[2][i])}</span><span>输出权重 ${number(finalLayer.weights[0][i])}</span><strong>贡献 ${number(terms[i])}</strong></button>`).join('')}</div></section>`:`<section class="geometry-training-note"><b>反对证据也参与判断</b><p>233 个违规目标只观察 3×3 区域中的 2～5 个像素。当前检测到 ${current.slice(119).filter(v=>v>=.5).length} 项，实际扣分仍由连续响应乘真实权重计算。</p></section>`}
      <section class="inspector-calculation"><h3>最终得分的完整求和</h3><p>${double?'五项几何条件':'119 项支持特征＋233 项违规特征'}的贡献之和 ${number(terms.reduce((a,b)=>a+b,0))} ＋ 输出偏置 ${number(finalLayer.biases[0])} ＝ ${number(output)}</p><details><summary>展开全部 ${terms.length} 项输出贡献</summary><div class="geometry-output-terms">${terms.map((v,i)=>`<button type="button" data-teaching-layer="${double?2:1}" data-teaching-neuron="${i}">${escape(neuronName(double?2:1,i))}<br>${number(previous[i])} × ${number(finalLayer.weights[0][i])} = ${number(v)}</button>`).join('')}</div></details></section>
      <section class="teaching-stage"><h3>全部第一层特征 · ${features.length} 项</h3>${groups.map(([n,a,b],index)=>`<details class="geometry-catalog" data-geometry-group="${index}" ${state.geometryOpenGroups?.includes(index)?'open':''}><summary>${n} · ${b-a} 项</summary><div class="geometry-feature-grid">${state.geometryOpenGroups?.includes(index)?renderGroup(state,forward,index):''}</div></details>`).join('')}</section>
      <details class="trained-provenance"><summary>训练与验证记录</summary><p>随机初始化，结构范围以外的连接固定为零。特征名称是人工监督目标；参数由实际梯度更新得到。</p><p>模型哈希 ${escape(artifact.modelSha256||'')}</p><p>正式六位小数模型：全部 33,554,432 种黑白输入验证 ${artifact.verification?.errors===0?'零误判':'以正式验证记录为准'}。</p></details>`;
  }
  return {ID,name,kind:'trained',geometry:true,double,features,examples,defaultCandidate:0,drawOnExampleSelect:true,artifact,createPreset,matchesPreset,neuronName,renderFeatureTarget,renderGeometryTarget,renderView,renderGroup,examplePixels,pixelsFor:examplePixels,provenance:()=>({modelSha256:artifact.modelSha256,training:artifact.training,verification:artifact.verification,featureDefinitions:features})};
}
const SquareGeometrySingleTeaching = typeof SquareGeometrySingleModelData !== 'undefined' ? createSquareGeometryTeaching(SquareGeometrySingleModelData,'square-trained-geometry-single-v1','全特征 · 单隐藏层 · 局部支持与违规') : null;
const SquareGeometryDoubleTeaching = typeof SquareGeometryDoubleModelData !== 'undefined' ? createSquareGeometryTeaching(SquareGeometryDoubleModelData,'square-trained-geometry-double-v1','全特征 · 双隐藏层 · 局部特征与几何组合') : null;
if(typeof module!=='undefined'&&module.exports)module.exports={createSquareGeometryTeaching};
