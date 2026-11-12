/* Actual first-layer sources and contextual pixel interventions; never equivalent pixel weights. */
(function(root,factory){
  const api=factory();
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.SecondLayerSpatial=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  const vector=(a,n)=>(Array.isArray(a)||ArrayBuffer.isView(a))&&a.length===n&&Array.from(a).every(Number.isFinite);
  function compute(model,result,forwardFn,neuronIndex){
    if(typeof forwardFn!=='function'||!model||!Array.isArray(model.layers)||model.layers.length!==3)return null;
    const sizes=[25];
    for(const layer of model.layers){
      if(!layer||!Array.isArray(layer.weights)||!layer.weights.length||!vector(layer.biases,layer.weights.length)||
        !layer.weights.every(row=>vector(row,sizes.at(-1))))return null;
      sizes.push(layer.weights.length);
    }
    if(model.topology&&(!vector(model.topology,4)||sizes.some((size,i)=>model.topology[i]!==size)))return null;
    if(!Number.isInteger(neuronIndex)||neuronIndex<0||neuronIndex>=sizes[2])return null;
    const validForward=f=>f&&Array.isArray(f.activations)&&f.activations.length===4&&
      f.activations.every((row,i)=>vector(row,sizes[i]))&&Array.isArray(f.preActivations)&&f.preActivations.length===3&&
      f.preActivations.every((row,i)=>vector(row,sizes[i+1]))&&vector(f.outputs,sizes[3]);
    if(!validForward(result))return null;
    const input=Array.from(result.activations[0]);
    if(input.some(v=>v<0||v>1))return null;
    const first=model.layers[0],second=model.layers[1];
    const sources=first.weights.map((weights,index)=>({
      index,weights:Array.from(weights),pixelContributions:Array.from(weights,(w,p)=>w*input[p]),
      activation:result.activations[1][index],weight:second.weights[neuronIndex][index],
      contribution:result.activations[1][index]*second.weights[neuronIndex][index],
      bias:first.biases[index],preActivation:result.preActivations[0][index]
    }));
    const weightedSum=sources.reduce((sum,s)=>sum+s.contribution,0),bias=second.biases[neuronIndex];
    const preActivation=result.preActivations[1][neuronIndex],activation=result.activations[2][neuronIndex];
    const tolerance=1e-10*Math.max(1,Math.abs(bias),...sources.map(s=>Math.abs(s.contribution)))*sources.length;
    if(!Number.isFinite(weightedSum)||Math.abs(weightedSum+bias-preActivation)>tolerance)return null;
    const pixels=[];
    for(let p=0;p<25;p++){
      const evaluate=value=>{
        if(input[p]===value)return result;
        const changed=input.slice();changed[p]=value;
        const f=forwardFn(changed,model);
        return validForward(f)&&changed.every((v,i)=>f.activations[0][i]===v)?f:null;
      };
      const off=evaluate(0),on=evaluate(1);
      if(!off||!on)return null;
      const z0=off.preActivations[1][neuronIndex],z1=on.preActivations[1][neuronIndex];
      const a0=off.activations[2][neuronIndex],a1=on.activations[2][neuronIndex];
      const deltaZ=z1-z0,deltaActivation=a1-a0;
      if(!Number.isFinite(deltaZ)||!Number.isFinite(deltaActivation))return null;
      pixels.push({index:p,inputValue:input[p],z0,z1,deltaZ,a0,a1,deltaActivation});
    }
    return {sources,weightedSum,bias,preActivation,activation,pixels,input,neuronIndex};
  }
  return {compute};
});
