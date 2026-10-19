const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const core=require('../backprop-core.js'),root=path.resolve(__dirname,'..');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);
let checked=0;
for(const version of ['v1','v2']){
 const dir=path.join(root,'training-trace',version),manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 const artifact=JSON.parse(fs.readFileSync(path.join(root,version==='v2'?'partial-features-random-model.json':'partial-features-shuffled-model.json')));
 const read=step=>{
   const entry=manifest.chunks[Math.floor((step-1)/100)],source=fs.readFileSync(path.join(dir,entry.file),'utf8');
   const raw=zlib.gunzipSync(Buffer.from(source.match(/,"([A-Za-z0-9+/=]+)"/)[1],'base64'));
   return core.reconstruct(core.readChunk(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)),step,manifest.optimizer);
 };
 const first=read(1);
 // All 1,351 parameter paths, including free neurons and both kinds of bias.
 const ids=Uint32Array.from([469440,469312,0,33554431,31,262143]),positives=1;
 for(const weights of [[0,1,1,0],[1,1,1,1]]){
   const full=core.gradients(first.before,ids,positives,artifact.featureDefinitions,artifact.compositionExamples,0,weights);
   for(let parameter=0;parameter<core.P;parameter++){
     const t=core.parameterTrace(first.before,ids,positives,artifact.featureDefinitions,artifact.compositionExamples,parameter,weights);
     for(let k=0;k<4;k++){
       close(t.rawGradients[k],full.rawParts[k][parameter]);close(t.parts[k],full.parts[k][parameter]);
       close(t.rows[k].reduce((s,r)=>s+r.gradient,0),t.parts[k]);
       for(const r of t.rows[k]){close(r.rawGradient,r.rawDerivative/r.divisor);close(r.gradient,r.rawDerivative*r.coefficient/r.divisor);}
     }
     close(t.total,full.total[parameter]);checked++;
   }
 }
 for(const step of [1,100,6000,6001,12000,manifest.steps]){
   const r=read(step),weights=core.optimizerAt(manifest.optimizer,step).lossWeights;
   const full=core.gradients(r.before,r.ids,r.positives,artifact.featureDefinitions,artifact.compositionExamples,0,weights);
   for(const parameter of [6,31,506,1250,1251,1270,1300,1301,1320,1350]){
     const t=core.parameterTrace(r.before,r.ids,r.positives,artifact.featureDefinitions,artifact.compositionExamples,parameter,weights);
     close(t.total,r.gradient[parameter]);t.parts.forEach((v,k)=>close(v,full.parts[k][parameter]));checked++;
     if(version==='v2'&&step<=6000&&[31,1251,1300,1301,1320,1350].includes(parameter)){assert.equal(t.total,0);assert.equal(r.before[parameter],r.after[parameter]);}
     if(parameter===1350)assert.deepEqual(t.rows.slice(1),[[],[],[]]);
   }
 }
}
assert.throws(()=>core.parameterTrace(new Float64Array(core.P),new Uint32Array(),0,[],[],0));
console.log(`PASS loss provenance: ${checked} parameter audits; all 1,351 paths in both versions and phase configurations; individual row factors, all four losses, real batch totals, zero gradients and both biases.`);
