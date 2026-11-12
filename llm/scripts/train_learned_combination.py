"""Learn second-layer combinations using final labels only.

H1 targets and sparse signed connections are explicit teaching constraints.
H2 and output are dense, signed freely, and receive classification gradients only.
No geometric-condition labels, gates, margins, sampling, or analytic initialization.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,base64,gzip,hashlib,itertools,json,math,time
from pathlib import Path
import numpy as np
import train_geometry_features as base
ROOT=Path(__file__).resolve().parents[1]
PRESET='square-trained-learned-combination-v1'
TRACE='learned-combination-v1'
MODE='learned-combination'
POS=base.POS_IDS

def definition(width=16):
    features=base.primitive_features(); mask=np.zeros((119,25),bool);sign=np.zeros((119,25))
    for f in features:
        j=f['neuronIndex'];mask[j,f['on']+f['off']]=True;sign[j,f['on']]=1;sign[j,f['off']]=-1
    masks=[mask,np.ones((width,119),bool),np.ones((1,width),bool)]
    signs=[sign,np.zeros((width,119)),np.zeros((1,width))]
    layers=[];mapping=[]
    for k,(mask,sign) in enumerate(zip(masks,signs),1):
        indices=np.full(mask.shape,-1,int)
        for j,i in zip(*np.nonzero(mask)):
            idx=len(mapping);indices[j,i]=idx;s=sign[j,i]
            mapping.append(dict(index=idx,layer=k,neuron=int(j),input=int(i),kind='weight',lower=.002 if s>0 else -128,upper=-.002 if s<0 else 128))
        bi=[]
        for j in range(mask.shape[0]):
            idx=len(mapping);bi.append(idx);mapping.append(dict(index=idx,layer=k,neuron=j,input=None,kind='bias',lower=-128,upper=128))
        layers.append(dict(mask=mask,sign=sign,indices=indices,biases=np.array(bi)))
    d=dict(kind=MODE,features=features,topology=[25,119,width,1],layers=layers,parameterMap=mapping,count=len(mapping),
        lossConfig=dict(classification=dict(positive=1.6,negative=.4),local=dict(coefficient=.2),geometry=dict(coefficient=0),boundary=dict(localMargin=8.,geometryEnabled=False,coefficient=1.),participation=dict(coefficient=0,minimum=0.)))
    d['on']=np.array([sum(1<<i for i in f['on']) for f in features],np.uint32);d['off']=np.array([sum(1<<i for i in f['off']) for f in features],np.uint32)
    d['violations']=[(f['neuronIndex'],i,0) for f in features for i in f['on']]+[(f['neuronIndex'],i,1) for f in features for i in f['off']]
    d['lo']=np.array([v['lower'] for v in mapping]);d['hi']=np.array([v['upper'] for v in mapping])
    return d

def loss_and_gradients(p,ids,labels,d,parts=False):
    x=base.pixels(ids);n=len(x);params=base.unpack(p,d);acts,zs=base.forward(x,p,d);h=acts[1]
    truth=(((ids[:,None]&d['on'])==d['on'])&((ids[:,None]&d['off'])==0)).astype(float)
    zero=lambda:[np.zeros_like(z) for z in zs]
    error=np.where(labels[:,None]>0,np.minimum(acts[-1]-1.6,0),np.maximum(acts[-1]-.4,0))
    delta=zero();delta[-1]=2*error/n;gc=base.backward(delta,acts,params,d);lc=float(np.mean(error**2))
    error=h-truth;delta=zero();delta[0]=.4*error*h*(1-h)/(n*119);gl=base.backward(delta,acts,params,d);ll=float(.2*np.mean(error**2))
    w,b=params[0];xp=(w<0).astype(float)
    for f in d['features']:xp[f['neuronIndex'],f['on']]=1;xp[f['neuronIndex'],f['off']]=0
    ep=np.minimum(np.sum(xp*w,axis=1)+b-8,0);gw=2*ep[:,None]*xp/119;gb=2*ep/119;lb=float(ep@ep/119)
    indices=np.array([r[0] for r in d['violations']]);bits=np.array([r[1] for r in d['violations']]);values=np.array([r[2] for r in d['violations']])
    xn=(w[indices]>0).astype(float);xn[np.arange(len(indices)),bits]=values
    en=np.maximum(np.sum(xn*w[indices],axis=1)+b[indices]+8,0);lb+=float(en@en/119)
    np.add.at(gw,indices,2*en[:,None]*xn/119);np.add.at(gb,indices,2*en/119)
    ws=[gw]+[np.zeros_like(v[0]) for v in params[1:]];bs=[gb]+[np.zeros_like(v[1]) for v in params[1:]]
    gbound=base.pack_gradient(ws,bs,d);null=np.zeros(d['count']);grads=[gc,gl,null,gbound,null]
    return np.array([lc,ll,0.,lb,0.]),gc+gl+gbound,grads if parts else None

def adam(p,m,v,g,step,rate,d):
    m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
    proposed=p-rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    p[:]=np.clip(proposed,d['lo'],d['hi']);return proposed

def training_pool():
    entries=set(int(v) for v in POS);near=set()
    for value in list(entries):
        for b in range(25):
            near.add(value^(1<<b))
            for c in range(b+1,25):near.add(value^(1<<b)^(1<<c))
    rectangles=set()
    for h in range(1,6):
      for w in range(1,6):
       for r in range(6-h):
        for c in range(6-w):
         rectangles.add(sum(1<<((r+i)*5+c+j) for i in range(h) for j in range(w) if i in (0,h-1) or j in (0,w-1)))
    for a,b in itertools.combinations(POS,2):rectangles.add(int(a)|int(b))
    entries.update(near);entries.update(rectangles);entries.add(0);entries.add((1<<25)-1)
    rng=np.random.default_rng(20261112);entries.update(rng.integers(1<<25,size=8192).tolist())
    return np.array(sorted(entries),np.uint32),np.array(sorted(near|rectangles),np.uint32)

def batch(rng,pool,near,hard):
    chunks=[POS[rng.integers(len(POS),size=128)],near[rng.integers(len(near),size=192)],pool[rng.integers(len(pool),size=64)],rng.integers(1<<25,size=128,dtype=np.uint32)]
    if len(hard):chunks.append(hard[rng.integers(len(hard),size=128)])
    ids=np.concatenate(chunks);return ids,np.isin(ids,POS).astype(float)

def make_tables(p,d):
    params=base.unpack(p,d);width=d['topology'][2];windows=[[5*(r+i)+c+j for i in range(3) for j in range(3)] for r in range(3) for c in range(3)]
    tables=[np.zeros((512,width)) for _ in windows];patterns=[]
    for bits in windows:
        x=np.zeros((512,25));x[:,bits]=(np.arange(512)[:,None]>>np.arange(9))&1;patterns.append(x)
    for f in d['features']:
        j=f['neuronIndex'];support=set(f['on']+f['off']);k=next(i for i,w in enumerate(windows) if support.issubset(w))
        h=base.sigmoid(patterns[k]@params[0][0][j]+params[0][1][j]);tables[k]+=h[:,None]*params[1][0][:,j]
    return params,windows,tables

def table_forward(ids,data,details=False):
    params,windows,tables=data;z=np.broadcast_to(params[1][1],(len(ids),len(params[1][1]))).copy()
    for bits,table in zip(windows,tables):
        code=np.zeros(len(ids),np.uint16)
        for k,bit in enumerate(bits):code|=((ids>>bit)&1).astype(np.uint16)<<k
        z+=table[code]
    score=base.sigmoid(z)@params[2][0][0]+params[2][1][0]
    return (score,z) if details else score

def verify_lookup_forward(p,d,label):
    rng=np.random.default_rng(20261113)
    ids=np.unique(np.concatenate([POS,np.array([int(x)^(1<<i) for x in POS for i in range(25)],np.uint32),np.array([0,(1<<25)-1],np.uint32),rng.integers(1<<25,size=4096,dtype=np.uint32)]))
    acts,zs=base.forward(base.pixels(ids),p,d);fast,z=table_forward(ids,make_tables(p,d),True)
    score_error=float(np.max(abs(fast-acts[-1][:,0])));hidden_error=float(np.max(abs(z-zs[1])))
    result=dict(label=label,inputCount=len(ids),maximumScoreError=score_error,maximumHiddenInputError=hidden_error,absoluteTolerance=1e-10,passed=score_error<1e-10 and hidden_error<1e-10)
    assert result['passed'],result
    return result

def exhaustive(p,d,cap=1<<25):
    start=time.time();data=make_tables(p,d);errors=0;maxneg=-math.inf;minpos=math.inf;hard=[]
    for begin in range(0,cap,1<<18):
        ids=np.arange(begin,min(begin+(1<<18),cap),dtype=np.uint32);score=table_forward(ids,data);positive=np.isin(ids,POS);bad=(score>1)!=positive
        errors+=int(bad.sum());neg=~positive
        if positive.any():minpos=min(minpos,float(score[positive].min()))
        if neg.any():maxneg=max(maxneg,float(score[neg].max()))
        candidates=np.flatnonzero(neg&(score>.55))
        if len(candidates):
            chosen=candidates[np.argsort(score[candidates])[-32:]];hard.extend((float(score[k]),int(ids[k])) for k in chosen)
        if begin%(1<<23)==0:print('EXHAUSTIVE',begin,'errors',errors,flush=True)
    verification=dict(inputCount=cap,positiveCount=int(np.sum(POS<cap)),errors=errors,threshold=1,comparison='gt',minimumPositive=minpos,maximumNegative=maxneg,seconds=time.time()-start)
    mined=np.array(sorted({x for _,x in sorted(hard,reverse=True)[:8192]}),np.uint32)
    print('EXHAUSTIVE DONE',verification,'mined',len(mined),flush=True);return verification,mined

def finite_differences(p,ids,labels,d):
    _,g,parts=loss_and_gradients(p,ids,labels,d,True);rows=[]
    for layer in range(1,4):
      for kind in ['weight','bias']:
       candidates=[a['index'] for a in d['parameterMap'] if a['layer']==layer and a['kind']==kind]
       for idx in candidates[:2]:
        plus=p.copy();minus=p.copy();plus[idx]+=1e-5;minus[idx]-=1e-5
        la,_,_=loss_and_gradients(plus,ids,labels,d);lb,_,_=loss_and_gradients(minus,ids,labels,d)
        numeric=(la-lb)/2e-5;analytic=np.array([g[idx] for g in parts]);error=float(np.max(abs(numeric-analytic)))
        rows.append(dict(parameter=idx,analytic=analytic.tolist(),numeric=numeric.tolist(),maximumError=error))
    return dict(passed=all(r['maximumError']<2e-4 for r in rows),epsilon=1e-5,rows=rows)

class Recorder:
    def __init__(self,d,rate):
        self.d=d;self.rate=rate;self.dest=ROOT/'training-trace'/TRACE
        self.dest.mkdir(parents=True,exist_ok=True);self.chunks=[];self.timeline=[];self.checked=0;self.maximum_error=0.
    def before(self,step,p,m,v,ids,labels,loss,g):
        if (step-1)%100==0:self.block=dict(format='geometry-trace-v1',firstStep=step,count=0,start=dict(parameters=p.tolist(),m=m.tolist(),v=v.tolist()),updates=[])
        self.block['updates'].append(dict(step=step,gradient=g.tolist(),ids=ids.astype(int).tolist(),labels=labels.astype(int).tolist(),losses=loss.tolist()))
        self.timeline.append([step,*loss.tolist()])
    def after(self,step,p,m,v,force=False):
        if step%100 and not force:return
        self.block['count']=len(self.block['updates']);self.block['end']=dict(parameters=p.tolist(),m=m.tolist(),v=v.tolist())
        self.verify(self.block)
        raw=json.dumps(self.block,separators=(',',':'),allow_nan=False).encode();packed=gzip.compress(raw,compresslevel=6,mtime=0)
        index=len(self.chunks);name=f'chunk-{index:03d}.js'
        (self.dest/name).write_text(f'SquareTraceChunks.receive("{TRACE}",{index},"{base64.b64encode(packed).decode()}");\n',encoding='ascii')
        self.chunks.append(dict(index=index,firstStep=self.block['firstStep'],lastStep=step,file=name,sha256=hashlib.sha256(packed).hexdigest(),bytes=len(packed)))
    def verify(self,block):
        p=np.array(block['start']['parameters']);m=np.array(block['start']['m']);v=np.array(block['start']['v'])
        for update in block['updates']:
            ids=np.array(update['ids'],np.uint32);labels=np.array(update['labels'],float)
            loss,g,_=loss_and_gradients(p,ids,labels,self.d);error=float(np.max(abs(g-np.array(update['gradient']))));self.maximum_error=max(self.maximum_error,error)
            assert error<1e-10 and np.allclose(loss,update['losses'],atol=1e-10,rtol=1e-10)
            adam(p,m,v,g,update['step'],self.rate,self.d);self.checked+=1
        assert all(np.array_equal(a,np.array(block['end'][key])) for a,key in [(p,'parameters'),(m,'m'),(v,'v')])

def publish(p,d,recorder,verification,feature_verification,checks,initial,args,step,history):
    model=base.model_of(p,d);sha=hashlib.sha256(json.dumps(model,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    initialization=dict(method='uniform',bounds=[-.1,.1],weightSeed=args.seed,biases='zero',inheritedParameters=False)
    training=dict(seed=args.seed,dataSeed=20261112,steps=step,optimizer='Adam',learningRate=args.rate,initialization=initialization,
        trainableParameters=d['count'],denseParameters=sum(a*b+b for a,b in zip(d['topology'][:-1],d['topology'][1:])),allLossesActiveFromStep=1,
        parameterBounds=[-128,128],structuralMask=True,parameterChangeL2=float(np.linalg.norm(p-initial)),lossConfig=d['lossConfig'],history=history,
        supervision=dict(firstLayer='119 human-defined local targets with signed sparse connections',secondLayer='No targets, geometric labels, margins or sign constraints; classification loss only',output='Classification loss only; signed unconstrained connections'),
        sampling=dict(description='All 30 positives; one/two bit flips, all rectangular outlines, pairwise positive unions, uniform random inputs, classification mistakes and high-scoring negative inputs (score > 0.55), no geometric targets',checkEvery=args.check_every,poolRefreshEvery=500),
        reproducibility=dict(pythonNumpyVersion=np.__version__,command=f'python llm/scripts/train_learned_combination.py --width {args.width} --steps {step} --rate {args.rate} --seed {args.seed} --check-every {args.check_every} --record'))
    artifact=dict(presetId=PRESET,learningMode=MODE,model=model,activations=model['activations'],featureDefinitions=d['features'],geometryDefinitions=[],compositionExamples=base.examples(),parameterMap=d['parameterMap'],
        training=training,verification=verification,featureVerification=feature_verification,gradientVerification=dict(passed=all(c['passed'] for c in checks),checks=checks),modelSha256=sha,
        name='第一层特征引导＋第二层自主组合',lookupForwardVerification=[verify_lookup_forward(initial,d,'random initialization'),verify_lookup_forward(np.round(p,6),d,'published six-decimal parameters')])
    definitions=[dict(id='classification',name='分类误差',formula='mean_n marginError(score_n)^2',coefficient=1),
        dict(id='local',name='第一层局部特征监督',formula='0.2 / (N F) sum_n,j (h_nj - target_nj)^2',coefficient=1),
        dict(id='geometry',name='第二层几何监督（未设置）',formula='0; no H2 targets or geometric condition labels',coefficient=0),
        dict(id='boundary',name='第一层检测边界',formula='sum_j [min(z_match-8,0)^2 + sum_bit max(z_violate+8,0)^2] / F; H1 only',coefficient=1),
        dict(id='participation',name='输出方向约束（未设置）',formula='0; signed output weights are free',coefficient=0)]
    stages=sorted(set([0,1,25,50,100,250,500,1000,step//2,step*3//4,step]));stages=[x for x in stages if x<=step]
    manifest=dict(version=3,id=TRACE,dataVersion=TRACE,dataPath=f'training-trace/{TRACE}/',presetId=PRESET,learningMode=MODE,modelSha256=sha,topology=d['topology'],activations=model['activations'],
        parameterCount=d['count'],parameterMap=d['parameterMap'],steps=step,chunkSize=100,stages=stages,precision=6,initialization=initialization,seed=args.seed,dataSeed=20261112,
        optimizer=dict(name='Adam',learningRate=args.rate,beta1=.9,beta2=.999,epsilon=1e-8,lossWeights=[1,1,0,1,0],bounds=[-128,128]),lossDefinitions=definitions,lossConfig=d['lossConfig'],chunks=recorder.chunks,
        numericTolerance=dict(absolute=1e-8,relative=1e-10),provenance='Recorded deterministic joint training from small random parameters; only H1 has feature targets, H2/output learn from final classification labels.',
        verification=dict(stepsChecked=recorder.checked,maximumGradientError=recorder.maximum_error,finalModelExactMatch=True,exhaustiveErrors=verification['errors']),timeline=recorder.timeline)
    (ROOT/'learned-combination-model.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (recorder.dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    (ROOT/'learned-combination-verification.json').write_text(json.dumps(dict(presetId=PRESET,modelSha256=sha,classification=verification,features=feature_verification,gradients=artifact['gradientVerification'],replay=manifest['verification'],lookupForward=artifact['lookupForwardVerification']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PUBLISHED',PRESET,sha,recorder.checked,'updates',sum(c['bytes'] for c in recorder.chunks),'compressed bytes',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--width',type=int,default=8);ap.add_argument('--steps',type=int,default=6000);ap.add_argument('--rate',type=float,default=.05);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--record',action='store_true');ap.add_argument('--check-every',type=int,default=3000);args=ap.parse_args()
    d=definition(args.width);rng=np.random.default_rng(args.seed);p=rng.uniform(-.1,.1,d['count']);p[[v['index'] for v in d['parameterMap'] if v['kind']=='bias']]=0;initial=p.copy();m=np.zeros_like(p);v=np.zeros_like(p)
    pool,near=training_pool();rng=np.random.default_rng(20261112);hard=np.array([],np.uint32);mined=hard.copy();start=time.time();history=[];checks=[];recorder=Recorder(d,args.rate) if args.record else None
    print('START',d['topology'],d['count'],'pool',len(pool),'near',len(near),'rate',args.rate,flush=True)
    for step in range(1,args.steps+1):
        ids,labels=batch(rng,pool,near,hard);loss,g,_=loss_and_gradients(p,ids,labels,d)
        if recorder and step in [1,500,1500]:checks.append(dict(step=step,**finite_differences(p,ids[:64],labels[:64],d)))
        if recorder:recorder.before(step,p,m,v,ids,labels,loss,g)
        adam(p,m,v,g,step,args.rate,d)
        if recorder:recorder.after(step,p,m,v,force=step==args.steps)
        if step%500==0:
            scores=table_forward(pool,make_tables(p,d));wrong=(scores>1)!=np.isin(pool,POS);hard=np.unique(np.concatenate([pool[wrong],mined]))
            print('STEP',step,'seconds',round(time.time()-start,1),'loss',loss.tolist(),'pool-errors',int(wrong.sum()),'hard',len(hard),flush=True)
            np.savez(ROOT/f'.learned-combination-w{args.width}-trial.npz',p=p,m=m,v=v,step=step)
        if step%args.check_every==0 or step==args.steps:
            verification,mined=exhaustive(np.round(p,6),d);history.append(dict(step=step,**verification));hard=np.unique(np.concatenate([hard,mined]))
            fv=base.verify_features(np.round(p,6),d);print('FEATURES',fv['passed'],flush=True)
            if verification['errors']==0 and fv['passed']:
                print('SUCCESS',step,args.width,flush=True)
                if recorder:
                    checks.append(dict(step=step,**finite_differences(p,ids[:64],labels[:64],d)));assert all(c['passed'] for c in checks)
                    publish(p,d,recorder,verification,fv,checks,initial,args,step,history)
                (ROOT/f'.learned-combination-w{args.width}-trial.json').write_text(json.dumps(dict(args=vars(args),step=step,history=history,verification=verification,featureVerification=fv),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                np.savez(ROOT/f'.learned-combination-w{args.width}-trial.npz',p=p,m=m,v=v,step=step)
                return
    print('NOT YET ACCEPTED',flush=True)
    if recorder:raise RuntimeError('No model published: exhaustive/feature acceptance not met')

if __name__=='__main__':main()
