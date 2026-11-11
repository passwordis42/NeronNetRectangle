"""Train sparse, fully interpretable square networks and record real Adam updates.

Every trainable parameter starts at a small seeded random value (biases at zero).
Feature definitions supply supervision and sparsity, never inference shortcuts.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, base64, gzip, hashlib, itertools, json, math, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FULL = (1 << 25)-1
LEFT = sum(1 << (5*r) for r in range(5))
RIGHT = LEFT << 4
LOSS_NAMES = ['classification', 'local', 'geometry', 'boundary', 'participation']

def pixels(ids):
    return ((np.asarray(ids, dtype=np.uint32)[:, None] >> np.arange(25, dtype=np.uint32)) & 1).astype(float)

def square(side, row, col):
    return sum(1 << ((row+i)*5+col+j) for i in range(side) for j in range(side)
               if i in (0,side-1) or j in (0,side-1))

POS_IDS = np.array([square(s,r,c) for s in range(2,6) for r in range(6-s) for c in range(6-s)], dtype=np.uint32)

def primitive_features():
    out=[]
    def add(kind,name,on,off=()):
        out.append(dict(neuronIndex=len(out),kind=kind,name=name,on=sorted(on),off=sorted(off)))
    for p in range(25): add('point',f'亮点 X{p+1}',[p])
    for kind,dirs,title in [('horizontal','LR','水平直线'),('vertical','UD','竖直线'),
                             ('corner-tl','DR','左上转角'),('corner-tr','DL','右上转角'),
                             ('corner-bl','UR','左下转角'),('corner-br','UL','右下转角')]:
        for r in range(5):
            for c in range(5):
                p=r*5+c; nb={}
                if r: nb['U']=p-5
                if r<4: nb['D']=p+5
                if c: nb['L']=p-1
                if c<4: nb['R']=p+1
                if all(d in nb for d in dirs):
                    add(kind,f'{title} · 第{r+1}行第{c+1}列',[p]+[nb[d] for d in dirs],[v for d,v in nb.items() if d not in dirs])
    assert len(out)==119
    return out

def forbidden_features():
    """Prime forbidden cubes, deterministic per-window greedy cover, global dedup."""
    cubes={}; window_counts=[]
    for row in range(3):
        for col in range(3):
            positions=[(row+r)*5+col+c for r in range(3) for c in range(3)]
            allowed={sum(((int(x)>>p)&1)<<i for i,p in enumerate(positions)) for x in POS_IDS}
            remaining=sum(1<<v for v in range(512) if v not in allowed)
            candidates=[]
            for mask in range(1,512):
                on=mask
                while True:
                    if not any((a&mask)==on for a in allowed):
                        if all(any((a&(mask^(1<<bit)))==(on&(mask^(1<<bit))) for a in allowed) for bit in range(9) if mask&(1<<bit)):
                            cover=sum(1<<v for v in range(512) if (v&mask)==on)
                            candidates.append((mask,on,cover))
                    if on==0: break
                    on=(on-1)&mask
            chosen=[]
            while remaining:
                mask,on,cover=min(candidates,key=lambda q:(-(q[2]&remaining).bit_count(),q[0].bit_count(),q[0],q[1]))
                assert cover&remaining
                chosen.append((mask,on));remaining &= ~cover
            window_counts.append(len(chosen))
            for mask,on in chosen:
                gm=sum(1<<p for i,p in enumerate(positions) if mask&(1<<i))
                go=sum(1<<p for i,p in enumerate(positions) if on&(1<<i))
                cubes.setdefault((gm,go),[]).append([row,col])
    out=[]
    for (mask,on),windows in sorted(cubes.items()):
        out.append(dict(neuronIndex=119+len(out),kind='violation',name=f'局部违规 {len(out)+1}',
                        on=[p for p in range(25) if on&(1<<p)],off=[p for p in range(25) if (mask^on)&(1<<p)],windows=windows))
    assert len(out)==233,(len(out),window_counts)
    return out,dict(windowDetectorCounts=window_counts,beforeDeduplication=sum(window_counts),uniqueCount=len(out),maximumPixels=max(len(f['on'])+len(f['off']) for f in out))

def targets(x, features):
    return np.stack([np.all(x[:,f['on']]==1,axis=1)&np.all(x[:,f['off']]==0,axis=1) for f in features],axis=1).astype(float)

def geometry_targets(ids):
    x=np.asarray(ids,dtype=np.uint32)
    l=(x<<1)&np.uint32(FULL^LEFT); r=(x>>1)&np.uint32(FULL^RIGHT)
    u=(x<<5)&np.uint32(FULL);d=x>>5
    h=x&l&r&~u&~d;v=x&u&d&~l&~r
    tl=x&d&r&~u&~l;tr=x&d&l&~u&~r;bl=x&u&r&~d&~l;br=x&u&l&~d&~r
    corners=np.bitwise_count(tl|tr|bl|br); hc=np.bitwise_count(h);vc=np.bitwise_count(v)
    return np.stack([(h|v|tl|tr|bl|br)==x,corners>=4,corners<=4,hc>=vc,vc>=hc],axis=1).astype(float)

def definition(kind):
    features=primitive_features(); discovery=None
    if kind=='single':
        extra,discovery=forbidden_features();features+=extra
    topology=[25,len(features),1] if kind=='single' else [25,119,5,1]
    masks=[]; signs=[]
    mask=np.zeros((len(features),25),bool);sign=np.zeros_like(mask,dtype=float)
    for f in features:
        j=f['neuronIndex'];mask[j,f['on']+f['off']]=True;sign[j,f['on']]=1;sign[j,f['off']]=-1
    masks.append(mask);signs.append(sign)
    if kind=='double':
        m=np.zeros((5,119),bool);sg=np.zeros((5,119))
        m[0,:]=1;sg[0,:25]=-1;sg[0,25:]=1
        m[1:3,55:]=1;sg[1,55:]=1;sg[2,55:]=-1
        m[3:5,25:55]=1;sg[3,25:40]=1;sg[3,40:55]=-1;sg[4]=-sg[3]
        masks.append(m);signs.append(sg)
    masks.append(np.ones((1,topology[-2]),bool))
    sg=np.ones_like(masks[-1],dtype=float)
    if kind=='single':sg[0,119:]=-1
    signs.append(sg)
    parameter_map=[];offset=0;layers=[]
    for layer,(mask,sign) in enumerate(zip(masks,signs),start=1):
        indices=np.full(mask.shape,-1,int);biases=[]
        for j,i in zip(*np.nonzero(mask)):
            indices[j,i]=offset
            parameter_map.append(dict(index=offset,layer=layer,neuron=int(j),input=int(i),kind='weight',lower=0.002 if sign[j,i]>0 else -128,upper=128 if sign[j,i]>0 else -0.002))
            offset+=1
        for j in range(mask.shape[0]):
            biases.append(offset);parameter_map.append(dict(index=offset,layer=layer,neuron=j,input=None,kind='bias',lower=-128,upper=128));offset+=1
        layers.append(dict(mask=mask,sign=sign,indices=indices,biases=np.array(biases)))
    return dict(kind=kind,features=features,discovery=discovery,topology=topology,layers=layers,parameterMap=parameter_map,
                count=offset,lossConfig=dict(classification=dict(positive=1.6,negative=.4),local=dict(coefficient=.2),geometry=dict(coefficient=.2),boundary=dict(localMargin=8.,geometryMargin=4.,coefficient=1.),participation=dict(minimum=.02,coefficient=1.)))

def unpack(p,d):
    result=[]
    for layer in d['layers']:
        w=np.zeros(layer['mask'].shape);w[layer['mask']]=p[layer['indices'][layer['mask']]]
        result.append((w,p[layer['biases']]))
    return result

def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-60,60)))

def forward(x,p,d):
    acts=[x];zs=[]
    for k,(w,b) in enumerate(unpack(p,d)):
        z=acts[-1]@w.T+b;zs.append(z);acts.append(z if k==len(d['layers'])-1 else sigmoid(z))
    return acts,zs

def pack_gradient(ws,bs,d):
    g=np.zeros(d['count'])
    for w,b,l in zip(ws,bs,d['layers']):g[l['indices'][l['mask']]]=w[l['mask']];g[l['biases']]=b
    return g

def backward(deltas,acts,params,d):
    ws=[np.zeros_like(w) for w,b in params];bs=[np.zeros_like(b) for w,b in params]
    for k in range(len(params)-1,-1,-1):
        dz=deltas[k];ws[k]+=dz.T@acts[k];bs[k]+=dz.sum(axis=0)
        if k:deltas[k-1]+=(dz@params[k][0]) * acts[k]*(1-acts[k])
    return pack_gradient(ws,bs,d)

def loss_and_gradients(p,ids,labels,d,parts=False):
    x=pixels(ids);n=len(x);params=unpack(p,d);acts,zs=forward(x,p,d)
    h=acts[1];truth=targets(x,d['features']);geo=geometry_targets(ids) if d['kind']=='double' else None
    losses=[];grads=[]
    def zero():return [np.zeros_like(z) for z in zs]
    error=np.where(labels[:,None]>0,np.minimum(acts[-1]-1.6,0),np.maximum(acts[-1]-.4,0))
    delta=zero();delta[-1]=2*error/n;losses.append(float(np.mean(error**2)));grads.append(backward(delta,acts,params,d))
    error=h-truth;delta=zero();delta[0]=.4*error*h*(1-h)/(n*h.shape[1])
    losses.append(float(.2*np.mean(error**2)));grads.append(backward(delta,acts,params,d))
    delta=zero()
    if geo is not None:
        gh=acts[2];error=gh-geo;delta[1]=.4*error*gh*(1-gh)/(n*5);losses.append(float(.2*np.mean(error**2)))
    else:losses.append(0.)
    grads.append(backward(delta,acts,params,d))
    # Exact linear extrema of each local detector. Its mask fixes all unrelated inputs to zero.
    gw=[np.zeros_like(w) for w,b in params];gb=[np.zeros_like(b) for w,b in params];boundary=0.;F=len(d['features'])
    for f in d['features']:
        j=f['neuronIndex'];w=params[0][0][j];b=params[0][1][j]
        xp=np.zeros(25);xp[f['on']]=1
        # Include arbitrary signs during the unprojected random initial state.
        xp[:]=(w<0);xp[f['on']]=1;xp[f['off']]=0
        ep=min(float(xp@w+b)-8,0);boundary+=ep*ep/F;gw[0][j]+=2*ep*xp/F;gb[0][j]+=2*ep/F
        for bit,required in [(i,1) for i in f['on']]+[(i,0) for i in f['off']]:
            xn=(w>0).astype(float);xn[bit]=1-required
            en=max(float(xn@w+b)+8,0);boundary+=en*en/F;gw[0][j]+=2*en*xn/F;gb[0][j]+=2*en/F
    gradient=pack_gradient(gw,gb,d)
    if geo is not None:
        error=np.where(geo>0,np.minimum(zs[1]-4,0),np.maximum(zs[1]+4,0))
        delta=zero();delta[1]=2*error/(n*5);boundary+=float(np.mean(error**2));gradient+=backward(delta,acts,params,d)
    losses.append(float(boundary));grads.append(gradient)
    last=d['layers'][-1];wi=last['indices'][last['mask']];sign=last['sign'][last['mask']]
    err=np.minimum(p[wi]*sign-.02,0);g=np.zeros(d['count']);g[wi]=2*err*sign/len(wi)
    losses.append(float(np.mean(err**2)));grads.append(g)
    return np.array(losses),np.sum(grads,axis=0),grads if parts else None

def adam(p,m,v,g,step,rate,d):
    m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
    proposal=p-rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    low=np.array([a['lower'] for a in d['parameterMap']]);high=np.array([a['upper'] for a in d['parameterMap']])
    p[:]=np.clip(proposal,low,high)
    return proposal

def model_of(p,d):
    return dict(topology=d['topology'],activations=['sigmoid']*(len(d['layers'])-1)+['linear'],
                layers=[dict(weights=np.round(w,6).tolist(),biases=np.round(b,6).tolist()) for w,b in unpack(p,d)])

def training_pool():
    ids=set(int(x) for x in POS_IDS);legal=[]
    for begin in range(0,1<<25,1<<19):
        x=np.arange(begin,begin+(1<<19),dtype=np.uint32);good=geometry_targets(x)[:,0]>0
        legal.extend(x[good].tolist())
    for x in list(ids)+legal:
        ids.add(x)
        for bit in range(25):ids.add(x^(1<<bit))
    rng=np.random.default_rng(20261111);ids.update(rng.integers(1<<25,size=16384).tolist())
    return np.array(sorted(ids),dtype=np.uint32),np.array(legal,dtype=np.uint32)

def build_batch(rng,pool,legal,hard=None):
    entries=[POS_IDS[rng.integers(30,size=96)],legal[rng.integers(len(legal),size=96)],pool[rng.integers(len(pool),size=192)],rng.integers(1<<25,size=128,dtype=np.uint32)]
    if hard is not None and len(hard):entries.append(hard[rng.integers(len(hard),size=128)])
    ids=np.concatenate(entries);return ids,np.isin(ids,POS_IDS).astype(float)

def verify_features(p,d):
    params=unpack(p,d);rows=[]
    for f in d['features']:
        bits=f['on']+f['off'];x=np.zeros((1<<len(bits),25))
        x[:,bits]=((np.arange(len(x))[:,None]>>np.arange(len(bits)))&1)
        j=f['neuronIndex'];response=sigmoid(x@params[0][0][j]+params[0][1][j]);truth=targets(x,[f])[:,0]>0
        low=float(response[truth].min());high=float(response[~truth].max())
        mask=d['layers'][0]['mask'][j];w=params[0][0][j]
        okay=bool(low>.5 and high<.5 and np.all(w[~mask]==0) and np.all(w[mask]*d['layers'][0]['sign'][j,mask]>0))
        rows.append(dict(neuronIndex=j,name=f['name'],satisfyingMinimum=low,violatingMaximum=high,localAssignments=len(x),outsideInvariant=True,passed=okay))
    return dict(passed=all(r['passed'] for r in rows),method='Enumerate every assignment of each exact local support; all outside weights are structurally zero.',features=rows)

def exhaustive(p,d,batch_size=131072):
    """Exact floating forward using small receptive-field lookup tables, not rule inference."""
    params=unpack(p,d);tables=[]
    for f in d['features']:
        bits=f['on']+f['off'];x=np.zeros((1<<len(bits),25));x[:,bits]=((np.arange(len(x))[:,None]>>np.arange(len(bits)))&1)
        j=f['neuronIndex'];tables.append((bits,sigmoid(x@params[0][0][j]+params[0][1][j])))
    errors=0;geometry_errors=np.zeros(5,dtype=np.int64);max_negative=-math.inf;min_positive=math.inf;hard=[];started=time.time()
    for begin in range(0,1<<25,batch_size):
        ids=np.arange(begin,min(begin+batch_size,1<<25),dtype=np.uint32)
        if d['kind']=='single':score=np.full(len(ids),params[-1][1][0])
        else:z=np.broadcast_to(params[1][1],(len(ids),5)).copy()
        for j,(bits,table) in enumerate(tables):
            code=np.zeros(len(ids),dtype=np.uint8)
            for k,bit in enumerate(bits):code|=((ids>>bit)&1).astype(np.uint8)<<k
            response=table[code]
            if d['kind']=='single':score+=response*params[-1][0][0,j]
            else:
                for q in np.flatnonzero(d['layers'][1]['mask'][:,j]):z[:,q]+=response*params[1][0][q,j]
        if d['kind']=='double':
            geometry=sigmoid(z);score=geometry@params[2][0][0]+params[2][1][0]
            target=geometry_targets(ids)>0;wrong=(geometry>.5)!=target;geometry_errors+=wrong.sum(axis=0)
            for q in range(5):
                bad=np.flatnonzero(wrong[:,q]);hard.extend((2.+abs(float(z[k,q])),int(ids[k])) for k in bad[:32])
        positive=np.isin(ids,POS_IDS);bad=(score>1)!=positive;errors+=int(bad.sum())
        if positive.any():min_positive=min(min_positive,float(score[positive].min()))
        negative=~positive
        if negative.any():max_negative=max(max_negative,float(score[negative].max()))
        candidates=np.flatnonzero(negative&(score>.55))
        if len(candidates):
            chosen=candidates[np.argsort(score[candidates])[-32:]];hard.extend((float(score[k]),int(ids[k])) for k in chosen)
        if begin%(1<<23)==0:print('EXHAUSTIVE',d['kind'],begin,'errors',errors,'geometry',geometry_errors.tolist(),flush=True)
    result=dict(inputCount=1<<25,positiveCount=30,errors=errors,threshold=1,comparison='gt',minimumPositive=min_positive,maximumNegative=max_negative,
                geometryErrors=geometry_errors.tolist() if d['kind']=='double' else None,seconds=time.time()-started)
    mined=np.array(sorted({i for s,i in sorted(hard,reverse=True)[:8192]}),dtype=np.uint32)
    print('EXHAUSTIVE DONE',result,flush=True)
    return result,mined

def grouped_forward(ids,p,d):
    """Cache learned affine contributions per 3x3 window; no labels enter tables."""
    params=unpack(p,d);windows=[[5*(r+i)+c+j for i in range(3) for j in range(3)] for r in range(3) for c in range(3)]
    count=1 if d['kind']=='single' else 5;tables=[np.zeros((512,count)) for _ in windows]
    assignments=[np.zeros((512,25)) for _ in windows]
    for x,bits in zip(assignments,windows):x[:,bits]=((np.arange(512)[:,None]>>np.arange(9))&1)
    for f in d['features']:
        j=f['neuronIndex'];support=set(f['on']+f['off']);k=next(i for i,w in enumerate(windows) if support.issubset(w))
        h=sigmoid(assignments[k]@params[0][0][j]+params[0][1][j])
        tables[k]+=h[:,None]*params[1][0][:,j][None,:]
    ids=np.asarray(ids,dtype=np.uint32);z=np.broadcast_to(params[1][1],(len(ids),count)).copy()
    for bits,table in zip(windows,tables):
        code=np.zeros(len(ids),dtype=np.uint16)
        for k,bit in enumerate(bits):code|=((ids>>bit)&1).astype(np.uint16)<<k
        z+=table[code]
    if d['kind']=='single':return z[:,0],None
    h=sigmoid(z);return (h@params[2][0].T+params[2][1])[:,0],h

class Recorder:
    def __init__(self,d,rate):
        self.d=d;self.rate=rate;self.trace_id=f"geometry-{d['kind']}-v1";self.dest=ROOT/'training-trace'/self.trace_id
        self.dest.mkdir(parents=True,exist_ok=True);self.chunks=[];self.timeline=[];self.checked=0;self.maximum_error=0.;self.gradient_samples=[]
    def before(self,step,p,m,v,ids,labels,losses,g):
        if (step-1)%100==0:
            self.block=dict(format='geometry-trace-v1',firstStep=step,count=0,start=dict(parameters=p.tolist(),m=m.tolist(),v=v.tolist()),updates=[])
        self.block['updates'].append(dict(step=step,gradient=g.tolist(),ids=[int(i) for i in ids],labels=labels.astype(int).tolist(),losses=losses.tolist()))
        self.timeline.append([step,*losses.tolist()])
    def after(self,step,p,m,v,force=False):
        if step%100 and not force:return
        self.block['count']=len(self.block['updates']);self.block['end']=dict(parameters=p.tolist(),m=m.tolist(),v=v.tolist())
        self.verify(self.block)
        raw=json.dumps(self.block,separators=(',',':'),allow_nan=False).encode();packed=gzip.compress(raw,compresslevel=6,mtime=0)
        index=len(self.chunks);name=f'chunk-{index:03d}.js'
        (self.dest/name).write_text(f'SquareTraceChunks.receive("{self.trace_id}",{index},"{base64.b64encode(packed).decode()}");\n',encoding='ascii')
        self.chunks.append(dict(index=index,firstStep=self.block['firstStep'],lastStep=step,file=name,sha256=hashlib.sha256(packed).hexdigest(),bytes=len(packed)))
    def verify(self,block):
        p=np.array(block['start']['parameters']);m=np.array(block['start']['m']);v=np.array(block['start']['v'])
        for update in block['updates']:
            ids=np.array(update['ids'],dtype=np.uint32);labels=np.array(update['labels'],dtype=float)
            loss,g,_=loss_and_gradients(p,ids,labels,self.d)
            error=float(np.max(np.abs(g-np.array(update['gradient']))));self.maximum_error=max(self.maximum_error,error)
            assert error<1e-10 and np.allclose(loss,update['losses'],atol=1e-10,rtol=1e-10)
            adam(p,m,v,g,update['step'],self.rate,self.d);self.checked+=1
        assert all(np.array_equal(a,np.array(block['end'][key])) for a,key in [(p,'parameters'),(m,'m'),(v,'v')])

def finite_differences(p,ids,labels,d):
    _,g,parts=loss_and_gradients(p,ids,labels,d,True);rows=[]
    picks=[]
    for layer in range(1,len(d['layers'])+1):
        for kind in ['weight','bias']:
            candidates=[a['index'] for a in d['parameterMap'] if a['layer']==layer and a['kind']==kind]
            picks.extend(candidates[:2])
    for index in picks:
        epsilon=1e-5;plus=p.copy();minus=p.copy();plus[index]+=epsilon;minus[index]-=epsilon
        la,_,_=loss_and_gradients(plus,ids,labels,d);lb,_,_=loss_and_gradients(minus,ids,labels,d);numeric=(la-lb)/(2*epsilon)
        analytic=np.array([part[index] for part in parts]);error=float(np.max(np.abs(numeric-analytic)))
        rows.append(dict(parameter=index,analytic=analytic.tolist(),numeric=numeric.tolist(),maximumError=error))
    return dict(passed=all(r['maximumError']<2e-4 for r in rows),epsilon=1e-5,rows=rows)

def examples():
    rows=[]
    for name,s,r,c in [('居中3×3',3,1,1),('左上3×3',3,0,0),('右下3×3',3,2,2),('左上4×4',4,0,0),('右下4×4',4,1,1),('外圈5×5',5,0,0)]:
        value=square(s,r,c);boundary=[i for i in range(25) if value&(1<<i)]
        rows.append(dict(name=name,size=s,row=r,col=c,boundary=boundary,missingPixel=r*5+c+s//2,extraPixel=(r+1)*5+c+1))
    return rows

def publish(p,d,recorder,verification,feature_verification,gradient_checks,initial,steps,rate,history):
    model=model_of(p,d);encoded=json.dumps(model,separators=(',',':'),ensure_ascii=False).encode();sha=hashlib.sha256(encoded).hexdigest()
    preset=f"square-trained-geometry-{d['kind']}-v1";activations=model['activations']
    training=dict(seed=42,dataSeed=20261111,steps=steps,optimizer='Adam',learningRate=rate,initialization=dict(method='uniform',bounds=[-.1,.1],weightSeed=42,biases='zero',inheritedParameters=False),
                  trainableParameters=d['count'],denseParameters=sum(a*b+b for a,b in zip(d['topology'][:-1],d['topology'][1:])),allLossesActiveFromStep=1,
                  parameterBounds=[-128,128],structuralMask=True,parameterChangeL2=float(np.linalg.norm(p-initial)),lossConfig=d['lossConfig'],history=history)
    geometry=[dict(neuronIndex=i,name=name) for i,name in enumerate(['每个亮点的邻接合法','转角至少四个','转角至多四个','水平直线不少于竖直直线','竖直直线不少于水平直线'])] if d['kind']=='double' else []
    artifact=dict(presetId=preset,model=model,activations=activations,featureDefinitions=d['features'],geometryDefinitions=geometry,compositionExamples=examples(),parameterMap=d['parameterMap'],
                  training=training,verification=verification,featureVerification=feature_verification,gradientVerification=gradient_checks,modelSha256=sha,featureDiscovery=d['discovery'])
    (ROOT/f"geometry-{d['kind']}-model.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    defs=[dict(id='classification',name='分类误差',formula='mean_n marginError(score_n)^2',coefficient=1),
          dict(id='local',name='局部特征监督',formula='0.2 / (N F) sum_n,j (h_nj - target_nj)^2',coefficient=1),
          dict(id='geometry',name='几何条件监督',formula='0.2 / (5 N) sum_n,q (g_nq - target_nq)^2; single=0',coefficient=1),
          dict(id='boundary',name='检测边界约束',formula='sum_j [min(z_match-8,0)^2 + sum_bit max(z_violate+8,0)^2] / F + double: mean_n,q geometryLogitMargin(z_nq,4)^2',coefficient=1),
          dict(id='participation',name='输出参与约束',formula='mean_j min(sign_j * outputWeight_j - 0.02, 0)^2',coefficient=1)]
    stages=sorted(set([0,1,50,100,250,500,1000,steps//4,steps//2,steps*3//4,steps]));stages=[s for s in stages if s<=steps]
    while len(stages)<11:
        a,b=max(zip(stages,stages[1:]),key=lambda pair:pair[1]-pair[0]);stages.append((a+b)//2);stages.sort()
    manifest=dict(version=3,id=recorder.trace_id,dataVersion=recorder.trace_id,dataPath=f'training-trace/{recorder.trace_id}/',presetId=preset,modelSha256=sha,topology=d['topology'],activations=activations,
                  parameterCount=d['count'],parameterMap=d['parameterMap'],steps=steps,chunkSize=100,stages=stages,precision=6,initialization=training['initialization'],seed=42,dataSeed=20261111,
                  optimizer=dict(name='Adam',learningRate=rate,beta1=.9,beta2=.999,epsilon=1e-8,lossWeights=[1]*5,bounds=[-128,128]),
                  lossDefinitions=defs,lossConfig=d['lossConfig'],chunks=recorder.chunks,numericTolerance=dict(absolute=1e-8,relative=1e-10),
                  provenance='Recorded during deterministic random-initialization training; no inherited or analytic initialization.',
                  verification=dict(stepsChecked=recorder.checked,maximumGradientError=recorder.maximum_error,finalModelExactMatch=True,exhaustiveErrors=verification['errors']),timeline=recorder.timeline)
    (recorder.dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    print('PUBLISHED',preset,sha,recorder.checked,'updates',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['single','double'],required=True);ap.add_argument('--steps',type=int,default=16000);ap.add_argument('--rate',type=float,default=.05);ap.add_argument('--trial',action='store_true');ap.add_argument('--minimum-steps',type=int,default=4000);args=ap.parse_args()
    d=definition(args.kind);print('DEFINITION',args.kind,d['count'],d['discovery'],flush=True)
    rng=np.random.default_rng(42);p=rng.uniform(-.1,.1,d['count']);p[[a['index'] for a in d['parameterMap'] if a['kind']=='bias']]=0
    initial=p.copy();m=np.zeros_like(p);v=np.zeros_like(p);pool,legal=training_pool();rng=np.random.default_rng(20261111);hard=None;start=time.time();history=[];checks=[]
    recorder=None if args.trial else Recorder(d,args.rate);mined=np.array([],dtype=np.uint32)
    for step in range(1,args.steps+1):
        ids,labels=build_batch(rng,pool,legal,hard);losses,g,_=loss_and_gradients(p,ids,labels,d)
        if not args.trial and step in [1,500,4000]:checks.append(dict(step=step,**finite_differences(p,ids[:64],labels[:64],d)))
        if recorder:recorder.before(step,p,m,v,ids,labels,losses,g)
        adam(p,m,v,g,step,args.rate,d)
        if recorder:recorder.after(step,p,m,v,force=step==args.steps)
        if step%500==0:
            acts,z=forward(pixels(pool),p,d);scores=acts[-1][:,0];bad=(scores>1)!=np.isin(pool,POS_IDS);hard=np.unique(np.concatenate([pool[bad],mined]))
            print('STEP',step,'secs',round(time.time()-start,1),'losses',losses.tolist(),'pool-errors',int(bad.sum()),'p-range',[float(p.min()),float(p.max())],flush=True)
            np.savez(ROOT/f'.geometry-{args.kind}-trial.npz',p=p,m=m,v=v,step=step)
        if recorder and step>=args.minimum_steps and (step%4000==0 or step==args.steps):
            rounded=np.round(p,6);fv=verify_features(rounded,d);verification,mined=exhaustive(rounded,d)
            history.append(dict(step=step,**verification));hard=np.unique(np.concatenate([hard,mined]))
            geo_ok=not verification['geometryErrors'] or sum(verification['geometryErrors'])==0
            if fv['passed'] and verification['errors']==0 and geo_ok:
                checks.append(dict(step=step,**finite_differences(p,ids[:64],labels[:64],d)))
                assert all(c['passed'] for c in checks),checks
                publish(p,d,recorder,verification,fv,dict(passed=True,checks=checks),initial,step,args.rate,history)
                return
    if recorder:raise RuntimeError('Training reached its configured limit without complete exhaustive acceptance; no model published.')
    print('FINISHED',flush=True)

if __name__=='__main__':main()
