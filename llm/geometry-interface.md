# Geometry training replay interface

`geometry-backprop-core.js` exports the self-contained global function
`createGeometryBackpropCore(manifest, artifact)`, plus
`GeometryBackpropCore.create` and a CommonJS export. The factory can be serialized
with `toString()` into a worker; it has no external numeric dependencies.

## Model and parameter order

- `artifact.model`: dense `{topology,layers:[{weights,biases}]}`.
- `manifest.activations` (or artifact activations): hidden sigmoid, output linear.
- `manifest.parameterMap`: ordered active parameters. Entries contain
  `{layer,neuron,input,kind,index,label?,lower,upper}`. `layer` is **1-based**;
  neuron/input are 0-based. A bias has `input:null`.
- Fixed absent connections are exactly zero; the trace never invents their updates.
- `engine.P`, `parameterMap`, `describe(index)`, `indexOf(layer,neuron,input,kind)`.
- `engine.model(parameters)` expands active values to a dense model;
  `flatten(model)` returns active values in the manifest order.
- `engine.forward(pixels, model, activations?)` returns
  `{activations,preActivations,outputs}`.

## Recorded chunks

Gzip JSON is delivered as `SquareTraceChunks.receive(traceId,index,base64)`.
After decompression, call `engine.readChunk(jsonStringOrObject)`:

```
{format:'geometry-trace-v1', firstStep, count,
 start:{parameters,m,v},
 updates:[{step,gradient,ids,labels,losses}],
 end:{parameters,m,v}}
```

Labels are explicit 0/1; positive samples need not occur first. The initial
parameters are not projected to sign constraints. After each actual Adam update,
the proposed parameter is projected to its descriptor's `[lower,upper]` range.

`reconstruct(chunk,step)` returns
`{step,before,after,gradient,ids,labels,losses,oldM,oldV,m,v,proposed,
correctedM,correctedV,clipped}`. Step zero is available only from the first chunk;
its `gradient` is null. Reconstruction validates the recorded block endpoint.

## Loss and provenance results

Loss IDs in order: `classification`, `local`, `geometry`, `boundary`,
`participation`. The single-hidden-layer model has a zero geometry component.
The trainer supplies explicit formulas, coefficients, and normalization in the
manifest. `engine.lossDefinitions` contains `{id,name,formula,...}`.

- `optimizerAt(step)` gives the actual phase optimizer and loss weights.
- `gradients(parameters,batch,step,sampleIndex=0)` returns
  `{parts,rawParts,total,losses,rawLosses,lossWeights,sample,...}`.
- `batch` is `{ids,labels}` (a reconstructed update also works).
- The selected `sample` contains `index,id,pixels,positive,label,activations,
  preActivations,outputs` plus the computed per-source derivative data.
- `sampleGradient(sample,parameter,lossId)` returns
  `{raw,coefficient,denominator,contribution,chain,...}`. `raw` is the selected
  sample derivative **before** the loss coefficient and batch averaging;
  `contribution=raw*coefficient/denominator`.
- `parameterTrace(parameters,batch,step,index)` returns
  `{parameter,parts,rawGradients,total,lossWeights,rows}`. `rows[k]` contains
  concrete terms for loss k. A row contains `label,index,id,pixels,loss,
  rawDerivative,rawGradient,coefficient,denominator,gradient,chain` as applicable.
  `rawGradient=rawDerivative/denominator`, `gradient=coefficient*rawGradient`.
- Each `chain` entry is `{formula,substitution,value}`. The last entry gives
  the complete product or sum, so UI can show formulas and substituted numbers.
- A constraint loss has constraint rows rather than a fabricated sample gradient.
- `findNonzeroSample(parameters,batch,step,parameter,lossId,start)` returns a batch
  sample index or -1. It does not change history.

The initial state has no gradient; zero coefficients mean a source is disabled,
not that the underlying raw loss vanished. All displayed input contributions
must use the same selected model and input as `forward()`.
