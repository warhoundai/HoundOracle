// SPDX-License-Identifier: AGPL-3.0-only
// CPU interface/profile smoke on zeros. Not a classification-quality benchmark.
import * as ort from 'onnxruntime-node';
import {readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
const dir=process.env.RECAPTCHA_SOLVER_MODEL_DIR;
const manifest=JSON.parse(await readFile(new URL('./model-manifest.json',import.meta.url),'utf8'));
const records=[];
for(const asset of manifest.models) {
 const start=performance.now();
 const session=await ort.InferenceSession.create(join(dir,asset.fileName),{
   executionProviders:['cpu'],intraOpNumThreads:2,interOpNumThreads:1,executionMode:'sequential'});
 const loaded=performance.now();
 const output=await session.run({[asset.inputName]:new ort.Tensor('float32',new Float32Array(3*640*640),[1,3,640,640])});
 records.push({id:asset.id,input_shape:[1,3,640,640],output_shape:output[asset.outputName].dims,
   load_seconds:(loaded-start)/1000,inference_seconds:(performance.now()-loaded)/1000});
 await session.release();
}
await writeFile(process.env.RECAPTCHA_MODEL_SMOKE,JSON.stringify({records,scope:'Single zero-tensor CPU interface/profile smoke, two intra-op threads. Not accuracy or Google success.'},null,2));
