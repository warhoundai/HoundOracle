// SPDX-License-Identifier: AGPL-3.0-only
// Separate opt-in helper. Never forwards cookies/tokens to Oracle's evidence database.
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {readFile,stat} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
let stage='VALIDATION';

export function normalizeResult(result) {
  const verified=result.status==='success' && ['widget_and_token_confirmed','navigation_confirmed'].includes(result.verification);
  return {status:verified?'WIDGET_SOLVED':'UNVERIFIED',widget_solved:verified,
    server_accepted:null, verification:['widget_and_token_confirmed','navigation_confirmed','widget_observed','token_observed'].includes(result.verification)?result.verification:'unverified',
    captcha_type:['dynamic_3x3','selection_3x3','square_4x4','no_challenge'].includes(result.captchaType)?result.captchaType:'unknown',
    image_challenge_solved:verified && ['dynamic_3x3','selection_3x3','square_4x4'].includes(result.captchaType),
    attempts:Number.isInteger(result.attempts)?result.attempts:null,
    local_inference:true,hosted_solver_calls:0};
}

export async function validateModels(directory) {
  const manifest=JSON.parse(await readFile(new URL('./model-manifest.json',import.meta.url),'utf8'));
  for (const model of manifest.models) {
    const file=join(directory,model.fileName);
    if ((await stat(file)).size!==model.size) throw new Error('MODEL_SIZE_MISMATCH');
    const digest=createHash('sha256');
    for await (const chunk of createReadStream(file)) digest.update(chunk);
    if (digest.digest('hex')!==model.sha256) throw new Error('MODEL_HASH_MISMATCH');
  }
}

export function runtimeRequestKind(value,port) {
  const url=new URL(value);
  if(url.username||url.password||url.hash) return null;
  if(url.protocol==='http:' && url.hostname==='127.0.0.1' && url.port===String(port)
     && ['/json/version','/json/list'].includes(url.pathname)) return 'cdp';
  if(url.protocol==='https:' && ['www.google.com','www.recaptcha.net'].includes(url.hostname)
     && !url.port && url.pathname==='/recaptcha/api2/payload') return 'challenge_image';
  return null;
}

export async function solve(request) {
  const target=new URL(request.target_url);
  if (target.protocol!=='https:' || target.origin!=='https://www.google.com' || target.username || target.password || target.hash)
    throw new Error('UNAPPROVED_GOOGLE_TARGET');
  if (!Number.isInteger(request.port) || request.port<1024 || request.port>65535) throw new Error('INVALID_LOCAL_CDP_PORT');
  const directory=resolve(request.model_dir);
  stage="MODEL_VALIDATION";
  await validateModels(directory);
  process.env.RECAPTCHA_SOLVER_MODEL_DIR=directory;
  // Runtime model downloads are disabled; setup must acquire and verify them first.
  const originalFetch=globalThis.fetch;
  let imageRequests=0,imageBytes=0;
  globalThis.fetch=async (input,options)=>{
    const value=typeof input==='string'?input:input.url??input.toString();
    const kind=runtimeRequestKind(value,request.port);
    if (!kind || (options?.method && options.method!=='GET')) throw new Error('RUNTIME_FETCH_BLOCKED');
    if (kind==='challenge_image' && ++imageRequests>32) throw new Error('IMAGE_REQUEST_BUDGET');
    const response=await originalFetch(input,{...options,redirect:'error'});
    if(kind==='cdp') return response;
    if(!response.ok || !response.headers.get('content-type')?.startsWith('image/')) {
      await response.body?.cancel();throw new Error('CHALLENGE_IMAGE_UNAVAILABLE');
    }
    const reader=response.body.getReader(),chunks=[];let bytes=0;
    try {
      for(;;) {
        const {done,value:chunk}=await reader.read();if(done)break;
        bytes+=chunk.byteLength;imageBytes+=chunk.byteLength;
        if(bytes>2000000 || imageBytes>16000000) {
          await reader.cancel();throw new Error('IMAGE_BYTE_BUDGET');
        }
        chunks.push(chunk);
      }
    } finally {reader.releaseLock()}
    return new Response(Buffer.concat(chunks),{status:response.status,headers:response.headers});
  };
  stage='RUNTIME_IMPORT';
  const {default:puppeteer}=await import('puppeteer-core');
  const {solveReCaptcha}=await import('@conghuy113/recaptcha-solver');
  stage="BROWSER_CONNECT";
  const browser=await puppeteer.connect({browserURL:`http://127.0.0.1:${request.port}`,protocolTimeout:15000});
  try {
    stage="EXACT_TAB_SELECTION";
    const pages=await browser.pages();
    const page=pages.find(p=>p.url()===request.target_url);
    if (!page) throw new Error('EXACT_TARGET_TAB_REQUIRED');
    stage="WIDGET_INTERACTION";
    const result=await solveReCaptcha({page,targetUrl:request.target_url,clickCheckbox:true});
    return {...normalizeResult(result),challenge_image_requests:imageRequests,challenge_image_bytes:imageBytes};
  } finally {
    await browser.disconnect(); // Existing browser/session remain owned by the caller.
    globalThis.fetch=originalFetch;
  }
}

if (process.argv[1] && resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  let input='';
  for await (const chunk of process.stdin) {
    input+=chunk;
    if (input.length>12000) throw new Error('REQUEST_TOO_LARGE');
  }
  let result;
  try {result=await solve(JSON.parse(input));}
  catch(error) {
    const message=String(error?.message??'').toLowerCase();
    const reason=/timeout|timed out/.test(message)?'TIMEOUT':
      /confidence/.test(message)?'LOW_CONFIDENCE':
      /not completed after/.test(message)?'ATTEMPTS_EXHAUSTED':
      /runtime_fetch_blocked|download.*disabled/.test(message)?'RUNTIME_FETCH_BLOCKED':
      /visible|checkbox|frame/.test(message)?'WIDGET_STATE_UNAVAILABLE':
      /model|onnx/.test(message)?'MODEL_RUNTIME_FAILED':'UNCLASSIFIED_FAILURE';
    const errorType=['Error','TypeError','CdpConnectionError','CdpProtocolError','TimeoutError'].includes(error?.name)?error.name:'Error';
    result={status:'SOLVER_FAILED',widget_solved:false,server_accepted:null,local_inference:true,
            failure_stage:stage,failure_reason:reason,failure_type:errorType};
  }
  process.stdout.write(JSON.stringify(result)+'\n');
}
