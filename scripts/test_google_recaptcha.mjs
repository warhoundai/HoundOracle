// SPDX-License-Identifier: AGPL-3.0-only
// Live public Google demo smoke test; no credentials or response tokens persisted.
import puppeteer from '../integrations/recaptcha-local/browser-runtime.mjs';
import {spawn} from 'node:child_process';
import {writeFileSync,readFileSync} from 'node:fs';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const records=[];
const browser=await puppeteer.launch({executablePath:process.env.CHROMIUM_EXECUTABLE,
  headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--remote-debugging-address=127.0.0.1','--remote-debugging-port=9222'],
  env:{PATH:process.env.PATH,LANG:'en_US.UTF-8'},protocolTimeout:15000});
try {
 const page=await browser.newPage();
 const target='https://www.google.com/recaptcha/api2/demo';
 const response=await page.goto(target,{waitUntil:'domcontentloaded',timeout:30000});
 await new Promise(r=>setTimeout(r,3000));
 const before={http_status:response?.status(),frame_count:page.frames().length,
   widget_frames:page.frames().filter(f=>f.url().includes('/recaptcha/')).length};
 // Separate Python process must invoke the shipped process boundary exactly as a user would.
 const program=`import importlib.util,json\nspec=importlib.util.spec_from_file_location('recaptcha',${JSON.stringify(root+'/src/houndoracle/recaptcha.py')})\nm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\nprint(json.dumps(m.solve_google_recaptcha(${JSON.stringify(page.url())},helper_directory=${JSON.stringify(root+'/integrations/recaptcha-local')},model_directory=${JSON.stringify(process.env.RECAPTCHA_SOLVER_MODEL_DIR)},max_seconds=120,allow_network=True)))`;
 const run=await new Promise((resolve,reject)=>{
   const child=spawn('python3',['-c',program],{timeout:125000,env:{PATH:process.env.PATH}});
   let stdout='';
   child.stdout.on('data',chunk=>{stdout+=chunk;if(stdout.length>65536)child.kill('SIGKILL')});
   child.stderr.on('data',()=>{});
   child.on('error',reject);
   child.on('close',status=>resolve({stdout,status}));
 });
 let result;
 try {result=JSON.parse(run.stdout)}catch{result={status:'HARNESS_FAILED',return_code:run.status}}
 records.push({test:'public_google_recaptcha_demo',before,result});
 // Capture only bounded nonsecret structural state; never CAPTCHA images, cookies or tokens.
 records[0].after={origin:new URL(page.url()).origin,frame_count:page.frames().length};
 records[0].diagnostics=[];
 for (const frame of page.frames()) {
   if (!frame.url().includes('/recaptcha/')) continue;
   try {
     const state=await frame.evaluate(()=>({checkbox:document.querySelector('#recaptcha-anchor')?.getAttribute('aria-checked')??null,
       prompt:document.querySelector('.rc-imageselect-desc-wrapper')?.textContent?.slice(0,150)??null,
       imageTiles:document.querySelectorAll('.rc-imageselect-tile').length,
       blocked:document.body?.innerText?.includes('automated queries')??false}));
     records[0].diagnostics.push(state);
   } catch {records[0].diagnostics.push({state_unavailable:true})}
 }

} catch(error){records.push({test:'public_google_recaptcha_demo',failure_type:error.name});}
finally {await browser.close();}
let model_smoke=null;
try {model_smoke=JSON.parse(readFileSync(process.env.RECAPTCHA_MODEL_SMOKE,'utf8'))} catch {}
writeFileSync(process.env.RECAPTCHA_TEST_RESULT,JSON.stringify({records,model_smoke,
 scope:'One public Google demo smoke attempt, not a success-rate benchmark or Google Search acceptance test. No stealth flags, proxy rotation, token replay or cloud inference.',
 hosted_solver_calls:0},null,2));
