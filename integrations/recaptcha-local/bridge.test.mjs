// SPDX-License-Identifier: AGPL-3.0-only
import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeResult,solve,validateModels,runtimeRequestKind} from './bridge.mjs';
test('token alone is not accepted and secrets never leave the helper',()=>{
 const result=normalizeResult({status:'success',verification:'token_observed',token:'secret',cookies:[{value:'secret'}]});
 assert.equal(result.widget_solved,false);assert.equal(JSON.stringify(result).includes('secret'),false);
});
test('widget evidence does not claim application server acceptance',()=>{
 const result=normalizeResult({status:'success',verification:'widget_and_token_confirmed',captchaType:'selection_3x3',attempts:1});
 assert.equal(result.widget_solved,true);assert.equal(result.image_challenge_solved,true);assert.equal(result.server_accepted,null);
});
test('private/unapproved target rejected before contacting Chrome',async()=>{
 await assert.rejects(solve({target_url:'http://169.254.169.254/',port:9222,model_dir:'/tmp'}),/UNAPPROVED/);
 await assert.rejects(solve({target_url:'https://www.google.com.evil.example/',port:9222,model_dir:'/tmp'}),/UNAPPROVED/);
});
test('missing models fail before runtime downloads',async()=>{
 await assert.rejects(validateModels('/tmp/houndoracle-nonexistent-model-fixture'));
});

test('Google payload permitted without allowing model downloads or arbitrary fetches',()=>{
 assert.equal(runtimeRequestKind('https://www.google.com/recaptcha/api2/payload?p=example',9222),'challenge_image');
 for(const url of ['https://github.com/model.onnx','http://169.254.169.254/latest',
   'https://www.google.com.evil.example/recaptcha/api2/payload','https://www.google.com/search?q=x'])
  assert.equal(runtimeRequestKind(url,9222),null);
 assert.equal(runtimeRequestKind('http://127.0.0.1:9222/json/version',9222),'cdp');
 assert.equal(runtimeRequestKind('http://127.0.0.1:9223/json/version',9222),null);
});

test('CLI entry point works in a directory containing spaces',async()=>{
 const {mkdtemp,copyFile,rm}=await import('node:fs/promises');
 const {tmpdir}=await import('node:os');
 const {join}=await import('node:path');
 const {spawnSync}=await import('node:child_process');
 const dir=await mkdtemp(join(tmpdir(),'oracle recaptcha '));
 try {
  const script=join(dir,'bridge.mjs');await copyFile(new URL('./bridge.mjs',import.meta.url),script);
  const child=spawnSync(process.execPath,[script],{encoding:'utf8',timeout:2000,
   input:JSON.stringify({target_url:'http://169.254.169.254/',port:9222,model_dir:dir})});
  assert.equal(child.status,0);assert.equal(JSON.parse(child.stdout).status,'SOLVER_FAILED');
 }finally{await rm(dir,{recursive:true,force:true})}
});
