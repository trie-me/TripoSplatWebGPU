import json, urllib.request, websocket
page = next(x for x in json.load(urllib.request.urlopen('http://127.0.0.1:9222/json')) if x['type'] == 'page')
ws = websocket.create_connection(page['webSocketDebuggerUrl'], timeout=900)
expression = r'''(async () => {
  const statuses = [];
  const { O: OrtWorkerClient } = await import('/assets/OrtWorkerClient-qIBpPhO9.js');
  const client = new OrtWorkerClient({ onStatus: value => statuses.push(value.stage) });
  const root = '/fixtures/generated/dit-context0-inv08';
  const tensor = async (name, dims) => ({type:'float32', data:new Float32Array(await (await fetch(`${root}/${name}.f32`)).arrayBuffer()), dims});
  const compare = (a,b) => { let max=0,sum=0,sq=0,dot=0,an=0,bn=0; for(let i=0;i<a.length;i++){const d=a[i]-b[i],x=Math.abs(d);max=Math.max(max,x);sum+=x;sq+=d*d;dot+=a[i]*b[i];an+=a[i]*a[i];bn+=b[i]*b[i]} return {max,mean:sum/a.length,rmse:Math.sqrt(sq/a.length),cosine:dot/Math.sqrt(an*bn)} };
  try {
    const externalData=[{path:'dit_step_webgpu_fp32.onnx.data',url:'/models/triposplat/dit_step_webgpu_fp32.onnx.data'}];
    const loaded=await client.loadContext0Split({sessionId:'smoke',graphs:{pre:{graphUrl:'/models/triposplat/dit_step_webgpu_fp32_context0_wgsl.pre.onnx',externalData},post:{graphUrl:'/models/triposplat/dit_step_webgpu_fp32_context0_wgsl.post.onnx',externalData}},options:{allowWasmFallback:false,graphOptimizationLevel:'disabled'}});
    const response=await client.runContext0Split({sessionId:'smoke',inputs:{latent:await tensor('latent',[1,8192,16]),camera:await tensor('camera',[1,1,5]),t:await tensor('t',[1]),feature1:await tensor('feature1',[1,4101,1280]),feature2:await tensor('feature2',[1,4101,128])},outputs:['pred_latent','pred_camera']});
    const rl=(await tensor('pred_latent',[1,8192,16])).data, rc=(await tensor('pred_camera',[1,1,5])).data;
    return {loaded,timings:response.timings,latent:compare(response.outputs.pred_latent.data,rl),camera:compare(response.outputs.pred_camera.data,rc),statuses};
  } catch(error) { return {error:{name:error.name,message:error.message},statuses}; }
  finally { await client.dispose().catch(()=>{}); }
})()'''
ws.send(json.dumps({'id':1,'method':'Runtime.evaluate','params':{'expression':expression,'awaitPromise':True,'returnByValue':True}}))
while True:
    message=json.loads(ws.recv())
    if message.get('id')==1:
        print(json.dumps(message,indent=2)); break
ws.close()
