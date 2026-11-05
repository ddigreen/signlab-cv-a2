'use strict';
(() => {
  const CLASSES = ['give_way', 'main_road', 'parking', 'pedestrian_crossing'];
  const NAMES = ['Give way', 'Main road', 'Parking', 'Pedestrian crossing'];
  const COLORS = ['#f7bf5b', '#9bd2ff', '#7de0b7', '#e7a2ff'];
  const SIZE = 320, IOU_THRESHOLD = 0.45;
  const DETECTOR_REVISION = 't4-model-choice-v7';
  const DEFAULT_DETECTOR = 'expanded_synthetic';
  const DETECTORS = {
    original: {file:'models/detector.onnx', title:'Original · 353 real', header:'YOLO11n · ORIGINAL DATA'},
    expanded_synthetic: {file:'models/detector-expanded-synthetic.onnx', title:'Expanded + synthetic · 453 real + 128 synthetic', header:'YOLO11n · EXPANDED + SYNTHETIC'}
  };
  const safeArm = value => Object.hasOwn(DETECTORS,value) ? value : DEFAULT_DETECTOR;
  const initialArm = safeArm(new URL(location.href).searchParams.get('detector'));
  const sessionWork = new WeakMap(), warmedSessions = new WeakSet(), warmingSessions = new WeakMap(), retiredSessions = new WeakSet();
  const ARM_NAMES = {original:'Original real data', expanded:'+100 real frames', expanded_synthetic:'+100 real +128 synthetic'};
  const SOURCE_LABELS = {camera: 'CAMERA', replay: 'SYNTHETIC EXAMPLE REPLAY', video: 'LOCAL VIDEO', image: 'LOCAL IMAGE', sample: 'SYNTHETIC SAMPLE'};
  const $ = id => document.getElementById(id);
  const target = $('detector-canvas'), video = $('detector-video');
  const state = {arm:initialArm, generation:0, zoom:1, autoZoomApplied:false, busyOwner:null, metrics:null, retirement:Promise.resolve(), modelSHA256:'', session: null, loading: null, running: false, starting: false, stream: null, videoURL: null, image: null, sourceName: '', sourceKind: '', token: 0, timing: [], elapsed: [], processingStarted: null, detections: [], frame: null, warmedUp: false};
  const conditionNotes = {clean:'Original input. No condition simulation applied.', dark:'Deterministic simulation: multiply source brightness by 0.35.', blur:'Deterministic simulation: 2 px browser blur on the source frame.', low_resolution:'Deterministic simulation: reduce the source to 40 × 40 before resizing.', occlusion:'Deterministic simulation: a central gray patch covers 20% of the source.'};
  const environment = $('device-name')?.textContent || navigator.platform || 'Browser';
  $('detector-device').textContent = environment;
  function status(message) { $('detector-status').textContent = message; }
  function resetAutomaticZoom() {
    if(!state.autoZoomApplied)return;
    state.zoom=1;state.autoZoomApplied=false;
    $('detector-zoom').value='1';$('detector-zoom-value').textContent='1.0×';
  }
  function resultFeedback(kind,title,detail) {
    $('detector-result').setAttribute('data-state',kind);
    $('detector-result-title').textContent=title;
    status(detail);
  }
  function error(message = '') { $('detector-error').textContent = message; $('detector-error').hidden = !message; if(message)resultFeedback('error','Recognition failed',message); }
  function escapeText(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function makeCanvas(width, height = width) { const c = document.createElement('canvas'); c.width = width; c.height = height; return c; }
  function resetTiming() { state.timing = []; state.elapsed = []; state.processingStarted = null; $('detector-latency').innerHTML = '—<small>ms</small>'; $('detector-fps').innerHTML = '—<small>FPS</small>'; $('detector-run-count').textContent = 'No timed frames'; }
  function updateTiming(inferenceMs, totalMs) {
    state.timing.push(inferenceMs); state.elapsed.push(totalMs);
    const mean = state.timing.reduce((a,b)=>a+b,0)/state.timing.length;
    const fps = state.processingStarted === null ? 1000/totalMs : state.timing.length*1000/(performance.now()-state.processingStarted);
    $('detector-latency').innerHTML = `${mean.toFixed(1)}<small>ms</small>`;
    $('detector-fps').innerHTML = `${fps.toFixed(1)}<small>FPS</small>`;
    $('detector-run-count').textContent = `${state.timing.length} timed frame${state.timing.length === 1 ? '' : 's'} · warm-up excluded`;
  }
  function stop(message = 'Detection stopped.') {
    hideBusy();
    state.running = false; state.starting = false; state.token++;
    if (state.stream) state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
    video.pause(); video.srcObject = null;
    if (state.videoURL) { URL.revokeObjectURL(state.videoURL); state.videoURL = null; } video.removeAttribute('src'); video.load();
    $('detector-live-badge').classList.add('hidden'); $('detector-stop').disabled = true; $('detector-start').disabled = false;
    if (message) resultFeedback('idle','Detection stopped',message);
  }
  function showBusy(headline,detail,token=state.token,generation=state.generation) {
    if(!activeOperation(token,generation))return;
    state.busyOwner={token,generation};
    resultFeedback('processing',headline,detail||'Please wait…');
    $('detector-busy-title').textContent=headline;$('detector-busy-detail').textContent=detail||'';
    $('detector-busy').hidden=false;$('detector-busy').classList.remove('hidden');$('detector-view').setAttribute('aria-busy','true');
  }
  function hideBusy(token,generation) {
    const owner=state.busyOwner;
    if(token!==undefined&&(!owner||owner.token!==token||owner.generation!==generation))return;
    state.busyOwner=null;$('detector-busy').hidden=true;$('detector-busy').classList.add('hidden');$('detector-view').setAttribute('aria-busy','false');
  }
  function currentModel(generation,arm) { return generation===state.generation&&arm===state.arm; }
  function activeOperation(token,generation) { return token===state.token&&generation===state.generation; }
  function runSession(session,feeds) {
    if(retiredSessions.has(session))return Promise.reject(new Error('Detector changed before this run began.'));
    const previous=sessionWork.get(session)||Promise.resolve();
    const pending=previous.catch(()=>{}).then(()=>session.run(feeds));
    sessionWork.set(session,pending.then(()=>{},()=>{}));
    return pending;
  }
  function retireSession(session) {
    if(!session||retiredSessions.has(session))return Promise.resolve();
    retiredSessions.add(session);
    return (sessionWork.get(session)||Promise.resolve()).catch(()=>{}).then(()=>session.release()).catch(e=>console.warn('Detector session release failed',e));
  }
  function clearDetectionOutput() {
    state.detections=[];state.frame=null;state.modelSHA256='';
    target.getContext('2d').clearRect(0,0,target.width,target.height);target.classList.remove('has-frame');
    $('detector-placeholder').classList.remove('hidden');$('detector-count').textContent='0';
    $('detector-list').innerHTML='<p class="tiny">Select an input to run the active detector.</p>';
    $('detector-source-badge').textContent='NO INPUT';
    $('detector-model-version').hidden=true;$('detector-model-version').textContent='';$('detector-model-version').removeAttribute('data-sha256');
    resetTiming();
    resultFeedback('idle','Ready to recognize','Start the camera, upload a photo or video, or choose an example.');
  }
  function updateActiveModel() {
    const model=DETECTORS[state.arm];
    $('detector-model-select').value=state.arm;
    $('detector-active-heading').textContent=model.header;
    const url=new URL(location.href);url.searchParams.set('detector',state.arm);url.hash='';
    $('detector-model-link').href=url.href;
    const data=state.metrics,rows=data?.comparison||[],row=rows.find(r=>r.arm===state.arm&&r.condition==='clean');
    const validationWinner=data?.selection?.selected_arm||data?.selected_arm||'original';
    const val=data?.selection?.validation_metrics?.find(r=>r.arm===state.arm);
    $('detector-selected-arm').hidden=false;
    $('detector-selected-arm').textContent=`Active: ${model.title}. ${state.arm===DEFAULT_DETECTOR?'App default. ':''}${state.arm===validationWinner?'Validation winner':'Original remains the validation winner'}${val?` · validation mAP@0.50:0.95 ${percent(val.mAP50_95)}`:''}.`;
    $('detector-map50').textContent=percent(row?.map50);$('detector-map5095').textContent=percent(row?.map50_95);
    $('detector-metrics-note').textContent=row?`${model.title} · clean test on ${row.n??row.images??96} original images. Browser simulations do not change these recorded scores.`:'Recorded scores for the active model are loading or unavailable.';
  }
  async function switchDetector(value,updateURL=true) {
    const arm=safeArm(value);if(arm===state.arm){updateActiveModel();return;}
    const image=state.image,name=state.sourceName,kind=state.sourceKind;
    stop('');state.generation++;state.arm=arm;
    const oldLoading=state.loading;state.loading=null;oldLoading?.controller.abort();
    const oldSession=state.session;state.session=null;state.warmedUp=false;
    state.retirement=retireSession(oldSession);
    clearDetectionOutput();error();updateActiveModel();
    if(updateURL){const url=new URL(location.href);url.searchParams.set('detector',arm);history.replaceState(null,'',url);}
    status(`${DETECTORS[arm].title} selected. ${image?'Recognizing the same image…':'Start the camera or choose an input to run this model.'}`);
    if(image)await detectImage(image,name,kind);
  }
  async function getSession() {
    const generation=state.generation,arm=state.arm,model=DETECTORS[arm],token=state.token;
    if(state.session)return state.session;
    if(state.loading&&state.loading.generation===generation)return state.loading.promise;
    if(!window.ort)throw new Error('The local ONNX Runtime library is missing. Restore web/vendor/ from the complete package.');
    ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('vendor/',location.href).href;
    const loading={generation,arm,controller:new AbortController(),promise:null};
    loading.promise=(async()=>{
      if(activeOperation(token,generation)){status(`Downloading ${model.title} · about 10 MB. First load depends on your connection…`);showBusy('Loading model · about 10 MB',`${model.title}. First load depends on your connection.`,token,generation);}
      const response=await fetch(`${model.file}?v=${DETECTOR_REVISION}`,{cache:'no-cache',signal:loading.controller.signal});
      if(!currentModel(generation,arm))return null;
      if(!response.ok)throw new Error(`The selected detector is unavailable (${response.status}). Please retry or choose the original model.`);
      const bytes=await response.arrayBuffer();if(!currentModel(generation,arm))return null;
      if(bytes.byteLength<1000)throw new Error('The selected detector file is incomplete.');
      let sha='';
      if(crypto.subtle){const digest=await crypto.subtle.digest('SHA-256',bytes);sha=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');}
      if(!currentModel(generation,arm))return null;
      if(activeOperation(token,generation)){status(`Preparing ${model.title} in this browser…`);showBusy('Preparing model…',model.title,token,generation);}
      const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
      if(!currentModel(generation,arm)){await retireSession(session);return null;}
      state.session=session;state.modelSHA256=sha;
      if(sha){$('detector-model-version').hidden=false;$('detector-model-version').textContent=`Model revision ${sha.slice(0,12)}`;$('detector-model-version').setAttribute('data-sha256',sha);}
      return session;
    })();
    state.loading=loading;
    try{return await loading.promise;}finally{if(state.loading===loading)state.loading=null;}
  }
  async function warmup(session) {
    if(warmedSessions.has(session))return;
    if(warmingSessions.has(session))return warmingSessions.get(session);
    if(session===state.session)status('Warming up the active detector…');
    const pending=runSession(session,{[session.inputNames[0]]:new ort.Tensor('float32',new Float32Array(3*SIZE*SIZE),[1,3,SIZE,SIZE])}).then(()=>{
      warmedSessions.add(session);if(session===state.session)state.warmedUp=true;
    });
    warmingSessions.set(session,pending);
    try{await pending;}finally{warmingSessions.delete(session);}
  }
  function sourceFrame(source,requestedZoom=state.zoom) {
    const naturalW=source.videoWidth||source.naturalWidth||source.width;
    const naturalH=source.videoHeight||source.naturalHeight||source.height;
    if(!naturalW||!naturalH)throw new Error('No source frame is ready.');
    const zoom=Math.max(1,Math.min(4,Number(requestedZoom)||1));
    const cropW=naturalW/zoom,cropH=naturalH/zoom,sx=(naturalW-cropW)/2,sy=(naturalH-cropH)/2;
    const resize=Math.min(1,1600/Math.max(cropW,cropH));
    const width=Math.max(1,Math.round(cropW*resize)),height=Math.max(1,Math.round(cropH*resize));
    const frame=makeCanvas(width,height),ctx=frame.getContext('2d');ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';
    const condition=$('detector-condition').value;
    if(condition==='low_resolution'){
      const low=makeCanvas(40);low.getContext('2d').drawImage(source,sx,sy,cropW,cropH,0,0,40,40);ctx.drawImage(low,0,0,width,height);
    }else{
      ctx.filter=condition==='dark'?'brightness(0.35)':condition==='blur'?'blur(2px)':'none';
      ctx.drawImage(source,sx,sy,cropW,cropH,0,0,width,height);ctx.filter='none';
    }
    if(condition==='occlusion'){const side=Math.sqrt(width*height*0.2);ctx.fillStyle='#808080';ctx.fillRect((width-side)/2,(height-side)/2,side,side);}
    return frame;
  }
  function prepare(frame) {
    const width=frame.width,height=frame.height,gain=Math.min(SIZE/width,SIZE/height),newW=Math.round(width*gain),newH=Math.round(height*gain);
    // Ultralytics LetterBox: centered padding, round(dw/2-0.1), RGB114.
    const left=Math.round((SIZE-newW)/2-0.1),top=Math.round((SIZE-newH)/2-0.1);
    const box=makeCanvas(SIZE),ctx=box.getContext('2d',{willReadFrequently:true});ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,SIZE,SIZE);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(frame,left,top,newW,newH);
    const rgba=ctx.getImageData(0,0,SIZE,SIZE).data,data=new Float32Array(3*SIZE*SIZE),plane=SIZE*SIZE;
    for(let p=0;p<plane;p++){data[p]=rgba[p*4]/255;data[plane+p]=rgba[p*4+1]/255;data[2*plane+p]=rgba[p*4+2]/255;}
    return {tensor:data,gain,left,top,width,height};
  }
  function intersectionOverUnion(a,b) {
    const intersection=Math.max(0,Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1))*Math.max(0,Math.min(a.y2,b.y2)-Math.max(a.y1,b.y1));
    const union=Math.max(0,a.x2-a.x1)*Math.max(0,a.y2-a.y1)+Math.max(0,b.x2-b.x1)*Math.max(0,b.y2-b.y1)-intersection;
    return union>0?intersection/union:0;
  }
  function nms(candidates,threshold=IOU_THRESHOLD,maxDetections=50) {
    const ordered=candidates.sort((a,b)=>b.score-a.score),kept=[];
    for(const candidate of ordered){if(kept.every(previous=>candidate.classId!==previous.classId||intersectionOverUnion(candidate,previous)<=threshold))kept.push(candidate);if(kept.length>=maxDetections)break;}
    return kept;
  }
  function decode(output,meta,confidence=Number($('detector-confidence').value)) {
    const dims=output.dims,data=output.data,candidates=[];
    if(dims.length!==3||dims[0]!==1)throw new Error(`Unsupported detector output shape [${dims.join(',')}]. Expected [1,8,N].`);
    const channelsFirst=dims[1]===8,channelsLast=dims[2]===8,processed=dims[1]!==8&&dims[2]===6;
    if(!channelsFirst&&!channelsLast&&!processed)throw new Error(`Detector output has ${dims.join(' × ')} values; expected 4 box coordinates and 4 class scores.`);
    const count=channelsFirst?dims[2]:dims[1];
    const value=(i,c)=>channelsFirst?Number(data[c*count+i]):Number(data[i*dims[2]+c]);
    for(let i=0;i<count;i++){
      let classId=0,score=-Infinity,x1,y1,x2,y2;
      if(processed){score=value(i,4);classId=Math.round(value(i,5));x1=value(i,0);y1=value(i,1);x2=value(i,2);y2=value(i,3);}
      else {for(let c=0;c<4;c++){const s=value(i,4+c);if(s>score){score=s;classId=c;}}const cx=value(i,0),cy=value(i,1),w=value(i,2),h=value(i,3);x1=cx-w/2;y1=cy-h/2;x2=cx+w/2;y2=cy+h/2;}
      if(score<confidence||classId<0||classId>=4||![x1,y1,x2,y2,score].every(Number.isFinite))continue;
      x1=Math.max(0,Math.min(meta.width,(x1-meta.left)/meta.gain));x2=Math.max(0,Math.min(meta.width,(x2-meta.left)/meta.gain));y1=Math.max(0,Math.min(meta.height,(y1-meta.top)/meta.gain));y2=Math.max(0,Math.min(meta.height,(y2-meta.top)/meta.gain));
      if(x2-x1>1&&y2-y1>1)candidates.push({x1,y1,x2,y2,score,classId,label:CLASSES[classId]});
    }
    return nms(candidates);
  }
  function draw(frame,detections) {
    target.width=frame.width;target.height=frame.height;target.classList.add('has-frame');
    const ctx=target.getContext('2d');ctx.drawImage(frame,0,0);const scale=Math.max(frame.width,frame.height)/640;
    for(const box of detections){
      const color=COLORS[box.classId],font=Math.max(13,Math.round(16*scale));ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,3*scale);ctx.strokeRect(box.x1,box.y1,box.x2-box.x1,box.y2-box.y1);
      const label=`${NAMES[box.classId]} ${(box.score*100).toFixed(1)}%`;ctx.font=`600 ${font}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;const pad=5*scale,labelW=ctx.measureText(label).width+pad*2,labelH=font+pad*2;
      const x=Math.max(0,Math.min(frame.width-labelW,box.x1)),y=box.y1>=labelH?box.y1-labelH:Math.min(frame.height-labelH,box.y1+2);
      ctx.fillStyle=color;ctx.fillRect(x,y,labelW,labelH);ctx.fillStyle='#0b1420';ctx.textBaseline='top';ctx.fillText(label,x+pad,y+pad);
    }
    $('detector-placeholder').classList.add('hidden');$('detector-count').textContent=String(detections.length);
    $('detector-list').innerHTML=detections.length?detections.map(box=>`<div class="detected-row"><span class="detected-label"><i class="detected-dot" style="background:${COLORS[box.classId]}"></i>${NAMES[box.classId]}</span><strong>${(box.score*100).toFixed(1)}%</strong></div>`).join(''):'<p class="tiny">No trained sign class exceeds the selected confidence threshold in this frame.</p>';
    const simulated = $('detector-condition').value !== 'clean';
    $('detector-source-badge').textContent = `${simulated ? 'SIMULATED · ' : ''}${SOURCE_LABELS[state.sourceKind] || 'IMAGE'}`;
    state.frame=frame;state.detections=detections;
  }
  async function predictFrame(source,session,token,generation,zoom) {
    if(!activeOperation(token,generation))return null;
    const frameStart=performance.now();if(state.processingStarted===null)state.processingStarted=frameStart;const frame=sourceFrame(source,zoom),meta=prepare(frame);
    const start=performance.now(),outputs=await runSession(session,{[session.inputNames[0]]:new ort.Tensor('float32',meta.tensor,[1,3,SIZE,SIZE])}),inferenceMs=performance.now()-start;
    if(!activeOperation(token,generation))return null;
    const output=outputs.output0||outputs[session.outputNames[0]],detections=decode(output,meta);
    return {frame,detections,zoom,inferenceMs,totalMs:performance.now()-frameStart};
  }
  function recordPrediction(prediction) { updateTiming(prediction.inferenceMs,prediction.totalMs); }
  function predictionFeedback(detail) {
    const boxes=state.detections,count=boxes.length;
    const title=count===1?`${NAMES[boxes[0].classId]} · ${(boxes[0].score*100).toFixed(1)}%`:count?`${count} signs found`:'No signs found';
    const labels=count>1?boxes.map(box=>`${NAMES[box.classId]} ${(box.score*100).toFixed(1)}%`).join(' · ')+'. ':'';
    resultFeedback(count?'success':'empty',title,labels+detail);
  }
  async function infer(source,session,token,generation=state.generation) {
    const zoom=state.zoom,prediction=await predictFrame(source,session,token,generation,zoom);
    if(!prediction)return false;
    if(zoom!==state.zoom)return true; // The next live frame uses the new crop; never display the old crop as new zoom.
    draw(prediction.frame,prediction.detections);recordPrediction(prediction);
    return true;
  }
  async function detectImage(image,name,sourceKind='image') {
    stop('');state.image=image;state.sourceName=name;state.sourceKind=sourceKind;resetTiming();error();
    const token=state.token,generation=state.generation,initialZoom=state.zoom,initialAutoZoom=state.autoZoomApplied;
    const autoZoom=sourceKind==='image'&&initialZoom===1&&$('detector-auto-zoom').checked;
    showBusy(state.session?'Recognizing image…':'Loading model · about 10 MB',DETECTORS[state.arm].title,token,generation);
    try{
      const session=await getSession();if(!session||!activeOperation(token,generation))return;
      showBusy('Preparing model…',DETECTORS[state.arm].title,token,generation);
      await warmup(session);if(!activeOperation(token,generation))return;
      showBusy('Recognizing image…',`${initialZoom.toFixed(1)}× centre crop`,token,generation);
      const initial=await predictFrame(image,session,token,generation,initialZoom);
      if(!initial||state.zoom!==initialZoom){hideBusy(token,generation);return;}
      recordPrediction(initial);let chosen=initial;const checked=[initialZoom];
      if(autoZoom&&!initial.detections.length){
        for(const zoom of [2,4]){
          showBusy(`Checking ${zoom}× zoom…`,'No signs found at 1×. Checking the centre of the original photo.',token,generation);
          const candidate=await predictFrame(image,session,token,generation,zoom);
          if(!candidate||state.zoom!==initialZoom){hideBusy(token,generation);return;}
          recordPrediction(candidate);checked.push(zoom);
          if(candidate.detections.length){chosen=candidate;break;}
        }
      }
      if(!activeOperation(token,generation))return;
      state.zoom=chosen.zoom;state.autoZoomApplied=chosen.zoom>1&&(checked.length>1||initialAutoZoom);
      $('detector-zoom').value=String(chosen.zoom);$('detector-zoom-value').textContent=`${chosen.zoom.toFixed(1)}×`;
      draw(chosen.frame,chosen.detections);hideBusy(token,generation);
      const zoomNote=checked.length>1?` Auto zoom checked ${checked.map(zoom=>`${zoom}×`).join(', ')}. ${chosen.zoom>1?`Showing the ${chosen.zoom}× centre crop.`:'Showing the full photo.'}`:` Zoom ${chosen.zoom.toFixed(1)}×.`;
      const emptyNote=chosen.detections.length?'':` No sign exceeded the ${Math.round(Number($('detector-confidence').value)*100)}% confidence threshold. Try a closer or sharper view.`;
      predictionFeedback(`Detection complete · ${state.detections.length} sign${state.detections.length===1?'':'s'} found.${zoomNote}${emptyNote} File: ${name}`);
    }catch(e){if(!activeOperation(token,generation))return;hideBusy(token,generation);console.error(e);error(e.message||String(e));status('Detection failed. No substitute result is shown.');}
  }
  async function loop(session,token,generation) {
    try{
      while(state.running&&activeOperation(token,generation)){
        if(video.ended){stop(state.sourceKind==='replay'?'Synthetic example replay finished.':'Uploaded video finished. Camera access is released.');break;}
        if(video.readyState>=2&&!video.paused){
          if(!await infer(video,session,token,generation))break;
          predictionFeedback(`${state.sourceKind==='camera'?'Live camera':state.sourceKind==='replay'?'Synthetic example replay':'Uploaded video'} · ${state.detections.length} sign${state.detections.length===1?'':'s'} in the latest frame. Zoom ${state.zoom.toFixed(1)}×.`);
        }
        await new Promise(resolve=>requestAnimationFrame(resolve));
      }
    }catch(e){if(!activeOperation(token,generation))return;console.error(e);stop('Detection stopped after an inference error.');error(e.message||String(e));}
  }
  async function startCamera() {
    resetAutomaticZoom();
    stop('');error();state.image=null;state.sourceKind='camera';state.sourceName='Live camera';resetTiming();
    const token=state.token,generation=state.generation;
    $('detector-start').disabled=true;$('detector-stop').disabled=false;state.starting=true;
    showBusy(state.session?'Opening camera…':'Loading model · about 10 MB',DETECTORS[state.arm].title,token,generation);
    try{
      const session=await getSession();if(!session||!activeOperation(token,generation))return;
      await warmup(session);if(!activeOperation(token,generation))return;
      if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera access requires HTTPS or localhost and a supported browser. Upload a photograph or video to continue.');
      status('Allow camera access in your browser to begin.');showBusy('Opening camera…','Allow camera access in your browser to begin.',token,generation);
      const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false});
      if(!activeOperation(token,generation)){stream.getTracks().forEach(track=>track.stop());return;}
      state.stream=stream;video.srcObject=stream;await video.play();
      if(!activeOperation(token,generation)){stream.getTracks().forEach(track=>track.stop());return;}
      hideBusy(token,generation);state.starting=false;state.running=true;$('detector-stop').disabled=false;$('detector-live-badge').classList.remove('hidden');
      await loop(session,token,generation);
    }catch(e){if(!activeOperation(token,generation))return;stop('Camera detection could not start.');error(e.message||String(e));}
  }
  async function startReplay() {
    resetAutomaticZoom();
    stop('');error();state.image=null;state.sourceKind='replay';state.sourceName='Synthetic example replay';resetTiming();
    const token=state.token,generation=state.generation;state.starting=true;$('detector-stop').disabled=false;$('detector-start').disabled=true;
    showBusy(state.session?'Opening replay…':'Loading model · about 10 MB',DETECTORS[state.arm].title,token,generation);
    try{
      const session=await getSession();if(!session||!activeOperation(token,generation))return;
      await warmup(session);if(!activeOperation(token,generation))return;
      status('Loading the synthetic example replay…');showBusy('Opening replay…','Prepared synthetic examples',token,generation);
      const response=await fetch('samples/detector-replay.mp4',{method:'HEAD'});if(!activeOperation(token,generation))return;
      if(!response.ok)throw new Error(`The synthetic example replay is unavailable (${response.status}).`);
      video.src='samples/detector-replay.mp4';video.loop=false;await video.play();if(!activeOperation(token,generation))return;
      hideBusy(token,generation);state.starting=false;state.running=true;await loop(session,token,generation);
    }catch(e){if(!activeOperation(token,generation))return;stop('Synthetic replay could not start.');error(e.message||String(e));}
  }
  async function loadFile(file) {
    if(!file)return;error();
    resetAutomaticZoom();
    if(file.type.startsWith('image/')){
      stop('');state.image=null;const token=state.token,generation=state.generation;
      showBusy('Opening photo…',file.name,token,generation);
      const url=URL.createObjectURL(file),image=new Image();
      try{await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('This image could not be decoded.'));image.src=url;});if(!activeOperation(token,generation))return;await detectImage(image,file.name);}
      catch(e){if(activeOperation(token,generation)){hideBusy(token,generation);error(e.message);}}finally{URL.revokeObjectURL(url);}
    }else if(file.type.startsWith('video/')){
      stop('');state.image=null;state.sourceKind='video';state.sourceName=file.name;resetTiming();
      const token=state.token,generation=state.generation;state.starting=true;$('detector-stop').disabled=false;$('detector-start').disabled=true;
      showBusy(state.session?'Opening video…':'Loading model · about 10 MB',DETECTORS[state.arm].title,token,generation);
      try{
        const session=await getSession();if(!session||!activeOperation(token,generation))return;
        await warmup(session);if(!activeOperation(token,generation))return;
        state.videoURL=URL.createObjectURL(file);video.src=state.videoURL;video.loop=false;await video.play();if(!activeOperation(token,generation))return;
        hideBusy(token,generation);state.starting=false;state.running=true;$('detector-stop').disabled=false;$('detector-start').disabled=true;await loop(session,token,generation);
      }catch(e){if(!activeOperation(token,generation))return;stop('Video detection could not start.');error(e.message||String(e));}
    }else error('Choose an image or video file.');
  }
  async function loadSamples() {
    try{
      const response=await fetch('samples/manifest.json');if(!response.ok)return;const data=await response.json(),rows=data.images||data;
      $('detector-samples').innerHTML=rows.map((row,i)=>`<button class="sample" data-detector-index="${i}" aria-label="Detect signs in synthetic ${escapeText(NAMES[CLASSES.indexOf(row.label)]||row.label)} example"><img src="samples/${escapeText(row.path)}" alt="Synthetic ${escapeText(row.label)} sign scene"><span>${escapeText(NAMES[CLASSES.indexOf(row.label)]||row.label)}</span></button>`).join('');
      $('detector-samples').querySelectorAll('button').forEach(button=>button.addEventListener('click',async()=>{
        resetAutomaticZoom();
        stop('');state.image=null;const token=state.token,generation=state.generation;
        showBusy('Opening photo…','Synthetic sign example',token,generation);
        const row=rows[Number(button.dataset.detectorIndex)],image=new Image();
        try{await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=reject;image.src=`samples/${row.path}`;});if(!activeOperation(token,generation))return;$('detector-samples').querySelectorAll('button').forEach(b=>b.classList.toggle('selected',b===button));await detectImage(image,row.title||row.path,'sample');}
        catch(e){if(activeOperation(token,generation)){hideBusy(token,generation);error(e.message||'The sample image could not be loaded.');}}
      }));
    }catch(e){console.warn('Detection samples unavailable',e);}
  }
  function percent(value){if(value===undefined||value===null||!Number.isFinite(Number(value)))return '—';const n=Number(value);return `${(n<=1?n*100:n).toFixed(2)}%`;}
  async function loadMetrics(){
    const content=$('detector-results-content');
    try{
      const response=await fetch(`results/detection.json?v=${DETECTOR_REVISION}`,{cache:'no-cache'});
      if(!response.ok)throw new Error('The latest detector results are unavailable.');
      const data=await response.json(),rows=(data.comparison||[]).filter(row=>row.condition==='clean');
      const selectedArm=data.selected_arm||data.selection?.selected_arm;
      const selected=rows.find(row=>row.arm===selectedArm);
      if(!selected||!rows.length)throw new Error('No validation-selected detector metrics are recorded.');
      const dataset=data.dataset||{},validation=data.selection?.validation_metrics||[];
      const selectedValidation=validation.find(row=>row.arm===selectedArm);
      state.metrics=data;updateActiveModel();
      $('detector-stress-note').textContent=data.stress_evaluation?.note||'No quantitative stress evaluation is available for this detector. Controls apply deterministic browser simulations.';
      $('detector-test-outcome').hidden=false;
      $('detector-test-outcome').textContent=`Clean test mAP@0.50:0.95: ${rows.map(row=>`${ARM_NAMES[row.arm]||row.arm} ${percent(row.map50_95)}`).join(' · ')}. Added training data did not improve this metric over the original-data arm.`;

      const wrapper=document.createElement('section');wrapper.className='detector-experiment-section';
      wrapper.setAttribute('aria-label','Latest three-arm detector experiment');
      wrapper.innerHTML='<p class="eyebrow">LATEST DETECTOR STUDY · TESLA T4</p><h2>Three training sets. Original data won validation.</h2><p class="result-note">Each detector checkpoint and the experiment winner were selected by validation mAP@0.50:0.95 before the test set was evaluated. The app defaults to Expanded + synthetic for the demonstration.</p>';
      wrapper.innerHTML+=`<div class="metrics-summary"><span class="metric-pill"><strong>353 / 453 / 581</strong>training images</span><span class="metric-pill"><strong>${escapeText(dataset.validation_images??120)}</strong>shared validation images</span><span class="metric-pill"><strong>${escapeText(dataset.test_images??96)}</strong>shared test images</span><span class="metric-pill"><strong>${escapeText(dataset.added_real_frames??100)} frames / ${escapeText(dataset.added_real_boxes??169)} boxes</strong>added real annotations</span><span class="metric-pill"><strong>${escapeText(dataset.synthetic_composites??128)}</strong>synthetic composites</span></div>`;
      wrapper.innerHTML+=`<p id="detector-validation-selection" class="selection-result">Validation winner: ${escapeText(ARM_NAMES[selectedArm]||selectedArm)} · validation mAP@0.50:0.95 ${percent(selectedValidation?.mAP50_95)}. App default: ${escapeText(ARM_NAMES[DEFAULT_DETECTOR])}. The detector selector changes inference, not this experiment result.</p>`;
      wrapper.innerHTML+='<p class="tiny">Clean evaluation on the original 96-image test set. mAP measures class and bounding-box agreement; these values do not change with browser simulations.</p>';
      wrapper.innerHTML+='<div class="table-wrap"><table id="detector-results-table"><thead><tr><th scope="col">Training arm</th><th scope="col">Train images / boxes</th><th scope="col">Validation mAP@0.50:0.95</th><th scope="col">Test mAP@0.50</th><th scope="col">Test mAP@0.50:0.95</th><th scope="col">Test precision</th><th scope="col">Test recall</th></tr></thead><tbody>'+rows.map(row=>`<tr${row.arm===selectedArm?' class="selected-detector-row"':''} data-arm="${escapeText(row.arm)}"><td>${escapeText(ARM_NAMES[row.arm]||row.model||row.arm)}${row.arm===selectedArm?' <span class="deployed-label">VALIDATION SELECTED</span>':''}${row.arm===DEFAULT_DETECTOR?' <span class="deployed-label">APP DEFAULT</span>':''}</td><td>${escapeText(row.train_images??'—')} / ${escapeText(row.train_boxes??'—')}</td><td>${percent(row.validation_map50_95)}</td><td>${percent(row.map50)}</td><td>${percent(row.map50_95)}</td><td>${percent(row.precision)}</td><td>${percent(row.recall)}</td></tr>`).join('')+'</tbody></table></div>';
      wrapper.innerHTML+=`<p id="detector-stress-results-note" class="study-status-note">${escapeText(data.stress_evaluation?.note||'Stress-condition scores have not been measured for this study.')}</p>`;
      if(data.protocol_description)wrapper.innerHTML+=`<details class="detector-protocol"><summary>Training and evaluation protocol</summary><p class="result-note">${escapeText(data.protocol_description)}</p></details>`;
      const notes=Array.isArray(data.notes)?data.notes:data.notes?[data.notes]:[];
      wrapper.innerHTML+=notes.map(note=>`<p class="result-note">${escapeText(note)}</p>`).join('');
      if(data.environment)wrapper.innerHTML+=`<p class="tiny">Detector experiment environment: ${escapeText(typeof data.environment==='string'?data.environment:JSON.stringify(data.environment))}. Recorded backend timing is not an arm-to-arm speed comparison; live browser timing is measured independently.</p>`;
      content.replaceChildren(wrapper);
    }catch(e){
      $('detector-map50').textContent='—';$('detector-map5095').textContent='—';
      $('detector-selected-arm').hidden=true;$('detector-test-outcome').hidden=true;
      $('detector-metrics-note').textContent='Latest recorded detector metrics are unavailable.';
      content.innerHTML='<p class="empty-message">The latest detector study could not be loaded. Recorded scores are unavailable; browser inference still uses the bundled model.</p>';
      console.warn('Recorded detection metrics unavailable',e);
    }
  }
  $('detector-zoom').addEventListener('input',e=>{state.autoZoomApplied=false;state.zoom=Math.max(1,Math.min(4,Number(e.target.value)||1));$('detector-zoom-value').textContent=`${state.zoom.toFixed(1)}×`;resetTiming();});
  $('detector-zoom').addEventListener('change',()=>{if(state.image)detectImage(state.image,state.sourceName,state.sourceKind);});
  $('detector-auto-zoom').addEventListener('change',()=>{if(!$('detector-auto-zoom').checked)state.autoZoomApplied=false;if(state.image&&state.sourceKind==='image')detectImage(state.image,state.sourceName,state.sourceKind);});
  $('detector-model-select').addEventListener('change',e=>{switchDetector(e.target.value).catch(e=>{console.error(e);error(e.message||String(e));});});
  $('detector-start').addEventListener('click',startCamera);$('detector-replay').addEventListener('click',startReplay);$('detector-stop').addEventListener('click',()=>stop());$('detector-upload').addEventListener('click',()=>$('detector-file').click());$('detector-file').addEventListener('change',e=>loadFile(e.target.files[0]));
  $('detector-condition').addEventListener('change',()=>{$('detector-condition-note').textContent=conditionNotes[$('detector-condition').value];if(state.image)detectImage(state.image,state.sourceName,state.sourceKind);else if(state.running)resetTiming();});
  $('detector-confidence').addEventListener('input',()=>{$('detector-confidence-value').textContent=Number($('detector-confidence').value).toFixed(2);});
  $('detector-confidence').addEventListener('change',()=>{if(state.image)detectImage(state.image,state.sourceName,state.sourceKind);});
  document.addEventListener('signlab:tabchange',e=>{if(e.detail!=='detection'&&(state.running||state.starting||state.stream||state.videoURL))stop('Detection stopped when leaving the detector tab.');});
  window.addEventListener('pagehide',()=>stop(''));document.addEventListener('visibilitychange',()=>{if(document.hidden&&state.sourceKind==='camera'&&(state.running||state.starting||state.stream))stop('Camera detection stopped while the page is hidden.');});
  updateActiveModel();loadSamples();loadMetrics();window.SignLabDetector={state,sourceFrame,prepare,decode,nms,intersectionOverUnion,detectImage,loadFile,startCamera,startReplay,stop,switchDetector,getSession};
})();
