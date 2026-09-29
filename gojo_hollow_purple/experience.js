import {createArena,cast,stepArena,aimDirection,TECHNIQUES} from './combat.mjs';
// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

const MERGE_DIST   = 0.13;
const CHARGE_TIME  = 0.8;
const BEAM_DUR     = 2.5;
const COOLDOWN_DUR = 1.0;
const CINEMATIC_BAR_MAX = 0.13;
const TWO_PI = Math.PI * 2;

const State = {
  IDLE:0, BLUE:1, RED:2, BOTH:3,
  MERGING:4, CHARGING:5, HOLDING:6, SHOOTING:7, COOLDOWN:8
};

let state = State.IDLE;
let mode = null, running = false, raf = 0, attempt = 0;
let reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
let arena=createArena();
let lastTrackingAt=0, lastDetectionAt=0;
let lastVideoTime = -1, lostSince = 0;
let previousState = -1;
const $ = id => document.getElementById(id);
let pulse = 0, chargeProgress = 0;
let mergeStart = 0, shootStart = 0, cooldownStart = 0;
let flashIntensity = 0, shakeIntensity = 0, chromaticIntensity = 0;
let mergeCenter = {x:.5,y:.5};

// Holding state
let holdingHand = null;        // Purple is held and aimed by the right hand.
let holdPurplePos = {x:.5,y:.5}; // smoothed sphere position while held
let aimAngle = 0;              // direction the controlling hand points
let holdArmed = false;         // requires crossed fingers before release can fire

// Shooting state
let purpleSpherePos = null;
let shootOrigin = {x:.5,y:.5};
let releaseBurst = 0;          // one-shot explosion intensity

// Hand data
let leftPalm=null, rightPalm=null;
let leftTips=[], rightTips=[];
let leftSign=false, rightSign=false;
let leftCrossed=false, rightCrossed=false;
let leftReleased=false, rightReleased=false;
let leftSize=0, rightSize=0;
let leftFingerDir=null, rightFingerDir=null; // normalized pointing direction
let leftCrossFrames=0, rightCrossFrames=0;
let leftReleaseFrames=0, rightReleaseFrames=0;

let particles = [];
let emissionClock=0;
let lastTime = performance.now()/1000;

// ═══════════════════════════════════════════════════════════════════════════
// CANVAS
// ═══════════════════════════════════════════════════════════════════════════

const video = document.getElementById("cam");
const distC = document.getElementById("distortion");
const fxC   = document.getElementById("fx");
const hudC  = document.getElementById("hud");
const dist  = distC.getContext("2d");
const fx    = fxC.getContext("2d");
const hud   = hudC.getContext("2d");

function resize(){
  const w=window.innerWidth, h=window.innerHeight;
  [distC,fxC,hudC].forEach(c=>{c.width=w;c.height=h});
}
window.addEventListener("resize",resize); resize();

// ═══════════════════════════════════════════════════════════════════════════
// MEDIAPIPE
// ═══════════════════════════════════════════════════════════════════════════

let handLandmarker = null;

async function initMediaPipe(){
  const {HandLandmarker, FilesetResolver} = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs');
  const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm');
  const options = {baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'GPU'},runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:.6,minHandPresenceConfidence:.6,minTrackingConfidence:.5};
  try { return await HandLandmarker.createFromOptions(vision, options); }
  catch { options.baseOptions.delegate='CPU'; return await HandLandmarker.createFromOptions(vision, options); }
}
function stopCamera(){
  video.srcObject?.getTracks().forEach(t=>t.stop());
  video.srcObject=null;
  handLandmarker?.close(); handLandmarker=null;
}
// Match mirrored object-fit:cover video, including portrait/landscape crop.
function screenLandmark(p){
  const W=innerWidth,H=innerHeight,vw=video.videoWidth||1280,vh=video.videoHeight||720;
  const scale=Math.max(W/vw,H/vh);
  return {...p,x:1-(p.x*vw*scale-(vw*scale-W)/2)/W,y:(p.y*vh*scale-(vh*scale-H)/2)/H};
}

// ═══════════════════════════════════════════════════════════════════════════
// HAND ANALYSIS — GOJO GESTURE DETECTION
// ═══════════════════════════════════════════════════════════════════════════

function palmCenter(lm){
  const ids=[0,5,9,13,17]; let sx=0,sy=0;
  for(const i of ids){sx+=lm[i].x;sy+=lm[i].y}
  return {x:sx/ids.length, y:sy/ids.length};
}

function palmSz(lm){
  const dx=lm[0].x-lm[9].x, dy=lm[0].y-lm[9].y;
  return Math.sqrt(dx*dx+dy*dy);
}

function fingerTipPositions(lm){
  return [4,8,12,16,20].map(i=>({x:lm[i].x,y:lm[i].y}));
}

function twoFingersUp(lm){
  // Wrist-relative distances work with rotated hands, unlike vertical y tests.
  const d=(a,b)=>Math.hypot((a.x-b.x)*innerWidth,(a.y-b.y)*innerHeight);
  const extended=(tip,pip)=>d(lm[tip],lm[0])>d(lm[pip],lm[0])*1.12;
  return extended(8,6)&&extended(12,10)&&!extended(16,14)&&!extended(20,18);
}
function crossingSign(lm){
  const bx=(lm[5].x-lm[9].x)*innerWidth,by=(lm[5].y-lm[9].y)*innerHeight;
  const tx=(lm[8].x-lm[12].x)*innerWidth,ty=(lm[8].y-lm[12].y)*innerHeight;
  return bx*tx+by*ty;
}
function isCrossed(lm){ return twoFingersUp(lm)&&crossingSign(lm)<0; }
function isReleased(lm){ return twoFingersUp(lm)&&crossingSign(lm)>=0; }

function getPointingDir(lm){
  // Direction from wrist toward index+middle fingertip midpoint
  const wx = lm[0].x, wy = lm[0].y;
  const fx = (lm[8].x + lm[12].x) / 2;
  const fy = (lm[8].y + lm[12].y) / 2;
  const dx = fx - wx, dy = fy - wy;
  const len = Math.sqrt(dx*dx + dy*dy) || 0.001;
  return { x: dx/len, y: dy/len };
}

function processHands(result){
  leftPalm=rightPalm=null; leftTips=[]; rightTips=[];
  leftSign=rightSign=false;
  leftCrossed=rightCrossed=false;
  leftReleased=rightReleased=false;
  leftFingerDir=rightFingerDir=null;

  if(!result.landmarks?.length){
    leftCrossFrames=rightCrossFrames=leftReleaseFrames=rightReleaseFrames=0;
    return;
  }

  for(let h=0;h<result.landmarks.length;h++){
    const lm=result.landmarks[h].map(screenLandmark);
    const label=result.handedness[h][0].categoryName;
    const palm=palmCenter(lm);
    const tips=fingerTipPositions(lm);
    const sign=twoFingersUp(lm);
    const crossed=isCrossed(lm);
    const released=isReleased(lm);
    const size=palmSz(lm);
    const dir=getPointingDir(lm);

    // Use the tracker label directly. screenLandmark mirrors only the drawing
    // coordinates; it must not reverse which physical hand owns the gesture.
    if(label==="Right"){
      rightPalm=palm; rightTips=tips; rightSign=sign; rightSize=size;
      rightCrossed=crossed; rightReleased=released;
      rightFingerDir=dir;
    } else {
      leftPalm=palm; leftTips=tips; leftSign=sign; leftSize=size;
      leftCrossed=crossed; leftReleased=released;
      leftFingerDir=dir;
    }
  }

  // Temporal smoothing for gesture stability
  leftCrossFrames = leftCrossed ? Math.min(6, leftCrossFrames + 1) : Math.max(0, leftCrossFrames - 1);
  rightCrossFrames = rightCrossed ? Math.min(6, rightCrossFrames + 1) : Math.max(0, rightCrossFrames - 1);
  leftReleaseFrames = leftReleased ? Math.min(6, leftReleaseFrames + 1) : Math.max(0, leftReleaseFrames - 1);
  rightReleaseFrames = rightReleased ? Math.min(6, rightReleaseFrames + 1) : Math.max(0, rightReleaseFrames - 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// PARTICLE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

function spawn(x,y,vx,vy,life,r,g,b,size,type='normal'){
  if(particles.length < (reduced?250:1400)) particles.push({x,y,vx,vy,life,maxLife:life,size,r,g,b,type,trail:[]});
}

function emitBlueVortex(cx,cy,n=8){
  for(let i=0;i<n;i++){
    const a=Math.random()*TWO_PI;
    const d=0.03+Math.random()*0.07;
    const px=cx+Math.cos(a)*d, py=cy+Math.sin(a)*d;
    const inA=a+Math.PI+0.8*(Math.random()>0.5?1:-1);
    const sp=0.04+Math.random()*0.1;
    const colors=[[30,100,255],[0,180,255],[80,160,255],[150,200,255]];
    const c=colors[Math.floor(Math.random()*colors.length)];
    spawn(px,py,Math.cos(inA)*sp,Math.sin(inA)*sp, 0.4+Math.random()*0.5, c[0],c[1],c[2], 1+Math.random()*2.5, 'blue');
  }
}

function emitRedBurst(cx,cy,n=7){
  for(let i=0;i<n;i++){
    const a=Math.random()*TWO_PI;
    const sp=0.08+Math.random()*0.2;
    const colors=[[255,40,40],[255,80,20],[255,120,60],[255,200,150]];
    const c=colors[Math.floor(Math.random()*colors.length)];
    spawn(cx,cy,Math.cos(a)*sp,Math.sin(a)*sp, 0.25+Math.random()*0.4, c[0],c[1],c[2], 1.5+Math.random()*3, 'red');
  }
}

function emitPurpleVortex(cx,cy,n=14){
  for(let i=0;i<n;i++){
    const a=Math.random()*TWO_PI;
    const d=0.04+Math.random()*0.14;
    const px=cx+Math.cos(a)*d, py=cy+Math.sin(a)*d;
    const inA=a+Math.PI+(Math.random()-0.5)*0.5;
    const sp=0.1+Math.random()*0.22;
    const colors=[[120,0,200],[180,50,255],[80,0,160],[200,100,255],[255,255,255]];
    const c=colors[Math.floor(Math.random()*colors.length)];
    spawn(px,py,Math.cos(inA)*sp,Math.sin(inA)*sp, 0.2+Math.random()*0.5, c[0],c[1],c[2], 1.5+Math.random()*4, 'purple');
  }
}

function emitHoldAura(cx,cy,n=6){
  for(let i=0;i<n;i++){
    const a=Math.random()*TWO_PI;
    const d=0.02+Math.random()*0.04;
    const px=cx+Math.cos(a)*d, py=cy+Math.sin(a)*d;
    const sp=0.01+Math.random()*0.03;
    const colors=[[120,0,200],[180,50,255],[60,0,120],[200,100,255]];
    const c=colors[Math.floor(Math.random()*colors.length)];
    spawn(px,py,Math.cos(a)*sp,Math.sin(a)*sp, 0.3+Math.random()*0.6, c[0],c[1],c[2], 1.5+Math.random()*3, 'hold');
  }
}

function emitSphereTrail(cx,cy,angle,n=10){
  for(let i=0;i<n;i++){
    const a=Math.random()*TWO_PI;
    const sp=0.02+Math.random()*0.06;
    const colors=[[100,0,180],[60,0,120],[180,80,255],[40,0,80]];
    const c=colors[Math.floor(Math.random()*colors.length)];
    // Trail drifts opposite to travel direction
    const tvx=Math.cos(a)*sp - Math.cos(angle)*0.05;
    const tvy=Math.sin(a)*sp - Math.sin(angle)*0.05;
    spawn(cx,cy,tvx,tvy, 0.4+Math.random()*0.8, c[0],c[1],c[2], 2+Math.random()*5, 'trail');
  }
}

function emitImpactBurst(cx,cy,n=200){
  for(let i=0;i<n;i++){
    const a=Math.random()*TWO_PI;
    const sp=0.05+Math.random()*0.5;
    const colors=[[120,0,200],[180,50,255],[60,0,120],[255,200,255],[255,255,255],[0,0,0]];
    const c=colors[Math.floor(Math.random()*colors.length)];
    spawn(cx,cy,Math.cos(a)*sp,Math.sin(a)*sp, 0.5+Math.random()*1.5, c[0],c[1],c[2], 2+Math.random()*8, 'impact');
  }
}

function updateParticles(dt){
  for(const p of particles){
    p.trail.push({x:p.x,y:p.y});
    if(p.trail.length>8) p.trail.shift();
    p.x+=p.vx*dt; p.y+=p.vy*dt;
    if(p.type==='blue') { p.vx*=0.96; p.vy*=0.96; }
    if(p.type==='trail'||p.type==='hold') { p.vx*=0.95; p.vy*=0.95; }
    p.life-=dt;
  }
  particles=particles.filter(p=>p.life>0);
}

function drawParticles(ctx,W,H){
  for(const p of particles){
    const alpha=Math.max(0,p.life/p.maxLife);
    const px=p.x*W, py=p.y*H;
    if(px<-80||px>W+80||py<-80||py>H+80) continue;
    const r=p.size*alpha;

    ctx.globalAlpha=alpha*0.15;
    for(let i=0;i<p.trail.length;i++){
      const ta=(i/p.trail.length)*0.3;
      ctx.globalAlpha=ta*alpha;
      const tr=r*(i/p.trail.length)*0.6;
      ctx.fillStyle=`rgb(${p.r},${p.g},${p.b})`;
      ctx.beginPath();
      ctx.arc(p.trail[i].x*W, p.trail[i].y*H, Math.max(0.5,tr), 0, TWO_PI);
      ctx.fill();
    }

    ctx.globalAlpha=alpha*0.9;
    ctx.fillStyle=`rgb(${p.r},${p.g},${p.b})`;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, TWO_PI);
    ctx.fill();
  }
  ctx.globalAlpha=1;
}

// ═══════════════════════════════════════════════════════════════════════════
// PROCEDURAL VISUAL EFFECTS
// ═══════════════════════════════════════════════════════════════════════════

function drawBlueOrb(ctx,W,H,nx,ny,size,p){
  const cx=nx*W, cy=ny*H;
  const baseR=size*W*0.32;
  const r=baseR*(1+0.08*Math.sin(p*5));

  for(let i=4;i>0;i--){
    const rr=r+i*18+6*Math.sin(p*4+i);
    ctx.globalAlpha=0.06*(1-i/5);
    ctx.strokeStyle='rgb(0,140,255)';
    ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(cx,cy,rr,0,TWO_PI); ctx.stroke();
  }

  const voidGrad=ctx.createRadialGradient(cx,cy,0,cx,cy,r);
  voidGrad.addColorStop(0,'rgba(0,0,20,0.9)');
  voidGrad.addColorStop(0.4,'rgba(0,20,80,0.7)');
  voidGrad.addColorStop(0.7,'rgba(0,80,200,0.4)');
  voidGrad.addColorStop(0.9,'rgba(0,150,255,0.15)');
  voidGrad.addColorStop(1,'rgba(0,150,255,0)');
  ctx.globalAlpha=1; ctx.fillStyle=voidGrad;
  ctx.beginPath(); ctx.arc(cx,cy,r,0,TWO_PI); ctx.fill();

  ctx.globalAlpha=0.7; ctx.strokeStyle='rgba(50,150,255,0.8)'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(cx,cy,r*0.85,0,TWO_PI); ctx.stroke();

  for(let i=0;i<3;i++){
    const a=p*3+i*TWO_PI/3;
    ctx.globalAlpha=0.5; ctx.strokeStyle='rgba(80,180,255,0.7)'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(cx,cy,r+8,a,a+1.2); ctx.stroke();
  }

  const flicker=2+1.5*Math.sin(p*12);
  ctx.globalAlpha=0.6; ctx.fillStyle='rgba(180,220,255,0.8)';
  ctx.beginPath(); ctx.arc(cx,cy,flicker,0,TWO_PI); ctx.fill();
  ctx.globalAlpha=1;
}

function drawRedOrb(ctx,W,H,nx,ny,size,p){
  const cx=nx*W, cy=ny*H;
  const baseR=size*W*0.3;
  const r=baseR*(1+0.12*Math.sin(p*7));

  for(let i=0;i<3;i++){
    const phase=(p*2+i*0.7)%2;
    const ringR=r+phase*40;
    const ringAlpha=Math.max(0, 0.4*(1-phase/2));
    ctx.globalAlpha=ringAlpha; ctx.strokeStyle='rgba(255,60,30,0.8)'; ctx.lineWidth=2-phase*0.8;
    ctx.beginPath(); ctx.arc(cx,cy,ringR,0,TWO_PI); ctx.stroke();
  }

  const coreGrad=ctx.createRadialGradient(cx,cy,0,cx,cy,r);
  coreGrad.addColorStop(0,'rgba(255,255,240,0.95)');
  coreGrad.addColorStop(0.2,'rgba(255,200,150,0.8)');
  coreGrad.addColorStop(0.5,'rgba(255,60,30,0.6)');
  coreGrad.addColorStop(0.8,'rgba(200,20,0,0.25)');
  coreGrad.addColorStop(1,'rgba(150,0,0,0)');
  ctx.globalAlpha=1; ctx.fillStyle=coreGrad;
  ctx.beginPath(); ctx.arc(cx,cy,r,0,TWO_PI); ctx.fill();

  ctx.globalAlpha=0.6; ctx.strokeStyle='rgba(255,100,50,0.9)'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(cx,cy,r*0.7,0,TWO_PI); ctx.stroke();

  ctx.globalAlpha=0.4; ctx.strokeStyle='rgba(255,200,100,0.6)'; ctx.lineWidth=1;
  for(let i=0;i<8;i++){
    const a=p*2+i*TWO_PI/8;
    const inner=r*0.8, outer=r+10+8*Math.sin(p*10+i*2);
    ctx.beginPath();
    ctx.moveTo(cx+Math.cos(a)*inner,cy+Math.sin(a)*inner);
    ctx.lineTo(cx+Math.cos(a)*outer,cy+Math.sin(a)*outer);
    ctx.stroke();
  }
  ctx.globalAlpha=1;
}

function drawPurpleVoid(ctx,W,H,nx,ny,radius,p,intensity=1){
  const cx=nx*W, cy=ny*H, r=radius;

  for(let i=6;i>0;i--){
    const rr=r+i*20+8*Math.sin(p*3+i*0.5);
    ctx.globalAlpha=0.04*(1-i/7)*intensity;
    ctx.strokeStyle='rgba(150,50,255,0.6)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.arc(cx,cy,rr,0,TWO_PI); ctx.stroke();
  }

  const voidGrad=ctx.createRadialGradient(cx,cy,0,cx,cy,r);
  voidGrad.addColorStop(0,'rgba(0,0,0,0.97)');
  voidGrad.addColorStop(0.3,'rgba(20,0,40,0.9)');
  voidGrad.addColorStop(0.5,'rgba(60,0,120,0.7)');
  voidGrad.addColorStop(0.7,'rgba(120,30,200,0.4)');
  voidGrad.addColorStop(0.85,'rgba(160,60,255,0.15)');
  voidGrad.addColorStop(1,'rgba(180,80,255,0)');
  ctx.globalAlpha=intensity; ctx.fillStyle=voidGrad;
  ctx.beginPath(); ctx.arc(cx,cy,r,0,TWO_PI); ctx.fill();

  ctx.globalAlpha=0.8*intensity; ctx.strokeStyle='rgba(180,60,255,0.9)'; ctx.lineWidth=2.5;
  ctx.beginPath(); ctx.arc(cx,cy,r*0.75,0,TWO_PI); ctx.stroke();

  const hazeGrad=ctx.createRadialGradient(cx,cy,r*0.8,cx,cy,r*2);
  hazeGrad.addColorStop(0,'rgba(140,40,255,0.2)');
  hazeGrad.addColorStop(0.5,'rgba(100,20,200,0.08)');
  hazeGrad.addColorStop(1,'rgba(80,0,160,0)');
  ctx.globalAlpha=intensity; ctx.fillStyle=hazeGrad;
  ctx.beginPath(); ctx.arc(cx,cy,r*2,0,TWO_PI); ctx.fill();

  for(let i=0;i<4;i++){
    const a=p*4+i*TWO_PI/4;
    ctx.globalAlpha=0.6*intensity; ctx.strokeStyle='rgba(200,120,255,0.7)'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(cx,cy,r+5,a,a+0.8); ctx.stroke();
  }
  for(let i=0;i<3;i++){
    const a=-p*6+i*TWO_PI/3;
    ctx.globalAlpha=0.4*intensity; ctx.strokeStyle='rgba(255,200,255,0.5)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.arc(cx,cy,r*0.5,a,a+0.6); ctx.stroke();
  }
  for(let i=0;i<5;i++){
    const fa=p*8+i*1.3;
    const fd=r*0.4*Math.abs(Math.sin(fa));
    const fx2=cx+Math.cos(fa)*fd, fy2=cy+Math.sin(fa)*fd;
    ctx.globalAlpha=0.3*Math.abs(Math.sin(fa*2))*intensity;
    ctx.fillStyle='rgba(200,150,255,0.8)';
    ctx.beginPath(); ctx.arc(fx2,fy2,1.5,0,TWO_PI); ctx.fill();
  }
  ctx.globalAlpha=1;
}

function drawFingerThreads(ctx,W,H,palm,tips,color,p){
  for(let i=0;i<tips.length;i++){
    const tx=tips[i].x*W, ty=tips[i].y*H;
    const px=palm.x*W, py=palm.y*H;
    const mx=(px+tx)/2+10*Math.sin(p*5+i*1.3);
    const my=(py+ty)/2+10*Math.cos(p*4+i*1.1);
    ctx.globalAlpha=0.35; ctx.strokeStyle=color; ctx.lineWidth=1.2;
    ctx.beginPath(); ctx.moveTo(tx,ty); ctx.quadraticCurveTo(mx,my,px,py); ctx.stroke();
    const gr=2+2*Math.sin(p*7+i);
    ctx.globalAlpha=0.7; ctx.fillStyle=color;
    ctx.beginPath(); ctx.arc(tx,ty,gr,0,TWO_PI); ctx.fill();
  }
  ctx.globalAlpha=1;
}

// Aim direction indicator while holding purple
function drawAimLine(ctx,W,H,fromX,fromY,angle,p){
  const cx=fromX*W, cy=fromY*H;
  const len=60+10*Math.sin(p*6);
  const ex=cx+Math.cos(angle)*len, ey=cy+Math.sin(angle)*len;

  ctx.globalAlpha=0.3;
  ctx.strokeStyle='rgba(180,80,255,0.6)';
  ctx.lineWidth=1.5;
  ctx.setLineDash([6,6]);
  ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(ex,ey); ctx.stroke();
  ctx.setLineDash([]);

  // Arrow tip
  const tipLen=8;
  ctx.beginPath();
  ctx.moveTo(ex,ey);
  ctx.lineTo(ex-Math.cos(angle-0.4)*tipLen, ey-Math.sin(angle-0.4)*tipLen);
  ctx.moveTo(ex,ey);
  ctx.lineTo(ex-Math.cos(angle+0.4)*tipLen, ey-Math.sin(angle+0.4)*tipLen);
  ctx.stroke();
  ctx.globalAlpha=1;
}

function drawTravelingSphere(ctx,W,H,nx,ny,radius,p,age,angle){
  drawPurpleVoid(ctx,W,H,nx,ny,radius,p,Math.min(1,age*3));

  const cx=nx*W, cy=ny*H;
  const trailLen=radius*6;
  // Trail in opposite direction of travel
  const tx=cx-Math.cos(angle)*trailLen;
  const ty=cy-Math.sin(angle)*trailLen;

  const trailGrad=ctx.createLinearGradient(tx,ty,cx,cy);
  trailGrad.addColorStop(0,'rgba(0,0,0,0)');
  trailGrad.addColorStop(0.4,'rgba(40,0,80,0.15)');
  trailGrad.addColorStop(0.8,'rgba(20,0,40,0.3)');
  trailGrad.addColorStop(1,'rgba(0,0,0,0.1)');
  ctx.globalAlpha=0.6; ctx.fillStyle=trailGrad;

  // Draw trail as rotated rect
  ctx.save();
  ctx.translate(cx,cy);
  ctx.rotate(angle+Math.PI);
  ctx.fillRect(0,-radius*1.5,trailLen,radius*3);
  ctx.restore();

  ctx.globalAlpha=0.2; ctx.strokeStyle='rgba(150,80,255,0.4)'; ctx.lineWidth=1;
  const perpX=-Math.sin(angle), perpY=Math.cos(angle);
  for(let i=0;i<6;i++){
    const off=(i-2.5)*radius*0.5;
    const sx=cx-Math.cos(angle)*radius+perpX*off;
    const sy=cy-Math.sin(angle)*radius+perpY*off;
    const ex2=cx-Math.cos(angle)*trailLen+perpX*off;
    const ey2=cy-Math.sin(angle)*trailLen+perpY*off;
    const waveAmp=5*Math.sin(p*10+i*2);
    const mx2=(sx+ex2)/2+perpX*waveAmp, my2=(sy+ey2)/2+perpY*waveAmp;
    ctx.beginPath(); ctx.moveTo(sx,sy); ctx.quadraticCurveTo(mx2,my2,ex2,ey2); ctx.stroke();
  }
  ctx.globalAlpha=1;
}

function drawSpeedLines(ctx,W,H,angle,intensity,p){
  if(intensity <= 0.01) return;
  const cx = W * 0.5;
  const cy = H * 0.5;
  const perpX = -Math.sin(angle);
  const perpY = Math.cos(angle);
  ctx.globalAlpha = 0.08 * intensity;
  ctx.strokeStyle = 'rgba(230,200,255,0.7)';
  for(let i=0;i<22;i++){
    const band = (i - 11) * (H * 0.03);
    const jitter = Math.sin(p * 10 + i) * 20;
    const sx = cx - Math.cos(angle) * W * 0.65 + perpX * band;
    const sy = cy - Math.sin(angle) * W * 0.65 + perpY * band;
    const ex = cx + Math.cos(angle) * W * 0.65 + perpX * band + jitter;
    const ey = cy + Math.sin(angle) * W * 0.65 + perpY * band + jitter * 0.25;
    ctx.lineWidth = 0.8 + (i % 3) * 0.35;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// DISTORTION LAYER
// ═══════════════════════════════════════════════════════════════════════════

function drawCameraGhost(offset,W,H){
  const scale=Math.max(W/video.videoWidth,H/video.videoHeight);
  const w=video.videoWidth*scale,h=video.videoHeight*scale;
  dist.save(); dist.translate(W+offset,0); dist.scale(-1,1);
  dist.drawImage(video,(W-w)/2,(H-h)/2,w,h); dist.restore();
}
function renderDistortion(W,H){
  dist.clearRect(0,0,W,H);

  if(chromaticIntensity>0.01 && video.readyState>=2 && !reduced){
    const offset=chromaticIntensity*8;
    dist.globalAlpha=chromaticIntensity*0.15;
    drawCameraGhost(offset,W,H);
    dist.globalCompositeOperation='multiply';
    dist.fillStyle='rgba(255,0,0,0.3)'; dist.fillRect(0,0,W,H);
    dist.globalCompositeOperation='source-over';
    dist.globalAlpha=chromaticIntensity*0.15;
    drawCameraGhost(-offset,W,H);
    dist.globalCompositeOperation='multiply';
    dist.fillStyle='rgba(0,0,255,0.3)'; dist.fillRect(0,0,W,H);
    dist.globalCompositeOperation='source-over';
    dist.globalAlpha=1;
  }

  if(state!==State.IDLE && state!==State.COOLDOWN){
    const vi=state>=State.CHARGING?0.5:0.25;
    const vg=dist.createRadialGradient(W/2,H/2,W*0.2,W/2,H/2,W*0.7);
    vg.addColorStop(0,'rgba(0,0,0,0)'); vg.addColorStop(1,`rgba(0,0,0,${vi})`);
    dist.fillStyle=vg; dist.fillRect(0,0,W,H);
  }

  // Cinematic letterbox bars during key beats
  const dramatic = (state===State.CHARGING || state===State.HOLDING || state===State.SHOOTING);
  const barFrac = dramatic ? CINEMATIC_BAR_MAX : 0;
  const barH = H * barFrac;
  if(barH > 1){
    dist.globalAlpha = 0.95;
    dist.fillStyle = 'rgba(0,0,0,0.95)';
    dist.fillRect(0,0,W,barH);
    dist.fillRect(0,H-barH,W,barH);
    dist.globalAlpha = 1;
  }

  if(state===State.SHOOTING&&!reduced){
    const age=(performance.now()/1000-shootStart)/BEAM_DUR;
    if(age<0.15){
      dist.globalAlpha=0.3*(1-age/0.15);
      dist.fillStyle='rgba(100,0,200,0.5)'; dist.fillRect(0,0,W,H);
      dist.globalAlpha=1;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE MACHINE
// ═══════════════════════════════════════════════════════════════════════════

function fingertipFocus(tips,palm){
  if(tips.length>=3){
    return {x:(tips[1].x+tips[2].x)/2,y:(tips[1].y+tips[2].y)/2};
  }
  return palm;
}
function leftFocus(){ return fingertipFocus(leftTips,leftPalm); }
function rightFocus(){ return fingertipFocus(rightTips,rightPalm); }
function mergedFocus(){
  const left=leftFocus(),right=rightFocus();
  if(!left||!right)return null;
  return {x:(left.x+right.x)/2,y:(left.y+right.y)/2};
}
function handDist(){
  const left=leftFocus(),right=rightFocus();
  if(!left||!right) return Infinity;
  const dx=left.x-right.x, dy=left.y-right.y;
  return Math.sqrt(dx*dx+dy*dy);
}

function updateState(dt,now){
  const d=handDist();
  const both=leftPalm&&rightPalm;
  const la=leftPalm&&leftSign;
  const ra=rightPalm&&rightSign;

  switch(state){
    case State.IDLE:
      chromaticIntensity*=0.9;
      if(both&&la&&ra) state=State.BOTH;
      else if(la) state=State.BLUE;
      else if(ra) state=State.RED;
      break;

    case State.BLUE:
      if(la&&ra) state=State.BOTH;
      else if(!la) state=State.IDLE;
      break;

    case State.RED:
      if(la&&ra) state=State.BOTH;
      else if(!ra) state=State.IDLE;
      break;

    case State.BOTH:
      if(!la||!ra){state=la?State.BLUE:ra?State.RED:State.IDLE;break;}
      if(d<MERGE_DIST){mergeCenter=mergedFocus()??mergeCenter;state=State.MERGING;mergeStart=now;}
      break;

    case State.MERGING:
      chromaticIntensity=Math.min(chromaticIntensity+dt*0.5,0.3);
      if(!la||!ra){state=State.IDLE;break;}
      if(d>MERGE_DIST*1.5){state=State.BOTH;break;}
      if(now-mergeStart>CHARGE_TIME*0.2){state=State.CHARGING;chargeProgress=0;}
      break;

    case State.CHARGING:
      chargeProgress=Math.min(chargeProgress+dt/CHARGE_TIME,1.0);
      chromaticIntensity=Math.min(0.3+chargeProgress*0.4,0.7);
      mergeCenter=mergedFocus()??mergeCenter;
      if((!la||!ra)&&chargeProgress<0.8){
        state=State.IDLE; chargeProgress=0; chromaticIntensity=0; break;
      }
      if(chargeProgress>=1.0){
        // Charged! Transition to HOLDING
        state=State.HOLDING;
        // The merge needs both hands, but Purple belongs to the right hand once
        // formed.  This makes its position, aim, arming, and release unambiguous.
        holdingHand='right';
        holdPurplePos={...(rightFocus()??mergeCenter)};
        holdArmed=false;
      }
      break;

    case State.HOLDING: {
      chromaticIntensity=Math.max(chromaticIntensity*0.98,0.35);

      // Purple is permanently controlled by the right hand after the merge.
      const activeFocus = rightFocus();

      if(activeFocus){
        // Purple remains at the controlling hand's index/middle fingertips.
        holdPurplePos.x += (activeFocus.x - holdPurplePos.x) * 0.25;
        holdPurplePos.y += (activeFocus.y - holdPurplePos.y) * 0.25;

      }

      const activeCrossFrames = rightCrossFrames;
      const activeReleaseFrames = rightReleaseFrames;

      const target=aimTarget(holdPurplePos);
      const dir=aimDirection(holdPurplePos,target,innerWidth,innerHeight);
      aimAngle=Math.atan2(dir.y,dir.x);
      // Arm only when fingers have clearly crossed
      if(activeCrossFrames >= 2) holdArmed = true;

      // FIRE when crossed -> uncrossed transition happens
      if(holdArmed && activeReleaseFrames >= 2){
        firePurple(now);
      }

      // Tracking loss is never a release gesture. Briefly hold, then cancel.
      if(!leftPalm&&!rightPalm){
        if(!lostSince) lostSince=now;
        if(now-lostSince>1.5) resetTechnique();
      } else lostSince=0;
      break;
    }

    case State.SHOOTING: {
      chromaticIntensity*=0.97;
      const elapsed=now-shootStart;
      if(elapsed>=BEAM_DUR){
        state=State.COOLDOWN; cooldownStart=now; purpleSpherePos=null;
      } else if(purpleSpherePos){
        const shot=arena.shots.find(s=>s.type==='purple');
        if(shot){purpleSpherePos.x=shot.x;purpleSpherePos.y=shot.y;}
      }
      break;
    }

    case State.COOLDOWN:
      chromaticIntensity*=0.9;
      if(now-cooldownStart>=COOLDOWN_DUR) state=State.IDLE;
      break;
  }
}

function firePurple(now){
  if(state!==State.HOLDING)return;
  const target=aimTarget(holdPurplePos);
  if(!cast(arena,'purple',holdPurplePos,target,innerWidth,innerHeight))return;
  const dir=aimDirection(holdPurplePos,target,innerWidth,innerHeight);
  aimAngle=Math.atan2(dir.y,dir.x);
  state=State.SHOOTING;
  shootStart=now;
  flashIntensity=1;
  shakeIntensity=1;

  shootOrigin={...holdPurplePos};
  purpleSpherePos={...holdPurplePos};



  emitImpactBurst(holdPurplePos.x,holdPurplePos.y,250);
  releaseBurst = 1;
  holdingHand=null;
  holdArmed=false;
}

// ═══════════════════════════════════════════════════════════════════════════
// PER-FRAME EMISSION
// ═══════════════════════════════════════════════════════════════════════════

function emitFrame(dt,now){
  const blueStates=[State.BLUE,State.BOTH,State.MERGING,State.CHARGING];
  const redStates=[State.RED,State.BOTH,State.MERGING,State.CHARGING];

  if(blueStates.includes(state)&&leftPalm) emitBlueVortex(leftPalm.x,leftPalm.y,6);
  if(redStates.includes(state)&&rightPalm) emitRedBurst(rightPalm.x,rightPalm.y,5);

  if((state===State.MERGING||state===State.CHARGING)&&leftPalm&&rightPalm){
    mergeCenter=mergedFocus()??mergeCenter;
    emitPurpleVortex(mergeCenter.x,mergeCenter.y, state===State.CHARGING?18:10);
  }

  if(state===State.HOLDING){
    emitHoldAura(holdPurplePos.x,holdPurplePos.y,8);
    emitPurpleVortex(holdPurplePos.x,holdPurplePos.y,5);
  }

  if(state===State.SHOOTING&&purpleSpherePos){
    emitSphereTrail(purpleSpherePos.x,purpleSpherePos.y,aimAngle,8);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// RENDER
// ═══════════════════════════════════════════════════════════════════════════

function renderFX(now){
  const W=fxC.width, H=fxC.height;
  fx.clearRect(0,0,W,H);
  drawTargets(fx,W,H);

  fx.globalCompositeOperation='lighter';

  // Blue orb
  const blueStates=[State.BLUE,State.BOTH,State.MERGING,State.CHARGING];
  if(blueStates.includes(state)&&leftPalm){
    drawBlueOrb(fx,W,H,leftPalm.x,leftPalm.y,leftSize,pulse);
    drawFingerThreads(fx,W,H,leftPalm,leftTips,'rgba(50,150,255,0.6)',pulse);
  }

  // Red orb
  const redStates=[State.RED,State.BOTH,State.MERGING,State.CHARGING];
  if(redStates.includes(state)&&rightPalm){
    drawRedOrb(fx,W,H,rightPalm.x,rightPalm.y,rightSize,pulse);
    drawFingerThreads(fx,W,H,rightPalm,rightTips,'rgba(255,80,50,0.6)',pulse);
  }

  fx.globalCompositeOperation='source-over';

  // Purple void (merging)
  if(state===State.MERGING){
    drawPurpleVoid(fx,W,H,mergeCenter.x,mergeCenter.y,20,pulse,0.6);
  }

  // Purple void (charging)
  if(state===State.CHARGING){
    const r=20+40*Math.min(1,chargeProgress);
    drawPurpleVoid(fx,W,H,mergeCenter.x,mergeCenter.y,r,pulse,Math.min(1,chargeProgress+0.3));

    if(leftPalm&&rightPalm){
      const cx=mergeCenter.x*W, cy=mergeCenter.y*H;
      fx.globalAlpha=0.3; fx.strokeStyle='rgba(50,130,255,0.5)'; fx.lineWidth=2;
      fx.beginPath(); fx.moveTo(leftPalm.x*W,leftPalm.y*H);
      fx.quadraticCurveTo(cx+15*Math.sin(pulse*6),cy-10*Math.cos(pulse*5),cx,cy);
      fx.stroke();
      fx.strokeStyle='rgba(255,60,40,0.5)';
      fx.beginPath(); fx.moveTo(rightPalm.x*W,rightPalm.y*H);
      fx.quadraticCurveTo(cx-15*Math.sin(pulse*5),cy+10*Math.cos(pulse*6),cx,cy);
      fx.stroke();
      fx.globalAlpha=1;
    }
  }

  // Purple void (HOLDING — follows hand, pulsing ominously)
  if(state===State.HOLDING){
    const r=55+5*Math.sin(pulse*4);
    drawPurpleVoid(fx,W,H,holdPurplePos.x,holdPurplePos.y,r,pulse,1);

    // Draw aim direction indicator
    drawAimLine(fx,W,H,holdPurplePos.x,holdPurplePos.y,aimAngle,pulse);

    // Energy tether from the right-hand controller to the sphere.
    const hp=rightPalm;
    if(hp){
      const sx=hp.x*W, sy=hp.y*H;
      const dx=holdPurplePos.x*W, dy=holdPurplePos.y*H;
      const mx2=(sx+dx)/2+12*Math.sin(pulse*7);
      const my2=(sy+dy)/2+12*Math.cos(pulse*6);
      fx.globalAlpha=0.4; fx.strokeStyle='rgba(160,60,255,0.7)'; fx.lineWidth=2;
      fx.beginPath(); fx.moveTo(sx,sy); fx.quadraticCurveTo(mx2,my2,dx,dy); fx.stroke();
      fx.globalAlpha=1;
    }

    // Cross-finger arm indicator ring
    const armedR = holdArmed ? 78 : 66;
    fx.globalAlpha = holdArmed ? 0.45 : 0.2;
    fx.strokeStyle = holdArmed ? 'rgba(220,140,255,0.9)' : 'rgba(140,80,180,0.6)';
    fx.lineWidth = holdArmed ? 2 : 1;
    fx.beginPath();
    fx.arc(holdPurplePos.x * W, holdPurplePos.y * H, armedR + Math.sin(pulse * 7) * 3, 0, TWO_PI);
    fx.stroke();
    fx.globalAlpha = 1;
  }

  drawArena(fx,W,H);
  drawReticle(fx,W,H);

  fx.globalCompositeOperation='lighter';
  drawParticles(fx,W,H);
  fx.globalCompositeOperation='source-over';

  // Screen flash
  if(flashIntensity>0.01 && !reduced){
    fx.globalAlpha=flashIntensity*0.5; fx.fillStyle='rgba(255,255,255,1)'; fx.fillRect(0,0,W,H);
    fx.globalAlpha=flashIntensity*0.3; fx.fillStyle='rgba(140,40,255,0.5)'; fx.fillRect(0,0,W,H);
    flashIntensity*=0.88; fx.globalAlpha=1;
  }

  // Release burst ring (single anime hit frame feel)
  if(releaseBurst > 0.01&&!reduced){
    const r = (1 - releaseBurst) * 260;
    fx.globalAlpha = releaseBurst * 0.6;
    fx.strokeStyle = 'rgba(235,210,255,0.9)';
    fx.lineWidth = 2.5;
    fx.beginPath();
    fx.arc(shootOrigin.x * W, shootOrigin.y * H, r, 0, TWO_PI);
    fx.stroke();
    fx.globalAlpha = releaseBurst * 0.28;
    fx.fillStyle = 'rgba(180,80,255,0.35)';
    fx.beginPath();
    fx.arc(shootOrigin.x * W, shootOrigin.y * H, r * 0.5, 0, TWO_PI);
    fx.fill();
    releaseBurst *= 0.82;
    fx.globalAlpha = 1;
  }

  // Screen shake
  if(shakeIntensity>0.01 && !reduced){
    const sx=(Math.random()-0.5)*shakeIntensity*25;
    const sy=(Math.random()-0.5)*shakeIntensity*25;
    fxC.style.transform=`translate(${sx}px,${sy}px)`;
    video.style.transform=`scaleX(-1) translate(${sx}px,${sy}px)`;
    distC.style.transform=`translate(${sx}px,${sy}px)`;
    shakeIntensity*=0.92;
  } else {
    fxC.style.transform=''; video.style.transform='scaleX(-1)'; distC.style.transform='';
    if(reduced){flashIntensity=shakeIntensity=releaseBurst=0;}
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// HUD
// ═══════════════════════════════════════════════════════════════════════════

function renderHUD(){
  hud.clearRect(0,0,hudC.width,hudC.height);
  const labels=[['NEUTRAL / 無限','Infinity awaits.','Show your left hand for Blue or your right hand for Red.'],['LAPSE / 蒼','Attraction.','Blue is summoned by your left hand.'],['REVERSAL / 赫','Repulsion.','Red is summoned by your right hand.'],['LIMITLESS / 無下限','Two infinities.','Touch both hands’ fingertips to converge them.'],['CONVERGENCE / 虚式','The collision.','Keep your fingertips together.'],['HOLLOW TECHNIQUE / 茈','Beyond infinity.','Hold steady. Purple is forming at your fingertips.'],['HOLLOW TECHNIQUE / 茈','Take aim.','Purple is on your right hand. Cross its fingertips, then uncross to release.'],['HOLLOW TECHNIQUE / 茈','Hollow Purple.','A destructive path through the training field.'],['RECOVERY / 無限','After the silence.','Let the energy settle.']];
  const [label,title,hint]=labels[state];
  $('phaseLabel').textContent=label;$('phaseTitle').textContent=title;
  $('phaseHint').textContent=mode==='camera'&&state===State.HOLDING?'Purple follows your right-hand fingertips. Cross them to arm it, then uncross to release.':hint;
  const energy=state===State.CHARGING?Math.round(chargeProgress*100):state===State.HOLDING?100:state===State.SHOOTING?Math.max(0,Math.round(100*(1-(performance.now()/1000-shootStart)/BEAM_DUR))):state===State.BOTH?40:state===State.BLUE||state===State.RED?20:0;
  $('energyFill').style.width=energy+'%';$('energyValue').textContent=energy+'%';
  $('trackingStatus').textContent=(leftPalm||rightPalm)?`${Number(!!leftPalm)+Number(!!rightPalm)} HAND${leftPalm&&rightPalm?'S':''} TRACKED`:'SHOW BOTH HANDS';
  $('targetStats').textContent=`${arena.stats.purple} TARGET${arena.stats.purple===1?'':'S'} CLEARED`;
  $('aimStatus').textContent=state===State.HOLDING?'RIGHT-HAND AIM · CROSS THEN UNCROSS TO RELEASE':'RIGHT-HAND AIM · WAITING FOR PURPLE';
  previousState=state;
  document.body.dataset.phase=Object.keys(State)[state].toLowerCase();
  document.body.dataset.shots=String(arena.shots.length);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN LOOP
// ═══════════════════════════════════════════════════════════════════════════

function loop(){
  if(!running) return;
  const now=performance.now()/1000,dt=Math.min(now-lastTime,.05);
  lastTime=now;pulse+=dt;
  if(mode==='camera'&&handLandmarker&&video.readyState>=2&&video.currentTime!==lastVideoTime&&now-lastDetectionAt>=1/30){
    lastDetectionAt=now;lastTrackingAt=now;
    lastVideoTime=video.currentTime;
    try { processHands(handLandmarker.detectForVideo(video,performance.now())); }
    catch { exitSession(); $('startStatus').textContent='Tracking stopped. Try practice, or reconnect your camera.'; return; }
  }
  if(mode==='camera'&&now-lastTrackingAt>.3)processHands({landmarks:[]});
  updateState(dt,now);stepArena(arena,dt,innerWidth,innerHeight);emissionClock+=dt;
  if(emissionClock>=1/(reduced?30:60)){emitFrame(dt,now);emissionClock=0;}
  updateParticles(dt);
  renderDistortion(fxC.width,fxC.height);renderFX(now);renderHUD();
  raf=requestAnimationFrame(loop);
}
function resetTechnique(){
  arena=createArena();
  state=State.IDLE;particles=[];chargeProgress=0;holdArmed=false;holdingHand=null;
  flashIntensity=shakeIntensity=chromaticIntensity=releaseBurst=0;
  purpleSpherePos=null;lostSince=0;
  processHands({landmarks:[]});
}
function begin(){
  resetTechnique();mode='camera';running=true;previousState=-1;lastVideoTime=-1;
  lastTrackingAt=performance.now()/1000;
  document.body.classList.add('camera');
  $('startup').hidden=true;$('sessionUI').hidden=false;$('exitBtn').hidden=false;
  $('modeLabel').textContent='LIVE TECHNIQUE';
  lastTime=performance.now()/1000;cancelAnimationFrame(raf);loop();
}
function exitSession(){
  attempt++;running=false;cancelAnimationFrame(raf);stopCamera();resetTechnique();
  document.body.classList.remove('camera');
  [dist,fx,hud].forEach(c=>c.clearRect(0,0,innerWidth,innerHeight));
  fxC.style.transform=distC.style.transform='';video.style.transform='scaleX(-1)';
  $('startup').hidden=false;$('sessionUI').hidden=true;$('exitBtn').hidden=true;
  $('startBtn').disabled=false;$('startStatus').textContent='Camera mode processes video in this browser; it does not record or upload it.';
  $('startBtn').focus();
}
$('startBtn').onclick=async()=>{
  const token=++attempt;let pendingStream=null,pendingTracker=null;
  const watchdog=setTimeout(()=>{if(token===attempt){attempt++;stopCamera();$('startBtn').disabled=false;$('startStatus').textContent='Camera setup timed out. Retry, or enter practice.';}},45000);
  $('startBtn').disabled=true;$('startStatus').textContent='Allow camera access. Your video stays in this browser.';
  try{
    if(!navigator.mediaDevices?.getUserMedia) throw new Error('Camera needs localhost or HTTPS. Practice is always available.');
    pendingStream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'},audio:false});
    if(token!==attempt){pendingStream.getTracks().forEach(t=>t.stop());return;}
    video.srcObject=pendingStream;await video.play();
    if(token!==attempt){pendingStream.getTracks().forEach(t=>t.stop());return;}
    $('startStatus').textContent='Loading hand tracking. First launch needs an internet connection…';
    pendingTracker=await initMediaPipe();
    if(token!==attempt){pendingTracker.close();pendingStream.getTracks().forEach(t=>t.stop());return;}
    handLandmarker=pendingTracker;begin();
  }catch(e){
    pendingStream?.getTracks().forEach(t=>t.stop());pendingTracker?.close();
    if(token===attempt){video.srcObject=null;$('startStatus').textContent=e.name==='NotAllowedError'?'Camera permission declined. You can still enter practice.':e.name==='NotFoundError'?'No camera found. Enter practice to wield the technique.':'Could not start camera tracking. Check your camera and connection, then retry or use practice.';}
  }finally{clearTimeout(watchdog);if(token===attempt)$('startBtn').disabled=false;}
};
$('exitBtn').onclick=exitSession;
window.addEventListener('pagehide',()=>{attempt++;running=false;cancelAnimationFrame(raf);stopCamera();});
document.addEventListener('visibilitychange',()=>{lastTime=performance.now()/1000;if(document.hidden&&running)resetTechnique();});

function aimTarget(origin){
  const dir=rightFingerDir;
  if(!dir)return {x:origin.x,y:Math.max(.05,origin.y-.35)};
  const unit=aimDirection({x:0,y:0},dir,innerWidth,innerHeight);
  // Project a ray to the viewport edge without distorting its direction.
  const dx=unit.x/innerWidth,dy=unit.y/innerHeight;
  const tx=Math.abs(dx)<1e-8?Infinity:((dx>0?.95:.05)-origin.x)/dx;
  const ty=Math.abs(dy)<1e-8?Infinity:((dy>0?.82:.2)-origin.y)/dy;
  const travel=Math.max(30,Math.min(...[tx,ty,420].filter(v=>v>0)));
  return {x:origin.x+dx*travel,y:origin.y+dy*travel};
}
function drawTargets(ctx,W,H){
  for(const t of arena.targets){
    if(t.dead)continue;
    const x=t.x*W,y=t.y*H,r=9+Math.sin(pulse*.7+t.id)*1.5;
    ctx.save();ctx.translate(x,y);ctx.rotate(Math.PI/4+Math.sin(pulse*.35+t.id)*.15);
    ctx.strokeStyle=t.hit?t.tint:'rgba(183,191,221,.55)';ctx.fillStyle=t.hit?t.tint+'22':'rgba(99,108,139,.12)';
    ctx.lineWidth=t.hit?2:1;ctx.fillRect(-r,-r,2*r,2*r);ctx.strokeRect(-r,-r,2*r,2*r);ctx.restore();
    ctx.fillStyle='rgba(190,196,218,.35)';ctx.font='8px monospace';ctx.textAlign='center';ctx.fillText(String(t.id+1).padStart(2,'0'),x,y+25);
  }
}
function drawArena(ctx,W,H){
  for(const s of arena.shots){
    const cfg=TECHNIQUES[s.type],x=s.x*W,y=s.y*H;
    ctx.save();ctx.globalCompositeOperation='lighter';ctx.strokeStyle=cfg.color;ctx.lineWidth=s.type==='purple'?12:3;
    if(s.trail.length){ctx.globalAlpha=.25;ctx.beginPath();ctx.moveTo(s.trail[0].x*W,s.trail[0].y*H);for(const p of s.trail)ctx.lineTo(p.x*W,p.y*H);ctx.lineTo(x,y);ctx.stroke();}
    ctx.globalAlpha=1;ctx.globalCompositeOperation='source-over';
    if(s.type==='blue'){
      drawBlueOrb(ctx,W,H,s.x,s.y,110/W,pulse);
      if(s.phase==='field'){
        const fade=Math.min(1,(cfg.duration-s.age)*2);ctx.strokeStyle=cfg.color;ctx.globalAlpha=.35*fade;
        for(let i=0;i<3;i++){const r=cfg.radius*(1-((s.age*.6+i/3)%1));ctx.beginPath();ctx.arc(x,y,r,0,TWO_PI);ctx.stroke();}
      }
    }else if(s.type==='red'){
      if(s.phase==='travel')drawRedOrb(ctx,W,H,s.x,s.y,100/W,pulse);
      else{const r=cfg.radius*Math.min(1,s.age/.35);ctx.globalAlpha=Math.max(0,1-s.age/cfg.duration);ctx.strokeStyle=cfg.color;ctx.lineWidth=4;ctx.beginPath();ctx.arc(x,y,r,0,TWO_PI);ctx.stroke();ctx.lineWidth=1;ctx.beginPath();ctx.arc(x,y,r*.72,0,TWO_PI);ctx.stroke();}
    }else{
      drawTravelingSphere(ctx,W,H,s.x,s.y,52,pulse,Math.min(1,s.age*2),Math.atan2(s.dir.y,s.dir.x));
      if(!reduced)drawSpeedLines(ctx,W,H,Math.atan2(s.dir.y,s.dir.x),Math.max(0,.7-s.age),pulse);
    }
    ctx.restore();
  }
}
function drawReticle(ctx,W,H){
  if(state!==State.HOLDING)return;
  const origin=holdPurplePos;
  const target=aimTarget(origin),color=TECHNIQUES.purple.color;
  const x=target.x*W,y=target.y*H;
  ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.globalAlpha=.55;ctx.lineWidth=1;ctx.setLineDash([4,8]);
  ctx.beginPath();ctx.moveTo(origin.x*W,origin.y*H);ctx.lineTo(x,y);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=.9;
  ctx.beginPath();ctx.arc(x,y,14,0,TWO_PI);ctx.moveTo(x-23,y);ctx.lineTo(x-8,y);ctx.moveTo(x+8,y);ctx.lineTo(x+23,y);ctx.moveTo(x,y-23);ctx.lineTo(x,y-8);ctx.moveTo(x,y+8);ctx.lineTo(x,y+23);ctx.stroke();
  ctx.font='9px monospace';ctx.textAlign='center';ctx.fillText('PURPLE TRAJECTORY',x,y+34);ctx.restore();
}
