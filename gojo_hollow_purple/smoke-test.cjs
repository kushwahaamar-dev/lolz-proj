// CPU-only regression checks; no camera permission, network, or packages needed.
const {createArena,cast,stepArena,aimDirection,TECHNIQUES}=require('./combat.mjs');
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
const ctx=new Proxy({}, {get:(_,key)=>key==='createRadialGradient'||key==='createLinearGradient'?()=>({addColorStop(){}}):()=>{}});
const element=id=>{if(!elements.has(id))elements.set(id,{id,style:{},dataset:{},hidden:false,srcObject:null,videoWidth:1280,videoHeight:720,getContext:()=>ctx,setAttribute(){},focus(){},addEventListener(){},getTracks:()=>[]});return elements.get(id)};
const box={createArena,cast,stepArena,aimDirection,TECHNIQUES,console,Math,performance:{now:()=>1000},innerWidth:1280,innerHeight:720,matchMedia:()=>({matches:false}),requestAnimationFrame:()=>1,cancelAnimationFrame(){},setTimeout,clearTimeout,navigator:{},document:{getElementById:element,body:{classList:{add(){},toggle(){},remove(){}},dataset:{}},addEventListener(){}},window:{innerWidth:1280,innerHeight:720,addEventListener(){}}};
vm.createContext(box);vm.runInContext(fs.readFileSync(__dirname+'/experience.js','utf8').replace(/^import .*;\n/,''),box);
function check(name,code){assert.equal(vm.runInContext(code,box),true,name);console.log('PASS',name)}
check('hands summon Blue and Red before Purple can form',`(()=>{begin('camera');leftPalm={x:.4,y:.5};rightPalm=null;leftSign=true;rightSign=false;updateState(.02,1);if(state!==State.BLUE)return false;rightPalm={x:.5,y:.5};rightSign=true;updateState(.02,1.02);return state===State.BOTH&&arena.shots.length===0})()`);
check('Purple merges at fingertips even when palms stay apart',`(()=>{resetTechnique();mode='camera';state=State.BOTH;leftPalm={x:.2,y:.5};rightPalm={x:.8,y:.5};leftTips=[{x:.45,y:.5},{x:.49,y:.5},{x:.5,y:.5}];rightTips=[{x:.55,y:.5},{x:.5,y:.5},{x:.51,y:.5}];leftSign=rightSign=true;updateState(.02,1);return state===State.MERGING&&Math.abs(mergeCenter.x-.5)<.02&&Math.abs(mergeCenter.y-.5)<.02})()`);
check('tracking loss cancels without firing',`(()=>{resetTechnique();mode='camera';state=State.HOLDING;holdArmed=true;updateState(.02,10);if(state!==State.HOLDING)return false;updateState(.02,12);return state===State.IDLE})()`);
check('missing landmarks reset gesture debounce',`(()=>{leftCrossFrames=rightReleaseFrames=6;processHands({landmarks:[]});return leftCrossFrames===0&&rightReleaseFrames===0})()`);
check('tracker Right label controls the user right hand',`(()=>{const hand=x=>Array.from({length:21},()=>({x,y:.5}));processHands({landmarks:[hand(.2),hand(.8)],handedness:[[{categoryName:'Right'}],[{categoryName:'Left'}]]});return rightPalm!==null&&leftPalm!==null&&rightPalm.x>leftPalm.x})()`);
check('camera cover coordinates mirror the video',`(()=>{const p=screenLandmark({x:.2,y:.3});return Math.abs(p.x-.8)<1e-9&&Math.abs(p.y-.3)<1e-9})()`);
check('portrait crop keeps center aligned',`(()=>{innerWidth=390;innerHeight=844;const p=screenLandmark({x:.5,y:.5});innerWidth=1280;innerHeight=720;return Math.abs(p.x-.5)<1e-9&&Math.abs(p.y-.5)<1e-9})()`);
check('camera stops all tracks on exit',`(()=>{let stopped=0;video.srcObject={getTracks:()=>[{stop(){stopped++}},{stop(){stopped++}}]};exitSession();return stopped===2&&video.srcObject===null&&!running})()`);
check('particle count has a hard upper bound',`(()=>{particles=[];for(let i=0;i<2000;i++)spawn(.5,.5,0,0,1,1,1,1,1);return particles.length===1400})()`);
check('Purple releases only after a right-hand crossed then uncrossed gesture',`(()=>{resetTechnique();begin('camera');state=State.HOLDING;holdPurplePos={x:.5,y:.5};holdingHand='right';rightPalm={x:.5,y:.5};rightFingerDir={x:1,y:0};rightCrossFrames=2;rightReleaseFrames=0;updateState(.02,1);if(!holdArmed||state!==State.HOLDING)return false;rightCrossFrames=0;rightReleaseFrames=2;updateState(.02,1.02);return state===State.SHOOTING&&arena.shots.length===1&&arena.shots[0].type==='purple'})()`);
check('Purple always locks to the right hand after charging',`(()=>{resetTechnique();mode='camera';state=State.CHARGING;chargeProgress=.99;mergeCenter={x:.5,y:.5};leftPalm={x:.2,y:.5};rightPalm={x:.8,y:.5};leftTips=[{x:.2,y:.5},{x:.3,y:.5},{x:.32,y:.5}];rightTips=[{x:.7,y:.5},{x:.78,y:.5},{x:.8,y:.5}];leftSign=rightSign=true;leftCrossFrames=4;rightCrossFrames=0;updateState(.02,1);return state===State.HOLDING&&holdingHand==='right'&&holdPurplePos.x>.75})()`);
check('held Purple frame renders without stopping the animation loop',`(()=>{resetTechnique();mode='camera';state=State.HOLDING;holdingHand='right';holdPurplePos={x:.5,y:.5};leftPalm={x:.2,y:.5};leftTips=[{x:.4,y:.5},{x:.49,y:.5},{x:.5,y:.5}];rightPalm={x:.8,y:.5};rightTips=[{x:.6,y:.5},{x:.5,y:.5},{x:.51,y:.5}];rightFingerDir={x:1,y:0};leftSign=rightSign=true;try{renderFX(2);return true;}catch{return false;}})()`);
check('application exposes no keyboard casting action',`(()=>typeof action==='undefined'&&typeof castTechnique==='undefined'&&typeof canCast==='undefined')()`);

check('Blue and Red never create projectiles',`(()=>{begin('camera');leftPalm={x:.5,y:.5};rightPalm={x:.6,y:.5};leftSign=rightSign=true;for(let i=0;i<6;i++){leftCrossFrames=rightCrossFrames=6;leftReleaseFrames=rightReleaseFrames=6;updateState(.02,1+i*.02)}return arena.shots.length===0})()`);
check('reset clears projectiles and restores targets',`(()=>{resetTechnique();return arena.shots.length===0&&arena.targets.every(t=>!t.dead)&&arena.stats.blue===0})()`);
for(const type of ['blue','red','purple']){
 const a=createArena(),o={x:.5,y:.5};
 for(const t of [{x:.8,y:.2},{x:.2,y:.8},{x:.5,y:.1}]){
  a.time+=1;const shot=cast(a,type,o,t,1280,720);assert.ok(shot);
  const d=aimDirection(o,t,1280,720);assert.ok(Math.abs(shot.dir.x-d.x)<1e-9&&Math.abs(shot.dir.y-d.y)<1e-9);
 }
 console.log('PASS',type+' aim in multiple directions');
}
function physics(type){const a=createArena();a.targets=[{id:0,x:.55,y:.5,homeX:.55,homeY:.5,vx:0,vy:0,dead:0,hit:0}];cast(a,type,{x:.3,y:.5},{x:.5,y:.5},1000,700);for(let i=0;i<100;i++)stepArena(a,.01,1000,700);return a;}
assert.ok(physics('blue').targets[0].x<.55);console.log('PASS Blue attracts target');
assert.ok(physics('red').targets[0].x>.55);console.log('PASS Red repels target');
assert.ok(physics('purple').targets[0].dead>0);console.log('PASS Purple swept collision clears target');
const limited=createArena();assert.ok(cast(limited,'red',{x:0,y:0},{x:1,y:1},100,100));assert.equal(cast(limited,'red',{x:0,y:0},{x:1,y:1},100,100),null);console.log('PASS rapid fire cooldown');
assert.equal(cast(limited,'invalid',{x:0,y:0},{x:1,y:1},100,100),null);console.log('PASS invalid technique rejected');
for(let i=0;i<1000;i++)stepArena(limited,.016,1280,720);assert.equal(limited.shots.length,0);console.log('PASS expired projectiles cleaned up');
