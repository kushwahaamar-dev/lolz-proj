// Gameplay interpretation of attraction / repulsion / a destructive swept path.
// Positions are normalized; distances, velocities and collision radii use pixels.
export const TECHNIQUES = Object.freeze({
  blue: {speed:760, radius:165, duration:3.2, color:'#69b5ff'},
  red: {speed:1100, radius:180, duration:.7, color:'#ff706e'},
  purple: {speed:900, radius:52, duration:2.5, color:'#c099ff'},
});
export function createArena(){
  return {time:0,shots:[],nextId:1,lastCast:{blue:-100,red:-100,purple:-100},stats:{blue:0,red:0,purple:0},
    targets:Array.from({length:18},(_,i)=>{const x=.16+(i%6)*.136,y=.31+Math.floor(i/6)*.16;return {id:i,x,y,homeX:x,homeY:y,vx:0,vy:0,dead:0,hit:0,tint:'#abb0c6'};})};
}
export function aimDirection(origin,target,W,H){
  const x=(target.x-origin.x)*W,y=(target.y-origin.y)*H,len=Math.hypot(x,y);
  return len<1?{x:0,y:-1}:{x:x/len,y:y/len};
}
export function segmentDistance(p,a,b,W,H){
  const dx=(b.x-a.x)*W,dy=(b.y-a.y)*H,px=(p.x-a.x)*W,py=(p.y-a.y)*H;
  const t=Math.max(0,Math.min(1,(px*dx+py*dy)/(dx*dx+dy*dy||1)));
  return Math.hypot(px-t*dx,py-t*dy);
}
export function cast(arena,type,origin,target,W,H){
  if(!TECHNIQUES[type]||![origin.x,origin.y,target.x,target.y,W,H].every(Number.isFinite)||W<=0||H<=0)return null;
  if(arena.shots.length>=8||arena.time-arena.lastCast[type]<.45)return null;
  const dir=aimDirection(origin,target,W,H),distance=Math.hypot((target.x-origin.x)*W,(target.y-origin.y)*H);
  const shot={id:arena.nextId++,type,x:origin.x,y:origin.y,start:{...origin},target:{...target},dir,travel:0,distance:Math.max(1,distance),age:0,phase:'travel',touched:new Set(),trail:[]};
  arena.shots.push(shot);arena.lastCast[type]=arena.time;return shot;
}
function touch(arena,shot,t){
  if(!shot.touched.has(t.id)){shot.touched.add(t.id);arena.stats[shot.type]++;}
  t.hit=.6;t.tint=TECHNIQUES[shot.type].color;
}
export function stepArena(arena,delta,W,H){
  if(!Number.isFinite(delta)||delta<=0||W<=0||H<=0)return;
  // Substeps prevent fast Red/Purple from tunnelling through targets.
  let remaining=Math.min(delta,.1);
  while(remaining>1e-8){const dt=Math.min(remaining,1/120);remaining-=dt;arena.time+=dt;
    for(const s of arena.shots){
      const cfg=TECHNIQUES[s.type],old={x:s.x,y:s.y};s.age+=dt;
      if(s.phase==='travel'){
        const move=s.type==='purple'?cfg.speed*dt:Math.min(cfg.speed*dt,Math.max(0,s.distance-s.travel));
        s.x+=s.dir.x*move/W;s.y+=s.dir.y*move/H;s.travel+=move;
        if(!s.trail.length||Math.hypot((s.x-s.trail.at(-1).x)*W,(s.y-s.trail.at(-1).y)*H)>12){s.trail.push({...old});if(s.trail.length>24)s.trail.shift();}
        if(s.type!=='purple'&&s.travel>=s.distance-1e-6){s.phase='field';s.age=0;}
        if(s.type==='purple')for(const t of arena.targets){if(!t.dead&&segmentDistance(t,old,s,W,H)<cfg.radius+9){touch(arena,s,t);t.dead=5;t.vx=t.vy=0;}}
      }
      if(s.phase==='field')for(const t of arena.targets){
        if(t.dead)continue;
        const dx=(s.x-t.x)*W,dy=(s.y-t.y)*H,d=Math.hypot(dx,dy);
        if(d>cfg.radius)continue;
        if(s.type==='blue'){
          touch(arena,s,t);const force=1300*Math.min(1,d/24)*(1-d/(cfg.radius+1));
          t.vx+=dx/(d||1)*force*dt;t.vy+=dy/(d||1)*force*dt;
        }else if(!s.touched.has(t.id)){
          touch(arena,s,t);const force=680*(1-d/(cfg.radius+1))+160;
          t.vx-=(d?dx/d:Math.cos(t.id))*force;t.vy-=(d?dy/d:Math.sin(t.id))*force;
        }
      }
    }
    arena.shots=arena.shots.filter(s=>s.type==='purple'?s.age<TECHNIQUES.purple.duration&&s.x>-.3&&s.x<1.3&&s.y>-.3&&s.y<1.3:s.phase==='travel'||s.age<TECHNIQUES[s.type].duration);
    for(const t of arena.targets){
      if(t.dead){t.dead=Math.max(0,t.dead-dt);if(!t.dead){t.x=t.homeX;t.y=t.homeY;}continue;}
      t.x+=t.vx*dt/W;t.y+=t.vy*dt/H;t.vx*=Math.exp(-2.8*dt);t.vy*=Math.exp(-2.8*dt);t.hit=Math.max(0,t.hit-dt);
      if(t.x<.04||t.x>.96){t.x=Math.max(.04,Math.min(.96,t.x));t.vx*=-.35;}
      if(t.y<.22||t.y>.8){t.y=Math.max(.22,Math.min(.8,t.y));t.vy*=-.35;}
    }
  }
}
