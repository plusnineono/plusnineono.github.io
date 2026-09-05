// New core vs. old v5.4 engine, arbitrated by the new core's rules.
const N=15, cell=(x,y)=>y*N+x;
const nameOf = c => `${'ABCDEFGHJKLMNOP'[c%N]}${N-((c/N)|0)}`;
const src = await Deno.readTextFile(new URL('./core.js', import.meta.url));
const RenjuCore = new Function(src + '\nreturn RenjuCore;')();
await import('./baseline_v5.js');
const OLD = globalThis.__OLD;
globalThis.__OLD_TIME = Number(Deno.args[2] ?? Deno.args[1] ?? 1500);

const TIME = Number(Deno.args[1] ?? 1500);
const OLD_TIME = Number(Deno.args[2] ?? TIME);
const GAMES = Number(Deno.args[0] ?? 6);

function oldMove(moves, color, timeMs){
  const g = new OLD.Game();
  for(let i=0;i<moves.length;i++){
    const c=moves[i], col=(i%2===0)?1:2;
    g.set(c%N,(c/N)|0,col); g.zhash ^= g.zTable[c][col-1];
    g.history.push({x:c%N,y:(c/N)|0,color:col}); g.lastMove=g.history[g.history.length-1];
  }
  g.turn = color;
  // the old engine reads its time budget from a stubbed <select>
  const res = OLD.searchBestMove(g, color, OLD.MAX_SEARCH_DEPTH);
  return res.move ? res.move.y*N+res.move.x : -1;
}

// Both engines are deterministic, so every game would otherwise be identical.
// Each game starts from a different short random opening near the centre.
function mulberry(a){ return () => { a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function randomOpening(rng, core){
  const seq=[cell(7,7)];
  core.setMoves(seq);
  for(let k=0;k<3;k++){
    const colour = seq.length%2===0 ? 1 : 2;
    const near=[];
    for(let c=0;c<225;c++){
      const x=c%N, y=(c/N)|0;
      if(Math.max(Math.abs(x-7),Math.abs(y-7))>3) continue;
      if(core.board()[c]!==0) continue;
      if(!core.isLegal(c,colour)) continue;
      near.push(c);
    }
    seq.push(near[Math.floor(rng()*near.length)]);
    core.setMoves(seq);
  }
  return seq;
}

let newWins=0, oldWins=0, draws=0;
for(let game=0; game<GAMES; game++){
  const newIsBlack = game % 2 === 0;
  const core = RenjuCore();
  const rng = mulberry(0x9e37 + game * 7919);
  const moves = randomOpening(rng, core);
  let result=null;
  for(let ply=moves.length; ply<225; ply++){
    const color = ply%2===0 ? 1 : 2;
    const newToMove = (color===1) === newIsBlack;
    let c;
    if(newToMove){
      core.setMoves(moves);
      c = core.think({side: color, timeMs: TIME, maxDepth: 20}).cell;
    } else {
      c = oldMove(moves, color, OLD_TIME);
    }
    core.setMoves(moves);
    if(c<0 || !core.isLegal(c, color)){
      // fall back to any legal move (old engine may propose a forbidden point)
      const b=core.board(); let alt=-1;
      for(let k=0;k<225;k++) if(b[k]===0 && core.isLegal(k,color)){ alt=k; break; }
      if(c>=0 && !core.isLegal(c,color) && color===1){ result = {winner:2, why:`black played forbidden ${nameOf(c)}`}; break; }
      c = alt;
      if(c<0){ result={winner:0, why:'no legal move'}; break; }
    }
    moves.push(c);
    core.setMoves(moves);
    if(core.winner()){ result={winner:core.winner(), why:'five'}; break; }
    if(moves.length>=225){ result={winner:0, why:'full board'}; break; }
  }
  const w = result?.winner ?? 0;
  const newWon = (w===1 && newIsBlack) || (w===2 && !newIsBlack);
  const oldWon = w!==0 && !newWon;
  if(newWon) newWins++; else if(oldWon) oldWins++; else draws++;
  console.log(`game ${game+1}: opening ${moves.slice(0,4).map(nameOf).join(' ')} new=${newIsBlack?'black':'white'} -> ${newWon?'NEW wins':oldWon?'old wins':'draw'} (${result?.why}, ${moves.length} moves)`);
}
console.log(`\nNEW ${newWins} - OLD ${oldWins} - draw ${draws}  (new ${TIME} ms, old ${OLD_TIME} ms per move)`);
