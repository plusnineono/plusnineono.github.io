// Play two builds of the engine against each other.
//
//   deno run --allow-read _renju_src/ab.js <games> <ms> <coreA.js> <coreB.js>
//
// Take the two builds from *files*. An earlier version of this harness set
// tuning flags on globalThis before constructing each engine, which meant a
// config of {} inherited whatever the previous build had set - the two sides
// were the same engine, and every "improvement" it reported was noise. Reading
// two files cannot go wrong that way.
//
// Calibrate before believing anything: the same file against itself scores 4-2
// over 6 games often enough. Forty games is about the minimum for a real
// signal, and a change that only helps at 300 ms may well hurt at 2 s, so
// measure at the time control people actually play at.
const N = 15, cell = (x, y) => y * N + x;
const GAMES = Number(Deno.args[0] ?? 40), MS = Number(Deno.args[1] ?? 600);
const pathA = Deno.args[2], pathB = Deno.args[3];
const mk = async p => new Function(await Deno.readTextFile(p) + '\nreturn RenjuCore;')()();
const A = await mk(pathA), B = await mk(pathB), judge = await mk(pathB);

function mulberry(a){ return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/** A different short opening per game, or every game would be the same one. */
function opening(rng){
  const seq = [cell(7, 7)];
  judge.setMoves(seq);
  for(let k = 0; k < 3; k++){
    const colour = seq.length % 2 === 0 ? 1 : 2, near = [];
    for(let c = 0; c < 225; c++){
      const x = c % N, y = (c / N) | 0;
      if(Math.max(Math.abs(x - 7), Math.abs(y - 7)) > 3) continue;
      if(judge.board()[c] !== 0 || !judge.isLegal(c, colour)) continue;
      near.push(c);
    }
    seq.push(near[Math.floor(rng() * near.length)]);
    judge.setMoves(seq);
  }
  return seq;
}

let aw = 0, bw = 0, dr = 0;
for(let g = 0; g < GAMES; g++){
  const aIsBlack = g % 2 === 0;                  // each side plays both colours
  const moves = opening(mulberry(0x9e37 + g * 7919));
  let res = 0;
  for(let ply = moves.length; ply < 225; ply++){
    const colour = ply % 2 === 0 ? 1 : 2;
    const eng = ((colour === 1) === aIsBlack) ? A : B;
    eng.setMoves(moves);
    const c = eng.think({ side: colour, timeMs: MS, maxDepth: 24 }).cell;
    judge.setMoves(moves);
    if(c < 0 || !judge.isLegal(c, colour)){ res = colour === 1 ? 2 : 1; break; }
    moves.push(c);
    judge.setMoves(moves);
    if(judge.winner()){ res = judge.winner(); break; }
  }
  const awon = (res === 1 && aIsBlack) || (res === 2 && !aIsBlack);
  if(res === 0) dr++; else if(awon) aw++; else bw++;
}
const pct = (100 * (aw + dr / 2) / GAMES).toFixed(1);
console.log(`${pathA}  ${aw} - ${bw} - ${dr}  ${pathB}   (${GAMES} games, ${MS} ms, A scores ${pct}%)`);
