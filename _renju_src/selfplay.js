// Generate labelled positions for tune_eval.js by playing the engine against
// itself.
//
//   deno run --allow-read --allow-write _renju_src/selfplay.js [games] [ms] [out] [seed]
//   deno run --allow-read --allow-write _renju_src/selfplay.js 1000 150 data.txt
//
// Each line of the output is:  <label> <16 feature values>
// where the label is 1 if the side to move went on to win, 0 if it lost, 0.5
// for a draw, and the features are those of the position from that side's point
// of view. That is all the tuner needs, because the evaluation is linear in its
// weights (see features() in core.js).
//
// Appends, so several runs accumulate into one file, and runs with different
// seeds can go in parallel into different files that you then concatenate.
// Roughly 20 usable positions per game: a few thousand games is far more than
// the 16 weights need, and even 200 games is enough to see movement.

const CORE = await Deno.readTextFile(new URL('./core.js', import.meta.url));
const RenjuCore = new Function(CORE + '\nreturn RenjuCore;')();

const GAMES = Number(Deno.args[0] ?? 200);
const MS = Number(Deno.args[1] ?? 150);
const OUT = Deno.args[2] ?? 'renju_data.txt';
const SEED = Number(Deno.args[3] ?? 1);

const N = 15, CENTRE = 112;
const BLACK = 1, WHITE = 2;
const engine = RenjuCore(), judge = RenjuCore();

function mulberry(a){
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/** A different short opening per game, else every game is the same one. */
function opening(rng){
  const seq = [CENTRE];
  judge.setMoves(seq);
  for(let k = 0; k < 3; k++){
    const colour = seq.length % 2 === 0 ? BLACK : WHITE;
    const near = [];
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

const lines = [];
let games = 0, positions = 0, blackWins = 0, whiteWins = 0, draws = 0;
const t0 = Date.now();

for(let g = 0; g < GAMES; g++){
  const rng = mulberry(SEED * 1000003 + g * 7919);
  const moves = opening(rng);
  const pending = [];                       // {side, features} awaiting the result
  let result = 0;                           // BLACK, WHITE, or 0 for a draw

  for(let ply = moves.length; ply < 225; ply++){
    const side = ply % 2 === 0 ? BLACK : WHITE;
    engine.setMoves(moves);
    const r = engine.think({ side, timeMs: MS, maxDepth: 20 });
    if(r.cell < 0) break;

    // Record quiet, undecided positions only: a position the search has already
    // resolved carries no information about the weights.
    const decided = Math.abs(r.score) > 900000;
    if(ply >= 6 && !decided) pending.push({ side, f: Array.from(engine.features(side)) });

    // Vary the play early so the data covers more than one opening plan.
    let cell = r.cell;
    if(ply < 16 && rng() < 0.3 && r.candidates && r.candidates.length > 1){
      const near = r.candidates.filter(m => Math.abs(m.score - r.score) < 400);
      if(near.length > 1) cell = near[Math.floor(rng() * near.length)].cell;
    }

    judge.setMoves(moves);
    if(!judge.isLegal(cell, side)) cell = r.cell;
    if(!judge.isLegal(cell, side)) break;
    moves.push(cell);
    judge.setMoves(moves);
    if(judge.winner()){ result = judge.winner(); break; }
  }

  for(const p of pending){
    const label = result === 0 ? 0.5 : (result === p.side ? 1 : 0);
    lines.push(label + ' ' + p.f.map(v => Math.round(v * 100) / 100).join(' '));
  }
  positions += pending.length;
  games++;
  if(result === BLACK) blackWins++; else if(result === WHITE) whiteWins++; else draws++;

  if(games % 10 === 0 || g === GAMES - 1){
    const mins = (Date.now() - t0) / 60000;
    const rate = games / mins;
    console.log(`${games}/${GAMES} games, ${positions} positions, ` +
      `black ${blackWins} white ${whiteWins} draw ${draws}, ` +
      `${rate.toFixed(1)} games/min, ~${((GAMES - games) / rate).toFixed(0)} min left`);
    await Deno.writeTextFile(OUT, lines.join('\n') + '\n', { append: true });
    lines.length = 0;
  }
}
if(lines.length) await Deno.writeTextFile(OUT, lines.join('\n') + '\n', { append: true });
console.log(`\ndone: ${positions} positions appended to ${OUT}`);
console.log(`next:  deno run --allow-read --allow-write _renju_src/tune_eval.js ${OUT}`);
