// Regression tests for the renju core. Run with:
//   /Applications/quarto/bin/tools/aarch64/deno run --allow-read test.js
const src = await Deno.readTextFile(new URL('./core.js', import.meta.url));
const RenjuCore = new Function(src + '\nreturn RenjuCore;')();

const N = 15;
let pass = 0, fail = 0;
function ok(cond, msg){ if(cond){ pass++; } else { fail++; console.log('  FAIL: ' + msg); } }
function eq(a, b, msg){ ok(a === b, `${msg} (got ${a}, want ${b})`); }

const cell = (x, y) => y * N + x;
const nameOf = c => `${'ABCDEFGHJKLMNOP'[c % N]}${N - ((c / N) | 0)}`;

// Build a position from an ASCII diagram: '.' empty, 'X' black, 'O' white.
function fromDiagram(rows){
  const black = [], white = [];
  rows.forEach((row, y) => {
    [...row.replace(/\s/g, '')].forEach((ch, x) => {
      if(ch === 'X') black.push(cell(x, y));
      else if(ch === 'O') white.push(cell(x, y));
    });
  });
  const core = RenjuCore();
  // Interleave so the move list stays legal-ish; extra stones of one colour are
  // appended at the end (setMoves alternates, so we place directly instead).
  return { core, black, white };
}
function setup(rows){
  const { core, black, white } = fromDiagram(rows);
  const seq = [];
  const n = Math.max(black.length, white.length);
  for(let i = 0; i < n; i++){
    if(i < black.length) seq.push(black[i]);
    if(i < white.length) seq.push(white[i]);
  }
  core.setMoves(seq);
  return core;
}
function show(core){
  const b = core.board();
  let out = '';
  for(let y = 0; y < N; y++){
    let row = '';
    for(let x = 0; x < N; x++) row += b[cell(x, y)] === 1 ? 'X' : b[cell(x, y)] === 2 ? 'O' : '.';
    out += row + '\n';
  }
  return out;
}

console.log('--- shape classification ---');
{
  const core = RenjuCore();
  const B = core.BLACK, W = core.WHITE;
  // Black stones at (5,7),(6,7),(7,7): playing (8,7) makes an open four.
  core.setMoves([cell(5,7), cell(0,0), cell(6,7), cell(1,0), cell(7,7), cell(2,0)]);
  const sh = core.shapeAt(cell(8,7), B);
  eq(sh[0], core.P_F4, 'X X X _ -> open four at the end');
  const sh2 = core.shapeAt(cell(4,7), B);
  eq(sh2[0], core.P_F4, 'other end also open four');
  const sh3 = core.shapeAt(cell(9,7), B);
  eq(sh3[0], core.P_B4, 'gap point X X X _ X is a simple four');
}
{
  const core = RenjuCore();
  // Open three for white: _ O O O _  at row 7, x=5..7 -> both ends are open four
  core.setMoves([cell(0,0), cell(5,7), cell(1,0), cell(6,7), cell(2,0), cell(7,7)]);
  eq(core.shapeAt(cell(8,7), core.WHITE)[0], core.P_F4, 'white open three -> open four point');
}
{
  const core = RenjuCore();
  // Blocked three  W X X X _   : the open end gives a simple four, not open.
  core.setMoves([cell(5,7), cell(4,7), cell(6,7), cell(0,0), cell(7,7), cell(1,0)]);
  eq(core.shapeAt(cell(8,7), core.BLACK)[0], core.P_B4, 'blocked three -> simple four');
  eq(core.shapeAt(cell(9,7), core.BLACK)[0], core.P_B4, 'gap after a blocked three is still a four');
}
{
  const core = RenjuCore();
  // Four X X X X _ -> five point
  core.setMoves([cell(4,7), cell(0,0), cell(5,7), cell(1,0), cell(6,7), cell(2,0), cell(7,7), cell(3,0)]);
  eq(core.shapeAt(cell(8,7), core.BLACK)[0], core.P_F5, 'four -> five point');
  eq(core.shapeAt(cell(3,7), core.BLACK)[0], core.P_F5, 'four -> five point (other end)');
  ok(core.isWinningPoint(cell(8,7), core.BLACK), 'winning point recognised');
}
{
  const core = RenjuCore();
  // Black overline: X X X X X _ ... playing the 6th makes an overline
  core.setMoves([cell(3,7), cell(0,0), cell(4,7), cell(1,0), cell(5,7), cell(2,0),
                 cell(6,7), cell(3,0), cell(2,7), cell(4,0)]);
  // black has 2,3,4,5,6 -> that is already five. Use a gapped setup instead.
  const c2 = RenjuCore();
  c2.setMoves([cell(3,7), cell(0,0), cell(4,7), cell(1,0), cell(5,7), cell(2,0), cell(6,7), cell(3,0),
               cell(8,7), cell(4,0)]);
  // black at 3,4,5,6 and 8 : playing 7 makes 3..8 = six -> overline
  eq(c2.shapeAt(cell(7,7), c2.BLACK)[0], c2.P_OL, 'black six is an overline');
  ok(!c2.isLegal(cell(7,7), c2.BLACK), 'overline is forbidden for black');
  ok(c2.isLegal(cell(7,7), c2.WHITE), 'white may make an overline');
  const c3 = RenjuCore();
  c3.setMoves([cell(3,7), cell(0,0), cell(4,7), cell(1,0), cell(5,7), cell(2,0), cell(6,7), cell(3,0),
               cell(8,7), cell(4,0)]);
  eq(c3.shapeAt(cell(2,7), c3.BLACK)[0], c3.P_F5, 'exact five still wins for black');
}

{
  const core = RenjuCore();
  // Isolated black pair: the point next to it makes an open three.
  core.setMoves([cell(6,7), cell(0,0), cell(7,7), cell(1,0)]);
  eq(core.shapeAt(cell(8,7), core.BLACK)[0], core.P_F3, 'open two -> open three point');
  eq(core.shapeAt(cell(9,7), core.BLACK)[0], core.P_F3, 'gapped open three point');
  eq(core.shapeAt(cell(10,7), core.BLACK)[0], core.P_B3, 'two gaps -> broken three only');
  const solo = RenjuCore();
  solo.setMoves([cell(7,7), cell(0,0)]);
  eq(solo.shapeAt(cell(5,7), solo.BLACK)[0], solo.P_F2, 'lone stone + 2 apart -> open two');
}

console.log('--- renju forbidden moves ---');
{
  // Double four for black: two separate broken fours meeting at one point.
  //  row  : . X X X . X X X .   -> the middle gap is a double four
  const core = RenjuCore();
  const bs = [cell(4,7), cell(5,7), cell(6,7), cell(8,7), cell(9,7), cell(10,7)];
  const ws = [cell(0,0), cell(1,0), cell(2,0), cell(0,1), cell(1,1), cell(2,1)];
  const seq = [];
  for(let i = 0; i < bs.length; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  ok(!core.isLegal(cell(7,7), core.BLACK), 'double four is forbidden');
  ok(core.isLegal(cell(7,7), core.WHITE), 'double four is fine for white');
}
{
  // Double three for black: two open threes crossing.
  const core = RenjuCore();
  const bs = [cell(6,7), cell(8,7), cell(7,6), cell(7,8)];
  const ws = [cell(0,0), cell(1,0), cell(2,0), cell(3,0)];
  const seq = [];
  for(let i = 0; i < bs.length; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  // playing (7,7) makes _X_X_ horizontally and vertically -> two open threes
  ok(!core.isLegal(cell(7,7), core.BLACK), 'double three is forbidden');
  ok(core.isLegal(cell(7,7), core.WHITE), 'double three fine for white');
}
{
  // A "three" that cannot become a straight four does not count: block one side.
  const core = RenjuCore();
  const bs = [cell(6,7), cell(8,7), cell(7,6), cell(7,8)];
  const ws = [cell(5,7), cell(9,7), cell(0,0), cell(1,0)];
  const seq = [];
  for(let i = 0; i < bs.length; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  // horizontal three is dead (blocked both sides at distance 2) -> only one real three
  ok(core.isLegal(cell(7,7), core.BLACK), 'blocked pseudo-three does not create a double three');
}
{
  // A five is legal even if the same point would be a double four.
  const core = RenjuCore();
  const bs = [cell(4,7), cell(5,7), cell(6,7), cell(8,7), cell(9,7), cell(10,7), cell(7,3), cell(7,4), cell(7,5), cell(7,6)];
  const ws = [cell(0,0), cell(1,0), cell(2,0), cell(0,1), cell(1,1), cell(2,1), cell(0,2), cell(1,2), cell(2,2), cell(3,2)];
  const seq = [];
  for(let i = 0; i < bs.length; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  ok(core.isLegal(cell(7,7), core.BLACK), 'five overrides the double-four ban');
  ok(core.isWinningPoint(cell(7,7), core.BLACK), 'and it is a win');
}

{
  // Two fours in a single line: renju counts this as a double four even though
  // it is not four-in-a-row, so Black may not play it but White may.
  //   . X . X X X . X    -> the point at index 5 makes 1-5 and 3-7 both fours
  const core = RenjuCore();
  const bs = [cell(1,7), cell(3,7), cell(4,7), cell(7,7)];
  const ws = [cell(0,0), cell(1,0), cell(2,0), cell(3,0)];
  const seq = [];
  for(let i = 0; i < bs.length; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  eq(core.shapeAt(cell(5,7), core.BLACK)[0], core.P_D4, 'two fours in one line are recognised');
  ok(!core.isLegal(cell(5,7), core.BLACK), 'two fours in one line is forbidden for Black');
  const core2 = RenjuCore();
  const seq2 = [];
  for(let i = 0; i < bs.length; i++){ seq2.push(ws[i]); seq2.push(bs[i]); }
  core2.setMoves(seq2);
  ok(core2.isLegal(cell(5,7), core2.BLACK), 'sanity: the same point is fine when the stones are White');
  // A genuine straight four stays legal.
  const core3 = RenjuCore();
  const bs3 = [cell(4,7), cell(5,7), cell(6,7)];
  const ws3 = [cell(0,0), cell(1,0), cell(2,0)];
  const seq3 = [];
  for(let i = 0; i < 3; i++){ seq3.push(bs3[i]); seq3.push(ws3[i]); }
  core3.setMoves(seq3);
  eq(core3.shapeAt(cell(7,7), core3.BLACK)[0], core3.P_F4, 'straight four is still a straight four');
  ok(core3.isLegal(cell(7,7), core3.BLACK), 'straight four is legal for Black');
}

console.log('--- make / unmake symmetry ---');
{
  const core = RenjuCore();
  const before = JSON.stringify(core.counts());
  const seq = [cell(7,7), cell(7,6), cell(8,8), cell(6,6), cell(6,8), cell(8,6), cell(5,9), cell(9,5)];
  core.setMoves(seq);
  const mid = JSON.stringify(core.counts());
  for(let i = 0; i < seq.length; i++) core.undo();
  eq(JSON.stringify(core.counts()), before, 'counters restored after undo');
  core.setMoves(seq);
  eq(JSON.stringify(core.counts()), mid, 'counters reproducible');
}

console.log('--- tactics ---');
function findsMove(core, side, want, ms, label){
  const r = core.think({ side, timeMs: ms, maxDepth: 14 });
  const good = want.includes(r.cell);
  ok(good, `${label}: played ${nameOf(r.cell)}, expected one of ${want.map(nameOf).join('/')} (score ${r.score}, depth ${r.depth}, nodes ${r.nodes}, ${r.mode})`);
  return r;
}
{
  // Black to move has four in a row -> must complete five.
  const core = RenjuCore();
  const seq = [];
  const bs = [cell(4,7), cell(5,7), cell(6,7), cell(7,7)];
  const ws = [cell(4,9), cell(5,9), cell(6,9), cell(0,0)];
  for(let i = 0; i < bs.length; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  findsMove(core, core.BLACK, [cell(3,7), cell(8,7)], 300, 'complete five');
}
{
  // White to move: black threatens five at one point, must block.
  const core = RenjuCore();
  const seq = [];
  const bs = [cell(4,7), cell(5,7), cell(6,7), cell(7,7)];
  const ws = [cell(3,7), cell(4,9), cell(5,9), cell(0,0)];
  for(let i = 0; i < bs.length; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  findsMove(core, core.WHITE, [cell(8,7)], 300, 'block the five');
}
{
  // White open three must be answered (black to move).
  const core = RenjuCore();
  const seq = [];
  const bs = [cell(2,2), cell(3,3), cell(10,10)];
  const ws = [cell(5,7), cell(6,7), cell(7,7)];
  for(let i = 0; i < Math.max(bs.length, ws.length); i++){
    if(i < bs.length) seq.push(bs[i]);
    if(i < ws.length) seq.push(ws[i]);
  }
  core.setMoves(seq);
  const r = core.think({ side: core.BLACK, timeMs: 800, maxDepth: 14 });
  ok([cell(4,7), cell(8,7), cell(3,7), cell(9,7)].includes(r.cell),
     `answer the open three: played ${nameOf(r.cell)} (score ${r.score}, depth ${r.depth})`);
}

{
  // A cross of two open threes: playing the crossing point makes a double four.
  const bs = [cell(1,1), cell(2,1), cell(3,1), cell(1,3), cell(2,3), cell(3,3)];
  const ws = [cell(4,7), cell(5,7), cell(6,7), cell(7,4), cell(7,5), cell(7,6)];
  const seq = [];
  for(let i = 0; i < 6; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  const core = RenjuCore();
  core.setMoves(seq);
  const r = core.think({ side: core.WHITE, timeMs: 1000, maxDepth: 16 });
  ok(r.cell === cell(7,7), `white finds the double four: ${nameOf(r.cell)}`);
  ok(r.mate !== 0 || r.score > core.MATE - 500, `and calls it a win (score ${r.score}, mode ${r.mode})`);

  // The mirrored shape is forbidden for Black, so Black must find something else.
  const core2 = RenjuCore();
  const seq2 = [];
  for(let i = 0; i < 6; i++){ seq2.push(ws[i]); seq2.push(bs[i]); }   // colours swapped
  core2.setMoves(seq2);
  ok(!core2.isLegal(cell(7,7), core2.BLACK), 'the same point is a forbidden double four for Black');
  const r2 = core2.think({ side: core2.BLACK, timeMs: 800, maxDepth: 16 });
  ok(r2.cell !== cell(7,7), 'engine as Black avoids the forbidden point');
  ok(core2.isLegal(r2.cell, core2.BLACK), 'engine as Black plays a legal point');
}
{
  // Victory by continuous four: white has two separate broken fours to chain.
  const core = RenjuCore();
  const ws = [cell(3,5), cell(4,5), cell(5,5), cell(6,6), cell(7,7), cell(8,8)];
  const bs = [cell(2,5), cell(0,0), cell(1,0), cell(2,0), cell(3,0), cell(0,1)];
  const seq = [];
  for(let i = 0; i < 6; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  const r = core.think({ side: core.WHITE, timeMs: 1500, maxDepth: 18 });
  ok(core.isLegal(r.cell, core.WHITE), 'vcf position: legal move chosen');
  console.log(`  vcf probe: ${nameOf(r.cell)} score ${r.score} mode ${r.mode}`);
}
{
  // The opponent threatens a four next move; the engine must not ignore it.
  const core = RenjuCore();
  const ws = [cell(5,5), cell(6,6), cell(7,7)];         // white open three on a diagonal
  const bs = [cell(0,0), cell(1,0), cell(14,14)];
  const seq = [];
  for(let i = 0; i < 3; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  const r = core.think({ side: core.BLACK, timeMs: 1200, maxDepth: 16 });
  const stops = [cell(4,4), cell(8,8), cell(3,3), cell(9,9)];
  ok(stops.includes(r.cell), `black stops the diagonal open three: played ${nameOf(r.cell)}`);
}

console.log('--- speed ---');
{
  const core = RenjuCore();
  core.setMoves([cell(7,7), cell(7,6), cell(8,8), cell(6,6), cell(6,8), cell(8,6)]);
  const t0 = performance.now();
  const r = core.think({ side: core.BLACK, timeMs: 2000, maxDepth: 20 });
  const dt = performance.now() - t0;
  console.log(`  depth ${r.depth} (sel ${r.seldepth}) nodes ${r.nodes} in ${dt.toFixed(0)} ms = ${(r.nodes / dt).toFixed(0)} knodes/s`);
  console.log(`  move ${nameOf(r.cell)} score ${r.score} pv ${r.pv.map(nameOf).join(' ')}`);
  ok(r.depth >= 6, `reaches at least depth 6 in 2 s (got ${r.depth})`);
  ok(r.seldepth >= 16, `and follows forcing lines deep (seldepth ${r.seldepth})`);
  ok(r.nodes / dt > 60, `at least 60k nodes/s (got ${(r.nodes / dt).toFixed(0)})`);
}

console.log(`\n${pass} passed, ${fail} failed`);
if(fail) Deno.exit(1);
