// ==== RENJU CORE BEGIN ====
// Renju engine core.
//
// Design notes
// ------------
// * Every line of 11 cells (centre +/- 5) is encoded in base 3 from the point of
//   view of one colour (0 = empty, 1 = own stone, 2 = blocked by opponent or by
//   the board edge).  A memoised table maps such a key to the *shape* that the
//   colour obtains by playing the centre point: five, overline, open four,
//   simple four, open three, broken three, ... 11 cells is exactly the width
//   needed to classify every shape up to "two" without edge artefacts.
// * The 4 keys of every cell are maintained incrementally.  Playing or taking
//   back a stone touches 4 directions x 10 neighbours, so make/unmake is O(1)
//   and the evaluation is a pair of running sums - the whole evaluation is O(1).
// * Renju forbidden moves (overline / double four / double three) are derived
//   from the same shape table; the double-three rule is checked recursively, so
//   a three only counts when it can actually be pushed to a straight four by a
//   legal move.
function RenjuCore(){
'use strict';

const N = 15, NN = 225;
const EMPTY = 0, BLACK = 1, WHITE = 2;

// Shape codes, ordered by strength (OL is special-cased, never "strong").
// P_D4 is two fours in one line: it has two five-points like a straight four,
// but it is not four in a row, and renju counts it as a double four.
const P_NONE = 0, P_B2 = 1, P_F2 = 2, P_B3 = 3, P_F3 = 4,
      P_B4 = 5, P_F4 = 6, P_D4 = 7, P_F5 = 8, P_OL = 9;
const NPAT = 10;
const SHAPE_NAME = ['-', 'two', 'open two', 'three', 'open three',
                    'four', 'open four', 'double four', 'five', 'overline'];

const MATE = 1000000;
const MAX_PLY = 60;

const POW3 = new Int32Array(12);
(function(){ let v = 1; for(let i = 0; i < 12; i++){ POW3[i] = v; v *= 3; } })();
const NKEY = POW3[11];
const CENTRE_POW = POW3[5];

const DX = [1, 0, 1, 1], DY = [0, 1, 1, -1];

const now = (typeof performance !== 'undefined' && performance.now)
  ? () => performance.now() : () => Date.now();

// ---------------------------------------------------------------- geometry --
// LINE[(d*NN+c)*11 + j] = cell at offset (j-5) from c along direction d, or -1.
const LINE = new Int32Array(4 * NN * 11);
for(let d = 0; d < 4; d++){
  for(let c = 0; c < NN; c++){
    const x = c % N, y = (c / N) | 0;
    for(let j = 0; j < 11; j++){
      const xx = x + (j - 5) * DX[d], yy = y + (j - 5) * DY[d];
      LINE[(d * NN + c) * 11 + j] = (xx >= 0 && xx < N && yy >= 0 && yy < N) ? yy * N + xx : -1;
    }
  }
}

const CENTREV = new Int32Array(NN);
for(let c = 0; c < NN; c++){
  const x = c % N, y = (c / N) | 0;
  CENTREV[c] = 7 - Math.max(Math.abs(x - 7), Math.abs(y - 7));
}

// ----------------------------------------------------------- shape tables --
function buildShapeTable(isBlack){
  const tbl = new Uint8Array(NKEY).fill(255);
  function shape(k){
    const cached = tbl[k];
    if(cached !== 255) return cached;
    const cells = new Int32Array(11);
    let kk = k;
    for(let i = 0; i < 11; i++){ const r = kk % 3; cells[i] = r; kk = (kk - r) / 3; }
    // cells[5] is always 1 (the stone we are testing).
    let run = 1;
    for(let i = 4; i >= 0 && cells[i] === 1; i--) run++;
    for(let i = 6; i < 11 && cells[i] === 1; i++) run++;
    let res;
    if(run >= 5){
      res = (!isBlack || run === 5) ? P_F5 : P_OL;
    } else {
      let fives = 0;
      for(let i = 1; i < 10; i++){
        if(cells[i] !== 0) continue;
        if(shape(k + POW3[i]) === P_F5){ fives++; if(fives >= 2) break; }
      }
      if(fives >= 2){
        // Four in a row with both ends empty is a straight four (one four, and
        // legal for Black).  Two five-points in any other arrangement means two
        // separate fours share this point.
        let l = 4; while(l >= 0 && cells[l] === 1) l--;
        let r = 6; while(r < 11 && cells[r] === 1) r++;
        const straight = run === 4 && l >= 0 && cells[l] === 0 && r < 11 && cells[r] === 0;
        res = straight ? P_F4 : P_D4;
      }
      else if(fives === 1) res = P_B4;
      else {
        let f4 = false, b4 = false, f3 = false, b3 = false;
        for(let i = 1; i < 10; i++){
          if(cells[i] !== 0) continue;
          const s = shape(k + POW3[i]);
          if(s === P_F4) f4 = true;
          // A point that would make two fours in one line is a four-making
          // point too, but it is not a straight four, so the shape is only a
          // broken three (and for Black that extension is forbidden anyway).
          else if(s === P_B4 || s === P_D4) b4 = true;
          else if(s === P_F3) f3 = true;
          else if(s === P_B3) b3 = true;
        }
        res = f4 ? P_F3 : b4 ? P_B3 : f3 ? P_F2 : b3 ? P_B2 : P_NONE;
      }
    }
    tbl[k] = res;
    return res;
  }
  return { tbl, shape };
}
const shTabB = buildShapeTable(true), shTabW = buildShapeTable(false);
const tblB = shTabB.tbl, tblW = shTabW.tbl;
const shapeBlack = shTabB.shape, shapeWhite = shTabW.shape;
// Hot-path lookups: the memo table is filled lazily, 255 means "not computed".
function shB(k){ const v = tblB[k]; return v !== 255 ? v : shapeBlack(k); }
function shW(k){ const v = tblW[k]; return v !== 255 ? v : shapeWhite(k); }

// ------------------------------------------------------- evaluation tables --
// Values are attached to *empty* points and describe what playing there would
// achieve, so an open three on the board shows up as two P_F4 points, a live
// two as two P_F3 points, and so on.  This gives a threat-flavoured evaluation
// that can be kept as a running sum.
const VAL = new Int32Array(NPAT);
VAL[P_NONE] = 0;
VAL[P_B2]   = 4;
VAL[P_F2]   = 18;
VAL[P_B3]   = 22;
VAL[P_F3]   = 180;
VAL[P_B4]   = 70;
VAL[P_F4]   = 1400;
VAL[P_D4]   = 1450;
VAL[P_F5]   = 9000;
VAL[P_OL]   = 0;

const COMBO = new Int32Array(NPAT * NPAT);
const isFour = p => p >= P_B4 && p <= P_D4;
for(let b = 0; b < NPAT; b++){
  for(let s = 0; s < NPAT; s++){
    let v = 0;
    if(isFour(b) && isFour(s)) v = 2600;                                // double four
    else if(isFour(b) && s === P_F3) v = 1600;                          // four-three
    else if(b === P_F3 && s === P_F3) v = 700;                          // double open three
    else if(b === P_F3 && s === P_B3) v = 90;
    else if(b === P_F3 && s === P_F2) v = 40;
    COMBO[b * NPAT + s] = v;
  }
}

// ------------------------------------------------------------ board state --
const board = new Uint8Array(NN);
const keyB = new Int32Array(4 * NN), keyW = new Int32Array(4 * NN);
const patB = new Uint8Array(4 * NN), patW = new Uint8Array(4 * NN);
const bstB = new Uint8Array(NN),  bstW = new Uint8Array(NN);
const secB = new Uint8Array(NN),  secW = new Uint8Array(NN);
const valB = new Int32Array(NN),  valW = new Int32Array(NN);
const cntB = new Int32Array(NPAT), cntW = new Int32Array(NPAT);
// qforb: 0 = fine, 1 = certainly forbidden (overline / double four),
//        2 = double open three, needs the recursive check.
const qforb = new Uint8Array(NN);
const nearCnt = new Uint8Array(NN);
// Bitmask of the points worth looking at: empty and within 2 of a stone. Move
// generation and the four-move scan walk this instead of all 225 points, in the
// same order, so the search behaves identically and just visits fewer cells.
// (Any point that can make a four has a stone within 2 of it: a four needs 3
// stones among the 4 other cells of a five-window, and only 2 of those can be
// further away than 2.)
const activeMask = new Uint32Array(8);
function updateActive(c){
  const w = c >> 5, bit = 1 << (c & 31);
  if(board[c] === EMPTY && nearCnt[c] !== 0) activeMask[w] |= bit;
  else activeMask[w] &= ~bit;
}
let sumB = 0, sumW = 0, forbCnt = 0, posSum = 0;

const moveList = [];
let winner = 0;            // 0 = none, BLACK, WHITE
let winLine = null;

// Zobrist (two independent 32 bit halves).
const ZB = new Int32Array(NN * 2), ZB2 = new Int32Array(NN * 2);
(function(){
  let s = 0x9e3779b9 | 0;
  const rnd = () => {
    s ^= s << 13; s |= 0; s ^= s >>> 17; s ^= s << 5; s |= 0;
    return s;
  };
  for(let i = 0; i < NN * 2; i++){ ZB[i] = rnd(); ZB2[i] = rnd(); }
})();
let hash1 = 0, hash2 = 0;

/**
 * Recompute one empty point's aggregate from its four direction patterns and
 * fold the difference into the running totals. Placing a stone touches ~25
 * empty neighbours, so this is the hottest function in the engine: it applies
 * deltas rather than removing and re-adding the point's whole contribution.
 */
function refreshCell(c){
  // Black
  {
    let b = 0, s = 0, ol = 0, fours = 0, threes = 0;
    for(let d = 0; d < 4; d++){
      const p = patB[d * NN + c];
      if(p === P_OL){ ol = 1; continue; }
      if(p > b){ s = b; b = p; } else if(p > s) s = p;
      if(isFour(p)) fours += (p === P_D4 ? 2 : 1);
      else if(p === P_F3) threes++;
    }
    let q = 0;
    if(b !== P_F5){
      if(ol || fours >= 2) q = 1;
      else if(threes >= 2) q = 2;
    }
    if(q !== 0){ b = 0; s = 0; }
    const v = q !== 0 ? 0 : VAL[b] + COMBO[b * NPAT + s];
    sumB += v - valB[c];
    const ob = bstB[c];
    if(ob !== b){ cntB[ob]--; cntB[b]++; bstB[c] = b; }
    if(qforb[c] !== q){
      if(qforb[c] === 1) forbCnt--;
      if(q === 1) forbCnt++;
      qforb[c] = q;
    }
    secB[c] = s; valB[c] = v;
  }
  // White
  {
    let b = 0, s = 0;
    for(let d = 0; d < 4; d++){
      const p = patW[d * NN + c];
      if(p > b){ s = b; b = p; } else if(p > s) s = p;
    }
    const v = VAL[b] + COMBO[b * NPAT + s];
    sumW += v - valW[c];
    const ob = bstW[c];
    if(ob !== b){ cntW[ob]--; cntW[b]++; bstW[c] = b; }
    secW[c] = s; valW[c] = v;
  }
}

/** The point is about to be occupied: take its contribution back out. */
function clearCell(c){
  sumB -= valB[c]; cntB[bstB[c]]--; cntB[0]++;
  sumW -= valW[c]; cntW[bstW[c]]--; cntW[0]++;
  if(qforb[c] === 1) forbCnt--;
  valB[c] = 0; valW[c] = 0; bstB[c] = 0; bstW[c] = 0;
  secB[c] = 0; secW[c] = 0; qforb[c] = 0;
}

function bumpNear(c, delta){
  const x = c % N, y = (c / N) | 0;
  const y0 = Math.max(0, y - 2), y1 = Math.min(N - 1, y + 2);
  const x0 = Math.max(0, x - 2), x1 = Math.min(N - 1, x + 2);
  for(let yy = y0; yy <= y1; yy++){
    const row = yy * N;
    for(let xx = x0; xx <= x1; xx++){ nearCnt[row + xx] += delta; updateActive(row + xx); }
  }
}

function resetBoard(){
  board.fill(EMPTY);
  patB.fill(P_NONE); patW.fill(P_NONE);
  bstB.fill(0); bstW.fill(0); secB.fill(0); secW.fill(0);
  valB.fill(0); valW.fill(0); qforb.fill(0); nearCnt.fill(0);
  cntB.fill(0); cntW.fill(0); activeMask.fill(0);
  sumB = 0; sumW = 0; forbCnt = 0; posSum = 0;
  hash1 = 0; hash2 = 0;
  moveList.length = 0; winner = 0; winLine = null;
  for(let d = 0; d < 4; d++){
    for(let c = 0; c < NN; c++){
      let k = 0;
      const base = (d * NN + c) * 11;
      for(let j = 0; j < 11; j++) if(LINE[base + j] < 0) k += 2 * POW3[j];
      keyB[d * NN + c] = k; keyW[d * NN + c] = k;
      patB[d * NN + c] = shapeBlack(k + CENTRE_POW);
      patW[d * NN + c] = shapeWhite(k + CENTRE_POW);
    }
  }
  cntB[0] = NN; cntW[0] = NN;
  for(let c = 0; c < NN; c++) refreshCell(c);
}

/** Does playing `color` at `c` complete a five (a win)? */
function isWinningPoint(c, color){
  const pat = color === BLACK ? patB : patW;
  for(let d = 0; d < 4; d++) if(pat[d * NN + c] === P_F5) return true;
  return false;
}

function place(c, color){
  clearCell(c);
  cntB[0]--; cntW[0]--;          // the point is occupied now, not an empty "none"
  board[c] = color;
  const cb = color === BLACK ? 1 : 2;
  const cw = color === BLACK ? 2 : 1;
  for(let d = 0; d < 4; d++){
    const base = (d * NN + c) * 11;
    const dOff = d * NN;
    for(let j = 0; j < 11; j++){
      if(j === 5) continue;
      const c2 = LINE[base + j];
      if(c2 < 0) continue;
      const pw = POW3[10 - j];
      const ix = dOff + c2;
      keyB[ix] += cb * pw; keyW[ix] += cw * pw;
      if(board[c2] === EMPTY){
        patB[ix] = shB(keyB[ix] + CENTRE_POW);
        patW[ix] = shW(keyW[ix] + CENTRE_POW);
        refreshCell(c2);
      }
    }
  }
  bumpNear(c, 1);
  posSum += (color === BLACK ? 1 : -1) * CENTREV[c];
  const zi = c * 2 + (color - 1);
  hash1 ^= ZB[zi]; hash2 ^= ZB2[zi];
}

function unplace(c){
  const color = board[c];
  const zi = c * 2 + (color - 1);
  hash1 ^= ZB[zi]; hash2 ^= ZB2[zi];
  posSum -= (color === BLACK ? 1 : -1) * CENTREV[c];
  bumpNear(c, -1);
  const cb = color === BLACK ? 1 : 2;
  const cw = color === BLACK ? 2 : 1;
  for(let d = 0; d < 4; d++){
    const base = (d * NN + c) * 11;
    const dOff = d * NN;
    for(let j = 0; j < 11; j++){
      if(j === 5) continue;
      const c2 = LINE[base + j];
      if(c2 < 0) continue;
      const pw = POW3[10 - j];
      const ix = dOff + c2;
      keyB[ix] -= cb * pw; keyW[ix] -= cw * pw;
      if(board[c2] === EMPTY){
        patB[ix] = shB(keyB[ix] + CENTRE_POW);
        patW[ix] = shW(keyW[ix] + CENTRE_POW);
        refreshCell(c2);
      }
    }
  }
  board[c] = EMPTY;
  cntB[0]++; cntW[0]++;          // empty again
  for(let d = 0; d < 4; d++){
    const ix = d * NN + c;
    patB[ix] = shB(keyB[ix] + CENTRE_POW);
    patW[ix] = shW(keyW[ix] + CENTRE_POW);
  }
  refreshCell(c);
  updateActive(c);
}

// ------------------------------------------------------- renju legality ----
/**
 * Full renju forbidden-move test for Black.  `depth` bounds the recursion used
 * by the double-three rule (a three only counts if it can be extended into a
 * straight four by a move that is itself legal).
 *
 * Returns 0 when the point is legal, otherwise which rule forbids it.
 */
const FORBID_NONE = 0, FORBID_OVERLINE = 1, FORBID_FOUR = 2, FORBID_THREE = 3;
const FORBID_TEXT = [
  '',
  'Overline: this would give Black six or more in a row, which does not count as a five.',
  'Double four: this move makes two fours at once.',
  'Double three: this move makes two open threes at once.'
];
function blackForbiddenWhy(c, depth){
  if(board[c] !== EMPTY) return FORBID_FOUR;
  let ol = 0, fours = 0, nThrees = 0;
  const threeDirs = [0, 0, 0, 0];
  for(let d = 0; d < 4; d++){
    const p = patB[d * NN + c];
    if(p === P_F5) return FORBID_NONE;       // a five always wins, even with an overline
    if(p === P_OL) ol = 1;
    else if(isFour(p)) fours += (p === P_D4 ? 2 : 1);
    else if(p === P_F3) threeDirs[nThrees++] = d;
  }
  if(ol) return FORBID_OVERLINE;
  if(fours >= 2) return FORBID_FOUR;
  if(nThrees < 2) return FORBID_NONE;
  if(depth >= 5) return FORBID_THREE;        // pathologically deep nesting: assume forbidden
  place(c, BLACK);
  let real = 0;
  for(let i = 0; i < nThrees && real < 2; i++){
    const d = threeDirs[i];
    const base = (d * NN + c) * 11;
    for(let j = 1; j < 10; j++){
      if(j === 5) continue;
      const e = LINE[base + j];
      if(e < 0 || board[e] !== EMPTY) continue;
      if(patB[d * NN + e] !== P_F4) continue;
      if(blackForbiddenWhy(e, depth + 1) === FORBID_NONE){ real++; break; }
    }
  }
  unplace(c);
  return real >= 2 ? FORBID_THREE : FORBID_NONE;
}
function blackForbidden(c, depth){ return blackForbiddenWhy(c, depth) !== FORBID_NONE; }

/** Why Black may not play `c`: '' when the point is legal. */
function forbiddenReason(c){
  if(c < 0 || c >= NN || board[c] !== EMPTY) return '';
  if(qforb[c] === 0) return '';
  return FORBID_TEXT[blackForbiddenWhy(c, 0)];
}

function isLegal(c, color){
  if(c < 0 || c >= NN || board[c] !== EMPTY) return false;
  if(color !== BLACK) return true;
  if(qforb[c] === 0) return true;
  if(qforb[c] === 1) return false;
  return !blackForbidden(c, 0);
}

// --------------------------------------------------------------- game API --
function play(c, color){
  if(!isLegal(c, color)) return false;
  const win = isWinningPoint(c, color);
  place(c, color);
  moveList.push(c);
  if(win){ winner = color; winLine = winningLine(c, color); }
  return true;
}

function winningLine(c, color){
  for(let d = 0; d < 4; d++){
    let n = 1;
    const cells = [c];
    const base = (d * NN + c) * 11;
    for(let j = 4; j >= 1; j--){
      const e = LINE[base + j];
      if(e < 0 || board[e] !== color) break;
      cells.unshift(e); n++;
    }
    for(let j = 6; j <= 9; j++){
      const e = LINE[base + j];
      if(e < 0 || board[e] !== color) break;
      cells.push(e); n++;
    }
    if(color === WHITE ? n >= 5 : n === 5) return cells;
  }
  return null;
}

function undo(){
  if(!moveList.length) return false;
  const c = moveList.pop();
  unplace(c);
  winner = 0; winLine = null;
  return true;
}

function setMoves(list){
  resetBoard();
  for(let i = 0; i < list.length; i++){
    const c = list[i];
    const color = (i % 2 === 0) ? BLACK : WHITE;
    const win = isWinningPoint(c, color);
    place(c, color);
    moveList.push(c);
    if(win){ winner = color; winLine = winningLine(c, color); break; }
  }
}

// ------------------------------------------------------------ evaluation --
function evaluate(side){
  const me = side === BLACK ? sumB : sumW;
  const op = side === BLACK ? sumW : sumB;
  const cntMe = side === BLACK ? cntB : cntW;
  // Having the move matters in proportion to the threats you can actually cash
  // in, not to your whole position: a bonus proportional to the total made the
  // reported score swing by a full open three every single ply.
  const tempo = Math.min(700, cntMe[P_F4] * 300 + cntMe[P_D4] * 300 + cntMe[P_F3] * 25);
  let s = me - op + tempo;
  // Black's forbidden points are permanent holes in Black's shape, and the
  // main thing White plays for. Double-three points are only counted at a
  // discount: the quick test that finds them can be wrong, the recursive one
  // is too slow to run over the board at every node.
  const forbBonus = forbCnt * 12;
  s += (side === WHITE ? forbBonus : -forbBonus);
  s += (side === BLACK ? posSum * 3 : -posSum * 3);
  if(s > 400000) s = 400000;
  if(s < -400000) s = -400000;
  return s;
}

// ------------------------------------------------------------- search ------
const TT_BITS = 19, TT_SIZE = 1 << TT_BITS, TT_MASK = TT_SIZE - 1;
const ttVerify = new Int32Array(TT_SIZE);
const ttScore  = new Int32Array(TT_SIZE);
const ttInfo   = new Int32Array(TT_SIZE);   // move | flag<<8 | depth<<10 | gen<<17
const ttStamp  = new Int32Array(TT_SIZE);
let ttGen = 0;
const FLAG_EXACT = 0, FLAG_LOWER = 1, FLAG_UPPER = 2;

const SIDE_H1 = 0x5bf03635 | 0, SIDE_H2 = 0x2545f491 | 0;

const moveBuf = new Int32Array(MAX_PLY * 48);
const scratch = new Int32Array(MAX_PLY * 24);
const killer  = new Int32Array(MAX_PLY * 2);
const histB   = new Int32Array(NN), histW = new Int32Array(NN);

// Quiescence budget: a forced block costs 1, playing a four costs 2, so the
// VCF chain at a leaf is bounded both in length and in width.
const QDEPTH = 10;      // quiescence: a block costs 1, playing a four costs 2
const QWIDTH = 8;       // four-moves tried per quiescence node
const VCT_MAXD = 11;    // attacker moves in the root threat search
const VCT_DEFW = 12;    // defender replies it considers
const LIM_HI = 16, LIM_MID = 13, LIM_LO = 10;   // moves per node, by depth
const ROOTW = 24;
let nodes = 0, deadline = 0, aborted = false, seldepth = 0;

function timeUp(){
  if(aborted) return true;
  if(now() >= deadline){ aborted = true; return true; }
  return false;
}

/** Collect up to `max` empty points whose best shape for `color` is in [lo,hi]. */
function collectShape(color, lo, hi, out, off, max){
  const bst = color === BLACK ? bstB : bstW;
  let n = 0;
  for(let w = 0; w < 8; w++){
    let bits = activeMask[w];
    while(bits !== 0){
      const c = (w << 5) + (31 - Math.clz32(bits & -bits));
      bits &= bits - 1;
      const b = bst[c];
      if(b >= lo && b <= hi){ out[off + n] = c; if(++n >= max) return n; }
    }
  }
  return n;
}

/**
 * Quiescence = an exact VCF search.  Only four-creating moves are tried; a
 * player facing a five threat is forced to block, which keeps the tree narrow
 * and makes every leaf tactically settled.
 */
function quiesce(alpha, beta, side, ply, budget){
  nodes++;
  if((nodes & 511) === 0 && timeUp()) return 0;
  if(ply > seldepth) seldepth = ply;
  const opp = side === BLACK ? WHITE : BLACK;
  const cntMe = side === BLACK ? cntB : cntW;
  const cntOp = side === BLACK ? cntW : cntB;

  if(cntMe[P_F5] > 0) return MATE - ply;
  if(cntOp[P_F5] > 0){
    // Forced: block or lose.  Blocking does not consume the four budget.
    const off = ply * 24;
    const n = collectShape(opp, P_F5, P_F5, scratch, off, 2);
    if(n >= 2) return -(MATE - ply - 1);
    const c = scratch[off];
    if(side === BLACK && !isLegal(c, BLACK)) return -(MATE - ply - 1);
    place(c, side);
    const v = -quiesce(-beta, -alpha, opp, ply + 1, budget - 1);
    unplace(c);
    return v;
  }
  const stand = evaluate(side);
  if(stand >= beta) return beta;
  if(alpha < stand) alpha = stand;
  if(budget <= 0 || ply >= MAX_PLY - 4) return stand;

  const off = ply * 24;
  const n = collectShape(side, P_B4, P_D4, scratch, off, 8);
  for(let i = 0; i < n; i++){
    const c = scratch[off + i];
    if(side === BLACK && !isLegal(c, BLACK)) continue;
    place(c, side);
    const v = -quiesce(-beta, -alpha, opp, ply + 1, budget - 2);
    unplace(c);
    if(aborted) return 0;
    if(v >= beta) return beta;
    if(v > alpha) alpha = v;
  }
  return alpha;
}

/** Fill moveBuf[ply*48 ...] with up to `limit` packed (score<<8|cell), best first. */
function genMoves(side, ply, limit, ttMove){
  const off = ply * 48;
  const valMe = side === BLACK ? valB : valW;
  const valOp = side === BLACK ? valW : valB;
  const bstMe = side === BLACK ? bstB : bstW;
  const bstOp = side === BLACK ? bstW : bstB;
  const hist  = side === BLACK ? histB : histW;
  const k0 = killer[ply * 2], k1 = killer[ply * 2 + 1];
  let n = 0, worst = 0x7fffffff, worstAt = -1;
  for(let w = 0; w < 8; w++){
  let bits = activeMask[w];
  while(bits !== 0){
    const c = (w << 5) + (31 - Math.clz32(bits & -bits));
    bits &= bits - 1;
    const vm = valMe[c], vo = valOp[c];
    if(side === BLACK){
      if(qforb[c] === 1) continue;
      if(qforb[c] === 2 && blackForbidden(c, 0)) continue;
    }
    const h = hist[c] >> 6;
    let sc = vm + ((vo * 3) >> 2) + CENTREV[c] * 4 + (h > 3000 ? 3000 : h);
    if(c === ttMove) sc = 0x1ffff0;
    else if(c === k0) sc += 5000;
    else if(c === k1) sc += 3000;
    if(sc > 0x1fffff) sc = 0x1fffff;
    if(sc < 0) sc = 0;
    const packed = (sc << 8) | c;
    if(n < limit){
      moveBuf[off + n] = packed;
      n++;
      if(n === limit){
        worst = 0x7fffffff; worstAt = -1;
        for(let i = 0; i < n; i++) if(moveBuf[off + i] < worst){ worst = moveBuf[off + i]; worstAt = i; }
      }
    } else if(packed > worst){
      moveBuf[off + worstAt] = packed;
      worst = 0x7fffffff; worstAt = -1;
      for(let i = 0; i < n; i++) if(moveBuf[off + i] < worst){ worst = moveBuf[off + i]; worstAt = i; }
    }
  }
  }
  // Insertion sort, descending: n is small (<= limit).
  for(let i = 1; i < n; i++){
    const v = moveBuf[off + i];
    let j = i - 1;
    while(j >= 0 && moveBuf[off + j] < v){ moveBuf[off + j + 1] = moveBuf[off + j]; j--; }
    moveBuf[off + j + 1] = v;
  }
  return n;
}

function ttProbe(side){
  const h1 = side === WHITE ? (hash1 ^ SIDE_H1) : hash1;
  const h2 = side === WHITE ? (hash2 ^ SIDE_H2) : hash2;
  const ix = h1 & TT_MASK;
  if(ttVerify[ix] !== h2 || ttStamp[ix] === 0) return -1;
  return ix;
}

// Mate scores are distances from the root, so they have to be re-based on the
// current ply before they can be shared between different paths.
const MATE_BAND = MATE - 1000;
function ttStore(side, depth, score, flag, move, ply){
  if(score >= MATE_BAND) score += ply;
  else if(score <= -MATE_BAND) score -= ply;
  const h1 = side === WHITE ? (hash1 ^ SIDE_H1) : hash1;
  const h2 = side === WHITE ? (hash2 ^ SIDE_H2) : hash2;
  const ix = h1 & TT_MASK;
  const prevDepth = (ttInfo[ix] >> 10) & 0x7f;
  if(ttStamp[ix] === ttGen && prevDepth > depth && ttVerify[ix] === h2) return;
  ttVerify[ix] = h2;
  ttScore[ix] = score;
  ttInfo[ix] = (move & 0xff) | (flag << 8) | (depth << 10);
  ttStamp[ix] = ttGen;
}

function search(depth, alpha, beta, side, ply){
  nodes++;
  if((nodes & 511) === 0 && timeUp()) return 0;
  if(ply > seldepth) seldepth = ply;
  const opp = side === BLACK ? WHITE : BLACK;
  const cntMe = side === BLACK ? cntB : cntW;
  const cntOp = side === BLACK ? cntW : cntB;

  if(cntMe[P_F5] > 0) return MATE - ply;
  if(cntOp[P_F5] > 0){
    const off = ply * 24;
    const n = collectShape(opp, P_F5, P_F5, scratch, off, 2);
    if(n >= 2) return -(MATE - ply - 1);
    const c = scratch[off];
    if(side === BLACK && !isLegal(c, BLACK)) return -(MATE - ply - 1);
    place(c, side);
    const v = -search(ply < 20 ? depth : depth - 1, -beta, -alpha, opp, ply + 1);   // forced reply
    unplace(c);
    return v;
  }
  if(depth <= 0 || ply >= MAX_PLY - 6) return quiesce(alpha, beta, side, ply, QDEPTH);

  const alpha0 = alpha;
  let ttMove = -1;
  const ix = ttProbe(side);
  if(ix >= 0){
    const info = ttInfo[ix];
    ttMove = info & 0xff;
    const tdepth = (info >> 10) & 0x7f;
    if(tdepth >= depth){
      let s = ttScore[ix];
      if(s >= MATE_BAND) s -= ply;
      else if(s <= -MATE_BAND) s += ply;
      const flag = (info >> 8) & 3;
      if(flag === FLAG_EXACT) return s;
      if(flag === FLAG_LOWER){ if(s > alpha) alpha = s; }
      else if(s < beta) beta = s;
      if(alpha >= beta) return s;
    }
  }

  const limit = depth >= 6 ? LIM_HI : depth >= 4 ? LIM_MID : LIM_LO;
  const n = genMoves(side, ply, limit, ttMove);
  if(n === 0) return evaluate(side);
  const off = ply * 48;
  const bstMe = side === BLACK ? bstB : bstW;
  const bstOp = side === BLACK ? bstW : bstB;

  let best = -Infinity, bestMove = moveBuf[off] & 0xff;
  for(let i = 0; i < n; i++){
    const c = moveBuf[off + i] & 0xff;
    const tag = bstMe[c], oppTag = bstOp[c];
    place(c, side);
    const forcing = tag >= P_B4 || oppTag >= P_B4;
    // Extend the forcing moves. Fours were always extended; extending open
    // threes as well is worth about 60% in self-play, because an open three
    // must be answered too - it turns the main search into a cheap threat
    // search rather than leaving all of that to the root VCT.
    const forcing3 = tag >= P_B4;
    const ext = (forcing3 && ply < 16) ? 1 : 0;
    const nd = depth - 1 + ext;
    let v;
    if(i === 0){
      v = -search(nd, -beta, -alpha, opp, ply + 1);
    } else {
      let red = 0;
      if(!forcing && depth >= 3 && i >= 4 && tag < P_F3 && oppTag < P_F3) red = i >= 10 ? 2 : 1;
      if(red > nd) red = nd;
      v = -search(nd - red, -alpha - 1, -alpha, opp, ply + 1);
      if(v > alpha && (red > 0 || beta > alpha + 1)) v = -search(nd, -beta, -alpha, opp, ply + 1);
    }
    unplace(c);
    if(aborted) return 0;
    if(v > best){ best = v; bestMove = c; }
    if(v > alpha){
      alpha = v;
      if(alpha >= beta){
        if(tag < P_B4){
          const kk = ply * 2;
          if(killer[kk] !== c){ killer[kk + 1] = killer[kk]; killer[kk] = c; }
          const hist = side === BLACK ? histB : histW;
          hist[c] += depth * depth;
        }
        break;
      }
    }
  }
  const flag = best <= alpha0 ? FLAG_UPPER : (best >= beta ? FLAG_LOWER : FLAG_EXACT);
  ttStore(side, depth, best, flag, bestMove, ply);
  return best;
}

// ------------------------------------------------- threat (VCT) solver -----
// A boolean AND/OR search in which the attacker plays only forcing moves
// (fours and open threes) and the defender answers with blocks plus its own
// fours.  Approximate on the defensive side (top-K replies) but a big source of
// strength: it finds the "four-three" style wins renju is made of.
function vctWin(attacker, side, depth, ply){
  if(timeUp()) return false;
  nodes++;
  const defender = attacker === BLACK ? WHITE : BLACK;
  const cntA = attacker === BLACK ? cntB : cntW;
  const cntD = defender === BLACK ? cntB : cntW;
  const off = ply * 24;

  if(side === attacker){
    if(cntA[P_F5] > 0) return true;
    if(cntD[P_F5] > 0){
      const n = collectShape(defender, P_F5, P_F5, scratch, off, 2);
      if(n >= 2) return false;
      const c = scratch[off];
      if(attacker === BLACK && !isLegal(c, BLACK)) return false;
      place(c, attacker);
      const r = vctWin(attacker, defender, depth, ply + 1);
      unplace(c);
      return r;
    }
    if(depth <= 0 || ply >= MAX_PLY - 6) return false;
    const n = genMoves(attacker, ply, 14, -1);
    const mo = ply * 48;
    const bstA = attacker === BLACK ? bstB : bstW;
    for(let i = 0; i < n; i++){
      const c = moveBuf[mo + i] & 0xff;
      if(bstA[c] < P_F3) continue;               // forcing moves only
      place(c, attacker);
      const r = vctWin(attacker, defender, depth - 1, ply + 1);
      unplace(c);
      if(aborted) return false;
      if(r) return true;
    }
    return false;
  }

  // Defender to move.
  if(cntD[P_F5] > 0) return false;               // defender makes five first
  if(cntA[P_F5] > 0){
    const n = collectShape(attacker, P_F5, P_F5, scratch, off, 2);
    if(n >= 2) return true;
    const c = scratch[off];
    if(defender === BLACK && !isLegal(c, BLACK)) return true;
    place(c, defender);
    const r = vctWin(attacker, attacker, depth, ply + 1);
    unplace(c);
    return r;
  }
  if(ply >= MAX_PLY - 6) return false;
  const n = genMoves(defender, ply, VCT_DEFW, -1);
  if(n === 0) return false;
  const mo = ply * 48;
  const cells = [];
  for(let i = 0; i < n; i++) cells.push(moveBuf[mo + i] & 0xff);
  for(let i = 0; i < cells.length; i++){
    const c = cells[i];
    place(c, defender);
    const r = vctWin(attacker, attacker, depth - 1, ply + 1);
    unplace(c);
    if(aborted) return false;
    if(!r) return false;
  }
  return true;
}

function findVctMove(attacker, depth, ply){
  const defender = attacker === BLACK ? WHITE : BLACK;
  const n = genMoves(attacker, ply, 16, -1);
  const mo = ply * 48;
  const bstA = attacker === BLACK ? bstB : bstW;
  const cells = [];
  for(let i = 0; i < n; i++){
    const c = moveBuf[mo + i] & 0xff;
    if(bstA[c] >= P_F3) cells.push(c);
  }
  for(const c of cells){
    place(c, attacker);
    const r = vctWin(attacker, defender, depth - 1, ply + 1);
    unplace(c);
    if(aborted) return -1;
    if(r) return c;
  }
  return -1;
}

// ------------------------------------------------------------- root --------
function principalVariation(side, maxLen){
  const pv = [], played = [];
  let s = side;
  for(let i = 0; i < maxLen; i++){
    const ix = ttProbe(s);
    if(ix < 0) break;
    const c = ttInfo[ix] & 0xff;
    if(board[c] !== EMPTY || !isLegal(c, s)) break;
    pv.push(c);
    const wins = isWinningPoint(c, s);
    place(c, s); played.push(c);
    s = s === BLACK ? WHITE : BLACK;
    if(wins) break;
  }
  for(let i = played.length - 1; i >= 0; i--) unplace(played[i]);
  return pv;
}

/**
 * opts = { side, timeMs, maxDepth, wantCandidates }
 * returns { cell, score, depth, seldepth, nodes, mate, pv, mode, candidates }
 */
function think(opts){
  const side = opts.side;
  const opp = side === BLACK ? WHITE : BLACK;
  const timeMs = Math.max(50, opts.timeMs || 1000);
  const maxDepth = Math.min(40, opts.maxDepth || 16);
  nodes = 0; aborted = false; seldepth = 0;
  deadline = now() + timeMs;
  ttGen = (ttGen % 120) + 1;
  killer.fill(-1);
  for(let i = 0; i < NN; i++){ histB[i] >>= 2; histW[i] >>= 2; }

  const cntMe = side === BLACK ? cntB : cntW;
  const cntOp = side === BLACK ? cntW : cntB;

  if(winner) return { cell: -1, score: 0, depth: 0, seldepth: 0, nodes: 0, mate: 0, pv: [], mode: 'game over', candidates: [] };
  if(moveList.length === 0)
    return { cell: 112, score: 0, depth: 0, seldepth: 0, nodes: 0, mate: 0, pv: [112], mode: 'opening', candidates: [{ cell: 112, score: 0, tag: 'centre' }] };

  // White's reply to a lone stone is a book move, not a search. With one stone
  // of each colour the evaluation is exactly symmetric - sumB === sumW whatever
  // the distance between them - so every candidate scores the same and the
  // search picks arbitrarily, often two or three lines away. Every named renju
  // opening has White adjacent to Black's first stone, so play the ring.
  if(moveList.length === 1 && side === WHITE){
    const b = moveList[0], bx = b % N, by = (b / N) | 0;
    const ring = [];
    let bestCentre = -1;
    for(let dy = -1; dy <= 1; dy++){
      for(let dx = -1; dx <= 1; dx++){
        if(!dx && !dy) continue;
        const x = bx + dx, y = by + dy;
        if(x < 0 || x >= N || y < 0 || y >= N) continue;
        const c = y * N + x;
        if(CENTREV[c] > bestCentre) bestCentre = CENTREV[c];
        ring.push(c);
      }
    }
    const best = ring.filter(c => CENTREV[c] === bestCentre);
    if(best.length){
      // Vary it so the practice room does not replay the same game every time.
      const cell = opts.varyOpening ? best[(Math.random() * best.length) | 0] : best[0];
      return { cell, score: 0, depth: 0, seldepth: 0, nodes: 0, mate: 0,
               pv: [cell], mode: 'opening', candidates: [{ cell, score: 0, tag: 'restrain' }] };
    }
  }

  // Immediate five.
  if(cntMe[P_F5] > 0){
    const n = collectShape(side, P_F5, P_F5, scratch, 0, 1);
    if(n) return { cell: scratch[0], score: MATE - 1, depth: 1, seldepth: 1, nodes: 0, mate: 1,
                   pv: [scratch[0]], mode: 'five', candidates: [{ cell: scratch[0], score: MATE - 1, tag: 'win' }] };
  }
  // Forced block.
  let forcedBlock = -1;
  if(cntOp[P_F5] > 0){
    const n = collectShape(opp, P_F5, P_F5, scratch, 0, 2);
    if(n === 1 && isLegal(scratch[0], side)) forcedBlock = scratch[0];
  }

  const rootCells = [];
  if(forcedBlock >= 0){
    rootCells.push(forcedBlock);
    // The move is forced; spend a fraction of the budget just to get a score.
    deadline = now() + Math.min(timeMs, 600);
  } else {
    const rootN = genMoves(side, 0, ROOTW, -1);
    for(let i = 0; i < rootN; i++) rootCells.push(moveBuf[i] & 0xff);
  }
  if(!rootCells.length){
    for(let c = 0; c < NN; c++) if(board[c] === EMPTY && isLegal(c, side)){ rootCells.push(c); break; }
    if(!rootCells.length) return { cell: -1, score: 0, depth: 0, seldepth: 0, nodes: 0, mate: 0, pv: [], mode: 'no move', candidates: [] };
  }
  const savedDeadline = deadline;

  if(forcedBlock < 0){
    // Threat search: try to prove a forced win before spending time on alpha-beta.
    deadline = now() + Math.min(timeMs * 0.30, 1500);
    let vctCell = -1, vctDepth = 0;
    for(let d = 3; d <= VCT_MAXD && !aborted; d += 2){
      const c = findVctMove(side, d, 1);
      if(c >= 0){ vctCell = c; vctDepth = d; break; }
    }
    aborted = false;
    deadline = savedDeadline;
    if(vctCell >= 0){
      return { cell: vctCell, score: MATE - 20, depth: vctDepth, seldepth, nodes, mate: 1,
               pv: [vctCell], mode: 'forced win', candidates: [{ cell: vctCell, score: MATE - 20, tag: 'forced win' }] };
    }

    // Defensive probe: would the opponent have a forced win if handed a free
    // move?  If so, pull the root moves that break the sequence to the front,
    // where alpha-beta will look at them first.
    deadline = now() + Math.min(timeMs * 0.14, 700);
    const threatened = findVctMove(opp, 7, 1) >= 0;
    aborted = false;
    if(threatened){
      deadline = now() + Math.min(timeMs * 0.22, 1100);
      const survivors = [], rest = [];
      for(let i = 0; i < rootCells.length; i++){
        const c = rootCells[i];
        if(i >= 12 || now() >= deadline){ rest.push(c); continue; }
        if(!isLegal(c, side)){ rest.push(c); continue; }
        place(c, side);
        const stillLost = findVctMove(opp, 5, 1) >= 0;
        unplace(c);
        (stillLost ? rest : survivors).push(c);
      }
      aborted = false;
      if(survivors.length && survivors.length < rootCells.length){
        rootCells.length = 0;
        for(const c of survivors) rootCells.push(c);
        for(const c of rest) rootCells.push(c);
      }
    }
    aborted = false;
    deadline = savedDeadline;
  }

  const scores = new Array(rootCells.length).fill(-Infinity);
  let bestCell = rootCells[0], bestScore = -Infinity, bestDepth = 0;
  // An odd-depth search gives one side an extra move, so consecutive iterations
  // straddle the true value ("odd-even effect"). The move always comes from the
  // deepest iteration; the number shown to the user is the mean of the last two
  // iterations, which removes most of that swing without hiding real changes.
  let prevScore = null, prevPrev = null, fullRanking = null;
  for(let depth = 2; depth <= maxDepth; depth++){
    let alpha = -Infinity, beta = Infinity;
    let localBest = -1, localScore = -Infinity, completed = 0;
    const localScores = new Array(rootCells.length).fill(-Infinity);
    for(let i = 0; i < rootCells.length; i++){
      const c = rootCells[i];
      place(c, side);
      let v;
      if(i === 0){
        v = -search(depth - 1, -beta, -alpha, opp, 1);
      } else {
        v = -search(depth - 1, -alpha - 1, -alpha, opp, 1);
        if(v > alpha) v = -search(depth - 1, -beta, -alpha, opp, 1);
      }
      unplace(c);
      if(aborted) break;
      completed++;
      localScores[i] = v;
      if(v > localScore){ localScore = v; localBest = c; }
      if(v > alpha) alpha = v;
    }
    // Safe to commit even when the iteration was cut short: the previous best
    // move is searched first, so localBest either is it or beat it *at this
    // depth*. (Comparing against the previous depth's score instead would fall
    // foul of the odd-even swing.)
    if(completed > 0 && localBest >= 0){
      bestCell = localBest; bestScore = localScore; bestDepth = depth;
      prevPrev = prevScore; prevScore = localScore;
      if(completed === rootCells.length){
        fullRanking = rootCells.map((c, i) => ({ cell: c, score: localScores[i] }))
          .filter(e => e.score > -Infinity).sort((a, b) => b.score - a.score);
      }
      for(let i = 0; i < rootCells.length; i++) if(localScores[i] > -Infinity) scores[i] = localScores[i];
      // Re-order for the next iteration.
      const order = rootCells.map((c, i) => [c, scores[i]]);
      order.sort((a, b) => b[1] - a[1]);
      for(let i = 0; i < order.length; i++){ rootCells[i] = order[i][0]; scores[i] = order[i][1]; }
    }
    if(aborted) break;
    if(bestScore >= MATE - 200 || bestScore <= -(MATE - 200)) break;
    if(now() >= deadline) break;
  }

  place(bestCell, side);
  const pvTail = principalVariation(opp, 8);
  unplace(bestCell);
  const pv = [bestCell, ...pvTail];

  const ranking = fullRanking || rootCells.map((c, i) => ({ cell: c, score: scores[i] })).filter(e => e.score > -Infinity);
  const candidates = ranking.slice(0, 8)
    .map(e => ({ cell: e.cell, score: e.score, tag: shapeTagFor(e.cell, side) }));

  let mate = 0;
  if(bestScore >= MATE - 200) mate = Math.ceil((MATE - bestScore) / 2);
  else if(bestScore <= -(MATE - 200)) mate = -Math.ceil((MATE + bestScore) / 2);
  let shownScore = bestScore;
  if(mate === 0 && prevPrev !== null && Math.abs(prevPrev) < MATE_BAND)
    shownScore = Math.round((prevScore + prevPrev) / 2);

  return { cell: bestCell, score: shownScore, depth: bestDepth, seldepth, nodes, mate,
           pv, mode: aborted ? 'search' : 'search (complete)', candidates };
}

function shapeTagFor(c, color){
  if(board[c] !== EMPTY) return '';
  const bst = color === BLACK ? bstB : bstW;
  const sec = color === BLACK ? secB : secW;
  const obst = color === BLACK ? bstW : bstB;
  const b = bst[c], s = sec[c];
  if(b === P_F5) return 'five';
  if(isFour(b) && isFour(s)) return 'double four';
  if(isFour(b) && s === P_F3) return 'four-three';
  if(b === P_F3 && s === P_F3) return 'double three';
  if(b === P_D4) return 'double four';
  if(b === P_F4) return 'open four';
  if(b === P_B4) return 'four';
  if(b === P_F3) return 'open three';
  if(obst[c] === P_F5) return 'block five';
  if(isFour(obst[c])) return 'block four';
  if(obst[c] === P_F3) return 'block three';
  return SHAPE_NAME[b];
}

/** All empty points that Black is not allowed to play (for the UI markers). */
function forbiddenPoints(){
  const out = [];
  for(let c = 0; c < NN; c++){
    if(board[c] !== EMPTY) continue;
    if(qforb[c] === 0) continue;
    if(qforb[c] === 1){ out.push(c); continue; }
    if(blackForbidden(c, 0)) out.push(c);
  }
  return out;
}

function shapeAt(c, color){
  const pat = color === BLACK ? patB : patW;
  return [pat[c], pat[NN + c], pat[2 * NN + c], pat[3 * NN + c]];
}

resetBoard();

return {
  N, NN, EMPTY, BLACK, WHITE, MATE,
  P_NONE, P_B2, P_F2, P_B3, P_F3, P_B4, P_F4, P_D4, P_F5, P_OL, SHAPE_NAME,
  reset: resetBoard,
  setMoves,
  play,
  undo,
  isLegal,
  isWinningPoint,
  forbiddenPoints,
  forbiddenReason,
  evaluate,
  think,
  shapeAt,
  shapeTagFor,
  board: () => board,
  moves: () => moveList.slice(),
  winner: () => winner,
  winLine: () => winLine,
  counts: () => ({ black: Array.from(cntB), white: Array.from(cntW), sumB, sumW, forbCnt })
};
}
// ==== RENJU CORE END ====
