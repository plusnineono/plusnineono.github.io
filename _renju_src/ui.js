// ==== RENJU UI BEGIN ====
(() => {
  const core = RenjuCore();                 // main-thread copy: rules, legality, markers
  const N = core.N, BLACK = core.BLACK, WHITE = core.WHITE, EMPTY = core.EMPTY;
  const MATE = core.MATE;
  const LETTERS = 'ABCDEFGHJKLMNOP';
  const coordName = c => `${LETTERS[c % N]}${N - ((c / N) | 0)}`;

  // ---------------------------------------------------------------- worker --
  const WORKER_SRC = `${RenjuCore.toString()}
const core = RenjuCore();
self.onmessage = (ev) => {
  const m = ev.data;
  if(m.cmd !== 'think') return;
  core.setMoves(m.moves);
  const r = core.think({ side: m.side, timeMs: m.timeMs, maxDepth: m.maxDepth, varyOpening: m.varyOpening });
  self.postMessage({ id: m.id, result: r });
};`;

  let worker = null, workerUrl = null, reqId = 0, workerBroken = false;
  const pending = new Map();

  function spawnWorker(){
    if(workerBroken || typeof Worker === 'undefined') return null;
    try {
      if(!workerUrl) workerUrl = URL.createObjectURL(new Blob([WORKER_SRC], { type: 'text/javascript' }));
      const w = new Worker(workerUrl);
      w.onmessage = ev => {
        const { id, result } = ev.data;
        const entry = pending.get(id);
        pending.delete(id);
        if(entry) entry.resolve(result);
      };
      w.onerror = () => { workerBroken = true; closeWorker('error'); };
      return w;
    } catch(e){ workerBroken = true; return null; }
  }
  function closeWorker(reason){
    if(worker){ try { worker.terminate(); } catch(e){} }
    worker = null;
    for(const [, entry] of pending) entry.resolve({ __stopped: reason });
    pending.clear();
  }
  /** Throw away whatever the engine is chewing on (used on undo / new game). */
  function abortThinking(){ closeWorker('abort'); }

  function thinkHere(moves, side, timeMs, maxDepth){
    core.setMoves(moves);
    const r = core.think({ side, timeMs, maxDepth, varyOpening: true });
    syncCore();
    return r;
  }

  /**
   * Ask the engine for a move. Resolves to null if the request was cancelled.
   * If the Worker cannot be used at all, the search runs on the main thread with
   * a much smaller budget so the page never locks up for seconds at a time.
   */
  async function think(moves, side, timeMs, maxDepth){
    if(!worker) worker = spawnWorker();
    if(worker){
      const id = ++reqId;
      const p = new Promise(resolve => pending.set(id, { resolve }));
      worker.postMessage({ cmd: 'think', id, moves, side, timeMs, maxDepth, varyOpening: true });
      const r = await p;
      if(r && !r.__stopped) return r;
      if(r && r.__stopped === 'abort') return null;
    }
    return thinkHere(moves, side, Math.min(timeMs, 1200), maxDepth);
  }

  // ----------------------------------------------------------------- state --
  const game = { moves: [], winner: 0, winLine: null };
  let humanColor = BLACK;
  let busy = false, analysisToken = 0;
  let showNumbers = true, showEval = true, showCandidates = false;
  let lastAnalysis = null;

  const LEVELS = {
    quick:  { ms: 500,   depth: 12, label: 'Quick' },
    normal: { ms: 2000,  depth: 18, label: 'Normal' },
    strong: { ms: 6000,  depth: 24, label: 'Strong' },
    deep:   { ms: 15000, depth: 30, label: 'Deep' }
  };
  let level = 'normal';

  const $ = id => document.getElementById(id);
  const canvas = $('board'), ctx = canvas.getContext('2d');
  const els = {
    turn: $('turnVal'), result: $('resultVal'), depth: $('depthVal'), nodes: $('nodesVal'),
    banner: $('resultBanner'), engine: $('engineVal'),
    barBlack: $('winRateBlack'), barWhite: $('winRateWhite'),
    labBlack: $('winRateBlackLabel'), labWhite: $('winRateWhiteLabel'),
    evalVal: $('evalVal'), evalNote: $('evalNote'), tip: $('forbidTip'),
    cand: $('cand'), candBtn: $('candidatesToggleBtn'), evalBtn: $('evalToggleBtn'),
    evalPanel: $('evalPanel'), numbersBtn: $('numbersToggleBtn'), sideLabel: $('sideLabel')
  };

  function syncCore(){ core.setMoves(game.moves); }

  // --------------------------------------------------------- full-bleed ----
  // Quarto's grid caps the article column well short of the window. Measure the
  // element's natural left edge and the viewport width (documentElement's
  // clientWidth already excludes the scrollbar, so this cannot introduce a
  // horizontal one) and stretch the app across the whole window.
  const appEl = document.querySelector('.renju-app');
  let fittedW = -1, fittedH = -1;
  function fitToWindow(force){
    if(!appEl || !appEl.getBoundingClientRect) return;
    const vw = document.documentElement ? document.documentElement.clientWidth : 0;
    const vh = self.innerHeight || 0;
    if(!vw || (!force && vw === fittedW && vh === fittedH)) return;  // measuring forces a reflow
    fittedW = vw; fittedH = vh;
    appEl.style.marginLeft = '';
    appEl.style.width = '';
    const left = appEl.getBoundingClientRect().left;
    appEl.style.marginLeft = `${-left}px`;
    appEl.style.width = `${vw}px`;
    // Cap the board by the height that is actually on screen at scroll 0, so it
    // never runs off the bottom of the window on any device.
    const box = canvas.parentElement;
    if(vh && box && box.getBoundingClientRect){
      canvas.style.maxWidth = '';
      const top = box.getBoundingClientRect().top + (self.scrollY || 0);
      let limit = vh - top - 20;
      // Below 900px the panels sit under the board, so leave part of the screen
      // for them instead of pushing the controls off the bottom.
      if(vw < 900) limit = Math.min(limit, Math.round(vh * 0.62));
      canvas.style.maxWidth = `${Math.max(240, limit)}px`;
    }
  }

  // ------------------------------------------------------------- rendering --
  let geom = { pad: 0, cellPx: 0, size: 0 };
  function layout(){
    const dpr = Math.min(3, self.devicePixelRatio || 1);
    const cssSize = canvas.clientWidth || 640;
    const want = Math.round(cssSize * dpr);
    // Assigning width/height clears the canvas, so only do it when it changed.
    if(canvas.width !== want){ canvas.width = want; canvas.height = want; }
    const s = canvas.width;
    geom.size = s;
    geom.pad = Math.round(s * 0.052);
    geom.cellPx = (s - 2 * geom.pad) / (N - 1);
  }
  const px = i => geom.pad + i * geom.cellPx;

  let forbidCache = { key: '', cells: [] };
  function forbiddenCells(){
    if(humanColor !== BLACK) return [];
    if(game.winner) return [];
    if(game.moves.length % 2 !== 0) return [];      // not Black's turn
    const key = game.moves.join(',');
    if(forbidCache.key === key) return forbidCache.cells;
    syncCore();
    forbidCache = { key, cells: core.forbiddenPoints() };
    return forbidCache.cells;
  }

  function draw(){
    fitToWindow();
    layout();
    const s = geom.size, cp = geom.cellPx;
    const styles = getComputedStyle(document.querySelector('.renju-app'));
    ctx.fillStyle = (styles.getPropertyValue('--board') || '#d8b574').trim();
    ctx.fillRect(0, 0, s, s);

    ctx.strokeStyle = 'rgba(90,56,23,0.85)';
    ctx.lineWidth = Math.max(1, s / 900);
    for(let i = 0; i < N; i++){
      ctx.beginPath(); ctx.moveTo(px(0), px(i)); ctx.lineTo(px(N - 1), px(i)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px(i), px(0)); ctx.lineTo(px(i), px(N - 1)); ctx.stroke();
    }
    ctx.fillStyle = '#5a3817';
    for(const sy of [3, 7, 11]) for(const sx of [3, 7, 11]){
      ctx.beginPath(); ctx.arc(px(sx), px(sy), cp * 0.09, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = 'rgba(59,42,22,0.9)';
    ctx.font = `${Math.round(cp * 0.38)}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for(let i = 0; i < N; i++){
      ctx.fillText(LETTERS[i], px(i), geom.pad * 0.45);
      ctx.fillText(String(N - i), geom.pad * 0.45, px(i));
    }

    // forbidden points for a human playing Black
    const forb = forbiddenCells();
    if(forb.length){
      ctx.strokeStyle = 'rgba(214,48,48,0.75)';
      ctx.lineWidth = Math.max(1.5, cp * 0.055);
      const r = cp * 0.2;
      for(const c of forb){
        const x = px(c % N), y = px((c / N) | 0);
        ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
        ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke();
      }
    }

    const winSet = new Set(game.winLine || []);
    const rStone = cp * 0.44;
    for(let i = 0; i < game.moves.length; i++){
      const c = game.moves[i], colour = i % 2 === 0 ? BLACK : WHITE;
      const x = px(c % N), y = px((c / N) | 0);
      ctx.beginPath(); ctx.arc(x, y, rStone, 0, Math.PI * 2);
      const g = ctx.createRadialGradient(x - rStone * 0.35, y - rStone * 0.35, rStone * 0.15, x, y, rStone * 1.1);
      if(colour === BLACK){ g.addColorStop(0, '#6a7688'); g.addColorStop(1, '#0d0f12'); }
      else { g.addColorStop(0, '#ffffff'); g.addColorStop(1, '#c9d1db'); }
      ctx.fillStyle = g; ctx.fill();
      ctx.lineWidth = Math.max(0.6, cp * 0.02);
      ctx.strokeStyle = colour === BLACK ? '#000' : '#93a0ae'; ctx.stroke();
      if(winSet.has(c)){
        ctx.strokeStyle = '#ffd166'; ctx.lineWidth = Math.max(2, cp * 0.08);
        ctx.beginPath(); ctx.arc(x, y, rStone * 0.96, 0, Math.PI * 2); ctx.stroke();
      }
      if(showNumbers){
        ctx.fillStyle = colour === BLACK ? '#e8edf5' : '#1a2028';
        ctx.font = `${Math.round(cp * 0.34)}px system-ui, sans-serif`;
        ctx.fillText(String(i + 1), x, y);
      }
    }
    if(game.moves.length){
      const c = game.moves[game.moves.length - 1];
      const x = px(c % N), y = px((c / N) | 0);
      ctx.strokeStyle = '#ff5252'; ctx.lineWidth = Math.max(1.5, cp * 0.06);
      ctx.beginPath(); ctx.arc(x, y, rStone * 1.22, 0, Math.PI * 2); ctx.stroke();
    }
  }

  // ------------------------------------------------------- score reporting --
  // The engine reports a score in "threat points" from the side to move's point
  // of view, or a mate score.  Everything the user sees is derived from that one
  // number, so the bar, the label and the PV can never disagree.
  const WIN_SCALE = 2500;
  function blackScoreOf(res){
    if(!res || typeof res.score !== 'number') return null;
    const side = res.side;
    return side === BLACK ? res.score : -res.score;
  }
  function isMateScore(s){ return Math.abs(s) >= MATE - 200; }
  function winRateFor(blackScore){
    if(blackScore >= MATE - 200) return 100;
    if(blackScore <= -(MATE - 200)) return 0;
    const p = 1 / (1 + Math.exp(-blackScore / WIN_SCALE));
    return Math.max(1, Math.min(99, Math.round(p * 100)));
  }
  function formatScore(blackScore){
    if(isMateScore(blackScore)){
      const plies = MATE - Math.abs(blackScore);
      const movesToWin = Math.max(1, Math.ceil(plies / 2));
      return `${blackScore > 0 ? 'Black' : 'White'} wins in ${movesToWin}`;
    }
    const v = Math.round(blackScore);
    if(Math.abs(v) < 150) return 'Level';
    return `${v > 0 ? 'Black' : 'White'} +${Math.abs(v)}`;
  }

  function renderEvaluation(){
    if(game.winner){
      const rate = game.winner === BLACK ? 100 : 0;
      paintBar(rate, game.winner === BLACK ? 'Black wins' : 'White wins', '');
      return;
    }
    if(!lastAnalysis){ paintBar(50, 'Level', 'thinking…'); return; }
    const bs = blackScoreOf(lastAnalysis);
    if(bs === null){ paintBar(50, 'Level', ''); return; }
    const note = `depth ${lastAnalysis.depth}${lastAnalysis.seldepth ? '/' + lastAnalysis.seldepth : ''} · ${fmtNodes(lastAnalysis.nodes)} nodes`;
    paintBar(winRateFor(bs), formatScore(bs), note, lastAnalysis.pv || []);
    renderCandidates(lastAnalysis);
  }
  function fmtNodes(n){
    if(n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if(n >= 1e3) return (n / 1e3).toFixed(0) + 'k';
    return String(n);
  }
  function paintBar(blackRate, label, note){
    els.barBlack.style.width = `${blackRate}%`;
    els.barWhite.style.width = `${100 - blackRate}%`;
    els.labBlack.textContent = `Black ${blackRate}%`;
    els.labWhite.textContent = `White ${100 - blackRate}%`;
    els.evalVal.textContent = label;
    els.evalVal.style.color = blackRate > 57 ? 'var(--good)' : blackRate < 43 ? '#ffb4b4' : 'var(--text)';
    els.evalNote.textContent = note;
  }
  function renderCandidates(res){
    if(!showCandidates){ return; }
    const list = res && res.candidates ? res.candidates : [];
    if(!list.length){ els.cand.textContent = 'No analysis yet.'; return; }
    const flip = res.side === BLACK ? 1 : -1;
    els.cand.innerHTML = list.map((m, i) => {
      const bs = m.score * flip;
      // Only the first move gets an exact score; alpha-beta proves the rest are
      // merely "no better than this", so they are shown with a bound.
      const num = isMateScore(bs) ? formatScore(bs) : (Math.round(bs) >= 0 ? '+' : '') + Math.round(bs);
      const txt = i === 0 ? num : (res.side === BLACK ? '\u2264 ' : '\u2265 ') + num;
      return `<span class="candRow"><b>${i + 1}.</b> ${coordName(m.cell)} <span class="candTag">${m.tag || ''}</span> <span class="candScore">${txt}</span></span>`;
    }).join('');
  }

  // ----------------------------------------------------------- status bar --
  function updateStatus(){
    const toMove = game.moves.length % 2 === 0 ? BLACK : WHITE;
    els.turn.textContent = game.winner ? '—' : (toMove === BLACK ? 'Black' : 'White');
    els.result.textContent = game.winner
      ? (game.winner === BLACK ? 'Black wins' : 'White wins')
      : (game.moves.length >= 225 ? 'Draw' : 'Playing');
    els.engine.textContent = busy ? 'Thinking…' : 'Idle';
    if(els.sideLabel){
      els.sideLabel.textContent = humanColor === BLACK ? 'You play Black' : 'You play White';
      if(els.sideLabel.style) els.sideLabel.style.setProperty('--stone', humanColor === BLACK ? '#14171b' : '#f2f5f9');
    }
    els.depth.textContent = lastAnalysis ? String(lastAnalysis.depth || 0) : '0';
    els.nodes.textContent = lastAnalysis ? fmtNodes(lastAnalysis.nodes || 0) : '0';
    els.banner.className = 'resultBanner';
    if(game.winner){
      els.banner.textContent = game.winner === BLACK ? 'Black wins' : 'White wins';
      els.banner.classList.add('show', game.winner === BLACK ? 'win-black' : 'win-white');
    } else if(game.moves.length >= 225){
      els.banner.textContent = 'Draw'; els.banner.classList.add('show');
    } else {
      els.banner.textContent = '';
    }
    draw();
  }

  // -------------------------------------------------------------- gameplay --
  function pushMove(c){
    syncCore();
    const colour = game.moves.length % 2 === 0 ? BLACK : WHITE;
    if(!core.play(c, colour)) return false;
    game.moves.push(c);
    game.winner = core.winner();
    game.winLine = core.winLine();
    return true;
  }

  /**
   * Play one engine move - or two, when the engine has just played the human's
   * own colour. "Engine move" then means "play this one for me", and the
   * opponent answers as usual, so when the engine is idle it is always the
   * human's turn again. Undo relies on that: it unwinds back to the human's
   * turn, and without the pairing a single Undo could take back two moves the
   * human never made (on a nearly empty board, the whole board).
   */
  async function engineMove(){
    if(busy || game.winner) return;
    busy = true; analysisToken++;
    abortThinking();                        // drop any background analysis still queued
    updateStatus();
    const lv = LEVELS[level];
    for(let step = 0; step < 2; step++){
      const side = game.moves.length % 2 === 0 ? BLACK : WHITE;
      const res = await think(game.moves.slice(), side, lv.ms, lv.depth);
      if(!res || res.cell < 0) break;
      lastAnalysis = { ...res, side };
      if(!pushMove(res.cell)) break;
      updateStatus();
      renderEvaluation();
      if(game.winner) break;
      if(side !== humanColor) break;        // the engine played its own colour: done
    }
    busy = false;
    updateStatus();
    if(!game.winner) scheduleAnalysis();
  }

  /** Background analysis of the position the human is now looking at. */
  function scheduleAnalysis(){
    if(!showEval && !showCandidates) return;
    if(game.winner) return;
    const token = ++analysisToken;
    const side = game.moves.length % 2 === 0 ? BLACK : WHITE;
    const lv = LEVELS[level];
    const ms = workerBroken ? 250 : Math.min(1600, Math.max(400, lv.ms / 2));
    const snapshot = game.moves.slice();
    think(snapshot, side, ms, lv.depth).then(res => {
      if(token !== analysisToken || !res) return;
      if(snapshot.join() !== game.moves.join()) return;
      lastAnalysis = { ...res, side };
      renderEvaluation();
      updateStatus();
    });
  }

  function cellFromEvent(ev){
    const rect = canvas.getBoundingClientRect();
    const scale = canvas.width / rect.width;
    const mx = (ev.clientX - rect.left) * scale, my = (ev.clientY - rect.top) * scale;
    const x = Math.round((mx - geom.pad) / geom.cellPx), y = Math.round((my - geom.pad) / geom.cellPx);
    if(x < 0 || x >= N || y < 0 || y >= N) return -1;
    if(Math.hypot(mx - px(x), my - px(y)) > geom.cellPx * 0.5) return -1;
    return y * N + x;
  }

  let pressCell = -1, pressX = 0, pressY = 0;
  function onPress(ev){
    pressCell = cellFromEvent(ev);
    pressX = ev.clientX; pressY = ev.clientY;
  }
  async function onRelease(ev){
    const start = pressCell;
    pressCell = -1;
    if(start < 0) return;
    if(Math.hypot(ev.clientX - pressX, ev.clientY - pressY) > 14) return;  // a drag, not a tap
    if(cellFromEvent(ev) !== start) return;
    if(busy || game.winner) return;
    const toMove = game.moves.length % 2 === 0 ? BLACK : WHITE;
    if(toMove !== humanColor) return;
    syncCore();
    if(!core.isLegal(start, toMove)){
      if(core.board()[start] === EMPTY){ flashIllegal(start); explainForbidden(start); }
      return;
    }
    hideTip();
    if(!pushMove(start)) return;
    analysisToken++;
    updateStatus();
    if(game.winner){ renderEvaluation(); return; }
    await engineMove();
  }
  let tipTimer = 0;
  function hideTip(){
    if(!els.tip || !els.tip.classList) return;
    els.tip.classList.remove('show');
    if(tipTimer) clearTimeout(tipTimer);
  }
  /** Say *why* a red cross is a red cross, next to the point that was tapped. */
  function explainForbidden(c){
    const tip = els.tip;
    if(!tip || !tip.classList || !canvas.getBoundingClientRect) return;
    syncCore();
    const why = core.forbiddenReason(c);
    if(!why) return;
    tip.innerHTML = `<b>${coordName(c)} is forbidden for Black.</b><br>${why}`;
    const box = canvas.parentElement;
    const cRect = canvas.getBoundingClientRect();
    const bRect = box && box.getBoundingClientRect ? box.getBoundingClientRect() : cRect;
    const scale = cRect.width / (canvas.width || 1);
    let x = cRect.left - bRect.left + px(c % N) * scale;
    const y = cRect.top - bRect.top + px((c / N) | 0) * scale;
    x = Math.max(120, Math.min(bRect.width - 120, x));
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
    tip.style.transform = y < 100 ? 'translate(-50%, 30%)' : 'translate(-50%, -125%)';
    tip.classList.add('show');
    if(tipTimer) clearTimeout(tipTimer);
    tipTimer = setTimeout(() => tip.classList.remove('show'), 5000);
  }

  function flashIllegal(c){
    draw();
    const x = px(c % N), y = px((c / N) | 0);
    ctx.strokeStyle = 'rgba(255,80,80,0.95)';
    ctx.lineWidth = Math.max(2, geom.cellPx * 0.09);
    ctx.beginPath(); ctx.arc(x, y, geom.cellPx * 0.42, 0, Math.PI * 2); ctx.stroke();
    setTimeout(draw, 420);
  }

  // ---------------------------------------------------------------- buttons --
  function newGame(){
    abortThinking();
    hideTip();
    busy = false; analysisToken++;
    game.moves = []; game.winner = 0; game.winLine = null;
    lastAnalysis = null; forbidCache = { key: '', cells: [] };
    syncCore();
    updateStatus(); renderEvaluation();
    if(humanColor !== BLACK) engineMove(); else scheduleAnalysis();
  }
  function undo(){
    abortThinking(); hideTip(); busy = false; analysisToken++;
    // step back to the human's turn
    if(game.moves.length) game.moves.pop();
    const toMove = () => game.moves.length % 2 === 0 ? BLACK : WHITE;
    if(game.moves.length && toMove() !== humanColor) game.moves.pop();
    game.winner = 0; game.winLine = null; lastAnalysis = null;
    syncCore();
    updateStatus(); renderEvaluation(); scheduleAnalysis();
  }
  $('newGameBtn').onclick = newGame;
  $('undoBtn').onclick = undo;
  $('engineMoveBtn').onclick = () => engineMove();
  $('switchBtn').onclick = () => {
    humanColor = humanColor === BLACK ? WHITE : BLACK;
    forbidCache = { key: '', cells: [] };
    updateStatus();
    const toMove = game.moves.length % 2 === 0 ? BLACK : WHITE;
    if(!game.winner && toMove !== humanColor) engineMove();
  };
  $('levelSel').onchange = e => { level = e.target.value; };
  els.evalBtn.onclick = () => {
    showEval = !showEval;
    els.evalBtn.textContent = showEval ? 'On' : 'Off';
    els.evalPanel.classList.toggle('hidden', !showEval);
    if(showEval) scheduleAnalysis();
  };
  els.candBtn.onclick = () => {
    showCandidates = !showCandidates;
    els.candBtn.textContent = showCandidates ? 'On' : 'Off';
    els.cand.classList.toggle('hidden', !showCandidates);
    if(showCandidates){ renderCandidates(lastAnalysis); scheduleAnalysis(); }
  };
  els.numbersBtn.onclick = () => {
    showNumbers = !showNumbers;
    els.numbersBtn.textContent = showNumbers ? 'On' : 'Off';
    draw();
  };
  // Small hook for debugging from the browser console.
  self.__renju = () => ({ moves: game.moves.slice(), winner: game.winner, analysis: lastAnalysis });

  canvas.addEventListener('pointerdown', onPress);
  canvas.addEventListener('pointerup', onRelease);
  canvas.addEventListener('pointercancel', () => { pressCell = -1; });
  self.addEventListener('resize', () => { fitToWindow(true); draw(); });

  newGame();
  self.addEventListener('load', () => { fitToWindow(true); draw(); });
  if(typeof requestAnimationFrame === 'function') requestAnimationFrame(draw);
})();
// ==== RENJU UI END ====
