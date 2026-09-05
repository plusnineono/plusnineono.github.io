// Headless smoke test of the generated page script: builds a fake DOM, runs the
// script, plays a few moves and checks the UI state that the user sees.
// Runs the generated page against a fake DOM: catches wiring mistakes that the
// core tests cannot see. Worker creation fails under Deno, which exercises the
// main-thread fallback path.
const qmd = await Deno.readTextFile(new URL('../renju_engine.qmd', import.meta.url));
const src = qmd.slice(qmd.indexOf('<script>') + 8, qmd.lastIndexOf('</script>'));

const noop = () => {};
function makeCtx(){
  return new Proxy({}, { get: (t, k) => {
    if(k === 'createRadialGradient') return () => ({ addColorStop: noop });
    if(k === 'measureText') return () => ({ width: 10 });
    return (typeof k === 'string' && k.startsWith('_')) ? undefined : (() => undefined);
  }, set: () => true });
}
const listeners = new Map();
function el(id){
  const e = {
    id, textContent: '', innerHTML: '', value: '', style: { setProperty(){}, removeProperty(){} },
    className: '', clientWidth: 640, width: 640, height: 640,
    classList: { _s: new Set(), add(...a){ a.forEach(x=>this._s.add(x)); }, remove(...a){ a.forEach(x=>this._s.delete(x)); },
                 toggle(x, on){ on ? this._s.add(x) : this._s.delete(x); }, contains(x){ return this._s.has(x); } },
    getContext: () => makeCtx(),
    parentElement: null,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 640, height: 640 }),
    addEventListener(type, fn){ listeners.set(id + ':' + type, fn); },
    onclick: null, onchange: null
  };
  return e;
}
const nodes = new Map();
const appEl = el('renju-app');
appEl.getBoundingClientRect = () => ({ left: 120, top: 0, width: 900, height: 900 });
globalThis.document = {
  getElementById: id => { if(!nodes.has(id)) nodes.set(id, el(id)); return nodes.get(id); },
  querySelector: () => appEl, addEventListener: noop,
  documentElement: { clientWidth: 1440, style: { setProperty: noop } }
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '#d8b574' });
globalThis.devicePixelRatio = 1;
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
globalThis.addEventListener = noop;
// give the canvas a parent so the height cap and the tooltip can be positioned
const boardBox = el('boardBox');
boardBox.getBoundingClientRect = () => ({ left: 0, top: 180, width: 640, height: 640 });

// the canvas is created lazily by getElementById; pre-create and wire it up
const canvasEl = document.getElementById('board');
canvasEl.parentElement = boardBox;
new Function(src)();

const get = id => nodes.get(id);
const cell = (x, y) => y * 15 + x;
const at = i => Math.round(0.052 * 640) + i * (640 - 2 * Math.round(0.052 * 640)) / 14;
const click = (x, y) => {
  const ev = { preventDefault: noop, clientX: at(x), clientY: at(y) };
  listeners.get('board:pointerdown')(ev);
  return listeners.get('board:pointerup')(ev);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

let fail = 0;
const ok = (c, m) => { if(!c){ console.log('  FAIL: ' + m); fail++; } };

get('levelSel').value = 'quick';
get('levelSel').onchange({ target: { value: 'quick' } });

console.log('turn after init:', get('turnVal').textContent);
ok(get('turnVal').textContent === 'Black', 'black to move at the start');
ok(appEl.style.width === '1440px', `app stretched to the viewport (got ${appEl.style.width})`);
ok(appEl.style.marginLeft === '-120px', `app pulled to the left edge (got ${appEl.style.marginLeft})`);
ok(canvasEl.style.maxWidth === '700px', `board capped by visible height (got ${canvasEl.style.maxWidth})`);

click(7, 7);
await sleep(2500);
console.log('after human H8 + engine reply:', get('turnVal').textContent,
            '| eval:', get('evalVal').textContent, '| note:', get('evalNote').textContent,
            '| depth:', get('depthVal').textContent,
            '| bar:', get('winRateBlack').style.width);
ok(get('turnVal').textContent === 'Black', 'engine replied, black to move again');
ok(/%$/.test(get('winRateBlack').style.width || ''), 'win rate bar has a width');

for(const [x, y] of [[8,8],[6,6],[9,9]]){ click(x, y); await sleep(2500); }
console.log('later:', '| eval:', get('evalVal').textContent, '| result:', get('resultVal').textContent,
            '| bar:', get('winRateBlack').style.width);

{
  const st = globalThis.__renju();
  const b = new Array(225).fill('.');
  st.moves.forEach((c, i) => { b[c] = i % 2 === 0 ? 'X' : 'O'; });
  console.log('board:');
  for(let y = 0; y < 15; y++) console.log('  ' + b.slice(y*15, y*15+15).join(''));
  console.log('moves:', st.moves.map(c => `${'ABCDEFGHJKLMNOP'[c%15]}${15-((c/15)|0)}`).join(' '));
}
// tapping a forbidden point should explain itself
{
  const core = new Function(await Deno.readTextFile(new URL('./core.js', import.meta.url)) + '\nreturn RenjuCore;')()();
  // Black double three: stones at G8/J8/H9/H7 make H8 a double three.
  const seq = [];
  const bs = [cell(6,7), cell(8,7), cell(7,6), cell(7,8)];
  const ws = [cell(0,0), cell(1,0), cell(2,0), cell(3,0)];
  for(let i = 0; i < 4; i++){ seq.push(bs[i]); seq.push(ws[i]); }
  core.setMoves(seq);
  const why = core.forbiddenReason(cell(7,7));
  console.log('reason for the cross at H8:', why);
  ok(/double three/i.test(why), 'core explains a double three');
  ok(core.forbiddenReason(cell(0,5)) === '', 'a legal point has no reason');
}

get('candidatesToggleBtn').onclick();
await sleep(2000);
console.log('candidates:', get('cand').innerHTML.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').slice(0, 200));
ok(get('cand').innerHTML.length > 10, 'candidate list rendered');

get('undoBtn').onclick(); await sleep(1500);
console.log('after undo, turn =', get('turnVal').textContent);
ok(get('turnVal').textContent === 'Black', 'undo returns the move to the human');

// Regression: "Engine move" plays the human's colour, so it must also let the
// opponent answer - otherwise a single Undo unwinds two moves the human never
// made, and on a nearly empty board that clears the whole thing.
get('newGameBtn').onclick(); await sleep(600);
{
  const before = globalThis.__renju().moves.length;
  get('engineMoveBtn').onclick(); await sleep(3000);
  const after = globalThis.__renju().moves.length;
  ok(after === before + 2, `engine move plays a pair (${before} -> ${after})`);
  get('undoBtn').onclick(); await sleep(1200);
  const undone = globalThis.__renju().moves.length;
  ok(undone === before, `undo returns exactly to where it started (${after} -> ${undone}, wanted ${before})`);
}
for(const [x, y] of [[7,7],[8,8]]){ click(x, y); await sleep(2200); }
{
  const before = globalThis.__renju().moves.length;
  get('engineMoveBtn').onclick(); await sleep(3000);
  get('undoBtn').onclick(); await sleep(1200);
  const now = globalThis.__renju().moves.length;
  ok(now === before, `same mid-game: ${before} -> ${now}`);
}

// switching sides should update the readout that replaced the dropdown
get('switchBtn').onclick(); await sleep(1200);
console.log('side label after switch:', get('sideLabel').textContent);
ok(get('sideLabel').textContent === 'You play White', 'switch sides updates the readout');
get('switchBtn').onclick(); await sleep(1200);
ok(get('sideLabel').textContent === 'You play Black', 'and switches back');

get('newGameBtn').onclick(); await sleep(500);
ok(get('resultVal').textContent === 'Playing', 'new game resets the result');
console.log(fail ? `\n${fail} FAILURES` : '\nui smoke test ok');
if(fail) Deno.exit(1);
