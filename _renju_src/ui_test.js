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
    id, textContent: '', innerHTML: '', value: '', style: {},
    className: '', clientWidth: 640, width: 640, height: 640,
    classList: { _s: new Set(), add(...a){ a.forEach(x=>this._s.add(x)); }, remove(...a){ a.forEach(x=>this._s.delete(x)); },
                 toggle(x, on){ on ? this._s.add(x) : this._s.delete(x); }, contains(x){ return this._s.has(x); } },
    getContext: () => makeCtx(),
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
globalThis.addEventListener = noop;

new Function(src)();

const get = id => nodes.get(id);
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

click(7, 7);
await sleep(2500);
console.log('after human H8 + engine reply:', get('turnVal').textContent,
            '| eval:', get('evalVal').textContent, '| note:', get('evalNote').textContent,
            '| pv:', get('pvLine').textContent, '| depth:', get('depthVal').textContent,
            '| bar:', get('winRateBlack').style.width);
ok(get('turnVal').textContent === 'Black', 'engine replied, black to move again');
ok(/%$/.test(get('winRateBlack').style.width || ''), 'win rate bar has a width');

for(const [x, y] of [[8,8],[6,6],[9,9]]){ click(x, y); await sleep(2500); }
console.log('later:', '| eval:', get('evalVal').textContent, '| result:', get('resultVal').textContent,
            '| bar:', get('winRateBlack').style.width, '| pv:', get('pvLine').textContent);

{
  const st = globalThis.__renju();
  const b = new Array(225).fill('.');
  st.moves.forEach((c, i) => { b[c] = i % 2 === 0 ? 'X' : 'O'; });
  console.log('board:');
  for(let y = 0; y < 15; y++) console.log('  ' + b.slice(y*15, y*15+15).join(''));
  console.log('moves:', st.moves.map(c => `${'ABCDEFGHJKLMNOP'[c%15]}${15-((c/15)|0)}`).join(' '));
}
get('candidatesToggleBtn').onclick();
await sleep(2000);
console.log('candidates:', get('cand').innerHTML.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').slice(0, 200));
ok(get('cand').innerHTML.length > 10, 'candidate list rendered');

get('undoBtn').onclick(); await sleep(1500);
console.log('after undo, turn =', get('turnVal').textContent);
ok(get('turnVal').textContent === 'Black', 'undo returns the move to the human');

get('newGameBtn').onclick(); await sleep(500);
ok(get('resultVal').textContent === 'Playing', 'new game resets the result');
console.log(fail ? `\n${fail} FAILURES` : '\nui smoke test ok');
if(fail) Deno.exit(1);
