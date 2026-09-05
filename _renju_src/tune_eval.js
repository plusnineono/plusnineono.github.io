// Fit the evaluation weights to self-play results (Texel-style tuning).
//
//   deno run --allow-read              _renju_src/tune_eval.js data.txt
//   deno run --allow-read --allow-write _renju_src/tune_eval.js data.txt --apply
//
// The evaluation is a weighted sum of counts, so evaluate(side) is exactly
// dot(features(side), weights) - the test suite asserts it. Fitting it to game
// outcomes is therefore plain logistic regression rather than hill-climbing
// with thousands of games, which is why this takes seconds once the data
// exists.
//
// The logistic scale K is held at the value the page uses to draw the win-rate
// bar, so the fitted weights come out already calibrated to that curve: a
// tuned "+2500" really does mean about 73%.
//
// --apply rewrites the TUNED WEIGHTS block in core.js. Always confirm with a
// match before keeping it:
//
//   cp _renju_src/core.js /tmp/core_tuned.js     # after --apply
//   git stash                                    # back to the old weights
//   deno run --allow-read _renju_src/ab.js 40 600 /tmp/core_tuned.js _renju_src/core.js
//
// Fitting the outcome is not the same thing as playing better. Forty games is
// about the minimum for a signal; see the README.

const CORE_PATH = new URL('./core.js', import.meta.url);
const CORE = await Deno.readTextFile(CORE_PATH);
const RenjuCore = new Function(CORE + '\nreturn RenjuCore;')();
const core = RenjuCore();

const args = Deno.args.filter(a => !a.startsWith('--'));
const flags = new Set(Deno.args.filter(a => a.startsWith('--')));
const DATA = args[0] ?? 'renju_data.txt';
const K = Number((Deno.args.find(a => a.startsWith('--k=')) ?? '--k=2500').slice(4));
const EPOCHS = Number((Deno.args.find(a => a.startsWith('--epochs=')) ?? '--epochs=600').slice(9));

const NF = core.N_FEATURES;
const names = core.FEATURE_NAMES;

// ------------------------------------------------------------------ data ---
const text = await Deno.readTextFile(DATA);
const rows = [];
for(const line of text.split('\n')){
  if(!line) continue;
  const parts = line.split(' ');
  if(parts.length !== NF + 1) continue;
  const y = Number(parts[0]);
  const f = new Float64Array(NF);
  for(let i = 0; i < NF; i++) f[i] = Number(parts[i + 1]);
  rows.push({ y, f });
}
if(rows.length < 500){
  console.error(`only ${rows.length} positions in ${DATA} - generate more with selfplay.js first`);
  Deno.exit(1);
}
// Deterministic shuffle, then hold out a tenth to check we are not overfitting.
let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
for(let i = rows.length - 1; i > 0; i--){
  const j = Math.floor(rnd() * (i + 1));
  [rows[i], rows[j]] = [rows[j], rows[i]];
}
const cut = Math.floor(rows.length * 0.9);
const train = rows.slice(0, cut), val = rows.slice(cut);
console.log(`${rows.length} positions (${train.length} train, ${val.length} validation), K = ${K}`);

// ------------------------------------------------------------------ fit ----
const w = Float64Array.from(core.weights());
const w0 = Array.from(w);

function loss(set, wv){
  let sum = 0;
  for(const r of set){
    let d = 0;
    for(let i = 0; i < NF; i++) d += r.f[i] * wv[i];
    const p = 1 / (1 + Math.exp(-d / K));
    const q = Math.min(1 - 1e-9, Math.max(1e-9, p));
    sum -= r.y * Math.log(q) + (1 - r.y) * Math.log(1 - q);
  }
  return sum / set.length;
}

console.log(`start:  train ${loss(train, w).toFixed(5)}   validation ${loss(val, w).toFixed(5)}`);

// Adam. The gradient of the logistic loss w.r.t. w is (p - y) * f / K.
const m = new Float64Array(NF), v = new Float64Array(NF);
const lr = 2.0, b1 = 0.9, b2 = 0.999, eps = 1e-8;
const BATCH = 4096;
let step = 0;
for(let epoch = 0; epoch < EPOCHS; epoch++){
  for(let start = 0; start < train.length; start += BATCH){
    const end = Math.min(train.length, start + BATCH);
    const g = new Float64Array(NF);
    for(let r = start; r < end; r++){
      const row = train[r];
      let d = 0;
      for(let i = 0; i < NF; i++) d += row.f[i] * w[i];
      const p = 1 / (1 + Math.exp(-d / K));
      const e = (p - row.y) / K;
      for(let i = 0; i < NF; i++) g[i] += e * row.f[i];
    }
    const n = end - start;
    step++;
    const c1 = 1 - Math.pow(b1, step), c2 = 1 - Math.pow(b2, step);
    for(let i = 0; i < NF; i++){
      const gi = g[i] / n;
      m[i] = b1 * m[i] + (1 - b1) * gi;
      v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      w[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
    }
  }
  if((epoch + 1) % 100 === 0)
    console.log(`epoch ${epoch + 1}: train ${loss(train, w).toFixed(5)}   validation ${loss(val, w).toFixed(5)}`);
}

const trainAfter = loss(train, w), valAfter = loss(val, w);
console.log(`\nfinal:  train ${trainAfter.toFixed(5)}   validation ${valAfter.toFixed(5)}`);
if(valAfter >= loss(val, w0))
  console.log('validation loss did not improve - more data, or the weights were already near a fit');

// --------------------------------------------------------------- report ----
console.log('\n' + 'feature'.padEnd(24) + 'before'.padStart(10) + 'after'.padStart(10));
for(let i = 0; i < NF; i++){
  const before = i === 13 ? w0[i] * 100 : w0[i];
  const after = i === 13 ? w[i] * 100 : w[i];
  console.log(names[i].padEnd(24) + before.toFixed(0).padStart(10) + after.toFixed(0).padStart(10));
}

// Sanity checks a fit can fail without anything crashing.
const shape = w.slice(0, 8);
if(shape.some(x => x < 0)) console.log('\nwarning: a shape weight came out negative');
for(let i = 1; i < 8; i++){
  if(shape[i] < shape[i - 1] && !(i === 4))     // B4 below F3 is normal: a four in one line
    console.log(`warning: "${names[i]}" is worth less than "${names[i - 1]}"`);
}
const scale = w[7] / w0[7];
if(scale < 0.5 || scale > 2)
  console.log(`warning: overall scale moved by ${scale.toFixed(2)}x - the tempo cap (700) and ` +
              'the win-rate curve assume roughly the current scale');

// ---------------------------------------------------------------- apply ----
if(flags.has('--apply')){
  const rounded = Array.from(w).map((x, i) => Math.round(i === 13 ? x * 100 : x));
  const keys = core.WEIGHT_KEYS;
  const block =
`// ==== TUNED WEIGHTS BEGIN ====
// Rewritten by tune_eval.js. Everything the evaluation is linear in lives here,
// in the order features() reports, so a tuned weight vector drops straight in.
// Fitted on ${rows.length} self-play positions, validation loss ${valAfter.toFixed(5)}.
const ${keys[0]} = ${rounded[0]}, ${keys[1]} = ${rounded[1]}, ${keys[2]} = ${rounded[2]}, ${keys[3]} = ${rounded[3]}, ${keys[4]} = ${rounded[4]}, ${keys[5]} = ${rounded[5]}, ${keys[6]} = ${rounded[6]}, ${keys[7]} = ${rounded[7]};
const ${keys[8]} = ${rounded[8]}, ${keys[9]} = ${rounded[9]}, ${keys[10]} = ${rounded[10]}, ${keys[11]} = ${rounded[11]}, ${keys[12]} = ${rounded[12]};
const ${keys[13]} = ${rounded[13]};                  // percent of the tempo term
const ${keys[14]} = ${rounded[14]}, ${keys[15]} = ${rounded[15]};
// ==== TUNED WEIGHTS END ====`;
  const start = CORE.indexOf('// ==== TUNED WEIGHTS BEGIN ====');
  const end = CORE.indexOf('// ==== TUNED WEIGHTS END ====') + '// ==== TUNED WEIGHTS END ===='.length;
  if(start < 0 || end < start){ console.error('could not find the weight block in core.js'); Deno.exit(1); }
  await Deno.writeTextFile(CORE_PATH, CORE.slice(0, start) + block + CORE.slice(end));
  console.log('\ncore.js updated. Now:');
  console.log('  deno run --allow-read _renju_src/test.js              # the linearity test must still pass');
  console.log('  cp _renju_src/core.js /tmp/core_tuned.js && git stash');
  console.log('  deno run --allow-read _renju_src/ab.js 40 600 /tmp/core_tuned.js _renju_src/core.js');
  console.log('  # keep it only if the tuned build actually wins; then git stash pop and rebuild');
} else {
  console.log('\n(dry run - pass --apply to write these into core.js)');
}
