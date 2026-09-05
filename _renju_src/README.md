# Renju engine sources

`renju_engine.qmd` at the repo root is **generated**. Edit the files here and
rebuild:

```sh
python3 _renju_src/build.py     # writes ../renju_engine.qmd
quarto render renju_engine.qmd  # writes docs/renju_engine.html
```

Directories starting with `_` are ignored by Quarto, so nothing here is
published.

| file | what it is |
| --- | --- |
| `core.js` | the engine: rules, renju forbidden moves, evaluation, search. Pure JS, no DOM. |
| `ui.js` | the page: canvas board, controls, evaluation display, Web Worker plumbing. |
| `page.html` | CSS and markup, plus the opening `<script>` tag. |
| `build.py` | concatenates the three into `renju_engine.qmd`. |
| `test.js` | engine regression tests (shapes, forbidden moves, tactics, speed). |
| `ui_test.js` | runs the generated page against a fake DOM: catches wiring mistakes. |
| `match.js` | plays the engine against the previous version. |
| `ab.js` | plays two builds against each other, both taken from files. |
| `selfplay.js` | records labelled positions for tuning. |
| `tune_eval.js` | fits the evaluation weights to them. |
| `baseline_v5.js` | the previous engine, kept only so `match.js` has something to measure against. |

Tests need a JavaScript runtime. Quarto ships one:

```sh
DENO=/Applications/quarto/bin/tools/aarch64/deno   # x86_64 on Intel Macs
$DENO run --allow-read _renju_src/test.js
$DENO run --allow-read _renju_src/ui_test.js
$DENO run --allow-read _renju_src/match.js 6 800        # 6 games, 800 ms/move
$DENO run --allow-read _renju_src/match.js 6 400 4000   # new gets 400 ms, old gets 4000 ms
```

## How the engine works

**Shapes.** Every point has, for each of the 4 directions, an 11-cell window
(centre ± 5) encoded in base 3 from one colour's point of view: 0 empty, 1 own
stone, 2 blocked (opponent stone or board edge). A memoised table maps that key
to what playing the centre point would achieve — five, overline, open four,
four, open three, broken three, open two, two. Eleven cells is exactly the width
needed to classify all of those without edge artefacts, and the table is derived
recursively ("a three is a shape that one more stone turns into a four"), so
broken shapes like `X.XX` are handled without any hand-written pattern list.

**Incremental state.** The four keys of every point are updated in place when a
stone is played or taken back: one stone touches 4 directions × 10 neighbours,
so make/unmake is O(1) and so is the evaluation, which is kept as a running sum
over the points each side could play. That is what lets the search run at
~150k nodes/s in a browser.

**Renju rules.** Overline, double four and double three all fall out of the same
shape table. Two of the fiddly cases are handled explicitly:

* *Two fours in one line.* A point with two five-points is a straight four only
  if it is four in a row with both ends empty; anything else (`.X.XXX.X`) is two
  fours sharing a point, which renju counts as a double four. The shape table
  gives those the separate code `P_D4`.
* *Double three.* Applied recursively: a three only counts if it can actually be
  pushed to a straight four by a move that is itself legal.

Forbidden points are excluded from Black's move generation, counted as a
positional asset for White, and drawn on the board for a human playing Black.

**Search.** Iterative-deepening alpha-beta (PVS) with a transposition table,
killer moves, history ordering and late-move reductions. Five-in-a-row threats
are handled as forced nodes rather than by search. Four-creating moves are
extended by a ply. Quiescence is an exact VCF search — only four-creating moves,
with the defender forced to block — so every leaf is tactically settled. Before the main search the root runs a VCT threat
search (attacker plays only fours and open threes) which is where most renju
wins actually come from, plus the same search from the opponent's side to spot
threats that have to be broken. The VCT search prunes the defender's replies to
the best dozen, so a "forced win" is shown as 99%, not 100%.

Against the previous engine (`baseline_v5.js`), with varied openings and each
side played once per opening, the new engine wins every game while getting a
tenth of the thinking time.

## Measuring a change

`ab.js` plays two builds against each other, `match.js` plays the current one
against `baseline_v5.js`. Two rules, both learned the hard way:

**Take the two builds from files, never from flags on a shared global.** An
earlier harness set tuning flags on `globalThis` before constructing each
engine, so a config of `{}` inherited whatever the previous build had set: the
two sides were the same engine and every "result" was noise. It reported a 60%
score for a change that, measured properly, scores 25%.

**Calibrate the noise first.** The same build against itself scores 4-2 over 6
games often enough. Forty games is about the minimum for a real signal, and a
change that helps at 300 ms can hurt at 2 s — measure at the time control
people actually play at.

### What has been tried

| change | result |
| --- | --- |
| extend a ply on open threes, not just fours | **9-31** (40 games, 600 ms) |
| the same plus defensive quiescence | **10-30** (40 games, 600 ms) |
| "tension" term so White simplifies and Black complicates | 4-6 overall, 0-5 as White |
| refusing to commit a part-finished search iteration | 5-7 |
| deeper VCF at the leaves (quiescence budget 16 vs 10) | 8-8 |
| walking a bitmask of live points instead of all 225 | search tree identical, ~3% faster |
| merging the per-point bookkeeping into one delta pass | identical, make/unmake 8% faster |

The engine sits at a local optimum for its *search* parameters: they have been
pushed in both directions and nothing moves. What has never been fitted is the
evaluation — see below.

## Tuning the evaluation from self-play

The evaluation is a weighted sum of counts, so it is **linear in its weights**:

```
evaluate(side) === dot(features(side), weights())
```

`test.js` asserts that identity, which is what makes fitting cheap. Because the
model is linear, tuning it against game outcomes is logistic regression, not
reinforcement learning in the usual sense: you need thousands of *positions*,
not thousands of *games*, and a game yields about twenty of them. This is the
method chess engines call Texel tuning.

```sh
DENO=/Applications/quarto/bin/tools/aarch64/deno

# 1. play games and record positions labelled with who eventually won.
#    Appends, so you can stop it, run it again, or run several seeds in
#    parallel into different files and concatenate them.
$DENO run --allow-read --allow-write _renju_src/selfplay.js 1000 150 data.txt

# 2. fit. Seconds, once the data exists. Dry run first.
$DENO run --allow-read _renju_src/tune_eval.js data.txt
$DENO run --allow-read --allow-write _renju_src/tune_eval.js data.txt --apply

# 3. VERIFY. Fitting outcomes is not the same as playing better.
$DENO run --allow-read _renju_src/test.js
cp _renju_src/core.js /tmp/core_tuned.js && git stash
$DENO run --allow-read _renju_src/ab.js 40 600 /tmp/core_tuned.js _renju_src/core.js
# keep it only if the tuned build actually wins; then git stash pop, build, render
```

Scale is worth knowing about. The fit holds the logistic constant K at the
value the page uses for the win-rate bar, so tuned weights come out already
calibrated to it — "+2500" really does mean about 73%. It also means the fit
picks the overall scale for you, and the tuner warns if that scale moves far
enough to matter, because the tempo cap (700) and the mate band assume roughly
the current one.

What the tuner cannot do: features that rarely fire cannot be fitted (with too
little data the "double four" combination weight will come back untouched,
because it almost never occurs), and it only tunes the 16 numbers listed in the
weight block — the shape classification, the search, and the inner constants of
the tempo term are all outside the model. Bigger ideas (a policy network, MCTS)
would have to run in the browser on GitHub Pages, which rules out anything
needing shared memory or a GPU.

**Opening.** White's reply to a lone stone is a book move, not a search. With
one stone of each colour the evaluation is exactly symmetric - `sumB === sumW`
whatever the distance between the stones - so every candidate scores the same
and the search picks arbitrarily, often two or three lines away. Every named
renju opening has White adjacent to Black's first stone, so the engine plays the
ring (varied between games when the page asks for it). From move 4 on there are
real shapes to evaluate and the search takes over.

The nominal depth understates the reach: forcing moves are extended and
quiescence chases fours, so depth 8 routinely follows lines past ply 25. The
page shows both numbers ("8/28") for that reason.

**Score.** One number, in "threat points", from the side to move's point of
view, with mate scores of the form `MATE - ply`. The page converts it to Black's
point of view and maps it through a logistic curve for the win-rate bar, so the
bar and the label can never disagree.

Two details keep that number readable. The tempo bonus is proportional to the
threats a side can actually cash in rather than to its whole position, and the
number reported is the mean of the last two search iterations: odd and even
depths straddle the true value ("odd-even effect"), and without the mean the
displayed score swung by an open three every single ply. The move played always
comes from the deepest iteration regardless.
