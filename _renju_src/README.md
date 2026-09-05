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
are handled as forced nodes rather than by search. Quiescence is an exact VCF
search — only four-creating moves, with the defender forced to block — so every
leaf is tactically settled. Before the main search the root runs a VCT threat
search (attacker plays only fours and open threes) which is where most renju
wins actually come from, plus the same search from the opponent's side to spot
threats that have to be broken. The VCT search prunes the defender's replies to
the best dozen, so a "forced win" is shown as 99%, not 100%.

Against the previous engine (`baseline_v5.js`), with varied openings and each
side played once per opening, the new engine wins every game while getting a
tenth of the thinking time.

**Score.** One number, in "threat points", from the side to move's point of
view, with mate scores of the form `MATE - ply`. The page converts it to Black's
point of view and maps it through a logistic curve for the win-rate bar, so the
bar, the label and the principal variation can never disagree.

Two details keep that number readable. The tempo bonus is proportional to the
threats a side can actually cash in rather than to its whole position, and the
number reported is the mean of the last two search iterations: odd and even
depths straddle the true value ("odd-even effect"), and without the mean the
displayed score swung by an open three every single ply. The move played always
comes from the deepest iteration regardless.
