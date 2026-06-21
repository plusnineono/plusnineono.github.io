# Renju Software Work Summary

Generated on 2026-05-23 from the local `plusnineono.github.io` repository.

This summary is based on the Renju-related files and git history in the website workspace, especially `renju_hard.qmd`, `renju_engine.html`, `train_renju.py`, `renju_home.qmd`, `renju_articles/ugetsu.qmd`, and the rendered assets under `docs/`.

## Overview

Over the past several months, the Renju work grew from a website page into a small playable software project: a browser-based Renju dojo, an improved standalone engine interface, a Python training/tuning loop for evaluation weights, and an article-style opening analysis with many generated board diagrams and animations.

The project now has two main user-facing modes:

- A Quarto-integrated practice page, `renju_hard.qmd`, published as the Renju Dojo.
- A standalone dark-themed engine page, `renju_engine.html`, with live evaluation, win-rate display, candidate move lists, and a more polished analysis interface.

## Timeline

### June 2025: Article and Diagram Workflow

The earliest Renju-related website artifacts are the `renju_figs/` screenshots and GIFs, many dated around 2025-06-22 to 2025-06-24. These support a written opening study in `renju_articles/ugetsu.qmd`.

That article analyzes the Ugetsu opening, especially white's triangular defense, using many concrete board sequences. It documents forcing lines, four-three threats, VCF-style continuations, and named tactical motifs such as three-move sets.

This phase seems to have established the workflow for:

- Capturing board positions as static screenshots and GIFs.
- Writing Renju analysis in Quarto/Markdown.
- Publishing the analysis through the website's `docs/` output.

### November 2025: Playable Renju Page and Mobile Support

The git history shows a burst of Renju commits around 2025-11-09 and 2025-11-10: adding the Renju page, rendering the site, upgrading the AI, fixing bugs, and supporting smartphone windows.

The older playable pages, `renju.qmd` and `renju_old.qmd`, already include:

- A 15x15 board rendered with HTML canvas.
- Human-vs-computer play.
- Side selection and undo.
- Basic legal move checks.
- Black forbidden-move handling for overlines, double-fours, and double-threes.
- Candidate move generation near existing stones.
- A negamax search with alpha-beta pruning and a transposition table.

This phase turned the Renju work from static article content into an interactive practice tool.

### February 2026: AI Improvements

The February 2026 commits are mostly labeled "Improved AI." The code reflects several stronger search and evaluation ideas:

- Pattern-based threat detection for fours, live threes, and broken threes.
- Candidate move ordering by local tactical value.
- Immediate win and block-win detection.
- Transposition-table reuse.
- Iterative search behavior under time limits.

This phase appears to have focused on making the AI less random and more tactically aware, especially around forcing lines.

### April 2026: Renju Dojo and Stronger Engine

In April 2026, the project gained the current Renju Dojo page and stronger engine work. The git history includes commits such as "Add Renju Dojo page and navbar link," "Strengthen Renju engine," and several rounds of "Improved winning rate calc."

The current `renju_hard.qmd` includes:

- A Quarto page titled `Renju Dojo 連珠道場`.
- A responsive canvas board.
- Human side selection.
- New game and undo controls.
- A visible training timestamp.
- Renju-specific forbidden move checks for Black.
- Tuned evaluation weights embedded in a `WEIGHTS` object.
- Zobrist hashing and a transposition table.
- Killer moves and history heuristic move ordering.
- Quiescence search for tactical positions.
- Null-move pruning and late-move reductions.
- VCF and VCT helper logic for forcing tactical sequences.

The standalone `renju_engine.html` goes further as a richer software interface:

- Dark UI with a main board and side panel.
- New game, undo, side switch, engine-move, and time-control buttons.
- Live evaluation toggle.
- Candidate move list with scores and tags.
- Node-count display.
- Winning-rate bar for Black and White.
- Result banner for wins.
- Coordinate labels on the board.
- Move numbers drawn on stones.
- Red markers for forbidden Black moves.
- Opening-book responses for early positions.
- Time-limited iterative deepening up to a fixed maximum depth.
- Tactical shortcuts for immediate wins and forced defenses.
- Candidate classification tags such as `win`, `block-win`, `double-four`, `four`, `double-three`, `three`, `block-four`, and `block-three`.

The April work also refined the win-rate calculation. The engine converts search scores into a black win-rate estimate with a logistic curve, and it separately checks for short forced wins so clearly winning sequences can snap to near-certain evaluations instead of looking like ordinary small advantages.

## Training and Weight Tuning

The file `train_renju.py` is a Python training script for tuning the evaluation weights used by the Quarto-based AI.

Its important pieces are:

- A Python port of the Renju board and evaluation logic.
- Zobrist hashing and transposition-table support.
- Candidate generation around existing stones.
- Forbidden move checks for Black.
- Threat counting for live fours, sleep fours, live threes, and sleep threes.
- VCF and VCT solvers.
- Alpha-beta/negamax search with quiescence search, null-move pruning, killer moves, history scores, and aspiration windows.
- An opening book containing standard Renju openings such as Horketsu, Kagetsu, Kougetsu, Matsugetsu, Ryuusei, Ungetsu, Hokusei, and Suisei.
- SPSA-style self-play tuning over active evaluation-weight keys.
- Automatic rewriting of the `WEIGHTS` block inside `renju_hard.qmd`.

In short, `train_renju.py` is not just a separate experiment; it is wired back into the website page by editing the embedded JavaScript weights.

## Current Architecture

The Renju software currently lives inside the static website rather than as a separate app package.

Important source files:

- `renju_home.qmd`: simple landing page introducing Renju and linking to the practice room.
- `renju_hard.qmd`: Quarto-integrated playable Renju Dojo.
- `renju_engine.html`: standalone engine/practice UI with richer analysis controls.
- `train_renju.py`: offline training and evaluation-weight tuning script.
- `renju_articles/ugetsu.qmd`: written opening analysis.
- `renju_figs/`: board screenshots and GIFs used in the written article.
- `docs/`: rendered website output.

The main technical stack is deliberately lightweight:

- Quarto for site generation.
- HTML canvas for the board.
- Plain JavaScript for game rules, rendering, search, and UI behavior.
- Python for offline self-play and tuning.

## Main Accomplishments

- Built a browser-playable Renju board with responsive rendering.
- Implemented core Renju legality rules, including Black forbidden moves.
- Added AI play with candidate generation, tactical evaluation, and search.
- Improved the AI through several rounds of stronger tactical detection and search optimizations.
- Added an offline training loop to tune evaluation weights through self-play.
- Created a more polished standalone engine UI with win-rate visualization and candidate-move inspection.
- Wrote a detailed Ugetsu opening analysis with many board diagrams.
- Integrated Renju into the website navigation and published rendered pages under `docs/`.

## Remaining Directions

Natural next steps would be:

- Decide whether `renju_hard.qmd` or `renju_engine.html` should be the canonical practice room.
- Move shared game/search logic into a reusable JavaScript module instead of duplicating it across pages.
- Add regression tests for forbidden-move detection and tactical solvers.
- Store opening-book data in a separate JSON file.
- Add a position editor and import/export notation for sharing board states.
- Turn the training output into a reproducible artifact, for example by saving the tuned weights and training logs.
- Continue the opening-article workflow beyond Ugetsu and connect article diagrams to the playable engine.

