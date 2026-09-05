#!/bin/sh
# Wrapper for the renju engine tools, so you do not have to remember where the
# JavaScript runtime lives. There is no `deno` or `node` on this machine's PATH;
# Quarto ships a Deno binary and that is what everything here uses.
#
#   ./_renju_src/run.sh help
#
# Works from anywhere - it locates the repository from its own path.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")

# Prefer a real deno if one is ever installed, else Quarto's, matching the CPU.
if command -v deno >/dev/null 2>&1; then
  DENO=deno
elif [ -x "/Applications/quarto/bin/tools/$(uname -m)/deno" ]; then
  DENO="/Applications/quarto/bin/tools/$(uname -m)/deno"
elif [ -x /Applications/quarto/bin/tools/aarch64/deno ]; then
  DENO=/Applications/quarto/bin/tools/aarch64/deno
elif [ -x /Applications/quarto/bin/tools/x86_64/deno ]; then
  DENO=/Applications/quarto/bin/tools/x86_64/deno
else
  echo "No JavaScript runtime found." >&2
  echo "Install Quarto (which bundles one), or 'brew install deno'." >&2
  exit 1
fi

R="$DENO run --allow-read"
RW="$DENO run --allow-read --allow-write"
cmd=${1:-help}
shift 2>/dev/null || true

case "$cmd" in
  test)
    $R "$HERE/test.js"
    $R "$HERE/ui_test.js"
    ;;

  selfplay)   # [games] [ms-per-move] [outfile] [seed]
    $RW "$HERE/selfplay.js" "${1:-1000}" "${2:-150}" "${3:-$ROOT/renju_data.txt}" "${4:-1}"
    ;;

  tune)       # <datafile> [--epochs=N] - dry run, prints the fit, changes nothing
    data=${1:-$ROOT/renju_data.txt}
    [ $# -gt 0 ] && shift
    $R "$HERE/tune_eval.js" "$data" "$@"
    ;;

  apply)      # <datafile> [--epochs=N] - same, but writes the weights into core.js
    data=${1:-$ROOT/renju_data.txt}
    [ $# -gt 0 ] && shift
    $RW "$HERE/tune_eval.js" "$data" --apply "$@"
    ;;

  ab)         # <games> <ms> <coreA.js> <coreB.js>
    $R "$HERE/ab.js" "$@"
    ;;

  match)      # [games] [ms] [ms-for-the-old-engine] - against the v5.4 baseline
    $R "$HERE/match.js" "$@"
    ;;

  build)      # regenerate renju_engine.qmd and the rendered page
    python3 "$HERE/build.py"
    (cd "$ROOT" && quarto render renju_engine.qmd)
    ;;

  help|--help|-h|*)
    cat <<EOF
Renju engine tools.  Runtime: $DENO

  ./_renju_src/run.sh test
      Engine tests plus the page smoke test. Takes about 15 seconds.

  ./_renju_src/run.sh selfplay [games] [ms] [outfile] [seed]
      Play games and record labelled positions for tuning. Appends to the file,
      so you can stop and resume, and prints an ETA as it goes.
      Roughly: 1000 games at 150 ms is about an hour and gives ~20000 positions.

  ./_renju_src/run.sh tune [datafile]
      Fit the evaluation weights and print them. Changes nothing.

  ./_renju_src/run.sh apply [datafile]
      Fit and write the weights into core.js. Verify before keeping them:
        ./_renju_src/run.sh test
        cp _renju_src/core.js /tmp/core_tuned.js && git stash
        ./_renju_src/run.sh ab 40 600 /tmp/core_tuned.js _renju_src/core.js
      Keep the fit only if the tuned build actually wins, then: git stash pop

  ./_renju_src/run.sh ab <games> <ms> <coreA.js> <coreB.js>
      Play two builds against each other. Calibrate first: the same file against
      itself scores 62% over 8 games often enough. Use 40 games or more.

  ./_renju_src/run.sh match [games] [ms] [old-ms]
      Play the current engine against the v5.4 baseline.

  ./_renju_src/run.sh build
      Rebuild renju_engine.qmd from these sources and render the page.

For a long self-play run, keep the Mac awake and let it survive the terminal
closing:

  caffeinate -is nohup ./_renju_src/run.sh selfplay 2000 150 data.txt > log.txt 2>&1 &
  tail -f log.txt          # watch it;  Ctrl-C just stops watching
EOF
    ;;
esac
