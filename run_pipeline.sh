#!/usr/bin/env bash
#
# Coach-swap pipeline runner.
#
#   ./run_pipeline.sh                 # steps 1-5, then stop at the checkpoint
#   ./run_pipeline.sh audience        # the 9-hour audience run
#   ./run_pipeline.sh analysis        # H2, H3, MDU, baselines
#   ./run_pipeline.sh page            # MP3s + listening page
#   ./run_pipeline.sh all             # every stage, no checkpoint pause
#   ./run_pipeline.sh 3               # one numbered step
#   ./run_pipeline.sh 3 5             # a range of steps
#
#   COACH=estimator_A3 ./run_pipeline.sh 1 3      # different coach name
#   DRY=1 ./run_pipeline.sh all                   # print commands, run nothing
#   KEEP=1 ./run_pipeline.sh all                  # do not clear previous outputs
#   ./run_pipeline.sh clean                       # clear everything, run nothing
#
# Outputs are REPLACED, not added to. Before a step runs, whatever it produced
# last time is deleted, so a rerun cannot leave orphans behind: stimuli from
# briefs no longer in the set, CSVs for judges no longer scored, or LLM logs
# spanning several studies. KEEP=1 disables this.
#
# Every step logs to logs/pipeline/<timestamp>/NN_<step>.log and the script stops
# at the first failure, so a stage can be re-run on its own without repeating the
# ones before it.

set -Eeuo pipefail

COACH="${COACH:-estimator_A2}"
JUDGE="${JUDGE:-estimator_B2}"
DEAM="${DEAM:-datasets/DEAM}"
PMEMO="${PMEMO:-datasets/PMEmo}"
BACKEND="${BACKEND:-ollama}"
DRY="${DRY:-0}"
KEEP="${KEEP:-0}"

cd "$(dirname "$0")"
ROOT="$PWD"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="logs/pipeline/$STAMP"
mkdir -p "$LOGDIR"

# Keep the last few runs' step logs; older ones are of no use once superseded.
KEEP_RUNS="${KEEP_RUNS:-5}"
if [[ -d logs/pipeline ]]; then
  ls -1dt logs/pipeline/*/ 2>/dev/null | tail -n +$((KEEP_RUNS + 1)) | xargs -r rm -rf
fi

bold=$'\e[1m'; dim=$'\e[2m'; red=$'\e[31m'; grn=$'\e[32m'; ylw=$'\e[33m'; off=$'\e[0m'

say()  { printf '\n%s==> %s%s\n' "$bold" "$1" "$off"; }
info() { printf '%s    %s%s\n' "$dim" "$1" "$off"; }
warn() { printf '%s    %s%s\n' "$ylw" "$1" "$off"; }
die()  { printf '\n%s!!  %s%s\n\n' "$red" "$1" "$off" >&2; exit 1; }

trap 'die "step failed (line $LINENO). Nothing after this point ran. Fix, then re-run just this step."' ERR

# Run a command, tee to its own log, and time it.
step() {
  local n="$1" name="$2"; shift 2
  local log="$LOGDIR/$(printf '%02d' "$n")_${name}.log"
  say "[$n] $name"
  info "$*"
  info "log: $log"
  if [[ "$DRY" == "1" ]]; then info "(dry run, not executed)"; return 0; fi
  local t0=$SECONDS
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  [[ $rc -eq 0 ]] || return $rc
  printf '%s    done in %dm%02ds%s\n' "$grn" $(( (SECONDS-t0)/60 )) $(( (SECONDS-t0)%60 )) "$off"
}

# ------------------------------------------------------------------- clean ---
# Remove what a step regenerates, so each run starts from a known state.
# Scoped to the requested range: running only `analysis` must not delete stimuli.
rm_glob() {
  local label="$1"; shift
  local found=0 f
  for f in "$@"; do
    [[ -e "$f" ]] || continue
    found=1
    if [[ "$DRY" == "1" ]]; then info "would remove: $f"; else rm -rf "$f"; fi
  done
  [[ $found -eq 1 ]] && info "cleared: $label"
  return 0
}

clean_range() {
  local from="$1" to="$2"
  [[ "$KEEP" == "1" ]] && { info "KEEP=1: previous outputs left in place"; return 0; }
  say "[0] clearing previous outputs for steps $from-$to"

  (( from<=1 && to>=1 )) && rm_glob "selection report" models/selection
  (( from<=2 && to>=2 )) && rm_glob "reachable region" models/reachable_va.json
  if (( from<=3 && to>=3 )); then
    # Every rendered file, not just *.wav: a stale .mid or .tmp left by a failed
    # render is still stale. manifest.json is rewritten with only the briefs in
    # the run, so orphan audio from a wider run must go with it.
    rm_glob "stimuli"         data/stimuli
    rm_glob "generation log"  logs/generation.jsonl logs/pilot.jsonl
  fi
  if (( from<=4 && to>=4 )); then
    rm_glob "scored distances" data/analysis
  fi
  (( from<=5 && to>=5 )) && rm_glob "H1 outputs" analysis/h1
  if (( from<=6 && to>=6 )); then
    # responses_*.csv are the timestamped copies run_audience.py sets aside when
    # it starts a new file, plus responses_current/_all from clean_audience.py.
    # Left in place they accumulate across studies and invite loading the wrong one.
    if [[ -f data/audience/responses.csv ]]; then
      local rows; rows=$(( $(wc -l < data/audience/responses.csv) - 1 ))
      warn "removing $rows audience responses and every responses_*.csv backup"
      warn "that is the ~9 hour run; copy them out first if you still need them"
    fi
    rm_glob "audience responses" data/audience
    rm_glob "audience log"       logs/audience.jsonl
  fi
  if (( from<=7 && to>=7 )); then
    rm_glob "analysis outputs" analysis/h1 analysis/h2 analysis/h3 \
                               analysis/mdu analysis/baselines analysis/tables
  fi
  if (( from<=8 && to>=8 )); then
    # create_index.py lives in this directory and is source, not output, so the
    # directory is emptied by pattern rather than removed.
    rm_glob "mp3s and page" data/stimuli_mp3/*.mp3 data/stimuli_mp3/*.html
  fi

  if [[ "$DRY" != "1" ]]; then
    mkdir -p data/stimuli data/analysis data/audience data/stimuli_mp3 logs
  fi
  printf '%s    cleared%s\n' "$grn" "$off"
}

# ---------------------------------------------------------------- preflight ---
preflight() {
  say "[0] preflight"
  [[ -n "${VIRTUAL_ENV:-}" ]] || warn "no virtualenv active - run: source .venv/bin/activate"

  local missing=0
  for f in src/estimators/select_estimator.py src/generator/probe_reachable.py \
           src/generator/generate_briefs.py src/generator/run_generation.py \
           src/analysis/score_estimator_b.py src/audience/run_audience.py \
           data/stimuli_mp3/create_index.py analysis/Stage6_h1.R \
           analysis/Check_quadrant_confidence.R; do
    [[ -f "$f" ]] || { warn "missing: $f"; missing=1; }
  done
  [[ -f src/estimators/select.py ]] && die "src/estimators/select.py still exists - it shadows Python's stdlib 'select' module. Delete it."
  [[ $missing -eq 0 ]] || die "files above are missing."

  # LFS pointer stubs are ~130 bytes and fail with KeyError: 118 on load
  for f in models/estimator_A.joblib models/estimator_B.joblib; do
    if [[ -f "$f" ]]; then
      local sz; sz=$(stat -c%s "$f")
      [[ $sz -gt 10000 ]] || die "$f is $sz bytes - a Git-LFS pointer. Run: git lfs pull"
    else
      warn "missing: $f"
    fi
  done

  [[ -d "$DEAM"  ]] || warn "DEAM not found at $DEAM (set DEAM=/path)"
  [[ -d "$PMEMO" ]] || warn "PMEmo not found at $PMEMO (set PMEMO=/path)"

  command -v Rscript  >/dev/null || warn "Rscript not on PATH"
  command -v ffmpeg   >/dev/null || warn "ffmpeg not on PATH (step 9 needs it)"
  curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 \
    || warn "Ollama not responding on :11434 - steps 4 and 7 need it (ollama serve)"

  info "coach=$COACH  judge=$JUDGE  backend=$BACKEND"
  printf '%s    preflight ok%s\n' "$grn" "$off"
}

# -------------------------------------------------------------------- steps ---
s1_select() {
  step 1 select_coach python src/estimators/select_estimator.py \
    --deam "$DEAM" --corpus DEAM \
    --name "$COACH" --role optimisation_coach --discriminate-on probe
}

s2_region() {
  step 2 probe_reachable python src/generator/probe_reachable.py --coach "$COACH"
  step 2 generate_briefs python src/generator/generate_briefs.py
  # generate_briefs.py already prints the region it used and warns when a
  # quadrant is unreachable. A hardcoded "compare against" line here goes stale
  # the moment the coach or the generator changes, and quoting a region from two
  # coaches ago is worse than saying nothing.
  warn "check the region printed above spans all four quadrants before continuing"
}

s3_generate() {
  step 3 run_generation python -W ignore src/generator/run_generation.py \
    --backend "$BACKEND" --coach "$COACH"
  if [[ "$DRY" != "1" ]]; then
    grep -q "coach: $COACH" "$LOGDIR/03_run_generation.log" 2>/dev/null \
      || warn "could not confirm '$COACH' in the generation log - check it used the right coach"
  fi
}

s4_score() {
  step 4 score_incumbent python src/analysis/score_estimator_b.py
  step 4 score_selected  python src/analysis/score_estimator_b.py --estimator "$JUDGE"
  # The coach is scored too, but only as a diagnostic column for the quadrant
  # check and the listening page. It is never passed to Stage6_h1.R: the loop
  # selected `best` by minimising the coach's own distance, so an H1 test against
  # the coach is true by construction.
  step 4 score_coach_diagnostic python src/analysis/score_estimator_b.py --estimator "$COACH"
}

s5_h1() {
  step 5 h1_incumbent Rscript analysis/Stage6_h1.R
  step 5 h1_selected  Rscript analysis/Stage6_h1.R "$JUDGE"
  step 5 quadrant_confidence Rscript analysis/Check_quadrant_confidence.R
}

s6_audience() {
  say "[6] synthetic audience - about 9 hours"
  info "resume after an interruption: python src/audience/run_audience.py --backend $BACKEND --resume"
  step 6 run_audience python -W ignore src/audience/run_audience.py --backend "$BACKEND"
}

# Every statistical test, H1 through baselines. H1 and the quadrant check are
# repeated here so this one target reproduces the full result set from whatever
# is currently in data/; they take seconds and are idempotent.
s7_analysis() {
  step 7 h1_incumbent        Rscript analysis/Stage6_h1.R
  step 7 h1_selected         Rscript analysis/Stage6_h1.R "$JUDGE"
  step 7 quadrant_confidence Rscript analysis/Check_quadrant_confidence.R
  step 7 h2                  Rscript analysis/Stage6_h2.R
  step 7 h3                  Rscript analysis/Stage6_h3_alignment.R
  step 7 mdu                 Rscript analysis/Stage6_mdu.R
  step 7 baselines           Rscript analysis/Stage6_baselines.R
}

s8_page() {
  step 8 convert_mp3 sh ./convert_2_mp3.sh
  step 8 build_page  python data/stimuli_mp3/create_index.py
  info "open: data/stimuli_mp3/index.html"
}

# Plain printf rather than a heredoc: an unquoted heredoc mixing expanded colour
# codes with multi-byte box characters is easy to corrupt in transit, and a broken
# heredoc makes bash mis-parse the rest of the file.
checkpoint() {
  printf '\n%s%s%s\n' "$bold" "== CHECKPOINT ===========================================================" "$off"
  printf '\n  Read before starting the audience run:\n\n'
  printf '      cat analysis/h1/%s/h1_results.txt\n' "$JUDGE"
  printf '      cat analysis/h1/quadrant_confidence.txt\n\n'
  printf '  Continue with:   ./run_pipeline.sh audience\n'
  printf '  Then:            ./run_pipeline.sh analysis\n'
  printf '                   ./run_pipeline.sh page\n'
  printf '\n%s%s%s\n' "$bold" "=========================================================================" "$off"
}

run_range() {
  local from="$1" to="$2"
  (( from<=1 && to>=1 )) && s1_select
  (( from<=2 && to>=2 )) && s2_region
  (( from<=3 && to>=3 )) && s3_generate
  (( from<=4 && to>=4 )) && s4_score
  (( from<=5 && to>=5 )) && s5_h1
  (( from<=6 && to>=6 )) && s6_audience
  (( from<=7 && to>=7 )) && s7_analysis
  (( from<=8 && to>=8 )) && s8_page
  return 0
}

# --------------------------------------------------------------------- main ---
target="${1:-checkpoint}"

case "$target" in
  checkpoint) preflight; clean_range 1 5; run_range 1 5;  checkpoint ;;
  audience)   preflight; clean_range 6 6; run_range 6 6 ;;
  analysis)   preflight; clean_range 7 7; run_range 7 7 ;;
  page)       preflight; clean_range 8 8; run_range 8 8 ;;
  all)        preflight; clean_range 1 8; run_range 1 8 ;;
  preflight)  preflight ;;
  clean)      preflight; clean_range 1 8 ;;
  [1-8])      preflight; clean_range "$target" "${2:-$target}"
              run_range "$target" "${2:-$target}" ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac

say "finished. logs in $LOGDIR"