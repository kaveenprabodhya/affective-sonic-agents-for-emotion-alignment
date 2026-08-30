#!/usr/bin/env bash
#
# Coach-swap pipeline runner.
#
#   ./run_pipeline.sh                 # steps 1-5, then stop at checkpoint
#   ./run_pipeline.sh audience        # fresh ~9-hour audience run
#   ./run_pipeline.sh audience-resume # resume interrupted audience; no cleanup
#   ./run_pipeline.sh h1-checks       # H1 + robustness checks only
#   ./run_pipeline.sh h2-checks       # H2 + baselines only
#   ./run_pipeline.sh h3-checks       # H3 + supplementary only
#   ./run_pipeline.sh analysis        # complete H1/H2/H3 + MDU + scale usage
#   ./run_pipeline.sh page            # MP3s + listening page
#   ./run_pipeline.sh all             # every stage, no checkpoint pause
#   ./run_pipeline.sh 3               # one numbered step
#   ./run_pipeline.sh 3 5             # inclusive numbered range
#
#   COACH=estimator_A3 ./run_pipeline.sh 1 3
#   DRY=1 ./run_pipeline.sh all
#   KEEP=1 ./run_pipeline.sh all
#   ./run_pipeline.sh clean

set -Eeuo pipefail

COACH="${COACH:-estimator_A2}"
JUDGE="${JUDGE:-estimator_B2}"
DEAM="${DEAM:-datasets/DEAM}"
PMEMO="${PMEMO:-datasets/PMEmo}"
BACKEND="${BACKEND:-ollama}"
DRY="${DRY:-0}"
KEEP="${KEEP:-0}"
KEEP_RUNS="${KEEP_RUNS:-5}"

cd "$(dirname "$0")"
ROOT="$PWD"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="logs/pipeline/$STAMP"

bold=$'\e[1m'
dim=$'\e[2m'
red=$'\e[31m'
grn=$'\e[32m'
ylw=$'\e[33m'
off=$'\e[0m'

say() {
  printf '\n%s==> %s%s\n' "$bold" "$1" "$off"
}

info() {
  printf '%s    %s%s\n' "$dim" "$1" "$off"
}

warn() {
  printf '%s    %s%s\n' "$ylw" "$1" "$off"
}

die() {
  printf '\n%s!!  %s%s\n\n' "$red" "$1" "$off" >&2
  exit 1
}

trap 'die "step failed (line $LINENO). Nothing after this point ran. Fix, then re-run just this step."' ERR

prune_pipeline_logs() {
  [[ -d logs/pipeline ]] || return 0
  ls -1dt logs/pipeline/*/ 2>/dev/null | tail -n +$((KEEP_RUNS + 1)) | xargs -r rm -rf || true
}

step() {
  local n="$1"
  local name="$2"
  shift 2

  local log="$LOGDIR/$(printf '%02d' "$n")_${name}.log"

  say "[$n] $name"
  info "$*"
  info "log: $log"

  if [[ "$DRY" == "1" ]]; then
    info "(dry run, not executed)"
    return 0
  fi

  mkdir -p "$LOGDIR"

  local t0=$SECONDS

  "$@" 2>&1 | tee "$log"

  local rc=${PIPESTATUS[0]}

  [[ $rc -eq 0 ]] || return "$rc"

  printf '%s    done in %dm%02ds%s\n' \
    "$grn" \
    $(((SECONDS - t0) / 60)) \
    $(((SECONDS - t0) % 60)) \
    "$off"
}

rm_glob() {
  local label="$1"
  shift

  local found=0
  local f

  for f in "$@"; do
    [[ -e "$f" ]] || continue

    found=1

    if [[ "$DRY" == "1" ]]; then
      info "would remove: $f"
    else
      rm -rf -- "$f"
    fi
  done

  [[ $found -eq 1 ]] && info "cleared: $label"

  return 0
}

clean_range() {
  local from="$1"
  local to="$2"

  if [[ "$KEEP" == "1" ]]; then
    info "KEEP=1: previous outputs left in place"
    return 0
  fi

  say "[0] clearing previous outputs for steps $from-$to"

  if (( from <= 1 && to >= 1 )); then
    rm_glob "selection report" models/selection
  fi

  if (( from <= 2 && to >= 2 )); then
    rm_glob "reachable region" models/reachable_va.json
    rm_glob "generated working briefs" config/briefs.yaml
  fi

  if (( from <= 3 && to >= 3 )); then
    rm_glob "generated stimuli" data/stimuli
    rm_glob "generation log" logs/generation.jsonl logs/pilot.jsonl
  fi

  if (( from <= 4 && to >= 4 )); then
    rm_glob "scored estimator outputs" data/analysis
  fi

  if (( from <= 5 && to >= 5 )); then
    rm_glob "H1 outputs" analysis/h1
  fi

  if (( from <= 6 && to >= 6 )); then
    rm_glob "audience feature reference" models/audience_feature_reference.json

    if [[ -f data/audience/responses.csv ]]; then
      local lines
      local rows

      lines=$(wc -l < data/audience/responses.csv)

      if (( lines > 0 )); then
        rows=$((lines - 1))
      else
        rows=0
      fi

      warn "removing $rows audience responses"
      warn "this is the expensive audience run; copy it elsewhere first if you still need it"
    fi

    rm_glob "audience responses" data/audience
    rm_glob "audience log" logs/audience.jsonl
  fi

  if (( from <= 7 && to >= 7 )); then
    rm_glob \
      "analysis outputs" \
      analysis/h1 \
      analysis/h2 \
      analysis/h3 \
      analysis/mdu \
      analysis/baselines \
      analysis/tables \
      analysis/stage7_scale_usage \
      analysis/diagnostics/h1_robustness
  fi

  if (( from <= 8 && to >= 8 )); then
    rm_glob "listening-page MP3 files" data/stimuli_mp3/*.mp3
    rm_glob "generated listening page" data/stimuli_mp3/*.html
    rm_glob "temporary listening-page files" data/stimuli_mp3/*.tmp
  fi

  if [[ "${target:-}" == "clean" ]]; then
    rm_glob "pipeline run logs" logs/pipeline
  fi

  if [[ "$DRY" != "1" ]]; then
    mkdir -p data/stimuli data/analysis data/audience data/stimuli_mp3 logs
  fi

  printf '%s    cleared%s\n' "$grn" "$off"
}

preflight() {
  say "[0] preflight"

  [[ -n "${VIRTUAL_ENV:-}" ]] || warn "no virtualenv active - run: source .venv/bin/activate"

  local missing=0

  local required_files=(
    src/estimators/select_estimator.py
    src/generator/probe_reachable.py
    src/generator/generate_briefs.py
    src/generator/run_generation.py
    src/analysis/score_estimator_b.py
    src/audience/build_feature_reference.py
    src/audience/run_audience.py
    data/stimuli_mp3/create_index.py
    convert_2_mp3.sh
    analysis/Stage6_h1.R
    analysis/Check_quadrant_confidence.R
    analysis/Check_h1_brief_level.R
    analysis/Check_h1_resampling.R
    analysis/Stage6_h2.R
    analysis/Stage6_baselines.R
    analysis/Stage6_h3_alignment.R
    analysis/Stage6_h3_supplementary.R
    analysis/Stage6_mdu.R
    analysis/Stage7_scale_usage.R
  )

  for f in "${required_files[@]}"; do
    [[ -f "$f" ]] || {
      warn "missing: $f"
      missing=1
    }
  done

  if [[ -f src/estimators/select.py ]]; then
    die "src/estimators/select.py still exists - it shadows Python's stdlib 'select' module. Delete it."
  fi

  [[ $missing -eq 0 ]] || die "files above are missing."

  for f in models/estimator_A.joblib models/estimator_B.joblib; do
    if [[ -f "$f" ]]; then
      local sz
      sz=$(stat -c%s "$f")

      [[ $sz -gt 10000 ]] || die "$f is $sz bytes - a Git-LFS pointer. Run: git lfs pull"
    else
      warn "missing: $f"
    fi
  done

  [[ -d "$DEAM" ]] || warn "DEAM not found at $DEAM (set DEAM=/path)"
  [[ -d "$PMEMO" ]] || warn "PMEmo not found at $PMEMO (set PMEMO=/path)"

  command -v Rscript >/dev/null || warn "Rscript not on PATH"
  command -v ffmpeg >/dev/null || warn "ffmpeg not on PATH (step 8 needs it)"

  curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 \
    || warn "Ollama not responding on :11434 - steps 3 and 6 need it (ollama serve)"

  info "coach=$COACH  judge=$JUDGE  backend=$BACKEND"

  printf '%s    preflight ok%s\n' "$grn" "$off"
}

s1_select() {
  step 1 select_coach python src/estimators/select_estimator.py --deam "$DEAM" --corpus DEAM --name "$COACH" --role optimisation_coach --discriminate-on probe
}

s2_region() {
  step 2 probe_reachable python src/generator/probe_reachable.py --coach "$COACH"
  step 2 generate_briefs python src/generator/generate_briefs.py

  warn "check the region printed above spans all four quadrants before continuing"
}

s3_generate() {
  step 3 run_generation python -W ignore src/generator/run_generation.py --backend "$BACKEND" --coach "$COACH"

  if [[ "$DRY" != "1" ]]; then
    grep -q "coach: $COACH" "$LOGDIR/03_run_generation.log" 2>/dev/null \
      || warn "could not confirm '$COACH' in the generation log - check it used the right coach"
  fi
}

s4_score() {
  step 4 score_incumbent python src/analysis/score_estimator_b.py
  step 4 score_selected python src/analysis/score_estimator_b.py --estimator "$JUDGE"
  step 4 score_coach_diagnostic python src/analysis/score_estimator_b.py --estimator "$COACH"
}

s5_h1() {
  step 5 h1_incumbent Rscript analysis/Stage6_h1.R
  step 5 h1_selected Rscript analysis/Stage6_h1.R "$JUDGE"
  step 5 quadrant_confidence Rscript analysis/Check_quadrant_confidence.R
}

s6_audience() {
  say "[6] synthetic audience - rebuild calibration reference, then run audience"

  step 6 build_audience_reference python -W ignore src/audience/build_feature_reference.py

  info "target-blind audience feature reference rebuilt from the frozen synthesis system"
  info "resume after an interruption: ./run_pipeline.sh audience-resume"

  step 6 run_audience python -W ignore src/audience/run_audience.py --backend "$BACKEND"
}

s6_audience_resume() {
  say "[6] resume synthetic audience"

  [[ -f data/audience/responses.csv ]] \
    || die "no data/audience/responses.csv exists - there is no audience run to resume"

  [[ -f models/audience_feature_reference.json ]] \
    || die "audience feature reference is missing - do not resume against a different calibration"

  info "existing responses and audience feature reference are preserved"

  step 6 resume_audience python -W ignore src/audience/run_audience.py --backend "$BACKEND" --resume
}

s7_h1_checks() {
  step 7 h1_incumbent Rscript analysis/Stage6_h1.R
  step 7 h1_selected Rscript analysis/Stage6_h1.R "$JUDGE"
  step 7 quadrant_confidence Rscript analysis/Check_quadrant_confidence.R
  step 7 h1_brief_level Rscript analysis/Check_h1_brief_level.R "$JUDGE"
  step 7 h1_resampling Rscript analysis/Check_h1_resampling.R "$JUDGE" 10000 20260829
}

s7_h2_checks() {
  step 7 h2 Rscript analysis/Stage6_h2.R
  step 7 baselines Rscript analysis/Stage6_baselines.R
}

s7_h3_checks() {
  step 7 h3 Rscript analysis/Stage6_h3_alignment.R
  step 7 h3_supplementary Rscript analysis/Stage6_h3_supplementary.R
}

s7_analysis() {
  s7_h1_checks
  s7_h2_checks
  s7_h3_checks
  step 7 mdu Rscript analysis/Stage6_mdu.R
  step 7 scale_usage Rscript analysis/Stage7_scale_usage.R
}

s8_page() {
  step 8 convert_mp3 sh ./convert_2_mp3.sh
  step 8 build_page python data/stimuli_mp3/create_index.py

  info "open: data/stimuli_mp3/index.html"
}

checkpoint() {
  printf '\n%s%s%s\n' "$bold" "== CHECKPOINT ===========================================================" "$off"

  printf '\n  Read before starting the audience run:\n\n'

  printf '      ls -1 analysis/h1/\n'
  printf '      cat analysis/h1/quadrant_confidence.txt\n\n'

  printf '  Optional H1 robustness checks: ./run_pipeline.sh h1-checks\n'
  printf '  Continue with:                 ./run_pipeline.sh audience\n'
  printf '  If interrupted:                ./run_pipeline.sh audience-resume\n'
  printf '  Then:                          ./run_pipeline.sh analysis\n'
  printf '                                 ./run_pipeline.sh page\n'

  printf '\n%s%s%s\n' "$bold" "=========================================================================" "$off"
}

run_range() {
  local from="$1"
  local to="$2"

  (( from <= 1 && to >= 1 )) && s1_select
  (( from <= 2 && to >= 2 )) && s2_region
  (( from <= 3 && to >= 3 )) && s3_generate
  (( from <= 4 && to >= 4 )) && s4_score
  (( from <= 5 && to >= 5 )) && s5_h1
  (( from <= 6 && to >= 6 )) && s6_audience
  (( from <= 7 && to >= 7 )) && s7_analysis
  (( from <= 8 && to >= 8 )) && s8_page

  return 0
}

target="${1:-checkpoint}"

if [[ "$target" != "clean" ]]; then
  prune_pipeline_logs
fi

case "$target" in
  checkpoint)      preflight; clean_range 1 5; run_range 1 5; checkpoint ;;
  audience)        preflight; clean_range 6 6; run_range 6 6 ;;
  audience-resume) preflight; s6_audience_resume ;;
  h1-checks)       preflight; s7_h1_checks ;;
  h2-checks)       preflight; s7_h2_checks ;;
  h3-checks)       preflight; s7_h3_checks ;;
  analysis)        preflight; clean_range 7 7; run_range 7 7 ;;
  page)            preflight; clean_range 8 8; run_range 8 8 ;;
  all)             preflight; clean_range 1 8; run_range 1 8 ;;
  preflight)       preflight ;;
  clean)           preflight; clean_range 1 8 ;;

  [1-8])
    end="${2:-$target}"

    [[ "$end" =~ ^[1-8]$ ]] \
      || die "range end must be a number from 1 to 8"

    (( end >= target )) \
      || die "range end ($end) must be >= start ($target)"

    preflight
    clean_range "$target" "$end"
    run_range "$target" "$end"
    ;;

  *)
    sed -n '2,25p' "$0"
    exit 1
    ;;
esac

if [[ "$target" == "clean" ]]; then
  say "finished. main pipeline outputs cleared"
elif [[ "$DRY" == "1" ]]; then
  say "finished. dry run only"
else
  say "finished. logs in $LOGDIR"
fi