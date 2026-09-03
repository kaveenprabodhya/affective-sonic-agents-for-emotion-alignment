# Affective Sonic Agents for Emotion Alignment

**Using a Multi-Agent Approach to Measure the Alignment between Brand-Intended and Audience-Perceived Emotional Positions in Synthetic Sonic Logos for Radio Advertising**

MSc dissertation project — WMG, University of Warwick.

A parametric synthesiser generates short sonic logos toward target valence–arousal (VA) coordinates. Two independently frozen estimators score them. A synthetic audience of 32 OCEAN personas rates them.

## Listen to the stimuli

Explore the generated sonic logos in the public interactive gallery:

**[Open the Sonic Logo Stimuli gallery](https://kaveenprabodhya.github.io/sonic-logo-stimuli/)**

The gallery contains all 16 brand briefs, 48 repeated generation runs, and 96 audio stimuli. Each matched pair shows the first and best candidates, estimator movement, the target VA position, and the synthesiser parameters changed during optimisation. The gallery is a convenient way to inspect the published stimuli; the full pipeline below explains how to reproduce the artefacts locally.

## Quick start

For the shortest path from a fresh checkout to the first H1 result:

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_pipeline.sh preflight
DRY=1 ./run_pipeline.sh all       # inspect the commands first
./run_pipeline.sh                 # generate and score stimuli, then stop at H1
```

The complete pipeline needs FluidSynth, FFmpeg, R, Ollama, the `qwen3:8b` model, and the DEAM/PMEmo corpora. See [Setup](#3-setup) for installation details. The audience stage takes approximately nine hours and is intentionally separated from the default run; use `./run_pipeline.sh audience` when you are ready to run it.

---

## 1. What runs where

| Layer | Language | Location |
|---|---|---|
| Generation, optimisation, audience harness | Python 3.13 | `src/` |
| Statistical analysis (H1–H3, MDU, baselines) | R | `analysis/*.R` |
| LLM (all agents) | Qwen3:8b via Ollama, local | — |
| Audio rendering | FluidSynth + GeneralUser-GS SoundFont | `assets/soundfonts/` |

The same LLM is used for the generator and synthetic-audience agents at every stage that requires an LLM.

---

## 2. Hypotheses and what tests them

**H1** — Optimisation of synthetic sonic logos will improve their emotional alignment with brand-intended valence–arousal targets.

**H2** — Audience-perceived emotional responses to the same sonic logo will differ across OCEAN personality profiles.

**H3** — The degree of alignment between brand-intended and audience-perceived emotional positions will differ across OCEAN-based synthetic audience personas.

| Hypothesis | What is compared | Test | Script | Input |
|---|---|---|---|---|
| **H1** | Intended VA target vs independent Estimator B2, optimised vs non-optimised | Paired t-test on 48 matched pairs, mixed-effects model, sign test, Wilcoxon | `Stage6_h1.R estimator_B2` | `data/analysis/h1_estimator_b_estimator_B2.csv` |
| **H2** | Perceived valence and arousal across OCEAN profiles, same stimuli | Separate mixed-effects models, omnibus LRT, Holm adjustment | `Stage6_h2.R` | `data/audience/responses.csv` |
| **H3** | Distance between intended and persona-perceived VA, by OCEAN trait | Mixed-effects, five traits jointly by LRT | `Stage6_h3_alignment.R` | `data/audience/responses.csv` |

H1 tests RO2, H2 tests RO3, H3 tests RO5. RO4 carries no hypothesis.

H1 is stimulus-level and uses only Stage 3 output. H2 and H3 are audience-level and need the full Stage 4 run.

Estimator B2 is the dissertation's primary independent H1 judge. The incumbent Estimator B remains available as a comparative judge through the commands that omit the estimator argument.

Supporting analyses:

| Script | Output folder |
|---|---|
| `Stage6_mdu.R` — multidimensional unfolding of personas and emotion terms | `analysis/mdu/` |
| `Stage6_baselines.R` — OCEAN spread vs the two control agents | `analysis/baselines/` |

---

## 3. Setup

### 3.1 Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install --upgrade pip
pip install -r requirements.txt
python -c "import librosa, soundfile, sklearn, pandas, matplotlib; print('ok', librosa.__version__)"
```

> Do not run with conda `(base)` active. If your prompt shows `(.venv) (base)`, run `conda deactivate` first.

Check the version the estimators were frozen under (recorded in `models/estimator_A.meta.json`):

```bash
python -c "import sklearn; print(sklearn.__version__)"   # expect 1.9.0
```

### 3.2 Git LFS — do this before anything else

`*.joblib`, `*.wav`, `*.mp3`, `*.jsonl`, `*.pkl` and `*.sf2` are stored in Git LFS. A fresh clone or a GitHub "Download ZIP" gives 131-byte pointer stubs.

```bash
git lfs install
git lfs pull
ls -la models/          # estimator_A.joblib should be ~437 MB, not 134 bytes
```

If files are still tiny: `git lfs fetch --all`.

### 3.3 System dependencies

```bash
sudo apt install fluidsynth ffmpeg r-base-core
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
ollama run qwen3:8b "reply with the single word: ok"     # smoke test
```

### 3.4 R packages

```bash
sudo Rscript -e 'install.packages(c("lme4","lmerTest","smacof","ggplot2","ggrepel","dplyr","tidyr"), repos="https://cloud.r-project.org")'
Rscript -e 'library(lme4); library(lmerTest); library(smacof); cat("R deps ok\n")'
```

---

## 4. Folder structure — which is which

```
.
├── config/                          # everything tunable lives here
│   ├── experiment.yaml              # stage sizes, optimisation rule, selection rule, model settings
│   ├── briefs.yaml                  # 16 brand briefs, targets rescaled into the reachable region  <- used by the pipeline
│   ├── briefs_full_range.yaml       # the original full-range targets, never overwritten
│   ├── personas.yaml                # 32 OCEAN profiles + 2 controls
│   ├── questionnaire.yaml           # the Q1–Q12 survey instrument
│   └── prompts/
│       ├── generator_initial.txt    # first parameter proposal from brand description
│       ├── audience_system.txt      # OCEAN persona system prompt
│       ├── audience_system_generic.txt   # generic-listener control
│       ├── audience_system_neutral.txt   # neutral control
│       └── audience_user.txt        # shared user message: feature block + survey
│
├── src/
│   ├── config_loader.py             # finds project root, loads config/*.yaml
│   ├── features/extracts.py         # 70-feature acoustic extraction (shared by A, B, audience)
│   ├── estimators/
│   │   ├── data.py                  # corpus loading, feature caching
│   │   ├── model.py                 # fit / freeze / load
│   │   ├── build.py                 # ENTRY: builds and freezes Estimator A and B
│   │   └── select_estimator.py      # ENTRY: architecture comparison for the judge
│   ├── generator/
│   │   ├── synth.py                 # parameter schema, validation, MIDI -> WAV render
│   │   ├── probe_reachable.py       # ENTRY: maps the reachable VA region
│   │   ├── generate_briefs.py       # ENTRY: rescales brief targets into that region
│   │   ├── loop.py                  # the optimisation controller (stopping rule lives here)
│   │   └── run_generation.py        # ENTRY: Stage 2, produces the 96 stimuli
│   ├── analysis/score_estimator_b.py # ENTRY: Stage 3, held-out scoring for H1
│   ├── analysis/clean_audience.py    # diagnoses/salvages merged audience runs
│   ├── audience/
│   │   ├── build_feature_reference.py # target-blind acoustic calibration for audience prompts
│   │   ├── survey.py                # prompt assembly, response parsing
│   │   └── run_audience.py          # ENTRY: Stage 4, the 9,792 responses
│   └── llm/client.py                # Ollama/Anthropic/mock backend, logs every call
│
├── analysis/                        # R scripts + their outputs
│   ├── Stage6_h1.R      -> analysis/h1/          stimulus alignment
│   ├── Stage6_h2.R      -> analysis/h2/          persona differentiation
│   ├── Stage6_h3_alignment.R -> analysis/h3/     audience alignment ~ OCEAN traits
│   ├── Stage6_mdu.R     -> analysis/mdu/         unfolding
│   │                                               ├── mdu_fit_summary.csv   (Stress-1, all solutions)
│   │                                               ├── pooled/               (all stimuli)
│   │                                               └── quadrants/            (4x, optimised only)
│   ├── Stage6_h3_supplementary.R -> analysis/h3/  centred distance + rank agreement
│   ├── Stage6_baselines.R -> analysis/baselines/  neutral + generic control comparison
│   ├── Stage7_scale_usage.R -> analysis/stage7_scale_usage/   response-scale descriptives
│   ├── Check_quadrant_confidence.R -> analysis/h1/quadrant_confidence.txt
│   │                                              held / marginal / crossed per estimator,
│   │                                              plus raw, chance and kappa agreement
│   ├── Check_mdu_degeneracy.R -> analysis/mdu/    ordinal vs interval degeneracy evidence
│   ├── Check_h1_brief_level.R -> analysis/diagnostics/h1_robustness/
│   │                                              brief-level H1 sensitivity analysis
│   ├── Check_h1_resampling.R -> analysis/diagnostics/h1_robustness/
│   │                                              cluster bootstrap + exact sign-flip test
│   ├── Check_audience_optimisation_effect.R        audience-rated opt vs non-opt alignment
│   ├── Check_estimator_b_range.R                  judge spread vs own RMSE (superseded)
│   └── Diagnose_baselines.R                       why the baselines pivot sees duplicates
│
├── data/
│   ├── stimuli/                     # 96 WAV files + manifest.json (generation record)
│   ├── stimuli_mp3/                 # MP3 copies + create_index.py + index.html
│   ├── analysis/
│   │   ├── h1_estimator_b.csv       # 48 matched pairs, held-out estimator distances -> input to H1
│   │   ├── h1_estimator_b_<judge>.csv  # same, for any additional judge scored
│   │   └── integrity.json           # per-stimulus sha256, peak, RMS, duration
│   └── audience/
│       ├── responses.csv             # 9,792 rows -> input to H2, H3, MDU, baselines
│       ├── audience_protocol_sha256.txt # hash binding responses to the audience protocol
│       ├── feature_reference.json    # calibration snapshot used for this audience run
│       └── stimulus_feature_blocks.json # exact target-blind prompt features by stimulus
│
├── models/
│   ├── estimator_A.joblib(.meta.json)   # DEAM, random forest — guides optimisation
│   ├── estimator_B.joblib(.meta.json)   # PMEmo, SVR — held out, scores H1
│   ├── estimator_B2.joblib(.meta.json)  # architecture-selected judge, if built
│   ├── estimator_A2.joblib(.meta.json)  # architecture-selected coach, if built
│   ├── selection/                       # candidate_comparison.csv + selection_report.txt
│   ├── reachable_va.json                # measured reachable region
│   ├── audience_feature_reference.json  # calibration distribution built before Stage 4
│   └── cache/                           # feature caches (safe to delete, slow to rebuild)
│
├── logs/                            # every LLM call, with ts / tokens / prompt / response
│   ├── generation.jsonl
│   ├── audience.jsonl
│   ├── pilot.jsonl
│   └── pipeline/<timestamp>/         # one log per run_pipeline.sh step
│
├── run_pipeline.sh                  # staged runner with preflight + per-step logs
│
└── spike/                           # exploratory work, not part of the pipeline
    ├── estimator_transfer_test.py   # window-length transfer check
    ├── estimator_family_comparison.py # feature-family comparison
    ├── persona_pilot.py             # persona differentiation pilot
    ├── audience_feature_pilot.py    # pilot of target-blind audience feature prompts
    ├── deterministic_search_test.py # deterministic optimiser experiment
    ├── local_search_test.py         # local-search experiment
    ├── render_soundfont_ab.py       # soundfont rendering comparison
    └── rhythm_gallery.py            # rhythm-pattern rendering gallery
```

`briefs.yaml` holds the rescaled targets used by the pipeline. `briefs_full_range.yaml` holds the originals and is never overwritten.

Estimator A guides the optimisation loop. Estimator B is held out and used only in Stage 3.

---

## 5. The pipeline

Later stages depend on earlier outputs, so run in order.

The `Stage` labels below describe the research procedure. `run_pipeline.sh` uses runner step numbers 1–8 for the same work, with research Stage 4 (the audience) implemented as runner step 6.

### Stage 0 — Build and freeze the estimators

```bash
python src/estimators/build.py --deam datasets/DEAM --pmemo datasets/PMEmo
```

Writes `models/estimator_{A,B}.joblib` and their `.meta.json` (R², RMSE, freeze timestamp, seed).

```bash
# quick build (caps songs per corpus)
python src/estimators/build.py --deam datasets/DEAM --pmemo datasets/PMEmo --songs 150
# rebuild just one
python src/estimators/build.py --pmemo datasets/PMEmo --only B --family-b svr
```

Stale cache: `rm -f models/cache/pmemo_*` then rebuild.

### Stage 1 — Map the reachable region and place the brief targets

```bash
python src/generator/probe_reachable.py          # -> models/reachable_va.json
python src/generator/generate_briefs.py          # -> rewrites config/briefs.yaml

# with a different coach
python src/generator/probe_reachable.py --coach estimator_A2
```

The reachable region is measured *through the coach*, so the coach used here must be the one used in Stage 2. The name is recorded in `reachable_va.json`.

`probe_reachable.py` scores a fixed sample of parameter combinations with Estimator A. `generate_briefs.py` rescales the full-range targets into that region, per axis and per sign.

### Stage 2 — Generate the stimuli

```bash
time python -W ignore src/generator/run_generation.py --backend ollama
```

16 briefs × 3 runs = 48 matched pairs = 96 stimuli, into `data/stimuli/` with `manifest.json`.

Stopping rule (`src/generator/loop.py`): the loop stops when the Estimator A prediction is within `threshold` of the target **and** on the same side of both axes as the target, or when `iteration_cap` is hit. `best` is tracked every iteration and prefers quadrant-holding candidates before shortest distance.

`manifest.json` is rewritten from scratch with only the briefs in that run, and matching WAV files are overwritten — so a `--briefs` pilot replaces a 48-run manifest with a 12-run one.

Output flags: `*` = both criteria met, `Q` = quadrant held.

```
 *Q B09 run0: non-opt d=0.124 -> opt d=0.124  (1 iters)
  x B01 run0: non-opt d=0.316 -> opt d=0.138  (10 iters)
```

Config knobs (`config/experiment.yaml` → `generation.optimisation`):

| Key | Meaning |
|---|---|
| `threshold` | proximity tolerance on Estimator A distance |
| `require_quadrant` | enforce sign agreement; `false` gives distance-only behaviour |
| `iteration_cap` | max iterations per run |

`--coach` selects the estimator that guides the loop (default `estimator_A`) and records the name in `manifest.json`:

```bash
python -W ignore src/generator/run_generation.py --backend ollama --coach estimator_A2
```

### Stage 2b — Architecture selection for the held-out judge

Optional. Runs after generation, because one of the two criteria is measured on the study stimuli.

For the **judge**, discrimination is measured on the 96 generated stimuli — the files it will score:

```bash
python src/estimators/select_estimator.py --pmemo datasets/PMEmo --no-freeze   # report only
python src/estimators/select_estimator.py --pmemo datasets/PMEmo --name estimator_B2
```

For the **coach**, use `--discriminate-on probe`. Discrimination is then measured on a deterministic sweep of the synthesiser parameter space rather than on the study stimuli, which were produced by the previous coach:

```bash
python src/estimators/select_estimator.py --deam datasets/DEAM --corpus DEAM \
    --name estimator_A2 --role optimisation_coach --discriminate-on probe
```

Compares ridge, linear SVR, RBF SVR (gamma grid), random forest, gradient boosting and MLP. Split three ways by song: train fits, validation ranks hyperparameters within a family, test gives the reported metric.

Selection criteria are read from `config/experiment.yaml` → `estimator_selection`:

| Criterion | What it measures |
|---|---|
| In-domain R² | held-out songs from the estimator's own corpus |
| Discrimination | prediction SD on the 96 study stimuli ÷ the candidate's own RMSE |

Candidates within `r2_tolerance` of the best mean R² pass the gate; among those the highest discrimination wins; ties break on R². The rule is hashed into the report and the frozen metadata.

Outputs:

```
models/selection/candidate_comparison.csv    # every family winner, both criteria
models/selection/selection_report.txt        # rule hash, domain-shift diagnostic, ranked table
models/<name>.joblib + .meta.json            # the frozen winner
```

| Flag | Default | Meaning |
|---|---|---|
| `--corpus` | `PMEmo` | which corpus to train on; pair with `--pmemo` or `--deam` |
| `--name` | `estimator_B2` | name to freeze under; must differ from `estimator_A` and `estimator_B` |
| `--role` | `held_out_H1_judge` | recorded in the metadata |
| `--discriminate-on` | `study` | `study` = the 96 stimuli, `probe` = the parameter sweep |
| `--probe-n` | `300` | probe grid size when `--discriminate-on probe` |
| `--no-freeze` | off | write the report only |
| `--songs` | all | cap songs for a quick run |

`--name` must differ from `estimator_A` and `estimator_B`; the script refuses to overwrite them.

Swapping the **coach** means re-running Stages 1 and 2 with `--coach <name>`, since the reachable region and the brief targets are both derived through the coach.

### Stage 3 — Score with the held-out judge

```bash
python src/analysis/score_estimator_b.py                              # estimator_B
python src/analysis/score_estimator_b.py --estimator estimator_B2     # any other judge
```

Writes `data/analysis/h1_estimator_b.csv` (or `..._estimator_B2.csv`) and `data/analysis/integrity.json`. Read-only on the stimuli. Prints the judge's spread across the stimuli next to its own RMSE.

The coach can be scored the same way, which adds it as a column to the quadrant check and the listening page:

```bash
python src/analysis/score_estimator_b.py --estimator estimator_A2
```

Do not pass the coach to `Stage6_h1.R`. The loop selected `best` by minimising the coach's own distance, so an H1 test against the coach is true by construction — its numbers are a description of what the optimiser did, not evidence about it.

### Stage 4 — Run the synthetic audience

```bash
python -W ignore src/audience/build_feature_reference.py
time python -W ignore src/audience/run_audience.py --backend ollama
```

34 agents (32 OCEAN + neutral + generic) × 96 stimuli × 3 repetitions = 9,792 responses → `data/audience/responses.csv`. Budget ~9 hours. Interruptible:

```bash
python src/audience/run_audience.py --backend ollama --resume    # skips rows already written
```

The first command renders 300 sounds sampled independently from the synthesiser space and writes `models/audience_feature_reference.json`. Audience prompts use this target-blind distribution to express acoustic features consistently; it contains no study target, condition, estimator output, emotion label, or brand brief. Build it once at the start of a fresh audience run. Do not rebuild it while resuming an interrupted run.

A run without `--resume` starts a new file, moving the previous one to `responses_<timestamp>.csv`. A run with `--resume` refuses to continue if the existing rows were scored against different stimuli: generation reuses filenames, so resuming after the brief targets change would merge two studies into one file with no way to separate them afterwards. `--force` overrides; `python src/analysis/clean_audience.py` reports and salvages a file that has already been merged.

### Stage 5 — Convert stimuli for listening (optional)

```bash
sh ./convert_2_mp3.sh
python data/stimuli_mp3/create_index.py
xdg-open data/stimuli_mp3/index.html
```

One card per matched pair. Each card holds the brand brief that seeded it, both audio versions labelled with the winning iteration, a VA plot for Estimator A and for every judge scored, a distance row per estimator marked closer/further, and the synthesiser parameters the optimiser changed. Filter by quadrant, by whether quadrant was held, or by unchanged pairs. Only one audio player runs at a time.

Judges are discovered automatically from `data/analysis/h1_estimator_b*.csv`, so anything scored in Stage 3 appears without a flag. A judge that has not been scored is simply absent, and the page prints the command that would add it.

| Flag | Default | Meaning |
|---|---|---|
| `--judges` | every estimator scored | comma-separated names, e.g. `estimator_B,estimator_B2`; also fixes column order |
| `--scale` | `shared` | `shared` puts every estimator on one axis; `per-judge` scales each to its own range |

```bash
python data/stimuli_mp3/create_index.py --scale per-judge          # read positions
python data/stimuli_mp3/create_index.py --judges estimator_B2      # one column only
```

**Where the numbers come from.** The coach column is read from `data/stimuli/manifest.json` (`non_optimised.est` / `optimised.est`) — the estimates recorded during generation. Every other column is read from a `data/analysis/h1_estimator_b*.csv` written by Stage 3. Nothing is recomputed or interpolated.

**Adding a column.** `score_estimator_b.py` accepts any frozen estimator, so any of them can be put on the page:

```bash
python src/analysis/score_estimator_b.py --estimator estimator_A2
python data/stimuli_mp3/create_index.py
```

The coach column is labelled from the `coach` field the generation run recorded, so if `estimator_A2` produced the stimuli the column says so, and the quadrant badge is attributed to it.

**Plot size and scrolling.** Plots are a fixed 168 px and never shrink as columns are added. The strip scrolls horizontally with scroll-snap, so four or more estimators are read by sliding rather than squinting. `PLOT_PX` at the top of `create_index.py` changes the size.

**Axis scale.** Axes are computed from the data with 10% headroom, so no point is ever clipped, and the scale is printed in each column heading (`Estimator B · held out · ±0.30`) rather than inside the plot, where it would sit under the data. Under `shared` every estimator uses one axis, which keeps columns comparable — a compressed estimator visibly stays compressed next to one that spreads, but its dots may overlap. Under `per-judge` each gets its own axis, so positions are readable and columns are **not** comparable; the legend says so. `DOMAIN_MIN` and `DOMAIN_HEADROOM` set the floor and the headroom.

The coordinates being plotted are printed under each plot and the target in the card header, so any plot can be checked against the CSV.

### Stage 6 — Statistical analysis

```bash
Rscript analysis/Stage6_h1.R                     # H1, estimator_B
Rscript analysis/Stage6_h1.R estimator_B2        # H1, another judge -> analysis/h1/h1_results_b2.txt
Rscript analysis/Check_quadrant_confidence.R     # quadrant classification + judge agreement
Rscript analysis/Check_h1_brief_level.R estimator_B2
Rscript analysis/Check_h1_resampling.R estimator_B2 10000 20260829
Rscript analysis/Stage6_h2.R
Rscript analysis/Stage6_h3_alignment.R
Rscript analysis/Stage6_h3_supplementary.R
Rscript analysis/Stage6_mdu.R
Rscript analysis/Stage6_baselines.R
Rscript analysis/Stage7_scale_usage.R
```

#### What each produces

**`Stage6_h1.R` → `analysis/h1/h1_results_<judge>.txt`**
The judge's spread across the stimuli, descriptives, mixed-effects model, paired t-test, tie count, sign test, Wilcoxon signed-rank, and a quadrant check counting how often optimised and non-optimised stimuli sit on the target's side of both axes. The default incumbent writes `h1_results_b.txt`; `estimator_B2` writes `h1_results_b2.txt`.

**`Check_quadrant_confidence.R` → `analysis/h1/quadrant_confidence.txt`**
Reads every `data/analysis/h1_estimator_b*.csv` and classifies each optimised stimulus per estimator as **held** (same sign as the target on both axes, both coordinates at least `margin` clear of the axes), **marginal** (same sign, but within `margin` of an axis, so a small error flips it) or **crossed** (opposite sign on at least one axis). The margin is not a property of the data, so the table is reported across several values — 0.00, 0.02, 0.05, 0.10 by default, or pass your own:

```bash
Rscript analysis/Check_quadrant_confidence.R 0.02 0.05 0.10 0.15
```

Also breaks results down by intended quadrant, and reports agreement between estimators as raw agreement, chance agreement, and Cohen's kappa. Kappa matters here because an estimator concentrated in one quadrant inflates raw agreement with everything else: where one estimator is near-constant, raw agreement equals chance by construction and kappa is 0. Only pairs where both estimators have real spread are interpretable.

**`Stage6_h2.R` → `analysis/h2/`**
`h2_results.txt` — mixed-effects models for perceived valence and perceived arousal, omnibus LRT across the five OCEAN traits, singularity check.

**`Stage6_h3_alignment.R` → `analysis/h3/`**
`h3_results.txt`, `h3_distance_by_quadrant.png`, `h3_offset_vectors.png` — alignment distance against the five traits with a joint LRT, plus descriptive quadrant distances, signed valence/arousal offsets, and an exploratory quadrant × OCEAN interaction test.

**`Stage6_h3_supplementary.R` → `analysis/h3/`**
Separates two things raw alignment distance cannot tell apart: a persona that rates everything higher or lower (already captured by H2) from a persona that orders the stimuli differently (genuinely new). Two checks. **Centred distance** subtracts each persona's own mean valence and arousal before recomputing distance, removing the level shift; traits that still predict it indicate differences in accuracy rather than in level. **Rank agreement** correlates personas on which stimuli they align best with, since a pure level shift leaves the ordering untouched. Diagnostic only — it does not replace or rewrite the primary H3 test.

**`Stage6_mdu.R` → `analysis/mdu/`**
Interval unfolding via `smacof`: one pooled solution and four quadrant-specific solutions on optimised logos only.

```
analysis/mdu/
├── mdu_fit_summary.csv           # Stress-1 for every solution
├── pooled/
│   ├── mdu_persona_emotion.png
│   ├── mdu_emotions_only.png
│   ├── mdu_persona_coords.csv
│   ├── mdu_emotion_coords.csv
│   └── persona_emotion_matrix.csv
└── quadrants/{HV_HA,HV_LA,LV_HA,LV_LA}/   # same five files each
```

The script prints a warning when Stress-1 falls below 0.01.

**`Stage6_baselines.R` → `analysis/baselines/`**
`baseline_results.txt` plus per-condition and per-stimulus CSVs and distribution plots. Repetitions are averaged per agent–stimulus, the 32 OCEAN personas are averaged to a stimulus-level mean, and that mean is compared against each control.

**`Stage7_scale_usage.R` → `analysis/stage7_scale_usage/`**
How the synthetic respondents used the two nine-point scales, and where perceived positions fell relative to the intended quadrants. Descriptive evidence for the response-scale paragraph in Section 4.3.1. Responses are nested (persona × stimulus × repetition), so no inferential test is run here.

**`Check_h1_brief_level.R` and `Check_h1_resampling.R` → `analysis/diagnostics/h1_robustness/`**
Robustness checks for the selected H1 judge. The first averages the three runs within each brief and tests the 16 brief means. The second keeps runs clustered by brief, reports a cluster-bootstrap confidence interval, and performs the exact `2^16` sign-flip test. Arguments are `[judge]` for the brief-level check and `[judge] [bootstrap repetitions] [seed]` for resampling.

---

### On-demand diagnostics

Not part of a normal run — each answers one question and is kept because the answer belongs in the write-up.

| Script | Question | Output |
|---|---|---|
| `Check_mdu_degeneracy.R` | Does ordinal unfolding degenerate on this data, and does interval avoid it? | `analysis/mdu/degeneracy_check.txt`, `degeneracy_comparison.csv`, `degeneracy_configurations.png` |
| `Check_audience_optimisation_effect.R` | Did optimisation change intended–perceived distance for the OCEAN audience? | mixed-model and 48-pair results printed to stdout |
| `Check_estimator_b_range.R` | How much does a judge's prediction vary across the stimuli, against its own RMSE? | prints to stdout |
| `Diagnose_baselines.R` | Why does the baselines pivot see duplicate rows? | prints to stdout |

`Check_mdu_degeneracy.R` fits the persona × emotion matrix under interval, ratio, and ordinal transformations at three penalty strengths, and reports Stress-1 alongside the coefficient of variation of the fitted distances. The CV is what separates a good solution from a degenerate one: a degenerate solution reaches *low* stress precisely by collapsing every point of one set onto every point of the other, so stress alone cannot detect it. Note that `smacof::unfolding()` minimises a penalised stress (Busing, Groenen & Heiser, 2005) designed to prevent exactly this, controlled by `omega`; the script varies `omega` down to 0 so the transformation's behaviour is visible with and without that protection.

Run the audience optimisation diagnostic after Stage 4 with:

```bash
Rscript analysis/Check_audience_optimisation_effect.R
```

`Check_estimator_b_range.R` is largely superseded — `Stage6_h1.R` prints the judge's spread at the top of every run and `Check_quadrant_confidence.R` reports it per estimator.

---

## 6. Running the pipeline

`run_pipeline.sh` wraps every stage below with preflight checks, per-step logging, and a stop at the first failure. Each stage can also be run on its own, so a crash part-way through does not mean repeating the stages before it.

```bash
chmod +x run_pipeline.sh
source .venv/bin/activate

DRY=1 ./run_pipeline.sh all      # print every command, run nothing
./run_pipeline.sh                # steps 1-5, then stop at the checkpoint
```

The default run stops after H1 so the result can be read before committing to the 9-hour audience stage:

```bash
cat analysis/h1/h1_results_b2.txt

./run_pipeline.sh audience       # step 6,  ~9 h
./run_pipeline.sh audience-resume # resume step 6 without replacing its calibration/data
./run_pipeline.sh analysis       # step 7,  complete statistical analysis
./run_pipeline.sh page           # step 8,  MP3s + listening page
```

### Targets

| Argument | Steps | What it does |
|---|---|---|
| *(none)* | 1–5 | select coach, re-measure region, regenerate, score, H1 — then stop |
| `audience` | 6 | rebuild the target-blind feature reference, then run 9,792 responses |
| `audience-resume` | 6 | resume existing responses while preserving the existing feature reference |
| `h1-checks` | 7 | H1, quadrant confidence, brief-level sensitivity, bootstrap and exact sign-flip checks |
| `h2-checks` | 7 | H2 and control-baseline analyses only |
| `h3-checks` | 7 | H3 and its supplementary analyses only |
| `analysis` | 7 | complete H1/H2/H3 analyses, robustness checks, MDU, baselines and scale usage |
| `page` | 8 | MP3 conversion and the listening page |
| `all` | 1–8 | everything, no checkpoint pause |
| `preflight` | 0 | environment checks only |
| `clean` | — | clear every generated output, run nothing |
| `3` | 3 | a single numbered step |
| `4 5` | 4–5 | a range of steps |

### Steps

| # | Step | Time |
|---|---|---|
| 0 | preflight — files, LFS stubs, venv, Ollama, Rscript, ffmpeg | instant |
| 1 | coach architecture selection (`--discriminate-on probe`) | ~15 min |
| 2 | re-measure the reachable region, re-place the brief targets | ~5 min |
| 3 | regenerate all 96 stimuli | ~20 min |
| 4 | score with both judges, plus the coach as a diagnostic | ~6 min |
| 5 | H1 for both judges, quadrant confidence | seconds |
| 6 | rebuild audience feature reference, then run the synthetic audience | ~9 h |
| 7 | H1/H2/H3 analyses, H1 robustness checks, MDU, baselines and scale usage | minutes |
| 8 | MP3 conversion, listening page | ~2 min |

### Outputs are replaced, not added to

Before a step runs, whatever it produced last time is deleted. A rerun therefore cannot leave orphans behind — stimuli from briefs no longer in the set, scored CSVs for judges no longer in use, or LLM logs spanning several studies.

Cleaning is scoped to the steps requested, so `./run_pipeline.sh analysis` clears the analysis outputs and leaves the stimuli and audience data alone.

| Steps | Cleared first |
|---|---|
| 1 | `models/selection/` |
| 2 | `models/reachable_va.json`, `config/briefs.yaml` |
| 3 | `data/stimuli/`, `logs/generation.jsonl`, `logs/pilot.jsonl` |
| 4 | `data/analysis/` |
| 5 | `analysis/h1/` |
| 6 | `models/audience_feature_reference.json`, `data/audience/`, `logs/audience.jsonl` |
| 7 | `analysis/{h1,h2,h3,mdu,baselines,tables,stage7_scale_usage}/`, `analysis/diagnostics/h1_robustness/` |
| 8 | `data/stimuli_mp3/*.mp3`, `*.html`, `*.tmp` |

Scripts, configs, corpora and frozen estimators are never touched — only generated artefacts.

```bash
DRY=1 ./run_pipeline.sh all       # lists exactly what would be removed
./run_pipeline.sh clean           # clear everything, run nothing
KEEP=1 ./run_pipeline.sh all      # leave previous outputs in place
```

Step logs under `logs/pipeline/<timestamp>/` are kept for the last `KEEP_RUNS` runs (default 5) and older ones are pruned.

### Environment overrides

| Variable | Default | Meaning |
|---|---|---|
| `COACH` | `estimator_A2` | coach to select and generate with |
| `JUDGE` | `estimator_B2` | additional judge to score and test |
| `DEAM` | `datasets/DEAM` | DEAM corpus root |
| `PMEMO` | `datasets/PMEmo` | PMEmo corpus root |
| `BACKEND` | `ollama` | LLM backend |
| `DRY` | `0` | `1` prints commands without running them |
| `KEEP` | `0` | `1` skips clearing previous outputs |
| `KEEP_RUNS` | `5` | how many runs of step logs to retain |

```bash
COACH=estimator_A3 ./run_pipeline.sh 1 3
DEAM=/mnt/data/DEAM ./run_pipeline.sh
```

`analysis` deliberately repeats H1 and the quadrant check, so that one target reproduces the complete result set from whatever is currently in `data/`. Both take seconds and are idempotent.

### Logs and failure behaviour

Every step writes to `logs/pipeline/<timestamp>/NN_<step>.log` and reports its own duration. The script uses `set -Eeuo pipefail` with an ERR trap, so the first failure halts the run rather than letting later stages work on stale inputs. Step 3 greps its own log to confirm the generation actually used the requested coach.

Preflight fails hard on a surviving `src/estimators/select.py` (it shadows Python's stdlib `select` module) and on Git-LFS pointer stubs, which otherwise surface as `KeyError: 118`. Missing Ollama, Rscript or ffmpeg are warnings, since not every target needs them.

### Running stages by hand

The runner is a convenience, not a requirement. Each stage is an ordinary command:

```bash
# after changing the judge only — no audience rerun needed
python src/estimators/select_estimator.py --pmemo datasets/PMEmo --name estimator_B2
python src/analysis/score_estimator_b.py --estimator estimator_B2
Rscript analysis/Stage6_h1.R estimator_B2
Rscript analysis/Check_quadrant_confidence.R

# after changing the coach — starts from Stage 1, because the reachable region
# and the brief targets are both derived through the coach
python src/estimators/select_estimator.py --deam datasets/DEAM --corpus DEAM \
    --name estimator_A2 --role optimisation_coach --discriminate-on probe
python src/generator/probe_reachable.py --coach estimator_A2
python src/generator/generate_briefs.py
time python -W ignore src/generator/run_generation.py --backend ollama --coach estimator_A2
python src/analysis/score_estimator_b.py
python src/analysis/score_estimator_b.py --estimator estimator_B2
Rscript analysis/Stage6_h1.R
Rscript analysis/Stage6_h1.R estimator_B2
Rscript analysis/Check_quadrant_confidence.R
Rscript analysis/Check_h1_brief_level.R estimator_B2
Rscript analysis/Check_h1_resampling.R estimator_B2 10000 20260829
python -W ignore src/audience/build_feature_reference.py
time python -W ignore src/audience/run_audience.py --backend ollama
Rscript analysis/Stage6_h2.R
Rscript analysis/Stage6_h3_alignment.R
Rscript analysis/Stage6_h3_supplementary.R
Rscript analysis/Stage6_mdu.R
Rscript analysis/Stage6_baselines.R
Rscript analysis/Stage7_scale_usage.R
```

Scoring is required before H1: `run_audience.py` reads only `manifest.json` and will run without it, but `score_estimator_b.py` is what writes the H1 input and refreshes `integrity.json`.

Changing the judge does not require an audience rerun — it is used only in Stage 3, and the audience agents receive acoustic features, not estimator output.

---

## 7. Fast dry runs

```bash
# Generation, one brief per quadrant (12 runs, ~5 min)
python -W ignore src/generator/run_generation.py --backend ollama --briefs B01,B05,B09,B13

# Generation, first N briefs
python -W ignore src/generator/run_generation.py --backend ollama --limit 2

# No LLM at all — checks wiring only
python src/generator/run_generation.py --backend mock --limit 2

# Audience, tiny slice
python src/audience/run_audience.py --backend ollama --limit-agents 8 --limit-stimuli 40

# Persona pilot
python spike/persona_pilot.py --backend ollama --model qwen3:8b

# Architecture selection, report only
python src/estimators/select_estimator.py --pmemo datasets/PMEmo --no-freeze

# The whole pipeline, printed but not executed
DRY=1 ./run_pipeline.sh all

# Coach selection, report only (renders the probe grid)
python src/estimators/select_estimator.py --deam datasets/DEAM --corpus DEAM \
    --discriminate-on probe --probe-n 60 --no-freeze
```

`--briefs` picks specific IDs. `--limit N` takes the *first* N briefs, and briefs are ordered in quadrant blocks of four, so `--limit 2` only covers HV_HA.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 118` loading a `.joblib` | File is a Git-LFS pointer (`118` = ASCII `v` of `version https://...`) | `git lfs pull` |
| Any file is exactly ~131–134 bytes | LFS content not fetched | `git lfs pull`, then `git lfs fetch --all` |
| `pickle` resolving from `~/anaconda3/` | conda `(base)` active alongside `.venv` | `conda deactivate`, re-activate `.venv` |
| `No soundfont found` | `.sf2` missing or an LFS stub | `git lfs pull`, or pass `--soundfont path/to.sf2` |
| Ollama connection refused | Daemon not running | `ollama serve`, check with `ollama ps` |
| R: `there is no package called 'smacof'` | Installed for the wrong user | `sudo Rscript -e 'install.packages("smacof", repos="https://cloud.r-project.org")'` |
| `No manifest at ...` from `select_estimator.py` | Generation has not run | Run Stage 2 first |
| `cannot import name 'grid_params'` | Stale copy of `probe_reachable.py` or `synth.py` | `grid_params` lives in `synth.py`; update both files |
| A step ran but used the wrong estimator | `--coach` omitted | Check the `coach:` line in the generation output, or run `./run_pipeline.sh preflight` |
| Estimator loads but predictions look wrong | sklearn version drift | Compare `python -c "import sklearn; print(sklearn.__version__)"` against the `.meta.json` |

Run `./run_pipeline.sh preflight` to check all of the above at once.

**Timing reference** (measured, Qwen3:8b local, ~2.9 s per LLM call):

| Stage | Calls | Time |
|---|---|---|
| Pilot (4 briefs × 3 runs) | ~12 initial proposals | ~5 min |
| Full generation (16 briefs × 3 runs) | ~48 initial proposals | ~20 min |
| Estimator scoring | 0 | ~1 min |
| Architecture selection (judge) | 0 | ~10 min (cached features) |
| Architecture selection (coach, probe grid) | 0 | ~15 min (300 renders) |
| Audience | 9,792 | ~9 h |
| R analysis (complete target) | 0 | minutes |

---

## 9. What gets recorded

- `models/*.meta.json` — corpus, model family, hyperparameters, held-out metrics, sklearn version, seed, freeze timestamp. Selected judges also record the selection rule, its hash, the families compared, and the domain-shift diagnostic.
- `logs/*.jsonl` — every LLM call with timestamp, model, temperature, full prompt, response and token counts.
- `data/stimuli/manifest.json` — per run: target, both conditions' parameters and estimator positions, distances, quadrant status, iteration count, and the full iteration history.
- `data/analysis/integrity.json` — sha256, peak, RMS and duration for every stimulus.
- `models/selection/` — the candidate comparison table and selection report.

Audio rendering is deterministic given the parameter set; LLM output is not. Audience agents receive only extracted acoustic features — not target coordinates, estimator outputs, brand descriptions, or the optimisation condition.
