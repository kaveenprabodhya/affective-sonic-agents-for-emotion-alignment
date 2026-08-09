# Affective Sonic Agents for Emotion Alignment

**Using a Multi-Agent Approach to Measure the Alignment between Brand-Intended and Audience-Perceived Emotional Positions in Synthetic Sonic Logos for Radio Advertising**

MSc dissertation project — WMG, University of Warwick.

A parametric synthesiser generates short sonic logos toward target valence–arousal (VA) coordinates. Two independently frozen estimators score them. A synthetic audience of 32 OCEAN personas rates them.

---

## 1. What runs where

| Layer | Language | Location |
|---|---|---|
| Generation, optimisation, audience harness | Python 3.13 | `src/` |
| Statistical analysis (H1–H3, MDU, baselines) | R | `analysis/*.R` |
| LLM (all agents) | Qwen3:8b via Ollama, local | — |
| Audio rendering | FluidSynth + GeneralUser-GS SoundFont | `assets/soundfonts/` |

The same LLM is used for every agent and every stage.

---

## 2. Hypotheses and what tests them

**H1** — Optimised sonic logos sit closer to their intended valence–arousal target than their matched non-optimised counterparts, as judged by the held-out estimator.

**H2** — Systematically varied OCEAN profiles produce distinguishable perceived valence and arousal ratings for the same stimulus.

**H3** — The distance between intended and perceived valence–arousal positions differs across OCEAN-based synthetic audience personas.

| Hypothesis | What is compared | Test | Script | Input |
|---|---|---|---|---|
| **H1** | Intended VA target vs held-out estimator, optimised vs non-optimised | Paired t-test on 48 matched pairs, mixed-effects model, sign test, Wilcoxon | `Stage6_h1.R` | `data/analysis/h1_estimator_b.csv` |
| **H2** | Perceived valence and arousal across OCEAN profiles, same stimuli | Separate mixed-effects models, omnibus LRT, Holm adjustment | `Stage6_h2.R` | `data/audience/responses.csv` |
| **H3** | Distance between intended and persona-perceived VA, by OCEAN trait | Mixed-effects, five traits jointly by LRT | `Stage6_h3_alignment.R` | `data/audience/responses.csv` |

H1 tests RO2, H2 tests RO3, H3 tests RO5. RO4 carries no hypothesis.

H1 is stimulus-level and uses only Stage 3 output. H2 and H3 are audience-level and need the full Stage 4 run.

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
│       ├── generator_revision.txt   # revision prompt (receives the signed VA gap)
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
│   ├── audience/
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
│   └── Stage6_baselines.R -> analysis/baselines/  neutral + generic control comparison
│
├── data/
│   ├── stimuli/                     # 96 WAV files + manifest.json (generation record)
│   ├── stimuli_mp3/                 # MP3 copies + index.html for listening
│   ├── analysis/
│   │   ├── h1_estimator_b.csv       # 48 matched pairs, held-out estimator distances -> input to H1
│   │   ├── h1_estimator_b_<judge>.csv  # same, for any additional judge scored
│   │   └── integrity.json           # per-stimulus sha256, peak, RMS, duration
│   └── audience/responses.csv       # 9,792 rows -> input to H2, H3, MDU, baselines
│
├── models/
│   ├── estimator_A.joblib(.meta.json)   # DEAM, random forest — guides optimisation
│   ├── estimator_B.joblib(.meta.json)   # PMEmo, SVR — held out, scores H1
│   ├── estimator_B2.joblib(.meta.json)  # architecture-selected judge, if built
│   ├── selection/                       # candidate_comparison.csv + selection_report.txt
│   ├── reachable_va.json                # measured reachable region
│   └── cache/                           # feature caches (safe to delete, slow to rebuild)
│
├── logs/                            # every LLM call, with ts / tokens / prompt / response
│   ├── generation.jsonl
│   ├── audience.jsonl
│   └── pilot.jsonl
│
└── spike/                           # exploratory work, not part of the pipeline
    ├── estimator_transfer_test.py   # window-length transfer check
    ├── estimator_family_comparison.py # feature-family comparison
    └── persona_pilot.py             # persona differentiation pilot
```

`briefs.yaml` holds the rescaled targets used by the pipeline. `briefs_full_range.yaml` holds the originals and is never overwritten.

Estimator A guides the optimisation loop. Estimator B is held out and used only in Stage 3.

---

## 5. The pipeline

Later stages depend on earlier outputs, so run in order.

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
```

`probe_reachable.py` scores a fixed sample of parameter combinations with Estimator A. `generate_briefs.py` rescales the full-range targets into that region, per axis and per sign.

### Stage 2 — Generate the stimuli

```bash
time python -W ignore src/generator/run_generation.py --backend ollama
```

16 briefs × 3 runs = 48 matched pairs = 96 stimuli, into `data/stimuli/` with `manifest.json`.

Stopping rule (`src/generator/loop.py`): the loop stops when the Estimator A prediction is within `threshold` of the target **and** on the same side of both axes as the target, or when `iteration_cap` is hit. `best` is tracked every iteration and prefers quadrant-holding candidates before shortest distance.

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

### Stage 2b — Architecture selection for the held-out judge

Optional. Runs after generation, because one of the two criteria is measured on the study stimuli.

```bash
python src/estimators/select_estimator.py --pmemo datasets/PMEmo --no-freeze   # report only
python src/estimators/select_estimator.py --pmemo datasets/PMEmo --name estimator_B2
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

`--name` must differ from `estimator_A` and `estimator_B`; the script refuses to overwrite them.

### Stage 3 — Score with the held-out judge

```bash
python src/analysis/score_estimator_b.py                              # estimator_B
python src/analysis/score_estimator_b.py --estimator estimator_B2     # any other judge
```

Writes `data/analysis/h1_estimator_b.csv` (or `..._estimator_B2.csv`) and `data/analysis/integrity.json`. Read-only on the stimuli. Prints the judge's spread across the stimuli next to its own RMSE.

### Stage 4 — Run the synthetic audience

```bash
time python -W ignore src/audience/run_audience.py --backend ollama
```

34 agents (32 OCEAN + neutral + generic) × 96 stimuli × 3 repetitions = 9,792 responses → `data/audience/responses.csv`. Budget ~9 hours. Interruptible:

```bash
python src/audience/run_audience.py --backend ollama --resume    # skips rows already written
```

### Stage 5 — Convert stimuli for listening (optional)

```bash
sh ./convert_2_mp3.sh
python data/stimuli_mp3/create_index.py
xdg-open data/stimuli_mp3/index.html
```

The page pairs each non-optimised and optimised stimulus with its brand brief, both estimators' VA plots, and the synthesiser parameters that changed. Filter by quadrant, by whether quadrant was held, or by unchanged pairs.

### Stage 6 — Statistical analysis

```bash
Rscript analysis/Stage6_h1.R              # H1, estimator_B
Rscript analysis/Stage6_h1.R estimator_B2 # H1, another judge -> analysis/h1/estimator_B2/
Rscript analysis/Stage6_h2.R
Rscript analysis/Stage6_h3_alignment.R
Rscript analysis/Stage6_mdu.R
Rscript analysis/Stage6_baselines.R
```

#### What each produces

**`Stage6_h1.R` → `analysis/h1/`** (or `analysis/h1/<judge>/` when a judge is named)
`h1_results.txt` — the judge's spread across the stimuli, descriptives, mixed-effects model, paired t-test, tie count, sign test, Wilcoxon signed-rank, and a quadrant check counting how often optimised and non-optimised stimuli sit on the target's side of both axes.

**`Stage6_h2.R` → `analysis/h2/`**
`h2_results.txt` — mixed-effects models for perceived valence and perceived arousal, omnibus LRT across the five OCEAN traits, singularity check.

**`Stage6_h3_alignment.R` → `analysis/h3/`**
`h3_results.txt`, `h3_distance_by_quadrant.png`, `h3_offset_vectors.png` — alignment distance against the five traits with a joint LRT, plus descriptive quadrant distances, signed valence/arousal offsets, and an exploratory quadrant × OCEAN interaction test.

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

---

## 6. Run order after a stopping-rule or estimator change

```bash
# 1. archive (Section 8)
# 2. regenerate all 96 stimuli                                      ~20 min
time python -W ignore src/generator/run_generation.py --backend ollama

# 3. score — writes the H1 input                                    ~2 min
python src/analysis/score_estimator_b.py

# 3b. optional: architecture selection, then score that judge too
python src/estimators/select_estimator.py --pmemo datasets/PMEmo --name estimator_B2
python src/analysis/score_estimator_b.py --estimator estimator_B2

# 4. H1 for every judge scored                                      seconds
Rscript analysis/Stage6_h1.R
Rscript analysis/Stage6_h1.R estimator_B2

# 5. audience                                                       ~9 h
time python -W ignore src/audience/run_audience.py --backend ollama

# 6. remaining analyses                                             minutes
Rscript analysis/Stage6_h2.R
Rscript analysis/Stage6_h3_alignment.R
Rscript analysis/Stage6_mdu.R
Rscript analysis/Stage6_baselines.R
```

Step 3 is required: `run_audience.py` reads only `manifest.json` and will run without it, but `score_estimator_b.py` is what writes the H1 input and refreshes `integrity.json`.

Changing the judge does not require an audience rerun — Estimator B is used only in Stage 3, and the audience agents receive acoustic features, not estimator output. Changing Estimator A does require full regeneration and the 9-hour audience run.

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
```

`--briefs` picks specific IDs. `--limit N` takes the *first* N briefs, and briefs are ordered in quadrant blocks of four, so `--limit 2` only covers HV_HA.

---

## 8. Before you regenerate — archive first

`run_generation.py` rewrites `manifest.json` with only the briefs in that run and overwrites matching WAV files. A `--briefs` pilot replaces a 48-run manifest with a 12-run one.

```bash
git lfs pull                                   # archive real files, not stubs
mkdir -p archive/pre_quadrant_fix
cp -r data analysis logs models config archive/pre_quadrant_fix/
```

`models/` and `config/` are included so the estimator versions and threshold settings that produced the old results are preserved alongside them.

`logs/*.jsonl` append across runs rather than resetting, so the archive copy is how you separate old calls from new.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 118` loading a `.joblib` | File is a Git-LFS pointer (`118` = ASCII `v` of `version https://...`) | `git lfs pull` |
| Any file is exactly ~131–134 bytes | LFS content not fetched | `git lfs pull`, then `git lfs fetch --all` |
| `pickle` resolving from `~/anaconda3/` | conda `(base)` active alongside `.venv` | `conda deactivate`, re-activate `.venv` |
| `No soundfont found` | `.sf2` missing or an LFS stub | `git lfs pull`, or pass `--soundfont path/to.sf2` |
| Ollama connection refused | Daemon not running | `ollama serve`, check with `ollama ps` |
| R: `there is no package called 'smacof'` | Installed for the wrong user | `sudo Rscript -e 'install.packages("smacof", repos="https://cloud.r-project.org")'` |
| `No manifest at ...` from `select_estimator.py` | Generation has not run | Run Stage 2 first |
| Estimator loads but predictions look wrong | sklearn version drift | Compare `python -c "import sklearn; print(sklearn.__version__)"` against the `.meta.json` |

**Timing reference** (measured, Qwen3:8b local, ~2.9 s per LLM call):

| Stage | Calls | Time |
|---|---|---|
| Pilot (4 briefs) | ~98 | ~5 min |
| Full generation | ~416 | ~17 min |
| Estimator scoring | 0 | ~1 min |
| Architecture selection | 0 | ~10 min (cached features) |
| Audience | 9,792 | ~9 h |
| R analysis (all five) | 0 | minutes |

---

## 10. What gets recorded

- `models/*.meta.json` — corpus, model family, hyperparameters, held-out metrics, sklearn version, seed, freeze timestamp. Selected judges also record the selection rule, its hash, the families compared, and the domain-shift diagnostic.
- `logs/*.jsonl` — every LLM call with timestamp, model, temperature, full prompt, response and token counts.
- `data/stimuli/manifest.json` — per run: target, both conditions' parameters and estimator positions, distances, quadrant status, iteration count, and the full iteration history.
- `data/analysis/integrity.json` — sha256, peak, RMS and duration for every stimulus.
- `models/selection/` — the candidate comparison table and selection report.

Audio rendering is deterministic given the parameter set; LLM output is not. Audience agents receive only extracted acoustic features — not target coordinates, estimator outputs, brand descriptions, or the optimisation condition.