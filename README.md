# Affective Sonic Agents for Emotion Alignment

**Using a Multi-Agent Approach to Measure the Alignment between Brand-Intended and Audience-Perceived Emotional Positions in Synthetic Sonic Logos for Radio Advertising**

MSc dissertation project — WMG, University of Warwick.

A parametric synthesiser generates short sonic logos toward target valence–arousal (VA) coordinates. Two independently frozen estimators judge them. A synthetic audience of 32 OCEAN personas rates them. The framework measures the gap between what the brand *intended* and what the audience *perceived*.

---

## 1. What runs where

| Layer | Language | Location |
|---|---|---|
| Generation, optimisation, audience harness | Python 3.13 | `src/` |
| Statistical analysis (H1–H3, MDU, baselines) | R | `analysis/*.R` |
| LLM (all agents) | Qwen3:8b via Ollama, local | — |
| Audio rendering | FluidSynth + GeneralUser-GS SoundFont | `assets/soundfonts/` |

The **same** LLM is used for every agent and every stage. This is deliberate: varying the model would answer a different research question.

---

## 2. Hypotheses and what tests them

**H1** — Optimised sonic logos sit closer to their intended valence–arousal target than their matched non-optimised counterparts, as judged by the held-out estimator.

**H2** — Systematically varied OCEAN profiles produce distinguishable perceived valence and arousal ratings for the same stimulus.

**H3** — The distance between intended and perceived valence–arousal positions differs across OCEAN-based synthetic audience personas.

| Hypothesis | What is compared | Test | Script | Input |
|---|---|---|---|---|
| **H1** | Intended VA target vs **Estimator B** estimate, optimised vs non-optimised | Paired t-test on 48 matched pairs, plus mixed-effects model; sign test and Wilcoxon as sensitivity analyses | `Stage6_h1.R` | `data/analysis/h1_estimator_b.csv` |
| **H2** | Perceived valence and arousal across systematically varied OCEAN profiles, same stimuli | Separate linear mixed-effects models with omnibus LRT; Holm adjustment across the two outcomes | `Stage6_h2.R` | `data/audience/responses.csv` |
| **H3** | Euclidean distance between intended and **persona-perceived** VA, by OCEAN trait | Linear mixed-effects, five traits tested jointly by likelihood-ratio test | `Stage6_h3_alignment.R` | `data/audience/responses.csv` |

H1 tests RO2, H2 tests RO3 and H3 tests RO5. RO4 is a design objective evidenced by the artefact itself and carries no hypothesis.

**H1 is stimulus-level.** It asks whether optimisation moved the artefact toward its target, judged by the held-out estimator. It never touches the audience. This is why it can be tested in minutes rather than hours.

**H2 and H3 are audience-level** and both require the full 9,792-response run.

**H1 gates the interpretation of H3.** If the independent judge does not confirm that optimised stimuli carry their intended VA, then any intended-versus-perceived distance measured later is a distance from a target the stimulus may not actually express. H3 remains testable, but its interpretation is conditional.

**Supporting analyses — not hypothesis tests.** Both are descriptive or exploratory and neither can confirm or reject H1–H3.

| Script | Question it answers | Output folder |
|---|---|---|
| `Stage6_mdu.R` | Do personas and emotion terms form an interpretable joint structure? Exploratory support for H2 | `analysis/mdu/` |
| `Stage6_baselines.R` | Do the 32 OCEAN personas vary *more* than the gap between the two control agents? | `analysis/baselines/` |

`Stage6_baselines.R` exists to answer a specific challenge: if OCEAN personas differ from one another no more than a neutral agent differs from a generic-listener agent, then apparent persona differentiation is role-framing artefact rather than trait effect. The controls are **never** entered into the H2 or H3 mixed models — they are compared descriptively against the OCEAN distribution.

`Stage6_mdu.R` is exploratory because the ten-term reduced emotion set cannot reproduce Russell's full 28-term circumplex structure, so the configuration is diagnostic rather than confirmatory.

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

> **Do not run with conda `(base)` active.** If your prompt shows `(.venv) (base)`, run `conda deactivate` first. Mixed environments cause `pickle`/`sklearn` to resolve from the wrong interpreter and can break estimator loading in ways that look like file corruption.

Verify you are on the version the estimators were frozen under (`models/estimator_A.meta.json` records it):

```bash
python -c "import sklearn; print(sklearn.__version__)"   # expect 1.9.0
```

### 3.2 Git LFS — do this before anything else

Large files (`*.joblib`, `*.wav`, `*.mp3`, `*.jsonl`, `*.pkl`, `*.sf2`) are stored in Git LFS. A fresh clone or a GitHub "Download ZIP" gives you **131-byte pointer stubs**, not the real files.

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
│   ├── experiment.yaml              # stage sizes, optimisation rule, model settings
│   ├── briefs.yaml                  # 16 brand briefs, targets RESCALED into reachable region  <- used by the pipeline
│   ├── briefs_full_range.yaml       # the ORIGINAL full-range targets, kept unaltered for audit
│   ├── personas.yaml                # 32 OCEAN profiles + 2 controls
│   ├── questionnaire.yaml           # the Q1–Q12 survey instrument
│   └── prompts/
│       ├── generator_initial.txt    # first parameter proposal from brand description
│       ├── generator_revision.txt   # revision prompt (receives the signed VA gap)
│       ├── audience_system.txt      # OCEAN persona system prompt
│       ├── audience_system_generic.txt   # generic-listener control
│       ├── audience_system_neutral.txt   # neutral control (no listener framing)
│       └── audience_user.txt        # shared user message: feature block + survey
│
├── src/
│   ├── config_loader.py             # finds project root, loads config/*.yaml
│   ├── features/extracts.py         # 70-feature acoustic extraction (shared by A, B, audience)
│   ├── estimators/
│   │   ├── data.py                  # corpus loading, feature caching
│   │   ├── model.py                 # fit / freeze / load
│   │   └── build.py                 # ENTRY: builds and freezes Estimator A and B
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
│   ├── Stage6_h1.R      -> analysis/h1/         stimulus alignment (Estimator B)
│   ├── Stage6_h2.R      -> analysis/h2/         persona differentiation
│   ├── Stage6_h3_alignment.R -> analysis/h3/    audience alignment ~ OCEAN traits
│   ├── Stage6_mdu.R     -> analysis/mdu/        exploratory unfolding
│   │                                              ├── mdu_fit_summary.csv   (Stress-1, all solutions)
│   │                                              ├── pooled/               (all stimuli)
│   │                                              └── quadrants/            (4x, optimised only)
│   └── Stage6_baselines.R -> analysis/baselines/ neutral + generic control comparison
│                                                  (descriptive; controls never enter H2/H3 models)
│
├── data/
│   ├── stimuli/                     # 96 WAV files + manifest.json (generation record)
│   ├── stimuli_mp3/                 # MP3 copies + index.html for listening
│   ├── analysis/
│   │   ├── h1_estimator_b.csv       # 48 matched pairs, Estimator B distances  -> input to H1
│   │   └── integrity.json           # per-stimulus sha256, peak, RMS, duration
│   └── audience/responses.csv       # 9,792 rows -> input to H2, H3, MDU, baselines
│
├── models/
│   ├── estimator_A.joblib(.meta.json)   # DEAM, random forest — GUIDES optimisation
│   ├── estimator_B.joblib(.meta.json)   # PMEmo, SVR — HELD OUT, judges H1
│   ├── reachable_va.json                # measured reachable region
│   └── cache/                           # extracted-feature caches (safe to delete, slow to rebuild)
│
├── logs/                            # every LLM call, with ts / tokens / prompt / response
│   ├── generation.jsonl
│   ├── audience.jsonl
│   └── pilot.jsonl
│
└── spike/                           # exploratory work, NOT part of the pipeline
    ├── estimator_transfer_test.py   # cross-corpus transfer check
    ├── estimator_family_comparison.py # model-family comparison
    └── persona_pilot.py             # persona-differentiation gate (Stage 2 go/no-go)
```

**Critical distinction:** `briefs.yaml` holds the *rescaled* targets actually used. `briefs_full_range.yaml` holds the *original* targets and is never overwritten — it exists so the rescaling is auditable.

**Estimator A vs B:** A is the coach (drives optimisation, sees everything). B is the judge (frozen, never touched during generation, used only to score H1). Never swap them.

---

## 5. The pipeline

Six stages. Later stages depend on earlier outputs, so run in order.

### Stage 0 — Build and freeze the estimators

Only needed once, or if you change the feature set.

```bash
python src/estimators/build.py --deam datasets/DEAM --pmemo datasets/PMEmo
```

Writes `models/estimator_{A,B}.joblib` and their `.meta.json` (R², RMSE, freeze timestamp, seed).

```bash
# quick build for testing (caps songs per corpus)
python src/estimators/build.py --deam datasets/DEAM --pmemo datasets/PMEmo --songs 150
# rebuild just one
python src/estimators/build.py --pmemo datasets/PMEmo --only B --family-b svr
```

> If a cache goes stale: `rm -f models/cache/pmemo_*` then rebuild.

### Stage 1 — Map the reachable region and place the brief targets

```bash
python src/generator/probe_reachable.py          # -> models/reachable_va.json
python src/generator/generate_briefs.py          # -> rewrites config/briefs.yaml
```

`probe_reachable.py` scores a fixed sample of parameter combinations with Estimator A to find what the synth can actually reach. `generate_briefs.py` rescales the full-range targets into that region, per axis and per sign, so quadrant labels still denote the sign of the intended position.

### Stage 2 — Generate the stimuli

```bash
time python -W ignore src/generator/run_generation.py --backend ollama
```

16 briefs × 3 runs = 48 matched pairs = 96 stimuli, into `data/stimuli/` with `manifest.json`.

**Stopping rule** (`src/generator/loop.py`): the loop stops when the Estimator A prediction is *both* within `threshold` of the target *and* on the same side of both axes as the target — or when `iteration_cap` is hit. Distance alone is a proximity tolerance and does not protect quadrant membership, which matters because several targets sit close to the origin relative to Estimator A's prediction error.

Output flags: `*` = both criteria met, `Q` = quadrant held.

```
 *Q B09 run0: non-opt d=0.124 -> opt d=0.124  (1 iters)
  x B01 run0: non-opt d=0.316 -> opt d=0.138  (10 iters)
```

Config knobs (`config/experiment.yaml` → `generation.optimisation`):

| Key | Meaning |
|---|---|
| `threshold` | proximity tolerance on Estimator A distance |
| `require_quadrant` | enforce sign agreement; `false` reproduces distance-only behaviour |
| `iteration_cap` | max iterations per run |

### Stage 3 — Score with the held-out Estimator B

```bash
python src/analysis/score_estimator_b.py
```

Writes `data/analysis/h1_estimator_b.csv` (the H1 input) and `data/analysis/integrity.json` (hashes + audio sanity checks). Read-only on the stimuli.

### Stage 4 — Run the synthetic audience

```bash
time python -W ignore src/audience/run_audience.py --backend ollama
```

34 agents (32 OCEAN + neutral + generic) × 96 stimuli × 3 repetitions = **9,792 responses** → `data/audience/responses.csv`.

**This is the long one — budget ~9 hours.** It is interruptible:

```bash
python src/audience/run_audience.py --backend ollama --resume    # skips rows already written
```

### Stage 5 — Convert stimuli for listening (optional)

```bash
sh ./convert_2_mp3.sh
python data/stimuli_mp3/create_index.py
xdg-open data/stimuli_mp3/index.html
```

### Stage 6 — Statistical analysis

```bash
Rscript analysis/Stage6_h1.R              # H1: did optimisation move stimuli toward target?
Rscript analysis/Stage6_h2.R              # H2: do OCEAN profiles perceive differently?
Rscript analysis/Stage6_h3_alignment.R    # H3: does alignment distance vary by OCEAN trait?
Rscript analysis/Stage6_mdu.R             # exploratory unfolding
Rscript analysis/Stage6_baselines.R       # neutral + generic control comparison
```

Each creates its own output folder and writes a `*_results.txt` plus CSVs/PNGs.

#### What each produces

**`Stage6_h1.R` → `analysis/h1/`**
`h1_results.txt` — descriptives, mixed-effects model, paired t-test, plus sign test and Wilcoxon signed-rank as sensitivity analyses (the paired differences are bounded below and frequently tied, so the mean-based test alone can hide a directional pattern). Also reports a quadrant verification: how often optimised and non-optimised stimuli sit on the target's side of both axes according to Estimator B. Crossings among *optimised* stimuli indicate Estimator A/B disagreement, not a failure of the stopping rule — the loop can only enforce the sign condition under the estimator that guides it.

**`Stage6_h2.R` → `analysis/h2/`**
`h2_results.txt` — separate mixed-effects models for perceived valence and perceived arousal, with an omnibus likelihood-ratio test across the five OCEAN traits and Holm adjustment over the two outcomes.

**`Stage6_h3_alignment.R` → `analysis/h3/`**
`h3_results.txt`, plus `h3_distance_by_quadrant.png` and `h3_offset_vectors.png`. Alignment distance is modelled against the five OCEAN traits, tested jointly by LRT. Quadrant-level distances and signed valence/arousal offsets are reported **descriptively only** — four briefs per quadrant cannot power a between-quadrant inferential test.

**`Stage6_mdu.R` → `analysis/mdu/`**
Interval unfolding via `smacof`, run five times: one pooled solution across all stimuli, and four quadrant-specific solutions using optimised logos only.

```
analysis/mdu/
├── mdu_fit_summary.csv           # Stress-1 for every solution — read this first
├── pooled/
│   ├── mdu_persona_emotion.png   # personas + emotion terms together
│   ├── mdu_emotions_only.png     # emotion terms alone (cleaner for the write-up)
│   ├── mdu_persona_coords.csv
│   ├── mdu_emotion_coords.csv
│   └── persona_emotion_matrix.csv
└── quadrants/{HV_HA,HV_LA,LV_HA,LV_LA}/   # same five files each
```

The script warns on Stress-1 below 0.01, which can indicate a **degenerate solution** — a configuration that fits almost perfectly because it has collapsed, not because it has found structure. Check `mdu_fit_summary.csv` before interpreting any plot.

**`Stage6_baselines.R` → `analysis/baselines/`**
`baseline_results.txt` plus per-condition and per-stimulus CSVs, distribution plots, and mean VA position plots. Aggregation runs in three steps: average the 3 repetitions per agent–stimulus, average the 32 OCEAN personas to a stimulus-level mean, then compare that mean against each control. What you are looking for is whether the OCEAN spread exceeds the neutral-versus-generic gap.

**H1 needs only Stage 3** — it is stimulus-level and never touches the audience. If you regenerate stimuli, you can check H1 in ~25 minutes without waiting on the 9-hour audience rerun.

---

## 6. Recommended run order after a stopping-rule or estimator change

Do **not** chain generation straight into the audience run. Check H1 first — it costs about 20 minutes and tells you whether the 9-hour audience run is worth starting.

```bash
# 1. archive (see Section 8) — the pilot overwrote manifest.json with only 12 runs
# 2. regenerate all 96 stimuli                                      ~20 min
time python -W ignore src/generator/run_generation.py --backend ollama

# 3. score with the held-out judge — REQUIRED, writes the H1 input   ~2 min
python src/analysis/score_estimator_b.py

# 4. test H1                                                        seconds
Rscript analysis/Stage6_h1.R

# --- STOP AND READ analysis/h1/h1_results.txt BEFORE CONTINUING ---

# 5. only then, the audience run                                    ~9 h
time python -W ignore src/audience/run_audience.py --backend ollama

# 6. the remaining analyses                                          minutes
Rscript analysis/Stage6_h2.R
Rscript analysis/Stage6_h3_alignment.R
Rscript analysis/Stage6_mdu.R
Rscript analysis/Stage6_baselines.R
```

**Why step 3 is not optional.** `run_audience.py` reads only `data/stimuli/manifest.json`, so it will happily run without it — but `score_estimator_b.py` is what writes `data/analysis/h1_estimator_b.csv`. Skip it and H1 cannot be tested at all, and `integrity.json` will still describe the *old* stimuli.

**Why regenerate at all.** The optimisation stopping rule was corrected: it previously accepted a candidate on distance alone, which allowed an accepted stimulus to sit in a different VA quadrant from the one its brief specified. All stimuli generated under the old rule were therefore produced by a rule that did not implement what quadrant membership was defined to mean. The fix requires regeneration; every downstream result derived from the old stimuli is superseded.

---

## 7. Fast dry runs

Use these before committing to a full run.

```bash
# Generation, one brief per quadrant (12 runs, ~5 min)
python -W ignore src/generator/run_generation.py --backend ollama --briefs B01,B05,B09,B13

# Generation, first N briefs
python -W ignore src/generator/run_generation.py --backend ollama --limit 2

# No LLM at all — checks wiring only
python src/generator/run_generation.py --backend mock --limit 2

# Audience, tiny slice
python src/audience/run_audience.py --backend ollama --limit-agents 8 --limit-stimuli 40

# Persona differentiation gate
python spike/persona_pilot.py --backend ollama --model qwen3:8b
```

> `--briefs` picks specific IDs and is the right choice for quadrant coverage — `--limit N` takes the *first* N briefs, and briefs are ordered in quadrant blocks of four, so `--limit 2` only ever tests HV_HA.

---

## 8. Before you regenerate — archive first

`run_generation.py` rewrites `manifest.json` from scratch with **only the briefs in that run**, and overwrites matching WAV files. A `--briefs` pilot will replace a 48-run manifest with a 12-run one.

```bash
git lfs pull                                   # make sure you are archiving real files, not stubs
mkdir -p archive/pre_quadrant_fix
cp -r data analysis logs models config archive/pre_quadrant_fix/
```

`models/` and `config/` are included deliberately — without the estimator versions and the threshold settings, old results are not interpretable later.

Note that `logs/*.jsonl` **append** across runs rather than resetting, so the archive copy is how you tell old calls from new.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 118` loading a `.joblib` | File is a Git-LFS pointer (`118` = ASCII `v` of `version https://...`) | `git lfs pull` |
| Any file is exactly ~131–134 bytes | Same — LFS content not fetched | `git lfs pull`, then `git lfs fetch --all` |
| `pickle` resolving from `~/anaconda3/` | conda `(base)` active alongside `.venv` | `conda deactivate`, re-activate `.venv` |
| `No soundfont found` | `.sf2` missing or an LFS stub | `git lfs pull`, or pass `--soundfont path/to.sf2` |
| Ollama connection refused | Daemon not running | `ollama serve`, check with `ollama ps` |
| R: `there is no package called 'smacof'` | Installed for the wrong user | `sudo Rscript -e 'install.packages("smacof", repos="https://cloud.r-project.org")'` |
| Estimator loads but predictions look wrong | sklearn version drift | Compare `python -c "import sklearn; print(sklearn.__version__)"` against `models/estimator_A.meta.json` |

**Timing reference** (measured, Qwen3:8b local, ~2.9 s/LLM call):

| Stage | Calls | Time |
|---|---|---|
| Pilot (4 briefs) | ~98 | ~5 min |
| Full generation | ~416 | ~17 min |
| Estimator B scoring | 0 | ~1 min |
| Audience | 9,792 | **~9 h** |
| R analysis (all five) | 0 | minutes |

---

## 10. Reproducibility notes

- Both estimators are **frozen before generation begins**; `*.meta.json` records corpus, model family, metrics, sklearn version, seed and freeze timestamp.
- Every LLM call is logged to `logs/*.jsonl` with timestamp, model, temperature, full prompt, response and token counts.
- `data/analysis/integrity.json` holds a sha256 and audio sanity check for every stimulus.
- Audio rendering is deterministic given the parameter set; **LLM output is not**. Exact replication is not claimed — stability is assessed by comparing repeated runs.
- Audience agents receive only extracted acoustic features. They never see the target coordinates, the estimator outputs, the brand description, or the optimisation condition.