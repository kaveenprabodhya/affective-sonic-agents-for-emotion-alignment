#!/usr/bin/env Rscript

# ---------------------------------------------------------------------------
# H1 robustness diagnostic: cluster bootstrap + exact sign-flip test
#
# Purpose:
#
# 1. Cluster bootstrap
#    Resample whole briefs with replacement, keeping their runs together.
#    This estimates uncertainty in the mean B2 optimisation effect without
#    pretending that the three runs of a brief are unrelated observations.
#
# 2. Exact sign-flip permutation test
#    Collapse runs to one mean reduction per brief, then evaluate every
#    possible +/- sign assignment across the 16 briefs.
#
#    With 16 briefs:
#       2^16 = 65,536
#
#    Therefore no Monte Carlo approximation is required.
#
# This is a robustness diagnostic only. It does not replace Stage6_h1.R.
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

judge <- if (length(args) >= 1) args[1] else "estimator_B2"

n_boot <- if (length(args) >= 2) {
  as.integer(args[2])
} else {
  10000L
}

seed <- if (length(args) >= 3) {
  as.integer(args[3])
} else {
  20260829L
}


if (judge == "estimator_B") {

  input_file <- "data/analysis/h1_estimator_b.csv"

} else {

  input_file <- paste0(
    "data/analysis/h1_estimator_b_",
    judge,
    ".csv"
  )
}


if (!file.exists(input_file)) {

  stop(
    "Input file not found: ",
    input_file,
    "\nRun the estimator scoring stage first."
  )
}


h1 <- read.csv(
  input_file,
  stringsAsFactors = FALSE
)


required <- c(
  "brief",
  "nonopt_B_dist",
  "opt_B_dist"
)


missing_cols <- setdiff(
  required,
  names(h1)
)


if (length(missing_cols) > 0) {

  stop(
    "Missing required columns: ",
    paste(
      missing_cols,
      collapse = ", "
    )
  )
}


h1$reduction <- (
  h1$nonopt_B_dist -
  h1$opt_B_dist
)


briefs <- unique(
  h1$brief
)

briefs <- sort(
  briefs
)

n_briefs <- length(
  briefs
)


# ---------------------------------------------------------------------------
# Brief-level effects
# ---------------------------------------------------------------------------

brief_effect <- aggregate(
  reduction ~ brief,
  data = h1,
  FUN = mean
)

brief_effect <- brief_effect[
  order(
    brief_effect$brief
  ),
]


observed <- mean(
  brief_effect$reduction
)


cat(
  "\nH1 RESAMPLING ROBUSTNESS ANALYSIS\n"
)

cat(
  paste0(
    strrep("=", 72),
    "\n"
  )
)

cat(
  "Judge: ",
  judge,
  "\n",
  sep = ""
)

cat(
  "Input: ",
  input_file,
  "\n",
  sep = ""
)

cat(
  "Brief clusters: ",
  n_briefs,
  "\n",
  sep = ""
)

cat(
  sprintf(
    "Observed mean reduction: %.6f\n\n",
    observed
  )
)


# ===========================================================================
# CLUSTER BOOTSTRAP
# ===========================================================================

cat(
  "-- cluster bootstrap by brief --\n\n"
)

cat(
  "Bootstrap samples: ",
  n_boot,
  "\n",
  sep = ""
)

cat(
  "Seed: ",
  seed,
  "\n",
  sep = ""
)


set.seed(
  seed
)


boot_effects <- numeric(
  n_boot
)


# Resample the 16 complete brief clusters.
#
# Because all briefs currently contain the same number of runs, the mean of
# the sampled brief effects equals the mean effect across their sampled rows.
# Operating on brief means makes that clustering explicit and transparent.

for (b in seq_len(n_boot)) {

  sampled <- sample(
    brief_effect$reduction,
    size = n_briefs,
    replace = TRUE
  )

  boot_effects[b] <- mean(
    sampled
  )
}


boot_ci <- quantile(
  boot_effects,
  probs = c(
    0.025,
    0.975
  ),
  names = FALSE,
  type = 7
)


boot_se <- sd(
  boot_effects
)


cat(
  sprintf(
    "Bootstrap mean:       %.6f\n",
    mean(boot_effects)
  )
)

cat(
  sprintf(
    "Bootstrap SE:         %.6f\n",
    boot_se
  )
)

cat(
  sprintf(
    "Bootstrap 95%% CI:     [%.6f, %.6f]\n",
    boot_ci[1],
    boot_ci[2]
  )
)

cat(
  sprintf(
    "Bootstrap P(effect>0): %.4f\n",
    mean(
      boot_effects > 0
    )
  )
)


# ===========================================================================
# EXACT SIGN-FLIP PERMUTATION TEST
# ===========================================================================

cat(
  "\n-- exact brief-level sign-flip permutation test --\n\n"
)


effects <- brief_effect$reduction

n_patterns <- 2^n_briefs


cat(
  "Exact sign configurations: ",
  format(
    n_patterns,
    big.mark = ","
  ),
  "\n",
  sep = ""
)


# Generate every binary pattern from 0 to 2^n - 1.
#
# Bit 0 -> -1
# Bit 1 -> +1

null_means <- numeric(
  n_patterns
)


for (i in 0:(n_patterns - 1)) {

  signs <- ifelse(
    bitwAnd(
      i,
      bitwShiftL(
        1L,
        0:(n_briefs - 1)
      )
    ) != 0,
    1,
    -1
  )

  null_means[i + 1] <- mean(
    effects * signs
  )
}


# Two-sided exact p-value:
# proportion of sign configurations with an absolute mean effect at least
# as large as the observed absolute effect.

perm_p_two <- mean(
  abs(null_means) >=
    abs(observed) - 1e-12
)


# One-sided version is also reported for diagnostic completeness.
# H1 predicts reduction > 0.

perm_p_one <- mean(
  null_means >=
    observed - 1e-12
)


cat(
  sprintf(
    "Observed mean reduction: %.6f\n",
    observed
  )
)

cat(
  sprintf(
    "Exact two-sided p:       %.6f\n",
    perm_p_two
  )
)

cat(
  sprintf(
    "Exact one-sided p:       %.6f\n",
    perm_p_one
  )
)


# ===========================================================================
# SAVE OUTPUTS
# ===========================================================================

out_dir <- file.path(
  "analysis",
  "diagnostics",
  "h1_robustness"
)

dir.create(
  out_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


boot_file <- file.path(
  out_dir,
  paste0(
    "bootstrap_",
    judge,
    ".csv"
  )
)


write.csv(
  data.frame(
    bootstrap_id = seq_len(n_boot),
    mean_reduction = boot_effects
  ),
  boot_file,
  row.names = FALSE
)


brief_file <- file.path(
  out_dir,
  paste0(
    "brief_effects_",
    judge,
    ".csv"
  )
)


write.csv(
  brief_effect,
  brief_file,
  row.names = FALSE
)


summary_file <- file.path(
  out_dir,
  paste0(
    "resampling_summary_",
    judge,
    ".txt"
  )
)


sink(
  summary_file
)

cat(
  "H1 RESAMPLING ROBUSTNESS SUMMARY\n"
)

cat(
  "Judge: ",
  judge,
  "\n\n",
  sep = ""
)

cat(
  sprintf(
    "Observed mean reduction: %.6f\n",
    observed
  )
)

cat(
  sprintf(
    "Cluster-bootstrap SE:    %.6f\n",
    boot_se
  )
)

cat(
  sprintf(
    "Bootstrap 95%% CI:        [%.6f, %.6f]\n",
    boot_ci[1],
    boot_ci[2]
  )
)

cat(
  sprintf(
    "Exact sign-flip p (2-sided): %.6f\n",
    perm_p_two
  )
)

cat(
  sprintf(
    "Exact sign-flip p (1-sided): %.6f\n",
    perm_p_one
  )
)

sink()


cat(
  "\nSaved:\n"
)

cat(
  "  ",
  brief_file,
  "\n",
  sep = ""
)

cat(
  "  ",
  boot_file,
  "\n",
  sep = ""
)

cat(
  "  ",
  summary_file,
  "\n",
  sep = ""
)