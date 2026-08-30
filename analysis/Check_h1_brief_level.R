#!/usr/bin/env Rscript

# ---------------------------------------------------------------------------
# H1 robustness diagnostic: brief-level sensitivity analysis
#
# Purpose:
#   The primary H1 analysis uses 48 paired brief/run observations.
#   This diagnostic averages the three runs within each of the 16 briefs,
#   so every brief contributes exactly one paired observation.
#
#   It asks:
#       Does the optimisation effect remain when the unit of analysis
#       is the brief rather than the individual run?
#
# This is a robustness diagnostic only. It does not replace Stage6_h1.R.
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

judge <- if (length(args) >= 1) args[1] else "estimator_B2"

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
    paste(missing_cols, collapse = ", ")
  )
}


# ---------------------------------------------------------------------------
# Aggregate the three runs within each brief
# ---------------------------------------------------------------------------

brief_level <- aggregate(
  cbind(
    nonopt_B_dist,
    opt_B_dist
  ) ~ brief,
  data = h1,
  FUN = mean
)

brief_level$reduction <- (
  brief_level$nonopt_B_dist -
  brief_level$opt_B_dist
)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

n_briefs <- nrow(brief_level)

mean_nonopt <- mean(
  brief_level$nonopt_B_dist
)

mean_opt <- mean(
  brief_level$opt_B_dist
)

mean_reduction <- mean(
  brief_level$reduction
)

relative_reduction <- (
  mean_reduction /
  mean_nonopt
) * 100

improved <- sum(
  brief_level$reduction > 0
)

worsened <- sum(
  brief_level$reduction < 0
)

ties <- sum(
  brief_level$reduction == 0
)


cat(
  "\nH1 BRIEF-LEVEL SENSITIVITY ANALYSIS\n"
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
  "\n\n",
  sep = ""
)

cat(
  "Unit of analysis: brief\n"
)

cat(
  "Briefs: ",
  n_briefs,
  "\n",
  sep = ""
)

cat(
  sprintf(
    "Mean non-optimised distance: %.4f\n",
    mean_nonopt
  )
)

cat(
  sprintf(
    "Mean optimised distance:     %.4f\n",
    mean_opt
  )
)

cat(
  sprintf(
    "Mean reduction:              %.4f\n",
    mean_reduction
  )
)

cat(
  sprintf(
    "Relative reduction:          %.1f%%\n\n",
    relative_reduction
  )
)

cat(
  "Briefs improved: ",
  improved,
  "/",
  n_briefs,
  "\n",
  sep = ""
)

cat(
  "Briefs worsened: ",
  worsened,
  "/",
  n_briefs,
  "\n",
  sep = ""
)

cat(
  "Briefs tied: ",
  ties,
  "/",
  n_briefs,
  "\n\n",
  sep = ""
)


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------

cat(
  "-- paired t-test on 16 brief means --\n\n"
)

tt <- t.test(
  brief_level$opt_B_dist,
  brief_level$nonopt_B_dist,
  paired = TRUE
)

print(tt)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank sensitivity test
# ---------------------------------------------------------------------------

cat(
  "\n-- Wilcoxon signed-rank test on brief means --\n\n"
)

wt <- suppressWarnings(
  wilcox.test(
    brief_level$opt_B_dist,
    brief_level$nonopt_B_dist,
    paired = TRUE,
    exact = FALSE
  )
)

print(wt)


# ---------------------------------------------------------------------------
# Sign test
# ---------------------------------------------------------------------------

nz <- brief_level$reduction[
  brief_level$reduction != 0
]

n_positive <- sum(
  nz > 0
)

cat(
  "\n-- exact sign test on non-tied briefs --\n\n"
)

cat(
  "Improved: ",
  n_positive,
  "   worsened: ",
  length(nz) - n_positive,
  "\n\n",
  sep = ""
)

if (length(nz) > 0) {

  sign_test <- binom.test(
    n_positive,
    length(nz),
    p = 0.5,
    alternative = "two.sided"
  )

  print(sign_test)

} else {

  cat(
    "All brief-level differences are zero.\n"
  )
}


# ---------------------------------------------------------------------------
# Per-brief table
# ---------------------------------------------------------------------------

brief_level <- brief_level[
  order(brief_level$brief),
]

cat(
  "\n-- per-brief means --\n\n"
)

print(
  brief_level,
  row.names = FALSE
)


# ---------------------------------------------------------------------------
# Save diagnostic table
# ---------------------------------------------------------------------------

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

out_csv <- file.path(
  out_dir,
  paste0(
    "brief_level_",
    judge,
    ".csv"
  )
)

write.csv(
  brief_level,
  out_csv,
  row.names = FALSE
)

cat(
  "\nSaved: ",
  out_csv,
  "\n",
  sep = ""
)