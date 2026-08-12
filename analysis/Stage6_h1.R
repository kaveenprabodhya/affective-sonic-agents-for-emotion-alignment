# H1 - STIMULUS alignment: did optimisation move the logo toward its intended VA?
#   Intended VA  <->  held-out judge VA   (estimator_B or estimator_B2)
# This is stimulus validity, NOT audience alignment. Reads the Estimator-B scoring output.
#
# The judge can be swapped so the pre-specified incumbent and any architecture-
# selected replacement are both testable. Both must be reported.
#
#   Rscript analysis/Stage6_h1.R                 # incumbent estimator_B
#   Rscript analysis/Stage6_h1.R estimator_B2    # architecture-selected judge
library(lme4); library(lmerTest)

args  <- commandArgs(trailingOnly = TRUE)
judge <- if (length(args) >= 1) args[1] else "estimator_B"
tag   <- if (judge == "estimator_B") "" else paste0("_", judge)
jl    <- sub("^estimator_", "", judge)          # short label: B, B2
jf    <- tolower(jl)                            # filename suffix: b, b2

IN  <- sprintf("data/analysis/h1_estimator_b%s.csv", tag)
OUT <- "analysis/h1"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(IN)) {
  stop(sprintf("%s not found. Run: python src/analysis/score_estimator_b.py --estimator %s",
               IN, judge))
}

h1 <- read.csv(IN, stringsAsFactors = FALSE)
h1$run   <- sub(".*_run([0-9]+)_.*", "\\1", paste0("x_run", h1$run, "_x"))  # run already a column
h1$brief <- factor(h1$brief)

RES <- file.path(OUT, sprintf("h1_results_%s.txt", jf))
sink(RES, split = TRUE)
cat(sprintf("H1 - stimulus alignment. Judge: %s (held out). Paired within brief/run.\n\n",
            judge))

# Discrimination gate. A judge whose predictions barely vary across the stimuli
# cannot detect movement of any size, so a null from it is uninformative about
# whether optimisation worked. Reported before the tests, not after.
sd_v <- sd(c(h1$nonopt_B_v, h1$opt_B_v)); sd_a <- sd(c(h1$nonopt_B_a, h1$opt_B_a))
cat(sprintf("judge spread across stimuli: valence SD %.4f, arousal SD %.4f\n",
            sd_v, sd_a))
cat(sprintf("judge valence range: %.3f   arousal range: %.3f\n",
            diff(range(c(h1$nonopt_B_v, h1$opt_B_v))),
            diff(range(c(h1$nonopt_B_a, h1$opt_B_a)))))
cat("Compare against this judge's own held-out RMSE in models/<judge>.meta.json.\n")
cat("If the SD is a small fraction of that RMSE, H1 is untestable with this judge\n")
cat("rather than rejected, and should be reported that way.\n\n")
cat(sprintf("mean non-optimised %s-distance: %.3f\n", jl, mean(h1$nonopt_B_dist)))
cat(sprintf("mean optimised %s-distance:     %.3f\n", jl, mean(h1$opt_B_dist)))
cat(sprintf("mean %s-reduction (non-opt - opt): %+.3f\n", jl, mean(h1$B_reduction)))
cat(sprintf("pairs improved: %d / %d\n\n", sum(h1$B_reduction > 0), nrow(h1)))

# long form for a paired mixed model: distance ~ condition, brief/run + (pair) structure
long <- data.frame(
  brief = rep(h1$brief, 2), run = rep(h1$run, 2),
  condition = rep(c("non_optimised","optimised"), each = nrow(h1)),
  Bdist = c(h1$nonopt_B_dist, h1$opt_B_dist))
long$condition <- relevel(factor(long$condition), ref = "non_optimised")
m <- lmer(Bdist ~ condition + (1|brief/run), data = long)
cat(sprintf("-- paired model: %s-distance ~ condition + (1|brief/run) --\n", jl)); print(summary(m)$coefficients)

# transparent primary: paired test on 48 brief/run differences
cat(sprintf("\n-- paired t-test on optimised vs non-optimised %s-distance (%d pairs) --\n", jl, nrow(h1)))
print(t.test(h1$opt_B_dist, h1$nonopt_B_dist, paired = TRUE))

# --- sensitivity analyses (Section 3.7.1): the paired differences include a high
# proportion of exact ties and a skewed distribution, so a sign test and a
# Wilcoxon signed-rank test are reported alongside the paired t-test. ---
diff   <- h1$opt_B_dist - h1$nonopt_B_dist   # positive = got worse, negative = improved
nz     <- diff[diff != 0]
n_tied <- sum(diff == 0)
n_up   <- sum(nz > 0)   # worsened
n_down <- sum(nz < 0)   # improved

cat(sprintf("\n-- tie check: %d / %d pairs show an exact zero difference --\n",
            n_tied, nrow(h1)))
cat("(structural candidate: the optimiser's first candidate may already have been\n")
cat(" its best Estimator-A-scored candidate, so optimised == non-optimised by\n")
cat(" construction; check history logs before treating these as noise.)\n\n")

cat(sprintf("-- sign test on the %d non-tied pairs (improved vs worsened) --\n", length(nz)))
cat(sprintf("improved: %d   worsened: %d\n", n_down, n_up))
print(binom.test(n_down, length(nz), p = 0.5))

cat(sprintf("\n-- Wilcoxon signed-rank test (optimised vs non-optimised %s-distance) --\n", jl))
print(wilcox.test(h1$opt_B_dist, h1$nonopt_B_dist, paired = TRUE))

cat("\nConclusion: report whether the INDEPENDENT judge confirms optimisation moved logos\n")
cat("toward target. If non-significant, H1 is the stimulus-validity LIMITATION: logos were\n")
cat("generated TOWARD targets, not proven to carry the intended VA. Note that the mean-based\n")
cat("paired t-test can mask a directional pattern that the sign/Wilcoxon tests reveal; report\n")
cat("all three and interpret the direction, not only significance.\n")

# --- quadrant verification: the stopping rule now requires distance <= threshold
# AND sign agreement with the target on both axes (Section 3.4.2.2). Estimator A
# enforces that rule inside the loop; this checks it against the independent
# the held-out judge, which never sees the optimisation. ---
cat(sprintf("\n-- quadrant verification (Estimator %s vs target sign, independent check) --\n", jl))
sgn <- function(x) ifelse(x >= 0, 1, -1)
cross_opt    <- (sgn(h1$opt_B_v)    != sgn(h1$target_v)) | (sgn(h1$opt_B_a)    != sgn(h1$target_a))
cross_nonopt <- (sgn(h1$nonopt_B_v) != sgn(h1$target_v)) | (sgn(h1$nonopt_B_a) != sgn(h1$target_a))
cat(sprintf("optimised stimuli crossing target quadrant sign:     %d / %d (%.1f%%)\n",
            sum(cross_opt), nrow(h1), 100 * mean(cross_opt)))
cat(sprintf("non-optimised stimuli crossing target quadrant sign: %d / %d (%.1f%%)\n",
            sum(cross_nonopt), nrow(h1), 100 * mean(cross_nonopt)))
cat("Non-optimised stimuli are iteration 0 and are NOT subject to the stopping rule,\n")
cat("so crossings there are expected. Crossings among OPTIMISED stimuli indicate\n")
cat(sprintf("Estimator A/%s disagreement on quadrant, not a failure of the rule: the loop can\n", jl))
cat("only enforce sign agreement under the estimator that guides it. Report this rate\n")
cat("as a limit on stimulus validity, alongside the pre-fix rate for comparison.\n")

sink()
cat(sprintf("Saved: %s\n", RES))