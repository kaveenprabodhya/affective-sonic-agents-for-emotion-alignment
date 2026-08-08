# H1 - STIMULUS alignment: did optimisation move the logo toward its intended VA?
#   Intended VA  <->  Estimator B VA   (independent, held-out judge)
# This is stimulus validity, NOT audience alignment. Reads the Estimator-B scoring output.
#
#   Rscript analysis/stage6_h1.R
library(lme4); library(lmerTest)


# --- all outputs go to analysis/h1/ ---
OUT <- "analysis/h1"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

h1 <- read.csv("data/analysis/h1_estimator_b.csv", stringsAsFactors = FALSE)
h1$run   <- sub(".*_run([0-9]+)_.*", "\\1", paste0("x_run", h1$run, "_x"))  # run already a column
h1$brief <- factor(h1$brief)

sink(file.path(OUT, "h1_results.txt"), split = TRUE)
cat("H1 - stimulus alignment (Estimator B, held-out judge). Paired within brief/run.\n\n")
cat(sprintf("mean non-optimised B-distance: %.3f\n", mean(h1$nonopt_B_dist)))
cat(sprintf("mean optimised B-distance:     %.3f\n", mean(h1$opt_B_dist)))
cat(sprintf("mean B-reduction (non-opt - opt): %+.3f\n", mean(h1$B_reduction)))
cat(sprintf("pairs improved: %d / %d\n\n", sum(h1$B_reduction > 0), nrow(h1)))

# long form for a paired mixed model: distance ~ condition, brief/run + (pair) structure
long <- data.frame(
  brief = rep(h1$brief, 2), run = rep(h1$run, 2),
  condition = rep(c("non_optimised","optimised"), each = nrow(h1)),
  Bdist = c(h1$nonopt_B_dist, h1$opt_B_dist))
long$condition <- relevel(factor(long$condition), ref = "non_optimised")
m <- lmer(Bdist ~ condition + (1|brief/run), data = long)
cat("-- paired model: B-distance ~ condition + (1|brief/run) --\n"); print(summary(m)$coefficients)

# transparent primary: paired test on 48 brief/run differences
cat("\n-- paired t-test on optimised vs non-optimised B-distance (48 pairs) --\n")
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

cat("\n-- Wilcoxon signed-rank test (optimised vs non-optimised B-distance) --\n")
print(wilcox.test(h1$opt_B_dist, h1$nonopt_B_dist, paired = TRUE))

cat("\nConclusion: report whether the INDEPENDENT judge confirms optimisation moved logos\n")
cat("toward target. If non-significant, H1 is the stimulus-validity LIMITATION: logos were\n")
cat("generated TOWARD targets, not proven to carry the intended VA. Note that the mean-based\n")
cat("paired t-test can mask a directional pattern that the sign/Wilcoxon tests reveal; report\n")
cat("all three and interpret the direction, not only significance.\n")

# --- quadrant verification: the stopping rule now requires distance <= threshold
# AND sign agreement with the target on both axes (Section 3.4.2.2). Estimator A
# enforces that rule inside the loop; this checks it against the independent
# Estimator B, which never sees the optimisation. ---
cat("\n-- quadrant verification (Estimator B vs target sign, independent check) --\n")
sgn <- function(x) ifelse(x >= 0, 1, -1)
cross_opt    <- (sgn(h1$opt_B_v)    != sgn(h1$target_v)) | (sgn(h1$opt_B_a)    != sgn(h1$target_a))
cross_nonopt <- (sgn(h1$nonopt_B_v) != sgn(h1$target_v)) | (sgn(h1$nonopt_B_a) != sgn(h1$target_a))
cat(sprintf("optimised stimuli crossing target quadrant sign:     %d / %d (%.1f%%)\n",
            sum(cross_opt), nrow(h1), 100 * mean(cross_opt)))
cat(sprintf("non-optimised stimuli crossing target quadrant sign: %d / %d (%.1f%%)\n",
            sum(cross_nonopt), nrow(h1), 100 * mean(cross_nonopt)))
cat("Non-optimised stimuli are iteration 0 and are NOT subject to the stopping rule,\n")
cat("so crossings there are expected. Crossings among OPTIMISED stimuli indicate\n")
cat("Estimator A/B disagreement on quadrant, not a failure of the rule: the loop can\n")
cat("only enforce sign agreement under the estimator that guides it. Report this rate\n")
cat("as a limit on stimulus validity, alongside the pre-fix rate for comparison.\n")

sink()
cat("Saved: analysis/h1/h1_results.txt\n")