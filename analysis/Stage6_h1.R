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
cat("\nConclusion: report whether the INDEPENDENT judge confirms optimisation moved logos\n")
cat("toward target. If non-significant, H1 is the stimulus-validity LIMITATION: logos were\n")
cat("generated TOWARD targets, not proven to carry the intended VA.\n")
sink()
cat("Saved: analysis/tables/h1_results.txt\n")