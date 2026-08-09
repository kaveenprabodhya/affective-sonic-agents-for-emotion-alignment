# analysis/check_estimator_b_range.R
h1 <- read.csv("data/analysis/h1_estimator_b.csv")
cat("Estimator B on study stimuli\n")
cat(sprintf("  valence SD %.4f  range %.3f\n", sd(h1$opt_B_v), diff(range(h1$opt_B_v))))
cat(sprintf("  arousal SD %.4f  range %.3f\n", sd(h1$opt_B_a), diff(range(h1$opt_B_a))))
cat("  compare against B's held-out RMSE: valence 0.236, arousal 0.247\n")