# H2 - do OCEAN personas differ in PERCEIVED valence/arousal? (not alignment - raw perception)
#   Fixed: OCEAN traits (+ condition, control). Random: persona + stimulus (crossed).
#
#   Rscript analysis/stage6_h2.R
library(lme4); library(lmerTest)


# --- all outputs go to analysis/h2/ ---
OUT <- "analysis/h2"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

d <- read.csv("data/audience/responses.csv", stringsAsFactors = TRUE)
traits <- c("openness","conscientiousness","extraversion","agreeableness","neuroticism")
oc <- droplevels(subset(d, agent_kind == "ocean"))
for (t in traits) oc[[t]] <- relevel(factor(oc[[t]]), ref = "low")
oc$condition <- relevel(factor(oc$condition), ref = "non_optimised")

sink(file.path(OUT, "h2_results.txt"), split = TRUE)
run_axis <- function(ax) {
  cat("\n==========", ax, "~ OCEAN + condition | (1|persona)+(1|stimulus) ==========\n")
  f  <- as.formula(paste0(ax, " ~ openness+conscientiousness+extraversion+agreeableness+neuroticism+condition+(1|persona_id)+(1|stimulus_file)"))
  m  <- lmer(f, data = oc)
  print(summary(m)$coefficients)
  f0 <- as.formula(paste0(ax, " ~ condition+(1|persona_id)+(1|stimulus_file)"))
  cat("\n-- LRT: OCEAN traits jointly (omnibus H2) --\n")
  print(anova(lmer(f0, data=oc, REML=FALSE), update(m, REML=FALSE)))
  cat("singular:", isSingular(m), "\n")
  m
}
mv <- run_axis("perceived_v")
ma <- run_axis("perceived_a")
# Holm across the two omnibus tests
cat("\nHolm-adjust the two omnibus p-values (valence, arousal) in the write-up.\n")
sink()
cat("Saved: analysis/tables/h2_results.txt\n")