# Does optimisation change intended-perceived alignment distance
# for the OCEAN synthetic audience?
#
# Run:
#   Rscript analysis/Check_audience_optimisation_effect.R

suppressMessages({
  library(dplyr)
  library(lme4)
  library(lmerTest)
})

d <- read.csv("data/audience/responses.csv", stringsAsFactors = FALSE)

req <- c("agent_kind", "persona_id", "stimulus_file", "brief", "condition",
         "target_v", "target_a", "perceived_v", "perceived_a", "rep")
miss <- setdiff(req, names(d))
if (length(miss) > 0)
  stop(paste("Missing columns:", paste(miss, collapse = ", ")))

# ------------------------------------------------------------------
# OCEAN audience only, BOTH conditions
# ------------------------------------------------------------------

oc <- d %>%
  filter(agent_kind == "ocean",
         condition %in% c("non_optimised", "optimised")) %>%
  mutate(
    distance = sqrt(
      (perceived_v - target_v)^2 +
      (perceived_a - target_a)^2
    ),
    run = sub(".*_run([0-9]+)_.*", "\\1", stimulus_file),
    pair_id = interaction(brief, run, drop = TRUE),
    persona_id = factor(persona_id),
    pair_id = factor(pair_id),
    condition = factor(
      condition,
      levels = c("non_optimised", "optimised")
    )
  )

cat("============================================================\n")
cat("OCEAN AUDIENCE: OPTIMISED VS NON-OPTIMISED ALIGNMENT\n")
cat("============================================================\n\n")

cat("Rows:", nrow(oc), "\n")
cat("Personas:", nlevels(oc$persona_id), "\n")
cat("Matched brief/run pairs:", nlevels(oc$pair_id), "\n\n")

# ------------------------------------------------------------------
# Descriptive result
# ------------------------------------------------------------------

desc <- oc %>%
  group_by(condition) %>%
  summarise(
    n = n(),
    mean_distance = mean(distance),
    sd_distance = sd(distance),
    .groups = "drop"
  )

print(desc)

m_non <- desc$mean_distance[desc$condition == "non_optimised"]
m_opt <- desc$mean_distance[desc$condition == "optimised"]
change <- m_opt - m_non
pct <- 100 * change / m_non

cat(sprintf(
  "\nMean change (optimised - non-optimised): %.4f\n", change
))
cat(sprintf(
  "Relative change: %+.2f%%\n\n", pct
))

# ------------------------------------------------------------------
# Main mixed-effects test
#
# pair_id matches the optimised and non-optimised generation from
# the same brief/run.
#
# persona_id accounts for repeated ratings by the same persona.
#
# persona_id:pair_id accounts for the repeated responses from the
# same persona to the matched pair.
# ------------------------------------------------------------------

m0 <- lmer(
  distance ~ 1 +
    (1 | persona_id) +
    (1 | pair_id) +
    (1 | persona_id:pair_id),
  data = oc,
  REML = FALSE
)

m1 <- lmer(
  distance ~ condition +
    (1 | persona_id) +
    (1 | pair_id) +
    (1 | persona_id:pair_id),
  data = oc,
  REML = FALSE
)

cat("------------------------------------------------------------\n")
cat("MAIN MIXED-EFFECTS MODEL\n")
cat("distance ~ condition + persona + brief/run pairing\n")
cat("------------------------------------------------------------\n\n")

print(summary(m1)$coefficients)

cat("\nLikelihood-ratio test for optimisation condition:\n")
print(anova(m0, m1))

cat("\n95% confidence intervals:\n")
print(confint(m1, parm = "conditionoptimised", method = "Wald"))

cat("\nSingular fit:", isSingular(m1), "\n")

# ------------------------------------------------------------------
# Simple 48-pair sensitivity check
#
# Average repetitions and personas within each brief/run/condition.
# This gives one audience distance for each side of each of the
# 48 matched generation pairs.
# ------------------------------------------------------------------

pair_means <- oc %>%
  group_by(pair_id, condition) %>%
  summarise(
    distance = mean(distance),
    .groups = "drop"
  ) %>%
  tidyr::pivot_wider(
    names_from = condition,
    values_from = distance
  ) %>%
  filter(!is.na(non_optimised), !is.na(optimised))

cat("\n------------------------------------------------------------\n")
cat("48-PAIR SENSITIVITY CHECK\n")
cat("------------------------------------------------------------\n\n")

cat("Matched pairs:", nrow(pair_means), "\n")

cat("\nPaired t-test:\n")
print(t.test(
  pair_means$optimised,
  pair_means$non_optimised,
  paired = TRUE
))

cat("\nWilcoxon signed-rank test:\n")
print(wilcox.test(
  pair_means$optimised,
  pair_means$non_optimised,
  paired = TRUE,
  exact = FALSE
))

diff <- pair_means$optimised - pair_means$non_optimised

cat("\nPair direction:\n")
cat("Closer after optimisation:", sum(diff < 0), "\n")
cat("Farther after optimisation:", sum(diff > 0), "\n")
cat("No change:", sum(diff == 0), "\n")

cat("\n============================================================\n")
cat("INTERPRETATION\n")
cat("Negative condition coefficient = better audience alignment\n")
cat("Positive condition coefficient = worse audience alignment\n")
cat("p < .05 = evidence that the condition difference is not only descriptive\n")
cat("============================================================\n")