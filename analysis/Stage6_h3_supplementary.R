# H3 SUPPLEMENTARY - does alignment differ across personas once each persona's
# overall response level is removed?
#
#   The primary H3 model tests raw intended-perceived distance. That distance is
#   built from the same perceived coordinates H2 already models, so a persona
#   that rates everything higher lands further from targets sitting below the
#   perceived cloud. Raw distance therefore cannot separate:
#
#     (a) a persona shifting all its ratings up or down   -> already H2
#     (b) a persona ordering the stimuli differently      -> genuinely new
#
#   Two checks separate them.
#
#   TEST 1 - centred distance.
#     Subtract each persona's own mean valence and arousal before recomputing
#     distance. This removes (a). If traits still predict centred distance,
#     personas differ in accuracy, not only in level.
#
#   TEST 2 - rank agreement.
#     Spearman correlation between personas on which stimuli are best aligned.
#     A pure level shift leaves the ordering untouched, so low agreement here
#     is evidence of genuinely different perception.
#
#   Run:
#     Rscript analysis/Stage6_h3_supplementary.R

library(lme4)
library(lmerTest)
library(dplyr)
library(tidyr)

OUT <- "analysis/h3"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

d <- read.csv("data/audience/responses.csv", stringsAsFactors = FALSE)

traits <- c("openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism")

oc <- d %>%
  filter(agent_kind == "ocean", condition == "optimised") %>%
  mutate(
    dist = sqrt((perceived_v - target_v)^2 + (perceived_a - target_a)^2),
    run  = sub(".*_run([0-9]+)_.*", "\\1", stimulus_file)
  ) %>%
  mutate(
    grand_v = mean(perceived_v),
    grand_a = mean(perceived_a)
  ) %>%
  group_by(persona_id) %>%
  mutate(
    # ratings re-expressed relative to this persona's own average,
    # then returned to the shared scale via the grand mean
    cent_v = perceived_v - mean(perceived_v) + first(grand_v),
    cent_a = perceived_a - mean(perceived_a) + first(grand_a)
  ) %>%
  ungroup() %>%
  mutate(
    dist_centred = sqrt((cent_v - target_v)^2 + (cent_a - target_a)^2)
  )

for (trait in traits) {
  oc[[trait]] <- relevel(factor(oc[[trait]]), ref = "low")
}
oc$persona_id <- factor(oc$persona_id)
oc$brief      <- factor(oc$brief)
oc$run        <- factor(oc$run)

sink(file.path(OUT, "h3_supplementary.txt"), split = TRUE)

cat("========== TEST 1: DISTANCE AFTER REMOVING EACH PERSONA'S LEVEL ==========\n\n")
cat(sprintf("mean raw distance:      %.3f\n", mean(oc$dist)))
cat(sprintf("mean centred distance:  %.3f\n\n", mean(oc$dist_centred)))

f <- as.formula(paste("dist_centred ~", paste(traits, collapse = " + "),
                      "+ (1|persona_id) + (1|brief/run)"))
m_full <- lmer(f, data = oc, REML = FALSE)
m_null <- lmer(dist_centred ~ 1 + (1|persona_id) + (1|brief/run),
               data = oc, REML = FALSE)

print(summary(m_full)$coefficients)
cat("\n-- omnibus test of all five traits --\n")
print(anova(m_null, m_full))

cat("\nReading: if this omnibus test is still significant, personas differ in\n")
cat("alignment beyond a simple shift in how high or low they rate, and H3 adds\n")
cat("something H2 does not. If it is not significant, H3 is largely a\n")
cat("restatement of H2 and should be reported as such.\n")

cat("\n\n========== TEST 2: DO PERSONAS RANK THE SAME STIMULI AS BEST ALIGNED? ==========\n\n")

wide <- oc %>%
  group_by(persona_id, stimulus_file) %>%
  summarise(d = mean(dist), .groups = "drop") %>%
  pivot_wider(names_from = persona_id, values_from = d) %>%
  select(-stimulus_file)

rho <- cor(wide, method = "spearman", use = "pairwise.complete.obs")
off <- rho[upper.tri(rho)]

cat(sprintf("persona pairs compared: %d\n", length(off)))
cat(sprintf("mean pairwise Spearman rho: %.3f\n", mean(off)))
cat(sprintf("range: %.3f to %.3f\n\n", min(off), max(off)))

cat("Reading: rho near 1 means every persona ranks the same logos as best\n")
cat("aligned, so persona differences are level shifts only. Lower values mean\n")
cat("personas genuinely disagree about which logos work, which is a finding\n")
cat("independent of H2.\n")

sink()
cat(sprintf("\nSaved: %s/h3_supplementary.txt\n", OUT))