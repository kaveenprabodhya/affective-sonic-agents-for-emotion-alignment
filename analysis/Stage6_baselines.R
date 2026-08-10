# Stage 6 - descriptive baseline analysis (neutral and generic controls).
#
# Answers one question (methodology 3.6.2): do the 32 OCEAN personas vary MORE
# than the gap between the neutral and generic controls? If so, the between-
# persona variation is trait-driven rather than an artefact of role framing.
#
# Descriptive only - does NOT test H1-H3, and the controls are NOT entered into
# the H2/H3 mixed models.
#
# Aggregation:
#   1. average the 3 repetitions per agent-stimulus,
#   2. average the 32 OCEAN personas to a stimulus-level OCEAN mean,
#   3. compare that stimulus-level OCEAN mean with the two controls.
#
# Run:  Rscript analysis/Stage6_baselines.R
# Packages: install.packages(c("dplyr","tidyr","ggplot2"))

library(dplyr); library(tidyr); library(ggplot2)

OUT <- "analysis/baselines"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# -------------------------------------------------------------------------
# Load and label
# -------------------------------------------------------------------------
d <- read.csv("data/audience/responses.csv", stringsAsFactors = FALSE)

req <- c("agent_kind","stimulus_file","condition","perceived_v","perceived_a","target_v","target_a")
miss <- setdiff(req, names(d))
if (length(miss) > 0) stop(paste("Missing required columns:", paste(miss, collapse = ", ")))
if (!"persona_id" %in% names(d)) d$persona_id <- NA_character_

base <- d %>%
  mutate(
    respondent_id = ifelse(is.na(persona_id) | persona_id == "", agent_kind, persona_id),
    audience_group = case_when(
      tolower(agent_kind) == "ocean"          ~ "OCEAN audience",
      grepl("neutral", tolower(agent_kind))   ~ "Neutral control",
      grepl("generic", tolower(agent_kind))   ~ "Generic-listener control",
      TRUE ~ NA_character_),
    alignment_distance = sqrt((perceived_v - target_v)^2 + (perceived_a - target_a)^2)) %>%
  filter(!is.na(audience_group))
if (nrow(base) == 0) stop("No OCEAN / neutral / generic responses found.")

# -------------------------------------------------------------------------
# 1. average repetitions per respondent-stimulus
# -------------------------------------------------------------------------
agent_stim <- base %>%
  group_by(audience_group, respondent_id, stimulus_file, condition, target_v, target_a) %>%
  summarise(perceived_v = mean(perceived_v, na.rm = TRUE),
            perceived_a = mean(perceived_a, na.rm = TRUE),
            alignment_distance = mean(alignment_distance, na.rm = TRUE),
            .groups = "drop")

# -------------------------------------------------------------------------
# 2. stimulus-level value per audience (OCEAN = mean of its 32 personas;
#    controls = their single repetition-averaged response). Persona SD is
#    retained for OCEAN so we can see within-stimulus persona spread.
# -------------------------------------------------------------------------
stim <- agent_stim %>%
  group_by(audience_group, stimulus_file, condition, target_v, target_a) %>%
  summarise(persona_sd_v = ifelse(n() > 1, sd(perceived_v, na.rm = TRUE), NA_real_),
            persona_sd_a = ifelse(n() > 1, sd(perceived_a, na.rm = TRUE), NA_real_),
            perceived_v = mean(perceived_v, na.rm = TRUE),
            perceived_a = mean(perceived_a, na.rm = TRUE),
            alignment_distance = mean(alignment_distance, na.rm = TRUE),
            respondents = n(), .groups = "drop")

# -------------------------------------------------------------------------
# Descriptive summaries (means + SDs only)
# -------------------------------------------------------------------------
summary_overall <- stim %>%
  group_by(audience_group) %>%
  summarise(n_stimuli = n(),
            mean_valence = mean(perceived_v, na.rm = TRUE),  sd_valence = sd(perceived_v, na.rm = TRUE),
            mean_arousal = mean(perceived_a, na.rm = TRUE),  sd_arousal = sd(perceived_a, na.rm = TRUE),
            mean_distance = mean(alignment_distance, na.rm = TRUE), sd_distance = sd(alignment_distance, na.rm = TRUE),
            .groups = "drop")

summary_by_condition <- stim %>%
  group_by(audience_group, condition) %>%
  summarise(n_stimuli = n(),
            mean_valence = mean(perceived_v, na.rm = TRUE),  sd_valence = sd(perceived_v, na.rm = TRUE),
            mean_arousal = mean(perceived_a, na.rm = TRUE),  sd_arousal = sd(perceived_a, na.rm = TRUE),
            mean_distance = mean(alignment_distance, na.rm = TRUE), sd_distance = sd(alignment_distance, na.rm = TRUE),
            .groups = "drop")

# -------------------------------------------------------------------------
# Stimulus-matched control-minus-OCEAN differences
# -------------------------------------------------------------------------
dup <- stim %>%
  count(audience_group, stimulus_file, condition, name = "n_rows") %>%
  filter(n_rows > 1)
if (nrow(dup) > 0) {
  cat("\n!! ", nrow(dup), " (audience_group, stimulus_file, condition) combinations\n", sep = "")
  cat("   appear more than once in `stim`. Run analysis/Diagnose_baselines.R to see why.\n")
  print(utils::head(dup, 10))
  stop("cannot pivot: stimulus-level rows are not unique")
}

wide <- stim %>%
  select(audience_group, stimulus_file, condition, target_v, target_a,
         perceived_v, perceived_a, alignment_distance) %>%
  pivot_wider(names_from = audience_group,
              values_from = c(perceived_v, perceived_a, alignment_distance),
              names_sep = "__")

diff_summary <- data.frame()
need <- c("perceived_v__OCEAN audience","perceived_a__OCEAN audience","alignment_distance__OCEAN audience",
          "perceived_v__Neutral control","perceived_a__Neutral control","alignment_distance__Neutral control",
          "perceived_v__Generic-listener control","perceived_a__Generic-listener control","alignment_distance__Generic-listener control")
if (all(need %in% names(wide))) {
  diffs <- wide %>% mutate(
    neutral_minus_ocean_v = `perceived_v__Neutral control` - `perceived_v__OCEAN audience`,
    neutral_minus_ocean_a = `perceived_a__Neutral control` - `perceived_a__OCEAN audience`,
    neutral_minus_ocean_d = `alignment_distance__Neutral control` - `alignment_distance__OCEAN audience`,
    generic_minus_ocean_v = `perceived_v__Generic-listener control` - `perceived_v__OCEAN audience`,
    generic_minus_ocean_a = `perceived_a__Generic-listener control` - `perceived_a__OCEAN audience`,
    generic_minus_ocean_d = `alignment_distance__Generic-listener control` - `alignment_distance__OCEAN audience`)
  diff_summary <- diffs %>% summarise(across(starts_with(c("neutral_minus","generic_minus")), ~ mean(.x, na.rm = TRUE)))
  write.csv(diffs, file.path(OUT, "baseline_stimulus_differences.csv"), row.names = FALSE)
} else {
  warning("One or more control groups not found; matched differences skipped.")
}

# -------------------------------------------------------------------------
# THE CONTROL ARGUMENT: OCEAN persona spread vs the neutral-generic gap
# (per-persona means across all stimuli, vs the gap between the two controls)
# -------------------------------------------------------------------------
ocean_pers <- base %>% filter(audience_group == "OCEAN audience") %>%
  group_by(respondent_id) %>%
  summarise(v = mean(perceived_v, na.rm = TRUE),
            a = mean(perceived_a, na.rm = TRUE),
            d = mean(alignment_distance, na.rm = TRUE), .groups = "drop")

ctrl_mean <- function(grp, col) mean(base[[col]][base$audience_group == grp], na.rm = TRUE)

# -------------------------------------------------------------------------
# Text results
# -------------------------------------------------------------------------
sink(file.path(OUT, "baseline_results.txt"), split = TRUE)
cat("========== DESCRIPTIVE BASELINE ANALYSIS ==========\n\n")
cat("Stimulus-level OCEAN mean vs neutral and generic controls. Repetitions averaged first.\n")
cat("Descriptive only; no significance tests.\n\n")

cat("---------- Response counts ----------\n"); print(table(base$audience_group))

cat("\n---------- Overall summary (stimulus level) ----------\n"); print(as.data.frame(summary_overall), row.names = FALSE)
cat("\n---------- By condition ----------\n"); print(as.data.frame(summary_by_condition), row.names = FALSE)

if (nrow(diff_summary) > 0) {
  cat("\n---------- Mean control-minus-OCEAN differences (per stimulus) ----------\n")
  print(as.data.frame(round(diff_summary, 3)), row.names = FALSE)
  cat("Positive = control scored higher than the OCEAN mean.\n")
}

cat("\n---------- CONTROL ARGUMENT: OCEAN persona spread vs baseline gap ----------\n")
for (ax in c("v","a","d")) {
  pm <- ocean_pers[[ax]]
  col <- c(v="perceived_v", a="perceived_a", d="alignment_distance")[[ax]]
  nb <- ctrl_mean("Neutral control", col); gb <- ctrl_mean("Generic-listener control", col)
  label <- c(v="valence", a="arousal", d="distance")[[ax]]
  cat(sprintf("  %-8s OCEAN persona means %.3f to %.3f (SD %.3f) | neutral %.3f, generic %.3f | gap %.3f\n",
              label, min(pm), max(pm), sd(pm), nb, gb, abs(nb - gb)))
}
cat("\nWhere the OCEAN persona range/SD exceeds the neutral-generic gap, the between-persona\n")
cat("variation is trait-driven rather than an artefact of role framing (descriptive; 3.6.2).\n")
sink()

# -------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------
write.csv(stim, file.path(OUT, "stimulus_level_baselines.csv"), row.names = FALSE)
write.csv(summary_overall, file.path(OUT, "baseline_summary_overall.csv"), row.names = FALSE)
write.csv(summary_by_condition, file.path(OUT, "baseline_summary_by_condition.csv"), row.names = FALSE)
write.csv(ocean_pers, file.path(OUT, "ocean_persona_means.csv"), row.names = FALSE)

# -------------------------------------------------------------------------
# Figure 1: boxplots - OCEAN vs controls, per measure and condition
# -------------------------------------------------------------------------
plot_data <- stim %>%
  select(audience_group, condition, stimulus_file, perceived_v, perceived_a, alignment_distance) %>%
  pivot_longer(c(perceived_v, perceived_a, alignment_distance), names_to = "measure", values_to = "value") %>%
  mutate(measure = recode(measure, perceived_v = "Perceived valence",
                          perceived_a = "Perceived arousal", alignment_distance = "Alignment distance"))

ggsave(file.path(OUT, "baseline_distributions.png"),
  ggplot(plot_data, aes(audience_group, value)) +
    geom_boxplot(fill = "grey90") +
    facet_grid(measure ~ condition, scales = "free_y") +
    labs(title = "Descriptive comparison of OCEAN audience and controls",
         subtitle = "Each value = one stimulus (repetitions averaged); points beyond 1.5xIQR are potential outliers",
         x = NULL, y = NULL) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(), axis.text.x = element_text(angle = 20, hjust = 1)),
  width = 11, height = 8.5, dpi = 300)

# -------------------------------------------------------------------------
# Figure 2: mean perceived VA position by audience
# -------------------------------------------------------------------------
va <- stim %>% group_by(audience_group, condition) %>%
  summarise(mean_v = mean(perceived_v, na.rm = TRUE), mean_a = mean(perceived_a, na.rm = TRUE), .groups = "drop")

ggsave(file.path(OUT, "baseline_mean_va_positions.png"),
  ggplot(va, aes(mean_v, mean_a, label = audience_group)) +
    geom_hline(yintercept = 0, colour = "grey85") + geom_vline(xintercept = 0, colour = "grey85") +
    geom_point(size = 3, colour = "firebrick") + geom_text(vjust = -1, size = 3.5) +
    facet_wrap(~ condition) + coord_fixed(xlim = c(-1, 1), ylim = c(-1, 1)) +
    labs(title = "Mean perceived valence-arousal position by audience",
         x = "Perceived valence", y = "Perceived arousal") +
    theme_minimal(base_size = 12) + theme(panel.grid.minor = element_blank()),
  width = 9, height = 5.5, dpi = 300)

cat("\nBaseline analysis complete. Outputs in", OUT, "\n")