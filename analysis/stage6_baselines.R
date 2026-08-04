# Stage 6 - descriptive baseline analysis
#
# Compares:
#   1. Mean response of the 32 OCEAN personas
#   2. Neutral control
#   3. Generic-listener control
#
# The analysis is descriptive only. It does not test H1-H3.
#
# Processing:
#   - First average the three repetitions for each agent-stimulus combination.
#   - Then average the 32 OCEAN personas for each stimulus.
#   - Compare the resulting stimulus-level OCEAN mean with the two controls.
#   - Check whether stimulus-level distributions support reporting means.
#
# Run:
#   Rscript analysis/stage6_baselines.R
#
# Required packages:
#   install.packages(c("dplyr", "tidyr", "ggplot2"))

library(dplyr)
library(tidyr)
library(ggplot2)


# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------

OUT <- "analysis/baselines"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)


# -------------------------------------------------------------------------
# Helper function: adjusted sample skewness
#
# Interpretation used for descriptive guidance:
#   absolute skewness below 0.50 = approximately symmetrical
#   absolute skewness from 0.50 to 1.00 = moderately skewed
#   absolute skewness above 1.00 = strongly skewed
#
# These are practical descriptive thresholds, not formal hypothesis tests.
# -------------------------------------------------------------------------

sample_skewness <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)

  if (n < 3) {
    return(NA_real_)
  }

  x_sd <- sd(x)

  if (!is.finite(x_sd) || x_sd == 0) {
    return(0)
  }

  n / ((n - 1) * (n - 2)) *
    sum(((x - mean(x)) / x_sd)^3)
}


# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

d <- read.csv(
  "data/audience/responses.csv",
  stringsAsFactors = FALSE
)

required_columns <- c(
  "agent_kind",
  "stimulus_file",
  "condition",
  "perceived_v",
  "perceived_a",
  "target_v",
  "target_a"
)

missing_columns <- setdiff(required_columns, names(d))

if (length(missing_columns) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

if (!"persona_id" %in% names(d)) {
  d$persona_id <- NA_character_
}


# -------------------------------------------------------------------------
# Prepare baseline data
# -------------------------------------------------------------------------

baseline_data <- d %>%
  mutate(
    respondent_id = ifelse(
      is.na(persona_id) | persona_id == "",
      agent_kind,
      persona_id
    ),
    audience_group = case_when(
      tolower(agent_kind) == "ocean" ~ "OCEAN audience",
      grepl("neutral", tolower(agent_kind)) ~ "Neutral control",
      grepl("generic", tolower(agent_kind)) ~ "Generic-listener control",
      TRUE ~ NA_character_
    ),
    alignment_distance = sqrt(
      (perceived_v - target_v)^2 +
        (perceived_a - target_a)^2
    )
  ) %>%
  filter(!is.na(audience_group))

if (nrow(baseline_data) == 0) {
  stop("No OCEAN, neutral-control or generic-control responses were found.")
}


# -------------------------------------------------------------------------
# Average the three repetitions for each respondent-stimulus combination
# -------------------------------------------------------------------------

agent_stimulus_means <- baseline_data %>%
  group_by(
    audience_group,
    respondent_id,
    stimulus_file,
    condition,
    target_v,
    target_a
  ) %>%
  summarise(
    perceived_v = mean(perceived_v, na.rm = TRUE),
    perceived_a = mean(perceived_a, na.rm = TRUE),
    alignment_distance = mean(alignment_distance, na.rm = TRUE),
    repetitions = n(),
    .groups = "drop"
  )


# -------------------------------------------------------------------------
# Create one stimulus-level value for each audience condition
#
# For the OCEAN audience, this averages the 32 persona means.
# For each control, this retains its repetition-averaged stimulus response.
# -------------------------------------------------------------------------

stimulus_level <- agent_stimulus_means %>%
  group_by(
    audience_group,
    stimulus_file,
    condition,
    target_v,
    target_a
  ) %>%
  summarise(
    perceived_v = mean(perceived_v, na.rm = TRUE),
    perceived_a = mean(perceived_a, na.rm = TRUE),
    alignment_distance = mean(alignment_distance, na.rm = TRUE),
    persona_sd_v = ifelse(
      n() > 1,
      sd(perceived_v, na.rm = TRUE),
      NA_real_
    ),
    persona_sd_a = ifelse(
      n() > 1,
      sd(perceived_a, na.rm = TRUE),
      NA_real_
    ),
    persona_sd_distance = ifelse(
      n() > 1,
      sd(alignment_distance, na.rm = TRUE),
      NA_real_
    ),
    respondents = n(),
    .groups = "drop"
  )


# -------------------------------------------------------------------------
# Mean-based descriptive summaries
# -------------------------------------------------------------------------

summary_by_condition <- stimulus_level %>%
  group_by(
    audience_group,
    condition
  ) %>%
  summarise(
    n_stimuli = n(),
    mean_valence = mean(perceived_v, na.rm = TRUE),
    sd_valence = sd(perceived_v, na.rm = TRUE),
    mean_arousal = mean(perceived_a, na.rm = TRUE),
    sd_arousal = sd(perceived_a, na.rm = TRUE),
    mean_alignment_distance = mean(alignment_distance, na.rm = TRUE),
    sd_alignment_distance = sd(alignment_distance, na.rm = TRUE),
    .groups = "drop"
  )

summary_overall <- stimulus_level %>%
  group_by(audience_group) %>%
  summarise(
    n_stimuli = n(),
    mean_valence = mean(perceived_v, na.rm = TRUE),
    sd_valence = sd(perceived_v, na.rm = TRUE),
    mean_arousal = mean(perceived_a, na.rm = TRUE),
    sd_arousal = sd(perceived_a, na.rm = TRUE),
    mean_alignment_distance = mean(alignment_distance, na.rm = TRUE),
    sd_alignment_distance = sd(alignment_distance, na.rm = TRUE),
    .groups = "drop"
  )


# -------------------------------------------------------------------------
# Stimulus-matched descriptive differences from the OCEAN mean
# -------------------------------------------------------------------------

paired_wide <- stimulus_level %>%
  select(
    audience_group,
    stimulus_file,
    condition,
    perceived_v,
    perceived_a,
    alignment_distance
  ) %>%
  pivot_wider(
    names_from = audience_group,
    values_from = c(
      perceived_v,
      perceived_a,
      alignment_distance
    ),
    names_sep = "__"
  )

needed_difference_columns <- c(
  "perceived_v__OCEAN audience",
  "perceived_a__OCEAN audience",
  "alignment_distance__OCEAN audience",
  "perceived_v__Neutral control",
  "perceived_a__Neutral control",
  "alignment_distance__Neutral control",
  "perceived_v__Generic-listener control",
  "perceived_a__Generic-listener control",
  "alignment_distance__Generic-listener control"
)

if (all(needed_difference_columns %in% names(paired_wide))) {
  paired_differences <- paired_wide %>%
    mutate(
      neutral_minus_ocean_v =
        `perceived_v__Neutral control` -
        `perceived_v__OCEAN audience`,
      neutral_minus_ocean_a =
        `perceived_a__Neutral control` -
        `perceived_a__OCEAN audience`,
      neutral_minus_ocean_distance =
        `alignment_distance__Neutral control` -
        `alignment_distance__OCEAN audience`,
      generic_minus_ocean_v =
        `perceived_v__Generic-listener control` -
        `perceived_v__OCEAN audience`,
      generic_minus_ocean_a =
        `perceived_a__Generic-listener control` -
        `perceived_a__OCEAN audience`,
      generic_minus_ocean_distance =
        `alignment_distance__Generic-listener control` -
        `alignment_distance__OCEAN audience`
    )

  difference_summary <- paired_differences %>%
    summarise(
      neutral_minus_ocean_v =
        mean(neutral_minus_ocean_v, na.rm = TRUE),
      neutral_minus_ocean_a =
        mean(neutral_minus_ocean_a, na.rm = TRUE),
      neutral_minus_ocean_distance =
        mean(neutral_minus_ocean_distance, na.rm = TRUE),
      generic_minus_ocean_v =
        mean(generic_minus_ocean_v, na.rm = TRUE),
      generic_minus_ocean_a =
        mean(generic_minus_ocean_a, na.rm = TRUE),
      generic_minus_ocean_distance =
        mean(generic_minus_ocean_distance, na.rm = TRUE)
    )
} else {
  warning(
    paste(
      "One or more control groups were not found.",
      "Stimulus-matched difference summaries were not produced."
    )
  )

  paired_differences <- data.frame()
  difference_summary <- data.frame()
}


# -------------------------------------------------------------------------
# Reshape stimulus-level values for distribution checks and plots
#
# The checks are performed at the same stimulus level used in the baseline
# comparison. They are not performed on the 9,792 unaggregated responses.
# -------------------------------------------------------------------------

plot_data <- stimulus_level %>%
  select(
    audience_group,
    condition,
    stimulus_file,
    perceived_v,
    perceived_a,
    alignment_distance
  ) %>%
  pivot_longer(
    cols = c(
      perceived_v,
      perceived_a,
      alignment_distance
    ),
    names_to = "measure",
    values_to = "value"
  ) %>%
  mutate(
    measure = recode(
      measure,
      perceived_v = "Perceived valence",
      perceived_a = "Perceived arousal",
      alignment_distance = "Alignment distance"
    )
  )


# -------------------------------------------------------------------------
# Distribution diagnostics for choosing mean or median
#
# Mean is supported when:
#   - the distribution is approximately symmetrical;
#   - the mean and median are close;
#   - there are no influential extreme observations.
#
# Median and IQR should be emphasised when:
#   - absolute skewness is high;
#   - the mean and median differ materially;
#   - several boxplot outliers are present.
#
# The recommendation is descriptive guidance. The researcher should also
# inspect the saved boxplots and consider the meaning of the measure.
# -------------------------------------------------------------------------

distribution_diagnostics <- plot_data %>%
  group_by(
    audience_group,
    condition,
    measure
  ) %>%
  summarise(
    n = sum(is.finite(value)),
    mean = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    mean_median_difference =
      mean(value, na.rm = TRUE) - median(value, na.rm = TRUE),
    absolute_mean_median_difference = abs(
      mean(value, na.rm = TRUE) - median(value, na.rm = TRUE)
    ),
    sd = sd(value, na.rm = TRUE),
    q1 = as.numeric(quantile(value, 0.25, na.rm = TRUE)),
    q3 = as.numeric(quantile(value, 0.75, na.rm = TRUE)),
    iqr = q3 - q1,
    lower_outlier_fence = q1 - 1.5 * iqr,
    upper_outlier_fence = q3 + 1.5 * iqr,
    skewness = sample_skewness(value),
    absolute_skewness = abs(skewness),
    outlier_count = sum(
      value < lower_outlier_fence |
        value > upper_outlier_fence,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  mutate(
    distribution_shape = case_when(
      is.na(absolute_skewness) ~ "Insufficient variation",
      absolute_skewness < 0.50 ~ "Approximately symmetrical",
      absolute_skewness <= 1.00 ~ "Moderately skewed",
      absolute_skewness > 1.00 ~ "Strongly skewed"
    ),
    descriptive_recommendation = case_when(
      is.na(absolute_skewness) ~
        "Inspect values directly",
      absolute_skewness > 1.00 ~
        "Emphasise median and IQR; also report mean for comparability",
      outlier_count > 0 & absolute_skewness >= 0.50 ~
        "Report mean and median; interpret the mean cautiously",
      absolute_skewness >= 0.50 ~
        "Report mean and median because moderate skewness is present",
      outlier_count > 0 ~
        "Mean is usable, but inspect and disclose potential outliers",
      TRUE ~
        "Mean and SD are appropriate primary summaries"
    )
  )


# -------------------------------------------------------------------------
# Text results
# -------------------------------------------------------------------------

sink(
  file.path(OUT, "baseline_results.txt"),
  split = TRUE
)

cat("========== DESCRIPTIVE BASELINE ANALYSIS ==========\n\n")
cat("The analysis compares the stimulus-level OCEAN audience mean with\n")
cat("the neutral and generic-listener controls. Repetitions are averaged first.\n")
cat("No inferential significance tests are used.\n\n")

cat("---------- OVERALL SUMMARY ----------\n")
print(summary_overall)

cat("\n---------- SUMMARY BY CONDITION ----------\n")
print(summary_by_condition)

if (nrow(difference_summary) > 0) {
  cat("\n---------- MEAN CONTROL MINUS OCEAN DIFFERENCES ----------\n")
  print(difference_summary)
  cat("\nPositive values mean that the control scored higher than the OCEAN mean.\n")
}

cat("\n---------- DISTRIBUTION DIAGNOSTICS ----------\n")
print(distribution_diagnostics)

cat("\nInterpretation guide:\n")
cat("|skewness| < 0.50: approximately symmetrical.\n")
cat("|skewness| 0.50-1.00: moderately skewed.\n")
cat("|skewness| > 1.00: strongly skewed.\n")
cat("Potential outliers use the standard 1.5 x IQR boxplot rule.\n")
cat("The final choice should also consider the mean-median difference and plots.\n")

sink()


# -------------------------------------------------------------------------
# Save tables
# -------------------------------------------------------------------------

write.csv(
  agent_stimulus_means,
  file.path(OUT, "agent_stimulus_means.csv"),
  row.names = FALSE
)

write.csv(
  stimulus_level,
  file.path(OUT, "stimulus_level_baselines.csv"),
  row.names = FALSE
)

write.csv(
  summary_by_condition,
  file.path(OUT, "baseline_summary_by_condition.csv"),
  row.names = FALSE
)

write.csv(
  summary_overall,
  file.path(OUT, "baseline_summary_overall.csv"),
  row.names = FALSE
)

write.csv(
  distribution_diagnostics,
  file.path(OUT, "baseline_distribution_diagnostics.csv"),
  row.names = FALSE
)

if (nrow(paired_differences) > 0) {
  write.csv(
    paired_differences,
    file.path(OUT, "baseline_stimulus_differences.csv"),
    row.names = FALSE
  )
}


# -------------------------------------------------------------------------
# Figure 1: boxplots for distribution shape and potential outliers
# -------------------------------------------------------------------------

distribution_plot <- ggplot(
  plot_data,
  aes(
    x = audience_group,
    y = value
  )
) +
  geom_boxplot() +
  facet_grid(
    measure ~ condition,
    scales = "free_y"
  ) +
  labs(
    title = "Descriptive comparison of OCEAN audience and controls",
    subtitle = paste(
      "Each value represents one stimulus after averaging repetitions;",
      "boxplot points beyond 1.5 x IQR indicate potential outliers"
    ),
    x = NULL,
    y = NULL
  ) +
  theme_minimal(
    base_size = 12
  ) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(
      angle = 20,
      hjust = 1
    )
  )

ggsave(
  filename = file.path(
    OUT,
    "baseline_distributions.png"
  ),
  plot = distribution_plot,
  width = 11,
  height = 9,
  dpi = 300
)


# -------------------------------------------------------------------------
# Figure 2: histograms for distribution symmetry
# -------------------------------------------------------------------------

histogram_plot <- ggplot(
  plot_data,
  aes(x = value)
) +
  geom_histogram(
    bins = 12,
    boundary = 0
  ) +
  geom_vline(
    data = distribution_diagnostics,
    aes(xintercept = mean),
    linetype = "solid",
    linewidth = 0.7
  ) +
  geom_vline(
    data = distribution_diagnostics,
    aes(xintercept = median),
    linetype = "dashed",
    linewidth = 0.7
  ) +
  facet_grid(
    measure + audience_group ~ condition,
    scales = "free"
  ) +
  labs(
    title = "Stimulus-level distribution checks",
    subtitle = "Solid line = mean; dashed line = median",
    x = "Stimulus-level value",
    y = "Number of stimuli"
  ) +
  theme_minimal(
    base_size = 11
  ) +
  theme(
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(
    OUT,
    "baseline_distribution_histograms.png"
  ),
  plot = histogram_plot,
  width = 12,
  height = 14,
  dpi = 300
)


# -------------------------------------------------------------------------
# Figure 3: mean perceived valence-arousal positions
# -------------------------------------------------------------------------

va_summary <- stimulus_level %>%
  group_by(
    audience_group,
    condition
  ) %>%
  summarise(
    mean_v = mean(perceived_v, na.rm = TRUE),
    mean_a = mean(perceived_a, na.rm = TRUE),
    .groups = "drop"
  )

va_plot <- ggplot(
  va_summary,
  aes(
    x = mean_v,
    y = mean_a,
    label = audience_group
  )
) +
  geom_hline(yintercept = 0) +
  geom_vline(xintercept = 0) +
  geom_point(size = 3) +
  geom_text(vjust = -1) +
  facet_wrap(~ condition) +
  coord_fixed(
    xlim = c(-1, 1),
    ylim = c(-1, 1)
  ) +
  labs(
    title = "Mean perceived valence-arousal position by audience condition",
    x = "Perceived valence",
    y = "Perceived arousal"
  ) +
  theme_minimal(
    base_size = 12
  ) +
  theme(
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(
    OUT,
    "baseline_mean_va_positions.png"
  ),
  plot = va_plot,
  width = 9,
  height = 5.5,
  dpi = 300
)


# -------------------------------------------------------------------------
# Completion message
# -------------------------------------------------------------------------

cat("\nBaseline analysis completed.\n")
cat("Saved outputs to:", OUT, "\n")
cat("Mean/median justification table:\n")
cat(file.path(OUT, "baseline_distribution_diagnostics.csv"), "\n")