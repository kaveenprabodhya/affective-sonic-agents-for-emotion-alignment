# H3 - AUDIENCE alignment: does intended-perceived DISTANCE differ across OCEAN personas?
#   Intended VA <-> persona-perceived VA, using OPTIMISED logos.
#
#   Primary:
#     distance ~ OCEAN + (1|persona_id) + (1|brief/run)
#
#   Omnibus H3 test:
#     compares the full OCEAN model with an identical model containing no OCEAN traits.
#
#   Descriptive diagnostics:
#     quadrant differences show where alignment is stronger or weaker;
#     valence and arousal offsets show the direction of misalignment.
#
#   The optimised-vs-non-optimised audience comparison is NOT part of H3.
#   Stimulus-side optimisation is assessed by H1 using independent Estimator B.
#
#   Run:
#     Rscript analysis/stage6_h3_alignment.R

library(lme4)
library(lmerTest)
library(ggplot2)
library(dplyr)


# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------

OUT <- "analysis/h3"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)


# -------------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------------

d <- read.csv(
  "data/audience/responses.csv",
  stringsAsFactors = FALSE
)

traits <- c(
  "openness",
  "conscientiousness",
  "extraversion",
  "agreeableness",
  "neuroticism"
)

# Primary H3 uses OCEAN personas and optimised sonic logos only.
oc <- d %>%
  filter(
    agent_kind == "ocean",
    condition == "optimised"
  ) %>%
  mutate(
    # Euclidean distance between intended and perceived VA positions
    dist = sqrt(
      (perceived_v - target_v)^2 +
      (perceived_a - target_a)^2
    ),

    # Separate differences show the direction of misalignment
    off_v = perceived_v - target_v,
    off_a = perceived_a - target_a,

    # Intended VA quadrant
    quadrant = paste0(
      ifelse(target_v >= 0, "H", "L"),
      "V_",
      ifelse(target_a >= 0, "H", "L"),
      "A"
    ),

    # Extract generation-run number from the stimulus filename
    run = sub(
      ".*_run([0-9]+)_.*",
      "\\1",
      stimulus_file
    )
  )

# Set low as the reference level for every OCEAN trait
for (trait in traits) {
  oc[[trait]] <- relevel(
    factor(oc[[trait]]),
    ref = "low"
  )
}

oc$quadrant <- relevel(
  factor(oc$quadrant),
  ref = "HV_HA"
)

oc$persona_id <- factor(oc$persona_id)
oc$brief <- factor(oc$brief)
oc$run <- factor(oc$run)


# -------------------------------------------------------------------------
# Results output
# -------------------------------------------------------------------------

sink(
  file.path(OUT, "h3_results.txt"),
  split = TRUE
)

cat(
  "========== PRIMARY H3: ALIGNMENT DISTANCE ACROSS OCEAN PERSONAS ==========\n"
)

cat(
  "Model: distance ~ OCEAN traits + (1|persona_id) + (1|brief/run)\n\n"
)

cat(
  "Quadrant is not a primary H3 factor. It is reported descriptively below.\n\n"
)


# -------------------------------------------------------------------------
# Primary H3 model
# -------------------------------------------------------------------------

m <- lmer(
  dist ~
    openness +
    conscientiousness +
    extraversion +
    agreeableness +
    neuroticism +
    (1 | persona_id) +
    (1 | brief/run),
  data = oc
)

cat("---------- FIXED-EFFECT ESTIMATES ----------\n")
print(summary(m)$coefficients)

cat(
  "\nSingular fit:",
  isSingular(m),
  "\n"
)


# -------------------------------------------------------------------------
# Singularity fallback
# -------------------------------------------------------------------------

if (isSingular(m)) {

  cat(
    "\nThe brief/run model was singular.\n",
    "Refitting with persona and brief random intercepts.\n\n"
  )

  m <- lmer(
    dist ~
      openness +
      conscientiousness +
      extraversion +
      agreeableness +
      neuroticism +
      (1 | persona_id) +
      (1 | brief),
    data = oc
  )

  cat("---------- REFITTED FIXED-EFFECT ESTIMATES ----------\n")
  print(summary(m)$coefficients)

  cat(
    "\nSingular fit after refitting:",
    isSingular(m),
    "\n"
  )
}


# -------------------------------------------------------------------------
# Omnibus H3 likelihood-ratio test
# -------------------------------------------------------------------------

cat(
  "\n========== OMNIBUS H3 TEST: OCEAN TRAITS JOINTLY ==========\n"
)

# Likelihood-ratio tests must compare models fitted using maximum likelihood
m_full <- update(
  m,
  REML = FALSE
)

# Same random-effects structure, but no OCEAN predictors
m_null <- update(
  m_full,
  . ~ . -
    openness -
    conscientiousness -
    extraversion -
    agreeableness -
    neuroticism,
  REML = FALSE
)

h3_omnibus <- anova(
  m_null,
  m_full
)

print(h3_omnibus)

cat(
  "\nThe omnibus test determines whether the five OCEAN traits jointly\n",
  "improve the model of intended-perceived alignment distance.\n"
)


# -------------------------------------------------------------------------
# Descriptive quadrant diagnostic
# -------------------------------------------------------------------------

cat(
  "\n========== DESCRIPTIVE DIAGNOSTIC: ALIGNMENT BY QUADRANT ==========\n"
)

quadrant_summary <- oc %>%
  group_by(quadrant) %>%
  summarise(
    mean_dist = round(mean(dist), 3),
    sd = round(sd(dist), 3),
    n = n(),
    .groups = "drop"
  )

print(quadrant_summary)

cat(
  "\nThese values are descriptive and are not used to test H3.\n"
)


# -------------------------------------------------------------------------
# Descriptive direction-of-misalignment diagnostic
# -------------------------------------------------------------------------

cat(
  "\n========== DESCRIPTIVE DIAGNOSTIC: DIRECTION OF MISALIGNMENT ==========\n"
)

offset_summary <- oc %>%
  group_by(quadrant) %>%
  summarise(
    valence_difference = mean(off_v),
    arousal_difference = mean(off_a),
    .groups = "drop"
  )

for (i in seq_len(nrow(offset_summary))) {

  cat(
    sprintf(
      "%s: valence %+.3f, arousal %+.3f\n",
      offset_summary$quadrant[i],
      offset_summary$valence_difference[i],
      offset_summary$arousal_difference[i]
    )
  )
}

cat(
  "\nPositive valence values mean that personas perceived the logos as more\n",
  "positive than intended. Positive arousal values mean that personas perceived\n",
  "the logos as more energetic than intended.\n"
)

sink()


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------

theme_set(
  theme_minimal(base_size = 13)
)


# Figure 1: alignment distance by intended quadrant
p_distance <- ggplot(
  oc,
  aes(
    x = quadrant,
    y = dist
  )
) +
  geom_boxplot(
    fill = "grey85"
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    colour = "firebrick",
    size = 3
  ) +
  labs(
    title = "Alignment distance by intended quadrant",
    x = "Intended quadrant",
    y = "Intended-perceived Euclidean distance"
  )

ggsave(
  filename = file.path(
    OUT,
    "h3_distance_by_quadrant.png"
  ),
  plot = p_distance,
  width = 8,
  height = 5.5,
  dpi = 300
)


# Figure 2: intended position to mean perceived position
centroids <- oc %>%
  group_by(quadrant) %>%
  summarise(
    intended_v = mean(target_v),
    intended_a = mean(target_a),
    perceived_v = mean(perceived_v),
    perceived_a = mean(perceived_a),
    .groups = "drop"
  )

p_offsets <- ggplot() +
  geom_hline(
    yintercept = 0,
    colour = "grey80"
  ) +
  geom_vline(
    xintercept = 0,
    colour = "grey80"
  ) +
  geom_segment(
    data = centroids,
    aes(
      x = intended_v,
      y = intended_a,
      xend = perceived_v,
      yend = perceived_a
    ),
    arrow = arrow(
      length = unit(0.2, "cm")
    ),
    colour = "firebrick"
  ) +
  geom_point(
    data = centroids,
    aes(
      x = intended_v,
      y = intended_a
    ),
    size = 3
  ) +
  geom_text(
    data = centroids,
    aes(
      x = intended_v,
      y = intended_a,
      label = quadrant
    ),
    vjust = -1,
    size = 3.5
  ) +
  coord_fixed(
    xlim = c(-1, 1),
    ylim = c(-1, 1)
  ) +
  labs(
    title = "Directional offset from intended to mean perceived position",
    x = "Valence",
    y = "Arousal"
  )

ggsave(
  filename = file.path(
    OUT,
    "h3_offset_vectors.png"
  ),
  plot = p_offsets,
  width = 7,
  height = 7,
  dpi = 300
)


cat(
  sprintf(
    "Saved: %s/h3_results.txt and two H3 figures\n",
    OUT
  )
)