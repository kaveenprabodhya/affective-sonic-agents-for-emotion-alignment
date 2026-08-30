# Stage 6 - exploratory MDU of persona x emotion-term structure.
#
# Outputs:
#   1. One pooled MDU across all stimuli and conditions.
#   2. Four quadrant-specific MDUs using optimised logos only:
#        HV_HA, HV_LA, LV_HA and LV_LA.
#
# Each analysis creates:
#   - persona-and-emotion plot
#   - emotion-only plot
#   - persona coordinates
#   - emotion coordinates
#   - aggregated persona-by-emotion matrix
#
# Run:
#   Rscript analysis/stage6_mdu.R
#
# Required packages:
#   install.packages(c("smacof", "ggplot2", "ggrepel"))

library(smacof)
library(ggplot2)
library(ggrepel)


# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------

OUT <- "analysis/mdu"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)


# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

d <- read.csv(
  "data/audience/responses.csv",
  stringsAsFactors = FALSE
)

ocean <- subset(
  d,
  agent_kind == "ocean"
)

emo_cols <- paste0("Q", 3:12)

emotion_labels <- c(
  "happy",
  "tense",
  "calm",
  "sad",
  "excited",
  "afraid",
  "content",
  "bored",
  "relaxed",
  "angry"
)


# -------------------------------------------------------------------------
# Create intended quadrant
# -------------------------------------------------------------------------

ocean$quadrant <- paste0(
  ifelse(ocean$target_v >= 0, "HV", "LV"),
  "_",
  ifelse(ocean$target_a >= 0, "HA", "LA")
)

quadrant_order <- c(
  "HV_HA",
  "HV_LA",
  "LV_HA",
  "LV_LA"
)


# -------------------------------------------------------------------------
# Function for fitting and saving one MDU solution
# -------------------------------------------------------------------------

run_mdu <- function(data_subset, analysis_name, output_directory) {

  dir.create(
    output_directory,
    recursive = TRUE,
    showWarnings = FALSE
  )

  if (nrow(data_subset) == 0) {
    stop(
      paste(
        "No observations were found for:",
        analysis_name
      )
    )
  }

  # Average Q3-Q12 ratings for each persona within this data subset.
  aggregated <- aggregate(
    data_subset[emo_cols],
    by = list(persona = data_subset$persona_id),
    FUN = mean,
    na.rm = TRUE
  )

  # Create the 32-persona x 10-emotion matrix.
  M <- as.matrix(
    aggregated[emo_cols]
  )

  rownames(M) <- aggregated$persona
  colnames(M) <- emotion_labels

  # Q3-Q12 use a 1-5 scale.
  # A high rating should become a short unfolding distance.
  D <- 5 - M

  set.seed(42)

  unfolding_fit <- unfolding(
    D,
    type = "interval"
  )

  stress_value <- unfolding_fit$stress

  cat("\n============================================================\n")
  cat("Analysis:", analysis_name, "\n")
  cat("Responses:", nrow(data_subset), "\n")
  cat("Personas:", nrow(M), "\n")
  cat("Emotion terms:", ncol(M), "\n")
  cat("Stress-1:", round(stress_value, 4), "\n")

  if (!is.finite(stress_value)) {

    cat(
      "WARNING: Stress could not be calculated.\n"
    )

  } else if (stress_value < 0.01) {

    cat(
      "WARNING: Near-zero stress may indicate a degenerate solution.\n"
    )

  } else {

    cat(
      "Stress is non-zero. Interpret the configuration cautiously.\n"
    )
  }

  # Honest read of the persona positions: measure how spread the personas are.
  persona_spread <- mean(
    sqrt(rowSums(scale(unfolding_fit$conf.row, scale = FALSE)^2))
  )
  emotion_spread <- mean(
    sqrt(rowSums(scale(unfolding_fit$conf.col, scale = FALSE)^2))
  )
  cat(sprintf("Mean persona distance from centre: %.3f\n", persona_spread))
  cat(sprintf("Mean emotion distance from centre: %.3f\n", emotion_spread))
  cat("NOTE: if personas sit much closer to the centre than the emotion words,\n")
  cat("      they differ little in AGGREGATE emotion vocabulary (this supports the\n")
  cat("      coherence of the emotion terms, NOT strong persona differentiation).\n")


  # -----------------------------------------------------------------------
  # Extract coordinates
  # -----------------------------------------------------------------------

  emotion_coordinates <- data.frame(
    dim1 = unfolding_fit$conf.col[, 1],
    dim2 = unfolding_fit$conf.col[, 2],
    term = rownames(unfolding_fit$conf.col)
  )

  persona_coordinates <- data.frame(
    dim1 = unfolding_fit$conf.row[, 1],
    dim2 = unfolding_fit$conf.row[, 2],
    id = rownames(unfolding_fit$conf.row)
  )


  # -----------------------------------------------------------------------
  # Persona-and-emotion plot
  # -----------------------------------------------------------------------

  persona_emotion_plot <- ggplot() +

    geom_point(
      data = persona_coordinates,
      aes(
        x = dim1,
        y = dim2
      ),
      colour = "grey70",
      size = 1.4
    ) +

    geom_point(
      data = emotion_coordinates,
      aes(
        x = dim1,
        y = dim2
      ),
      colour = "firebrick",
      size = 2.6
    ) +

    geom_text_repel(
      data = emotion_coordinates,
      aes(
        x = dim1,
        y = dim2,
        label = term
      ),
      colour = "firebrick",
      fontface = "bold",
      size = 4.2,
      box.padding = 0.6,
      max.overlaps = Inf,
      seed = 1
    ) +

    geom_text_repel(
      data = persona_coordinates,
      aes(
        x = dim1,
        y = dim2,
        label = id
      ),
      colour = "grey55",
      size = 2.6,
      box.padding = 0.25,
      segment.alpha = 0.3,
      max.overlaps = Inf,
      seed = 1
    ) +

    labs(
      title = paste0(
        "Multidimensional Unfolding (MDU)\n",
        "Personas and emotion words: ", analysis_name
      ),
      subtitle = sprintf(
        "Emotion words rated similarly appear close together. Interval unfolding, Stress-1 = %.3f",
        stress_value
      ),
      x = "Dimension 1",
      y = "Dimension 2"
    ) +

    theme_minimal(
      base_size = 13
    ) +

    theme(
      panel.grid.minor = element_blank(),
      plot.title    = element_text(size = 12, lineheight = 1.1,
                                   margin = margin(b = 4)),
      plot.subtitle = element_text(size = 10,
                                   margin = margin(b = 8)),
      plot.title.position = "plot"
    )


  ggsave(
    filename = file.path(
      output_directory,
      "mdu_persona_emotion.png"
    ),
    plot = persona_emotion_plot,
    width = 10,
    height = 7,
    dpi = 300
  )


  # -----------------------------------------------------------------------
  # Emotion-only plot
  # -----------------------------------------------------------------------

  emotion_plot <- ggplot(
    emotion_coordinates,
    aes(
      x = dim1,
      y = dim2,
      label = term
    )
  ) +

    geom_point(
      colour = "firebrick",
      size = 3
    ) +

    geom_text_repel(
      colour = "firebrick",
      fontface = "bold",
      size = 5,
      box.padding = 0.8,
      max.overlaps = Inf,
      seed = 1
    ) +

    labs(
      title = paste(
        "Multidimensional Unfolding (MDU) - emotion words:",
        analysis_name
      ),
      subtitle = sprintf(
        "Emotion words rated similarly by the audience appear close together. Stress-1 = %.3f",
        stress_value
      ),
      x = "Dimension 1",
      y = "Dimension 2"
    ) +

    theme_minimal(
      base_size = 13
    ) +

    theme(
      panel.grid.minor = element_blank(),
      plot.title    = element_text(size = 12, lineheight = 1.1,
                                   margin = margin(b = 4)),
      plot.subtitle = element_text(size = 10,
                                   margin = margin(b = 8)),
      plot.title.position = "plot"
    )


  ggsave(
    filename = file.path(
      output_directory,
      "mdu_emotions_only.png"
    ),
    plot = emotion_plot,
    width = 9,
    height = 6.5,
    dpi = 300
  )


  # -----------------------------------------------------------------------
  # Save coordinates and input matrix
  # -----------------------------------------------------------------------

  write.csv(
    persona_coordinates,
    file.path(
      output_directory,
      "mdu_persona_coords.csv"
    ),
    row.names = FALSE
  )

  write.csv(
    emotion_coordinates,
    file.path(
      output_directory,
      "mdu_emotion_coords.csv"
    ),
    row.names = FALSE
  )

  write.csv(
    M,
    file.path(
      output_directory,
      "persona_emotion_matrix.csv"
    )
  )


  # Return fit information for the summary table.
  data.frame(
    analysis = analysis_name,
    responses = nrow(data_subset),
    personas = nrow(M),
    emotions = ncol(M),
    stress_1 = stress_value
  )
}


# -------------------------------------------------------------------------
# 1. Original pooled analysis
# -------------------------------------------------------------------------

fit_results <- list()

# Pooled map uses optimised logos only, matching the quadrant maps and the
# primary H3 analysis (avoids mixing optimised and non-optimised stimuli).
fit_results[[1]] <- run_mdu(
  data_subset = subset(ocean, condition == "optimised"),
  analysis_name = "Pooled (optimised logos)",
  output_directory = file.path(
    OUT,
    "pooled"
  )
)


# -------------------------------------------------------------------------
# 2. Quadrant-specific analyses
# -------------------------------------------------------------------------

# Use optimised logos only so each map represents the stimuli used for
# the primary H3 audience-alignment analysis.
optimised_ocean <- subset(
  ocean,
  condition == "optimised"
)

for (quadrant_name in quadrant_order) {

  quadrant_data <- subset(
    optimised_ocean,
    quadrant == quadrant_name
  )

  fit_results[[length(fit_results) + 1]] <- run_mdu(
    data_subset = quadrant_data,
    analysis_name = paste(
      quadrant_name,
      "optimised logos"
    ),
    output_directory = file.path(
      OUT,
      "quadrants",
      quadrant_name
    )
  )
}


# -------------------------------------------------------------------------
# Save fit summary
# -------------------------------------------------------------------------

fit_summary <- do.call(
  rbind,
  fit_results
)

write.csv(
  fit_summary,
  file.path(
    OUT,
    "mdu_fit_summary.csv"
  ),
  row.names = FALSE
)



# -------------------------------------------------------------------------
# 3. Combined quadrant figure - Procrustes-aligned persona + emotion maps
# -------------------------------------------------------------------------
#
# Each quadrant MDU is fitted independently, so its axes can be arbitrarily
# rotated or reflected. Before comparing the four solutions visually, align
# the 10 common emotion-term coordinates to one reference orientation using
# an orthogonal Procrustes transformation.
#
# IMPORTANT: unfolding has TWO configurations in the same space:
#   - row configuration = personas
#   - column configuration = emotion terms
#
# The exact SAME translation and rotation/reflection must therefore be applied
# to the persona coordinates as to the emotion coordinates. Otherwise the
# combined plot would no longer represent the fitted unfolding geometry.
#
# No scaling is applied. Translation + orthogonal rotation/reflection preserve
# all within-panel Euclidean distances.

orthogonal_procrustes <- function(configuration, reference) {

  configuration <- as.matrix(configuration)
  reference <- as.matrix(reference)

  if (!all(dim(configuration) == dim(reference))) {
    stop("Procrustes configurations must have the same dimensions.")
  }

  configuration_center <- colMeans(configuration)
  reference_center <- colMeans(reference)

  configuration_centred <- sweep(
    configuration,
    2,
    configuration_center,
    "-"
  )

  reference_centred <- sweep(
    reference,
    2,
    reference_center,
    "-"
  )

  sv <- svd(
    t(configuration_centred) %*% reference_centred
  )

  rotation <- sv$u %*% t(sv$v)

  aligned <- configuration_centred %*% rotation

  list(
    coordinates = aligned,
    rotation = rotation,
    configuration_center = configuration_center,
    reference_center = reference_center
  )
}


# Apply an already-estimated Procrustes transformation to another set of
# coordinates from the SAME unfolding solution (here: the personas).
apply_procrustes <- function(coordinates, center, rotation) {

  coordinates <- as.matrix(coordinates)

  sweep(
    coordinates,
    2,
    center,
    "-"
  ) %*% rotation
}


quadrant_labels <- c(
  HV_HA = "High valence / high arousal",
  HV_LA = "High valence / low arousal",
  LV_HA = "Low valence / high arousal",
  LV_LA = "Low valence / low arousal"
)


# HV_HA only fixes the arbitrary visual orientation. It is not treated as a
# statistical baseline and does not alter any within-quadrant distances.
reference_quadrant <- "HV_HA"


reference_emotions <- read.csv(
  file.path(
    OUT,
    "quadrants",
    reference_quadrant,
    "mdu_emotion_coords.csv"
  ),
  stringsAsFactors = FALSE
)


# Force identical emotion-term order before matching configurations.
reference_emotions <- reference_emotions[
  match(emotion_labels, reference_emotions$term),
]


if (anyNA(reference_emotions$term)) {
  stop("Reference quadrant is missing one or more emotion terms.")
}


reference_matrix <- as.matrix(
  reference_emotions[, c("dim1", "dim2")]
)


# The reference configuration is translated to its emotion centroid.
reference_center <- colMeans(reference_matrix)

reference_matrix_centred <- sweep(
  reference_matrix,
  2,
  reference_center,
  "-"
)


aligned_emotion_list <- list()
aligned_persona_list <- list()


for (quadrant_name in quadrant_order) {

  # -----------------------------------------------------------------------
  # Read both configurations from the same fitted quadrant solution
  # -----------------------------------------------------------------------

  emotion_coordinates <- read.csv(
    file.path(
      OUT,
      "quadrants",
      quadrant_name,
      "mdu_emotion_coords.csv"
    ),
    stringsAsFactors = FALSE
  )

  persona_coordinates <- read.csv(
    file.path(
      OUT,
      "quadrants",
      quadrant_name,
      "mdu_persona_coords.csv"
    ),
    stringsAsFactors = FALSE
  )


  emotion_coordinates <- emotion_coordinates[
    match(emotion_labels, emotion_coordinates$term),
  ]


  if (anyNA(emotion_coordinates$term)) {
    stop(
      paste(
        "Quadrant",
        quadrant_name,
        "is missing one or more emotion terms."
      )
    )
  }


  emotion_matrix <- as.matrix(
    emotion_coordinates[, c("dim1", "dim2")]
  )

  persona_matrix <- as.matrix(
    persona_coordinates[, c("dim1", "dim2")]
  )


  # -----------------------------------------------------------------------
  # Estimate transformation from the common emotion terms
  # -----------------------------------------------------------------------

  if (quadrant_name == reference_quadrant) {

    configuration_center <- colMeans(emotion_matrix)
    rotation <- diag(2)

    aligned_emotions_matrix <- sweep(
      emotion_matrix,
      2,
      configuration_center,
      "-"
    )

  } else {

    transformation <- orthogonal_procrustes(
      emotion_matrix,
      reference_matrix_centred
    )

    configuration_center <- transformation$configuration_center
    rotation <- transformation$rotation
    aligned_emotions_matrix <- transformation$coordinates
  }


  # -----------------------------------------------------------------------
  # Apply EXACTLY the same transformation to the persona configuration
  # -----------------------------------------------------------------------

  aligned_personas_matrix <- apply_procrustes(
    persona_matrix,
    configuration_center,
    rotation
  )


  # -----------------------------------------------------------------------
  # Panel label
  # -----------------------------------------------------------------------

  stress_value <- fit_summary$stress_1[
    fit_summary$analysis == paste(
      quadrant_name,
      "optimised logos"
    )
  ]


  if (length(stress_value) != 1) {
    stop(
      paste(
        "Could not uniquely recover Stress-1 for",
        quadrant_name
      )
    )
  }


  panel_label <- paste0(
    quadrant_labels[[quadrant_name]],
    "\nStress-1 = ",
    sprintf("%.3f", stress_value)
  )


  # -----------------------------------------------------------------------
  # Store aligned coordinates
  # -----------------------------------------------------------------------

  aligned_emotion_list[[quadrant_name]] <- data.frame(
    dim1 = aligned_emotions_matrix[, 1],
    dim2 = aligned_emotions_matrix[, 2],
    term = emotion_labels,
    quadrant = quadrant_name,
    panel = panel_label,
    point_type = "Emotion term",
    stringsAsFactors = FALSE
  )


  aligned_persona_list[[quadrant_name]] <- data.frame(
    dim1 = aligned_personas_matrix[, 1],
    dim2 = aligned_personas_matrix[, 2],
    id = persona_coordinates$id,
    quadrant = quadrant_name,
    panel = panel_label,
    point_type = "OCEAN persona",
    stringsAsFactors = FALSE
  )
}


aligned_emotions <- do.call(
  rbind,
  aligned_emotion_list
)

aligned_personas <- do.call(
  rbind,
  aligned_persona_list
)


# -------------------------------------------------------------------------
# Fix facet ordering to a meaningful 2 x 2 quadrant arrangement
# -------------------------------------------------------------------------

panel_levels <- vapply(
  quadrant_order,
  function(q) {

    stress_value <- fit_summary$stress_1[
      fit_summary$analysis == paste(
        q,
        "optimised logos"
      )
    ]

    paste0(
      quadrant_labels[[q]],
      "\nStress-1 = ",
      sprintf("%.3f", stress_value)
    )
  },
  character(1)
)


aligned_emotions$panel <- factor(
  aligned_emotions$panel,
  levels = panel_levels
)

aligned_personas$panel <- factor(
  aligned_personas$panel,
  levels = panel_levels
)


# -------------------------------------------------------------------------
# Shared plot limits
# -------------------------------------------------------------------------
#
# Limits include BOTH personas and emotion terms. This is essential because
# the row configuration is part of the unfolding solution, not decoration.

padded_range <- function(x, proportion = 0.10) {

  r <- range(x, finite = TRUE)
  span <- diff(r)

  if (!is.finite(span) || span == 0) {
    span <- 1
  }

  r + c(-1, 1) * span * proportion
}


all_dim1 <- c(
  aligned_emotions$dim1,
  aligned_personas$dim1
)

all_dim2 <- c(
  aligned_emotions$dim2,
  aligned_personas$dim2
)

x_limits <- padded_range(all_dim1)
y_limits <- padded_range(all_dim2)


# -------------------------------------------------------------------------
# Combined persona + emotion unfolding figure
# -------------------------------------------------------------------------

aligned_quadrant_plot <- ggplot() +

  geom_hline(
    yintercept = 0,
    colour = "grey88",
    linewidth = 0.35
  ) +

  geom_vline(
    xintercept = 0,
    colour = "grey88",
    linewidth = 0.35
  ) +

  # Persona configuration: all 32 OCEAN agents.
  geom_point(
    data = aligned_personas,
    aes(
      x = dim1,
      y = dim2,
      shape = point_type
    ),
    colour = "grey68",
    size = 1.7,
    alpha = 0.78
  ) +

  # Emotion-term configuration.
  geom_point(
    data = aligned_emotions,
    aes(
      x = dim1,
      y = dim2,
      shape = point_type
    ),
    colour = "firebrick",
    size = 3
  ) +

  geom_text_repel(
    data = aligned_emotions,
    aes(
      x = dim1,
      y = dim2,
      label = term
    ),
    colour = "firebrick",
    fontface = "bold",
    size = 4.0,
    box.padding = 0.55,
    point.padding = 0.25,
    segment.alpha = 0.45,
    max.overlaps = Inf,
    seed = 1
  ) +

  facet_wrap(
    ~ panel,
    ncol = 2
  ) +

  coord_equal(
    xlim = x_limits,
    ylim = y_limits,
    clip = "off"
  ) +

  scale_shape_manual(
    name = NULL,
    values = c(
      "OCEAN persona" = 16,
      "Emotion term" = 16
    ),
    breaks = c(
      "OCEAN persona",
      "Emotion term"
    )
  ) +

  guides(
    shape = guide_legend(
      override.aes = list(
        colour = c("grey68", "firebrick"),
        size = c(2.2, 3)
      )
    )
  ) +

  labs(
    title = "Aligned persona–emotion unfolding configurations by intended quadrant",
    subtitle = paste(
      "Interval unfolding of optimised sonic logos;",
      "independent solutions aligned by orthogonal Procrustes rotation/reflection"
    ),
    x = "Aligned Dimension 1",
    y = "Aligned Dimension 2",
    caption = paste(
      "Grey points = 32 OCEAN persona configurations;",
      "red labelled points = emotion-term configurations."
    )
  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 11,
      lineheight = 1.05
    ),
    plot.title = element_text(
      size = 17,
      margin = margin(b = 5)
    ),
    plot.subtitle = element_text(
      size = 11,
      margin = margin(b = 12)
    ),
    plot.caption = element_text(
      size = 9.5,
      colour = "grey45",
      hjust = 0
    ),
    plot.title.position = "plot",
    legend.position = "top",
    legend.justification = "left",
    panel.spacing = unit(1.0, "lines"),
    plot.margin = margin(12, 18, 12, 12)
  )


ggsave(
  filename = file.path(
    OUT,
    "mdu_persona_emotion_by_quadrant_aligned.png"
  ),
  plot = aligned_quadrant_plot,
  width = 11,
  height = 9.4,
  dpi = 300
)


# -------------------------------------------------------------------------
# Save aligned coordinates for audit/reproduction
# -------------------------------------------------------------------------

write.csv(
  aligned_emotions,
  file.path(
    OUT,
    "mdu_emotion_coords_by_quadrant_aligned.csv"
  ),
  row.names = FALSE
)

write.csv(
  aligned_personas,
  file.path(
    OUT,
    "mdu_persona_coords_by_quadrant_aligned.csv"
  ),
  row.names = FALSE
)


cat("\n============================================================\n")
cat("All MDU analyses completed.\n")
cat("Outputs saved to:", OUT, "\n\n")
print(fit_summary)