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

cat("\n============================================================\n")
cat("All MDU analyses completed.\n")
cat("Outputs saved to:", OUT, "\n\n")
print(fit_summary)