# Why the MDU uses interval rather than ordinal transformation.
#
# Ordinal (non-metric) unfolding is prone to degenerate solutions: the algorithm
# reaches near-zero stress by collapsing the row points into one tight cluster
# and the column points into another, so every row-column distance becomes
# almost identical. Stress looks excellent and the configuration means nothing
# (Busing, Groenen & Heiser, 2005).
#
# smacof guards against this by default. unfolding() minimises a PENALISED
# stress whose penalty term punishes collapsed disparities, controlled by
# `omega` (default 1). Running type = "ordinal" with the default penalty may
# therefore look perfectly healthy and demonstrate nothing. To show the
# underlying behaviour the penalty has to be weakened.
#
# This script fits the same matrix under several transformations and penalty
# settings and reports degeneracy diagnostics for each, so the choice of
# interval unfolding rests on evidence from this data rather than on assertion.
#
#   Rscript analysis/Check_mdu_degeneracy.R
#
# Outputs:
#   analysis/mdu/degeneracy_check.txt
#   analysis/mdu/degeneracy_comparison.csv
#   analysis/mdu/degeneracy_configurations.png

library(smacof)
library(ggplot2)

OUT <- "analysis/mdu"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
sink(file.path(OUT, "degeneracy_check.txt"), split = TRUE)

emo_cols <- paste0("Q", 3:12)
emotion_labels <- c("happy", "tense", "calm", "sad", "excited",
                    "afraid", "content", "bored", "relaxed", "angry")

d <- read.csv("data/audience/responses.csv", stringsAsFactors = FALSE)
ocean <- subset(d, agent_kind == "ocean" & condition == "optimised")
if (nrow(ocean) == 0) stop("No optimised OCEAN responses found.")

agg <- aggregate(ocean[emo_cols], by = list(persona = ocean$persona_id),
                 FUN = mean, na.rm = TRUE)
M <- as.matrix(agg[emo_cols])
rownames(M) <- agg$persona
colnames(M) <- emotion_labels
D <- 5 - M                      # high rating -> short unfolding distance

cat("MDU TRANSFORMATION AND DEGENERACY CHECK\n")
cat(strrep("=", 72), "\n")
cat(sprintf("matrix: %d personas x %d emotion terms, pooled over optimised logos\n",
            nrow(M), ncol(M)))
cat(sprintf("input dissimilarities: min %.3f, max %.3f, SD %.3f\n\n",
            min(D), max(D), sd(as.vector(D))))

# ---------------------------------------------------------------------------
# Degeneracy diagnostics.
#
# A degenerate unfolding solution is not identified by stress alone - low stress
# is exactly what it produces. It is identified by the configuration collapsing,
# which shows up as near-constant fitted distances and near-constant disparities.
# ---------------------------------------------------------------------------
cv <- function(x) {
  x <- as.vector(x)
  x <- x[is.finite(x)]
  if (!length(x) || mean(x) == 0) return(NA_real_)
  sd(x) / mean(x)
}

spread <- function(cfg) mean(sqrt(rowSums(scale(cfg, scale = FALSE)^2)))

diagnose <- function(fit) {
  # Component names have varied across smacof versions, so take whichever is
  # present rather than assuming.
  dh <- if (!is.null(fit$dhat)) fit$dhat else fit$obsdiss
  cd <- fit$confdist
  list(
    stress        = fit$stress,
    cv_dhat       = cv(dh),
    cv_dist       = cv(cd),
    n_distinct    = length(unique(round(as.vector(dh), 4))),
    spread_row    = spread(fit$conf.row),
    spread_col    = spread(fit$conf.col),
    dist_min      = min(as.vector(cd)),
    dist_max      = max(as.vector(cd))
  )
}

# ---------------------------------------------------------------------------
# The configurations to compare.
#
# omega controls the strength of the anti-degeneracy penalty: 1 is the smacof
# default, 0 removes it entirely and leaves plain stress minimisation.
# ---------------------------------------------------------------------------
specs <- list(
  list(name = "interval (used in this study)", type = "interval", omega = 1),
  list(name = "ratio",                         type = "ratio",    omega = 1),
  list(name = "ordinal, default penalty",      type = "ordinal",  omega = 1),
  list(name = "ordinal, weak penalty",         type = "ordinal",  omega = 0.1),
  list(name = "ordinal, no penalty",           type = "ordinal",  omega = 0)
)

rows <- list()
fits <- list()

for (sp in specs) {
  set.seed(42)
  fit <- tryCatch(
    unfolding(D, type = sp$type, omega = sp$omega),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    cat(sprintf("-- %-30s FAILED: %s\n", sp$name, conditionMessage(fit)))
    rows[[length(rows) + 1]] <- data.frame(
      configuration = sp$name, type = sp$type, omega = sp$omega,
      stress_1 = NA, cv_disparities = NA, cv_distances = NA,
      distinct_disparities = NA, spread_personas = NA, spread_emotions = NA,
      verdict = "did not converge"
    )
    next
  }

  g <- diagnose(fit)
  fits[[sp$name]] <- fit

  # A solution is called degenerate here when the fitted distances have almost
  # no variation: every persona sits the same distance from every emotion term,
  # which is what a collapsed configuration produces.
  verdict <- if (!is.finite(g$stress)) {
    "stress not computable"
  } else if (is.finite(g$cv_dist) && g$cv_dist < 0.05) {
    "DEGENERATE - distances collapsed"
  } else if (g$stress < 0.01 && is.finite(g$cv_dist) && g$cv_dist < 0.15) {
    "DEGENERATE - near-zero stress with little distance variation"
  } else if (g$stress < 0.01) {
    "suspicious - near-zero stress, inspect the configuration"
  } else {
    "interpretable"
  }

  cat(sprintf("-- %s  (type = %s, omega = %.2f)\n", sp$name, sp$type, sp$omega))
  cat(sprintf("     Stress-1                      %.5f\n", g$stress))
  cat(sprintf("     CV of fitted distances        %.4f   <- near 0 means collapsed\n", g$cv_dist))
  cat(sprintf("     CV of disparities             %.4f\n", g$cv_dhat))
  cat(sprintf("     distinct disparity values     %d of %d cells\n", g$n_distinct, length(D)))
  cat(sprintf("     fitted distance range         %.4f .. %.4f\n", g$dist_min, g$dist_max))
  cat(sprintf("     mean spread: personas %.3f, emotions %.3f\n", g$spread_row, g$spread_col))
  cat(sprintf("     verdict: %s\n\n", verdict))

  rows[[length(rows) + 1]] <- data.frame(
    configuration = sp$name, type = sp$type, omega = sp$omega,
    stress_1 = round(g$stress, 5),
    cv_disparities = round(g$cv_dhat, 4),
    cv_distances = round(g$cv_dist, 4),
    distinct_disparities = g$n_distinct,
    spread_personas = round(g$spread_row, 3),
    spread_emotions = round(g$spread_col, 3),
    verdict = verdict
  )
}

comparison <- do.call(rbind, rows)
write.csv(comparison, file.path(OUT, "degeneracy_comparison.csv"), row.names = FALSE)

cat(strrep("-", 72), "\n")
cat("READING THIS TABLE\n\n")
cat("Stress-1 alone cannot separate a good solution from a degenerate one: a\n")
cat("degenerate solution achieves LOW stress precisely by collapsing. The CV of\n")
cat("the fitted distances is what distinguishes them. When it approaches zero,\n")
cat("every persona sits at almost the same distance from every emotion term, so\n")
cat("the configuration carries no information whatever its stress value.\n\n")
cat("smacof minimises a penalised stress (Busing, Groenen & Heiser, 2005) whose\n")
cat("penalty exists to prevent this. omega sets its strength; omega = 0 removes\n")
cat("it. The ordinal rows below therefore show what the transformation does with\n")
cat("and without that protection.\n\n")
print(comparison[, c("configuration", "stress_1", "cv_distances", "verdict")],
      row.names = FALSE)

# ---------------------------------------------------------------------------
# Side-by-side configurations: a collapsed solution is obvious on sight.
# ---------------------------------------------------------------------------
plot_rows <- list()
for (nm in names(fits)) {
  f <- fits[[nm]]
  plot_rows[[length(plot_rows) + 1]] <- rbind(
    data.frame(dim1 = f$conf.row[, 1], dim2 = f$conf.row[, 2],
               set = "persona", label = "", configuration = nm),
    data.frame(dim1 = f$conf.col[, 1], dim2 = f$conf.col[, 2],
               set = "emotion", label = rownames(f$conf.col), configuration = nm)
  )
}

if (length(plot_rows)) {
  pd <- do.call(rbind, plot_rows)
  pd$configuration <- factor(pd$configuration, levels = names(fits))

  p <- ggplot(pd, aes(dim1, dim2, colour = set)) +
    geom_point(aes(size = set), alpha = 0.85) +
    geom_text(aes(label = label), size = 3.0, vjust = -0.9, show.legend = FALSE) +
    facet_wrap(~ configuration, scales = "free", ncol = 2) +
    scale_colour_manual(values = c(persona = "grey60", emotion = "firebrick")) +
    scale_size_manual(values = c(persona = 1.2, emotion = 2.4), guide = "none") +
    labs(title = "MDU configurations under different transformations",
         subtitle = paste("A degenerate solution collapses each set into a tight",
                          "cluster or ring, leaving all cross-set distances equal."),
         x = "Dimension 1", y = "Dimension 2", colour = NULL) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          legend.position = "bottom",
          plot.title.position = "plot")

  # Sized for a portrait page: two panels across, as many rows as needed.
  n_rows <- ceiling(length(fits) / 2)
  ggsave(file.path(OUT, "degeneracy_configurations.png"), p,
         width = 9, height = 4.6 * n_rows, dpi = 300, limitsize = FALSE)
}

sink()
cat(sprintf("\nSaved: %s/degeneracy_check.txt\n", OUT))
cat(sprintf("       %s/degeneracy_comparison.csv\n", OUT))
cat(sprintf("       %s/degeneracy_configurations.png\n", OUT))