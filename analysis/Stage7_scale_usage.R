# stage7_scale_usage.R
# ---------------------------------------------------------------------------
# Descriptive evidence for the response-scale paragraph in Section 4.3.1.
# Reports how the synthetic respondents used the Q1 and Q2 nine-point scales,
# and where perceived positions fell relative to intended quadrants.
#
# Descriptive only. Responses are nested (persona x stimulus x repetition),
# so no inferential test is run here.
#
# Run from the project root:  Rscript stage7_scale_usage.R
# ---------------------------------------------------------------------------

responses_csv <- "data/audience/responses.csv"
out_dir       <- "analysis/stage7_scale_usage"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

d <- read.csv(responses_csv, stringsAsFactors = FALSE)

# persona responses only; the two controls are described separately
if (!"agent_kind" %in% names(d)) stop("agent_kind column not found")
cat("agent_kind values:", paste(unique(d$agent_kind), collapse = ", "), "\n\n")
p <- d[grepl("ocean", tolower(d$agent_kind)), ]
cat("persona responses:", nrow(p), "\n\n")

# --- 1. scale-point distribution -------------------------------------------
scale_table <- function(x, label) {
  tab <- table(factor(x, levels = 1:9))
  data.frame(item      = label,
             point     = as.integer(names(tab)),
             n         = as.integer(tab),
             percent   = round(100 * as.integer(tab) / length(x), 2))
}

dist_q <- rbind(scale_table(p$Q1, "Q1 (valence)"),
                scale_table(p$Q2, "Q2 (arousal)"))
write.csv(dist_q, file.path(out_dir, "scale_point_distribution.csv"), row.names = FALSE)

cat("--- scale-point distribution ---------------------------------------\n")
print(dist_q[dist_q$n > 0, ], row.names = FALSE)

# --- 2. the two claims in the paragraph -------------------------------------
pct <- function(x) round(100 * mean(x), 1)

at_or_above <- c(Q1 = pct(p$Q1 >= 5), Q2 = pct(p$Q2 >= 5))
odd_579     <- c(Q1 = pct(p$Q1 %in% c(5, 7, 9)), Q2 = pct(p$Q2 %in% c(5, 7, 9)))

cat("\n--- claims 1 and 2 -------------------------------------------------\n")
cat(sprintf("at or above midpoint (>= 5):   Q1 %.1f%%   Q2 %.1f%%\n",
            at_or_above["Q1"], at_or_above["Q2"]))
cat(sprintf("values 5, 7 or 9:              Q1 %.1f%%   Q2 %.1f%%\n",
            odd_579["Q1"], odd_579["Q2"]))

write.csv(data.frame(item = c("Q1", "Q2"),
                     pct_at_or_above_midpoint = as.numeric(at_or_above),
                     pct_values_5_7_9         = as.numeric(odd_579)),
          file.path(out_dir, "scale_usage_summary.csv"), row.names = FALSE)

# --- 3. intended vs perceived quadrant --------------------------------------
quad <- function(v, a) {
  ifelse(
    v == 0 | a == 0,
    "BOUNDARY",
    paste0(
      ifelse(v > 0, "HV", "LV"),
      ifelse(a > 0, "_HA", "_LA")
    )
  )
}
p$intended  <- quad(p$target_v,    p$target_a)
p$perceived <- quad(p$perceived_v, p$perceived_a)

ct <- table(intended = p$intended, perceived = p$perceived)
write.csv(as.data.frame.matrix(ct), file.path(out_dir, "quadrant_crosstab.csv"))

cat("\n--- intended (rows) x perceived (columns) --------------------------\n")
print(ct)

row_tot   <- rowSums(ct)
hvha_n    <- if ("HV_HA" %in% colnames(ct)) ct[, "HV_HA"] else rep(0, nrow(ct))
hvha_pct  <- round(100 * hvha_n / row_tot, 1)

cat("\n--- claim 3: share perceived as HV_HA, by intended quadrant --------\n")
print(data.frame(intended = names(row_tot), n_HV_HA = as.integer(hvha_n),
                 total = as.integer(row_tot), percent = as.numeric(hvha_pct)),
      row.names = FALSE)

overall_hvha <- round(100 * mean(p$perceived == "HV_HA"), 1)
cat(sprintf("\noverall perceived HV_HA: %.1f%% of %d responses\n",
            overall_hvha, nrow(p)))

if ("LV_LA" %in% names(row_tot)) {
  cat(sprintf("low-valence low-arousal briefs: %d of %d perceived as HV_HA\n",
              as.integer(hvha_n["LV_LA"]), as.integer(row_tot["LV_LA"])))
}

# --- 4. paragraph, regenerated from the values above ------------------------
cat("\n--- sentence check -------------------------------------------------\n")
cat(sprintf(paste0(
  "Responses were also strongly concentrated in the upper half of both scales.\n",
  "Q1 and Q2 were at or above the midpoint in %.1f%% and %.1f%% of responses\n",
  "respectively, while values 5, 7 and 9 accounted for %.1f%% and %.1f%%.\n",
  "Consequently, perceived positions fell in the high-valence, high-arousal\n",
  "quadrant for about %.0f%% of responses across all intended quadrants,\n",
  "including %d of %d responses to low-valence, low-arousal briefs.\n"),
  at_or_above["Q1"], at_or_above["Q2"], odd_579["Q1"], odd_579["Q2"],
  overall_hvha,
  if ("LV_LA" %in% names(row_tot)) as.integer(hvha_n["LV_LA"]) else NA,
  if ("LV_LA" %in% names(row_tot)) as.integer(row_tot["LV_LA"]) else NA))

cat("\nwrote CSVs to", out_dir, "\n")