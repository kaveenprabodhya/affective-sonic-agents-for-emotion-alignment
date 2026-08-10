# Quadrant confidence, per judge.
#
# There is no ground truth for the emotion a synthetic logo carries, so a judge
# cannot be scored against truth. What can be measured is whether it places a
# stimulus on the same side of both axes as the brief target, and how close to a
# boundary that placement sits.
#
#   held     - same sign as the target on both axes, and both coordinates at
#              least `margin` clear of their axis
#   marginal - same sign, but at least one coordinate within `margin` of an axis,
#              so the classification would flip under a small error
#   crossed  - opposite sign on at least one axis
#
# The margin is not a property of the data, so the table is reported across
# several values rather than one. If the held/marginal split moves sharply
# between margins, the quadrant result is resting on near-boundary calls.
#
#   Rscript analysis/Check_quadrant_confidence.R
#   Rscript analysis/Check_quadrant_confidence.R 0.02 0.05 0.10 0.15

args    <- commandArgs(trailingOnly = TRUE)
margins <- if (length(args)) as.numeric(args) else c(0.00, 0.02, 0.05, 0.10)

OUT <- "analysis/h1"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
sink(file.path(OUT, "quadrant_confidence.txt"), split = TRUE)

files <- list.files("data/analysis", pattern = "^h1_estimator_b.*\\.csv$", full.names = TRUE)
if (!length(files)) stop("No h1_estimator_b*.csv in data/analysis/. Run score_estimator_b.py first.")

judge_of <- function(f) {
  b <- sub("\\.csv$", "", basename(f))
  if (b == "h1_estimator_b") "estimator_B" else sub("^h1_estimator_b_", "", b)
}

classify <- function(pv, pa, tv, ta, margin) {
  same <- sign(pv) == sign(tv) & sign(pa) == sign(ta)
  near <- abs(pv) < margin | abs(pa) < margin
  ifelse(!same, "crossed", ifelse(near, "marginal", "held"))
}

cat("QUADRANT CONFIDENCE\n")
cat(strrep("=", 72), "\n")
cat("held     = same sign on both axes, both coordinates clear of the axes by >= margin\n")
cat("marginal = same sign, but within margin of an axis (would flip under small error)\n")
cat("crossed  = opposite sign on at least one axis\n\n")

store <- list()

for (f in files) {
  j  <- judge_of(f)
  d  <- read.csv(f, stringsAsFactors = FALSE)
  store[[j]] <- d

  cat(sprintf("-- %s  (n = %d pairs) --\n", j, nrow(d)))
  cat(sprintf("   spread on these stimuli: valence SD %.4f, arousal SD %.4f\n",
              sd(c(d$nonopt_B_v, d$opt_B_v)), sd(c(d$nonopt_B_a, d$opt_B_a))))
  cat(sprintf("   %-8s %8s %10s %9s   %s\n", "margin", "held", "marginal", "crossed", "held %"))
  for (m in margins) {
    cls <- classify(d$opt_B_v, d$opt_B_a, d$target_v, d$target_a, m)
    cat(sprintf("   %-8.2f %8d %10d %9d   %5.1f%%\n",
                m, sum(cls == "held"), sum(cls == "marginal"),
                sum(cls == "crossed"), 100 * mean(cls == "held")))
  }

  # non-optimised for comparison: iteration 0, never subject to the stopping rule
  cls0 <- classify(d$nonopt_B_v, d$nonopt_B_a, d$target_v, d$target_a, 0)
  cls1 <- classify(d$opt_B_v,    d$opt_B_a,    d$target_v, d$target_a, 0)
  cat(sprintf("   sign only: non-optimised %d/%d held -> optimised %d/%d held\n\n",
              sum(cls0 == "held"), nrow(d), sum(cls1 == "held"), nrow(d)))

  # which quadrants fail
  d$cls <- cls1
  tab <- table(d$quadrant, d$cls)
  cat("   by intended quadrant (sign only):\n")
  print(tab)
  cat("\n")
}

# ---- do the judges agree with each other? --------------------------------
if (length(store) > 1) {
  cat(strrep("-", 72), "\n")
  cat("AGREEMENT BETWEEN JUDGES (optimised stimuli, quadrant of the prediction)\n\n")
  quad <- function(d) paste0(ifelse(d$opt_B_v >= 0, "H", "L"), "V_",
                             ifelse(d$opt_B_a >= 0, "H", "L"), "A")

  # Cohen's kappa. Raw agreement is not interpretable on its own here: an
  # estimator that puts almost everything in one quadrant will agree with any
  # other estimator at a rate set by that concentration alone. kappa subtracts
  # the agreement expected from the two marginal distributions.
  #   kappa = (po - pe) / (1 - pe)
  # It is undefined when pe = 1, which happens when both estimators are constant
  # and identical - in that case the raw agreement carries no information either.
  kappa2 <- function(x, y) {
    n <- length(x)
    cats <- union(unique(x), unique(y))
    po <- mean(x == y)
    px <- as.numeric(table(factor(x, levels = cats))) / n
    py <- as.numeric(table(factor(y, levels = cats))) / n
    pe <- sum(px * py)
    list(po = po, pe = pe, k = if (pe < 1) (po - pe) / (1 - pe) else NA_real_)
  }

  cat("quadrant of each estimator's prediction, across the 48 optimised stimuli:\n")
  for (n in names(store)) {
    q <- quad(store[[n]])
    tb <- table(factor(q, levels = c("HV_HA", "HV_LA", "LV_HA", "LV_LA")))
    cat(sprintf("  %-14s %s\n", n,
                paste(sprintf("%s %d", names(tb), as.integer(tb)), collapse = "   ")))
  }
  cat("\nAn estimator concentrated in one quadrant inflates raw agreement with\n")
  cat("everything else, which is why kappa is reported alongside it.\n\n")

  cat(sprintf("%-30s %8s %8s %8s\n", "pair", "raw", "chance", "kappa"))
  ns <- names(store)
  for (i in seq_along(ns)) for (k in seq_along(ns)) {
    if (k <= i) next
    a <- store[[ns[i]]]; b <- store[[ns[k]]]
    m  <- match(paste(a$brief, a$run), paste(b$brief, b$run))
    ok <- !is.na(m)
    r  <- kappa2(quad(a)[ok], quad(b)[m[ok]])
    cat(sprintf("%-30s %7.1f%% %7.1f%% %8s\n",
                paste(ns[i], "vs", ns[k]), 100 * r$po, 100 * r$pe,
                if (is.na(r$k)) "n/a" else sprintf("%+.3f", r$k)))
  }

  cat("\nkappa near 0 means the estimators agree no more than their own marginal\n")
  cat("distributions would produce by chance. Negative means less than chance.\n")
  cat("The usual reference bands (Landis & Koch, 1977) are conventions rather than\n")
  cat("thresholds: <=0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,\n")
  cat("0.61-0.80 substantial, >0.80 almost perfect.\n\n")
  cat("Low agreement means the estimators disagree about where these stimuli sit,\n")
  cat("not that one of them is wrong. Nothing here establishes which is closer to\n")
  cat("what a listener would report. A pair involving an estimator with almost no\n")
  cat("spread should be read as a discrimination failure, not as disagreement.\n")
}

sink()
cat(sprintf("\nSaved: %s/quadrant_confidence.txt\n", OUT))