# What is duplicated in the baselines pivot?
#
#   Rscript analysis/Diagnose_baselines.R

suppressMessages(library(dplyr))

d <- read.csv("data/audience/responses.csv", stringsAsFactors = FALSE)
cat(sprintf("responses.csv: %d rows, %d columns\n\n", nrow(d), ncol(d)))

cat("-- agent_kind values --\n"); print(table(d$agent_kind))
cat("\n-- rows per (stimulus_file, condition, agent_kind) --\n")
print(table(table(paste(d$stimulus_file, d$condition, d$agent_kind))))
cat("(3 = the three repetitions, as expected)\n")

cat("\n-- distinct targets per stimulus_file --\n")
tp <- d %>%
  distinct(stimulus_file, target_v, target_a) %>%
  count(stimulus_file, name = "n_targets") %>%
  filter(n_targets > 1)
if (nrow(tp) == 0) {
  cat("every stimulus_file has exactly one target: OK\n")
} else {
  cat(sprintf("%d stimulus_file(s) carry more than one target -- this breaks the pivot:\n",
              nrow(tp)))
  print(head(tp, 20))
  cat("\nthe offending rows:\n")
  print(d %>% filter(stimulus_file %in% tp$stimulus_file) %>%
          distinct(stimulus_file, brief, condition, target_v, target_a) %>%
          arrange(stimulus_file) %>% head(20))
}

cat("\n-- distinct (brief, condition) per stimulus_file --\n")
bp <- d %>%
  distinct(stimulus_file, brief, condition) %>%
  count(stimulus_file, name = "n") %>%
  filter(n > 1)
if (nrow(bp) == 0) cat("every stimulus_file maps to one brief and condition: OK\n") else print(head(bp, 20))

cat("\n-- condition values --\n"); print(table(d$condition))
cat("\n-- reps per respondent-stimulus --\n")
print(table(table(paste(d$stimulus_file, d$persona_id, d$agent_kind))))