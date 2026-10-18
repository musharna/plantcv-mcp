# Figures and statistics for the agent benchmark, from results.jsonl and
# baselines.jsonl directly.
#
#   Rscript benchmarks/agent_qa/plot.R [results.jsonl] [outdir]
#
# Writes
#   summary.tsv   verdicts, validity, cost, time and calls per setup (the two
#                 no-model controls included), and per setup x kind of question
#   stats.tsv     bootstrap 95% intervals over questions, and the paired
#                 mcp-vs-alt test, on all questions and on the questions the
#                 colour-cut control gets wrong
#   agent_qa.png                right / wrong / declined / invalid per setup
#   agent_qa_accuracy_ci.png    share right with its 95% interval per setup
#   agent_qa_cost_time.png      cost and seconds per answer, every run
#   agent_qa_by_category.png    verdicts per setup x kind of question
# Every number is computed here from the logs; nothing is typed in by hand.
#
# Validity is decided per run by run.py (assess(): one rule for every setup).
# Invalid runs are drawn and counted as invalid, and left out of every share,
# interval and test.

library(ggplot2)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
results_path <- if (length(args) >= 1) args[1] else "benchmarks/agent_qa/results.jsonl"
outdir <- if (length(args) >= 2) args[2] else "benchmarks/agent_qa"
baselines_path <- file.path(dirname(results_path), "baselines.jsonl")
source("benchmarks/theme_pcvmcp.R")
set.seed(20261016)
n_boot <- 4000

r <- jsonlite::stream_in(file(results_path), verbose = FALSE)
b <- jsonlite::stream_in(file(baselines_path), verbose = FALSE)
stopifnot(all(c(r$verdict, b$verdict) %in% names(pal_verdict)))
stopifnot(is.logical(r$valid), !anyNA(r$valid))
# The plotted verdict: a run's score, or "invalid" if it did not test its setup.
r$shown <- ifelse(r$valid, r$verdict, "invalid")
b$shown <- b$verdict

arm_labels <- c(
  baseline_constant = "No model: median answer of its kind",
  baseline_threshold = "No model: one colour cut",
  bare = "Claude alone",
  alt = "Claude + Bash/PlantCV/OpenCV + web",
  mcp = "Claude + plantcv-mcp + Read"
)
stopifnot(all(c(r$arm, b$arm) %in% names(arm_labels)))
model_arms <- intersect(c("bare", "alt", "mcp"), unique(r$arm))
all_arms <- c("baseline_constant", "baseline_threshold", model_arms)

cat_labels <- c(
  area = "Plant area (px)",
  extent = "Plant height or width (px)",
  area_mm = "Plant area (mm², from a size marker)",
  tray = "Tray: plants present, or one plant's area",
  leaf_count = "Leaf count",
  color = "Colour: mean hue, or yellow share"
)
stopifnot(all(r$category %in% names(cat_labels)))

n_q <- length(unique(b$qid))
n_rep <- max(r$rep)
stopifnot(all(table(r$arm, r$qid) == n_rep)) # every setup ran every question n_rep times
# Questions a one-colour cut does not answer: the subset where image analysis
# is needed. Fixed by the no-model control, not by any model's results.
hard <- b$qid[b$arm == "baseline_threshold" & b$verdict != "right"]

cols <- c("arm", "qid", "category", "rep", "shown", "verdict", "valid")
d <- rbind(
  r[, cols],
  data.frame(b[, c("arm", "qid", "category", "rep", "shown", "verdict")], valid = TRUE)
)
d$arm_label <- factor(arm_labels[d$arm], levels = rev(arm_labels[all_arms]))
d$shown <- factor(d$shown, levels = names(pal_verdict))

# --- tables ---------------------------------------------------------------

# Share right among valid runs, per question first, then averaged: repeats of
# one question are not independent, so the question is the unit everywhere.
per_q_right <- function(x) tapply(x$verdict == "right", x$qid, mean)
boot_ci <- function(qids, stat) {
  if (length(qids) == 0) {
    return(c(est = NA, lo = NA, hi = NA))
  } # a setup with no valid run here
  est <- stat(qids)
  reps <- replicate(n_boot, stat(sample(qids, replace = TRUE)))
  c(est = est, lo = unname(quantile(reps, 0.025, na.rm = TRUE)), hi = unname(quantile(reps, 0.975, na.rm = TRUE)))
}

tally <- function(x) {
  v <- x[x$valid, ]
  data.frame(
    answers = nrow(x), valid = nrow(v), invalid = sum(!x$valid),
    right = sum(v$verdict == "right"), wrong = sum(v$verdict == "wrong"),
    declined = sum(v$verdict == "declined"), no_answer = sum(v$verdict == "no_answer"),
    questions_right_every_run = sum(tapply(x$valid & x$verdict == "right", x$qid, all))
  )
}
num_or_na <- function(x, f, digits) if (length(x)) round(f(x), digits) else NA
run_stats <- function(a) {
  x <- r[r$arm == a, ]
  v <- x[x$valid, ]
  slips <- if ("slip" %in% names(v)) sum(!is.na(v$slip)) else NA
  data.frame(
    slips = slips,
    touched_key = sum(x$touched_key), leak_hits = sum(lengths(x$leaks) > 0),
    cost_usd_mean = num_or_na(v$cost_usd, mean, 4),
    cost_usd_median = num_or_na(v$cost_usd, median, 4),
    cost_usd_total = round(sum(x$cost_usd, na.rm = TRUE), 2),
    seconds_mean = num_or_na(v$seconds, mean, 1),
    seconds_median = num_or_na(v$seconds, median, 1),
    seconds_max = num_or_na(v$seconds, max, 1),
    tool_calls_mean = num_or_na(lengths(v$tool_calls), mean, 2),
    tool_calls_median = num_or_na(lengths(v$tool_calls), median, 1)
  )
}
empty_stats <- run_stats(model_arms[1])[0, ][1, ]
per_arm <- do.call(rbind, lapply(all_arms, function(a) {
  x <- d[d$arm == a, ]
  cbind(
    level = "arm", arm = a, category = "all", tally(x),
    if (a %in% model_arms) run_stats(a) else empty_stats
  )
}))
per_cat <- do.call(rbind, lapply(all_arms, function(a) {
  do.call(rbind, lapply(names(cat_labels), function(k) {
    cbind(level = "arm_x_category", arm = a, category = k, tally(d[d$arm == a & d$category == k, ]), empty_stats)
  }))
}))
invalid_why <- table(r$arm[!r$valid], r$invalid_reason[!r$valid])
# na = "NA", not "": a line ending in tabs is trailing whitespace the
# pre-commit hook strips, which would drop those columns.
write.table(rbind(per_arm, per_cat), file.path(outdir, "summary.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE, na = "NA"
)

# Intervals and the paired test.
stat_rows <- list()
for (subset in c("all", "colour_cut_wrong")) {
  qs <- if (subset == "all") unique(b$qid) else hard
  for (a in all_arms) {
    x <- d[d$arm == a & d$valid & d$qid %in% qs, ]
    pq <- per_q_right(x)
    acc <- boot_ci(names(pq), function(s) mean(pq[s]))
    row <- data.frame(
      subset = subset, arm = a, questions = length(pq),
      share_right = round(acc["est"], 4), share_right_lo = round(acc["lo"], 4), share_right_hi = round(acc["hi"], 4),
      cost_median = NA, cost_median_lo = NA, cost_median_hi = NA,
      seconds_median = NA, seconds_median_lo = NA, seconds_median_hi = NA,
      calls_median = NA, calls_median_lo = NA, calls_median_hi = NA
    )
    if (a %in% model_arms) {
      rr <- r[r$arm == a & r$valid & r$qid %in% qs, ]
      by_q <- split(seq_len(nrow(rr)), rr$qid)
      med <- function(col) {
        function(s) median(unlist(lapply(by_q[s], function(i) col[i])))
      }
      for (m in list(
        c("cost_median", "cost_usd"), c("seconds_median", "seconds"), c("calls_median", "calls")
      )) {
        colv <- if (m[2] == "calls") lengths(rr$tool_calls) else rr[[m[2]]]
        ci <- boot_ci(names(by_q), med(colv))
        row[[m[1]]] <- round(ci["est"], 4)
        row[[paste0(m[1], "_lo")]] <- round(ci["lo"], 4)
        row[[paste0(m[1], "_hi")]] <- round(ci["hi"], 4)
      }
    }
    stat_rows[[length(stat_rows) + 1]] <- row
  }
}
stats <- do.call(rbind, stat_rows)
# Paired, per question: right in every valid run (and at least one valid run).
all_right <- function(a, qs) {
  x <- r[r$arm == a & r$valid & r$qid %in% qs, ]
  tapply(x$verdict == "right", x$qid, all)
}
paired <- do.call(rbind, lapply(c("all", "colour_cut_wrong"), function(subset) {
  qs <- if (subset == "all") unique(b$qid) else hard
  m <- all_right("mcp", qs)
  a <- all_right("alt", qs)
  both <- intersect(names(m), names(a))
  only_mcp <- sum(m[both] & !a[both])
  only_alt <- sum(!m[both] & a[both])
  p <- if (only_mcp + only_alt > 0) binom.test(only_mcp, only_mcp + only_alt, 0.5)$p.value else 1
  data.frame(
    subset = subset, questions_paired = length(both),
    mcp_all_right = sum(m[both]), alt_all_right = sum(a[both]),
    only_mcp = only_mcp, only_alt = only_alt, exact_mcnemar_p = signif(p, 3)
  )
}))
write.table(stats, file.path(outdir, "stats.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
write.table(paired, file.path(outdir, "stats_paired.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")

# --- figures --------------------------------------------------------------

setup_note <- sprintf(
  "Claude Opus 5.5 in Claude Code, %d questions × %d runs per setup; the no-model controls answer once.",
  n_q, n_rep
)
verdict_text <- c(`TRUE` = "grey15", `FALSE` = "white")
light <- c("declined", "invalid")

# Figure 1: one bar per setup, segments = verdict share, labelled with counts.
counts <- as.data.frame(table(arm_label = d$arm_label, shown = d$shown))
counts <- counts[counts$Freq > 0, ]
counts$share <- counts$Freq / ave(counts$Freq, counts$arm_label, FUN = sum)
p1 <- ggplot(counts, aes(x = share, y = arm_label, fill = shown)) +
  geom_col(width = 0.62, position = position_stack(reverse = TRUE)) +
  geom_text(
    aes(label = ifelse(share >= 0.04, Freq, ""), colour = shown %in% light),
    position = position_stack(vjust = 0.5, reverse = TRUE),
    fontface = "bold", size = 4, show.legend = FALSE
  ) +
  scale_colour_manual(values = verdict_text, guide = "none") +
  scale_fill_manual(values = pal_verdict, name = NULL, drop = TRUE) +
  scale_x_continuous(labels = scales::percent, expand = c(0, 0)) +
  labs(
    x = "Share of answers (counts on the bars)", y = NULL,
    title = "Plant image measurements: right, wrong, declined or invalid",
    subtitle = setup_note
  ) +
  theme_pcvmcp(base_size = 13) +
  theme(panel.grid.major.y = element_blank(), plot.margin = margin(8, 24, 8, 8))
ggsave(file.path(outdir, "agent_qa.png"), p1, width = 11.5, height = 4.8, dpi = 200, bg = "white")

# Figure 2: share right with its bootstrap interval, all questions and the
# questions the colour cut gets wrong.
sub_labels <- c(
  all = sprintf("All %d questions", n_q),
  colour_cut_wrong = sprintf("The %d questions a colour cut gets wrong", length(hard))
)
s2 <- stats
s2$arm_label <- factor(arm_labels[s2$arm], levels = rev(arm_labels[all_arms]))
s2$subset_label <- factor(sub_labels[s2$subset], levels = sub_labels)
p2 <- ggplot(s2, aes(x = share_right, y = arm_label)) +
  geom_errorbar(aes(xmin = share_right_lo, xmax = share_right_hi), width = 0.25, orientation = "y") +
  geom_point(
    shape = 21, size = pcvmcp_point_size + 0.8, stroke = pcvmcp_point_stroke,
    fill = pal_verdict[["right"]], colour = pcvmcp_point_outline
  ) +
  facet_wrap(~subset_label, ncol = 2) +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(
    x = "Share right (mean over questions of the share of valid runs right), 95% bootstrap interval over questions",
    y = NULL, title = "Accuracy with its uncertainty",
    subtitle = setup_note
  ) +
  theme_pcvmcp(base_size = 12) +
  theme(panel.grid.major.y = element_blank(), strip.text = element_text(hjust = 0, face = "bold"))
ggsave(file.path(outdir, "agent_qa_accuracy_ci.png"), p2, width = 12, height = 4.6, dpi = 200, bg = "white")

# Figure 3: cost and seconds per answer, one point per valid run.
rv <- r[r$valid, ]
rv$arm_label <- factor(arm_labels[rv$arm], levels = rev(arm_labels[model_arms]))
long <- rbind(
  data.frame(arm_label = rv$arm_label, metric = "Cost per answer (US$, log scale)", value = rv$cost_usd),
  data.frame(arm_label = rv$arm_label, metric = "Seconds per answer (log scale)", value = rv$seconds)
)
p3 <- ggplot(long, aes(x = value, y = arm_label)) +
  geom_boxplot(width = 0.55, outlier.shape = NA, fill = "white", colour = "grey20") +
  geom_point(
    position = position_jitter(height = 0.18, seed = 1),
    shape = 21, size = pcvmcp_point_size - 0.6, stroke = pcvmcp_point_stroke,
    fill = pcvmcp_point_fill, colour = pcvmcp_point_outline, alpha = 0.55
  ) +
  facet_wrap(~metric, ncol = 2, scales = "free_x") +
  scale_x_log10(labels = scales::label_number(drop0trailing = TRUE)) +
  labs(
    x = NULL, y = NULL,
    title = "Cost and time per answer",
    subtitle = paste0(
      setup_note, "\nOne point per valid run; the box spans the middle half, the line in it is the median."
    )
  ) +
  theme_pcvmcp(base_size = 13) +
  theme(
    panel.grid.major.y = element_blank(), strip.text = element_text(hjust = 0, face = "bold"),
    plot.margin = margin(8, 24, 8, 8)
  )
ggsave(file.path(outdir, "agent_qa_cost_time.png"), p3, width = 12, height = 4.2, dpi = 200, bg = "white")

# Figure 4: verdicts per setup x kind of question; strips name the counts.
present <- names(cat_labels)[names(cat_labels) %in% unique(d$category)]
n_cat <- table(b$category[b$arm == "baseline_constant"])
strip <- setNames(sprintf("%s (%d questions)", cat_labels[present], n_cat[present]), present)
d$cat_strip <- factor(strip[d$category], levels = strip)
cc <- as.data.frame(table(arm_label = d$arm_label, cat_strip = d$cat_strip, shown = d$shown))
cc <- cc[cc$Freq > 0, ]
cc$share <- cc$Freq / ave(cc$Freq, cc$arm_label, cc$cat_strip, FUN = sum)
p4 <- ggplot(cc, aes(x = share, y = arm_label, fill = shown)) +
  geom_col(width = 0.62, position = position_stack(reverse = TRUE)) +
  geom_text(
    aes(label = Freq, colour = shown %in% light),
    position = position_stack(vjust = 0.5, reverse = TRUE),
    fontface = "bold", size = 3.2, show.legend = FALSE
  ) +
  scale_colour_manual(values = verdict_text, guide = "none") +
  facet_wrap(~cat_strip, ncol = 1) +
  scale_fill_manual(values = pal_verdict, name = NULL, drop = TRUE) +
  scale_x_continuous(labels = scales::percent, expand = c(0, 0)) +
  labs(
    x = "Share of answers (counts on the bars)", y = NULL,
    title = "Right, wrong, declined or invalid, by kind of question",
    subtitle = setup_note
  ) +
  theme_pcvmcp(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(), strip.text = element_text(hjust = 0, face = "bold"),
    plot.margin = margin(8, 24, 8, 8)
  )
ggsave(file.path(outdir, "agent_qa_by_category.png"), p4, width = 10, height = 13, dpi = 200, bg = "white")

print(per_arm)
print(stats)
print(paired)
print(invalid_why)
