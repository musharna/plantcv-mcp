# Figures for the agent benchmark, from results.jsonl directly.
#
#   Rscript benchmarks/agent_qa/plot.R [results.jsonl] [outdir]
#
# Writes summary.tsv (verdicts, cost and time per setup; verdicts per setup x
# kind of question) and three figures:
#   agent_qa.png             right / wrong / declined per setup (the README one)
#   agent_qa_cost_time.png   cost and seconds per answer, every run
#   agent_qa_by_category.png right / wrong / declined per setup x kind of question
# Bars count answers, not questions: 46 questions x 3 runs = 138 per setup.
# Every number on the figures is computed here from the log; nothing is typed
# in by hand. Green, red and grey mean right, wrong and declined in every
# figure; setups are told apart by their axis labels, not by colour.

library(ggplot2)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
results_path <- if (length(args) >= 1) args[1] else "benchmarks/agent_qa/results.jsonl"
outdir <- if (length(args) >= 2) args[2] else "benchmarks/agent_qa"
source("benchmarks/theme_pcvmcp.R")

r <- jsonlite::stream_in(file(results_path), verbose = FALSE)
stopifnot(all(r$verdict %in% names(pal_verdict)))
stopifnot(all(r$exit == 0))

arm_labels <- c(
  bare = "Claude alone",
  alt = "Claude + Bash/PlantCV/OpenCV + web",
  mcp = "Claude + plantcv-mcp"
)
stopifnot(all(r$arm %in% names(arm_labels)))
arms <- names(arm_labels)[names(arm_labels) %in% unique(r$arm)]
r$arm_label <- factor(arm_labels[r$arm], levels = rev(arm_labels[arms]))
r$verdict <- factor(r$verdict, levels = names(pal_verdict))

cat_labels <- c(
  area = "Plant area (px)",
  extent = "Plant height or width (px)",
  area_mm = "Plant area (mm², from a size marker)",
  tray = "Tray: plants present, or one plant's area",
  leaf_count = "Leaf count",
  color = "Colour: mean hue, or yellow share"
)
stopifnot(all(r$category %in% names(cat_labels)))

n_per_arm <- table(r$arm)
stopifnot(length(unique(n_per_arm)) == 1)
n_ans <- as.numeric(n_per_arm[1])
n_q <- length(unique(r$qid))
n_rep <- length(unique(r$rep))

# Tables first: the figures draw these numbers, and the page quotes them.
tally <- function(x) {
  data.frame(
    right = sum(x$verdict == "right"), wrong = sum(x$verdict == "wrong"),
    declined = sum(x$verdict == "declined"), no_answer = sum(x$verdict == "no_answer"),
    # A wrong answer that matched a named other quantity (score.py: slip).
    slips = sum(!is.na(x$slip)),
    # A tool call that named the answer key or the server's code.
    touched_key = sum(x$touched_key),
    # Repeats of one question are not independent: also count questions.
    questions_right_every_run = sum(tapply(x$verdict == "right", x$qid, all))
  )
}
per_arm <- do.call(rbind, lapply(arms, function(a) {
  x <- r[r$arm == a, ]
  cbind(
    level = "arm", arm = a, category = "all", tally(x),
    cost_usd_mean = round(mean(x$cost_usd), 4),
    cost_usd_median = round(median(x$cost_usd), 4),
    cost_usd_total = round(sum(x$cost_usd), 2),
    seconds_mean = round(mean(x$seconds), 1),
    seconds_median = round(median(x$seconds), 1),
    seconds_max = max(x$seconds),
    tool_calls_mean = round(mean(lengths(x$tool_calls)), 2),
    tool_calls_median = median(lengths(x$tool_calls))
  )
}))
per_cat <- do.call(rbind, lapply(arms, function(a) {
  do.call(rbind, lapply(names(cat_labels), function(k) {
    cbind(
      level = "arm_x_category", arm = a, category = k,
      tally(r[r$arm == a & r$category == k, ]),
      cost_usd_mean = NA, cost_usd_median = NA, cost_usd_total = NA,
      seconds_mean = NA, seconds_median = NA, seconds_max = NA,
      tool_calls_mean = NA, tool_calls_median = NA
    )
  }))
}))
# na = "NA", not "": a line ending in tabs is trailing whitespace the
# pre-commit hook strips, which would drop those columns.
write.table(rbind(per_arm, per_cat), file.path(outdir, "summary.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE, na = "NA"
)

setup_note <- sprintf("Claude Opus 5.5 in Claude Code, %d questions × %d runs.", n_q, n_rep)
key_note <- "Synthetic images: every answer is the pixel count of the shape that was drawn."
# Count labels: white on green and red, dark on the light grey of declined.
verdict_text <- c(`TRUE` = "grey15", `FALSE` = "white")
legend_counts <- function(v) {
  sprintf("%s (%s)", v, vapply(v, function(k) {
    paste(vapply(arms, function(a) {
      as.character(sum(r$verdict == k & r$arm == a))
    }, ""), collapse = " / ")
  }, ""))
}

# Figure 1: one bar per setup, segments = verdict share, labelled with counts.
counts <- as.data.frame(table(arm_label = r$arm_label, verdict = r$verdict))
counts <- counts[counts$Freq > 0, ]
counts$share <- counts$Freq / n_ans
p1 <- ggplot(counts, aes(x = share, y = arm_label, fill = verdict)) +
  geom_col(width = 0.62, position = position_stack(reverse = TRUE)) +
  geom_text(
    aes(label = ifelse(share >= 0.05, Freq, ""), colour = verdict == "declined"),
    position = position_stack(vjust = 0.5, reverse = TRUE),
    fontface = "bold", size = 4.2, show.legend = FALSE
  ) +
  scale_colour_manual(values = verdict_text, guide = "none") +
  # Slices too thin to hold a number are counted in the legend instead, per
  # setup in the order the title names.
  scale_fill_manual(
    values = pal_verdict, labels = legend_counts, drop = TRUE,
    name = sprintf("Answers (%s):", paste(c(
      bare = "alone", alt = "Bash/PlantCV", mcp = "plantcv-mcp"
    )[arms], collapse = " / "))
  ) +
  guides(fill = guide_legend(title.position = "top", title.hjust = 0.5)) +
  scale_x_continuous(labels = scales::percent, expand = c(0, 0)) +
  labs(
    x = sprintf("Share of %d answers per setup", n_ans), y = NULL,
    title = "Plant image measurements: right, wrong or declined",
    subtitle = paste0(setup_note, "\n", key_note)
  ) +
  theme_pcvmcp(base_size = 13) +
  theme(panel.grid.major.y = element_blank(), plot.margin = margin(8, 24, 8, 8))
ggsave(file.path(outdir, "agent_qa.png"), p1, width = 11, height = 4.2, dpi = 200, bg = "white")

# Figure 2: cost and seconds per answer, one point per run, box over them.
long <- rbind(
  data.frame(arm_label = r$arm_label, metric = "Cost per answer (US$, log scale)", value = r$cost_usd),
  data.frame(arm_label = r$arm_label, metric = "Seconds per answer (log scale)", value = r$seconds)
)
p2 <- ggplot(long, aes(x = value, y = arm_label)) +
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
      setup_note, "\nOne point per run; the box spans the middle half, the line in it is the median."
    )
  ) +
  theme_pcvmcp(base_size = 13) +
  theme(
    panel.grid.major.y = element_blank(), strip.text = element_text(hjust = 0, face = "bold"),
    plot.margin = margin(8, 24, 8, 8)
  )
ggsave(file.path(outdir, "agent_qa_cost_time.png"), p2, width = 12, height = 4.2, dpi = 200, bg = "white")

# Figure 3: verdicts per setup x kind of question; strips name the counts.
present <- names(cat_labels)[names(cat_labels) %in% unique(r$category)]
n_cat <- table(r$category[r$arm == arms[1]])
strip <- setNames(
  sprintf("%s (%d questions × %d runs)", cat_labels[present], n_cat[present] / n_rep, n_rep),
  present
)
r$cat_strip <- factor(strip[r$category], levels = strip)
cc <- as.data.frame(table(arm_label = r$arm_label, cat_strip = r$cat_strip, verdict = r$verdict))
cc <- cc[cc$Freq > 0, ]
p3 <- ggplot(cc, aes(x = Freq, y = arm_label, fill = verdict)) +
  geom_col(width = 0.62, position = position_stack(reverse = TRUE)) +
  geom_text(
    aes(label = Freq, colour = verdict == "declined"),
    position = position_stack(vjust = 0.5, reverse = TRUE),
    fontface = "bold", size = 3.4, show.legend = FALSE
  ) +
  scale_colour_manual(values = verdict_text, guide = "none") +
  facet_wrap(~cat_strip, ncol = 1, scales = "free_x") +
  scale_fill_manual(values = pal_verdict, name = NULL, drop = TRUE) +
  scale_x_continuous(expand = c(0, 0)) +
  labs(
    x = "Answers", y = NULL, title = "Right, wrong or declined, by kind of question",
    subtitle = setup_note
  ) +
  theme_pcvmcp(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(), strip.text = element_text(hjust = 0, face = "bold"),
    plot.margin = margin(8, 24, 8, 8)
  )
ggsave(file.path(outdir, "agent_qa_by_category.png"), p3, width = 9, height = 10, dpi = 200, bg = "white")

print(per_arm)
