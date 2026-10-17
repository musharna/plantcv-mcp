# House ggplot2 theme + palette for plantcv-mcp figures.
#
# `git ls-files | grep -iE '\.R$|theme'` found no R file in this repo, so this
# is the first and the only place figure styling lives: plotting scripts
# source it and add only the geoms and scales specific to one figure. Modelled
# on plant-genomics-mcp's theme_pgmcp.R so the sibling servers' benchmark
# figures read alike.

theme_pcvmcp <- function(base_size = 12) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "bottom",
      # Several legends stack in their own rows instead of one row that runs
      # off the right edge at a fixed figure width.
      legend.box = "vertical",
      legend.spacing.y = grid::unit(0, "pt"),
      legend.box.spacing = grid::unit(6, "pt"),
      legend.margin = ggplot2::margin(0, 0, 0, 0),
      plot.title.position = "plot"
    )
}

# Verdicts of benchmarks/agent_qa/score.py. Green / red / grey mean right /
# wrong / declined in every figure; no_answer (a reply without an ANSWER
# line) gets Okabe-Ito orange so a format failure is never read as either.
pal_verdict <- c(
  right = "#2E7D5B",
  wrong = "#C0392B",
  declined = "#BDBDBD",
  no_answer = "#E69F00",
  # A run that did not test its setup (run.py assess(): exit not 0, tools or
  # server missing, no tool call, a leak marker). Shown, never counted as a
  # verdict.
  invalid = "#56B4E9"
)

# Per-run points (cost, seconds): a filled circle with a thin white outline,
# so overlapping runs stay countable where they merely overlap.
pcvmcp_point_size <- 2.4
pcvmcp_point_stroke <- 0.5
pcvmcp_point_outline <- "white"
pcvmcp_point_fill <- "grey35"
