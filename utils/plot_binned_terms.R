args = commandArgs(trailingOnly=TRUE)

# options(warn=-1, verbose = FALSE)

suppressMessages(library(readr))
suppressMessages(library(ggplot2))

filename <- paste(args[1], ".txt", sep = "")

df <- read_tsv(filename)
cols <- colnames(df)

p <- ggplot(df) +
  stat_summary_2d(aes_string(x = cols[1], y = cols[2], z = "z")) +
  labs(fill = "")
ggsave(plot = p, filename = paste(args[1], ".png", sep = ""))
