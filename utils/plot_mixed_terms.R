args = commandArgs(trailingOnly=TRUE)

# options(warn=-1, verbose = FALSE)

suppressMessages(library(readr))
suppressMessages(library(ggplot2))
suppressMessages(library(reshape2))

filename <- paste(args[1], ".txt", sep = "")

discr_var <- args[2]
cont_var <- args[3]

df <- melt(read_tsv(filename), id.vars = "x", variable.name = "state")

p <- ggplot(df, mapping = aes(x = x, y = value, color = state)) +
  geom_point() +
  geom_line() +
  labs(x = cont_var, y = "", color = discr_var)
ggsave(plot = p, filename = paste(args[1], ".png", sep = ""))
