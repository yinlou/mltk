args = commandArgs(trailingOnly=TRUE)

# options(warn=-1, verbose = FALSE)

suppressMessages(library(readr))
suppressMessages(library(ggplot2))
suppressMessages(library(scales))

filename <- paste(args[1], ".txt", sep = "")

df <- read_tsv(filename)
cols <- colnames(df)

p <- ggplot(df, mapping = aes(x = df[,1], y = y)) +
  geom_point() +
  geom_line() +
  labs(x = cols[1] , y = "")
print(p)
ggsave(plot = p, filename = paste(args[1], ".png", sep = ""))