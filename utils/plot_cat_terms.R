args = commandArgs(trailingOnly=TRUE)

# options(warn=-1, verbose = FALSE)

suppressMessages(library(readr))
suppressMessages(library(ggplot2))

filename <- paste(args[1], ".txt", sep = "")

df <- read_tsv(filename)
cols <- colnames(df)

p <- ggplot(df) +
  geom_bar(aes(x = factor(df[,1]), y = z, fill = factor(df[,2])),
           stat = "identity",
           position = "dodge") +
  labs(x = cols[1] , y = "", fill = cols[2])
print(p)
ggsave(plot = p, filename = paste(args[1], ".png", sep = ""))