args = commandArgs(trailingOnly=TRUE)

if (length(args) != 2) {
  stop("Exactly two argument must be supplied (input file, output file).n", call.=FALSE)
}

options(warn=-1, verbose = FALSE)

suppressMessages(library(readr))
suppressMessages(library(ggplot2))
suppressMessages(library(dplyr))

roc_file <- read_csv(args[1], col_names = FALSE)
roc_file <- roc_file %>% as.matrix() %>% t()

roc_df <- roc_file %>% as.data.frame() %>% rename(TPR = V1, FPR = V2)

plot <- roc_df %>%
  ggplot() +
  geom_point(aes(x = FPR, y = TPR))

ggsave(plot, file = args[2])