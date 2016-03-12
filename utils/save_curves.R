args = commandArgs(trailingOnly=TRUE)

if (length(args) != 2) {
  stop("Exactly two argument must be supplied (input file, output file).n", call.=FALSE)
}

options(warn=-1, verbose = FALSE)

suppressMessages(library(readr))
suppressMessages(library(ggplot2))
suppressMessages(library(magrittr))
suppressMessages(library(plyr))

roc_file <- read_csv(args[1], col_names = FALSE)
roc_file <- roc_file %>% as.matrix() %>% t()

model <- strsplit(args[2], split = "/")[[1]][2]

roc_df <- roc_file %>%
  as.data.frame() %>%
  setNames(c("TPR", "FPR", "Precision")) %>%
  subset(TPR < 1.1 & FPR < 1.1)

plot1 <- roc_df %>%
  ggplot() +
  geom_point(aes(x = FPR, y = TPR)) +
  labs(title = model)

ggsave(plot1, file = paste(args[2], "_roc", ".png", sep = ""))


plot2 <- roc_df %>%
  ggplot() +
  geom_point(aes(x = TPR, y = Precision)) +
  labs(x = "Recall", title = model)

ggsave(plot2, file = paste(args[2], "_pr", ".png", sep = ""))
