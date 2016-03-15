# Author: Sebastien Dubois

# Libraries ---------------------------------------------------------------
library(readr)
library(dplyr)
library(stringr)



# Load --------------------------------------------------------------------

dir <- "sutter/"
# healogics/
# sutter/

val_name <- "val-12-4-15"
# "test-2013-only-4-3-15"  # 
train_name <- "train-12-4-15"
# "train-2013-only-4-3-15"  # 

data.val <- str_c("data/", dir, val_name,".txt", sep = "")
data.train <- str_c("data/", dir, train_name, ".txt", sep = "")

df.val <- read_tsv(file = data.val)
df.train <- read_tsv(file = data.train)

df.val %>% dim()
df.train %>% dim()

# change target name for rbind
names(df.val)[1] <- 'Y'
names(df.train)[1] <- 'Y'
df <- rbind(df.val, df.train)



# Create attribute file ---------------------------------------------------


# clean variable names
p <- " |-|&|/|\\*|%"
names(df) <- names(df) %>%
  lapply(function(s) str_replace_all(s, pattern = p, replacement = "_")) %>%
  unlist()

# attribute file
file_str <- ""
for(i in 1:ncol(df)) {
  if(class(df[[i]]) == 'numeric') {
    file_str <- str_c(file_str, names(df)[i], ": cont")
  } else {
    
    # if integer, check if it's a discrete variable
    uniques <- df[,i] %>% unique() 
    nval <- uniques %>% nrow()
    
    if(sum(uniques < 0 | uniques >= nval) == 0) {
      # discrete variable
      vals <- str_c(c(0:(nval-1)), sep = "", collapse = ", ")
      vals <- str_c(names(df)[i], ": {", vals, "}")
      file_str <- str_c(file_str, vals)
    } else {
      file_str <- str_c(file_str, names(df)[i], ": cont")
    }
  }
  
  # add 'class' for the target variable
  if(i ==1) {
    file_str <- str_c(file_str, " (class)\n")
  } else {
    file_str <- str_c(file_str, "\n")
  }
}

# remove last '\n'
file_str <- substr(file_str, 1, (nchar(file_str)-1))

# save attr file
write(file_str,
      file = str_c("data/", dir, "data_attr.txt", sep= ""))


# Rewrite files -----------------------------------------------------------

# nrow(df.train)
# 0.75 * nrow(df.train)
# nrow(df.val)

ntrain <- 21568
# sutter : 99081
# healogics : 21568

# shuffle training set
df.train <- df.train[sample(nrow(df.train)),]

test.set <- df.val
valid.set <- df.train %>% slice((ntrain + 1):nrow(df.train))
train.set <- df.train %>% slice(1:ntrain)

# rewrite data without headers and with space separators

# training
write_delim(train.set,
            path = str_c("data/", dir, "train.txt", sep = ""),
            col_names = FALSE,
            delim = " ")

# validation set, a part from original training data
write_delim(valid.set,
            path = str_c("data/", dir, "val.txt", sep = ""),
            col_names = FALSE,
            delim = " ")

# all = training + validation
write_delim(df.train,
            path =str_c("data/", dir, "all.txt", sep = ""),
            col_names = FALSE,
            delim = " ")

# a test set we should never use for training, original 'validation' set
write_delim(test.set,
            path = str_c("data/", dir, "test.txt", sep = ""),
            col_names = FALSE,
            delim = " ")


