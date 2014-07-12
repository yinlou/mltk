# Overview

MLTK is a collection of various supervised machine learning algorithms, which is designed for directly training models and further development. For questions or suggestions with the code, please email <a href="mailto:yinlou@cs.cornell.edu">Yin Lou</a>.

Currently MLTK supports:
* Generalized Linear Models
  * Ridge
  * Lasso
  * Elastic Net
* Generalized Additive Models
* Regression Trees
* Random Forests
* Boosted Trees

# Dataset Format

## Dense Input Format

Typical input to MLTK is a text file containing the data matrix. An optional attribute file may also be provided to specify the target attribute. Datasets should be provided in separate white-space-delimited text files without any headers. MLTK supports continuous, nominal and binned attributes. All dense datasets should have the same number and order of columns. The structure of the attribute description is the following:

        attribute_name: type [(class)]

There are two types of binned attributes. One is specified using the number of bins, and the other is specified using number of bins, upper bounds and medians for each bin.

# Building Models

# Evaluating Models


