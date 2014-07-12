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

```
attribute_name: type [(class)]
```

There are two types of binned attributes. One is specified using the number of bins, and the other is specified using number of bins, upper bounds and medians for each bin.

#### Example attribute file
```
f1: cont
f2: {a, b, c}
f3: binned (256) 
f4: binned (3;[1, 5, 6];[0.5, 2.5, 3])
label: cont (class) 
```

#### Example data file
```
0.1 1 2 0 5
-2.3 0 255 1 2
3.1 2 128 2 -3
5 1 0 1 0.2
0.1 1 37 0 0.1
```

# Building Models

# Evaluating Models


