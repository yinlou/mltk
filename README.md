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

## Sparse Input Format

MLTK uses the following structure for sparse input format:
```
target feature:value ... feature:value
```

Feature/value pairs must be ordered by increasing feature number. MLTK does not skip zero valued features. For classification problems, make sure the target is in {0, ..., K - 1}, where K is the number of classes.

#### Example data file

```
0 1:0.2 3:0.5
0 2:-0.4
1 1:3.2 5:-3
```

## Dataset I/O

MLTK provides two classes to perform reading/writing of datasets: `mltk.core.io.InstancesReader` and `mltk.core.io.InstancesWriter`.

#### Example

`Instances instances = InstancesReader.read(< attr file path >, < dataset file path >)`
It reads a dense dataset from attribute file and data file.
`Instances instances = InstancesReader.read(< dataset file path >, < class index >)`
It reads a dense dataset from data file and a specified class index. A negative class index (e.g., -1) means no target is specified. Instances instances = `InstancesReader.read(< dataset file path >)` 
It reads a (maybe sparse) dataset from data file

# Building Models

# Evaluating Models


