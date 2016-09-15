# TensorFlow-Paper

A review paper on TensorFlow, the coolest thing since sliced bread.

## 1. Introduction

Two main points:

1. A brief, high-level discussion of machine intelligence, it's impact on
   society and current fields of application, concluding with observations about
   the particular effectiveness of deep learning models and reasons for this
   (and the delay of popularity for deep-learning).

2. Deep-Learning algorithms and models must be programmed somehow. For this,
   there existed libraries such as Theano, ... Now there is also Tensorflow,
   released by Google in ... (general information).

## 2. History of Machine Learning Frameworks

Give a brief overview and timeline of important programming frameworks in the
field of machine learning in the last ~30 years.

### 2.1 General Machine Learning Frameworks

#### 2.1.1 Open-Source

Cover:

1. MLC++: http://ai.stanford.edu/~ronnyk/mlcj.pdf (1994)
2. OpenCV: http://opencv.org (2000)
3. scikit-learn: http://dl.acm.org/citation.cfm?id=2078195 (2007)
4. Accord.NET: https://github.com/accord-net/framework (2008)
5. MOA: http://jmlr.csail.mit.edu/proceedings/papers/v11/bifet10a/bifet10a.pdf
   (2010)
6. Mahout: http://mahout.apache.org 2011
8. pattern: https://github.com/clips/pattern,
  http://www.jmlr.org/papers/volume13/desmedt12a/desmedt12a.pdf (2012)
9. spark mllib: http://spark.apache.org/mllib,
  http://arxiv.org/pdf/1505.06807v1.pdf (2015)
### 2.2 Focus on Deep Learning Toolkits

Note that this should be a brief listing and history of these frameworks,
*not* a comparison with TensorFlow (see Section 5).

Deep Learning or NN?

* Theano: http://arxiv.org/pdf/1605.02688.pdf (2008)
* Caffe: http://arxiv.org/abs/1408.5093 (2014)
* Torch: http://publications.idiap.ch/downloads/reports/2002/rr02-46.pdf (2002)
* DL4J: (2014)
* Lasagne: https://github.com/Lasagne/Lasagne (2014)
* CuDNN: https://developer.nvidia.com/cudnn
  https://developer.nvidia.com/deep-learning-software (2014)
* Nervana NeOn: http://www.nervanasys.com (2014)
* cuda-convnet: https://code.google.com/p/cuda-convnet/ (2014)

## 3. TensorFlow: Interface

Describe the abstract concepts of the TensorFlow Interface, nothing about
programming.

### 3.1 Elements of a TensorFlow Graph

The data-flow graph, `Variables`, `Operations`, `Tensors`, `Sessions`, `Graphs` etc.

Sparse Tensors.

### 3.2 Execution Model

Speak to the execution of a Graph:

* Devices
* Placement Algorithm
* Single-Machine Execution
* Many-Machine Execution

### 3.3. Optimizations

Optimizations implemented and aimed for by the TensorFlow team:

* Common-subexpression elimination
* Scheduling
* Lossy Compression
* (Async. Kernels)

^ Careful not to just repeat the paper.

### 3.4 Extensions

Extensions to the basic elements of a dataflow graph:

* Backpropagation Nodes
* Read/Write Nodes for Checkpoints
* Control-Flow
* Queues

## 4 Programming Interface

### 4.1. Overview

Currently the Python API is best developed, while the C++ API does not yet allow
for graph building. In the future, they expect more language frontends.

### 4.2 Basic Walkthrough

Explain the Python API by walking through a practical example.

### 4.3 Abstractions

Give an overview of the various abstraction libraries out there (for rapid
prototyping):

* PrettyTensor
* Keras
* TFLearn

## 5 Visualization



## 6. Comparison With Other DL Frameworks

### 6.1 Comparison of Basic Paradigms

Caffe's basic units are layers rather than nodes, Theano also uses symbolic
graphs, Torch is imperative while TensorFlow is declarative etc.

* Theano vs. TensorFlow
* Torch vs. TensorFlow
* Caffe vs. TensorFlow
* DL4J vs TensorFlow (maybe not)

### 6.2 Performance Comparison

Speak to Benchmarks:

http://arxiv.org/abs/1511.06435v3
http://arxiv.org/pdf/1605.02688v1.pdfA

## 7 Conclusion
