# lazy-tensor-samples

## Table of Contents

- [lazy-tensor-samples](#lazy-tensor-samples)
  * [How to Run Examples](#how-to-run-examples)
    + [Install Torchvision and Lazy Tensor Core](#install-torchvision-and-lazy-tensor-core)
    + [Bert](#bert)
      - [Setup](#setup)
      - [Running Example](#running-example)
    + [MaskRCNN](#maskrcnn)
      - [Setup](#setup-1)
      - [Running Example](#running-example-1)
    + [Resnet-18 Inference and Training](#resnet-18-inference-and-training)
      - [Setup](#setup-2)
      - [Additional steps for Inference](#additional-steps-for-inference)
      - [Additional steps for Training](#additional-steps-for-training)
      - [Running Inference Example](#running-inference-example)
      - [Running Training Example](#running-training-example)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## How to Run Examples

### Install Torchvision and Lazy Tensor Core

Install `torchvision`:

```shell
python -m pip install torchvision
python -m pip uninstall torch # if it was automatically installed by torchvision
```

Install the `lazy-tensor-core` Python package by following their [instructions](https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md).

Update `PYTHONPATH`:

```shell
export PYTHONPATH=/path/to/pytorch/lazy_tensor_core:$PYTHONPATH
```

### Bert

#### Setup

First [install Lazy Tensor Core](#install-torchvision-and-lazy-tensor-core).

Install the following Python packages:

```shell
python -m pip transformers datasets
```

#### Running Example

From inside the `lazy-tensor-samples` directory, run:

```shell
python lazytensor_bert_example.py
```

The output of this example can be found in [lazytensor_bert_example_output.txt](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_bert_example_output.txt).


### MaskRCNN

#### Setup

First [install Torchvision and Lazy Tensor Core](#install-torchvision-and-lazy-tensor-core).

Install the `maskrcnn-benchmark` using my fork, which includes some changes to make the benchmark run on LTC:

```shell
git clone https://github.com/ramiro050/maskrcnn-benchmark.git
cd maskrcnn-benchmark
git checkout lazy-tensor-maskrcnn
```

Follow the `maskrcnn-benchmark` installation [instructions](https://github.com/ramiro050/maskrcnn-benchmark/blob/lazy-tensor-maskrcnn/INSTALL.md).

Update `PYTHONPATH`:

```shell
export PYTHONPATH=/path/to/maskrcnn-benchmark/demo:$PYTHONPATH
```

#### Running Example

From inside the `lazy-tensor-samples` directory, run:

```shell
python lazytensor_maskrcnn_example.py path/to/image.png path/to/maskrcnn-benchmark
```
where `img.png` is the image to run the model on.

The output of this example can be found in [lazytensor_maskrcnn_example_output.txt](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_maskrcnn_example_output.txt).

### Resnet-18 Inference and Training

#### Setup

First [install Torchvision and Lazy Tensor Core](#install-torchvision-and-lazy-tensor-core).

#### Additional steps for Inference

Install the following Python packages:

```shell
python -m pip install pillow request
```

#### Additional steps for Training

Install the library `libsndfile`. On Ubuntu, simply run

```shell
sudo apt-get install libsndfile-dev
```

Install the PyTorch benchmarks using my fork, which includes some changes to make the benchmark run on LTC (the changes are based on [this](https://github.com/pytorch/benchmark/pull/456) patch by @alanwaketan):

```shell
git clone https://github.com/ramiro050/benchmark.git
cd benchmark
git checkout lazytensor_support
```

Then follow [these](https://github.com/ramiro050/benchmark#building-from-source) instructions to install the benchmark.


#### Running Inference Example

From inside the `lazy-tensor-samples` directory, run:

```shell
python lazytensor_resnet18_example.py
```

The output of this example can be found in [lazytensor_resnet18_example_output.txt](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_resnet18_example_output.txt).

#### Running Training Example

From inside the [benchmark](https://github.com/ramiro050/benchmark) directory, run:

```shell
python run.py resnet18 -d lazy -t train
```
