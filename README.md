# lazy-tensor-samples

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

### Resnet-18

#### Setup

First [install Torchvision and Lazy Tensor Core](#install-torchvision-and-lazy-tensor-core).

Install the following Python packages:

```shell
python -m pip install pillow request
```

#### Running Example

From inside the `lazy-tensor-samples` directory, run:

```shell
python lazytensor_resnet18_example.py
```

The output of this example can be found in [lazytensor_resnet18_example_output.txt](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_resnet18_example_output.txt).
