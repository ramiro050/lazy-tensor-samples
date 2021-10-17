# lazy-tensor-samples

## How to Run Examples

### MaskRCNN

#### Setup

Install `torchvision`:

```shell
python -m pip install torchvision
python -m pip uninstall torch # if it was automatically installed by torchvision
```

Install the `lazy-tensor-core` Python package by following their [instructions](https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md).

Install the `maskrcnn-benchmark` using my fork, which includes some changes to make the benchmark run on LTC:

```shell
git clone https://github.com/ramiro050/maskrcnn-benchmark.git
cd maskrcnn-benchmark
git checkout lazy-tensor-maskrcnn
```

Follow the `maskrcnn-benchmark` installation [instructions](https://github.com/ramiro050/maskrcnn-benchmark/blob/lazy-tensor-maskrcnn/INSTALL.md).

Setup `PYTHONPATH`:

```shell
export PYTHONPATH=/path/to/maskrcnn-benchmark/demo:/path/to/pytorch/lazy_tensor_core:$PYTHONPATH
```

#### Running Example

```shell
python lazytensor_maskrcnn_example.py path/to/image.png path/to/maskrcnn-benchmark
```
where `img.png` is the image to run the model on.

The output of this example can be found in [lazytensor_maskrcnn_example_output.txt](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_maskrcnn_example_output.txt).
