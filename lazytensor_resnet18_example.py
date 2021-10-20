"""
Runs the ResNet18 model using the Lazy Tensor Core with the TorchScript backend.

Requirements to run example:
- `torchvision` Python package
- `pillow` Python package
- `requests` Python package
- `lazy_tensor_core` Python package
    For information on how to obtain the `lazy_tensor_core` Python package,
    see here:

    https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md

To run the example, make sure `/path/to/pytorch/lazy_tensor_core` is in your
PYTHONPATH. Then, run

    python lazytensor_resnet18_example.py

The output of this example can be found in
    `lazytensor_resnet18_example_output.txt`

Most of the code in this example was barrowed from
    https://github.com/llvm/torch-mlir/blob/main/examples/torchscript_resnet18_e2e.py
"""

from torchvision import models, transforms
from PIL import Image
import requests

import torch
import lazy_tensor_core as ltc
from lazy_tensor_core.debug import metrics

ltc._LAZYC._ltc_init_ts_backend()

DEVICE = 'lazy'
IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg'

def load_and_preprocess_image(url: str):
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    img = Image.open(requests.get(url, headers=headers,
                                  stream=True).raw).convert('RGB')
    # preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels():
    classes_text = requests.get(
        'https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt',
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels


def top3_possibilities(res, labels):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3


class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True).to(DEVICE)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


def main():
    print('Loading image...')
    img = load_and_preprocess_image(IMAGE_URL).to(DEVICE)
    print('Loading labels...')
    labels = load_labels()

    resnet_module = ResNet18Module()
    print('Running resnet18.forward...')
    result = resnet_module.forward(img)
    print('Top 3 predictions:')
    print(top3_possibilities(result, labels))

    print('\nMetrics report:')
    print(metrics.metrics_report())
    graph_str = ltc._LAZYC._get_ltc_tensors_backend([resnet_module.forward(img)])
    print(graph_str)


if __name__ == '__main__':
    main()
