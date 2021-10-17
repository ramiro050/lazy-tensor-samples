"""
Runs the MaskRCNN on input image with Lazy Tensor Core/TorchScript backend.

Requirements to run example:
- `opencv-python` Python package
- `lazy_tensor_core` Python package
    For information on how to obtain the `lazy_tensor_core` Python package,
    see here:

    https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md

- `maskrcnn_benchmark` Python package
    For information on how to obtain the this Python package, see here:

    https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/INSTALL.md

To run the example, make sure `/path/to/maskrcnn-benchmark/demo` and
`/path/to/pytorch/lazy_tensor_core` is in your PYTHONPATH. Then, run

    python lazytensor_maskrcnn_example.py img_path

where `img_path` is the path to the sample image to run the model on.

The output of this example can be found in
    `lazytensor_maskrcnn_example_output.txt`
"""
import argparse
import pathlib

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import lazy_tensor_core as ltc
from lazy_tensor_core.debug import metrics
import cv2

ltc._LAZYC._ltc_init_ts_backend()


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run MaskRCNN on Lazy Tensor Core')
    parser.add_argument('img_path', type=pathlib.Path,
                        help='Path to image to run model on')
    parser.add_argument('maskrcnn_path', type=pathlib.Path,
                        help='Path to `maskrcnn-benchmark` repo. ' \
                        'Used to get configuration information.')
    return parser


def main():
    args = setup_argparse().parse_args()

    print('Loading image...')
    image = cv2.imread(str(args.img_path))

    rel_config_file = 'configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'
    config_file = args.maskrcnn_path / rel_config_file
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "lazy"])
    coco_demo = COCODemo(cfg, min_image_size=50,
                         confidence_threshold=0.7)

    print('Running model on image...')
    coco_demo.run_on_opencv_image(image)

    print('Metrics Report:')
    print(metrics.metrics_report())


if __name__ == '__main__':
    main()
