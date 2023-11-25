import argparse
import os
import warnings
import math
import numbers
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pickle
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from PIL import Image
from tqdm import tqdm

from lib.model_zoo.migan_inference import Generator as MIGAN


warnings.filterwarnings("ignore")


def read_mask(mask_path, invert=False):
    mask = Image.open(mask_path)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            _r, _g, _b, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 2:
            _l, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 3:
            _r, _g, _b = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_r, _r, _r])
    else:
        mask = np.dstack([mask, mask, mask])
    if invert:
        mask = 255 - mask
    mask[mask < 255] = 0
    return Image.fromarray(mask).convert("L")


# GaussianSmoothing is taken from https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/utils/gaussian_smoothing.py
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = (kernel_size - 1) // 2

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, (self.padding, self.padding, self.padding, self.padding), mode='reflect') 
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups, padding='valid')


class MIGAN_Pipeline(nn.Module):

    def __init__(self, model_path, resolution, padding = 128, device='cpu'):
        super().__init__()

        self.model = MIGAN(resolution=resolution)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.model.eval()
        self.gaussian_blur = GaussianSmoothing(
            channels=1, kernel_size=5, sigma=1.0, dim=2).to(device)
        self.res = torch.tensor(resolution)
        self.padding = torch.tensor(padding)

    def get_masked_bbox(self, mask):
        # We implement various hacks to make this method dynamic and ONNX-compatible

        padded_inverted_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=1)
        padded_inverted_mask = padded_inverted_mask.squeeze()
        h = (padded_inverted_mask.sum(dim=0) - torch.tensor(2))[0]
        w = (padded_inverted_mask.sum(dim=1) - torch.tensor(2))[0]
        mask = mask.squeeze().to(torch.float32)
        
        h_indices = torch.arange(0, h, step=1, dtype=torch.int32)
        w_indices = torch.arange(0, w, step=1, dtype=torch.int32)
        
        xx = mask.mean(dim=0)
        yy = mask.mean(dim=1)
        xx_masked_ids = w_indices[xx < torch.tensor(255.0)]
        yy_masked_ids = h_indices[yy < torch.tensor(255.0)]

        x_min = torch.min(torch.cat((xx_masked_ids.reshape(-1), torch.tensor(w).reshape(-1))))
        x_max = torch.max(torch.cat((xx_masked_ids.reshape(-1), torch.tensor(0).reshape(-1))))
        y_min = torch.min(torch.cat((yy_masked_ids.reshape(-1), torch.tensor(h).reshape(-1))))
        y_max = torch.max(torch.cat((yy_masked_ids.reshape(-1), torch.tensor(0).reshape(-1))))
        
        x_min = torch.min(torch.cat((
            x_min.reshape(-1),
            x_max.reshape(-1)
        )))

        x_max = torch.max(torch.cat((
            x_min.reshape(-1),
            x_max.reshape(-1)
        )))

        y_min = torch.min(torch.cat((
            y_min.reshape(-1),
            y_max.reshape(-1)
        )))

        y_max = torch.max(torch.cat((
            y_min.reshape(-1),
            y_max.reshape(-1)
        )))

        cnt_x = (x_min + x_max) // torch.tensor(2)
        cnt_y = (y_min + y_max) // torch.tensor(2)

        masked_w = x_max - x_min
        masked_h = y_max - y_min
        crop_size = torch.max(torch.cat((masked_w.reshape(-1), masked_h.reshape(-1))))
        crop_size = crop_size + self.padding * torch.tensor(2)
        crop_size = torch.max(torch.cat((
            crop_size.reshape(-1),
            torch.tensor(self.res).reshape(-1)
        )))

        offset = crop_size // torch.tensor(2)
        x_min = torch.max(torch.cat((
            (cnt_x - offset).reshape(-1),
            torch.tensor(0).reshape(-1)
        )))
        x_max = torch.min(torch.cat((
            (cnt_x + offset).reshape(-1),
            torch.tensor(w).reshape(-1)
        )))
        y_min = torch.max(torch.cat((
            (cnt_y - offset).reshape(-1),
            torch.tensor(0).reshape(-1)
        )))
        y_max = torch.min(torch.cat((
            (cnt_y + offset).reshape(-1),
            torch.tensor(h).reshape(-1)
        )))

        x_offset_excess = torch.max(torch.cat((
            (crop_size - (x_max - x_min)).reshape(-1),
            torch.tensor(0).reshape(-1)
        )))
        y_offset_excess = torch.max(torch.cat((
            (crop_size - (y_max - y_min)).reshape(-1),
            torch.tensor(0).reshape(-1)
        )))

        x_min = torch.max(torch.cat((
            (x_min - x_offset_excess).reshape(-1),
            torch.tensor(0).reshape(-1)
        )))
        x_max = torch.min(torch.cat((
            (x_max + x_offset_excess).reshape(-1),
            torch.tensor(w).reshape(-1)
        )))
        
        y_min = torch.max(torch.cat((
            (y_min - y_offset_excess).reshape(-1),
            torch.tensor(0).reshape(-1)
        )))
        y_max = torch.min(torch.cat((
            (y_max + y_offset_excess).reshape(-1),
            torch.tensor(h).reshape(-1)
        )))

        return x_min, x_max, y_min, y_max

    def preprocess(self, image, mask):
        image = tvF.resize(image, (self.res, self.res), interpolation=Image.BILINEAR)
        mask = tvF.resize(mask, (self.res, self.res), interpolation=Image.NEAREST)
        image = image.to(torch.float32) * 2 / 255 - 1
        mask = mask.to(torch.float32) / 255
        model_input = torch.cat([mask - 0.5, image * mask], dim=1)
        return model_input

    def postprocess(self, image, mask, model_output):
        model_output = ((model_output * 0.5 + 0.5) * 255).clamp(0, 255)
        model_output = tvF.resize(model_output, (image.size(2), image.size(3)), interpolation=Image.BILINEAR)
        image = image.to(torch.float32)
        mask = mask.to(torch.float32)
        mask = F.max_pool2d(mask, 3, stride=1, padding=1)
        mask = self.gaussian_blur(mask)
        mask = mask / torch.tensor(255)
        composed_img = image * mask + model_output * (1 - mask)
        return composed_img.clamp(0, 255).to(torch.uint8)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        image: (1, 3, H, W) torch.uint8 Tensor
        mask: (1, 1, H, W) torch.uint8 Tensor
        """
        mask = tvF.resize(mask, (image.size(2), image.size(3)), interpolation=Image.NEAREST)
        x_min, x_max, y_min, y_max = self.get_masked_bbox(mask)
        cropped_img, cropped_mask = image[:, :, y_min:y_max, x_min:x_max], mask[:, :, y_min:y_max, x_min:x_max]
        model_input = self.preprocess(cropped_img, cropped_mask)
        model_output = self.model(model_input)
        post_result = self.postprocess(cropped_img, cropped_mask, model_output)
        image[:, :, y_min:y_max, x_min:x_max] = post_result
        return image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, help="256 or 512", required=True)
    parser.add_argument("--model-path", type=str, help="Saved .pt model path.", required=True)
    parser.add_argument("--images-dir", type=Path, help="Path to images directory.", required=True)
    parser.add_argument("--masks-dir", type=Path, help="Path to masks directory.", required=True)
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask? (make 0-known, 1-hole)")
    parser.add_argument("--output-dir", type=Path, help="Output directory.", required=True)
    parser.add_argument("--device", type=str, help="Device.", default="cpu")
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    migan_pipeline = MIGAN_Pipeline(model_path=args.model_path, resolution=args.resolution, device=args.device)

    print("Exporting ONNX model...")
    dummy_image = torch.ones(1, 3, 256, 256, device="cpu", dtype=torch.uint8) * 255
    dummy_mask = torch.ones(1, 1, 256, 256, device="cpu", dtype=torch.uint8) * 255
    dummy_mask[:, :, 10, 10] = 0
    dummy_mask[:, :, 20, 30] = 0
    input_names = ["image", "mask"]
    output_names = ["result"]
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    onnx_export_path = args.output_dir / "models" / "migan.onnx"
    torch.onnx.export(
        migan_pipeline,
        (dummy_image, dummy_mask),
        onnx_export_path,
        verbose=False,
        export_params=True,
        dynamic_axes={'image' : {
                        0 : 'batch_size',
                        2: 'height',
                        3: 'width'
                    },
                    'mask' : {
                        0 : 'batch_size',
                        2: 'height',
                        3: 'width'
                    },
                    'output': {
                        0 : 'batch_size',
                        2: 'height',
                        3: 'width'
                    }},
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=17
    )
    print("ONNX model exported")

    ort_sess = ort.InferenceSession(str(onnx_export_path))

    img_extensions = {".jpg", ".jpeg", ".png"}
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(os.path.join(args.images_dir, "**", f"*{img_extension}"), recursive=True)

    img_paths = sorted(img_paths)
    (args.output_dir / "sample_results").mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(img_paths):
        # print(img_path)
        mask_path = os.path.join(args.masks_dir, "".join(os.path.basename(img_path).split('.')[:-1]) + ".png")
        img = Image.open(img_path).convert("RGB")
        mask = read_mask(mask_path, invert=args.invert_mask)

        with torch.no_grad():
            img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
            mask = np.expand_dims(mask, (0, 1))
            # result_image = migan_pipeline(torch.tensor(img), torch.tensor(mask)).detach().cpu().numpy()
            result_image = ort_sess.run(None, {'image': img, 'mask':  mask})[0]

        result_image = result_image[0].transpose(1, 2, 0)
        result_image = Image.fromarray(result_image)
        result_image.save(args.output_dir / "sample_results" / f"{Path(img_path).stem}.png")


if __name__ == '__main__':
    main()
