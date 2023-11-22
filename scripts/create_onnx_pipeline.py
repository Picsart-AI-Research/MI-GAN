import argparse
import os
import warnings
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pickle
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
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


class MIGAN_Pipeline(nn.Module):

    def __init__(self, model_path, resolution, device='cpu'):
        super().__init__()

        self.model = MIGAN(resolution=resolution)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.model.eval()
        self.res = resolution

    def preprocess(self, image, mask):
        image = F.resize(image, (self.res, self.res), interpolation=Image.BILINEAR)
        mask = F.resize(mask, (self.res, self.res), interpolation=Image.NEAREST)
        image = image.to(torch.float32) * 2 / 255 - 1
        mask = mask.to(torch.float32) / 255
        model_input = torch.cat([mask - 0.5, image * mask], dim=1)
        return model_input

    def postprocess(self, image, mask, model_output):
        model_output = ((model_output * 0.5 + 0.5) * 255).clamp(0, 255)
        model_output = F.resize(model_output, (image.size(2), image.size(3)), interpolation=Image.BILINEAR)
        image = image.to(torch.float32)
        mask = mask.to(torch.uint8)
        mask = mask / 255
        composed_img = image * mask + model_output * (1 - mask)
        return composed_img.clamp(0, 255).to(torch.uint8)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        image: (1, 3, H, W) torch.uint8 Tensor
        mask: (1, 1, H, W) torch.uint8 Tensor
        """
        model_input = self.preprocess(image, mask)
        model_output = self.model(model_input)
        return self.postprocess(image, mask, model_output)


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
    dummy_image = torch.ones(1, 3, 512, 512, device="cpu", dtype=torch.uint8) * 255
    dummy_mask = torch.ones(1, 1, 512, 512, device="cpu", dtype=torch.uint8) * 255
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
        opset_version=12
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
        mask_path = os.path.join(args.masks_dir, "".join(os.path.basename(img_path).split('.')[:-1]) + ".png")
        img = Image.open(img_path).convert("RGB")
        mask = read_mask(mask_path, invert=args.invert_mask)

        with torch.no_grad():
            img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
            mask = np.expand_dims(mask, (0, 1))
            result_image = ort_sess.run(None, {'image': img, 'mask':  mask})[0]

        result_image = result_image[0].transpose(1, 2, 0)
        result_image = Image.fromarray(result_image)
        result_image.save(args.output_dir / "sample_results" / f"{Path(img_path).stem}.png")


if __name__ == '__main__':
    main()
