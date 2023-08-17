import argparse
import os
import math
import random
import shutil
import warnings
from glob import glob
from pathlib import Path
from typing import Tuple, Optional

import lpips
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lib.model_zoo.migan_inference import Generator as MIGAN
from lib.model_zoo.comodgan import (
    Generator as CoModGANGenerator,
    Mapping as CoModGANMapping,
    Encoder as CoModGANEncoder,
    Synthesis as CoModGANSynthesis
)

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="One of [migan-256, migan-512, comodgan-256, comodgan-512]")
    parser.add_argument("--model-path", type=Path, help="Saved model path.", required=True)
    parser.add_argument("--real-dir", type=Path, help="Real dataset directory path.", required=True)
    parser.add_argument("--mask-dir", type=Path, help="Pre-generated mask directory path.", required=False)
    parser.add_argument("--device", type=str, help="Device to use.", required=False, default="cuda")
    parser.add_argument("--batch-size", type=int, help="Batch size to use.", required=False, default=8)
    parser.add_argument("--num-workers", type=int, help="Number of workers to use.", required=False, default=2)
    return parser.parse_args()


def RandomBrush(
    max_tries,
    s,
    min_num_vertex=4,
    max_num_vertex=18,
    mean_angle=2*math.pi / 5,
    angle_range=2*math.pi / 15,
    min_width=12,
    max_width=48
):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask


def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)

        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0

        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)

        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return (mask * 255).astype(np.uint8)


class InferenceDataset(Dataset):
    def __init__(
        self,
        real_dir: Path,
        mask_dir: Optional[Path] = None,
        resolution: int = None
    ):
        super(InferenceDataset, self).__init__()

        img_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
        self.img_paths = [i for i in Path(real_dir).iterdir() if i.suffix in img_extensions]
        self.mask_dir = mask_dir
        self.resolution = resolution

    def __getitem__(self, index) -> Tuple[torch.Tensor, np.array, np.array, str]:
        img_path = Path(self.img_paths[index])
        img_name = img_path.stem
        img = Image.open(img_path).convert("RGB")
        if img.size[0] != self.resolution or img.size[1] != self.resolution:
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        assert img.size[0] == self.resolution

        if self.mask_dir is not None:
            mask_path = self.mask_dir / f"{img_name}.png"
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
            assert mask.size[0] == self.resolution
        else:
            mask = RandomMask(img.size[0])
            mask = Image.fromarray(mask).convert("L")

        img = np.array(img)
        mask = np.array(mask)[:, :, np.newaxis] // 255
        img = torch.Tensor(img).float() * 2 / 255 - 1
        mask = torch.Tensor(mask).float()
        img = img.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        x = torch.cat([mask - 0.5, img * mask], dim=0)
        return x, np.array(img), mask, img_name

    def __len__(self):
        return len(self.img_paths)


def get_activations(model, imgs):
    with torch.no_grad():
        activations = model(imgs)[0]
    if activations.size(2) != 1 or activations.size(3) != 1:
        activations = adaptive_avg_pool2d(activations, output_size=(1, 1))
    activations = activations.squeeze(3).squeeze(2).cpu().numpy()
    return activations


def main():
    args = get_args()

    np.random.seed(0)
    random.seed(0)
    cuda = False
    if args.device == "cuda":
        cuda = True

    if args.model_name == "migan-256":
        resolution = 256
        model = MIGAN(resolution=256)
    elif args.model_name == "migan-512":
        resolution = 512
        model = MIGAN(resolution=512)
    elif args.model_name == "comodgan-256":
        resolution = 256
        comodgan_mapping = CoModGANMapping(num_ws=14)
        comodgan_encoder = CoModGANEncoder(resolution=resolution)
        comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
        model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
    elif args.model_name == "comodgan-512":
        resolution = 512
        comodgan_mapping = CoModGANMapping(num_ws=16)
        comodgan_encoder = CoModGANEncoder(resolution=resolution)
        comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
        model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
    else:
        raise Exception("Unsupported model name.")

    model.load_state_dict(torch.load(args.model_path))
    if cuda:
        model = model.to("cuda")
    model.eval()

    compute_lpips = lpips.LPIPS(net="alex")
    if cuda:
        compute_lpips = compute_lpips.to("cuda")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx]).to(args.device)
    inception_v3.eval()

    args = get_args()

    inference_dataset = InferenceDataset(real_dir=args.real_dir, mask_dir=args.mask_dir, resolution=resolution)
    inference_dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("Starting the inference... ")

    lpips_vals = []
    real_activations_arr = np.empty((len(inference_dataset), 2048))
    fake_activations_arr = np.empty((len(inference_dataset), 2048))
    start_idx = 0
    for batch in tqdm(iter(inference_dataloader)):
        inputs, imgs, masks, img_names = batch
        if cuda:
            inputs = inputs.to("cuda")
            imgs = imgs.to("cuda")
            masks = masks.to("cuda")

        with torch.no_grad():
            output_imgs = model(inputs)
       
        composed_imgs = masks * imgs + (1 - masks) * output_imgs

        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        composed_imgs = (composed_imgs * 0.5 + 0.5).clamp(0, 1)

        with torch.no_grad():
            lpips_val = compute_lpips(imgs, composed_imgs, normalize=True)
        lpips_vals.extend(lpips_val.detach().cpu().numpy().tolist())

        # Calculate activations for FID
        bs = imgs.shape[0]
        real_activations_arr[start_idx:start_idx + bs] = get_activations(inception_v3, imgs)
        fake_activations_arr[start_idx:start_idx + bs] = get_activations(inception_v3, composed_imgs)
        start_idx += bs

    print("Calculating Frechet distance. Please wait... ")

    mu_real = np.mean(real_activations_arr, axis=0)
    sigma_real = np.cov(real_activations_arr, rowvar=False)
    mu_fake = np.mean(fake_activations_arr, axis=0)
    sigma_fake = np.cov(fake_activations_arr, rowvar=False)
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    print("FID:", fid_value)
    print("LPIPS: ", np.mean(lpips_vals))


if __name__ == "__main__":
    main()
