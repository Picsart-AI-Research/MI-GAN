import argparse
import warnings
import os
import math
import random
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

warnings.filterwarnings("ignore")


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, help="Path to original images.", required=True)
    parser.add_argument("--output-dir", type=Path, help="Masks output directory.", required=True)
    parser.add_argument("--resolution", type=int, help="Mask resolution.", required=True)
    return parser.parse_args()


def main():
    np.random.seed(0)
    random.seed(0)

    args = get_args()
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    resolution = args.resolution

    img_extensions = {".jpg", ".jpeg", ".png"}
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(os.path.join(args.images_dir, "**", f"*{img_extension}"), recursive=True)

    img_paths = sorted(img_paths)

    for img_path in tqdm(img_paths):
        mask = RandomMask(resolution, [0, 1])
        save_path = output_dir / f"{Path(img_path).stem}.png"
        Image.fromarray(mask).convert("L").save(save_path)


if __name__ == "__main__":
    main()
