import argparse
import pickle
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from lib.model_zoo.migan_inference import Generator as MIGANGenerator


def copy_weights(source, dest, resolution=256):
    def get_source_w(module):
        if module.reparametrize:
            w = getattr(module, "w0")
            for i in range(1, module.num_reparam_tensors):
                w = w + getattr(module, f"w{i}")
            w = w / np.sqrt(module.num_reparam_tensors)
        else:
            w = module.weight
        w = w * (w.square().sum(dim=[1, 2, 3]) + 1e-8).rsqrt().reshape(-1, 1, 1, 1)
        return torch.nn.Parameter(w)

    # Copy Encoder
    for res in [2 ** i for i in range(2, int(np.log2(resolution)) + 1)]:
        source_module = getattr(source.encoder, f"b{res}")
        dest_module = getattr(dest.encoder, f"b{res}")

        if dest_module.fromrgb is not None:
            dest_module.fromrgb.weight = get_source_w(source_module.fromrgb)
            if dest_module.fromrgb.bias is not None:
                dest_module.fromrgb.bias = nn.Parameter(source_module.fromrgb.bias.to(torch.float32))

        dest_module.conv1.conv1.weight = get_source_w(source_module.conv1.conv1)
        dest_module.conv1.conv2.weight = get_source_w(source_module.conv1.conv2)
        if dest_module.conv1.conv1.bias is not None:
            dest_module.conv1.conv1.bias = nn.Parameter(source_module.conv1.conv1.bias.to(torch.float32))
        if dest_module.conv1.conv2.bias is not None:
            dest_module.conv1.conv2.bias = nn.Parameter(source_module.conv1.conv2.bias.to(torch.float32))

        dest_module.conv2.conv1.weight = get_source_w(source_module.conv2.conv1)
        dest_module.conv2.conv2.weight = get_source_w(source_module.conv2.conv2)
        if dest_module.conv2.conv1.bias is not None:
            dest_module.conv2.conv1.bias = nn.Parameter(source_module.conv2.conv1.bias.to(torch.float32))
        if dest_module.conv2.conv2.bias is not None:
            dest_module.conv2.conv2.bias = nn.Parameter(source_module.conv2.conv2.bias.to(torch.float32))

        setattr(dest.encoder, f"b{res}", dest_module)

    # Copy Synthesis
    for res in [2 ** i for i in range(2, int(np.log2(resolution)) + 1)]:
        source_module = getattr(source.synthesis, f"b{res}")
        dest_module = getattr(dest.synthesis, f"b{res}")

        if dest_module.torgb is not None:
            dest_module.torgb.weight = get_source_w(source_module.torgb)
            if dest_module.torgb.bias is not None:
                dest_module.torgb.bias = nn.Parameter(source_module.torgb.bias.to(torch.float32))

        dest_module.conv1.conv1.weight = get_source_w(source_module.conv1.conv1)
        dest_module.conv1.conv2.weight = get_source_w(source_module.conv1.conv2)
        if dest_module.conv1.conv1.bias is not None:
            dest_module.conv1.conv1.bias = nn.Parameter(source_module.conv1.conv1.bias.to(torch.float32))
        if dest_module.conv1.conv2.bias is not None:
            dest_module.conv1.conv2.bias = nn.Parameter(source_module.conv1.conv2.bias.to(torch.float32))
        if dest_module.conv1.use_noise:
            dest_module.conv1.noise_const = source_module.conv1.conv2.noise_const
            dest_module.conv1.noise_strength = source_module.conv1.conv2.noise_strength

        dest_module.conv2.conv1.weight = get_source_w(source_module.conv2.conv1)
        dest_module.conv2.conv2.weight = get_source_w(source_module.conv2.conv2)
        if dest_module.conv2.conv1.bias is not None:
            dest_module.conv2.conv1.bias = nn.Parameter(source_module.conv2.conv1.bias.to(torch.float32))
        if dest_module.conv2.conv2.bias is not None:
            dest_module.conv2.conv2.bias = nn.Parameter(source_module.conv2.conv2.bias.to(torch.float32))
        if dest_module.conv2.use_noise:
            dest_module.conv2.noise_const = source_module.conv2.conv2.noise_const
            dest_module.conv2.noise_strength = source_module.conv2.conv2.noise_strength

        setattr(dest.synthesis, f"b{res}", dest_module)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to model (pkl).", required=True)
    parser.add_argument("--origs-dir", type=Path, help="Path to origs directory.", required=True)
    parser.add_argument("--masks-dir", type=Path, help="Path to masks directory.", required=True)
    parser.add_argument("--output-dir", type=Path, help="Output directory.", required=True)
    parser.add_argument("--resolution", type=int, help="Model resolution.", required=True)
    parser.add_argument("--num-samples", type=int, help="Num of samples to test on.", default=10)
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_out_dir = args.output_dir / "samples"
    samples_out_dir.mkdir(parents=True, exist_ok=True)
    original_out_dir = samples_out_dir / "original_result"
    converted_out_dir = samples_out_dir / "converted_result"
    original_out_dir.mkdir(parents=True, exist_ok=True)
    converted_out_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png"}
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(os.path.join(args.origs_dir, "**", f"*{img_extension}"), recursive=True)

    img_paths = sorted(img_paths)[:args.num_samples]

    if Path(args.model_path).suffix == ".pkl":
        with open(args.model_path, 'rb') as f:
            resume_data = pickle.Unpickler(f).load()
        source_G = resume_data["G_ema"]
    else:
        raise Exception("Not implemented")

    source_G = source_G.eval()
    source_G = source_G.to("cpu")

    model = MIGANGenerator(resolution=args.resolution)
    model = model.eval()

    print("Copying weights...")
    copy_weights(source_G, model, resolution=args.resolution)

    print("Calculating diff statistic...")
    diff_sum = 0
    for img_path in tqdm(img_paths):
        mask_path = os.path.join(args.masks_dir, f"{Path(img_path).stem}.png")

        # Read image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        img = (transforms.ToTensor()(img) - 0.5) * 2
        mask = transforms.ToTensor()(mask)[0:1, :, :]
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # Prepare inputs
        x = torch.cat([mask - 0.5, img * mask], dim=1)

        original_result = source_G(x, noise_mode="const")
        converted_result = model(x)
        diff_sum += torch.sum(~torch.isclose(original_result, converted_result, rtol=0.001))

        converted_output = (img * mask + converted_result * (1 - mask))[0]
        converted_output = (converted_output.permute(1, 2, 0) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        converted_output = converted_output.detach().to('cpu').numpy()
        Image.fromarray(converted_output).save(converted_out_dir / f"{Path(img_path).stem}.png")

        original_output = (img * mask + original_result * (1 - mask))[0]
        original_output = (original_output.permute(1, 2, 0) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        original_output = original_output.detach().to('cpu').numpy()
        Image.fromarray(original_output).save(original_out_dir / f"{Path(img_path).stem}.png")

    diff_mean = diff_sum / args.num_samples
    print(f"Average diff %: {diff_mean / (args.resolution ** 2) * 100:.2f}%")

    print("Exporting ONNX model...")
    dummy_input = torch.randn(1, 4, args.resolution, args.resolution, device="cpu")
    input_names = ["input"]
    output_names = ["output"]
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy_input, args.output_dir / "models" / "migan.onnx", verbose=False,
                      input_names=input_names, output_names=output_names, opset_version=12)
    print("ONNX model exported")

    print("Exporting pt model...")
    dummy_input = torch.ones((1, 4, args.resolution, args.resolution), dtype=torch.float32, device="cpu")
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "models" / "migan.pt")
    m = torch.jit.trace(model, dummy_input)
    m.save(args.output_dir / "models" / "migan_traced.pt")
    print(".pt models exported")


if __name__ == "__main__":
    main()
