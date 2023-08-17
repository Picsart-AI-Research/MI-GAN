# MI-GAN

This repository is the official implementation of MI-GAN.

**MI-GAN: A Simple Baseline for Image Inpainting on Mobile Devices**
</br>
Andranik Sargsyan,
Shant Navasardyan,
Xingqian Xu,
[Humphrey Shi](https://www.humphreyshi.com)
</br>

<p align="center">
<img src="__assets__/github/teaser.png" width="800px"/>  
<br>
<em>Our method MI-GAN can produce plausible results both on complex scene images as well as on face images. The bubble chart on the right shows
the advantage of our network over state-of-the-art approaches. The size of the bubble signifies the relative number of parameters of each
approach and the number inside of the bubble shows the absolute number of model parameters. Our approach achieves a low FID, while
being one order of magnitude smaller and faster than recent SOTA approaches.</em>
</p>

## Prepare environment

Prepare the environment (Python>=3.8 is needed):
```commandline
sudo apt install python3-venv
python3 -m venv venv
source ./venv/bin/activate
pip install pip --upgrade
pip install -r requirements.txt
```

Download pre-trained MI-GAN models from [here](https://drive.google.com/drive/folders/1xNtvN2lto0p5yFKOEEg9RioMjGrYM74w?usp=share_link) and put into `./models` directory.
If you also want to test with Co-Mod-GAN models, download pre-trained models from [here](https://drive.google.com/drive/folders/1VATyNQQJW2VpuHND02bc-3_4ukJMHQ44?usp=share_link) and put into `./models` directory.

## Quick Test
For getting visual results, use `demo.py` script like this

```
python -m scripts.demo \
    --model-name migan-512 \
    --model-path ./models/migan_512_places2.pt \
    --images-dir ./examples/places2_512_object/images \
    --masks-dir ./examples/places2_512_object/masks \
    --output-dir ./examples/places2_512_object/results/migan \
    --device cuda \
    --invert-mask
```
For running with Co-Mod-GAN, change `--model-name` and `--model-path` accordingly.
Also, depending on what the white color means in your masks, you may or may not need `--invert-mask` flag.
By default white color denotes the known region.

To run inference on provided FFHQ samples, run

```
python -m scripts.demo \
    --model-name migan-256 \
    --model-path ./models/migan_256_ffhq.pt \
    --images-dir ./examples/ffhq_256_freeform/images \
    --masks-dir ./examples/ffhq_256_freeform/masks \
    --output-dir ./examples/ffhq_256_freeform/results/migan \
    --device cuda
```

You can find more example images and masks in `examples` directory, and modify the above command to reproduce all
MI-GAN and Co-Mod-GAN results presented in the Supplementary material.

**Note:** With our provided free-form masks, `--invert-mask` option **should not** be used with the command.

## Evaluation
For MI-GAN evaluation on Places2 download Places365-Standard 256x256 and high resolution validation images 
from [Places2 official website](http://places2.csail.mit.edu/download.html).

Then, for evaluating MI-GAN FID and LPIPS on Places2 256x256 run
```
python -m scripts.evaluate_fid_lpips \
    --model-name migan-256 \
    --model-path ./models/migan_256_places2.pt \
    --real-dir PATH_TO_PLACES256_VALIDATION_IMAGES
```

For evaluating MI-GAN FID and LPIPS on Places2 512x512 run
```
python -m scripts.evaluate_fid_lpips \
    --model-name migan-512 \
    --model-path ./models/migan_512_places2.pt \
    --real-dir PATH_TO_PLACES512_VALIDATION_IMAGES
```

For MI-GAN evaluation on FFHQ 256x256 download FFHQ dataset from the [official source](https://github.com/NVlabs/ffhq-dataset), copy the first 10,000 samples
into a separate validation directory, and then run
```
python -m scripts.evaluate_fid_lpips \
    --model-name migan-256 \
    --model-path ./models/migan_256_ffhq.pt \
    --real-dir PATH_TO_FFHQ_VALIDATION_IMAGES
```

The above commands will generate new free-form masks on the fly for doing the inference, but you can also download
our pre-generated masks for Places2 256x256, Places2 512x512 and FFHQ 256x256 validation sets from
[here](https://drive.google.com/drive/folders/1LG34GoCKX5ZnkOjLckFaF2v3ctCAYvfk?usp=share_link)
and use those for metrics calculations by providing an additional `--mask-dir` argument in above commands, where
`--mask-dir` will point to the respective pre-generated mask directory.

Besides, you can generate your own free form masks using our script like so
```
python -m scripts.generate_masks \
    --images-dir PATH_TO_ORIGINAL_IMAGES_DIR \
    --output-dir MASK_OUTPUT_PATH \
    --resolution MASK_RESOLUTION
```

For Co-Mod-GAN metrics calculation change `--model-name` and `--model-path` accordingly. For example, in order to calculate Co-Mod-GAN metrics on FFHQ 256x256, run
```
python -m scripts.evaluate_fid_lpips \
    --model-name comodgan-256 \
    --model-path ./models/comodgan_256_ffhq.pt \
    --real-dir PATH_TO_FFHQ_VALIDATION_IMAGES
```

If your GPU doesn't have enough memory for running the evaluation with default `batch-size`, you can reduce batch size
by providing `--batch-size` argument with a low batch size in above commands. 

### FLOPs and parameter count calculation
In order to calculate FLOPs and parameter counts for MI-GAN and Co-Mod-GAN models you can run
```
python -m scripts.calculate_flops
```

## Training

In order to smoothly run the training without changes, you need to have the following directory structure for datasets (data folder should be in the root project folder).

```
data
 | 
 +-- ffhq
 |  |  
 |  +-- ffhq256x256.zip
 |    
 +-- Places2
 |  |  
 |  +-- train_256
 |  +-- val_256
 |  +-- train_512
 |  +-- val_512
```

Also you need to download pre-trained Co-Mod-GAN models from [here](https://drive.google.com/drive/folders/1VATyNQQJW2VpuHND02bc-3_4ukJMHQ44?usp=share_link)
and put into `models` directory. `models` directory should have the following structure:
```
models
|  
| +-- comodgan_256_ffhq.pt
| +-- comodgan_256_places2.pt
| +-- comodgan_512_places2.pt
```

Then, for training MI-GAN 256 on Places 2 using 8 GPUs, run
```
./run.sh migan_places256 gpu01234567
```


For training MI-GAN 512 on Places 2 using 8 GPUs, run
```
./run.sh migan_places512 gpu01234567
```


For training MI-GAN 256 on FFHQ using 8 GPUs, run
```
./run.sh migan_ffhq256 gpu01234567
```

You can choose to run on specific GPUs. For example, if you want to run on GPUs 0,1,2,3, run the above commands with argument gpu0123. For more details, please see `run.sh` helper script.

After training you can generate the inference model from the saved best checkpoint with this command:
```bash
python -m scripts.export_inference_model --model-path PATH_TO_BEST_PLACES512_PKL --origs-dir PATH_TO_PLACES512_IMAGES --masks-dir PATH_TO_PLACES512_MASKS --output-dir ./exported_models/migan_places512 --resolution 512 
```
The above example is for *places2-512* model. You need to change the `--resolution` argument to `256` for 256 models.
The exported onnx and pt models and sample results will be saved in the specified `--output-dir` directory.

## Acknowledgement

Our code is built upon [SH-GAN](https://github.com/SHI-Labs/SH-GAN).
