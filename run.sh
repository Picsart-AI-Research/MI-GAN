#!/bin/bash

# Ablation on 256 Places 2

if [ $1 = "ablation_dw_places256" ]; then               # Baseline
    exp=ablation_dw_places256
    sigmodel=ablation_dw_places256
elif [ $1 = "ablation_dw_reparam_places256" ]; then     # Baseline + Re-param
    exp=ablation_dw_reparam_places256
    sigmodel=ablation_dw_reparam_places256
elif [ $1 = "ablation_dw_reparam_kd_places256" ]; then  # Baseline + Re-param + KD
  exp=ablation_dw_reparam_kd_places256
  sigmodel=ablation_dw_reparam_kd_places256
fi

# Final MI-GAN experiments

if [ $1 = "migan_places256" ]; then               
    exp=migan_places256
    sigmodel=migan_places256
elif [ $1 = "migan_ffhq256" ]; then 
    exp=migan_ffhq256
    sigmodel=migan_ffhq256
elif [ $1 = "migan_places512" ]; then
  exp=migan_places512
  sigmodel=migan_places512
fi


if [ $2 = "gpu0" ]; then
    gpu="0"
    port="11211"
    siggpu=1gpu
elif [ $2 = "gpu1" ]; then
    gpu="1"
    port="11201"
    siggpu=1gpu
elif [ $2 = "gpu2" ]; then
    gpu="2"
    port="11202"
    siggpu=1gpu
elif [ $2 = "gpu3" ]; then
    gpu="3"
    port="11203"
    siggpu=1gpu
elif [ $2 = "gpu4" ]; then
    gpu="4"
    port="11204"
    siggpu=1gpu
elif [ $2 = "gpu5" ]; then
    gpu="5"
    port="11205"
    siggpu=1gpu
elif [ $2 = "gpu6" ]; then
    gpu="6"
    port="11206"
    siggpu=1gpu
elif [ $2 = "gpu7" ]; then
    gpu="7"
    port="11207"
    siggpu=1gpu
elif [ $2 = "gpu01" ]; then
    gpu="0 1"
    port="11210"
    siggpu=2gpu
elif [ $2 = "gpu23" ]; then
    gpu="2 3"
    port="11212"
    siggpu=2gpu
elif [ $2 = "gpu45" ]; then
    gpu="4 5"
    port="11214"
    siggpu=2gpu
elif [ $2 = "gpu67" ]; then
    gpu="6 7"
    port="11216"
    siggpu=2gpu
elif [ $2 = "gpu0123" ]; then
    gpu="0 1 2 3"
    port="21220"
    siggpu=4gpu
elif [ $2 = "gpu4567" ]; then
    gpu="4 5 6 7"
    port="11224"
    siggpu=4gpu
elif [ $2 = "gpu0167" ]; then
    gpu="0 1 6 7"
    port="10167"
    siggpu=4gpu
elif [ $2 = "gpu01234567" ]; then
    gpu="0 1 2 3 4 5 6 7"
    port="11230"
    siggpu=8gpu
fi

python main.py \
    --experiment $exp \
    --dscache 0 \
    --gpu $gpu \
    --port $port \
    --seed 0 \
    --signature $sigmodel $siggpu mau
