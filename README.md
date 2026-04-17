

## Citation

If you find this work useful, please cite the paper:

```
@article{liu2025spatial,
  title={Spatial frequency domain transformation for infrared small target detection},
  author={Liu, Y. and Tu, B. and Liu, B. and He, Y. and Li, J. and Plaza, A.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```

## Features

- Support for multiple datasets: NUAA-SIRST, NUDT-SIRST, IRSTD-1K, etc.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- scikit-image
- matplotlib
- einops
- thop
- tqdm


## Dataset Preparation

Place datasets in the `dataset/` directory with the following structure:

```
dataset/
├── NUAA-SIRST/
│   ├── images/
│   ├── masks/
│   └── img_idx/
│       ├── train_NUAA-SIRST.txt
│       └── test_NUAA-SIRST.txt
├── NUDT-SIRST/
│   └── ...
└── IRSTD-1K/
    └── ...
```

## Training

```bash
python train.py --model_names SFDTNet --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 800
```

Main parameters:
- `--model_names`: Model name 
- `--dataset_names`: Dataset name 
- `--batchSize`: Batch size 
- `--patchSize`: Training patch size 
- `--nEpochs`: Number of epochs 
- `--lr`: Learning rate 

## Testing

```bash
python testing.py
```

```

