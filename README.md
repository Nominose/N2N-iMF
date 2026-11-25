# N2N-iMF

Training and prediction code for **Noise2Noise Flow Model for Self-Supervised CT Denoising with Expedited Inference**.

N2N-iMF learns to generate one noisy CT observation conditioned on a paired noisy observation. At prediction time it generates K independent samples and averages them in Hounsfield units (HU). Each sample takes NFE network evaluations.

## Installation

Use Python 3.10 or newer. For CUDA, install the PyTorch build that matches your GPU environment in the virtual environment before installing the remaining dependencies.

```sh
git clone https://github.com/Nominose/N2N-iMF.git
cd N2N-iMF
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The commands use CUDA when available. Pass `--device cpu` to run on a CPU.

## Training data

Use axial 3D NIfTI volumes in HU. The third array dimension must index consecutive slices. List the training volumes in a CSV file with one row per scan. Relative paths are resolved from the CSV directory.

For **simulated brain CT and PCCT**, supply one noisy volume per scan. The two neighbors of an interior slice form the input channels, and the central noisy slice is the training target.

```csv
volume
scans/noisy_brain.nii.gz
```

For **abdominal CT**, supply registered reconstructions from the odd and even projections of the same acquisition. The two reconstructions exchange the roles of input and target during training.

```csv
odd,even
scans/abdominal_odd.nii.gz,scans/abdominal_even.nii.gz
```

Provide reconstructed HU values directly. The loader applies the study's intensity mapping and clipping for each dataset. The brain CT histogram mapping is included in `n2nimf/assets/`.

## Train

```sh
python train.py --dataset brain --manifest data/brain_train.csv --output runs/brain_base.pt
python train.py --dataset pcct --manifest data/pcct_train.csv --output runs/pcct_base.pt
python train.py --dataset abdominal --manifest data/abdominal_train.csv --output runs/abdominal.pt
```

Base training uses Adam with a learning rate of 0.0001 for 200 epochs. The generator predicts the average velocity along a conditional flow. A second output head supplies the auxiliary velocity loss during training. Each completed epoch updates a checkpoint containing the generator, its exponential moving average (EMA), and the optimizer state. To continue an interrupted run, pass `--resume` with the same output file, data and training settings. When resuming fine-tuning, replace `--fine-tune-from` with `--resume`.

The brain CT and PCCT models receive a second training stage. A PatchGAN discriminator compares high-frequency residuals of generated and observed noisy slices. Set the number of fine-tuning epochs for your training run, for example:

```sh
python train.py --dataset brain --manifest data/brain_train.csv --fine-tune-from runs/brain_base.pt --epochs 50 --output runs/brain.pt
python train.py --dataset pcct --manifest data/pcct_train.csv --fine-tune-from runs/pcct_base.pt --epochs 50 --output runs/pcct.pt
```

## Predict

```sh
python predict.py --dataset brain --checkpoint runs/brain.pt \
  --volume data/brain_test.nii.gz --output brain_denoised.nii.gz

python predict.py --dataset pcct --checkpoint runs/pcct.pt \
  --volume data/pcct_test.nii.gz --output pcct_denoised.nii.gz

python predict.py --dataset abdominal --checkpoint runs/abdominal.pt \
  --odd data/abdominal_odd.nii.gz --even data/abdominal_even.nii.gz \
  --output abdominal_denoised.nii.gz
```

For brain CT the defaults are NFE = 3 and K = 10. PCCT uses NFE = 5 and K = 20. Abdominal CT uses NFE = 3 and K = 10 **for each split reconstruction**, so the denoised slice averages 20 generated samples. At the first and last slices of a brain volume, prediction repeats the boundary slice for the missing neighbor.

Prediction reads the EMA generator by default. Use `--weights raw` for the online generator, `--k` and `--nfe` to set the sampling budget, and `--seed` for a repeatable draw on the same device. The output is a floating-point NIfTI volume in HU on the input grid. Supply a checkpoint produced by training or a matching two-head study checkpoint. The repository distributes the source code and brain intensity mapping.

The U-Net includes components adapted from Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). The upstream MIT notice is in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
