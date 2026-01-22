# Pix2Pix-MFP: Image-to-Image Translation with Multi-Feature Perceptual Loss

This project extends the standard [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) framework by integrating a **Multi-Feature Perceptual (MFP) Loss**. This novel loss function leverages a pre-trained Vision Transformer (ViT, specifically DINO) to enforce both global semantic consistency and structural integrity between generated and ground-truth images.

## Features

- **MFP Loss Module**: Located in `MFP_loss/LossM.py`. It extracts deep features using a DINO-pretrained ViT backbone.
- **Dual Consistency Check**:
  - **Global Semantic Consistency**: Minimizes the distance between [CLS] tokens of the generated and real images (`lambda_global_cls`).
  - **Structural Self-Similarity**: Minimizes the difference in self-similarity keys (from intermediate layers) to preserve geometric structure (`lambda_global_ssim`).
- **Seamless Integration**: Integrated directly into the `Pix2PixModel` generator optimization step.
- **Configurable**: Easy tuning of loss components via a YAML configuration file.

## Getting Started

### Dataset
The dataset can be accessed via the [link](https://pan.baidu.com/s/15UgCcFBsl13UmjGi13yanw?pwd=tdx5).
### Prerequisites

- Linux or macOS
- Python 3
- PyTorch >= 1.7
- NVIDIA GPU + CUDA
- `pyyaml` (for loading configuration)

### Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/your-username/pix2pix-mfp.git
cd pix2pix-mfp
pip install -r requirements.txt
```

*(Note: Ensure you have `torch` and `torchvision` installed compatible with your CUDA version)*

### Configuration (`options/config.yaml`)

The behavior of the MFP loss is controlled by `options/config.yaml`. You can adjust the weights of the different loss components:

```yaml
# MFP Loss parameters
dino_model_name: dino_vitb8  # ViT Backbone: ['dino_vitb8', 'dino_vits8', 'dino_vitb16', 'dino_vits16']
lambda_global_cls: 10        # Weight for Global Semantic (CLS token) loss
lambda_global_ssim: 1        # Weight for Structural (Self-Similarity) loss
```

## Training

To train the model with the MFP loss, you must specify `--loss_function MFP` and optionally adjust `--lambda_MFP`.

### Basic Command

```bash
python train.py \
  --dataroot ./datasets/your_dataset \
  --name experiment_name \
  --model pix2pix \
  --direction BtoA \
  --loss_function MFP \
  --lambda_MFP 1.0 \
  --gpu_ids 0
```

### Key Arguments

- `--loss_function MFP`: **[Required]** Activates the Multi-Feature Perceptual loss.
- `--lambda_MFP`: Weight of the MFP loss added to the standard GAN + L1 loss (default: 1.0).
- `--lambda_L1`: Weight for L1 loss (default: 100.0).
- `--dataroot`: Path to the dataset training images.

### Example

Train on the `facades` dataset (assuming standard structure) with MFP loss:

```bash
python train.py --dataroot ./datasets/facades --name facades_mfp --model pix2pix --loss_function MFP --lambda_MFP 0.5
```

## Code Structure

- **`MFP_loss/`**: Contains the core logic for the new loss.
  - `LossM.py`: Implements `LossM` class which computes the weighted sum of CLS and SSIM losses.
- **`models/pix2pix_model.py`**: Modified standard Pix2Pix model.
  - Initializes `LossM` when `opt.loss_function == 'MFP'`.
  - In `backward_G`, calculates `loss_G_MFP` and adds it to the total generator loss.
- **`options/config.yaml`**: centralized configuration for ViT backbone selection and loss weights.

## Acknowledgments

This code is built upon the excellent [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
