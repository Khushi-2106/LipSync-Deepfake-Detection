# Motion-Aware LipFD: DeepFake Detection using Optical Flow and Region-Aware Attention

This project extends the LipFD framework proposed in *“Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes”* by integrating explicit motion modeling using optical flow.

The proposed framework combines:

- CLIP-based visual feature extraction
- Optical flow-based temporal motion analysis
- Region-aware attention over head, face, and lip regions
- Motion-visual feature fusion for DeepFake classification

The goal is to improve detection of lip-synchronization inconsistencies in manipulated videos by leveraging both spatial and temporal information.

---

# Project Architecture

## Pipeline Overview

Input Video
↓
Frame Extraction + Optical Flow Computation
↓
CLIP Visual Encoder + Motion Encoder
↓
Attention-based Region Fusion
↓
Region-Aware Classification
↓
Real / Fake Prediction

---

# Features

- Optical flow-based temporal inconsistency modeling
- Region-specific motion analysis (head, face, lips)
- Attention-based motion fusion
- CLIP visual embeddings
- Region-aware DeepFake detection
- Fine-tuning support from pretrained LipFD checkpoints

---

# Requirements

```bash
conda create -n LipFD python=3.10
conda activate LipFD

pip install -r requirements.txt
```

Dataset Structure

The dataset should follow the structure below:
datasets
├── train
│   ├── 0_real
│   └── 1_fake
├── val
│   ├── 0_real
│   └── 1_fake
└── wav
    ├── 0_real
    └── 1_fake

Each folder contains:

Real videos
Fake videos
Corresponding audio files (.wav)

Dataset Preprocessing

The preprocessing pipeline performs:

Frame extraction
Spectrogram generation
Optical flow computation
Region cropping
Sample generation

Run preprocessing:

```bash
python preprocess.py
```

Generated samples are stored in:

datasets/AVLips
├── 0_real
└── 1_fake

Training
Train from Scratch
```bash
python train.py
```

Fine-tuning from Pretrained LipFD Checkpoint
```bash
python train.py \
--fine-tune \
--pretrained_model ./checkpoints/ckpt.pth \
--batch_size 2 \
--lr 1e-6 \
--gpu_ids 0
```

Validation
```bash
python validate.py \
--real_list_path ./datasets/val/0_real \
--fake_list_path ./datasets/val/1_fake \
--ckpt ./checkpoints/ckpt.pth \
--gpu 0 \
--batch_size 2
```

Acknowledgements

This work is based on the LipFD framework proposed in:

Liu et al., “Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes”, NeurIPS 2024.

Citation
```bash
@inproceedings{liu2024lips,
 author = {Liu, Weifeng and She, Tianyi and Liu, Jiawei and Li, Boheng and Yao, Dongyu and Liang, Ziyou and Wang, Run},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes},
 year = {2024}
}
```

Authors
Khushi Sharma
Himanshi
