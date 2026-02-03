# CROPS: Source-Free UDA for Crop-Type Semantic Segmentation

This repository implements **CROPS** for **source-free unsupervised domain adaptation (SF-UDA)** in satellite-image crop-type semantic segmentation. A segmentation model is first trained on a labeled **source** domain and then adapted to an **unlabeled target** domain using only pretrained parameters and target images (no source images or labels are used during adaptation).

## Method overview

CROPS combines three components:

- **Image pre-alignment (NDVI-QM)**: each domain is converted to an NDVI-based quality mosaic that emphasizes peak vegetation conditions and suppresses low-quality observations, providing a stabilized representation for downstream segmentation.

- **Confidence-aware easy/hard patch partition**: a source-pretrained model infers class probabilities and uncertainty on the target composite. Target patches are split into **easy** (high-confidence) and **hard** (low-confidence) subsets to form an explicit easy-to-hard curriculum.

- **Semantic Prototypical Contrastive Learning (SPCL)**: adaptation is performed in a teacher–student (EMA) framework. Class prototypes are estimated from easy regions and used as semantic anchors. Hard-region features are aligned to these prototypes via prototype-based contrastive learning, coupled with confidence-weighted ClassMix supervision.

SPCL is backbone-agnostic. This repository provides an instantiation with **TransUNet**, and the same SPCL module can be integrated with other segmentation backbones.

## Repository entry points

- Main runner (adaptation + inference): `run_train_spcl_loopPlus_lowRAM.py`  
- Easy/Hard partition builder: `build_block_easyhard_h5v52.py`  
- Streaming inference (GeoTIFF writer): `infer_steamingPlus.py`  
- UDA / SPCL core: `uda/`  
- Backbone factory: `models.py`  
- Config: `config_HDF5loop_01Test.py`

## Requirements

- Python 3.8+
- PyTorch (CUDA build recommended)
- GDAL (GeoTIFF I/O and streaming inference)
- numpy, h5py, tqdm, etc.

## Data preparation

CROPS is designed for block-based training/adaptation (commonly 224×224). You will typically prepare:

- **Source domain**: image blocks + label blocks (for supervised pretraining)
- **Target domain**: image blocks (unlabeled, for adaptation)
- **Composites**: NDVI-QM (or your pipeline’s stabilized composite representation)

All file locations and run-time options are provided through the configuration returned by `get_config(config_suffix)`.

## How to run

1) Configure your experiment in `config_HDF5loop_01Test.py` (via `get_config(config_suffix)`), including:
- pretrained weights to load
- target-domain HDF5 blocks and the easy/hard split
- test image directory and output directory

2) Set the suffix list in `run_train_spcl_loopPlus_lowRAM.py`, then run:
```bash
python run_train_spcl_loopPlus_lowRAM.py
