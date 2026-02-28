# Paper Code Analysis (Landfill Semantic Segmentation)

This repository contains notebook-exported Python scripts for the thesis pipeline.  
The implementation intent is clear, but the scripts are not directly runnable as local production code without cleanup.

## What The Paper Code Is Doing

The main pipeline in files like `landfillpipeline_resnet_fcn.py`, `landfillpipeline_resnet_unet.py`, `landfillpipeline_vgg_fcn.py`, and `landfillpipeline_vgg_unet.py` is:

1. Load multispectral or pansharpened TIFF imagery (`4` or `8` channels).
2. Map CSV metadata (`Idx`, `Image Index`, `IsLandfill`, `json index`) to image and polygon labels.
3. Build binary segmentation masks from GeoJSON polygons (landfill footprint).
4. Convert spectral channels into model input:
   - 8-band: `[4,2,1,6]` (R,G,B,NIR1)
   - 4-band: `[2,1,0,3]` (R,G,B,NIR)
5. Train FCN or UNet decoder heads with VGG16 or ResNet34 encoders.
6. Evaluate with pixel accuracy, IoU, precision, recall, specificity.

## Why Existing Scripts Fail Locally

The exported `.py` files still contain notebook/Colab-only constructs:

- shell magics and apt installs (for example `!pip ...`, `!apt-get ...`)
- hardcoded Dropbox downloads during import/execution
- hardcoded relative dataset paths
- plotting and interactive cells tied to notebook flow
- no CLI entrypoint for dataset paths/hyperparameters

Because of this, running a script directly in a local terminal usually fails before training starts.

## Local Runnable Implementation Added

A new runner was added:

- `Code/Source/run_paper_resnet_fcn.py`

It keeps the paper-aligned behavior while making it local and reproducible:

1. Explicit CLI arguments for dataset paths and hyperparameters.
2. CSV + GeoJSON + TIFF data flow preserved.
3. Same 4-channel band-selection logic used by the original code.
4. ResNet34-FCN architecture aligned with original decoder design.
5. Train/validation split with stratification by `IsLandfill`.
6. Saves:
   - best checkpoint (`best_resnet34_fcn.pth`)
   - epoch metrics (`metrics.csv`)
   - sample predictions (`sample_predictions.png`)
7. Mac runtime defaults applied:
   - `--num-workers 3`
   - `--torch-threads 3`
   - `--max-ram-gb 11` (soft cap, best effort)

## Expected Dataset Layout

You can point the runner to your own paths, but the CSV should include:

- `Idx`
- `Image Index`
- `IsLandfill`
- optionally `json index` (recommended)

If `json index` is absent, the runner tries `<image_stem>.geojson` then `<image_stem>.json`.
