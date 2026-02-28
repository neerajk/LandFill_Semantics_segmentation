# Automatic Detection of Landfill Using Deep Learning



## Local Run (ResNet34-FCN)

The original scripts in `Code/Source` are notebook exports and include Colab-only commands.  
Use the local runner below instead:

`Code/Source/run_paper_resnet_fcn.py`

### Install

```bash
pip install -r Code/Source/requirements_runner.txt
```

### Train

```bash
python Code/Source/run_paper_resnet_fcn.py \
  --images-dir /ABS/PATH/HR_TIF_Files \
  --labels-csv /ABS/PATH/PanSharpenedData.csv \
  --json-dir /ABS/PATH/LandfillCoordPolygons \
  --output-dir /ABS/PATH/runs/resnet34_fcn \
  --num-workers 3 \
  --torch-threads 3 \
  --max-ram-gb 11 \
  --epochs 50 \
  --batch-size 4
```

### Outputs

- `best_resnet34_fcn.pth` (best validation checkpoint)
- `metrics.csv` (epoch-wise training/validation metrics)
- `sample_predictions.png` (quick qualitative validation preview)

For implementation notes and mapping to the original thesis code, see:
`Code/Source/PAPER_CODE_ANALYSIS.md`
