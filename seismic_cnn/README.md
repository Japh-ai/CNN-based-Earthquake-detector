# Seismic Event Detector

A 1D CNN trained to classify 60-second seismograms as **earthquake** or **noise** using the [STEAD dataset](https://github.com/smousavi05/STEAD).

## Project structure

```
seismic_cnn/
├── dataset.py   # loads STEAD HDF5/CSV, normalizes, returns a PyTorch Dataset
├── model.py     # 1D CNN architecture (3 conv blocks + global avg pool + classifier)
├── train.py     # training loop, evaluation, loss tracking
└── main.py      # full pipeline: load → train → evaluate → save
```

## Data

Download the STEAD dataset chunks from the [STEAD repository](https://github.com/smousavi05/STEAD):
- `chunk2.hdf5` / `chunk2.csv` — earthquake waveforms
- `chunk1.hdf5` / `chunk1.csv` — noise waveforms

Place them in `../data/` relative to this folder.

## Usage

Edit the configuration block at the top of `main.py`, then run it cell by cell in VS Code interactive mode or as a script:

```bash
python main.py
```

## Model

Input: `(batch, 3, 6000)` — 3-component seismogram at 100 Hz (60 seconds)  
Output: probability of being an earthquake (0 = noise, 1 = earthquake)

| Layer | Output shape |
|---|---|
| Conv block 1 | (batch, 32, 3000) |
| Conv block 2 | (batch, 64, 1500) |
| Conv block 3 | (batch, 128, 300) |
| Global avg pool | (batch, 128) |
| Classifier | (batch, 1) |

## Results

Trained on 8000 samples per class (16,000 total), 15 epochs:

| Metric | Score |
|---|---|
| Test accuracy | 99% |
| Precision | 0.99 |
| Recall | 0.99 |
| F1 | 0.99 |
