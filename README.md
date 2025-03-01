# Stepformer: Measure-Based Transformer for DDR Chart Generation

Stepformer is a deep learning-based system for automatic DDR/StepMania chart generation. Leveraging measure-based tokenization and a Transformer-based sequence-to-sequence architecture, it integrates insights from MT3 for music transcription and DDC for baseline performance.

## Project Structure

.
├── data/          # Contains raw audio & simfiles, processed token data, and optional features
│   ├── raw/       # Raw audio files & simfiles
│   ├── processed/ # Tokenized music and dance data
│   └── features/  # Precomputed or intermediate features
├── ddc/           # Reference code from the DDC baseline
├── mt3/           # Transcription tools based on MT3
├── simfile/       # Simfile parsing libraries/tools
├── src/           # Source code for data processing, models, training, and utilities
├── models/        # Saved model checkpoints and model architectures
├── notebooks/     # Jupyter notebooks for exploratory analysis
├── scripts/       # CLI or automation scripts
├── tests/         # Unit tests
└── README.md
```

## Setup

You can set up the project environment using one of the following methods:

### Option 1: Conda Environment

```bash
conda create -n stepformer python=3.11
conda activate stepformer
pip install -r requirements.txt
```

### Option 2: Python Virtual Environment (venv)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Development

- Data preprocessing scripts are in `src/data/`
- Model implementations are in `src/models/`
- Utility functions are in `src/utils/`

## License

MIT License
