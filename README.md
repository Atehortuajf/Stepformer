# Stepformer

A deep learning-based system for automatic DDR/StepMania chart generation, inspired by Dance Dance Convolution.

## Project Structure
```
.
├── data/
│   ├── raw/         # Raw audio files and step charts
│   ├── processed/   # Preprocessed data
│   └── features/    # Extracted audio features
├── models/          # Saved model checkpoints
├── src/
│   ├── data/        # Data loading and preprocessing
│   ├── models/      # Model architectures
│   └── utils/       # Utility functions
├── notebooks/       # Jupyter notebooks for analysis
└── tests/          # Unit tests
```

## Setup

1. Create and activate conda environment:
```bash
conda create -n stepformer python=3.11
conda activate stepformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development

- Data preprocessing scripts are in `src/data/`
- Model implementations are in `src/models/`
- Utility functions are in `src/utils/`

## License

MIT License
