# Complete Implementation Plan

This document provides a high-level roadmap and technical details for implementing the **Stepformer** project, which applies a measure-based tokenization scheme using the MT3 transcription approach and a Transformer-based sequence-to-sequence architecture for DDR chart generation.

---

## 1. Repository & Environment Setup

1. **Project Structure**  
   Here's a recommended folder layout:
   ```
   stepformer/
   ├── data/
   │   ├── raw/           # Raw audio files & simfiles
   │   ├── processed/     # Tokenized music and dance data
   │   └── features/      # (Optional) Precomputed or intermediate features
   ├── ddc/               # Reference code from DDC
   ├── mt3/               # Reference code from MT3
   ├── simfile/           # (Optional) for simfile parsing if not globally installed
   ├── src/
   │   ├── data/          # Data loading & preprocessing scripts
   │   ├── models/        # Model definitions (Transformer, etc.)
   │   ├── training/      # Training loops, evaluation scripts
   │   └── utils/         # Utility functions
   ├── notebooks/         # Jupyter notebooks for exploratory work
   ├── scripts/           # CLI or automation scripts
   ├── tests/             # Unit tests
   └── README.md
   ```

2. **Environment**  
   - Python 3.11 (or ~3.10).  
   - Recommended: Create a conda environment (or virtualenv).  
   - Install main dependencies (example):
     ```bash
     conda create -n stepformer python=3.11
     conda activate stepformer
     pip install torch~=2.0 wandb simfile
     # If using the MT3 code directly: 
     # pip install -e path/to/mt3  (if you have a local copy)
     ```
   - (Optional) If the project includes additional packages (e.g., for specialized audio processing), add them to `requirements.txt`.

---

## 2. Data Collection & Preprocessing

### 2.1 Acquire Audio + Step Charts
- charts are found in data/raw/ inside of song packs. each song pack has a folder for each song with the song audio and a simfile with one or more charts.

### 2.2 Simfile Parsing & Chart Representation
- Use `simfile` library (or custom parser) to load simfiles:
  - Extract BPM changes, measure divisions, steps per measure, etc.
  - For each chart, note the exact time frames or measure subdivisions for each arrow press.

### 2.3 Music Transcription (MT3 or Similar)
1. Either:
   - **Option A**: Run a pretrained MT3 to transcribe each audio into a note-level or drum-level sequence (converted to tokens).  
   - **Option B**: Use a simpler offline transcription approach if MT3 is too complex.
2. Align transcribed events to the simfile’s measure structure:
   - For each measure or sub-measure, gather the note events that fall within that time range.
   - Create tokens such as `[measure_number, subdiv_index, note_pitch, note_onset, instrument_label]`.

### 2.4 Dance Tokenization
- Convert DDR steps into a parallel sequence of tokens with matching measure and subdivision indexing.
   - For each sub-measure, record which arrows are pressed (or no-op if none).
   - If DDR charts use multiple arrows at once, encode them as combined tokens (e.g., “Up+Left”).
   - (Optional) If charts have hold notes, represent hold starts and ends with special tokens.

### 2.5 Store Token Sequences
- Save music tokens and corresponding dance tokens to `data/processed/`.  
  - Example format: JSON, CSV, or PyTorch `pt` file with structure:
    ```json
    {
      "track_id": "songA",
      "music_tokens": [12, 45, 67, ...],
      "dance_tokens": [3, 7, 15, ...]
    }
    ```

---

## 3. Model Architecture (PyTorch)

### 3.1 Overview
We will implement a Transformer-based seq2seq model that encodes music tokens and decodes dance tokens.

### 3.2 Encoder-Decoder Definition
- In `src/models/stepformer_transformer.py`, create a class `Stepformer`.
- Include:
  1. **Embedding layers** for music and dance tokens.  
  2. **Positional Encoding** (either absolute or relative).  
  3. **Transformer Encoder** (multilayer, e.g., 6 layers, 8 heads).  
  4. **Transformer Decoder** (same config).  
  5. **Decoder output projection** to dance token vocabulary size.

Example skeleton:python:src/models/stepformer_transformer.py
import torch
import torch.nn as nn
class Stepformer(nn.Module):
def init(
self,
vocab_size_music: int,
vocab_size_dance: int,
d_model: int = 512,
nhead: int = 8,
num_layers: int = 6,
ff_dim: int = 2048
):
super().init()
self.music_embedding = nn.Embedding(vocab_size_music, d_model)
self.dance_embedding = nn.Embedding(vocab_size_dance, d_model)
encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, ff_dim)
self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
self.output_proj = nn.Linear(d_model, vocab_size_dance)
def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
src_emb = self.music_embedding(src_tokens) # (batch, seq, d_model)
tgt_emb = self.dance_embedding(tgt_tokens)

# If using nn.Transformer, we might need shapes: (seq, batch, d_model)
src_emb = src_emb.permute(1, 0, 2)
tgt_emb = tgt_emb.permute(1, 0, 2)
memory = self.encoder(src_emb, mask=src_mask)
out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
out = out.permute(1, 0, 2) # back to (batch, seq, d_model)
logits = self.output_proj(out) # (batch, seq, vocab_size)
return logits

- Adapt shape permutations as needed (PyTorch’s `nn.Transformer` expects `(sequence, batch, hidden_size)`).
- Optionally add positional embeddings or advanced features (e.g., relative attention).

### 3.3 Vocabulary
- Decide on a numeric indexing for music tokens and dance tokens.  
- Typically, you’d have:
  - `vocab_size_music =` number of distinct music event tokens  
  - `vocab_size_dance =` number of distinct dance step tokens  
- Incorporate **special tokens** (e.g., `[PAD]`, `[BOS]`, `[EOS]`, `[UNK]`).

---

## 4. Training Pipeline

### 4.1 Dataset & DataLoader
Create a PyTorch dataset:

python:src/data/stepformer_dataset.py
import torch
from torch.utils.data import Dataset
class StepformerDataset(Dataset):
def init(self, data_path):
# Load pre-tokenized data from disk (e.g., JSON or pickle).
# Suppose it contains a list of {music_tokens, dance_tokens}.
self.samples = load_data(data_path)
def len(self):
return len(self.samples)
def getitem(self, idx):
item = self.samples[idx]
return {
"music_tokens": torch.tensor(item["music_tokens"]),
"dance_tokens": torch.tensor(item["dance_tokens"])
}

Create a custom collate function to pad sequences to the same length within a batch.

### 4.2 Training Script
In `src/training/train_stepformer.py`, define:
1. Arg parsing (epochs, batch_size, lr, etc.).  
2. Instantiate `StepformerDataset`, `DataLoader`.  
3. Build `Stepformer` model.  
4. Define `optimizer` (AdamW) and `loss_fn` (cross-entropy).  
5. Training loop:
   ```python
   for epoch in range(num_epochs):
       model.train()
       for batch in train_loader:
           music_tokens = batch["music_tokens"].to(device)
           dance_tokens = batch["dance_tokens"].to(device)

           # Shift dance tokens for teacher forcing
           dance_tokens_in = dance_tokens[:, :-1]
           dance_tokens_out = dance_tokens[:, 1:]

           optimizer.zero_grad()
           logits = model(music_tokens, dance_tokens_in)
           loss = loss_fn(
               logits.view(-1, logits.size(-1)),
               dance_tokens_out.view(-1)
           )
           loss.backward()
           optimizer.step()

       # Validation step ...
   ```

---

## 5. Logging & Monitoring (Weights & Biases)

1. `pip install wandb`, then:
   ```python
   import wandb

   wandb.init(project="stepformer", name="baseline_run")

   # Inside training loop:
   wandb.log({"train_loss": loss.item()})
   ```
2. For validation, also log metrics like perplexity or F1.

---

## 6. Evaluation

1. **Token-Level Accuracy & Perplexity**  
   Compare predicted tokens vs. ground-truth tokens.
2. **Step Placement F1**  
   Specifically check how many correct “step vs. no-step” tokens are predicted.
3. **Qualitative Review**  
   - Convert predicted token sequences back to DDR steps.
   - Test them in a DDR simulator or manual inspection.

---

## 7. Baseline & Comparisons

1. **DDC Baseline**  
   - Optionally run or reference the `ddc/` code to replicate the CNN+RNN approach.  
   - Compare objective metrics and subjective results with Stepformer.
2. **Ablations**  
   - E.g., test removing measure-based chunking vs. including it, or test different subdivision granularities.

---

## 8. Extensions & Next Steps

1. **Conditioning on Difficulty or Style**  
   - Add special tokens indicating “Beginner,” “Expert,” etc.
2. **Multi-Resolution**  
   - Use sub-measures for fast rhythms and full measures for slow sections.
3. **Human-in-the-Loop Tools**  
   - Let users modify partial charts and have the model regenerate the next steps.

---

## 9. Project Timeline

- **Weeks 1–2**: Data ingestion, simfile parsing, measure-based token alignment.  
- **Weeks 3–4**: Implement and test the Transformer code; initial training runs with small data.  
- **Weeks 5–6**: Refine token vocab, run hyperparam tuning, track results on wandb.  
- **Weeks 7–8**: Comparison with DDC baseline; gather subjective feedback from testers.  
- **Week 9**: Final cleanup, documentation, and blog post updates.

---

## 10. Conclusion

With this plan, we aim to build a measure-based, tokenized, Transformer-driven pipeline to generate DDR step charts from transcribed music events. By leveraging established code from MT3 for transcription and referencing the DDC approach, Stepformer will combine modern sequence modeling with domain-specific measure chunking to produce more coherent and musical choreographies.
