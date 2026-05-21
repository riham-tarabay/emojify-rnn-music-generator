# 😄🎵 Emojify → RNN Music Generator

> Two NLP/sequence projects: emoji sentiment prediction via word embeddings, and LSTM-based melody generation with temperature sampling.

## Part 1 — Emojify

Maps sentences to emojis using the average of their GloVe word embeddings.

| Model | Accuracy |
|---|---|
| Mean Pooling | ~97% (training) |
| LSTM | ~98% (training) |

**Key insight:** Sentence embedding = average of word embeddings captures semantic meaning without sequential modeling.

## Part 2 — RNN Music Generator

Auto-regressive LSTM trained to predict the next note given a sequence of 16 notes.

```
[C4, E4, G4, E4, C4, ...] → LSTM → P(next_note) → sample with temperature
```

| Temperature | Effect |
|---|---|
| 0.5 | Conservative, repetitive, stays close to training |
| 0.8 | Balanced creativity (recommended) |
| 1.2 | Exploratory, more varied, occasionally surprising |

## Quick Start
```bash
pip install -r requirements.txt
python emojify_music.py
```
Generated melodies saved as `.abc` files — open with [abcjs](https://www.abcjs.net/) or [MuseScore](https://musescore.org/).

## What I Learned
- Word embeddings encode semantic similarity: similar words cluster in embedding space
- Temperature scaling reshapes probability distributions for controllable generation
- LSTM hidden state carries long-range dependencies that plain RNNs lose
- ABC notation as a lightweight, parseable music representation

## Tech Stack
`TensorFlow/Keras` · `NumPy` · `Matplotlib`
