# Grapheme-to-Phoneme Conversion with Transformer Architecture

A neural sequence-to-sequence model that converts written text (graphemes) into phonetic representations (phonemes) using the Transformer architecture. The system is trained on the CMU Pronouncing Dictionary and achieves accurate pronunciation predictions for English words.

## Overview

This project implements a Transformer-based model for automatic phoneme generation from graphemes. The model learns the complex relationship between spelling and pronunciation in English, making it useful for text-to-speech systems, pronunciation learning tools, and linguistic research.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)
- python-Levenshtein

## Installation

Install the required dependencies:

```bash
pip install torch
pip install python-Levenshtein
```

The CMU Dictionary will be downloaded automatically on first run.

## Usage

Run the training script:

```bash
python train.py
```

The model will:
1. Download and prepare the CMU Pronouncing Dictionary
2. Train for 100 epochs with automatic validation
3. Report phoneme error rate (PER) and word error rate (WER)

### Model Architecture

- **Encoder**: 3 layers, 128-dimensional embeddings
- **Decoder**: 1 layer with masked self-attention
- **Attention heads**: 2 (128/64)
- **Feedforward dimension**: 512
- **Training batch size**: 1024
- **Optimizer**: AdamW with gradient clipping

### Example Predictions

```python
Input:  "otorhinolaryngologist"
Output: ['OW', 'T', 'OW', 'R', 'AY', 'N', 'OW', 'L', 'AE', 'R', 'AH', 'N', 'G', 'AA', 'L', 'AH', 'JH', 'AH', 'S', 'T']
```

## Dataset

The model is trained on the CMU Pronouncing Dictionary, which contains over 134,000 word-pronunciation pairs. The dataset is automatically split into 90% training and 10% validation.

## Evaluation Metrics

- **Phoneme Error Rate (PER)**: Levenshtein distance between predicted and target phoneme sequences
- **Word Error Rate (WER)**: Percentage of words with at least one incorrect phoneme

## Project Motivation

Grapheme-to-phoneme conversion is a fundamental problem in natural language processing with applications in:

- **Speech synthesis**: Converting text to speech requires accurate phonetic representations
- **Language learning**: Helping learners understand pronunciation patterns
- **Linguistic analysis**: Studying the relationship between orthography and phonology

Traditional rule-based approaches struggle with English's irregular spelling patterns. This project demonstrates that neural sequence-to-sequence models can learn these complex mappings directly from data, achieving robust performance even on out-of-vocabulary words through learned patterns.

The Transformer architecture was chosen for its ability to capture long-range dependencies in sequences and its superior training efficiency compared to recurrent models. The positional encoding allows the model to understand character order without recurrence, while the attention mechanism learns which graphemes are most relevant for predicting each phoneme.

## Model Performance

Expected validation metrics after 100 epochs:
- Validation Loss: ~0.05
- Phoneme Error Rate: <5%
- Word Error Rate: <15%

## License

This project uses the CMU Pronouncing Dictionary, which is in the public domain.
