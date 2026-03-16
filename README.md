***This repo is being modularized and corrected based on the ipynb.***

# Identifying Argument Structure in John Searle's Texts Using NLI

This project analyzes the argument structure of John Searle's Chinese Room Argument using Natural Language Processing (NLP) techniques and Natural Language Inference (NLI). It examines two source texts:

- **CRA** — *Chinese Room Argument*, Searle (2001) in Wilson et al. (Eds.)
- **MBP** — *Minds, Brains, and Programs*, Searle (1980)

## Overview

The analysis proceeds in four stages:

1. **Text Analysis** — Preprocess the texts, compute word/sentence statistics, generate word clouds, extract topics with LDA, visualize sentence embeddings with t-SNE, and summarize with LongT5.
2. **BERT Fine-tuning** — Fine-tune a pretrained BERT model on the SNLI dataset to demonstrate NLI classification (~0.8 accuracy).
3. **NLI Argument Analysis** — Use a high-accuracy CrossEncoder (DeBERTa v3) to predict entailment/contradiction/neutral labels for all premise-conclusion permutations across four argument formulations (MBP, CRA, LARG, LARG_PARA).
4. **Limits & Discussion** — Evaluate the limits of NLI for argument structure identification, covering label accuracy, argument mining complexity, probabilistic analysis, and justification.

## Project Structure

```
NLIstruct/
├── main.py              # Run the full analysis pipeline
├── preprocessing.py     # Download, clean, and tokenize source texts
├── exploratory.py       # Word statistics and word cloud visualization
├── topic_modeling.py    # LDA topic extraction with word clouds
├── embedding.py         # Sentence embedding and t-SNE visualization
├── summarization.py     # Text summarization using LongT5
├── bert_snli.py         # BERT fine-tuning on SNLI
├── nli_analysis.py      # NLI prediction and argument analysis
├── discussion.py        # Limits and discussion demonstrations
├── requirements.txt     # Python dependencies
└── Identifying Argument Structure.ipynb  # Original notebook
```

## Setup

### Requirements

- Python 3.8+
- GPU recommended (CUDA-compatible) for BERT training; CPU fallback is supported

### Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python main.py
```

This executes all stages sequentially: text preprocessing, exploratory analysis, topic modeling, sentence embedding, summarization, BERT fine-tuning, NLI argument analysis, and the discussion of limits.

### Using Individual Modules

Each module can be imported independently:

```python
from preprocessing import load_cra, load_mbp
from nli_analysis import load_nli_model, predict_nli, analyze_argument

# Load texts
cra_raw, cra_clean, cra_tokens = load_cra()

# Load NLI model and make predictions
model = load_nli_model()
label, scores = predict_nli(model, "Syntax is formal.", "Minds are semantic.")
```

## Key Dependencies

| Package | Purpose |
|---|---|
| `gdown` | Download source texts from Google Drive |
| `nltk` | Tokenization and stopword removal |
| `scikit-learn` | LDA topic modeling, t-SNE dimensionality reduction |
| `sentence-transformers` | Sentence embedding and CrossEncoder NLI |
| `transformers` | LongT5 summarization pipeline |
| `torch` | Deep learning backend |
| `d2l` | BERT model utilities and SNLI data loading |
| `plotly` | Interactive embedding visualization |
| `matplotlib` / `wordcloud` | Static plots and word clouds |
| `pandas` | Argument analysis data management |

## References

- Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large annotated corpus for learning natural language inference. *EMNLP*.
- Searle, J. R. (1980). Minds, brains, and programs. *Behavioral and Brain Sciences*, 3(3), 417–424.
- Searle, J. R. (2001). Chinese room argument. In R. A. Wilson & F. C. Keil (Eds.), *The MIT Encyclopedia of the Cognitive Sciences*.
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into Deep Learning*.
