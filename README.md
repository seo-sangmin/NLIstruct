\****This repo is being modularized and corrected based on the original python notebook (written in Sep 2023).***

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
nli-argument-structure/
├── main.ipynb                 # Colab-ready notebook running the full pipeline
├── preprocessing.py           # Download, clean, and tokenize source texts
├── exploratory.py             # Word statistics and word cloud visualization
├── topic_modeling.py          # LDA topic extraction with word clouds
├── embedding.py               # Sentence embedding and t-SNE visualization
├── summarization.py           # Text summarization using LongT5
├── bert_snli.py               # BERT fine-tuning on SNLI
├── nli_analysis.py            # NLI prediction and argument analysis
├── discussion.py              # Limits and discussion demonstrations
├── requirements.txt           # Python dependencies
└── original_notebook/
    └── Identifying Argument Structure.ipynb  # Original Sep 2023 notebook
```

The `original_notebook/` folder preserves the unmodified notebook from September 2023 for reference. The active pipeline lives in `main.ipynb` and the modular `*.py` files at the repository root.

## Setup

### Requirements

- Python 3.8+
- GPU recommended (CUDA-compatible) for BERT training; CPU fallback is supported

### Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

The recommended way to run the full pipeline is via `main.ipynb`, which is designed for Google Colab with a T4 GPU runtime. Open it in Colab (or a local Jupyter environment with GPU access) and execute the cells in order. The setup cells clone this repository, install dependencies, verify CUDA, and download NLTK corpora before the pipeline runs.

In Colab: **Runtime → Change runtime type → T4 GPU → Save**, then run all cells.

### Using Individual Modules

Each module can also be imported independently in Python:

```python
from preprocessing import load_cra, load_mbp
from nli_analysis import load_nli_model, predict_nli, analyze_argument

# Load texts
cra_raw, cra_clean, cra_tokens = load_cra()

# Load NLI model and make predictions
model = load_nli_model()
label, scores = predict_nli(model, "Syntax is formal.", "Minds are semantic.")
```

The modules are organized so that text analysis (`preprocessing`, `exploratory`, `topic_modeling`, `embedding`, `summarization`), BERT fine-tuning (`bert_snli`), and NLI argument analysis (`nli_analysis`) are independent. The `discussion` module depends on outputs from text analysis and NLI argument analysis.

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
