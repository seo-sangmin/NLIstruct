"""
Sentence embedding module.

Embeds sentences using SentenceTransformer, reduces dimensionality
with t-SNE, and visualizes the results with Plotly.
"""

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def embed_sentences(text, model_name="all-MiniLM-L6-v2"):
    """Tokenize text into sentences and compute embeddings.

    Args:
        text: Raw text string.
        model_name: SentenceTransformer model to use.

    Returns:
        tuple: (sentences, embeddings)
    """
    sentences = sent_tokenize(text)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return sentences, embeddings


def reduce_to_2d(embeddings, perplexity=30, random_state=30):
    """Reduce high-dimensional embeddings to 2D using t-SNE.

    Returns:
        tuple: (x_values, y_values)
    """
    tsne = TSNE(perplexity=perplexity, n_components=2, init="pca", random_state=random_state)
    reduced = tsne.fit_transform(embeddings)
    return reduced[:, 0], reduced[:, 1]


def plot_embeddings(x_cra, y_cra, cra_labels, x_mbp, y_mbp, mbp_labels):
    """Plot 2D sentence embeddings for both texts side by side."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Chinese Room Argument", "Minds, Brains, and Programs"),
    )

    fig.add_trace(
        go.Scatter(x=x_cra, y=y_cra, mode="markers", text=cra_labels),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_mbp, y=y_mbp, mode="markers", text=mbp_labels),
        row=1, col=2,
    )

    fig.update_layout(title="Sentence Embeddings")
    fig.show()
