"""
Topic modeling module.

Uses bag-of-words vectorization and Latent Dirichlet Allocation (LDA)
to identify topics across the CRA and MBP texts.
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def run_lda(documents, n_topics=4, n_top_words=8):
    """Run LDA topic modeling on a list of text documents.

    Args:
        documents: List of text strings.
        n_topics: Number of topics to extract.
        n_top_words: Number of top words per topic.

    Returns:
        dict: Mapping from topic index to list of top words.
    """
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(bag_of_words)

    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic_weights in enumerate(lda.components_):
        top_indices = topic_weights.argsort()[: -n_top_words - 1 : -1]
        topics[idx] = [feature_names[i] for i in top_indices]

    return topics


def plot_topic_wordclouds(topics):
    """Plot word clouds for each LDA topic."""
    n_topics = len(topics)
    cols = min(n_topics, 2)
    rows = (n_topics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    axes = axes.flatten() if n_topics > 1 else [axes]

    for i, words in topics.items():
        cloud = WordCloud(background_color="white").generate(" ".join(words))
        axes[i].imshow(cloud, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(f"Topic {i}")

    fig.suptitle("Word Clouds of Topics")
    plt.tight_layout()
    plt.show()
