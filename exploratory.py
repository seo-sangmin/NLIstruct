"""
Exploratory analysis module.

Provides word counting statistics and word cloud visualizations
for the CRA and MBP texts.
"""

from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def print_text_stats(name, num_sentences, num_words, num_unique_words):
    """Print sentence, word, and unique word counts for a text."""
    print(f"--- {name} ---")
    print(f"  Sentences:    {num_sentences}")
    print(f"  Words:        {num_words}")
    print(f"  Unique words: {num_unique_words}")
    print()


def generate_wordcloud(tokens, title, ax):
    """Generate and display a word cloud on a given matplotlib axis."""
    word_freq = Counter(tokens)
    cloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(word_freq)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)


def plot_wordclouds(cra_tokens, mbp_tokens):
    """Plot word clouds for both texts side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    generate_wordcloud(cra_tokens, "Chinese Room Argument", ax1)
    generate_wordcloud(mbp_tokens, "Minds, Brains, and Programs", ax2)
    plt.tight_layout()
    plt.show()
