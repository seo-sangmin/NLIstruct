"""
Text summarization module.

Uses the LongT5 model (fine-tuned for book summarization)
to generate summaries of the CRA and MBP texts.
"""

import textwrap

import torch
from transformers import pipeline


def create_summarizer():
    """Create a summarization pipeline using LongT5."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        device=device,
    )


def summarize_text(summarizer, text):
    """Summarize a text and return the summary string."""
    result = summarizer(text)
    return result[0]["summary_text"]


def wrap_text(text, width=100):
    """Wrap text for convenient reading."""
    return textwrap.fill(text, width=width)


def summarize_and_display(summarizer, cra_text, mbp_text):
    """Summarize both texts and print wrapped results.

    Returns:
        tuple: (wrapped_cra_summary, wrapped_mbp_summary)
    """
    cra_summary = summarize_text(summarizer, cra_text)
    mbp_summary = summarize_text(summarizer, mbp_text)

    wrapped_cra = wrap_text(cra_summary)
    wrapped_mbp = wrap_text(mbp_summary)

    print("Summary of CRA:")
    print(wrapped_cra)
    print()
    print("Summary of MBP:")
    print(wrapped_mbp)
    print()

    return wrapped_cra, wrapped_mbp
