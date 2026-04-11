"""
Main runner for the NLI Argument Structure Analysis.

Identifies argument structure in John Searle's texts using
Natural Language Inference (NLI). This script runs the full
pipeline from text preprocessing through analysis and discussion.

Usage:
    python main.py                  # Run the full pipeline
    python main.py text_analysis    # Section 2 only
    python main.py fine_tuning      # Section 3.1 only
    python main.py nli_analysis     # Section 3.2 only
    python main.py discussion       # Section 4 only (runs 2 & 3.2 first)

Author: Sangmin Seo
"""

import argparse

from preprocessing import load_cra, load_mbp, count_text_stats
from exploratory import print_text_stats, plot_wordclouds
from topic_modeling import run_lda, plot_topic_wordclouds
from embedding import embed_sentences, reduce_to_2d, plot_embeddings
from summarization import create_summarizer, summarize_and_display
from bert_snli import load_pretrained_bert, get_devices, train_classifier
from nli_analysis import load_nli_model, run_all_argument_analyses, plot_match_percentages
from discussion import run_discussion


def run_text_analysis():
    """Section 2: Analysing the Target Texts."""
    print("=" * 60)
    print("SECTION 2: Analysing the Target Texts")
    print("=" * 60)

    # 2.1 Pre-processing
    print("\n--- 2.1 Pre-processing ---")
    cra_raw, cra_clean, cra_tokens = load_cra()
    mbp_raw, mbp_clean, mbp_tokens = load_mbp()

    # 2.2 Exploratory Analysis
    print("\n--- 2.2 Exploratory Analysis ---")
    cra_stats = count_text_stats(cra_raw)
    mbp_stats = count_text_stats(mbp_raw)
    print_text_stats("Chinese Room Argument (CRA)", *cra_stats)
    print_text_stats("Minds, Brains, and Programs (MBP)", *mbp_stats)
    plot_wordclouds(cra_tokens, mbp_tokens)

    # 2.3 Topic Modelling
    print("\n--- 2.3 Topic Modelling ---")
    topics = run_lda([cra_clean, mbp_clean])
    print("Identified topics:", topics)
    plot_topic_wordclouds(topics)

    # 2.4 Sentence Embedding
    print("\n--- 2.4 Sentence Embedding ---")
    cra_sentences, cra_embeddings = embed_sentences(cra_raw)
    mbp_sentences, mbp_embeddings = embed_sentences(mbp_raw)

    x_cra, y_cra = reduce_to_2d(cra_embeddings)
    x_mbp, y_mbp = reduce_to_2d(mbp_embeddings)
    plot_embeddings(x_cra, y_cra, cra_sentences, x_mbp, y_mbp, mbp_sentences)

    # 2.5 Text Summarization
    print("\n--- 2.5 Text Summarization ---")
    summarizer = create_summarizer()
    wrapped_cra_sum, wrapped_mbp_sum = summarize_and_display(
        summarizer, cra_raw, mbp_raw
    )

    return {
        "cra_stats": cra_stats,
        "mbp_stats": mbp_stats,
        "wrapped_cra_sum": wrapped_cra_sum,
        "wrapped_mbp_sum": wrapped_mbp_sum,
    }


def run_fine_tuning():
    """Section 3.1: Fine-tuning BERT on SNLI."""
    print("=" * 60)
    print("SECTION 3.1: Fine-tuning BERT on SNLI")
    print("=" * 60)

    bert_model, _vocabulary = load_pretrained_bert()
    devices = get_devices()
    train_classifier(bert_model, devices)


def run_nli_analysis():
    """Section 3.2: Using NLI for argument analysis."""
    print("=" * 60)
    print("SECTION 3.2: NLI Argument Analysis")
    print("=" * 60)

    nli_model = load_nli_model()
    combined_df = run_all_argument_analyses(nli_model)

    print("\n--- Combined Results ---")
    print(combined_df)
    plot_match_percentages(combined_df)

    return {"nli_model": nli_model, "combined_df": combined_df}


def run_discussion_section(text_results, nli_results):
    """Section 4: Limits & Discussion."""
    print("=" * 60)
    print("SECTION 4: Limits & Discussion")
    print("=" * 60)

    run_discussion(
        nli_results["nli_model"],
        nli_results["combined_df"],
        text_results["cra_stats"],
        text_results["mbp_stats"],
        (text_results["wrapped_cra_sum"], text_results["wrapped_mbp_sum"]),
    )


def print_conclusion():
    """Section 5: Conclusion."""
    print("=" * 60)
    print("SECTION 5: Conclusion")
    print("=" * 60)
    print(
        "Analysis complete. Texts from Searle (1980) and Searle (2001) have "
        "been analyzed using NLP techniques and NLI applied to specific "
        "arguments extracted from those texts."
    )


def main():
    parser = argparse.ArgumentParser(
        description="NLI Argument Structure Analysis pipeline."
    )
    parser.add_argument(
        "section",
        nargs="?",
        choices=["text_analysis", "fine_tuning", "nli_analysis", "discussion"],
        default=None,
        help="Run a specific section instead of the full pipeline.",
    )
    args = parser.parse_args()

    if args.section is None:
        # Full pipeline
        text_results = run_text_analysis()
        run_fine_tuning()
        nli_results = run_nli_analysis()
        run_discussion_section(text_results, nli_results)
        print_conclusion()

    elif args.section == "text_analysis":
        run_text_analysis()

    elif args.section == "fine_tuning":
        run_fine_tuning()

    elif args.section == "nli_analysis":
        run_nli_analysis()

    elif args.section == "discussion":
        # Discussion depends on text_analysis and nli_analysis
        text_results = run_text_analysis()
        nli_results = run_nli_analysis()
        run_discussion_section(text_results, nli_results)


if __name__ == "__main__":
    main()
