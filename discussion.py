"""
Discussion and limits analysis module.

Demonstrates the limitations of using NLI for argument structure
identification, covering:
- Label accuracy limits
- Need for argument mining
- Need for probabilistic analysis
- Need for justification
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

from nli_analysis import predict_nli, define_mbp_argument, define_larger_argument


def show_entailment_cases(combined_df):
    """Display only the cases where the human prediction is entailment."""
    entail_df = combined_df[combined_df["Human Prediction"] == "entailment"]
    print("Cases where human prediction is 'entailment':")
    print(entail_df)
    print()
    return entail_df


def plot_nli_complexity(max_n=20):
    """Plot the growth of possible NLI comparisons vs 2^n.

    Shows why exhaustive NLI analysis is infeasible for texts
    with many sentences.
    """
    def nli_execution_count(n):
        """Number of different NLI comparisons for n sentences."""
        total = 0
        for k in range(1, n):
            total += comb(n, k) * comb(n - k, 1)
        return total

    n_values = np.arange(1, max_n + 1)
    nli_counts = [nli_execution_count(n) for n in n_values]
    power_of_two = [2**n for n in n_values]

    plt.figure(figsize=(10, 6))
    plt.plot(
        n_values, nli_counts,
        label=r"$\sum_{k=1}^{n-1} \binom{n}{k} \cdot \binom{n-k}{1}$",
        marker="o",
    )
    plt.plot(n_values, power_of_two, label=r"$2^n$", marker="x")
    plt.xlabel("n")
    plt.ylabel("Value")
    plt.title(r"$2^n$ and the number of different NLI for $n$ sentences")
    plt.legend()
    plt.grid(True)
    plt.show()


def demonstrate_probabilistic_need(nli_model):
    """Show examples where probabilistic labels would be more informative."""
    print("=== Probabilistic Analysis Examples ===\n")

    # General example
    p1 = "A boy hits a ball with a bat."
    h1 = "The boy is playing in a baseball game."
    print(f"P: {p1}")
    print(f"H: {h1}")
    print(f"Prediction: {predict_nli(nli_model, p1, h1)}")
    print()

    # MBP single premise to conclusion
    mbp_premises, mbp_conclusion = define_mbp_argument()
    room_mbp = mbp_premises[0]
    print(f"P: {room_mbp}")
    print(f"H: {mbp_conclusion}")
    print(f"Prediction: {predict_nli(nli_model, room_mbp, mbp_conclusion)}")
    print()

    # LARG partial premises
    larg_premises, larg_conclusion = define_larger_argument()
    prog, _mind, suf = larg_premises
    combined = prog + " " + suf
    print(f"P: {combined}")
    print(f"H: {larg_conclusion}")
    print(f"Prediction: {predict_nli(nli_model, combined, larg_conclusion)}")
    print()


def demonstrate_justification_need(nli_model):
    """Show examples where NLI predictions need further justification."""
    print("=== Justification Examples ===\n")

    # Analytic truth example
    p2 = "Quine is married."
    h2 = "Quine is a bachelor."
    print(f"P: {p2}")
    print(f"H: {h2}")
    print(f"Prediction: {predict_nli(nli_model, p2, h2)}")
    print()

    # Argument with different premise orderings
    larg_premises, larg_conclusion = define_larger_argument()
    mind, prog = larg_premises[1], larg_premises[0]

    print(f"P (mind+prog): {mind} {prog}")
    print(f"H: {larg_conclusion}")
    print(f"Prediction: {predict_nli(nli_model, mind + ' ' + prog, larg_conclusion)}")
    print()

    print(f"P (prog+mind): {prog} {mind}")
    print(f"H: {larg_conclusion}")
    print(f"Prediction: {predict_nli(nli_model, prog + ' ' + mind, larg_conclusion)}")
    print()

    # Intermediate conclusion examples
    inter = "Programs with minds are mental and semantic."
    inter2 = "Minds with programs are mental and semantic."

    print("--- With intermediate conclusions ---")
    print(f"P: {mind} {prog}")
    print(f"H: {larg_conclusion}")
    print(f"  Direct:  {predict_nli(nli_model, mind + ' ' + prog, larg_conclusion)}")
    print()
    print(f"  Via '{inter}':")
    print(f"    Step 1: {predict_nli(nli_model, mind + ' ' + prog, inter)}")
    print(f"    Step 2: {predict_nli(nli_model, inter, larg_conclusion)}")
    print()
    print(f"  Via '{inter2}':")
    print(f"    Step 1: {predict_nli(nli_model, mind + ' ' + prog, inter2)}")
    print(f"    Step 2: {predict_nli(nli_model, inter2, larg_conclusion)}")


def run_discussion(nli_model, combined_df, cra_stats, mbp_stats, wrapped_summaries):
    """Run the full discussion/limits analysis.

    Args:
        nli_model: The CrossEncoder NLI model.
        combined_df: Combined DataFrame from argument analysis.
        cra_stats: Tuple of (num_sentences, num_words, num_unique_words) for CRA.
        mbp_stats: Tuple of (num_sentences, num_words, num_unique_words) for MBP.
        wrapped_summaries: Tuple of (wrapped_cra_summary, wrapped_mbp_summary).
    """
    cra_num_sentences, cra_num_words, _ = cra_stats
    mbp_num_sentences, mbp_num_words, _ = mbp_stats
    wrapped_cra_sum, wrapped_mbp_sum = wrapped_summaries

    # 4.1 Limits of labeling argument types
    print("=" * 60)
    print("4.1 Limits of Labeling Argument Types")
    print("=" * 60)
    from nli_analysis import plot_match_percentages
    plot_match_percentages(combined_df)
    show_entailment_cases(combined_df)

    # 4.2 Need for argument mining
    print("=" * 60)
    print("4.2 The Need for Argument Mining")
    print("=" * 60)
    print(f"CRA word count: {cra_num_words}")
    print(f"MBP word count: {mbp_num_words}")
    print()
    print(f"CRA sentence count: {cra_num_sentences}")
    print(f"MBP sentence count: {mbp_num_sentences}")
    print()
    plot_nli_complexity()

    print("Summaries (for reference - may be inaccurate):")
    print("Summary of CRA:")
    print(wrapped_cra_sum)
    print()
    print("Summary of MBP:")
    print(wrapped_mbp_sum)
    print()

    # Print conclusions for reference
    _, conc_cra = define_mbp_argument()  # uses MBP's conclusion
    _, conc_larg = define_larger_argument()
    from nli_analysis import define_cra_argument
    _, conc_cra_text = define_cra_argument()
    print("Conclusions:")
    print(f"  CRA: {conc_cra_text}")
    print(f"  LARG: {conc_larg}")
    print(f"  MBP: {conc_cra}")
    print()

    # 4.3 Need for probabilistic analysis
    print("=" * 60)
    print("4.3 The Need for Probabilistic Analysis")
    print("=" * 60)
    demonstrate_probabilistic_need(nli_model)

    # 4.4 Need for justification
    print("=" * 60)
    print("4.4 The Need for Justification")
    print("=" * 60)
    demonstrate_justification_need(nli_model)
