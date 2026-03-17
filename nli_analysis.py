"""
NLI-based argument analysis module.

Uses a CrossEncoder NLI model (DeBERTa) to predict entailment,
contradiction, or neutral labels for premise-conclusion pairs.
Analyzes argument structures from Searle's texts.
"""

from itertools import chain, permutations

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import CrossEncoder

LABEL_MAPPING = ["contradiction", "entailment", "neutral"]


def load_nli_model():
    """Load the CrossEncoder NLI model (DeBERTa v3 base).

    Returns:
        CrossEncoder: The loaded model.
    """
    model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    # Verify with the developer's example
    scores = model.predict([
        ("A man is eating pizza", "A man eats something"),
        ("A black race car starts up in front of a crowd of people.",
         "A man is driving down a lonely road."),
    ])
    labels = [LABEL_MAPPING[s.argmax()] for s in scores]
    print(f"Model verification - scores: {scores}")
    print(f"Model verification - labels: {labels}")

    return model


def predict_nli(model, premise, hypothesis):
    """Predict the NLI label for a premise-hypothesis pair.

    Returns:
        tuple: (label_string, numerical_scores)
            where scores are [contradiction, entailment, neutral].
    """
    scores = model.predict([(premise, hypothesis)])
    label = LABEL_MAPPING[scores.argmax(axis=1)[0]]
    numerical = scores.tolist()[0]
    return label, numerical


def count_words(sentence):
    """Print the word count of a sentence."""
    words = sentence.split()
    print(f"{len(words)} words in: '{sentence}'")
    print()


def count_words_total(sentences):
    """Print the total word count of multiple sentences combined."""
    combined = " ".join(str(s) for s in sentences)
    count_words(combined)


def analyze_argument(model, premises, conclusion):
    """Analyze all premise permutations against a conclusion using NLI.

    For each possible subset and ordering of premises, predicts the NLI
    label. The full set of premises (in any order) is assigned 'entailment'
    as the human prediction; partial premise sets get 'neutral'.

    Args:
        model: The CrossEncoder NLI model.
        premises: List of premise strings.
        conclusion: The conclusion string.

    Returns:
        pd.DataFrame: Results with columns for premise, conclusion,
            model predictions, human predictions, and match status.
    """
    human_predictions = {}
    rows = []

    # All orderings of the full premise set count as "entailment" for humans
    full_premise_orderings = {" ".join(p) for p in permutations(premises)}

    # Generate all subsets of premises in all orderings
    all_orderings = chain.from_iterable(
        permutations(premises, r) for r in range(1, len(premises) + 1)
    )

    for premise_ordering in all_orderings:
        combined_premise = " ".join(premise_ordering)
        label, scores = predict_nli(model, combined_premise, conclusion)

        # Human prediction: entailment for full premise set, neutral otherwise
        if combined_premise in full_premise_orderings:
            human_pred = "entailment"
        else:
            human_pred = human_predictions.setdefault(
                (combined_premise, conclusion), "neutral"
            )
        human_predictions[(combined_premise, conclusion)] = human_pred

        matched = label == human_pred

        rows.append({
            "Premise": combined_premise,
            "Conclusion": conclusion,
            "Model Numerical Prediction (contradiction, entailment, neutral)": scores,
            "Model Prediction": label,
            "Human Prediction": human_pred,
            "Matched": matched,
        })

    return pd.DataFrame(rows)


# --- Argument definitions from Searle's texts ---

def define_mbp_argument():
    """Define the Chinese Room Argument as stated in MBP (Searle, 1980).

    Returns:
        tuple: (premises_list, conclusion_string)
    """
    room_mbp = (
        "As far as the Chinese is concerned, I simply behave like a computer. "
        "I have inputs and outputs that are indistinguishable from those of "
        "the native Chinese speaker, but I still understand nothing."
    )
    com_mbp = (
        "In the Chinese case the computer is me, and in cases where the "
        "computer is not me, the computer has nothing more than I have in "
        "the case where I understand nothing."
    )
    conclusion = (
        "Computer understands nothing of any stories, whether in Chinese, "
        "English, or whatever."
    )
    return [room_mbp, com_mbp], conclusion


def define_cra_argument():
    """Define the Chinese Room Argument as stated in CRA (Searle, 2001).

    Returns:
        tuple: (premises_list, conclusion_string)
    """
    room_cra = (
        "The man in the room does not understand Chinese on the basis of "
        "implementing the appropriate program for understanding Chinese."
    )
    com_cra = "No computer, qua computer, has anything the man does not have."
    conclusion = (
        "Any digital computer does not understand Chinese soley on the basis "
        "of implementing the appropriate program for understanding Chinese."
    )
    return [room_cra, com_cra], conclusion


def define_larger_argument():
    """Define the larger structure of the Chinese Room Argument.

    Returns:
        tuple: (premises_list, conclusion_string)
    """
    prog = "Implemented programs are by definition purely formal or syntactical."
    mind = "Minds have mental or semantic contents."
    suf = "Syntax is not by itself sufficient for, nor constitutive of, semantics."
    conclusion = "Implemented programs are not constitutive of minds."
    return [prog, mind, suf], conclusion


def define_paraphrased_argument():
    """Define a paraphrased version of the larger argument.

    Returns:
        tuple: (premises_list, conclusion_string)
    """
    prog = (
        "If a computer program has been executed, the program is necessarily "
        "a set of formal rules or syntax."
    )
    mind = "Human cognitive processes contain elements that are mental or semantic."
    suf = "Syntax in isolation is not enough to form semantics."
    conclusion = "Executed computer programs do not make up minds."
    return [prog, mind, suf], conclusion


def run_all_argument_analyses(model):
    """Run NLI analysis on all four argument structures.

    Returns:
        pd.DataFrame: Combined results for all argument classes.
    """
    argument_definitions = {
        "MBP": define_mbp_argument,
        "CRA": define_cra_argument,
        "LARG": define_larger_argument,
        "LARG_PARA": define_paraphrased_argument,
    }

    dataframes = []
    for class_name, define_fn in argument_definitions.items():
        premises, conclusion = define_fn()
        print(f"\n--- Analyzing {class_name} ---")
        count_words_total(premises)
        df = analyze_argument(model, premises, conclusion)
        df["Argument Class"] = class_name
        dataframes.append(df)
        print(df)

    combined = pd.concat(dataframes, ignore_index=True)
    return combined


def plot_match_percentages(combined_df):
    """Plot the percentage of model-human prediction matches per argument class."""
    grouped = combined_df.groupby("Argument Class")["Matched"]
    match_percentage = (grouped.sum() / grouped.count()) * 100

    plt.figure(figsize=(10, 6))
    plt.bar(match_percentage.index, match_percentage.values)
    plt.xlabel("Argument Class")
    plt.ylabel("Percentage of Matches (%)")
    plt.title("Percentage of Matched Entries for Each Argument Class")
    plt.show()
