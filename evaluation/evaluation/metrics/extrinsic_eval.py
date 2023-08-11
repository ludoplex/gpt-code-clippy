from metrics.bleu import compute_bleu


def compute_exact_match(references, generated) -> float:
    """
    Computes Exact Match Accuracy.
    args:
        reference: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
        translation: list of translations to score. Each translation
          should be tokenized into a list of tokens.
    returns:
        exact_match_accuracy : Float
    """
    exact_match_count = 0.0
    for gen, ref in zip(generated, references[0]):
        if gen == ref:
            exact_match_count += 1
    return exact_match_count / len(generated)


def compute_metrics(references, generated) -> dict:
    """
    Calculates various metrics and returns the calculated dict of these matrics.
    args:
        reference: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
        translation: list of translations to score. Each translation
          should be tokenized into a list of tokens.
    returns:
        A dicitonary with different metrics intact.
    """
    metrics_dict = {
        "bleu_4": None,
        "exact_match_acc": None,
        "smoothed_bleu_4": compute_bleu(references, generated, smooth=True),
    }
    metrics_dict["bleu_4"] = compute_bleu(references, generated, smooth=False)
    metrics_dict["exact_match_acc"] = compute_exact_match(references, generated)
    return metrics_dict
