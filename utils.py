
import string
import re
import unicodedata
from collections import Counter

import torch

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def drqa_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def drqa_exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediciton, ground_truth):
    """Check if the ground truth is a (soft) exact match substring of the prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediciton) 


def drqa_metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    # ground truth could be a string or a list of strings or a list of list of strings
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif isinstance(ground_truths[0], list):
        ground_truths = [ground_truth for ground_truths_list in ground_truths for ground_truth in ground_truths_list]

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def get_top_tokens(logits, tokenizer, top_k=10):
    """Get the top tokens and their probabilities from the logits."""
    top_tokens = []
    for logit in logits:
        a, b = torch.topk(torch.softmax(logit, dim=-1), top_k, dim=-1)
        l = [(y, f"{x*100:.02f}") for x, y in zip(a[0], tokenizer.convert_ids_to_tokens(b[0]))]
        top_tokens.append(l)
    return top_tokens


def nll_acc(nll, gold):
    pred = nll.argmax().item() 
    return pred, int(pred == gold)


def nll_acc_norm(nll, gold, length):
    pred = (nll / length).argmax().item()
    return pred, int(pred == gold) 


def nll_acc_calibrated(nll, gold, calibrated_nll):
    pred = (nll - calibrated_nll).argmax().item()
    return pred, int(pred == gold)


def nll_acc_calibrated_norm(nll, gold, calibrated_nll, length):
    pred = ((nll - calibrated_nll) / length).argmax().item()
    return pred, int(pred == gold)
