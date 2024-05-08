import pandas as pd
from scipy.special import softmax
from transformers import logging

logging.set_verbosity_warning()

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


def polarity_scores_vader(text):
    vader_result = sia.polarity_scores(text)
    vader_dict = {}
    for vader_k, vader_v in vader_result.items():
        vader_dict[f'vader_{vader_k}'] = vader_v

    return vader_dict


def polarity_scores_roberta(token, tokenizer, model):
    """
    Compute the sentiment polarity scores for a given token using a RoBERTa model.

    Args:
        token (str): The text snippet to analyze.
        tokenizer: The tokenizer compatible with the RoBERTa model.
        model: The pre-trained RoBERTa model.

    Returns:
        dict: A dictionary with keys 'roberta_neg', 'roberta_neu', and 'roberta_pos'
              representing the negative, neutral, and positive sentiment scores,
              respectively.
    """
    # Encode the text using the provided tokenizer
    encoded_text = tokenizer(token, return_tensors='pt')

    # Pass the encoded text through the model
    output = model(**encoded_text)

    # Extract the logits and apply softmax to get probabilities
    scores = output[0][0].detach().numpy()

    # Specific to MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    assert len(scores) == 3

    scores = softmax(scores)

    # Create a dictionary mapping sentiment types to their respective scores
    return dict(roberta_neg=scores[0], roberta_neu=scores[1], roberta_pos=scores[2])


def map_scores(row: pd.Series, model: str) -> int:
    """
    Maps sentiment scores from -1 to 1 scale into an integer scale of 1 to 5 based on the model.

    Args:
        row (pd.Series): A pandas Series containing the sentiment scores.
        model (str): The model type, either 'roberta' or 'vader', to specify which scores to use.

    Returns:
        int: The final score, mapped to an integer scale from 1 to 5.

    Raises:
        ValueError: If `model` is not 'roberta' or 'vader'.
    """

    # Validate the model parameter to ensure it is either 'roberta' or 'vader'
    if model not in ['roberta', 'vader']:
        raise ValueError("Model must be either 'roberta' or 'vader'")

    # Calculate the net positive score based on the model
    if model == 'roberta':
        net_score = row['roberta_pos'] - row['roberta_neg']
    elif model == 'vader':
        net_score = row['vader_pos'] - row['vader_neg']

    # Normalize this score to a 1-5 scale
    # net_score ranges from -1 to 1, we translate this to 0 to 2
    scaled_score = (net_score + 1) * 2
    # Convert 0-4 scale to 1-5 scale
    final_score = int(scaled_score / 4 * 5) + 1

    return final_score
