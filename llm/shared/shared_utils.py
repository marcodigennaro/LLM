import pandas as pd
from llm.config import DATA_DIR
from llm.core.functions import read_text_file


def basic_token():
    # Create token
    return "This is a token. I just love coding!"


def normal_cases():
    return pd.DataFrame({
        'roberta_pos': [0.6, -0.3, 0.0],
        'roberta_neg': [0.2, -0.6, 0.1],
        'vader_pos': [0.8, -0.4, 0.1],
        'vader_neg': [0.1, -0.5, 0.2]
    })


def edge_cases():
    return pd.DataFrame({
        'roberta_pos': [1, -1],
        'roberta_neg': [0, 0],
        'vader_pos': [1, -1],
        'vader_neg': [0, 0]
    })


def sample_text():
    return "TensorFlow is a platform for ML-powered applications. TensorFlow is a open source project."

