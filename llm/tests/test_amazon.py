import pytest
from llm.shared.shared_utils import basic_token
from llm.core.amazon import polarity_scores_roberta  # Ensure your function is in 'your_module'
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


@pytest.fixture
def roberta_setup():
    # Load tokenizer and model
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model


@pytest.fixture
def token_setup():
    # Create token
    return basic_token()


def test_polarity_scores_roberta(roberta_setup, token_setup):
    tokenizer, model = roberta_setup
    token = token_setup
    score_dict = polarity_scores_roberta(token, tokenizer, model)
    assert isinstance(score_dict, dict), "Output should be a dictionary"
    assert set(score_dict.keys()) == {'roberta_neg', 'roberta_neu', 'roberta_pos'}, "Keys mismatch"
    assert score_dict['roberta_neg'] == pytest.approx(0.0026655, abs=1e-5)
    assert score_dict['roberta_neu'] == pytest.approx(0.0169531, abs=1e-5)
    assert score_dict['roberta_pos'] == pytest.approx(0.9803814, abs=1e-5)

