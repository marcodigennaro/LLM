import pytest
from llm.shared.shared_utils import basic_token, normal_cases, edge_cases
from llm.core.sentiment_analysis import polarity_scores_roberta  # Ensure your function is in 'your_module'
from llm.core.sentiment_analysis import map_scores
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

@pytest.fixture
def sample_data():
    return normal_cases()

@pytest.fixture
def boundary_data():
    return edge_cases()

def test_polarity_scores_roberta(roberta_setup, token_setup):
    tokenizer, model = roberta_setup
    token = token_setup
    score_dict = polarity_scores_roberta(token, tokenizer, model)
    assert isinstance(score_dict, dict), "Output should be a dictionary"
    assert set(score_dict.keys()) == {'roberta_neg', 'roberta_neu', 'roberta_pos'}, "Keys mismatch"
    assert score_dict['roberta_neg'] == pytest.approx(0.0026655, abs=1e-5)
    assert score_dict['roberta_neu'] == pytest.approx(0.0169531, abs=1e-5)
    assert score_dict['roberta_pos'] == pytest.approx(0.9803814, abs=1e-5)


# Tests to check if the function works as expected
def test_map_scores_roberta(sample_data):
    # Testing 'roberta' model scenario
    result = map_scores(sample_data.iloc[0], 'roberta')
    assert result == 4, "Expected 4 based on the data"


def test_map_scores_vader(sample_data):
    # Testing 'vader' model scenario
    result = map_scores(sample_data.iloc[0], 'vader')
    assert result == 5, "Expected 5 based on the data"


def test_invalid_model(sample_data):
    # Testing with an invalid model name
    with pytest.raises(ValueError):
        map_scores(sample_data.iloc[0], 'bert')


def test_edge_cases(boundary_data):
    # Testing boundary conditions
    # Assuming net_score at boundaries -1 and 1
    print(boundary_data)
    print(map_scores(boundary_data.iloc[0], model='roberta'))
    # For net_score = 1 (max boundary)
    assert map_scores(boundary_data.iloc[0], 'roberta') == 5, "Expected maximum score 5 at upper boundary"
    # For net_score = -1 (min boundary)
    assert map_scores(boundary_data.iloc[1], 'roberta') == 1, "Expected minimum score 1 at lower boundary"

