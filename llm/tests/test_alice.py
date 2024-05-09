import numpy as np
import pytest
from llm.core.alice import encode_text
from llm.shared.shared_utils import sample_text
import re
from llm.core.alice import create_sequences
from llm.core.alice import pad_sequences_to_same_length
from llm.core.alice import split_sequences
from llm.core.alice import one_hot_encode_labels


@pytest.fixture
def hello_text():
    return 'Hello world, hello all, hello world'


@pytest.fixture
def tf_text():
    return sample_text()


@pytest.fixture
def expected_tokens(tf_text):
    all_words = re.findall(r'\b\w+\b', tf_text.lower())
    unique_words = list(set(all_words))

    expected_dict = {}
    for k in all_words:
        if k not in expected_dict:
            expected_dict[k] = len(expected_dict) + 1

    return all_words, unique_words, expected_dict


def test_encode_text(tf_text, expected_tokens):
    encoded, tokenizer = encode_text(tf_text)
    word_index = tokenizer.word_index
    all_words, unique_words, expected_dict = expected_tokens

    # Check if the output is a list of integers
    assert isinstance(encoded, list), "The encoded output should be a list."
    assert all(isinstance(num, int) for num in encoded), "All items in the encoded list should be integers."

    # Check if the vocabulary size matches the number of unique words

    assert len(word_index) == len(unique_words), "The word index should contain all unique words."
    assert expected_dict == word_index

    # Check if the encoding matches the manual check
    expected_encoding = [word_index[word] for word in all_words]
    assert encoded == expected_encoding, "The encoding does not match the expected output."


def test_encode_results(hello_text):
    encoded, tokenizer = encode_text(hello_text)
    word_index = tokenizer.word_index

    # Prepare the sequences used by the Neural Network
    vocab_size = len(word_index) + 1  # Including zero index
    sequences = create_sequences(encoded)

    padded_sequences = pad_sequences_to_same_length(sequences)
    X, y = split_sequences(padded_sequences)
    y = one_hot_encode_labels(y, vocab_size)

    assert word_index == {'hello': 1, 'world': 2, 'all': 3}
    assert encoded == [1, 2, 1, 3, 1, 2]
    assert np.array_equal(X[-1], [1, 2, 1, 3, 1] ), 'Hello world, hello all, hello'
    assert np.array_equal(y[-1], [0., 0., 1., 0.] ), 'world'
