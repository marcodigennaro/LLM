import numpy as np
import pytest
from llm.core.rnn import encode_text
from llm.shared.shared_utils import sample_text
import re
from llm.core.rnn import create_sequences
from llm.core.rnn import pad_sequences_to_same_length
from llm.core.rnn import split_sequences
from llm.core.rnn import one_hot_encode_labels
from llm.core.rnn import create_rnn_model


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
    assert np.array_equal(X[-1], [1, 2, 1, 3, 1]), 'Hello world, hello all, hello'
    assert np.array_equal(y[-1], [0., 0., 1., 0.]), 'world'


def test_create_sequences_typical_case():
    # Testing with a typical list of integers
    encoded = [1, 2, 3, 4]
    expected = [[1, 2], [1, 2, 3], [1, 2, 3, 4]]
    assert create_sequences(encoded) == expected, "Failed on a typical input list"


def test_create_sequences_empty_list():
    # Testing with an empty list
    encoded = []
    expected = []
    assert create_sequences(encoded) == expected, "Failed on an empty input list"


def test_create_sequences_single_element():
    # Testing with a single element list
    encoded = [1]
    expected = []
    assert create_sequences(encoded) == expected, "Failed on a single element list"


def test_create_sequences_content_and_length():
    # Testing to ensure each subsequence is correct in both content and length
    encoded = [1, 2, 3, 4, 5]
    result = create_sequences(encoded)
    # Check that all subsequences are correct
    assert all(result[i] == encoded[:i + 2] for i in range(len(encoded) - 1)), ("Subsequence content or length is "
                                                                                "incorrect")


@pytest.fixture
def sample_sequence():
    sequence = [[1], [1, 2], [1, 2, 1]]
    pre_pad_sequence = np.array([[0, 0, 1], [0, 1, 2], [1, 2, 1]])
    post_pad_sequence = np.array([[1, 0, 0], [1, 2, 0], [1, 2, 1]])
    return sequence, pre_pad_sequence, post_pad_sequence


def test_padding(sample_sequence):
    sequence, pre_pad_sequence, post_pad_sequence = sample_sequence
    expected_pre_pad_sequence = pad_sequences_to_same_length(sequence, 'pre')
    expected_post_pad_sequence = pad_sequences_to_same_length(sequence, 'post')

    assert np.array_equal(pre_pad_sequence, expected_pre_pad_sequence)
    assert np.array_equal(post_pad_sequence, expected_post_pad_sequence)


def test_create_rnn_model():
    vocab_size = 100  # For testing, small vocabulary size
    max_length = 10  # Short sequences for simplicity
    model = create_rnn_model(vocab_size)
    # Check the model's configuration
    assert model.layers[0].input_dim == (None, max_length - 1), "Embedding input shape is incorrect."
    assert model.layers[2].output_dim == (None, vocab_size), "Output layer shape is incorrect."
    assert len(model.layers) == 3, "Unexpected number of layers in the model."

# To run this test, ensure you have pytest installed and execute:
# pytest path_to_test_script.py
