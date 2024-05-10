from typing import Tuple, Dict, List

from numpy import ndarray
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')


def encode_text(text: str) -> Tuple[List[int], Dict[str, int]]:
    """
    Encodes a given text string into a sequence of integers where each integer represents a unique word in the text,
    based on the tokenizer's vocabulary. Additionally, it returns the mapping of words to their respective indices
    in the tokenizer's internal dictionary.

    Args:
        text (str): The text string to be encoded.

    Returns:
        Tuple[List[int], Dict[str, int]]:
            - A list of integers representing the encoded text.
            - A dictionary where keys are words and values are the corresponding unique indices used for encoding.

    Example:
        >>> encode_text("hello world hello")
        ([1, 2, 1], {'hello': 1, 'world': 2})
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    encoded = tokenizer.texts_to_sequences([text])[0]
    return encoded, tokenizer


def create_sequences(encoded: List[int]) -> List[List[int]]:
    """
    Create a list of subsequences from a list of encoded integers. Each subsequence consists of the
    elements from the start-up to the current position in the list, inclusively. This is typically
    used to prepare training data for sequence prediction models where each input sequence is used
    to predict the next element.

    Args:
        encoded (List[int]): A list of integers where each integer represents an encoded version
                             of a word or token in a sequence.

    Returns:
        List[List[int]]: A list of subsequences, where each subsequence contains the first `i` elements
                         of the `encoded` list, with `i` ranging from 1 to the length of `encoded`.

    Example:
        >>> input_list = [1, 2, 3, 4]
        >>> create_sequences(input_list)
        [[1, 2], [1, 2, 3], [1, 2, 3, 4]]
    """
    sequences = []
    for i in range(1, len(encoded)):
        sequence = encoded[:i + 1]
        sequences.append(sequence)
    return sequences


def pad_sequences_to_same_length(sequences: List[List[int]], padding: str = 'pre') -> np.ndarray:
    """
    Pad a list of integer sequences to the same length using a specified padding method.

    Args:
        sequences (List[List[int]]): A list of sequences, where each sequence is a list of integers.
        padding (str, optional): The padding method to use ('pre' or 'post'). Defaults to 'pre'.

    Returns:
        np.ndarray: An array of padded sequences, all the same length.

    Raises:
        ValueError: If `sequences` is empty, as `max_length` cannot be determined.

    Example:
        >>> input_list = [[1, 2, 3], [1, 2], [1]]
        >>> pad_sequences_to_same_length(input_list, padding='post')
        array([[1, 2, 3],
               [1, 2, 0],
               [1, 0, 0]])
    """
    if not sequences:
        raise ValueError("The list of sequences is empty. Cannot determine max_length.")
    max_length = max(len(seq) for seq in sequences)
    return pad_sequences(sequences, maxlen=max_length, padding=padding)


def split_sequences(sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X, y = sequences[:, :-1], sequences[:, -1]
    return X, y


def one_hot_encode_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return to_categorical(labels, num_classes=num_classes)


def prepare_data(text: str) -> tuple[ndarray, ndarray, int, int, dict[str, int]]:
    encoded, tokenizer = encode_text(text)
    vocab_size = len(tokenizer.word_index) + 1  # Including zero index
    sequences = create_sequences(encoded)
    padded_sequences = pad_sequences_to_same_length(sequences)
    X, y = split_sequences(padded_sequences)
    y = one_hot_encode_labels(y, vocab_size)
    max_length = padded_sequences.shape[1]  # Obtain the length of the padded sequences
    return X, y, vocab_size, max_length, tokenizer


def create_rnn_model(vocab_size: int) -> Sequential:
    """
    Creates a simple recurrent neural network (RNN) model for text prediction.

    Args:
        vocab_size (int): The size of the vocabulary in the text data, which determines the number
                          of neurons in the output layer and the size of the one-hot encoded vectors.

    Returns:
        Sequential: A compiled Keras Sequential model with an embedding layer, a simple RNN layer,
                    and a dense output layer with softmax activation suitable for classification.

    Example:
        >>> input_model = create_rnn_model(5000)
        >>> input_model.summary()
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100))
    model.add(SimpleRNN(100))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generate_text(model, tokenizer, seed_text, num_words, max_length):
    """
    Generate text using a trained RNN model based on a seed text.

    Args:
        model (Sequential): The trained Keras model for text generation.
        tokenizer (Tokenizer): The Keras Tokenizer object used for text encoding.
        seed_text (str): Initial text to start the text generation process.
        num_words (int): Number of words to generate after the seed text.
        max_length (int): The maximum length of the sequence that the model was trained on.

    Returns:
        str: The generated text.
    """
    result = seed_text
    for _ in range(num_words):
        # Encode the current set of words
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_length - 1, truncating='pre')

        # Predict the probability of the next word
        predictions = model.predict(encoded, verbose=0).flatten()
        next_word_index = np.argmax(predictions)
        next_word = tokenizer.index_word.get(next_word_index, '')

        # Append to the result and update seed text
        result += ' ' + next_word
        seed_text += ' ' + next_word  # Update seed text to include the new word

    return result


def extract_embeddings(model, tokenizer):
    """
    Extracts embeddings from the model's first layer and gets corresponding words.

    Args:
        model (Sequential): Trained Keras model.
        tokenizer (Tokenizer): Tokenizer used for the model training.

    Returns:
        numpy.ndarray: Embeddings extracted from the model.
        list: List of words corresponding to the indices in the tokenizer.
    """
    # Assuming the first layer of the model is the Embedding layer
    embeddings = model.layers[0].get_weights()[0]
    # Assuming embeddings[0] is for padding and should be ignored
    actual_embeddings = embeddings[1:]  # Skip the first embedding
    words = list(tokenizer.word_index.keys())  # Getting the words from the tokenizer
    return actual_embeddings, words


def extract_grammar_labels(words):
    # Assuming 'words' is a list of words from your tokenizer
    # We'll tag each word with its part of speech
    pos_tags = pos_tag(words)
    # Reduce the number of categories by generalizing tags (NN, NNP -> Noun, etc.)
    pos_simple = {
        'NN': 'Noun', 'NNS': 'Noun', 'NNP': 'Noun', 'NNPS': 'Noun',
        'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb', 'VBN': 'Verb', 'VBP': 'Verb', 'VBZ': 'Verb',
        'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective',
        'RB': 'Adverb', 'RBR': 'Adverb', 'RBS': 'Adverb'
    }
    category_labels = [pos_simple.get(tag, 'Other') for _, tag in pos_tags]
    return category_labels


def extract_sentiment_labels(words):
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Compute the sentiment score for each word
    sentiments = []
    for word in words:
        # VADER works better at the sentence level, so individual words might have less meaningful scores
        score = sia.polarity_scores(word)
        # Append the compound score of each word to the list
        sentiments.append(score['compound'])

    return sentiments
